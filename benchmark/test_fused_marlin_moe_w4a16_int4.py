# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import re
from collections import defaultdict
from statistics import median

import pytest
import torch
import triton
import triton.language as tl

# vLLM imports (baseline). Optional: when vllm is not installed (e.g. in CI),
# the entire benchmark is skipped via the skipif marker below.
try:
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        fused_marlin_moe as vllm_fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        quantize_weights,
    )
    from vllm.scalar_type import scalar_types

    VLLM_QUANT_TYPE = scalar_types.uint4b8
    HAS_VLLM_FUSED_MARLIN_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MARLIN_MOE = False

import flaggems_vllm

# FlagGems wrapper under test
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT4B8
from flaggems_vllm.ops.fused_marlin_moe import fused_marlin_moe as gems_fused_marlin_moe
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_no_tle,
    moe_align_block_size_small_grouped,
)
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.runtime.backend._metax.fused.moe_sum import moe_sum

from . import base


def is_cuda_available():
    if flaggems_vllm.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()

GROUP_SIZE = 128


def _wna16_quantize_per_expert(w_fp):
    """
    Per-expert GPTQ-style INT4 quantization for FlagGems wna16 kernel layout.

    Input  w_fp: (E, out_dim, in_dim), bf16/fp16
    Output w_q:   (E, out_dim, in_dim // 2), uint8 (two nibbles per byte)
           scales: (E, out_dim, in_dim // GROUP_SIZE), same dtype as w_fp
    """
    E, out_dim, in_dim = w_fp.shape
    assert in_dim % GROUP_SIZE == 0
    w_q = torch.empty(E, out_dim, in_dim // 2, device=w_fp.device, dtype=torch.uint8)
    scales = torch.empty(
        E, out_dim, in_dim // GROUP_SIZE, device=w_fp.device, dtype=w_fp.dtype
    )
    for e in range(E):
        _, q_e, sc_e, _ = quantize_weights(
            w_fp[e].T, VLLM_QUANT_TYPE, GROUP_SIZE, False, False
        )
        q_e = q_e.T.contiguous().to(torch.uint8)
        sc_e = sc_e.T
        w_q[e] = q_e[:, 1::2] * 16 + q_e[:, ::2]
        scales[e] = sc_e
    return w_q, scales


def _marlin_quantize_per_expert(w_fp):
    """
    Per-expert Marlin-layout INT4 quantization for vLLM's fused_marlin_moe.

    Input  w_fp: (E, out_dim, in_dim), bf16/fp16
    Output qweight: stacked (E, ...), int32 (Marlin packed layout)
           scales:  stacked (E, ...), same dtype as w_fp
    """
    qweight_l, scales_l = [], []
    E = w_fp.shape[0]
    for e in range(E):
        # marlin_quantize expects (in_dim, out_dim)
        _, qw, sc, _, _, _ = marlin_quantize(
            w_fp[e].T.contiguous(), VLLM_QUANT_TYPE, GROUP_SIZE, act_order=False
        )
        qweight_l.append(qw)
        scales_l.append(sc)
    qweight = torch.stack(qweight_l, dim=0).contiguous()
    scales = torch.stack(scales_l, dim=0).contiguous()
    return qweight, scales


class FusedMarlinMoEW4A16INT4Benchmark(base.Benchmark):
    """
    Benchmark for fused_marlin_moe W4A16 INT4 (fused-dequant MoE GEMM).

    Compares FlagGems' Triton wna16 kernel against vLLM's Marlin CUDA kernel.
    Both consume per-group-128 GPTQ uint4b8 weights (different packed layouts).
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # The three production MoE architectures from profile_fused_marlin_moe.py
        # over the decode token range (1 .. 256).
        self.shapes = [
            # Mixtral-8x7B
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (8, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (32, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            # DeepSeek-V3 (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (8, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (32, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
            # Qwen3-5-397B-A17B
            (1, 512, 4096, 1024, 10),
            (4, 512, 4096, 1024, 10),
            (8, 512, 4096, 1024, 10),
            (16, 512, 4096, 1024, 10),
            (32, 512, 4096, 1024, 10),
            (64, 512, 4096, 1024, 10),
            (128, 512, 4096, 1024, 10),
            (256, 512, 4096, 1024, 10),
            # DeepSeek-V4-Flash
            (1, 256, 4096, 2048, 6),
            (4, 256, 4096, 2048, 6),
            (8, 256, 4096, 2048, 6),
            (16, 256, 4096, 2048, 6),
            (32, 256, 4096, 2048, 6),
            (64, 256, 4096, 2048, 6),
            (128, 256, 4096, 2048, 6),
            (256, 256, 4096, 2048, 6),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._gen(config, cur_dtype)

    def _gen(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flaggems_vllm.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Original FP weights (kept only as source for both quantizers).
        w1_fp = (
            torch.randn(
                num_experts,
                intermediate_size * 2,
                hidden_size,
                device=device,
                dtype=dtype,
            )
            / 10.0
        )
        w2_fp = (
            torch.randn(
                num_experts,
                hidden_size,
                intermediate_size,
                device=device,
                dtype=dtype,
            )
            / 10.0
        )

        # FlagGems wna16 layout
        w1_q_wna16, w1_scale_wna16 = _wna16_quantize_per_expert(w1_fp)
        w2_q_wna16, w2_scale_wna16 = _wna16_quantize_per_expert(w2_fp)

        # vLLM Marlin layout
        w1_q_marlin, w1_scale_marlin = _marlin_quantize_per_expert(w1_fp)
        w2_q_marlin, w2_scale_marlin = _marlin_quantize_per_expert(w2_fp)

        del w1_fp, w2_fp
        torch.cuda.empty_cache()

        # Routing
        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        # vLLM requires fp32 topk_weights; FlagGems wrapper is dtype-agnostic.

        # Both ops get the same tuple; each picks what it needs.
        yield (
            hidden_states,
            w1_q_wna16,
            w2_q_wna16,
            w1_scale_wna16,
            w2_scale_wna16,
            w1_q_marlin,
            w2_q_marlin,
            w1_scale_marlin,
            w2_scale_marlin,
            topk_weights,
            topk_ids,
        )


def _vllm_baseline(
    hidden_states,
    w1_q_wna16,
    w2_q_wna16,
    w1_scale_wna16,
    w2_scale_wna16,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    topk_weights,
    topk_ids,
):
    """Baseline: vLLM's CUDA Marlin fused_marlin_moe."""
    return vllm_fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1_q_marlin,
        w2=w2_q_marlin,
        bias1=None,
        bias2=None,
        w1_scale=w1_scale_marlin,
        w2_scale=w2_scale_marlin,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=VLLM_QUANT_TYPE.id,
    )


def _gems_call(
    hidden_states,
    w1_q_wna16,
    w2_q_wna16,
    w1_scale_wna16,
    w2_scale_wna16,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    topk_weights,
    topk_ids,
):
    """FlagGems' Triton wna16 fused_marlin_moe (Phase 2)."""
    return gems_fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1_q_wna16,
        w2=w2_q_wna16,
        bias1=None,
        bias2=None,
        w1_scale=w1_scale_wna16,
        w2_scale=w2_scale_wna16,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=QUANT_TYPE_UINT4B8,
    )


SHAPE_PATTERN = re.compile(
    r"flag_gems\.ops\.fused_marlin_moe\.fused_marlin_moe, "
    r"\[shape info\]: \[([^]]+)\].*\[count\]: (\d+)"
)


def _load_metax_shapes(shape_glob):
    counts = defaultdict(int)
    for path in glob.glob(shape_glob):
        with open(path, encoding="utf-8") as shape_file:
            for line in shape_file:
                match = SHAPE_PATTERN.search(line)
                if match is None:
                    continue
                shape = tuple(int(dim) for dim in match.group(1).split(","))
                if len(shape) != 5:
                    raise ValueError(f"expected [T, E, H, I, topk] in {path}")
                counts[shape] += int(match.group(2))
    return sorted(counts.items(), key=lambda item: (item[0][1:], item[0][0]))


def _metax_align_routes(topk_ids, num_experts, block_m):
    num_routes = topk_ids.numel()
    max_grouped_routes = 64 if num_experts >= 128 else 512
    if num_routes <= max_grouped_routes and num_experts <= 1024:
        return moe_align_block_size_small_grouped(topk_ids, num_experts, block_m)
    return moe_align_block_size_no_tle(topk_ids, block_m, num_experts)


def _metax_native_gemm(
    kernel,
    activation,
    weight,
    scale,
    output,
    topk_weights,
    sorted_ids,
    expert_ids,
    num_post_padded,
    block_m,
    block_n,
    block_k,
    topk,
    mul_routed_weight,
):
    num_valid = topk_weights.numel()
    problem_m = sorted_ids.shape[0]
    if activation.shape[0] < block_m:
        problem_m = min(problem_m, activation.shape[0] * topk * block_m)
    stride_cm = output.stride(1) if output.ndim == 3 else output.stride(0)
    stride_cn = output.stride(2) if output.ndim == 3 else output.stride(1)
    grid = (triton.cdiv(problem_m, block_m) * triton.cdiv(weight.shape[1], block_n),)
    kernel(
        grid,
        activation,
        weight,
        output,
        scale,
        None,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_post_padded,
        weight.shape[1],
        activation.shape[1],
        problem_m,
        num_valid,
        activation.stride(0),
        activation.stride(1),
        weight.stride(0),
        weight.stride(2),
        weight.stride(1),
        stride_cm,
        stride_cn,
        scale.stride(0),
        scale.stride(2),
        scale.stride(1),
        0,
        0,
        0,
        block_k_diviable=activation.shape[1] % block_k == 0,
        group_size=GROUP_SIZE,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=topk,
        compute_type=tl.bfloat16,
        has_zp=False,
        use_int4_w4a16=True,
        use_int8_w8a16=False,
        GROUP_SIZE_M=1,
        SPLIT_K=1,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
    )


def _metax_native_moe(kernel, hidden, w1, w2, s1, s2, weights, ids):
    tokens, hidden_size = hidden.shape
    num_experts, fused_intermediate, _ = w1.shape
    intermediate_size = fused_intermediate // 2
    topk = ids.shape[1]
    block_m = 16 if tokens <= 20 else 32 if tokens <= 40 else 64
    block_n = 32 if tokens == 1 else 64
    block_k = 64 if tokens == 1 else 32
    gate_up = torch.empty(
        (tokens * topk, fused_intermediate), device=hidden.device, dtype=hidden.dtype
    )
    activated = torch.empty(
        (tokens * topk, intermediate_size), device=hidden.device, dtype=hidden.dtype
    )
    routed = torch.empty(
        (tokens, topk, hidden_size), device=hidden.device, dtype=hidden.dtype
    )
    sorted_ids, expert_ids, num_post_padded = _metax_align_routes(
        ids, num_experts, block_m
    )
    _metax_native_gemm(
        kernel,
        hidden,
        w1,
        s1,
        gate_up,
        weights,
        sorted_ids,
        expert_ids,
        num_post_padded,
        block_m,
        block_n,
        block_k,
        topk,
        False,
    )
    silu_and_mul_out(
        gate_up[:, :intermediate_size], gate_up[:, intermediate_size:], activated
    )
    _metax_native_gemm(
        kernel,
        activated,
        w2,
        s2,
        routed,
        weights,
        sorted_ids,
        expert_ids,
        num_post_padded,
        block_m,
        block_n,
        block_k,
        1,
        True,
    )
    output = torch.empty_like(hidden)
    moe_sum(routed, output)
    return output


def _metax_bench(fn, iterations):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def _run_metax_int4_benchmark():
    pattern = os.environ.get("FLAGOSTUNE_MARLIN_SHAPE_GLOB")
    if not pattern:
        pytest.skip("set FLAGOSTUNE_MARLIN_SHAPE_GLOB to exported shape files")
    shapes = _load_metax_shapes(pattern)
    if not shapes:
        pytest.skip(f"no fused_marlin_moe shape exports match {pattern}")
    native_kernel = pytest.importorskip(
        "mcoplib.triton_fused_moe"
    ).fused_moe_triton_kernel_gptq_awq
    torch.manual_seed(0)
    weights = None
    geometry = None
    weighted_speedup = 0.0
    total_count = 0
    minimum_speedup = float("inf")
    for shape, count in shapes:
        tokens, num_experts, hidden_size, intermediate_size, topk = shape
        if hidden_size % GROUP_SIZE or intermediate_size % GROUP_SIZE:
            raise ValueError(f"group size does not divide {shape}")
        if geometry != shape[1:4]:
            geometry = shape[1:4]
            w1 = torch.randint(
                0,
                256,
                (num_experts, 2 * intermediate_size, hidden_size // 2),
                device="cuda",
                dtype=torch.uint8,
            )
            w2 = torch.randint(
                0,
                256,
                (num_experts, hidden_size, intermediate_size // 2),
                device="cuda",
                dtype=torch.uint8,
            )
            s1 = (
                torch.rand(
                    num_experts,
                    2 * intermediate_size,
                    hidden_size // GROUP_SIZE,
                    device="cuda",
                )
                * 0.02
                + 0.01
            ).to(torch.bfloat16)
            s2 = (
                torch.rand(
                    num_experts,
                    hidden_size,
                    intermediate_size // GROUP_SIZE,
                    device="cuda",
                )
                * 0.02
                + 0.01
            ).to(torch.bfloat16)
            weights = (w1, w2, s1, s2)
        w1, w2, s1, s2 = weights
        hidden = (
            torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16) / 10
        )
        gating = torch.randn(tokens, num_experts, device="cuda")
        routes, ids = torch.topk(torch.softmax(gating, -1), topk, dim=-1)
        routes = (routes / routes.sum(-1, keepdim=True)).to(torch.bfloat16)
        inputs = (hidden, w1, w2, s1, s2, routes, ids)

        def ours():
            return flaggems_vllm.fused_marlin_moe(
                hidden, w1, w2, None, None, s1, s2, routes, ids, 0
            )

        def baseline():
            return _metax_native_moe(native_kernel, *inputs)

        actual, expected = ours(), baseline()
        error = (
            (actual.float() - expected.float()).abs().mean()
            / expected.float().abs().mean().clamp_min(1e-12)
        ).item()
        assert error < 0.04, f"{shape}: relative error={error}"
        iterations = 4 if tokens >= 1024 else 8
        ours_us = median(_metax_bench(ours, iterations) for _ in range(3))
        baseline_us = median(_metax_bench(baseline, iterations) for _ in range(3))
        speedup = baseline_us / ours_us
        minimum_speedup = min(minimum_speedup, speedup)
        assert speedup >= 1.0, f"{shape}: native INT4 is faster ({speedup:.3f}x)"
        weighted_speedup += count * speedup
        total_count += count
        print(
            f"METAX_INT4 shape={shape} count={count} error={error:.6f} "
            f"ours_us={ours_us:.1f} vllm_us={baseline_us:.1f} "
            f"speedup={speedup:.3f}",
            flush=True,
        )
    print(
        f"METAX_INT4 weighted_speedup={weighted_speedup / total_count:.3f} "
        f"sum_count={total_count} shapes={len(shapes)} "
        f"minimum_speedup={minimum_speedup:.3f}",
        flush=True,
    )


@pytest.mark.fused_marlin_moe_w4a16_int4
def test_fused_marlin_moe_w4a16_int4():
    """Compare W4A16 INT4 against the corresponding device's native INT4 path."""
    if flaggems_vllm.vendor_name == "metax":
        return _run_metax_int4_benchmark()
    if not HAS_VLLM_FUSED_MARLIN_MOE:
        pytest.skip("vllm not installed; baseline unavailable")
    if not CUDA_AVAILABLE:
        pytest.skip("requires NVIDIA Hopper architecture")
    bench = FusedMarlinMoEW4A16INT4Benchmark(
        op_name="fused_marlin_moe_w4a16_int4",
        torch_op=_vllm_baseline,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_call)
    bench.run()
