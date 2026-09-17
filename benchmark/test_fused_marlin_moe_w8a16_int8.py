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

import pytest
import torch

# vLLM imports (baseline). Optional: when vllm is not installed (e.g. in CI),
# the entire benchmark is skipped via the skipif marker below.
try:
    import vllm._custom_ops as vllm_ops
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        fused_marlin_moe as vllm_fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        quantize_weights,
    )
    from vllm.scalar_type import scalar_types

    VLLM_QUANT_TYPE_INT8 = scalar_types.uint8b128
    HAS_VLLM_FUSED_MARLIN_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MARLIN_MOE = False

import flaggems_vllm

# FlagGems wrapper under test
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT8B128, fused_marlin_moe

from . import base


def is_cuda_available():
    if flaggems_vllm.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()
ASCEND_AVAILABLE = flaggems_vllm.vendor_name == "ascend"

GROUP_SIZE = 128


def _wna16_quantize_per_expert_int8(w_fp):
    """
    Per-expert GPTQ-style INT8 quantization for FlagGems wna16 kernel layout.
    INT8 is one byte per element — no nibble packing — so K-dim stays in_dim.

    Input  w_fp: (E, out_dim, in_dim), bf16/fp16
    Output w_q:   (E, out_dim, in_dim), uint8
           scales: (E, out_dim, in_dim // GROUP_SIZE), same dtype as w_fp
    """
    E, out_dim, in_dim = w_fp.shape
    assert in_dim % GROUP_SIZE == 0
    w_q = torch.empty(E, out_dim, in_dim, device=w_fp.device, dtype=torch.uint8)
    scales = torch.empty(
        E, out_dim, in_dim // GROUP_SIZE, device=w_fp.device, dtype=w_fp.dtype
    )
    for e in range(E):
        _, q_e, sc_e, _ = quantize_weights(
            w_fp[e].T, VLLM_QUANT_TYPE_INT8, GROUP_SIZE, False, False
        )
        q_e = q_e.T.contiguous().to(torch.uint8)
        sc_e = sc_e.T
        w_q[e] = q_e
        scales[e] = sc_e
    return w_q, scales


def _marlin_repack_per_expert_int8(w_q, scales):
    """Repack the same UINT8 codes and scales for vLLM's CUDA Marlin kernel."""
    qweight_l, scales_l = [], []
    E, out_dim, in_dim = w_q.shape
    perm = torch.empty(0, dtype=torch.int, device=w_q.device)
    for e in range(E):
        qw = vllm_ops.gptq_marlin_repack(
            w_q[e].view(torch.int32).T.contiguous(), perm, in_dim, out_dim, 8
        )
        sc = marlin_permute_scales(
            scales[e].T.contiguous(), in_dim, out_dim, GROUP_SIZE
        )
        qweight_l.append(qw)
        scales_l.append(sc)
    qweight = torch.stack(qweight_l, dim=0).contiguous()
    scales = torch.stack(scales_l, dim=0).contiguous()
    return qweight, scales


def _ascend_weights(e, k, n, dtype):
    """Deterministic W8A16 INT8 weights for the Ascend candidate and baseline.

    Seeded full-precision weights are symmetric-quantized per group of 128
    along the reduction dim: stored code c = value + 128 (real int8 = c - 128),
    scales in the activation dtype.

    Per matrix returns (code, scale, dequant_bf16):
      code:         (E, out, in), uint8, candidate wna16 layout
      scale:        (E, out, in // 128), dtype
      dequant_bf16: (E, in, out), dtype, transposed for the unquant baseline GMM
    """
    torch.manual_seed(7)
    result = []
    for ni, ki in [(2 * n, k), (k, n)]:
        assert ki % 128 == 0
        w = torch.randn((e, ni, ki), device="npu", dtype=dtype).float() / 10.0
        amax = w.reshape(e, ni, ki // 128, 128).abs().amax(-1).clamp(min=1e-30)
        s = (amax / 127.0).to(dtype)
        q = torch.round(w.reshape(e, ni, ki // 128, 128) / s.float()[..., None])
        q = q.clamp(-127, 127)
        code = (q.to(torch.int32) + 128).to(torch.uint8).reshape(e, ni, ki)
        dequant = (q * s.float()[..., None]).reshape(e, ni, ki).to(dtype)
        result.append((code, s, dequant.transpose(1, 2).contiguous()))
    return result


def _ascend_baseline(x, ww, p, ids, *, vllm_dispatch=False):
    import torch_npu

    e = ww[0][0].shape[0]
    a, idx, counts, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        ids.to(torch.int32),
        expert_num=e,
        active_num=x.shape[0] * ids.shape[1],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        active_expert_range=[0, e],
        quant_mode=-1,
    )
    if vllm_dispatch:
        counts = counts.to(torch.int64)
    for j in range(2):
        _, _, w = ww[j]
        a = torch_npu.npu_grouped_matmul(
            x=[a],
            weight=[w],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=counts,
            output_dtype=x.dtype,
        )[0]
        if j == 0:
            a = torch_npu.npu_swiglu(a)
    if vllm_dispatch:
        idx = torch.abs(idx)
        p = p.to(a.dtype)
    return torch_npu.npu_moe_token_unpermute(a, idx, probs=p)


def _ascend_vllm_baseline(x, ww, p, ids):
    """Single-device vLLM-Ascend BF16-unquantized AllGather/GMM path, EP=1.

    vLLM-Ascend has no same-precision W8A16 INT8 MoE (W8A16 is a per-channel
    linear method only), so the baseline is AscendUnquantizedFusedMoEMethod
    style: tokens are gathered with the same AllGather dispatcher and run
    against BF16-dequantized weights. Dequantization is outside timing, as in
    process_weights_after_loading.
    """
    return _ascend_baseline(x, ww, p, ids, vllm_dispatch=True)


def _ascend_reference(x, ww, p, ids):
    x = x.cpu()
    p = p.cpu()
    ids = ids.cpu()
    y = torch.zeros_like(x, dtype=torch.float32)
    decoded = []
    for w, s, *_ in ww:
        w = w.cpu().float() - 128
        s = s.cpu()
        decoded.append(
            (w * s.float().repeat_interleave(128, dim=-1)).to(x.dtype).float()
        )
    for m in range(x.shape[0]):
        for t in range(ids.shape[1]):
            e = int(ids[m, t])
            h = (x[m].float() @ decoded[0][e].T).to(x.dtype).float()
            a, b = h.chunk(2)
            a = (torch.nn.functional.silu(a) * b).to(x.dtype).float()
            z = (a @ decoded[1][e].T).to(x.dtype).float()
            y[m] += z * p[m, t]
    return y.to(x.dtype)


def _ascend_gems_call(x, ww, p, ids):
    import flaggems_vllm
    from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT8B128

    return flaggems_vllm.fused_marlin_moe_w8a16_int8(
        x,
        ww[0][0],
        ww[1][0],
        None,
        None,
        ww[0][1],
        ww[1][1],
        p,
        ids,
        QUANT_TYPE_UINT8B128,
    )


def _ascend_inputs(m, e, k, t, seed=7):
    torch.manual_seed(seed + m)
    x = torch.randn((m, k), device="npu", dtype=torch.bfloat16) * 0.1
    ids = torch.rand((m, e), device="npu").topk(t, -1).indices.to(torch.int32)
    p = torch.softmax(torch.randn((m, t), device="npu"), -1)
    return x, p, ids


class FusedMarlinMoEW8A16INT8Benchmark(base.Benchmark):
    """
    Benchmark for fused_marlin_moe W8A16 INT8 (fused-dequant MoE GEMM).

    Compares FlagGems against vLLM Marlin on CUDA or the Ascend BF16-dequant
    MoE chain (vLLM-Ascend has no same-precision W8A16 INT8 MoE). All paths
    consume symmetric per-group-128 uint8b128 weights.
    Sister of FusedMarlinMoEW4A16INT4Benchmark (W4A16 INT4).
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (tokens, experts, hidden, intermediate, topk)
            for experts, hidden, intermediate, topk in (
                (8, 4096, 14336, 2),  # Mixtral-8x7B
                (256, 7168, 2048, 8),  # DeepSeek-V3 (TP=8)
                (512, 4096, 1024, 10),  # Qwen3.5-397B-A17B
                (256, 4096, 2048, 6),  # DeepSeek-V4-Flash
            )
            for tokens in (1, 16, 64, 256, 1024, 4096, 16384)
        ]

    def _get_ascend_input_iter(self, dtype):
        geometry = None
        ww = None
        for m, e, k, n, t in self.shapes:
            if geometry != (e, k, n):
                ww = _ascend_weights(e, k, n, dtype)
                geometry = (e, k, n)
            x, p, ids = _ascend_inputs(m, e, k, t)
            got = _ascend_gems_call(x, ww, p, ids)
            torch.testing.assert_close(
                got, _ascend_baseline(x, ww, p, ids), rtol=0.02, atol=0.02
            )
            torch.testing.assert_close(
                got, _ascend_vllm_baseline(x, ww, p, ids), rtol=0.02, atol=0.02
            )
            if m <= 256:
                torch.testing.assert_close(
                    got.cpu(),
                    _ascend_reference(x, ww, p, ids),
                    rtol=0.02,
                    atol=0.02,
                )
            yield (x, ww, p, ids)

    def get_input_iter(self, cur_dtype):
        if ASCEND_AVAILABLE:
            yield from self._get_ascend_input_iter(cur_dtype)
            return
        for config in self.shapes:
            yield from self._gen(config, cur_dtype)

    def _gen(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flaggems_vllm.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

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

        # FlagGems wna16 INT8 layout (unpacked)
        w1_q_wna16, w1_scale_wna16 = _wna16_quantize_per_expert_int8(w1_fp)
        w2_q_wna16, w2_scale_wna16 = _wna16_quantize_per_expert_int8(w2_fp)

        # Repacking preserves quantized values; it is outside benchmark timing.
        w1_q_marlin, w1_scale_marlin = _marlin_repack_per_expert_int8(
            w1_q_wna16, w1_scale_wna16
        )
        w2_q_marlin, w2_scale_marlin = _marlin_repack_per_expert_int8(
            w2_q_wna16, w2_scale_wna16
        )

        del w1_fp, w2_fp
        torch.cuda.empty_cache()

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        inputs = (
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
        yield inputs


def _vllm_baseline_int8(
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
    """Baseline: vLLM's CUDA Marlin fused_marlin_moe (INT8)."""
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
        quant_type_id=VLLM_QUANT_TYPE_INT8.id,
    )


def _gems_call_int8(
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
    """FlagGems' Triton wna16 fused_marlin_moe W8A16."""
    return fused_marlin_moe(
        bias1=None,
        bias2=None,
        quant_type_id=QUANT_TYPE_UINT8B128,
        hidden_states=hidden_states,
        w1=w1_q_wna16,
        w2=w2_q_wna16,
        w1_scale=w1_scale_wna16,
        w2_scale=w2_scale_wna16,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


@pytest.mark.fused_marlin_moe
@pytest.mark.skipif(
    not ASCEND_AVAILABLE and not HAS_VLLM_FUSED_MARLIN_MOE,
    reason="vllm not installed; CUDA baseline unavailable",
)
@pytest.mark.skipif(
    not (CUDA_AVAILABLE or ASCEND_AVAILABLE),
    reason="requires NVIDIA Hopper or Ascend",
)
def test_fused_marlin_moe_w8a16_int8():
    """
    Benchmark the active backend using its same-precision W8A16 INT8 baseline.
    CUDA uses vLLM Marlin; Ascend uses the BF16-dequantized torch_npu MoE
    primitive chain (no vLLM-Ascend W8A16 INT8 MoE exists).
    """
    baseline_op, gems_op = _vllm_baseline_int8, _gems_call_int8
    if ASCEND_AVAILABLE:
        baseline_op, gems_op = _ascend_vllm_baseline, _ascend_gems_call
    bench = FusedMarlinMoEW8A16INT8Benchmark(
        op_name="fused_marlin_moe_w8a16_int8",
        torch_op=baseline_op,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems_op)
    bench.run()
