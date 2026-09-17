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

try:
    import vllm._custom_ops as vllm_ops
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        fused_marlin_moe as vllm_fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        fp8_fused_exponent_bias_into_scales,
        pack_fp8_to_int32,
    )
    from vllm.scalar_type import scalar_types

    VLLM_QUANT_TYPE_FP8 = scalar_types.float8_e4m3fn
    HAS_VLLM_FUSED_MARLIN_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MARLIN_MOE = False

import flaggems_vllm
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP8_E4M3, fused_marlin_moe

from . import base


def is_cuda_available():
    if flaggems_vllm.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return 90 <= sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()
ASCEND_AVAILABLE = flaggems_vllm.vendor_name == "ascend"
GROUP_SIZE = 128


def _quantize_per_expert_fp8(w_fp):
    """Quantize each expert to E4M3 with one scale per 128 weights."""
    num_experts, out_dim, in_dim = w_fp.shape
    assert in_dim % GROUP_SIZE == 0
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    num_groups = in_dim // GROUP_SIZE
    w_q = torch.empty(num_experts, out_dim, in_dim, device=w_fp.device, dtype=fp8_dtype)
    scales = torch.empty(
        num_experts,
        out_dim,
        num_groups,
        device=w_fp.device,
        dtype=w_fp.dtype,
    )
    for expert in range(num_experts):
        w_grouped = w_fp[expert].reshape(out_dim, num_groups, GROUP_SIZE).float()
        scales_fp = (w_grouped.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(
            min=1e-8
        )
        q_expert = (
            (w_grouped / scales_fp).clamp(fp8_info.min, fp8_info.max).to(fp8_dtype)
        )
        w_q[expert] = q_expert.reshape(out_dim, in_dim)
        scales[expert] = scales_fp.squeeze(-1).to(w_fp.dtype)
    return w_q, scales.contiguous()


def _marlin_repack_per_expert_fp8(w_q, scales, dtype):
    """Convert E4M3 weights and per-group scales to vLLM Marlin layout."""
    num_experts, out_dim, in_dim = w_q.shape
    perm = torch.empty(0, dtype=torch.int, device=w_q.device)
    qweight_list = []
    scale_list = []
    for expert in range(num_experts):
        qweight = pack_fp8_to_int32(w_q[expert], size_k_first=False)
        qweight = vllm_ops.gptq_marlin_repack(
            b_q_weight=qweight.T.contiguous(),
            perm=perm,
            size_k=in_dim,
            size_n=out_dim,
            num_bits=8,
        )
        marlin_scales = marlin_permute_scales(
            s=scales[expert].T.to(dtype).contiguous(),
            size_k=in_dim,
            size_n=out_dim,
            group_size=GROUP_SIZE,
        )
        marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)
        qweight_list.append(qweight)
        scale_list.append(marlin_scales)
    return (
        torch.stack(qweight_list, dim=0).contiguous(),
        torch.stack(scale_list, dim=0).contiguous(),
    )


def _decode_e4m3(codes):
    """
    Decode raw E4M3FN codes stored as bytes.

    The sign bit moves to bit 31 and the 7-bit payload to bits 20..26, so the
    float32 view only needs a 2**120 exponent shift (bias 127 vs 7). The
    reserved payload 0x7F decodes to NaN.
    """
    c = codes.to(torch.int32)
    bits = ((c & 0x7F) << 20) | ((c & 0x80) << 24)
    value = bits.view(torch.float32) * 1.329227995784916e36
    return torch.where((c & 0x7F) == 0x7F, torch.full_like(value, float("nan")), value)


def _ascend_weights(e, k, n, dtype):
    import torch_npu

    torch.manual_seed(7)
    result = []
    for ni, ki in [(2 * n, k), (k, n)]:
        # Quantize on CPU so the E4M3 rounding is identical on every
        # torch_npu version. Codes are the uint8 view of the E4M3FN bytes.
        w_fp = torch.randn((e, ni, ki), dtype=dtype) / 10.0
        w_q, s = _quantize_per_expert_fp8(w_fp)
        w_deq = (
            _decode_e4m3(w_q.view(torch.uint8))
            * s.float().repeat_interleave(GROUP_SIZE, dim=-1)
        ).to(dtype)
        w = w_q.view(torch.uint8).to("npu")
        s = s.to("npu")
        # The baseline consumes dequantized BF16 weights, transposed to
        # (in_dim, out_dim). FRACTAL_NZ casting is not serviceable on 910B4
        # with torch_npu 2.10.0.post2, so plain transposed BF16 is used.
        native = []
        for ei in range(e):
            native.append(w_deq[ei].T.contiguous().to("npu"))
        wp = torch.stack(native)
        result.append((w, s, wp))
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
        w = ww[j][2]
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
    """De-facto reference: BF16-dequantized MoE, AllGather/GMM path, EP=1.

    vLLM-Ascend has no FP8-weights + BF16-activations MoE, so the baseline is
    the unquantized chain (AscendUnquantizedFusedMoEMethod) running on the
    dequantized weights. Weight dequant/NZ repacking is outside timing, as in
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
        q = _decode_e4m3(w.cpu())
        s = s.cpu()
        decoded.append(
            (q * s.float().repeat_interleave(GROUP_SIZE, dim=-1)).to(x.dtype).float()
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
    from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP8_E4M3

    return flaggems_vllm.fused_marlin_moe_w8a16_fp8(
        x,
        ww[0][0],
        ww[1][0],
        None,
        None,
        ww[0][1],
        ww[1][1],
        p,
        ids,
        QUANT_TYPE_FP8_E4M3,
    )


def _ascend_inputs(m, e, k, t, seed=7):
    torch.manual_seed(seed + m)
    x = torch.randn((m, k), device="npu", dtype=torch.bfloat16) * 0.1
    ids = torch.rand((m, e), device="npu").topk(t, -1).indices.to(torch.int32)
    p = torch.softmax(torch.randn((m, t), device="npu"), -1)
    return x, p, ids


class FusedMarlinMoEW8A16FP8Benchmark(base.Benchmark):
    """Compare the same E4M3 codes/scales in native and Marlin-repacked layouts.

    On Ascend, FlagGems runs against the BF16-dequantized MoE chain.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self._weight_cache = {}

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
            torch.testing.assert_close(
                _ascend_gems_call(x, ww, p, ids),
                _ascend_baseline(x, ww, p, ids),
                rtol=0.02,
                atol=0.02,
            )
            torch.testing.assert_close(
                _ascend_gems_call(x, ww, p, ids),
                _ascend_vllm_baseline(x, ww, p, ids),
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

    def _get_quantized_weights(
        self, dtype, device, num_experts, hidden_size, intermediate_size
    ):
        cache_key = (dtype, str(device), num_experts, hidden_size, intermediate_size)
        cached = self._weight_cache.get(cache_key)
        if cached is not None:
            return cached
        self._weight_cache.clear()

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
        w1_q_fp8, w1_scale_fp8 = _quantize_per_expert_fp8(w1_fp)
        w2_q_fp8, w2_scale_fp8 = _quantize_per_expert_fp8(w2_fp)
        w1_q_marlin, w1_scale_marlin = _marlin_repack_per_expert_fp8(
            w1_q_fp8, w1_scale_fp8, dtype
        )
        w2_q_marlin, w2_scale_marlin = _marlin_repack_per_expert_fp8(
            w2_q_fp8, w2_scale_fp8, dtype
        )
        cached = (
            w1_q_marlin,
            w1_scale_marlin,
            w2_q_marlin,
            w2_scale_marlin,
            w1_q_fp8,
            w1_scale_fp8,
            w2_q_fp8,
            w2_scale_fp8,
        )
        self._weight_cache[cache_key] = cached
        del w1_fp, w2_fp
        torch.cuda.empty_cache()
        return cached

    def _gen(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flaggems_vllm.device
        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        (
            w1_q_marlin,
            w1_scale_marlin,
            w2_q_marlin,
            w2_scale_marlin,
            w1_q_fp8,
            w1_scale_fp8,
            w2_q_fp8,
            w2_scale_fp8,
        ) = self._get_quantized_weights(
            dtype, device, num_experts, hidden_size, intermediate_size
        )

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        inputs = (
            hidden_states,
            w1_q_marlin,
            w2_q_marlin,
            w1_scale_marlin,
            w2_scale_marlin,
            w1_q_fp8,
            w2_q_fp8,
            w1_scale_fp8,
            w2_scale_fp8,
            topk_weights,
            topk_ids,
        )
        yield inputs


def _vllm_baseline_fp8(
    hidden_states,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    w1_q_fp8,
    w2_q_fp8,
    w1_scale_fp8,
    w2_scale_fp8,
    topk_weights,
    topk_ids,
):
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
        quant_type_id=VLLM_QUANT_TYPE_FP8.id,
    )


def _gems_call_fp8(
    hidden_states,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    w1_q_fp8,
    w2_q_fp8,
    w1_scale_fp8,
    w2_scale_fp8,
    topk_weights,
    topk_ids,
):
    return fused_marlin_moe(
        bias1=None,
        bias2=None,
        quant_type_id=QUANT_TYPE_FP8_E4M3,
        hidden_states=hidden_states,
        w1=w1_q_fp8,
        w2=w2_q_fp8,
        w1_scale=w1_scale_fp8,
        w2_scale=w2_scale_fp8,
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
def test_fused_marlin_moe_w8a16_fp8():
    """
    Benchmark the active backend against its reference W8A16 chain.
    CUDA uses vLLM Marlin; Ascend uses the BF16-dequantized primitive chain.
    """
    baseline_op, gems_op = _vllm_baseline_fp8, _gems_call_fp8
    if ASCEND_AVAILABLE:
        baseline_op, gems_op = _ascend_vllm_baseline, _ascend_gems_call
    bench = FusedMarlinMoEW8A16FP8Benchmark(
        op_name="fused_marlin_moe_w8a16_fp8",
        torch_op=baseline_op,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems_op)
    bench.run()
