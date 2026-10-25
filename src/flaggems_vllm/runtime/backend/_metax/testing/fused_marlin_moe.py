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

from importlib import import_module

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP8_E4M3, QUANT_TYPE_UINT4B8
from flaggems_vllm.runtime import torch_device_fn

GROUP_SIZE = 128


def quantize_uint4b8(w):
    """
    Symmetric per-group GPTQ uint4b8 quantization without vLLM.

    Matches vllm quantize_weights(w.T, scalar_types.uint4b8, GROUP_SIZE,
    zero_points=False): scale = max(|max| / 7, |min| / 8) per group of
    GROUP_SIZE input channels, codes clamped to [-8, 7] and biased by +8.

    Input  w:      (out_dim, in_dim), bf16/fp16
    Output codes:  (out_dim, in_dim), uint8 in [0, 15]
           scales: (out_dim, in_dim // GROUP_SIZE), same dtype as w
    """
    (out_dim, in_dim) = w.shape
    grouped = w.reshape(out_dim, in_dim // GROUP_SIZE, GROUP_SIZE)
    max_val = grouped.amax(dim=-1, keepdim=True)
    min_val = grouped.amin(dim=-1, keepdim=True)
    scales = torch.maximum((max_val / 7).abs(), (min_val / -8).abs())
    codes = torch.round(grouped / scales).int().clamp(-8, 7) + 8
    return (codes.reshape(out_dim, in_dim).to(torch.uint8), scales.squeeze(-1))


def quantize_int4_weights(w_fp):
    """
    Per-expert GPTQ-style INT4 quantization for FlagGems wna16 kernel layout.

    Input  w_fp: (E, out_dim, in_dim), bf16/fp16
    Output w_q:   (E, out_dim, in_dim // 2), uint8 (two nibbles per byte)
           scales: (E, out_dim, in_dim // GROUP_SIZE), same dtype as w_fp
    """
    (E, out_dim, in_dim) = w_fp.shape
    assert in_dim % GROUP_SIZE == 0
    w_q = torch.empty(E, out_dim, in_dim // 2, device=w_fp.device, dtype=torch.uint8)
    scales = torch.empty(
        E, out_dim, in_dim // GROUP_SIZE, device=w_fp.device, dtype=w_fp.dtype
    )
    for e in range(E):
        (q_e, sc_e) = quantize_uint4b8(w_fp[e])
        w_q[e] = q_e[:, 1::2] * 16 + q_e[:, ::2]
        scales[e] = sc_e
    return (w_q, scales)


def int4_inputs(benchmark, config, dtype):
    (num_tokens, num_experts, hidden_size, intermediate_size, topk) = config
    device = flaggems_vllm.device
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1_fp = (
        torch.randn(
            num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
        )
        / 10.0
    )
    w2_fp = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
        )
        / 10.0
    )
    (w1_q_wna16, w1_scale_wna16) = quantize_int4_weights(w1_fp)
    (w2_q_wna16, w2_scale_wna16) = quantize_int4_weights(w2_fp)
    (w1_q_marlin, w1_scale_marlin) = (w1_q_wna16, w1_scale_wna16)
    (w2_q_marlin, w2_scale_marlin) = (w2_q_wna16, w2_scale_wna16)
    del w1_fp, w2_fp
    torch_device_fn.empty_cache()
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    (topk_weights, topk_ids) = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
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


def fp8_weights(benchmark, dtype, device, num_experts, hidden_size, intermediate_size):
    from benchmark.test_fused_marlin_moe_w8a16_fp8 import _quantize_per_expert_fp8

    cache_key = (dtype, str(device), num_experts, hidden_size, intermediate_size)
    cached = benchmark._weight_cache.get(cache_key)
    if cached is not None:
        return cached
    benchmark._weight_cache.clear()
    w1_fp = (
        torch.randn(
            num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
        )
        / 10.0
    )
    w2_fp = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
        )
        / 10.0
    )
    (w1_q_fp8, w1_scale_fp8) = _quantize_per_expert_fp8(w1_fp)
    (w2_q_fp8, w2_scale_fp8) = _quantize_per_expert_fp8(w2_fp)
    for quantized, scale, decoded in (
        (w1_q_fp8, w1_scale_fp8, w1_fp),
        (w2_q_fp8, w2_scale_fp8, w2_fp),
    ):
        for expert in range(num_experts):
            values = quantized[expert].float().view(-1, GROUP_SIZE)
            values.mul_(scale[expert].float().view(-1, 1))
            decoded[expert].copy_(values.view_as(decoded[expert]))
    (w1_q_marlin, w2_q_marlin) = (w1_fp, w2_fp)
    w1_scale_marlin = w2_scale_marlin = None
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
    benchmark._weight_cache[cache_key] = cached
    del w1_fp, w2_fp
    torch_device_fn.empty_cache()
    return cached


def fp8_inputs(benchmark, config, dtype):
    (num_tokens, num_experts, hidden_size, intermediate_size, topk) = config
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
    ) = fp8_weights(
        benchmark, dtype, device, num_experts, hidden_size, intermediate_size
    )
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    (topk_weights, topk_ids) = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
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


def make_test_inputs(precision, router_weight_on_input, factory, *args):
    if precision not in ("int4", "fp8"):
        pytest.skip("MetaX Marlin MoE implements INT4 and FP8 weights")
    return factory(*args)


def supports_benchmark(precision):
    return precision in ("int4", "fp8")


def has_baseline(precision):
    if not supports_benchmark(precision):
        return False
    try:
        import_module("vllm_metax.model_executor.layers.fused_moe.fused_moe")
    except ImportError:
        return False
    return True


def benchmark_inputs(precision, benchmark, dtype):
    generate = int4_inputs if precision == "int4" else fp8_inputs
    for config in benchmark.shapes:
        yield from generate(benchmark, config, dtype)


def baseline(precision, *inputs):
    native = import_module("vllm_metax.model_executor.layers.fused_moe.fused_moe")
    (x, w1, w2, s1, s2, nw1, nw2, _ns1, _ns2, weights, ids) = inputs
    if precision == "int4":
        return native.outplace_fused_experts(
            x,
            w1,
            w2,
            weights,
            ids,
            use_int4_w4a16=True,
            w1_scale=s1,
            w2_scale=s2,
            block_shape=[0, GROUP_SIZE],
        )
    return native.outplace_fused_experts(x, nw1, nw2, weights, ids)


def candidate(precision, *inputs):
    (x, w1, w2, s1, s2, _nw1, _nw2, _ns1, _ns2, weights, ids) = inputs
    return flaggems_vllm.fused_marlin_moe(
        hidden_states=x,
        w1=w1,
        w2=w2,
        bias1=None,
        bias2=None,
        w1_scale=s1,
        w2_scale=s2,
        topk_weights=weights,
        topk_ids=ids,
        quant_type_id=(
            QUANT_TYPE_UINT4B8 if precision == "int4" else QUANT_TYPE_FP8_E4M3
        ),
    )
