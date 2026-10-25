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

import math
from importlib import import_module

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.fused_marlin_moe import (
    MXFP4_GROUP_SIZE,
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
)
from flaggems_vllm.runtime import torch_device_fn

GROUP_SIZE = 128


def pack_int4(weight):
    """Convert output-major packed INT4 bytes to GPTQ INT32 packing."""
    (experts, output_size, packed_k) = weight.shape
    packed = torch.empty(
        (experts, packed_k // 4, output_size), device=weight.device, dtype=torch.int32
    )
    for expert in range(experts):
        bytes4 = weight[expert].to(torch.int32).reshape(output_size, packed_k // 4, 4)
        words = (
            bytes4[..., 0]
            | bytes4[..., 1] << 8
            | bytes4[..., 2] << 16
            | bytes4[..., 3] << 24
        )
        packed[expert].copy_(words.transpose(0, 1))
    return packed


def int4_native_layout(weight, scales, size_k, size_n):
    from vllm import _custom_ops as vllm_ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_permute_scales,
    )

    qweight = pack_int4(weight)
    empty_perm = torch.empty(
        (weight.size(0), 0), device=weight.device, dtype=torch.int32
    )
    qweight = vllm_ops.gptq_marlin_moe_repack(
        qweight, empty_perm, size_k=size_k, size_n=size_n, num_bits=4
    )
    scales = marlin_moe_permute_scales(
        scales.transpose(1, 2).contiguous(),
        size_k=size_k,
        size_n=size_n,
        group_size=GROUP_SIZE,
    )
    return (qweight, scales)


def int4_weights(num_experts, hidden_size, intermediate_size, dtype):
    torch.manual_seed(7)
    device = flaggems_vllm.device
    w1 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w1_scale = (
        torch.rand(
            (num_experts, 2 * intermediate_size, hidden_size // GROUP_SIZE),
            device=device,
            dtype=dtype,
        )
        * 0.03
    )
    w2_scale = (
        torch.rand(
            (num_experts, hidden_size, intermediate_size // GROUP_SIZE),
            device=device,
            dtype=dtype,
        )
        * 0.03
    )
    (vllm_w1, vllm_w1_scale) = int4_native_layout(
        w1, w1_scale, hidden_size, 2 * intermediate_size
    )
    (vllm_w2, vllm_w2_scale) = int4_native_layout(
        w2, w2_scale, intermediate_size, hidden_size
    )
    return (w1, w2, w1_scale, w2_scale, vllm_w1, vllm_w2, vllm_w1_scale, vllm_w2_scale)


def mxfp4_native_layout(weight, scales, size_k, size_n):
    from vllm import _custom_ops as vllm_ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_permute_scales,
    )

    qweight = pack_int4(weight)
    empty_perm = torch.empty(
        (weight.size(0), 0), device=weight.device, dtype=torch.int32
    )
    qweight = vllm_ops.gptq_marlin_moe_repack(
        qweight, empty_perm, size_k=size_k, size_n=size_n, num_bits=4
    )
    scales = marlin_moe_permute_scales(
        scales.transpose(1, 2).contiguous(),
        size_k=size_k,
        size_n=size_n,
        group_size=MXFP4_GROUP_SIZE,
    )
    scales = (
        scales.reshape(-1, 4)[:, [0, 2, 1, 3]]
        .reshape(weight.size(0), size_k // MXFP4_GROUP_SIZE, size_n)
        .contiguous()
        .view(torch.float8_e8m0fnu)
    )
    return (qweight, scales)


def mxfp4_weights(num_experts, hidden_size, intermediate_size):
    torch.manual_seed(7)
    device = flaggems_vllm.device
    w1 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w1_scale = torch.randint(
        120,
        124,
        (num_experts, 2 * intermediate_size, hidden_size // MXFP4_GROUP_SIZE),
        device=device,
        dtype=torch.uint8,
    )
    w2_scale = torch.randint(
        120,
        124,
        (num_experts, hidden_size, intermediate_size // MXFP4_GROUP_SIZE),
        device=device,
        dtype=torch.uint8,
    )
    (vllm_w1, vllm_w1_scale) = mxfp4_native_layout(
        w1, w1_scale, hidden_size, 2 * intermediate_size
    )
    (vllm_w2, vllm_w2_scale) = mxfp4_native_layout(
        w2, w2_scale, intermediate_size, hidden_size
    )
    return (w1, w2, w1_scale, w2_scale, vllm_w1, vllm_w2, vllm_w1_scale, vllm_w2_scale)


def pack_int8(weight):
    """Pack four INT8 or E4M3 bytes per GPTQ INT32 word."""
    (experts, output_size, packed_k) = weight.shape
    packed = torch.empty(
        (experts, packed_k // 4, output_size), device=weight.device, dtype=torch.int32
    )
    for expert in range(experts):
        bytes4 = weight[expert].to(torch.int32).reshape(output_size, packed_k // 4, 4)
        words = (
            bytes4[..., 0]
            | bytes4[..., 1] << 8
            | bytes4[..., 2] << 16
            | bytes4[..., 3] << 24
        )
        packed[expert].copy_(words.transpose(0, 1))
    return packed


def w8_native_layout(weight, scales, size_k, size_n, fp8):
    from vllm import _custom_ops as vllm_ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_permute_scales,
    )

    if fp8:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            fp8_fused_exponent_bias_into_scales,
        )
    qweight = pack_int8(weight.view(torch.uint8))
    empty_perm = torch.empty(
        (weight.size(0), 0), device=weight.device, dtype=torch.int32
    )
    qweight = vllm_ops.gptq_marlin_moe_repack(
        qweight, empty_perm, size_k=size_k, size_n=size_n, num_bits=8
    )
    scales = marlin_moe_permute_scales(
        scales.transpose(1, 2).contiguous(),
        size_k=size_k,
        size_n=size_n,
        group_size=GROUP_SIZE,
    )
    if fp8:
        scales = fp8_fused_exponent_bias_into_scales(scales)
    return (qweight, scales)


def w8_weights(num_experts, hidden_size, intermediate_size, dtype, fp8):
    has_baseline = True if fp8 else True
    torch.manual_seed(7)
    weights = []
    scales = []
    native_weights = []
    native_scales = []
    for output_size, input_size in (
        (2 * intermediate_size, hidden_size),
        (hidden_size, intermediate_size),
    ):
        raw = torch.randint(
            0,
            256,
            (num_experts, output_size, input_size),
            device=flaggems_vllm.device,
            dtype=torch.uint8,
        )
        weight = raw
        if fp8:
            raw = torch.where(raw & 127 == 127, raw - 1, raw)
            weight = raw.view(torch.float8_e4m3fn)
        scale = (
            torch.rand(
                (num_experts, output_size, input_size // GROUP_SIZE),
                device=flaggems_vllm.device,
            )
            * 0.001
            + 0.0005
        ).to(dtype)
        if has_baseline:
            (native_weight, native_scale) = w8_native_layout(
                weight, scale, input_size, output_size, fp8
            )
        else:
            native_weight = native_scale = None
        weights.append(weight)
        scales.append(scale)
        native_weights.append(native_weight)
        native_scales.append(native_scale)
    return (*weights, *scales, *native_weights, *native_scales)


def e4m3_lut(device):
    values = []
    for q in range(256):
        exponent = q >> 3 & 15
        mantissa = q & 7
        value = (
            math.ldexp(mantissa, -9)
            if exponent == 0
            else math.ldexp(1 + mantissa / 8, exponent - 7)
        )
        if exponent == 15 and mantissa == 7:
            value = float("nan")
        values.append(-value if q & 128 else value)
    return torch.tensor(values, dtype=torch.float32, device=device)


def make_fp8_test_inputs(
    num_tokens,
    num_experts,
    hidden_size,
    intermediate_size,
    topk,
    dtype,
    device,
    group_size,
):
    torch.manual_seed(37)
    hidden_states = (
        torch.randn((num_tokens, hidden_size), dtype=dtype, device=device) * 0.1
    )
    weights = []
    refs = []
    scales = []
    lut = e4m3_lut(device)
    for output_size, input_size in (
        (2 * intermediate_size, hidden_size),
        (hidden_size, intermediate_size),
    ):
        raw = torch.randint(
            0,
            256,
            (num_experts, output_size, input_size),
            dtype=torch.uint8,
            device=device,
        )
        raw = torch.where(raw & 127 == 127, raw - 1, raw)
        groups = 1 if group_size == -1 else input_size // group_size
        scale = (
            torch.rand(
                (num_experts, output_size, groups), dtype=torch.float32, device=device
            )
            * 0.001
            + 0.0005
        ).to(dtype)
        expanded_scale = (
            scale.float().expand(num_experts, output_size, input_size)
            if group_size == -1
            else scale.float().repeat_interleave(group_size, dim=-1)
        )
        weights.append(raw.view(torch.float8_e4m3fn))
        refs.append((lut[raw.long()] * expanded_scale).to(dtype))
        scales.append(scale)
    topk_ids = (
        torch.rand((num_tokens, num_experts), device=device).topk(topk, dim=-1).indices
    )
    topk_weights = torch.softmax(
        torch.randn((num_tokens, topk), device=device), dim=-1
    ).to(dtype)
    return (
        hidden_states,
        weights[0],
        weights[1],
        refs[0],
        refs[1],
        topk_weights,
        topk_ids,
        scales[0],
        scales[1],
    )


@pytest.mark.skipif(False, reason="T-Head dispatch is only available on T-Head devices")
def test_fused_marlin_moe_uses_thead_specialization():
    assert (
        flaggems_vllm.fused_marlin_moe.__module__
        == "flaggems_vllm.runtime.backend._thead.fused.fused_marlin_moe"
    )


@pytest.mark.fused_marlin_moe_w4a16
@pytest.mark.skipif(False, reason="This test forces the T-Head expert-grouped AIU path")
def test_fused_marlin_moe_w4a16_int4_ppu_grouped():
    from tests.test_fused_marlin_moe import (
        _make_inputs_w4a16_int4,
        _reference_swiglu_moe,
        compute_max_diff,
    )

    config = (32, 8, 128, 256, 2)
    (hs, w1_q, w2_q, w1_ref, w2_ref, tw, ti, w1s, w2s) = _make_inputs_w4a16_int4(
        *config, torch.bfloat16, flaggems_vllm.device
    )
    result = flaggems_vllm.fused_marlin_moe(
        hidden_states=hs,
        w1=w1_q,
        w2=w2_q,
        bias1=None,
        bias2=None,
        w1_scale=w1s,
        w2_scale=w2s,
        topk_weights=tw,
        topk_ids=ti,
        quant_type_id=QUANT_TYPE_UINT4B8,
    )
    ref = _reference_swiglu_moe(hs, w1_ref, w2_ref, tw, ti)
    torch_device_fn.synchronize()
    max_diff = compute_max_diff(result.float(), ref)
    assert max_diff < 0.04, f"max_diff={max_diff:.4f}"


@pytest.mark.fused_marlin_moe_w4a16
@pytest.mark.skipif(
    False, reason="This test covers the T-Head fused GEMM2/top-k reduction path"
)
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_fused_marlin_moe_w4a16_int4_ppu_reduce_direct(apply_router_weight_on_input):
    from tests.test_fused_marlin_moe import (
        _make_inputs_w4a16_int4,
        _reference_swiglu_moe,
        compute_max_diff,
    )

    config = (2, 8, 128, 256, 2)
    (hs, w1_q, w2_q, w1_ref, w2_ref, tw, ti, w1s, w2s) = _make_inputs_w4a16_int4(
        *config, torch.bfloat16, flaggems_vllm.device
    )
    result = flaggems_vllm.fused_marlin_moe(
        hidden_states=hs,
        w1=w1_q,
        w2=w2_q,
        bias1=None,
        bias2=None,
        w1_scale=w1s,
        w2_scale=w2s,
        topk_weights=tw,
        topk_ids=ti,
        quant_type_id=QUANT_TYPE_UINT4B8,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    ref = _reference_swiglu_moe(
        hs,
        w1_ref,
        w2_ref,
        tw,
        ti,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    torch_device_fn.synchronize()
    max_diff = compute_max_diff(result.float(), ref)
    assert max_diff < 0.04, f"max_diff={max_diff:.4f}"


@pytest.mark.fused_marlin_moe_w8a16
@pytest.mark.skipif(
    False, reason="W8A16 FP8 specialization is only available on T-Head PPU"
)
@pytest.mark.parametrize("num_tokens", [1, 33])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group_size", [-1, 32, 128])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_fused_marlin_moe_w8a16_fp8_ppu(
    num_tokens, dtype, group_size, apply_router_weight_on_input
):
    from tests.test_fused_marlin_moe import _reference_swiglu_moe, compute_max_diff

    args = make_fp8_test_inputs(
        num_tokens, 4, 128, 256, 2, dtype, flaggems_vllm.device, group_size
    )
    (hs, w1, w2, w1_ref, w2_ref, tw, ti, w1s, w2s) = args
    result = flaggems_vllm.fused_marlin_moe(
        hidden_states=hs,
        w1=w1,
        w2=w2,
        bias1=None,
        bias2=None,
        w1_scale=w1s,
        w2_scale=w2s,
        topk_weights=tw,
        topk_ids=ti,
        quant_type_id=QUANT_TYPE_FP8_E4M3,
        apply_router_weight_on_input=apply_router_weight_on_input,
        group_size=group_size,
    )
    ref = _reference_swiglu_moe(
        hs,
        w1_ref,
        w2_ref,
        tw,
        ti,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    torch_device_fn.synchronize()
    max_diff = compute_max_diff(result.float(), ref)
    assert max_diff < 0.04, f"max_diff={max_diff:.4f}"


def make_test_inputs(precision, router_weight_on_input, factory, *args):
    if precision == "fp8":
        return make_fp8_test_inputs(*args)
    return factory(*args)


def supports_benchmark(precision):
    return precision in ("int4", "mxfp4", "int8", "fp8")


def has_baseline(precision):
    try:
        import_module("vllm.model_executor.layers.fused_moe.fused_marlin_moe")
        if precision == "fp8":
            import_module(
                "vllm.model_executor.layers.quantization.utils.marlin_utils_fp8"
            )
    except ImportError:
        return False
    return True


def benchmark_inputs(precision, benchmark, dtype):
    geometry = None
    weights = None
    for config in benchmark.shapes:
        (tokens, experts, hidden, intermediate, topk) = config
        next_geometry = (experts, hidden, intermediate)
        if geometry != next_geometry:
            if precision == "int4":
                weights = int4_weights(experts, hidden, intermediate, dtype)
            elif precision == "mxfp4":
                weights = mxfp4_weights(experts, hidden, intermediate)
            else:
                weights = w8_weights(
                    experts, hidden, intermediate, dtype, precision == "fp8"
                )
            geometry = next_geometry
        torch.manual_seed(7 + tokens)
        x = (
            torch.randn((tokens, hidden), device=flaggems_vllm.device, dtype=dtype)
            * 0.1
        )
        ids = (
            torch.rand((tokens, experts), device=flaggems_vllm.device)
            .topk(topk, dim=-1)
            .indices
        )
        probabilities = torch.softmax(
            torch.randn((tokens, topk), device=flaggems_vllm.device), dim=-1
        ).to(torch.float32)
        yield (x, *weights, probabilities, ids)


def baseline(precision, *inputs):
    native = import_module("vllm.model_executor.layers.fused_moe.fused_marlin_moe")
    scalar_types = import_module("vllm.scalar_type").scalar_types
    types = {
        "int4": scalar_types.uint4b8,
        "mxfp4": scalar_types.float4_e2m1f,
        "int8": scalar_types.uint8b128,
        "fp8": scalar_types.float8_e4m3fn,
    }
    (x, _w1, _w2, _s1, _s2, w1, w2, s1, s2, weights, ids) = inputs
    return native.fused_marlin_moe(
        hidden_states=x,
        w1=w1,
        w2=w2,
        bias1=None,
        bias2=None,
        w1_scale=s1,
        w2_scale=s2,
        topk_weights=weights,
        topk_ids=ids,
        quant_type_id=types[precision].id,
    )


def candidate(precision, *inputs):
    types = {
        "int4": QUANT_TYPE_UINT4B8,
        "mxfp4": QUANT_TYPE_FP4_E2M1,
        "int8": QUANT_TYPE_UINT8B128,
        "fp8": QUANT_TYPE_FP8_E4M3,
    }
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
        quant_type_id=types[precision],
        group_size=MXFP4_GROUP_SIZE if precision == "mxfp4" else GROUP_SIZE,
    )
