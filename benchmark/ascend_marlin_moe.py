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

"""Native single-device vLLM-Ascend MoE baselines for the Marlin benchmarks."""

from functools import lru_cache
from types import SimpleNamespace

import torch
from vllm_ascend.ops.fused_moe import moe_mlp
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.quantization.quant_type import QuantType

import flaggems_vllm

if hasattr(moe_mlp, "apply_moe_mlp"):
    from vllm_ascend.ops.fused_moe.dataclass.fused_experts import (
        build_fused_experts_input,
    )
    from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import build_mlp_compute_input
    from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
        build_token_dispatch_input,
    )
else:
    from vllm_ascend.ops.fused_moe.moe_runtime_args import (
        build_fused_experts_input,
        build_mlp_compute_input,
        build_token_dispatch_input,
    )


@lru_cache(maxsize=16)
def dispatcher(experts, topk):
    from vllm_ascend.utils import enable_custom_op

    # Register the plugin extension when available. The native 0.32 MoE
    # also supports the torch_npu kernels when this returns False.
    enable_custom_op()
    return TokenDispatcherWithAllGather(
        num_experts=experts, num_local_experts=experts, top_k=topk
    )


@lru_cache(maxsize=2)
def quant_method(quant_type):
    # Initialize only the inference hooks used by apply_moe_mlp. The normal
    # constructors also require a model runner and its global vLLM config.
    if quant_type == QuantType.W4A16:
        from vllm_ascend.quantization.methods.wna16.w4a16 import (
            AscendW4A16FusedMoEMethod,
        )

        method = object.__new__(AscendW4A16FusedMoEMethod)
        method.num_bits = 4
        method.pack_factor = 8
        method.group_size = 128
    else:
        from vllm_ascend.ops.fused_moe.routed_experts import (
            AscendUnquantizedFusedMoEMethod,
        )

        method = object.__new__(AscendUnquantizedFusedMoEMethod)
        method.moe = SimpleNamespace(has_bias=False)
        method.lora_context = None
        method._lora_routing = None
    method.dynamic_eplb = False
    return method


def native_moe(x, w1, w2, probabilities, ids, s1=None, s2=None, z1=None, z2=None):
    routing = dispatcher(w1.shape[0], ids.shape[1])
    quant_type = QuantType.NONE if s1 is None else QuantType.W4A16
    kwargs = {}
    if s1 is not None:
        kwargs.update(w1_scale=s1, w2_scale=s2, w1_offset=z1, w2_offset=z2)
    if hasattr(moe_mlp, "apply_moe_mlp"):
        kwargs["layer"] = SimpleNamespace(
            w13_weight=w1,
            w2_weight=w2,
            w13_weight_packed=w1,
            w2_weight_packed=w2,
            w13_weight_scale=s1,
            w2_weight_scale=s2,
            w13_weight_offset=z1,
            w2_weight_offset=z2,
        )
    inputs = build_fused_experts_input(
        hidden_states=x,
        topk_weights=probabilities.to(x.dtype),
        topk_ids=ids,
        w1=w1,
        w2=w2,
        quant_type=quant_type,
        dynamic_eplb=False,
        need_trans=False,
        **kwargs,
    )
    routed = routing.token_dispatch(
        build_token_dispatch_input(fused_experts_input=inputs)
    )
    mlp = build_mlp_compute_input(
        fused_experts_input=inputs, token_dispatch_output=routed, use_fusion_ops=False
    )
    if hasattr(moe_mlp, "apply_moe_mlp"):
        projected, _ = moe_mlp.apply_moe_mlp(mlp, quant_method(quant_type))
    else:
        projected = moe_mlp.unified_apply_mlp(mlp_compute_input=mlp)
    return routing.token_combine(projected, routed.combine_metadata)


def make_weights(experts, output_size, input_size, dtype, bits):
    """Prepare PR, reference and vendor layouts outside the timed call, one expert at a time."""
    from tests.test_fused_marlin_moe import (
        _gptq_quantize_uint4b8,
        _gptq_quantize_uint8b128,
    )

    device = flaggems_vllm.device
    codes = torch.empty(
        (experts, output_size, input_size // (8 // bits)),
        device=device,
        dtype=torch.uint8,
    )
    scales = torch.empty(
        (experts, output_size, input_size // 128), device=device, dtype=dtype
    )
    decoded = torch.empty(
        (experts, input_size, output_size), device=device, dtype=torch.bfloat16
    )
    native = decoded
    if bits == 4:
        from vllm_ascend.quantization.methods.wna16.w4a16 import pack_to_int32

        native = torch.empty(
            (experts, input_size, output_size // 8), device=device, dtype=torch.int32
        )
    for expert in range(experts):
        source = (
            torch.randn((output_size, input_size), device=device, dtype=dtype) / 10.0
        )
        if bits == 4:
            ref, nibbles, scale = _gptq_quantize_uint4b8(source, 128)
            codes[expert] = nibbles[:, ::2] | (nibbles[:, 1::2] << 4)
            signed = (nibbles.to(torch.int32) - 8).T.contiguous()
            native[expert] = pack_to_int32(signed[None])[0]
        else:
            ref, codes[expert], scale = _gptq_quantize_uint8b128(source, 128)
        scales[expert] = scale
        decoded[expert].copy_(ref.T)
    return codes, scales, decoded, native
