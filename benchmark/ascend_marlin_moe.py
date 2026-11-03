# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Call the installed vLLM-Ascend 0.24.0rc1 BF16/FP16 MoE implementation."""

from functools import lru_cache

import torch
from vllm_ascend.ops.fused_moe.moe_mlp import MoECommType, unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    build_fused_experts_input,
    build_mlp_compute_input,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_custom_op


@lru_cache(maxsize=16)
def dispatcher(experts, topk):
    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend custom operators could not be loaded")
    return TokenDispatcherWithAllGather(
        num_experts=experts, num_local_experts=experts, top_k=topk
    )


def native_moe(
    hidden_states,
    w1,
    w2,
    probabilities,
    expert_ids,
    *,
    quant_type=QuantType.NONE,
    w1_scale=None,
    w2_scale=None,
    w1_offset=None,
    w2_offset=None,
):
    routing = dispatcher(w1.shape[0], expert_ids.shape[1])
    inputs = build_fused_experts_input(
        hidden_states=hidden_states,
        topk_weights=probabilities.to(hidden_states.dtype),
        topk_ids=expert_ids,
        w1=w1,
        w2=w2,
        quant_type=quant_type,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_offset=w1_offset,
        w2_offset=w2_offset,
        dynamic_eplb=False,
        need_trans=False,
    )
    routed = routing.token_dispatch(
        build_token_dispatch_input(fused_experts_input=inputs)
    )
    expert_outputs, _ = unified_apply_mlp(
        mlp_compute_input=build_mlp_compute_input(
            fused_experts_input=inputs,
            token_dispatch_output=routed,
            use_fusion_ops=False,
        )
    )
    return routing.token_combine(expert_outputs, routed.combine_metadata)


native_bf16 = native_moe


def native_int4_moe(hidden_states, w1, w2, probabilities, expert_ids, s1, s2, o1, o2):
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    with override_forward_context(ForwardContext({}, {}, {})):
        _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
        return native_moe(
            hidden_states,
            w1,
            w2,
            probabilities,
            expert_ids,
            quant_type=QuantType.W4A16,
            w1_scale=s1,
            w2_scale=s2,
            w1_offset=o1,
            w2_offset=o2,
        )


def int8_input_iter(shapes, dtype):
    import flaggems_vllm
    from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT8B128
    from tests.accuracy_utils import gems_assert_close
    from tests.test_fused_marlin_moe import (
        _make_inputs_w8a16_int8,
        _reference_w8a16_grouped,
        compute_max_diff,
    )

    geometry = None
    for num_tokens, experts, hidden, intermediate, topk in shapes:
        if geometry != (experts, hidden, intermediate):
            _, w1, w2, ref1, ref2, _, _, s1, s2 = _make_inputs_w8a16_int8(
                1, experts, hidden, intermediate, topk, dtype, flaggems_vllm.device
            )
            baseline1 = ref1.to(torch.bfloat16).transpose(1, 2).contiguous()
            baseline2 = ref2.to(torch.bfloat16).transpose(1, 2).contiguous()
            fp16_references = (
                (ref1.transpose(1, 2).contiguous(), ref2.transpose(1, 2).contiguous())
                if dtype == torch.float16
                else None
            )
            geometry = (experts, hidden, intermediate)
        else:
            pass
        torch.manual_seed(num_tokens)
        hidden_states = torch.randn(
            (num_tokens, hidden), device=flaggems_vllm.device, dtype=dtype
        )
        probabilities, ids = torch.randn(
            (num_tokens, experts), device=flaggems_vllm.device
        ).topk(topk)
        probabilities = torch.softmax(probabilities, dim=-1)
        ids = ids.to(torch.int32)
        expected = _reference_w8a16_grouped(
            hidden_states, ref1, ref2, probabilities, ids
        )
        baseline_hidden = hidden_states.to(torch.bfloat16)
        baseline_expected = (
            expected
            if dtype == torch.bfloat16
            else _reference_w8a16_grouped(
                baseline_hidden,
                baseline1.transpose(1, 2),
                baseline2.transpose(1, 2),
                probabilities,
                ids,
            )
        )
        outputs = (
            native_moe(baseline_hidden, baseline1, baseline2, probabilities, ids),
            flaggems_vllm.fused_marlin_moe(
                hidden_states,
                w1,
                w2,
                None,
                None,
                s1,
                s2,
                probabilities,
                ids,
                QUANT_TYPE_UINT8B128,
            ),
        )
        for output, reference in zip(outputs, (baseline_expected, expected)):
            error = compute_max_diff(output.float(), reference)
            gems_assert_close(error, torch.zeros_like(error), torch.float32, atol=0.04)
        yield hidden_states, w1, w2, s1, s2, baseline1, baseline2, (
            baseline_hidden if dtype == torch.float16 else None
        ), fp16_references, probabilities, ids


def int4_input_iter(shapes, dtype):
    from vllm_ascend.quantization.methods.w4a16 import pack_to_int32

    import flaggems_vllm
    from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT4B8
    from tests.accuracy_utils import gems_assert_close
    from tests.test_fused_marlin_moe import (
        _quantize_moe_weight,
        _reference_w8a16_grouped,
        compute_max_diff,
    )

    if dtype != torch.bfloat16:
        raise ValueError("Ascend INT4 requires BF16 activations and scales")
    geometry = None
    for tokens, experts, hidden, intermediate, topk in shapes:
        if geometry != (experts, hidden, intermediate):
            weights = []
            torch.manual_seed(0)
            for output_features, input_features in (
                (2 * intermediate, hidden),
                (hidden, intermediate),
            ):
                packed = torch.empty(
                    (experts, output_features, input_features // 2),
                    device=flaggems_vllm.device,
                    dtype=torch.uint8,
                )
                scales = torch.empty(
                    (experts, output_features, input_features // 128),
                    device=flaggems_vllm.device,
                    dtype=dtype,
                )
                decoded = torch.empty(
                    (experts, input_features, output_features),
                    device=flaggems_vllm.device,
                    dtype=dtype,
                )
                native = torch.empty(
                    (experts, input_features, output_features // 8),
                    device=flaggems_vllm.device,
                    dtype=torch.int32,
                )
                # Quantizing one expert at a time bounds preparation memory.
                for expert in range(experts):
                    source = (
                        torch.randn(
                            (1, output_features, input_features),
                            device=flaggems_vllm.device,
                            dtype=dtype,
                        )
                        / 10.0
                    )
                    codes, reference, scale = _quantize_moe_weight(source, 128)
                    packed[expert] = codes[0]
                    scales[expert] = scale[0]
                    decoded[expert] = reference[0].T
                    unpacked = torch.stack(
                        (codes[0] & 15, codes[0] >> 4), dim=-1
                    ).reshape(output_features, input_features)
                    signed = (unpacked.to(torch.int32) - 8).T.contiguous()
                    native[expert] = pack_to_int32(signed[None, :, :])[0]
                native_scale = scales.transpose(1, 2).contiguous()
                weights.append(
                    (
                        packed,
                        scales,
                        decoded,
                        native,
                        native_scale,
                        torch.zeros_like(native_scale),
                    )
                )
            (w1, s1, ref1, baseline1, bs1, bo1), (w2, s2, ref2, baseline2, bs2, bo2) = (
                weights
            )
            geometry = (experts, hidden, intermediate)
        torch.manual_seed(tokens)
        hidden_states = (
            torch.randn((tokens, hidden), device=flaggems_vllm.device, dtype=dtype)
            / 10.0
        )
        probabilities, ids = torch.softmax(
            torch.randn((tokens, experts), device=flaggems_vllm.device), dim=-1
        ).topk(topk)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        ids = ids.to(torch.int32)
        expected = _reference_w8a16_grouped(
            hidden_states,
            ref1.transpose(1, 2),
            ref2.transpose(1, 2),
            probabilities,
            ids,
        )
        outputs = (
            native_int4_moe(
                hidden_states,
                baseline1,
                baseline2,
                probabilities,
                ids,
                bs1,
                bs2,
                bo1,
                bo2,
            ),
            flaggems_vllm.fused_marlin_moe(
                hidden_states,
                w1,
                w2,
                None,
                None,
                s1,
                s2,
                probabilities,
                ids,
                QUANT_TYPE_UINT4B8,
            ),
        )
        for output in outputs:
            error = compute_max_diff(output.float(), expected)
            gems_assert_close(error, torch.zeros_like(error), torch.float32, atol=0.04)
        yield hidden_states, w1, w2, s1, s2, baseline1, baseline2, bs1, bs2, probabilities, ids, bo1, bo2
