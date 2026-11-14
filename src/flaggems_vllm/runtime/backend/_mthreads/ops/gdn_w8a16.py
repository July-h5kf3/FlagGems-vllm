# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.gdn_w8a16 import _gdn_step, _prepare_qk, _store_dynamic_fp8_state
from flaggems_vllm.ops.gdn_w8a16 import gdn_w8a16 as _gdn_w8a16
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def _mthreads_gdn_decode_w8a16_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    raw_a,
    dt_bias,
    raw_b,
    output,
    state_fp8,
    state_scale,
    cu_seqlens,
    state_indices,
    num_accepted_tokens,
    scale,
    stride_q_b: tl.constexpr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_k: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_k: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_t: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_v: tl.constexpr,
    stride_g_b: tl.constexpr,
    stride_g_t: tl.constexpr,
    stride_g_h: tl.constexpr,
    stride_beta_b: tl.constexpr,
    stride_beta_t: tl.constexpr,
    stride_beta_h: tl.constexpr,
    stride_raw_a_b: tl.constexpr,
    stride_raw_a_t: tl.constexpr,
    stride_raw_a_h: tl.constexpr,
    stride_raw_b_b: tl.constexpr,
    stride_raw_b_t: tl.constexpr,
    stride_raw_b_h: tl.constexpr,
    USE_RAW_GATES: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_v: tl.constexpr,
    stride_state_s: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_scale_s: tl.constexpr,
    stride_scale_h: tl.constexpr,
    stride_scale_v: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_PACKED_INPUT: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    USE_SPEC_DECODING: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    LOW_PRECISION_AMAX: tl.constexpr,
):
    value_tile = tl.program_id(0)
    sequence_head = tl.program_id(1)
    sequence = sequence_head // HV
    value_head = sequence_head % HV
    query_head = value_head // (HV // H)

    if USE_PACKED_INPUT:
        batch = 0
        token = tl.load(cu_seqlens + sequence).to(tl.int64)
    else:
        batch = sequence
        token = 0

    if USE_STATE_INDICES:
        index_offset = sequence * stride_indices_seq
        initial_offset = index_offset
        if USE_SPEC_DECODING:
            accepted = tl.load(num_accepted_tokens + sequence).to(tl.int64)
            initial_offset += (accepted - 1) * stride_indices_tok
        initial_slot = tl.load(state_indices + initial_offset).to(tl.int64)
        output_slot = tl.load(state_indices + index_offset).to(tl.int64)
        if initial_slot < 0:
            return
        output_valid = output_slot >= 0
    else:
        initial_slot = sequence
        output_slot = sequence
        output_valid = True

    keys = tl.arange(0, BK)
    columns = value_tile * BV + tl.arange(0, BV)
    mask_k = keys < K
    mask_v = columns < V
    mask_state = mask_k[:, None] & mask_v[None, :]
    state_offsets = (
        initial_slot * stride_state_s
        + value_head * stride_state_h
        + keys[:, None] * stride_state_k
        + columns[None, :] * stride_state_v
    )
    scale_offsets = (
        initial_slot * stride_scale_s
        + value_head * stride_scale_h
        + columns * stride_scale_v
    )
    state_acc = _load_state(
        state_fp8,
        state_scale,
        state_offsets,
        scale_offsets,
        mask_state,
        mask_v,
        ACTIVATION_TYPE=q.dtype.element_ty,
    )

    q_offsets = (
        batch * stride_q_b
        + token * stride_q_t
        + query_head * stride_q_h
        + keys * stride_q_k
    )
    k_offsets = (
        batch * stride_k_b
        + token * stride_k_t
        + query_head * stride_k_h
        + keys * stride_k_k
    )
    q_values = tl.load(q + q_offsets, mask=mask_k, other=0.0)
    k_values = tl.load(k + k_offsets, mask=mask_k, other=0.0)
    q_low, k_low = _prepare_qk(
        q_values,
        k_values,
        scale,
        ACTIVATION_TYPE=q.dtype.element_ty,
        USE_QK_L2NORM=USE_QK_L2NORM,
    )
    value_offsets = (
        batch * stride_v_b
        + token * stride_v_t
        + value_head * stride_v_h
        + columns * stride_v_v
    )
    value = tl.load(v + value_offsets, mask=mask_v, other=0.0)
    gate_offset = batch * stride_g_b + token * stride_g_t + value_head * stride_g_h
    beta_offset = (
        batch * stride_beta_b + token * stride_beta_t + value_head * stride_beta_h
    )
    if USE_RAW_GATES:
        raw_a_offset = (
            batch * stride_raw_a_b
            + token * stride_raw_a_t
            + value_head * stride_raw_a_h
        )
        raw_b_offset = (
            batch * stride_raw_b_b
            + token * stride_raw_b_t
            + value_head * stride_raw_b_h
        )
        gate_input = tl.load(raw_a + raw_a_offset).to(tl.float32)
        gate_input += tl.load(dt_bias + value_head).to(tl.float32)
        softplus = tl.where(
            gate_input <= 20.0, tl.log(1.0 + tl.exp(gate_input)), gate_input
        )
        log_decay = -tl.exp(tl.load(A_log + value_head).to(tl.float32)) * softplus
        decay = tl.exp(log_decay)
        update = tl.sigmoid(tl.load(raw_b + raw_b_offset).to(tl.float32))
    else:
        decay = tl.exp(tl.load(g + gate_offset).to(tl.float32))
        update = tl.load(beta + beta_offset)
    state_acc, result = _gdn_step(
        state_acc,
        q_low,
        k_low,
        value,
        decay,
        update,
        ACTIVATION_TYPE=q.dtype.element_ty,
        FP32_STATE_PRODUCTS=True,
        FP32_UPDATE_PRODUCTS=True,
    )
    output_offsets = (
        batch * stride_o_b
        + token * stride_o_t
        + value_head * stride_o_h
        + columns * stride_o_v
    )
    tl.store(output + output_offsets, result.to(output.dtype.element_ty), mask=mask_v)

    safe_output_slot = tl.where(output_valid, output_slot, 0)
    output_state_offsets = (
        safe_output_slot * stride_state_s
        + value_head * stride_state_h
        + keys[:, None] * stride_state_k
        + columns[None, :] * stride_state_v
    )
    output_scale_offsets = (
        safe_output_slot * stride_scale_s
        + value_head * stride_scale_h
        + columns * stride_scale_v
    )
    _store_dynamic_fp8_state(
        state_acc,
        state_fp8,
        state_scale,
        output_state_offsets,
        output_scale_offsets,
        mask_state & output_valid,
        mask_v & output_valid,
        ACTIVATION_TYPE=q.dtype.element_ty,
        LOW_PRECISION_AMAX=LOW_PRECISION_AMAX,
    )


@triton.jit
def _load_state(
    state_fp8,
    state_scale,
    state_offsets,
    scale_offsets,
    mask_state,
    mask_v,
    ACTIVATION_TYPE: tl.constexpr,
):
    quantized = tl.load(state_fp8 + state_offsets, mask_state, 0.0).to(ACTIVATION_TYPE)
    scales = tl.load(state_scale + scale_offsets, mask_v, 0.0).to(ACTIVATION_TYPE)
    return (quantized * scales[None, :]).to(tl.float32)


def _mthreads_gdn_decode_w8a16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor | None,
    scale: float,
    state_fp8: torch.Tensor,
    state_scale: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    A_log: torch.Tensor | None,
    raw_a: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    raw_b: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, tokens, query_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    sequence_count = batch if cu_seqlens is None else cu_seqlens.numel() - 1
    if cu_seqlens is not None and tokens != sequence_count:
        raise NotImplementedError(
            "packed MThreads GDN decode requires one token per sequence"
        )
    if key_dim != 128 or value_dim != 128:
        raise NotImplementedError("MThreads GDN decode currently supports K=V=128")
    if state_indices is not None and state_indices.ndim not in (1, 2):
        raise ValueError("state indices must be one- or two-dimensional")
    if num_accepted_tokens is not None:
        if state_indices is None or state_indices.ndim != 2:
            raise ValueError("accepted-token metadata requires per-token state indices")
        if num_accepted_tokens.shape != (sequence_count,):
            raise ValueError("accepted-token metadata must have shape [num_sequences]")
    use_raw_gates = A_log is not None
    if use_raw_gates and any(tensor is None for tensor in (raw_a, dt_bias, raw_b)):
        raise ValueError("raw GDN gating requires A_log, a, dt_bias, and b")
    if not use_raw_gates and (g is None or beta is None):
        raise ValueError("precomputed GDN gating requires g and beta")
    gate_tensor = g if g is not None else A_log
    beta_tensor = beta if beta is not None else raw_b
    output = torch.empty(
        batch, tokens, value_heads, value_dim, device=q.device, dtype=q.dtype
    )
    if not sequence_count:
        return output, state_fp8, state_scale
    block_k = triton.next_power_of_2(key_dim)
    block_v = 32
    with torch_device_fn.device(q.device):
        _mthreads_gdn_decode_w8a16_kernel[
            (triton.cdiv(value_dim, block_v), sequence_count * value_heads)
        ](
            q,
            k,
            v,
            gate_tensor,
            beta_tensor,
            A_log if A_log is not None else gate_tensor,
            raw_a if raw_a is not None else q,
            dt_bias if dt_bias is not None else gate_tensor,
            raw_b if raw_b is not None else beta_tensor,
            output,
            state_fp8,
            state_scale,
            cu_seqlens,
            state_indices,
            num_accepted_tokens,
            scale,
            stride_q_b=q.stride(0),
            stride_q_t=q.stride(1),
            stride_q_h=q.stride(2),
            stride_q_k=q.stride(3),
            stride_k_b=k.stride(0),
            stride_k_t=k.stride(1),
            stride_k_h=k.stride(2),
            stride_k_k=k.stride(3),
            stride_v_b=v.stride(0),
            stride_v_t=v.stride(1),
            stride_v_h=v.stride(2),
            stride_v_v=v.stride(3),
            stride_g_b=gate_tensor.stride(0),
            stride_g_t=gate_tensor.stride(1) if gate_tensor.ndim == 3 else 0,
            stride_g_h=gate_tensor.stride(2) if gate_tensor.ndim == 3 else 0,
            stride_beta_b=beta_tensor.stride(0),
            stride_beta_t=beta_tensor.stride(1),
            stride_beta_h=beta_tensor.stride(2) if beta_tensor.ndim == 3 else 0,
            stride_raw_a_b=raw_a.stride(0) if raw_a is not None else 0,
            stride_raw_a_t=raw_a.stride(1) if raw_a is not None else 0,
            stride_raw_a_h=raw_a.stride(2) if raw_a is not None else 0,
            stride_raw_b_b=raw_b.stride(0) if raw_b is not None else 0,
            stride_raw_b_t=raw_b.stride(1) if raw_b is not None else 0,
            stride_raw_b_h=raw_b.stride(2) if raw_b is not None else 0,
            USE_RAW_GATES=use_raw_gates,
            stride_o_b=output.stride(0),
            stride_o_t=output.stride(1),
            stride_o_h=output.stride(2),
            stride_o_v=output.stride(3),
            stride_state_s=state_fp8.stride(0),
            stride_state_h=state_fp8.stride(1),
            stride_state_k=state_fp8.stride(2),
            stride_state_v=state_fp8.stride(3),
            stride_scale_s=state_scale.stride(0),
            stride_scale_h=state_scale.stride(1),
            stride_scale_v=state_scale.stride(2),
            stride_indices_seq=1 if state_indices is None else state_indices.stride(0),
            stride_indices_tok=(
                0
                if state_indices is None or state_indices.ndim == 1
                else state_indices.stride(1)
            ),
            H=query_heads,
            HV=value_heads,
            K=key_dim,
            V=value_dim,
            S=state_fp8.shape[0],
            BK=block_k,
            BV=block_v,
            USE_PACKED_INPUT=cu_seqlens is not None,
            USE_STATE_INDICES=state_indices is not None,
            USE_SPEC_DECODING=num_accepted_tokens is not None,
            USE_QK_L2NORM=use_qk_l2norm_in_kernel,
            LOW_PRECISION_AMAX=True,
            num_warps=1 if block_v == 4 else 4,
            num_stages=1,
        )
    return output, state_fp8, state_scale


def gdn_w8a16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor | None,
    scale: float,
    state_fp8: torch.Tensor,
    state_scale: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    max_sequence_length: int | None = None,
    A_log: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run GDN with MThreads-native raw or precomputed gates."""
    single_token_decode = (
        max_sequence_length == 1 if cu_seqlens is not None else q.shape[1] == 1
    )
    raw_gates = A_log is not None
    if single_token_decode:
        return _mthreads_gdn_decode_w8a16(
            q,
            k,
            v,
            g,
            beta,
            scale,
            state_fp8,
            state_scale,
            cu_seqlens,
            ssm_state_indices,
            num_accepted_tokens,
            use_qk_l2norm_in_kernel,
            A_log,
            a,
            dt_bias,
            b,
        )
    if raw_gates:
        if num_accepted_tokens is not None:
            raise NotImplementedError(
                "raw GDN prefill does not produce speculative checkpoints"
            )
        from flaggems_vllm.runtime.backend._mthreads.ops.gdn_w8a16_decode import (
            raw_prefill_w8a16_fp8,
        )

        return raw_prefill_w8a16_fp8(
            q,
            k,
            v,
            scale,
            state_fp8,
            state_scale,
            cu_seqlens,
            ssm_state_indices,
            use_qk_l2norm_in_kernel,
            A_log,
            a,
            dt_bias,
            b,
        )
    if num_accepted_tokens is None and (
        ssm_state_indices is None or ssm_state_indices.ndim == 1
    ):
        from flaggems_vllm.runtime.backend._mthreads.ops.gdn_w8a16_decode import (
            recurrent_prefill_w8a16_fp8,
        )

        return recurrent_prefill_w8a16_fp8(
            q,
            k,
            v,
            g,
            beta,
            scale,
            state_fp8,
            state_scale,
            cu_seqlens,
            ssm_state_indices,
            use_qk_l2norm_in_kernel,
        )
    return _gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        scale,
        state_fp8,
        state_scale,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        max_sequence_length=max_sequence_length,
    )


__all__ = ["gdn_w8a16"]
