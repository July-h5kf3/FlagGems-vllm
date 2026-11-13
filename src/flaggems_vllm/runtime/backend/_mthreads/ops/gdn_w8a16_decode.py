# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._mthreads.ops.gdn_w8a16_tle import (
    HAS_TLE,
    store_scaled_state,
)
from flaggems_vllm.utils import libentry

if HAS_TLE:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    from flaggems_vllm.runtime.backend._mthreads.ops.gdn_w8a16_packed import (
        GDN_VECTOR_LIBRARY,
        decode_quart,
    )
    from flaggems_vllm.runtime.backend._mthreads.ops.gdn_w8a16_tle import (
        decode_single_tle,
    )


@libentry()
@triton.jit
def decode_vk_kernel(
    Q,
    Key,
    Value,
    State,
    Scales,
    Alog,
    A,
    Bias,
    B,
    Out,
    Index,
    History,
    HistScale,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    T: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    K3: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    V3: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    O2: tl.constexpr,
    O3: tl.constexpr,
    IX: tl.constexpr,
    HIST_T: tl.constexpr,
    SCALE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    INDEXED: tl.constexpr,
    SAVE_STATE: tl.constexpr,
    SAVE_HISTORY: tl.constexpr,
    Cu,
    VARLEN: tl.constexpr,
    RAW_GATES: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    FinalState=None,
    FinalScale=None,
    HasInitial=None,
    PREFILL: tl.constexpr = False,
    HAS_INITIAL_MASK: tl.constexpr = False,
    HAS_INITIAL_STATE: tl.constexpr = False,
    HAS_FORGET_GATE: tl.constexpr = False,
    HAS_UPDATE_GATE: tl.constexpr = False,
    LOG_SPACE: tl.constexpr = True,
    KEY_HEADS: tl.constexpr = 1,
    VALUE_HEADS: tl.constexpr = 1,
    HAS_INITIAL_STRIDE: tl.constexpr = 0,
    FS0: tl.constexpr = 0,
    FS1: tl.constexpr = 0,
    FS2: tl.constexpr = 0,
    FS3: tl.constexpr = 0,
    FC0: tl.constexpr = 0,
    FC1: tl.constexpr = 0,
    FC2: tl.constexpr = 0,
    TLE_PREFILL: tl.constexpr = False,
):
    sequence_head = tl.program_id(0)
    sequence, head = sequence_head // HV, sequence_head % HV
    query_head = head // (HV // H)
    if VARLEN:
        batch = 0
        begin = tl.load(Cu + sequence)
        length = tl.load(Cu + sequence + 1) - begin
    else:
        batch, begin, length = sequence, 0, T
    if not PREFILL and length <= 0:
        return
    if TLE_PREFILL:
        # One warp reduces a complete 128-key row without exchanging partial sums.
        layout: tl.constexpr = tle.gpu.BlockEncoding([1, 4], [1, 32], [4, 1], [1, 0])
        row = tle.gpu.set_layout(tl.arange(0, BV), tle.gpu.SlicedEncoding(1, layout))
        reduction = tle.gpu.set_layout(
            tl.arange(0, BK), tle.gpu.SlicedEncoding(0, layout)
        )
    else:
        row = tl.arange(0, BV)
        reduction = tl.arange(0, BK)
    columns = tl.program_id(1) * BV + row
    slot = (
        tl.load(Index + sequence * IX).to(tl.int64)
        if INDEXED
        else sequence.to(tl.int64)
    )
    valid_v = columns < DV
    mask = valid_v[:, None] & (reduction[None, :] < DK)
    if slot < 0:
        for step in range(length):
            token = begin + step
            tl.store(
                Out + batch * O0 + token * O1 + head * O2 + columns * O3,
                0.0,
                valid_v,
            )
        return
    state_offset = (
        slot * S0 + head * S1 + columns[:, None] * S2 + reduction[None, :] * S3
    )
    scale_offset = slot * C0 + head * C1 + columns * C2
    if PREFILL:
        key_head = head // (HV // KEY_HEADS)
        value_head = head // (HV // VALUE_HEADS)
        state = tl.full((BV, BK), 0.0, tl.float32)
        if HAS_INITIAL_STATE:
            read_initial = (
                tl.load(HasInitial + sequence * HAS_INITIAL_STRIDE)
                if HAS_INITIAL_MASK
                else True
            )
            state = tl.load(State + state_offset, mask & read_initial, 0.0).to(
                tl.float32
            )
            state *= tl.load(Scales + scale_offset, valid_v & read_initial, 0.0)[
                :, None
            ]
    else:
        key_head, value_head = query_head, head
        state = tl.load(State + state_offset, mask, 0.0).to(tl.float32)
        state *= tl.load(Scales + scale_offset, valid_v, 0.0)[:, None]
    if RAW_GATES:
        alpha = tl.exp2(tl.load(Alog + head).to(tl.float32) * 1.4426950408889634)
        bias = tl.load(Bias + head).to(tl.float32)
    for step in range(length):
        token = begin + step
        query = tl.load(
            Q + batch * Q0 + token * Q1 + query_head * Q2 + reduction * Q3,
            reduction < DK,
            0.0,
        ).to(tl.float32)
        key = tl.load(
            Key + batch * K0 + token * K1 + key_head * K2 + reduction * K3,
            reduction < DK,
            0.0,
        ).to(tl.float32)
        if NORMALIZE:
            # The stock MP31 rsqrt can return NaN for +inf. Finite BF16
            # components with an overflowing squared norm must normalize to zero.
            qnorm = tl.sum(query * query, 0) + 1.0e-6
            knorm = tl.sum(key * key, 0) + 1.0e-6
            query = tl.where(qnorm == float("inf"), 0.0, query * tl.rsqrt(qnorm))
            key = tl.where(knorm == float("inf"), 0.0, key * tl.rsqrt(knorm))
        query *= SCALE
        if RAW_GATES:
            gate = (
                tl.load(A + batch * A0 + token * A1 + head * A2).to(tl.float32) + bias
            )
            softplus = tl.where(
                gate > 20.0,
                gate,
                tl.log2(1.0 + tl.exp2(gate * 1.4426950408889634)) * 0.6931471805599453,
            )
            decay = tl.exp2(-alpha * softplus * 1.4426950408889634)
            beta = tl.sigmoid(
                tl.load(B + batch * B0 + token * B1 + head * B2).to(tl.float32)
            )
        else:
            if PREFILL:
                decay = 1.0
                beta = 1.0
                if HAS_FORGET_GATE:
                    gate = tl.load(A + batch * A0 + token * A1 + head * A2).to(
                        tl.float32
                    )
                    decay = tl.exp2(gate * 1.4426950408889634) if LOG_SPACE else gate
                if HAS_UPDATE_GATE:
                    beta = tl.load(B + batch * B0 + token * B1 + head * B2).to(
                        tl.float32
                    )
            else:
                decay = tl.exp2(
                    tl.load(A + batch * A0 + token * A1 + head * A2).to(tl.float32)
                    * 1.4426950408889634
                )
                beta = tl.load(B + batch * B0 + token * B1 + head * B2).to(tl.float32)
        state *= decay
        value = tl.load(
            Value + batch * V0 + token * V1 + value_head * V2 + columns * V3,
            valid_v,
            0.0,
        ).to(tl.float32)
        delta = (value - tl.sum(state * key[None, :], 1)) * beta
        state = tl.fma(delta[:, None], key[None, :], state)
        result = tl.sum(state * query[None, :], 1)
        tl.store(
            Out + batch * O0 + token * O1 + head * O2 + columns * O3, result, valid_v
        )
        if SAVE_HISTORY:
            history_group = (sequence * HIST_T + step) * HV + head
            history_offset = (history_group * DV + columns[:, None]) * DK + reduction[
                None, :
            ]
            store_scaled_state(
                state,
                History,
                HistScale,
                history_offset,
                history_group * DV + columns,
                mask,
                valid_v,
            )
    if SAVE_STATE:
        if PREFILL:
            final_offset = (
                slot * FS0
                + head * FS1
                + columns[:, None] * FS2
                + reduction[None, :] * FS3
            )
            final_scale_offset = slot * FC0 + head * FC1 + columns * FC2
            store_scaled_state(
                state,
                FinalState,
                FinalScale,
                final_offset,
                final_scale_offset,
                mask,
                valid_v,
            )
        else:
            store_scaled_state(
                state, State, Scales, state_offset, scale_offset, mask, valid_v
            )


def gated_delta_rule_decode_w8a16_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    state_layout: str = "VK",
    state_indices: torch.Tensor | None = None,
    scale: float | None = None,
    output: torch.Tensor | None = None,
    intermediate_states_buffer: torch.Tensor | None = None,
    disable_state_update: bool = False,
    use_qk_l2norm: bool = True,
    *,
    state_scale: torch.Tensor,
    intermediate_state_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MATE decode/MTP contract with FP8 VK state and explicit channel scales."""
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("expected Q/K[B,T,H,K] and V[B,T,HV,V]")
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if (
        min(heads, key_dim, value_heads, value_dim) < 1
        or value_heads % heads
        or v.shape[:2] != q.shape[:2]
    ):
        raise ValueError("invalid QKV dimensions")
    if state_layout != "VK":
        raise NotImplementedError("MThreads native GDN supports VK state only")
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError("QKV must share BF16 or FP16 dtype")
    if state.ndim != 4 or state.shape[1:] != (value_heads, value_dim, key_dim):
        raise ValueError("state must have shape [pool,HV,V,K]")
    if state.dtype != torch.float8_e4m3fn or state_scale.dtype != torch.float32:
        raise TypeError("expected E4M3 state and FP32 scales")
    if state_scale.shape != state.shape[:3]:
        raise ValueError("state_scale must have shape [pool,HV,V]")
    if (
        a.shape != (batch, tokens, value_heads)
        or b.shape != a.shape
        or a.dtype != q.dtype
        or b.dtype != q.dtype
    ):
        raise ValueError("a/b must have shape [B,T,HV] and match activations")
    if (
        A_log.shape != (value_heads,)
        or dt_bias.shape != A_log.shape
        or A_log.dtype != torch.float32
        or dt_bias.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError("invalid A_log or dt_bias")
    if state_indices is not None:
        if state_indices.shape != (batch,) or state_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("state_indices must be int32/int64 [B]")
    elif state.shape[0] != batch:
        raise ValueError("identity mapping requires one state per sequence")
    if scale is not None and type(scale) not in (float, int):
        raise TypeError("scale must be a Python number")
    if type(use_qk_l2norm) is not bool or type(disable_state_update) is not bool:
        raise TypeError("controls must be Python booleans")
    if tokens == 1 and (disable_state_update or intermediate_states_buffer is not None):
        raise NotImplementedError(
            "MATE T=1 does not support rollback or read-only state"
        )
    if tokens > 1 and (key_dim != 128 or value_dim != 128):
        raise NotImplementedError("MATE MTP supports K=V=128")
    history = intermediate_states_buffer
    tensors = [q, k, v, state, state_scale, A_log, a, dt_bias, b]
    if state_indices is not None:
        tensors.append(state_indices)
    if history is not None:
        if (
            history.ndim != 5
            or history.shape[0] < batch
            or history.shape[1] < tokens
            or history.shape[2:] != state.shape[1:]
            or history.dtype != state.dtype
            or not history.is_contiguous()
        ):
            raise ValueError(
                "rollback state must be contiguous FP8 [buffer,steps,HV,V,K]"
            )
        if (
            intermediate_state_scale is None
            or intermediate_state_scale.shape != history.shape[:4]
            or intermediate_state_scale.dtype != torch.float32
            or not intermediate_state_scale.is_contiguous()
        ):
            raise ValueError(
                "rollback FP8 state requires FP32 [buffer,steps,HV,V] scales"
            )
        tensors.extend([history, intermediate_state_scale])
    elif intermediate_state_scale is not None:
        raise ValueError("rollback scales require a rollback state buffer")
    if output is not None:
        if output.shape != (
            batch,
            tokens,
            value_heads,
            value_dim,
        ) or output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("output must be [B,T,HV,V] FP16/BF16/FP32")
        tensors.append(output)
    if q.device.type != "musa" or any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must be on one MUSA device")
    if any(t.requires_grad for t in tensors):
        raise NotImplementedError("GDN is forward-only")
    if any(stride < 0 for t in tensors for stride in t.stride()):
        raise ValueError("negative strides are unsupported")
    if output is None:
        output = torch.empty(
            (
                (batch, value_heads, value_dim)
                if tokens == 1
                else (batch, tokens, value_heads, value_dim)
            ),
            device=q.device,
            dtype=q.dtype,
        )
    out_view = output.unsqueeze(1) if output.ndim == 3 else output
    if not batch or not tokens:
        return output, state
    single = HAS_TLE and tokens == 1 and key_dim == value_dim == 128 and use_qk_l2norm
    packed = (
        single
        and torch.musa.get_device_capability(q.device) == (3, 1)
        and state.is_contiguous()
        and state.data_ptr() % 16 == 0
        and all(
            t.stride(3) == 1
            and t.data_ptr() % 8 == 0
            and t.stride(0) % 4 == 0
            and t.stride(2) % 4 == 0
            for t in (q, k)
        )
    )
    use_tme = (
        packed
        and os.environ.get("FLAGGEMS_MTHREADS_GDN_TME") == "1"
        and state.numel() // 128 < 2**31
    )
    if single and (packed or batch <= 3):
        value_tile = 8 if batch == 1 else 16
        args = (
            q,
            k,
            v,
            state,
            state_scale,
            A_log,
            a,
            dt_bias,
            b,
            out_view,
            state_indices,
            (
                heads,
                value_heads,
                q.stride(0),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(2),
                v.stride(3),
                a.stride(0),
                a.stride(2),
                b.stride(0),
                b.stride(2),
                *state.stride(),
                *state_scale.stride(),
                out_view.stride(0),
                out_view.stride(2),
                out_view.stride(3),
                0 if state_indices is None else state_indices.stride(0),
                key_dim**-0.5 if scale is None else float(scale),
                state_indices is not None,
                value_tile,
            ),
        )
        grid = (batch * value_heads * (128 // value_tile),)
        if packed:
            desc = None
            if use_tme:
                desc = TensorDescriptor(
                    state.view(torch.float32),
                    [state.numel() // 128, 32],
                    [32, 1],
                    [value_tile, 32],
                )
            interleave_heads = batch <= 2 or 4 <= batch <= 7
            decode_quart[grid](
                *args[:-1],
                desc,
                args[-1],
                use_tme,
                16 if interleave_heads else 8,
                interleave_heads,
                num_warps=4,
                num_stages=1,
                enable_backend_opt=True,
                extern_libs={"gdn_fp8_quart": GDN_VECTOR_LIBRARY},
            )
        else:
            decode_single_tle[grid](*args, num_warps=4, num_stages=1)
        return output, state
    # Live recurrence cannot be replayed for runtime autotuning.
    value_tile, warps = (8 if batch == 1 else (16 if batch <= 3 else 32)), 4
    decode_vk_kernel[(batch * value_heads, triton.cdiv(value_dim, value_tile))](
        q,
        k,
        v,
        state,
        state_scale,
        A_log,
        a,
        dt_bias,
        b,
        out_view,
        state_indices,
        history,
        intermediate_state_scale,
        heads,
        value_heads,
        key_dim,
        value_dim,
        tokens,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *a.stride(),
        *b.stride(),
        *state.stride(),
        *state_scale.stride(),
        *out_view.stride(),
        0 if state_indices is None else state_indices.stride(0),
        0 if history is None else history.shape[1],
        key_dim**-0.5 if scale is None else float(scale),
        use_qk_l2norm,
        state_indices is not None,
        not disable_state_update,
        history is not None,
        None,
        False,
        True,
        triton.next_power_of_2(key_dim),
        value_tile,
        num_warps=warps,
        num_stages=1,
    )
    return output, state


@triton.jit
def normalize_query_key_kernel(
    Q,
    Key,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    K3: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
    NORMALIZE_KEY: tl.constexpr = True,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    reduction = tl.arange(0, BK)
    head = rows % H
    token = (rows // H) % TOKENS
    batch = rows // (TOKENS * H)
    qo = (
        batch[:, None] * Q0
        + token[:, None] * Q1
        + head[:, None] * Q2
        + reduction[None, :] * Q3
    )
    ko = (
        batch[:, None] * K0
        + token[:, None] * K1
        + head[:, None] * K2
        + reduction[None, :] * K3
    )
    mask = (rows[:, None] < ROWS) & (reduction[None, :] < DK)
    query = tl.load(Q + qo, mask, 0.0).to(tl.float32)
    if NORMALIZE_KEY:
        key = tl.load(Key + ko, mask, 0.0).to(tl.float32)
    query *= tl.rsqrt(tl.sum(query * query, 1) + 1.0e-6)[:, None]
    if NORMALIZE_KEY:
        key *= tl.rsqrt(tl.sum(key * key, 1) + 1.0e-6)[:, None]
    tl.store(Q + qo, query, mask)
    if NORMALIZE_KEY:
        tl.store(Key + ko, key, mask)


def recurrent_prefill_w8a16_fp8(
    q,
    k,
    v,
    g,
    beta,
    scale,
    state_fp8,
    state_scale,
    cu_seqlens,
    state_indices,
    normalize,
):
    """FP32 recurrence without intermediate activation rounding."""
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    sequence_count = batch if cu_seqlens is None else cu_seqlens.numel() - 1
    state_vk = state_fp8.transpose(-1, -2)
    output = torch.empty(v.shape, device=q.device, dtype=q.dtype)
    if not sequence_count or not tokens:
        return output, state_fp8, state_scale
    if normalize:
        normalize_query_key_kernel[(triton.cdiv(batch * tokens * heads, 16),)](
            q,
            k,
            batch * tokens * heads,
            tokens,
            heads,
            key_dim,
            *q.stride(),
            *k.stride(),
            triton.next_power_of_2(key_dim),
            16,
            num_warps=4,
            num_stages=1,
        )
    tle_prefill = HAS_TLE and key_dim == 128
    value_tile = 8 if tle_prefill else 16
    decode_vk_kernel[
        (sequence_count * value_heads, triton.cdiv(value_dim, value_tile))
    ](
        q,
        k,
        v,
        state_vk,
        state_scale,
        None,
        g,
        None,
        beta,
        output,
        state_indices,
        None,
        None,
        heads,
        value_heads,
        key_dim,
        value_dim,
        tokens,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *g.stride(),
        *beta.stride(),
        *state_vk.stride(),
        *state_scale.stride(),
        *output.stride(),
        0 if state_indices is None else state_indices.stride(0),
        0,
        scale,
        False,
        state_indices is not None,
        True,
        False,
        cu_seqlens,
        cu_seqlens is not None,
        False,
        triton.next_power_of_2(key_dim),
        value_tile,
        TLE_PREFILL=tle_prefill,
        num_warps=4,
        num_stages=1,
    )
    return output, state_fp8, state_scale


def chunk_gated_delta_rule_w8a16_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    is_log_space: bool = True,
    output: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    *,
    initial_state_scale: torch.Tensor | None = None,
    output_state_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """MATE MP31 prefill with FP8 VK state; return output, final state and scales.

    Initial state is read-only. Supplied output state/scales are returned by
    identity; both final values are None when output_final_state is false.
    Indexed active slots must be in bounds and unique, as required by MATE.
    """
    if scale is not None and type(scale) not in (float, int):
        raise TypeError("scale must be a Python number")
    if chunk_size != 64 or checkpoint_every_n_tokens:
        raise NotImplementedError(
            "MP31 prefill requires chunk_size=64 and no checkpoints"
        )
    unbatched = q.ndim == 3
    if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError("QKV must all be rank 3 or rank 4")
    if unbatched:
        if cu_seqlens is None:
            raise ValueError("rank-3 QKV requires cu_seqlens")
        query, key, value = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    else:
        query, key, value = q, k, v
    batch, tokens, heads, key_dim = query.shape
    key_heads, value_heads, value_dim = key.shape[2], value.shape[2], value.shape[3]
    state_heads = max(heads, value_heads)
    if (
        min(heads, key_heads, value_heads, key_dim, value_dim) < 1
        or key.shape[:2] != query.shape[:2]
        or value.shape[:2] != query.shape[:2]
        or key.shape[-1] != key_dim
        or not (
            (heads == key_heads and value_heads % heads == 0)
            or (value_heads == key_heads and heads % key_heads == 0)
        )
    ):
        raise ValueError("invalid QKV dimensions or GQA/GVA head groups")
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError("QKV must share BF16 or FP16 dtype")
    tensors = [q, k, v]
    if cu_seqlens is not None:
        if (
            batch != 1
            or cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 1
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("cu_seqlens requires packed B=1 and an integer vector")
        # The vendor kernel addresses cumulative lengths contiguously.
        if not cu_seqlens.is_contiguous():
            raise ValueError("cu_seqlens must be contiguous")
        sequences = cu_seqlens.numel() - 1
        tensors.append(cu_seqlens)
    else:
        sequences = batch
    if state_indices is not None:
        if state_indices.shape != (sequences,) or state_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("state_indices must be int32/int64 [N]")
        tensors.append(state_indices)
    if has_initial_state is not None:
        if (
            initial_state is None
            or has_initial_state.shape != (sequences,)
            or has_initial_state.dtype != torch.bool
        ):
            raise ValueError("has_initial_state requires an initial state and bool [N]")
        tensors.append(has_initial_state)
    for state, scales in (
        (initial_state, initial_state_scale),
        (output_state, output_state_scale),
    ):
        if state is None:
            if scales is not None:
                raise ValueError("state scales require the corresponding state")
        else:
            if (
                state.ndim != 4
                or state.shape[1:] != (state_heads, value_dim, key_dim)
                or state.dtype != torch.float8_e4m3fn
            ):
                raise ValueError("state must be FP8 E4M3 [pool,H,V,K]")
            if state_indices is None and state.shape[0] != sequences:
                raise ValueError("sequence-order state requires N slots")
            if (
                scales is None
                or scales.shape != state.shape[:3]
                or scales.dtype != torch.float32
            ):
                raise ValueError("FP8 state requires FP32 [pool,H,V] scales")
            tensors.extend((state, scales))
    gate_shape = (*q.shape[:-2], state_heads)
    for gate in (g, beta):
        if gate is not None:
            if gate.shape != gate_shape or gate.dtype not in (q.dtype, torch.float32):
                raise ValueError("g/beta must match token dimensions and state heads")
            tensors.append(gate)
    output_shape = (*q.shape[:-2], state_heads, value_dim)
    if output is not None:
        if output.shape != output_shape or output.dtype != q.dtype:
            raise ValueError("output must match Q dtype and [B,T,H,V] or [T,H,V]")
        tensors.append(output)
    if q.device.type != "musa" or any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all tensors must be on one MUSA device")
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("GDN is forward-only")
    if output_final_state and state_indices is not None and output_state is None:
        raise ValueError(
            "indexed final state requires an explicit output pool and scales"
        )
    if output is None:
        output = torch.empty(output_shape, device=q.device, dtype=q.dtype)
    if output_final_state:
        if output_state is None:
            output_state = torch.empty(
                (sequences, state_heads, value_dim, key_dim),
                device=q.device,
                dtype=torch.float8_e4m3fn,
            )
            output_state_scale = torch.empty(
                (sequences, state_heads, value_dim),
                device=q.device,
                dtype=torch.float32,
            )
        final_state, final_scale = output_state, output_state_scale
    else:
        final_state, final_scale = None, None
    if sequences:
        if use_qk_l2norm_in_kernel and tokens:
            for tensor in (query, key):
                rows = batch * tokens * tensor.shape[2]
                normalize_query_key_kernel[(triton.cdiv(rows, 16),)](
                    tensor,
                    None,
                    rows,
                    tokens,
                    tensor.shape[2],
                    key_dim,
                    *tensor.stride(),
                    0,
                    0,
                    0,
                    0,
                    triton.next_power_of_2(key_dim),
                    16,
                    False,
                    num_warps=4,
                    num_stages=1,
                )
        gate = None if g is None else (g.unsqueeze(0) if unbatched else g)
        update = None if beta is None else (beta.unsqueeze(0) if unbatched else beta)
        out = output.unsqueeze(0) if unbatched else output
        # Empty sequences only initialize final state; no K reduction is needed.
        tle_prefill = HAS_TLE and key_dim == 128 and tokens > 0
        value_tile = 8 if tle_prefill else 16
        decode_vk_kernel[(sequences * state_heads, triton.cdiv(value_dim, value_tile))](
            query,
            key,
            value,
            initial_state,
            initial_state_scale,
            None,
            gate,
            None,
            update,
            out,
            state_indices,
            None,
            None,
            heads,
            state_heads,
            key_dim,
            value_dim,
            tokens,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *((0, 0, 0) if gate is None else gate.stride()),
            *((0, 0, 0) if update is None else update.stride()),
            *((0, 0, 0, 0) if initial_state is None else initial_state.stride()),
            *(
                (0, 0, 0)
                if initial_state_scale is None
                else initial_state_scale.stride()
            ),
            *out.stride(),
            0 if state_indices is None else state_indices.stride(0),
            0,
            key_dim**-0.5 if scale is None else float(scale),
            False,
            state_indices is not None,
            output_final_state,
            False,
            cu_seqlens,
            cu_seqlens is not None,
            False,
            triton.next_power_of_2(key_dim),
            value_tile,
            final_state,
            final_scale,
            has_initial_state,
            True,
            *(
                has_initial_state is not None,
                initial_state is not None,
                g is not None,
                beta is not None,
                is_log_space,
                key_heads,
                value_heads,
                0 if has_initial_state is None else has_initial_state.stride(0),
                *((0, 0, 0, 0) if final_state is None else final_state.stride()),
                *((0, 0, 0) if final_scale is None else final_scale.stride()),
            ),
            TLE_PREFILL=tle_prefill,
            num_warps=4,
            num_stages=1,
        )
    return output, final_state, final_scale
