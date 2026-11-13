# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import triton
import triton.language as tl
from triton.experimental.tle import language as tle
from triton.language import core

# Link source IR directly: no runtime compilation tool or package writes are required.
GDN_VECTOR_LIBRARY = str(Path(__file__).with_name("gdn_w8a16_mp31.ll"))


@core.extern
def norm4(x0, x1, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x0, x1],
        {
            (
                tl.uint64,
                tl.int32,
            ): ("gdn_norm4", tl.float32)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def normalized_pair(x0, x1, x2, x3, _semantic=None):
    """Four normalized FP32 K/Q elements; select the low or high register pair."""
    return core.extern_elementwise(
        "",
        "",
        [x0, x1, x2, x3],
        {
            (
                tl.uint64,
                tl.float32,
                tl.int32,
                tl.int32,
            ): ("gdn_normalized_pair", tl.uint64)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def scaled_query_pair(x0, x1, x2, x3, x4, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x0, x1, x2, x3, x4],
        {
            (tl.uint64, tl.float32, tl.float32, tl.int32, tl.int32): (
                "gdn_scaled_query_pair",
                tl.uint64,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def state_pair(x0, x1, x2, x3, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x0, x1, x2, x3],
        {
            (
                tl.uint32,
                tl.float32,
                tl.float32,
                tl.int32,
            ): ("gdn_state_pair", tl.uint64)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def dot4(x0, x1, x2, x3, _semantic=None):
    """Four FP32 products with pairwise summation."""
    return core.extern_elementwise(
        "",
        "",
        [x0, x1, x2, x3],
        {
            (
                tl.uint64,
                tl.uint64,
                tl.uint64,
                tl.uint64,
            ): ("gdn_dot4", tl.float32)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def updated_pair(x0, x1, x2, x3, x4, x5, _semantic=None):
    """Updated FP32 values are retained for both projection and quantization."""
    return core.extern_elementwise(
        "",
        "",
        [x0, x1, x2, x3, x4, x5],
        {
            (
                tl.uint64,
                tl.uint64,
                tl.uint64,
                tl.uint64,
                tl.float32,
                tl.int32,
            ): ("gdn_updated_pair", tl.uint64)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def absmax4(x0, x1, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x0, x1],
        {
            (
                tl.uint64,
                tl.uint64,
            ): ("gdn_absmax4", tl.float32)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def encode4(x0, x1, x2, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x0, x1, x2],
        {
            (
                tl.uint64,
                tl.uint64,
                tl.float32,
            ): ("gdn_encode4", tl.uint32)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def _rsqrt(x, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(tl.float32,): ("llvm.musa.rsq", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def _raw_rcp(x, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(tl.float32,): ("llvm.musa.rcp", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _reciprocal(x):
    return tl.where(x == float("inf"), 0.0, _raw_rcp(x))


@triton.jit
def decode_quart(
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
    Desc,
    META: tl.constexpr,
    TME: tl.constexpr,
    LANES: tl.constexpr,
    INTERLEAVE_HEADS: tl.constexpr,
):
    # A lane owns four contiguous K values. A subgroup reduces one value row.
    H: tl.constexpr = META[0]
    HV: tl.constexpr = META[1]
    Q0: tl.constexpr = META[2]
    Q2: tl.constexpr = META[3]
    K0: tl.constexpr = META[5]
    K2: tl.constexpr = META[6]
    V0: tl.constexpr = META[8]
    V2: tl.constexpr = META[9]
    V3: tl.constexpr = META[10]
    A0: tl.constexpr = META[11]
    A2: tl.constexpr = META[12]
    B0: tl.constexpr = META[13]
    B2: tl.constexpr = META[14]
    S0: tl.constexpr = META[15]
    S1: tl.constexpr = META[16]
    S2: tl.constexpr = META[17]
    C0: tl.constexpr = META[19]
    C1: tl.constexpr = META[20]
    C2: tl.constexpr = META[21]
    O0: tl.constexpr = META[22]
    O2: tl.constexpr = META[23]
    O3: tl.constexpr = META[24]
    IX: tl.constexpr = META[25]
    SCALE: tl.constexpr = META[26]
    INDEXED: tl.constexpr = META[27]
    BV: tl.constexpr = META[28]
    if TME:
        # Describe the contiguous FP8 pool as words so each lane receives four codes.
        buf = tle.gpu.alloc((BV, 32), dtype=tl.float32, nv_mma_shared_layout=False)
        full = tle.gpu.alloc_barrier(expect_bytes=BV * 128)
    chunks: tl.constexpr = 128 // BV
    if INTERLEAVE_HEADS:
        head = tl.program_id(0) % HV
        value_chunk = (tl.program_id(0) // HV) % chunks
        sequence = tl.program_id(0) // (HV * chunks)
    else:
        sequence_head = tl.program_id(0) // chunks
        value_chunk = tl.program_id(0) % chunks
        sequence, head = sequence_head // HV, sequence_head % HV
    query_head = head // (HV // H)
    layout: tl.constexpr = tle.gpu.BlockEncoding(
        [1, 1], [32 // LANES, LANES], [4, 1], [1, 0]
    )
    columns = value_chunk * BV + tle.gpu.set_layout(
        tl.arange(0, BV), tle.gpu.SlicedEncoding(1, layout)
    )
    reduction = tle.gpu.set_layout(tl.arange(0, 32), tle.gpu.SlicedEncoding(0, layout))
    valid_v = columns < 128
    mask = valid_v[:, None] & (reduction[None, :] < 32)
    slot = (
        tl.load(Index + sequence * IX).to(tl.int64)
        if INDEXED
        else sequence.to(tl.int64)
    )
    al = tl.load(Alog + head).to(tl.float32)
    ga = tl.load(A + sequence * A0 + head * A2).to(tl.float32)
    gb = tl.load(B + sequence * B0 + head * B2).to(tl.float32)
    bi = tl.load(Bias + head).to(tl.float32)
    pwq = tl.load(
        (Q + sequence * Q0 + query_head * Q2).to(tl.pointer_type(tl.uint64))
        + reduction[None, :]
    )
    pwk = tl.load(
        (Key + sequence * K0 + query_head * K2).to(tl.pointer_type(tl.uint64))
        + reduction[None, :]
    )
    value = tl.load(Value + sequence * V0 + head * V2 + columns * V3, valid_v, 0.0).to(
        tl.float32
    )
    output_offset = sequence * O0 + head * O2 + columns * O3
    live = slot >= 0
    if TME and not live:
        tl.store(Out + output_offset, 0.0, valid_v)
        return
    valid_v = valid_v & live
    mask = mask & live
    slot = tl.maximum(slot, 0)
    state_offset = (
        slot * (S0 // 4)
        + head * (S1 // 4)
        + columns[:, None] * (S2 // 4)
        + reduction[None, :]
    )
    scale_offset = slot * C0 + head * C1 + columns * C2
    if TME:
        tile_row = ((slot * HV + head) * 128 + value_chunk * BV).to(tl.int32)
        tle.gpu.copy(Desc, buf, (BV, 32), (tile_row, 0), barrier=full)
        tle.gpu.barrier_wait(full, phaseIdx=0)
        r = tl.broadcast_to((columns - value_chunk * BV)[:, None], (BV, 32))
        c = tl.broadcast_to(reduction[None, :], (BV, 32))
        packed = tl.load(tle.gpu.local_ptr(buf, (r, c))).to(tl.uint32, bitcast=True)
    else:
        packed = tl.load(
            State.to(tl.pointer_type(tl.uint32)) + state_offset,
            mask,
            0,
        )
    # The device helpers unpack four activation/state elements per lane.
    bf = tl.full((), 1 if Q.dtype.element_ty == tl.bfloat16 else 0, tl.int32)
    nq = tl.reshape(tl.sum(norm4(pwq, bf), 1), ()) + 1.0e-6
    nk = tl.reshape(tl.sum(norm4(pwk, bf), 1), ()) + 1.0e-6
    iq = tl.where(nq == float("inf"), 0.0, _rsqrt(nq))
    ik = tl.where(nk == float("inf"), 0.0, _rsqrt(nk))
    alpha = tl.exp2(al * 1.4426950408889634)
    x = ga + bi
    softplus = tl.where(
        x > 20.0, x, tl.log2(1.0 + tl.exp2(x * 1.4426950408889634)) * 0.6931471805599453
    )
    decay = tl.exp2(-alpha * softplus * 1.4426950408889634)
    beta = _reciprocal(1.0 + tl.exp2(-gb * 1.4426950408889634))
    channel = tl.load(Scales + scale_offset, valid_v, 0.0)
    k0 = normalized_pair(pwk, ik, bf, 0)
    k1 = normalized_pair(pwk, ik, bf, 1)
    s0 = state_pair(packed, channel[:, None], decay, 0)
    s1 = state_pair(packed, channel[:, None], decay, 1)
    delta = (value - tl.sum(dot4(s0, s1, k0, k1), 1)) * beta
    t0 = updated_pair(s0, s1, k0, k1, delta[:, None], 0)
    t1 = updated_pair(s0, s1, k0, k1, delta[:, None], 1)
    if SCALE == 128**-0.5:
        q0 = normalized_pair(pwq, iq * SCALE, bf, 0)
        q1 = normalized_pair(pwq, iq * SCALE, bf, 1)
    else:
        q0 = scaled_query_pair(pwq, iq, tl.full((), SCALE, tl.float32), bf, 0)
        q1 = scaled_query_pair(pwq, iq, tl.full((), SCALE, tl.float32), bf, 1)
    result = tl.sum(dot4(t0, t1, q0, q1), 1)
    tl.store(Out + output_offset, tl.where(valid_v, result, 0.0), columns < 128)
    mag = tl.max(absmax4(t0, t1), 1)
    safe = tl.where(mag > 0.0, mag, 448.0)
    scale = safe * (1.0 / 448.0)
    quantized = encode4(t0, t1, (448.0 / safe)[:, None])
    tl.store(State.to(tl.pointer_type(tl.uint32)) + state_offset, quantized, mask)
    tl.store(Scales + scale_offset, scale, valid_v)
