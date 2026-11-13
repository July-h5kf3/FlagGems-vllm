# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from hashlib import sha256

import triton
import triton.language as tl
from triton.experimental.tle import language as tle
from triton.language import core
from triton.runtime.cache import get_cache_manager

# MP31 converts four adjacent FP8 values together; Triton's scalar cast
# does not select the four-element instruction.
_CONVERSION_IR = r"""
define protected i64 @gdn_decode_e4m3_quart(i32 noundef %0) alwaysinline nounwind readnone {
  %2 = bitcast i32 %0 to <4 x i8>
  %3 = tail call contract <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8> %2)
  %4 = bitcast <4 x half> %3 to i64
  ret i64 %4
}

declare <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8>) nounwind readnone

define i32 @gdn_encode_e4m3_quart(float %0, float %1, float %2, float %3)
    alwaysinline nounwind readnone {
  %5 = insertelement <4 x float> undef, float %0, i64 0
  %6 = insertelement <4 x float> %5, float %1, i64 1
  %7 = insertelement <4 x float> %6, float %2, i64 2
  %8 = insertelement <4 x float> %7, float %3, i64 3
  %9 = tail call <4 x i8> @llvm.musa.f2e4m3.rn.bst4(<4 x float> %8)
  %10 = bitcast <4 x i8> %9 to i32
  ret i32 %10
}

declare <4 x i8> @llvm.musa.f2e4m3.rn.bst4(<4 x float>) nounwind readnone
"""


@lru_cache(maxsize=1)
def conversion_library():
    cache = get_cache_manager(sha256(_CONVERSION_IR.encode()).hexdigest())
    name = "gdn_fp8_quart.ll"
    return cache.get_file(name) or cache.put(_CONVERSION_IR, name, binary=False)


@core.extern
def _decode_four(packed, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [packed],
        {(tl.uint32,): ("gdn_decode_e4m3_quart", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def _encode_four(first, second, third, fourth, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [first, second, third, fourth],
        {
            (tl.float32, tl.float32, tl.float32, tl.float32): (
                "gdn_encode_e4m3_quart",
                tl.uint32,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _unpack(word, shift: tl.constexpr, dtype: tl.constexpr):
    return (word >> shift).to(tl.uint16).to(dtype, bitcast=True).to(tl.float32)


@triton.jit
def _load_four(ptr, columns):
    word = tl.load(ptr.to(tl.pointer_type(tl.uint64)) + columns[None, :])
    return (
        _unpack(word, 0, ptr.dtype.element_ty),
        _unpack(word, 16, ptr.dtype.element_ty),
        _unpack(word, 32, ptr.dtype.element_ty),
        _unpack(word, 48, ptr.dtype.element_ty),
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


@triton.jit
def _inverse_norm(x0, x1, x2, x3):
    norm = tl.sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3, 1) + 1.0e-6
    # MP31 rsq(+inf) returns NaN; overflowing BF16 square sums normalize to zero.
    return tl.where(norm == float("inf"), 0.0, _rsqrt(norm))[:, None]


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
):
    # Groups of 16 lanes handle independent V rows, four FP8 bytes per lane load.
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
        buf = tle.gpu.alloc((BV, 32), dtype=tl.float32, nv_mma_shared_layout=False)
        full = tle.gpu.alloc_barrier(expect_bytes=BV * 128)
    chunks: tl.constexpr = 128 // BV
    sequence_head = tl.program_id(0) // chunks
    value_chunk = tl.program_id(0) % chunks
    (sequence, head) = (sequence_head // HV, sequence_head % HV)
    query_head = head // (HV // H)
    layout: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [2, 16], [4, 1], [1, 0])
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
    qptr = Q + sequence * Q0 + query_head * Q2
    kptr = Key + sequence * K0 + query_head * K2
    query0, query1, query2, query3 = _load_four(qptr, reduction)
    key0, key1, key2, key3 = _load_four(kptr, reduction)
    value = tl.load(Value + sequence * V0 + head * V2 + columns * V3, valid_v, 0.0).to(
        tl.float32
    )
    output_offset = sequence * O0 + head * O2 + columns * O3
    live = slot >= 0
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
        if not live:
            tl.store(Out + output_offset, 0.0, columns < 128)
            return
        tile_row = ((slot * HV + head) * 128 + value_chunk * BV).to(tl.int32)
        tle.gpu.copy(Desc, buf, (BV, 32), (tile_row, 0), barrier=full)
        tle.gpu.barrier_wait(full, phaseIdx=0)
        r = tl.broadcast_to((columns - value_chunk * BV)[:, None], (BV, 32))
        c = tl.broadcast_to(reduction[None, :], (BV, 32))
        packed = tl.load(tle.gpu.local_ptr(buf, (r, c))).to(tl.uint32, bitcast=True)
    else:
        packed = tl.load(State.to(tl.pointer_type(tl.uint32)) + state_offset, mask, 0)
    qinv = _inverse_norm(query0, query1, query2, query3)
    kinv = _inverse_norm(key0, key1, key2, key3)
    query0, query1 = (query0 * qinv) * SCALE, (query1 * qinv) * SCALE
    query2, query3 = (query2 * qinv) * SCALE, (query3 * qinv) * SCALE
    key0, key1, key2, key3 = key0 * kinv, key1 * kinv, key2 * kinv, key3 * kinv
    alpha = tl.exp2(al * 1.4426950408889634)
    x = ga + bi
    softplus = tl.where(
        x > 20.0, x, tl.log2(1.0 + tl.exp2(x * 1.4426950408889634)) * 0.6931471805599453
    )
    decay = tl.exp2(-alpha * softplus * 1.4426950408889634)
    beta = tl.sigmoid(gb)
    halves = _decode_four(packed)
    channel_scale = tl.load(Scales + scale_offset, valid_v, 0.0)[:, None]
    state0 = _unpack(halves, 0, tl.float16) * channel_scale * decay
    state1 = _unpack(halves, 16, tl.float16) * channel_scale * decay
    state2 = _unpack(halves, 32, tl.float16) * channel_scale * decay
    state3 = _unpack(halves, 48, tl.float16) * channel_scale * decay
    delta = (
        value - tl.sum(state0 * key0 + state1 * key1 + state2 * key2 + state3 * key3, 1)
    ) * beta
    state0 = tl.fma(delta[:, None], key0, state0)
    state1 = tl.fma(delta[:, None], key1, state1)
    state2 = tl.fma(delta[:, None], key2, state2)
    state3 = tl.fma(delta[:, None], key3, state3)
    result = tl.sum(
        state0 * query0 + state1 * query1 + state2 * query2 + state3 * query3, 1
    )
    tl.store(Out + output_offset, tl.where(valid_v, result, 0.0), columns < 128)
    magnitude = tl.max(
        tl.maximum(
            tl.maximum(tl.abs(state0), tl.abs(state1)),
            tl.maximum(tl.abs(state2), tl.abs(state3)),
        ),
        1,
    )
    safe = tl.where(magnitude > 0.0, magnitude, 448.0)
    scale = safe * (1.0 / 448.0)
    factor = 448.0 / safe
    quantized = _encode_four(
        state0 * factor[:, None],
        state1 * factor[:, None],
        state2 * factor[:, None],
        state3 * factor[:, None],
    )
    tl.store(State.to(tl.pointer_type(tl.uint32)) + state_offset, quantized, mask)
    tl.store(Scales + scale_offset, scale, valid_v)
