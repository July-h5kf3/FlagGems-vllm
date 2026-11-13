# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl

from flaggems_vllm.utils import libentry
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

HAS_TLE = has_triton_tle(3, 6, 0)


@triton.jit
def store_scaled_state(values, State, Scales, offset, scale_offset, mask, valid_v):
    magnitude = tl.max(tl.abs(values), 1)
    safe = tl.where(magnitude > 0.0, magnitude, 448.0)
    scale = safe * (1.0 / 448.0)
    quantized = (values * (448.0 / safe)[:, None]).to(State.dtype.element_ty)
    tl.store(State + offset, quantized, mask)
    tl.store(Scales + scale_offset, scale, valid_v)


if HAS_TLE:
    import triton.experimental.tle.language as tle

    @libentry()
    @triton.jit
    def decode_single_tle(
        Q, Key, Value, State, Scales, Alog, A, Bias, B, Out, Index, META: tl.constexpr
    ):
        H: tl.constexpr = META[0]
        HV: tl.constexpr = META[1]
        Q0: tl.constexpr = META[2]
        Q2: tl.constexpr = META[3]
        Q3: tl.constexpr = META[4]
        K0: tl.constexpr = META[5]
        K2: tl.constexpr = META[6]
        K3: tl.constexpr = META[7]
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
        S3: tl.constexpr = META[18]
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
        chunks: tl.constexpr = 128 // BV
        sequence_head = tl.program_id(0) // chunks
        value_chunk = tl.program_id(0) % chunks
        sequence, head = sequence_head // HV, sequence_head % HV
        query_head = head // (HV // H)
        layout: tl.constexpr = tle.gpu.BlockEncoding([1, 4], [1, 32], [4, 1], [1, 0])
        columns = value_chunk * BV + tle.gpu.set_layout(
            tl.arange(0, BV), tle.gpu.SlicedEncoding(1, layout)
        )
        reduction = tle.gpu.set_layout(
            tl.arange(0, 128), tle.gpu.SlicedEncoding(0, layout)
        )
        valid_v = columns < 128
        mask = valid_v[:, None] & (reduction[None, :] < 128)
        slot = (
            tl.load(Index + sequence * IX).to(tl.int64)
            if INDEXED
            else sequence.to(tl.int64)
        )
        output_offset = sequence * O0 + head * O2 + columns * O3
        if slot < 0:
            tl.store(Out + output_offset, 0.0, valid_v)
            return
        state_offset = (
            slot * S0 + head * S1 + columns[:, None] * S2 + reduction[None, :] * S3
        )
        scale_offset = slot * C0 + head * C1 + columns * C2
        state = tl.load(State + state_offset, mask, 0.0).to(tl.float32)
        state *= tl.load(Scales + scale_offset, valid_v, 0.0)[:, None]
        query = tl.load(
            Q + sequence * Q0 + query_head * Q2 + reduction[None, :] * Q3
        ).to(tl.float32)
        key = tl.load(
            Key + sequence * K0 + query_head * K2 + reduction[None, :] * K3
        ).to(tl.float32)
        query *= tl.exp2(-0.5 * tl.log2(tl.sum(query * query, 1) + 1.0e-6))[:, None]
        key *= tl.exp2(-0.5 * tl.log2(tl.sum(key * key, 1) + 1.0e-6))[:, None]
        query *= SCALE
        alpha = tl.exp2(tl.load(Alog + head).to(tl.float32) * 1.4426950408889634)
        x = tl.load(A + sequence * A0 + head * A2).to(tl.float32) + tl.load(
            Bias + head
        ).to(tl.float32)
        softplus = tl.where(
            x > 20.0,
            x,
            tl.log2(1.0 + tl.exp2(x * 1.4426950408889634)) * 0.6931471805599453,
        )
        decay = tl.exp2(-alpha * softplus * 1.4426950408889634)
        beta = tl.sigmoid(tl.load(B + sequence * B0 + head * B2).to(tl.float32))
        state *= decay
        value = tl.load(
            Value + sequence * V0 + head * V2 + columns * V3, valid_v, 0.0
        ).to(tl.float32)
        delta = (value - tl.sum(state * key, 1)) * beta
        state = tl.fma(delta[:, None], key, state)
        result = tl.sum(state * query, 1)
        tl.store(Out + output_offset, result, valid_v)
        store_scaled_state(
            state, State, Scales, state_offset, scale_offset, mask, valid_v
        )
