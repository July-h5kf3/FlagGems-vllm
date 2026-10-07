# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.language.extra.cann import libdevice


@triton.jit
def merge_half_state(first, second):
    combined = tl.full((32,), 0.0, tl.float32)
    combined = tle.dsa.insert_slice(
        combined, first, [tl.full((), 0, tl.int32)], [16], [1]
    )
    return tle.dsa.insert_slice(
        combined, second, [tl.full((), 16, tl.int32)], [16], [1]
    )


@triton.jit
def softmax_head_half(
    Score,
    maximum,
    denominator,
    query_scale,
    KeyScale,
    batch,
    kvhead,
    core,
    tile,
    nk,
    query_length,
    qstart,
    real_rows,
    row_start,
    N: tl.constexpr,
    META: tl.constexpr,
):
    rows = row_start + tl.arange(0, 16)
    cols = tl.arange(0, N)
    base = (core * 2 + tile % 2) * META[14] * N
    score = tl.load(Score + base + rows[:, None] * N + cols[None, :]).to(tl.float32)
    if META[25] == 0:
        key_scale = tl.load(KeyScale + batch * META[23] + kvhead * META[24])
        score = score * (query_scale * key_scale)
    else:
        key_blocks = tile * (N // 128) + tl.arange(0, N // 128)
        key_scale = tl.load(
            KeyScale + batch * META[23] + kvhead * META[24] + key_blocks * META[25],
            key_blocks < tl.cdiv(nk, 128),
            0,
        )
        key_columns = tl.broadcast_to(key_scale[:, None], (N // 128, 128)).reshape((N,))
        score = score * (query_scale * key_columns[None, :])
    if META[18]:
        allowed = nk - query_length + qstart + rows + 1
    else:
        allowed = tl.full((16,), nk, tl.int32)
    allowed = tl.where(rows < real_rows, allowed, 0)
    valid_count = tl.minimum(N, tl.maximum(0, allowed - tile * N))
    score = tl.where(cols[None, :] < valid_count[:, None], score, -float("inf"))
    local_maximum = tl.where(valid_count > 0, tl.max(score, 1), -3.4e38)
    probability = tl.exp(score - local_maximum[:, None])
    local_sum = tl.sum(probability, 1)
    new_maximum = tl.maximum(maximum, local_maximum)
    alpha = tl.exp(maximum - new_maximum)
    beta = tl.where(valid_count > 0, tl.exp(local_maximum - new_maximum), 0.0)
    denominator = denominator * alpha + local_sum * beta
    return new_maximum, denominator, alpha, beta, probability, local_sum, valid_count


@triton.jit
def quantize_head_half(
    Prob,
    ExtraProb,
    probability,
    local_sum,
    adaptive,
    magnitude,
    scratch16,
    scratch8,
    core,
    tile,
    row_start,
    N: tl.constexpr,
    Q: tl.constexpr,
):
    COUNT: tl.constexpr = 16 * N
    rows = row_start + tl.arange(0, 16)
    cols = tl.arange(0, N)
    pbase = (core * 2 + tile % 2) * (2 * Q + 16) * N
    if adaptive:
        scaled = probability * 255.0
        scratch16 = al.custom(
            "cast_fp32_to_int16", scaled.reshape((COUNT,)), 4, COUNT, out=scratch16
        )
        high_adaptive = al.custom(
            "cast_fp16_to_int8",
            (scratch16 - 128).to(tl.float16),
            5,
            COUNT,
            out=scratch8,
        )
        tl.store(
            Prob + pbase + rows[:, None] * N + cols[None, :],
            high_adaptive.reshape((16, N)),
        )
        residual = scaled - scratch16.reshape((16, N)).to(tl.float32)
        span = tl.maximum(tl.max(tl.abs(residual), 1), 2.0**-119)
        low_scale = span / 127.0
        scratch16 = al.custom(
            "cast_fp32_to_int16",
            (residual * (127.0 / span[:, None])).reshape((COUNT,)),
            4,
            COUNT,
            out=scratch16,
        )
        remaining = (
            residual - scratch16.reshape((16, N)).to(tl.float32) * low_scale[:, None]
        )
        error = (
            tl.sum(tl.abs(remaining), 1)
            * 128.0
            * magnitude
            / tl.maximum(local_sum, 1.0)
            / 255.0
        )
        needs_third = tl.max((error > 0.01).to(tl.int32), 0) > 0
        scratch8 = al.custom(
            "cast_fp16_to_int8", scratch16.to(tl.float16), 5, COUNT, out=high_adaptive
        )
        tl.store(
            Prob + pbase + Q * N + rows[:, None] * N + cols[None, :],
            scratch8.reshape((16, N)),
        )
        if needs_third:
            third_scale, scratch16, scratch8 = quantize_third_half(
                ExtraProb, remaining, scratch16, scratch8, core, tile, row_start, N, Q
            )
        else:
            third_scale = tl.full((16,), 0.0, tl.float32)
    else:
        scratch16 = al.custom(
            "cast_fp32_to_int16",
            (probability * 65024.0 - 32512.0).reshape((COUNT,)),
            4,
            COUNT,
            out=scratch16,
        )
        high_fixed = (scratch16 + 128) >> 8
        high_byte = al.custom(
            "cast_fp16_to_int8", high_fixed.to(tl.float16), 5, COUNT, out=scratch8
        )
        tl.store(
            Prob + pbase + rows[:, None] * N + cols[None, :], high_byte.reshape((16, N))
        )
        low = scratch16 - (high_fixed << 8)
        scratch8 = al.custom(
            "cast_fp16_to_int8", low.to(tl.float16), 5, COUNT, out=high_byte
        )
        low_scale = tl.full((16,), 1.0 / 256.0, tl.float32)
        third_scale = tl.full((16,), 0.0, tl.float32)
        needs_third = False
        tl.store(
            Prob + pbase + Q * N + rows[:, None] * N + cols[None, :],
            scratch8.reshape((16, N)),
        )
    if N == 512 and not needs_third:
        tl.store(
            ExtraProb
            + (core * 2 + tile % 2) * Q * N
            + rows[:, None] * N
            + cols[None, :],
            tl.full((16, N), 0, tl.int8),
        )
    else:
        pass
    return low_scale, third_scale, needs_third, scratch16, scratch8


@triton.jit
def quantize_third_half(
    ExtraProb,
    remaining,
    scratch16,
    scratch8,
    core,
    tile,
    row_start,
    N: tl.constexpr,
    Q: tl.constexpr,
):
    COUNT: tl.constexpr = 16 * N
    span = tl.maximum(tl.max(tl.abs(remaining), 1), 2.0**-119)
    third_scale = span / 127.0
    scratch16 = al.custom(
        "cast_fp32_to_int16",
        (remaining * (127.0 / span[:, None])).reshape((COUNT,)),
        4,
        COUNT,
        out=scratch16,
    )
    scratch8 = al.custom(
        "cast_fp16_to_int8", scratch16.to(tl.float16), 5, COUNT, out=scratch8
    )
    rows = row_start + tl.arange(0, 16)
    cols = tl.arange(0, N)
    tl.store(
        ExtraProb + (core * 2 + tile % 2) * Q * N + rows[:, None] * N + cols[None, :],
        scratch8.reshape((16, N)),
    )
    return third_scale, scratch16, scratch8


@triton.jit
def prepare_independent_half(
    Score,
    Prob,
    ExtraProb,
    maximum,
    denominator,
    query_scale,
    KeyScale,
    ValueScale,
    batch,
    kvhead,
    core,
    tile,
    nk,
    query_length,
    qstart,
    real_rows,
    scratch16,
    scratch8,
    row_start,
    N: tl.constexpr,
    META: tl.constexpr,
):
    maximum, denominator, alpha, beta, probability, local_sum, valid_count = (
        softmax_head_half(
            Score,
            maximum,
            denominator,
            query_scale,
            KeyScale,
            batch,
            kvhead,
            core,
            tile,
            nk,
            query_length,
            qstart,
            real_rows,
            row_start,
            N,
            META,
        )
    )
    value_scale = tl.load(
        ValueScale + batch * META[26] + kvhead * META[27] + tile * META[28]
    )
    magnitude = tl.abs(value_scale)
    threshold = (
        tl.maximum(valid_count - 1, 0).to(tl.float32)
        * 128.0
        * magnitude
        / (64770.0 * 0.01)
    )
    adaptive = (
        N == 512
        and tl.sum(((local_sum > 0.0) & (local_sum < threshold)).to(tl.int32), 0) > 0
    )
    low_scale, third_scale, needs_third, scratch16, scratch8 = quantize_head_half(
        Prob,
        ExtraProb,
        probability,
        local_sum,
        adaptive,
        magnitude,
        scratch16,
        scratch8,
        core,
        tile,
        row_start,
        N,
        META[14],
    )
    beta = beta * tl.where(adaptive, 1.0 / 255.0, 1.0 / 254.0)
    correction = tl.full((16,), tl.where(adaptive, 1.0, 127.0 / 128.0), tl.float32)
    return (
        maximum,
        denominator,
        alpha,
        beta,
        low_scale,
        third_scale,
        correction,
        needs_third,
        scratch16,
        scratch8,
    )


@triton.jit
def prepare_head_slice(
    Score,
    Prob,
    ExtraProb,
    maximum,
    denominator,
    query_scale,
    KeyScale,
    ValueScale,
    batch,
    kvhead,
    core,
    sub,
    tile,
    nk,
    query_length,
    qstart,
    real_rows,
    scratch16,
    scratch8,
    row_start,
    N: tl.constexpr,
    META: tl.constexpr,
):
    maximum0 = tle.dsa.extract_slice(maximum, [tl.full((), 0, tl.int32)], [16], [1])
    maximum1 = tle.dsa.extract_slice(maximum, [tl.full((), 16, tl.int32)], [16], [1])
    denominator0 = tle.dsa.extract_slice(
        denominator, [tl.full((), 0, tl.int32)], [16], [1]
    )
    denominator1 = tle.dsa.extract_slice(
        denominator, [tl.full((), 16, tl.int32)], [16], [1]
    )
    (
        maximum0,
        denominator0,
        alpha0,
        beta0,
        low0,
        third0,
        correction0,
        needs0,
        scratch16,
        scratch8,
    ) = prepare_independent_half(
        Score,
        Prob,
        ExtraProb,
        maximum0,
        denominator0,
        query_scale,
        KeyScale,
        ValueScale,
        batch,
        kvhead,
        core,
        tile,
        nk,
        query_length,
        qstart,
        real_rows,
        scratch16,
        scratch8,
        row_start,
        N,
        META,
    )
    (
        maximum1,
        denominator1,
        alpha1,
        beta1,
        low1,
        third1,
        correction1,
        needs1,
        scratch16,
        scratch8,
    ) = prepare_independent_half(
        Score,
        Prob,
        ExtraProb,
        maximum1,
        denominator1,
        query_scale,
        KeyScale,
        ValueScale,
        batch,
        kvhead,
        core,
        tile,
        nk,
        query_length,
        qstart,
        real_rows,
        scratch16,
        scratch8,
        row_start + 16,
        N,
        META,
    )
    return (
        merge_half_state(maximum0, maximum1),
        merge_half_state(denominator0, denominator1),
        merge_half_state(alpha0, alpha1),
        merge_half_state(beta0, beta1),
        merge_half_state(low0, low1),
        merge_half_state(third0, third1),
        merge_half_state(correction0, correction1),
        needs0 | needs1,
        scratch16,
        scratch8,
    )


@triton.jit
def accumulate_compact(
    Product,
    ExtraProduct,
    ValueScale,
    accumulator,
    alpha,
    beta,
    low_scale,
    third_scale,
    correction_scale,
    batch,
    kvhead,
    core,
    sub,
    tile,
    N: tl.constexpr,
    Q: tl.constexpr,
    META: tl.constexpr,
):
    R: tl.constexpr = Q // 2
    rows = sub * R + tl.arange(0, R)
    cols = tl.arange(0, 128)
    base = (core * 2 + tile % 2) * (2 * Q + 1) * 128
    high = tl.load(Product + base + rows[:, None] * 128 + cols[None, :]).to(tl.float32)
    low = tl.load(Product + base + (Q + rows[:, None]) * 128 + cols[None, :]).to(
        tl.float32
    )
    correction = tl.load(Product + base + 2 * Q * 128 + cols).to(tl.float32)
    product = tl.fma(
        low, low_scale[:, None], high - correction[None, :] * correction_scale[:, None]
    )
    if N == 512:
        extra = tl.load(
            ExtraProduct
            + (core * 2 + tile % 2) * Q * 128
            + rows[:, None] * 128
            + cols[None, :]
        ).to(tl.float32)
        product = tl.fma(extra, third_scale[:, None], product)
    value_scale = tl.load(
        ValueScale + batch * META[26] + kvhead * META[27] + tile * META[28]
    )
    return accumulator * alpha[:, None] + product * (beta * value_scale)[:, None]


@triton.jit
def vector_wide(
    QueryScale,
    KeyScale,
    ValueScale,
    Output,
    Workspace,
    WorkspaceI32,
    Used,
    Cuq,
    N: tl.constexpr,
    MODE: tl.constexpr,
    META: tl.constexpr,
):
    Q: tl.constexpr = META[14]
    R: tl.constexpr = Q // 2
    tl.static_assert(Q == 32 or Q == 64)
    tl.static_assert(N == 256 or N == 512)
    with al.scope(core_mode="vector"):
        core = tl.program_id(0)
        sub = al.sub_vec_id().to(tl.int32)
        heads: tl.constexpr = META[0]
        kvheads: tl.constexpr = META[1]
        blocks: tl.constexpr = META[9]
        groups: tl.constexpr = META[10]
        qlen: tl.constexpr = META[15]
        qgroups: tl.constexpr = META[16]
        Flags = WorkspaceI32 + META[38]
        Score = WorkspaceI32
        address = WorkspaceI32.to(tl.uint64)
        Prob = (address + META[30]).to(tl.pointer_type(tl.int8))
        Product = WorkspaceI32 + (META[30] + META[31]) // 4
        ExtraProb = (address + META[30] + META[31] + META[32]).to(
            tl.pointer_type(tl.int8)
        )
        ExtraProduct = (address + META[30] + META[31] + META[32] + META[36]).to(
            tl.pointer_type(tl.int32)
        )
        if sub == 0:
            padding_rows = tl.arange(0, 16)
            words = tl.arange(0, N // 4)
            padding = tl.where(padding_rows[:, None] == 0, -2139062144, 0) + tl.full(
                (16, N // 4), 0, tl.int32
            )
            for ring in tl.static_range(2):
                tl.store(
                    WorkspaceI32
                    + META[30] // 4
                    + (core * 2 + ring) * (2 * Q + 16) * (N // 4)
                    + (2 * Q + padding_rows[:, None]) * (N // 4)
                    + words[None, :],
                    padding,
                )
        scratch16 = tl.full((16 * N,), 0, tl.int16)
        scratch8 = tl.full((16 * N,), 0, tl.int8)
        for ordinal in range(tl.cdiv(groups, blocks)):
            group = core + ordinal * blocks
            if group < groups:
                batch = group // (qgroups * heads)
                head = group % heads
                kvhead = head // (heads // kvheads)
                qstart = group // heads % qgroups * Q
                if MODE == 4:
                    query_base = tl.load(Cuq + batch)
                    query_length = tl.load(Cuq + batch + 1) - query_base
                    active_group = (query_length > 4) & (qstart < query_length)
                else:
                    query_base = batch * qlen
                    query_length = qlen
                    active_group = True
                nk = tl.load(Used + batch)
                if MODE == 5:
                    twice_visible = 2 * nk.to(tl.int64) - (qlen - 1 if META[18] else 0)
                    active_group = (twice_visible >= (6144 if META[18] else 4096)) == (
                        N == 512
                    )
                fast_group = (batch * META[39] + qstart // 128) * heads + head
                replay = tl.load(Flags + fast_group * 2) | tl.load(
                    Flags + fast_group * 2 + 1
                )
                active_group = active_group & (replay != 0)
                if active_group:
                    real_rows = tl.minimum(Q, query_length - qstart)
                    if META[18]:
                        visible = tl.minimum(
                            nk, tl.maximum(0, nk - query_length + qstart + real_rows)
                        )
                    else:
                        visible = nk
                    tiles = tl.cdiv(visible, N)
                    maximum = tl.full((R,), -3.4e38, tl.float32)
                    denominator = tl.full((R,), 0.0, tl.float32)
                    accumulator = tl.full((R, 128), 0.0, tl.float32)
                    previous_alpha = tl.full((R,), 1.0, tl.float32)
                    previous_beta = tl.full((R,), 0.0, tl.float32)
                    previous_low = tl.full((R,), 0.0, tl.float32)
                    previous_third = tl.full((R,), 0.0, tl.float32)
                    previous_correction = tl.full((R,), 1.0, tl.float32)
                    query_scale = (
                        tl.load(
                            QueryScale
                            + batch * META[20]
                            + head * META[21]
                            + (qstart // 128) * META[22]
                        )
                        * 0.08838834764831845
                    )
                    for tile in range(tiles):
                        al.sync_block_wait("cube", "vector", 2 + 6 * (tile % 2))
                        if Q == 32:
                            (
                                maximum,
                                denominator,
                                alpha,
                                beta,
                                low,
                                third,
                                correction,
                                needs,
                                scratch16,
                                scratch8,
                            ) = prepare_independent_half(
                                Score,
                                Prob,
                                ExtraProb,
                                maximum,
                                denominator,
                                query_scale,
                                KeyScale,
                                ValueScale,
                                batch,
                                kvhead,
                                core,
                                tile,
                                nk,
                                query_length,
                                qstart,
                                real_rows,
                                scratch16,
                                scratch8,
                                sub * R,
                                N,
                                META,
                            )
                        else:
                            (
                                maximum,
                                denominator,
                                alpha,
                                beta,
                                low,
                                third,
                                correction,
                                needs,
                                scratch16,
                                scratch8,
                            ) = prepare_head_slice(
                                Score,
                                Prob,
                                ExtraProb,
                                maximum,
                                denominator,
                                query_scale,
                                KeyScale,
                                ValueScale,
                                batch,
                                kvhead,
                                core,
                                sub,
                                tile,
                                nk,
                                query_length,
                                qstart,
                                real_rows,
                                scratch16,
                                scratch8,
                                sub * R,
                                N,
                                META,
                            )
                        al.sync_block_set(
                            "vector",
                            "cube",
                            3 + 6 * (tile % 2),
                            sender_pipe=al.PIPE.PIPE_MTE3,
                            receiver_pipe=al.PIPE.PIPE_S,
                        )
                        if tile > 0:
                            al.sync_block_wait(
                                "cube", "vector", 4 + 6 * ((tile - 1) % 2)
                            )
                            accumulator = accumulate_compact(
                                Product,
                                ExtraProduct,
                                ValueScale,
                                accumulator,
                                previous_alpha,
                                previous_beta,
                                previous_low,
                                previous_third,
                                previous_correction,
                                batch,
                                kvhead,
                                core,
                                sub,
                                tile - 1,
                                N,
                                Q,
                                META,
                            )
                        previous_alpha, previous_beta = alpha, beta
                        previous_low, previous_third, previous_correction = (
                            low,
                            third,
                            correction,
                        )
                    if tiles > 0:
                        al.sync_block_wait("cube", "vector", 4 + 6 * ((tiles - 1) % 2))
                        accumulator = accumulate_compact(
                            Product,
                            ExtraProduct,
                            ValueScale,
                            accumulator,
                            previous_alpha,
                            previous_beta,
                            previous_low,
                            previous_third,
                            previous_correction,
                            batch,
                            kvhead,
                            core,
                            sub,
                            tiles - 1,
                            N,
                            Q,
                            META,
                        )
                    rows = sub * R + tl.arange(0, R)
                    cols = tl.arange(0, 128)
                    normalized = (
                        accumulator
                        * libdevice.reciprocal(tl.maximum(denominator, 1.0))[:, None]
                    )
                    offsets = (
                        (query_base + qstart + rows[:, None]) * heads + head
                    ) * 128 + cols[None, :]
                    tl.store(
                        Output + offsets,
                        normalized.to(tl.bfloat16),
                        rows[:, None] < real_rows,
                    )
