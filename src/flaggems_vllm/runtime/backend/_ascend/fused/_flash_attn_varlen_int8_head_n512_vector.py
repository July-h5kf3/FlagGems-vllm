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
    base = (core * 2 + tile % 2) * 128 * N
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
def prepare_fixed_half(
    Score,
    Prob,
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
    threshold = (
        tl.maximum(valid_count - 1, 0).to(tl.float32)
        * 128.0
        * tl.abs(value_scale)
        / (64770.0 * 0.01)
    )
    needs_replay = (
        N == 512
        and tl.max(((local_sum > 0.0) & (local_sum < threshold)).to(tl.int32), 0) > 0
    )
    COUNT: tl.constexpr = 16 * N
    scratch16 = al.custom(
        "cast_fp32_to_int16",
        (probability * 65024.0 - 32512.0).reshape((COUNT,)),
        4,
        COUNT,
        out=scratch16,
    )
    high = (scratch16 + 128) >> 8
    high_byte = al.custom(
        "cast_fp16_to_int8", high.to(tl.float16), 5, COUNT, out=scratch8
    )
    rows = row_start + tl.arange(0, 16)
    cols = tl.arange(0, N)
    base = (core * 2 + tile % 2) * 272 * N
    tl.store(
        Prob + base + rows[:, None] * N + cols[None, :], high_byte.reshape((16, N))
    )
    low = scratch16 - (high << 8)
    scratch8 = al.custom(
        "cast_fp16_to_int8", low.to(tl.float16), 5, COUNT, out=high_byte
    )
    tl.store(
        Prob + base + 144 * N + rows[:, None] * N + cols[None, :],
        scratch8.reshape((16, N)),
    )
    return (
        maximum,
        denominator,
        alpha,
        beta * (1.0 / 254.0),
        needs_replay,
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
    maximum0, denominator0, alpha0, beta0, replay0, scratch16, scratch8 = (
        prepare_fixed_half(
            Score,
            Prob,
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
    )
    maximum1, denominator1, alpha1, beta1, replay1, scratch16, scratch8 = (
        prepare_fixed_half(
            Score,
            Prob,
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
    )
    return (
        merge_half_state(maximum0, maximum1),
        merge_half_state(denominator0, denominator1),
        merge_half_state(alpha0, alpha1),
        merge_half_state(beta0, beta1),
        tl.full((32,), 1.0 / 256.0, tl.float32),
        tl.full((32,), 0.0, tl.float32),
        127.0 / 128.0,
        False,
        scratch16,
        scratch8,
        replay0 | replay1,
    )


@triton.jit
def merge_state(first, second):
    combined = tl.full((64,), 0.0, tl.float32)
    combined = tle.dsa.insert_slice(
        combined, first, [tl.full((), 0, tl.int32)], [32], [1]
    )
    return tle.dsa.insert_slice(
        combined, second, [tl.full((), 32, tl.int32)], [32], [1]
    )


@triton.jit
def accumulate_head_native(
    Product,
    ExtraProduct,
    ValueScale,
    accumulator,
    alpha0,
    alpha1,
    beta0,
    beta1,
    low_scale0,
    low_scale1,
    third_scale0,
    third_scale1,
    correction0,
    correction1,
    needs_third0,
    needs_third1,
    batch,
    kvhead,
    core,
    sub,
    tile,
    N: tl.constexpr,
    META: tl.constexpr,
):
    rows = sub * 64 + tl.arange(0, 64)
    cols = tl.arange(0, 128)
    alpha = merge_state(alpha0, alpha1)
    beta = merge_state(beta0, beta1)
    base = (core * 2 + tile % 2) * 257 * 128
    high = tl.load(Product + base + rows[:, None] * 128 + cols[None, :]).to(tl.float32)
    correction = tl.load(Product + base + 128 * 128 + cols).to(tl.float32)
    low = tl.load(Product + base + 129 * 128 + rows[:, None] * 128 + cols[None, :]).to(
        tl.float32
    )
    if N == 256:
        product = tl.fma(low, 1.0 / 256.0, high - correction[None, :] * (127.0 / 128.0))
    else:
        low_scale = merge_state(low_scale0, low_scale1)
        correction_scale = merge_state(
            tl.full((32,), correction0, tl.float32),
            tl.full((32,), correction1, tl.float32),
        )
        product = tl.fma(
            low,
            low_scale[:, None],
            high - correction[None, :] * correction_scale[:, None],
        )
        for slice in tl.static_range(2):
            needs_third = needs_third0 if slice == 0 else needs_third1
            third_scale = third_scale0 if slice == 0 else third_scale1
            if needs_third:
                extra_rows = sub * 64 + slice * 32 + tl.arange(0, 32)
                extra = tl.load(
                    ExtraProduct
                    + (core * 2 + tile % 2) * 128 * 128
                    + extra_rows[:, None] * 128
                    + cols[None, :]
                ).to(tl.float32)
                current = tle.dsa.extract_slice(
                    product,
                    [tl.full((), slice * 32, tl.int32), tl.full((), 0, tl.int32)],
                    [32, 128],
                    [1, 1],
                )
                current = tl.fma(extra, third_scale[:, None], current)
                product = tle.dsa.insert_slice(
                    product,
                    current,
                    [tl.full((), slice * 32, tl.int32), tl.full((), 0, tl.int32)],
                    [32, 128],
                    [1, 1],
                )
            else:
                pass
    value_scale = tl.load(
        ValueScale + batch * META[26] + kvhead * META[27] + tile * META[28]
    )
    coefficient = beta * value_scale
    return accumulator * alpha[:, None] + product * coefficient[:, None]


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
        Control = (
            WorkspaceI32 + (META[30] + META[31] + META[32] + META[36] + META[37]) // 4
        )
        if sub == 0:
            words = tl.arange(0, N // 4)
            for ring in tl.static_range(2):
                tl.store(
                    WorkspaceI32
                    + META[30] // 4
                    + ((core * 2 + ring) * 272 + 128) * (N // 4)
                    + words,
                    -2139062144,
                )
        else:
            pass
        scratch16 = tl.full((16 * N,), 0, tl.int16)
        scratch8 = tl.full((16 * N,), 0, tl.int8)
        for ordinal in range(tl.cdiv(groups, blocks)):
            group = core + ordinal * blocks
            if group < groups:
                tl.store(Flags + group * 2 + sub, 0)
                batch = group // (qgroups * heads)
                head = group % heads
                kvhead = head // (heads // kvheads)
                qstart = group // heads % qgroups * 128
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
                else:
                    pass
                if active_group:
                    real_rows = tl.minimum(128, query_length - qstart)
                    if META[18]:
                        visible = tl.minimum(
                            nk, tl.maximum(0, nk - query_length + qstart + real_rows)
                        )
                    else:
                        visible = nk
                    tiles = tl.cdiv(visible, N)
                    maximum0 = tl.full((32,), -3.4e38, tl.float32)
                    maximum1 = tl.full((32,), -3.4e38, tl.float32)
                    denominator0 = tl.full((32,), 0.0, tl.float32)
                    denominator1 = tl.full((32,), 0.0, tl.float32)
                    accumulator = tl.full((64, 128), 0.0, tl.float32)
                    previous_alpha0 = tl.full((32,), 1.0, tl.float32)
                    previous_alpha1 = tl.full((32,), 1.0, tl.float32)
                    previous_beta0 = tl.full((32,), 0.0, tl.float32)
                    previous_beta1 = tl.full((32,), 0.0, tl.float32)
                    previous_low_scale0 = tl.full((32,), 0.0, tl.float32)
                    previous_low_scale1 = tl.full((32,), 0.0, tl.float32)
                    previous_third_scale0 = tl.full((32,), 0.0, tl.float32)
                    previous_third_scale1 = tl.full((32,), 0.0, tl.float32)
                    previous_correction0 = 1.0
                    previous_correction1 = 1.0
                    previous_third0 = False
                    previous_third1 = False
                    query_scale = (
                        tl.load(
                            QueryScale
                            + batch * META[20]
                            + head * META[21]
                            + (qstart // 128) * META[22]
                        )
                        * 0.08838834764831845
                    )
                    # Block scales require the smaller replay tile for accuracy.
                    replay_group = (META[19] > 1 and META[22] != 0) or (
                        META[7] > 1 and META[25] != 0
                    )
                    for tile in range(tiles):
                        al.sync_block_wait("cube", "vector", 2 + 6 * (tile % 2))
                        (
                            maximum0,
                            denominator0,
                            alpha0,
                            beta0,
                            low_scale0,
                            third_scale0,
                            correction0,
                            third0,
                            scratch16,
                            scratch8,
                            replay0,
                        ) = prepare_head_slice(
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
                            sub,
                            tile,
                            nk,
                            query_length,
                            qstart,
                            real_rows,
                            scratch16,
                            scratch8,
                            sub * 64,
                            N,
                            META,
                        )
                        (
                            maximum1,
                            denominator1,
                            alpha1,
                            beta1,
                            low_scale1,
                            third_scale1,
                            correction1,
                            third1,
                            scratch16,
                            scratch8,
                            replay1,
                        ) = prepare_head_slice(
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
                            sub,
                            tile,
                            nk,
                            query_length,
                            qstart,
                            real_rows,
                            scratch16,
                            scratch8,
                            sub * 64 + 32,
                            N,
                            META,
                        )
                        replay_group = replay_group | replay0 | replay1
                        if N == 512:
                            mask = third0.to(tl.int32) | (third1.to(tl.int32) << 1)
                            tl.store(
                                Control
                                + (core * 2 + tile % 2) * 16
                                + sub * 8
                                + tl.arange(0, 8),
                                mask,
                            )
                            al.sync_block_set(
                                "vector",
                                "cube",
                                3 + 6 * (tile % 2),
                                sender_pipe=al.PIPE.PIPE_MTE3,
                                receiver_pipe=al.PIPE.PIPE_S,
                            )
                        else:
                            al.sync_block_set("vector", "cube", 3 + 6 * (tile % 2))
                        if tile > 0:
                            al.sync_block_wait(
                                "cube", "vector", 4 + 6 * ((tile - 1) % 2)
                            )
                            accumulator = accumulate_head_native(
                                Product,
                                ExtraProduct,
                                ValueScale,
                                accumulator,
                                previous_alpha0,
                                previous_alpha1,
                                previous_beta0,
                                previous_beta1,
                                previous_low_scale0,
                                previous_low_scale1,
                                previous_third_scale0,
                                previous_third_scale1,
                                previous_correction0,
                                previous_correction1,
                                previous_third0,
                                previous_third1,
                                batch,
                                kvhead,
                                core,
                                sub,
                                tile - 1,
                                N,
                                META,
                            )
                        else:
                            pass
                        previous_alpha0, previous_alpha1 = alpha0, alpha1
                        previous_beta0, previous_beta1 = beta0, beta1
                        previous_low_scale0, previous_low_scale1 = (
                            low_scale0,
                            low_scale1,
                        )
                        previous_third_scale0, previous_third_scale1 = (
                            third_scale0,
                            third_scale1,
                        )
                        previous_correction0, previous_correction1 = (
                            correction0,
                            correction1,
                        )
                        previous_third0, previous_third1 = third0, third1
                    if tiles > 0:
                        al.sync_block_wait("cube", "vector", 4 + 6 * ((tiles - 1) % 2))
                        accumulator = accumulate_head_native(
                            Product,
                            ExtraProduct,
                            ValueScale,
                            accumulator,
                            previous_alpha0,
                            previous_alpha1,
                            previous_beta0,
                            previous_beta1,
                            previous_low_scale0,
                            previous_low_scale1,
                            previous_third_scale0,
                            previous_third_scale1,
                            previous_correction0,
                            previous_correction1,
                            previous_third0,
                            previous_third1,
                            batch,
                            kvhead,
                            core,
                            sub,
                            tiles - 1,
                            N,
                            META,
                        )
                    else:
                        pass
                    tl.store(Flags + group * 2 + sub, replay_group.to(tl.int32))
                    cols = tl.arange(0, 128)
                    rows = sub * 64 + tl.arange(0, 64)
                    denominator = merge_state(denominator0, denominator1)
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
                else:
                    pass
            else:
                pass
