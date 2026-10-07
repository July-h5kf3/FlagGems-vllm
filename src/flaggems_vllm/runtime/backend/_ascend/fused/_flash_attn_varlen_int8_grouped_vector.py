# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.experimental.tle.language.dsa.ascend.custom_ops.registry import (  # noqa: F401
    cast_fp16_to_int8 as cast_fp16_to_int8,
)


@triton.jit
def round_i16(values, scratch, COUNT: tl.constexpr):
    return al.custom(
        "cast_fp32_to_int16", values.reshape((COUNT,)), 4, COUNT, out=scratch
    )


@triton.jit
def pack_i8(values, scratch, COUNT: tl.constexpr):
    return al.custom(
        "cast_fp16_to_int8",
        values.to(tl.float16).reshape((COUNT,)),
        5,
        COUNT,
        out=scratch,
    )


@triton.jit
def update_grouped_output(
    Product,
    ExtraProduct,
    accumulator,
    alpha,
    beta,
    value_scale,
    low_scale,
    third_scale,
    correction_factor,
    core,
    sub,
    physical_rows,
    tile,
    QT: tl.constexpr,
    N: tl.constexpr,
):
    al.sync_block_wait("cube", "vector", 4 + 6 * (tile % 2))
    ROWS: tl.constexpr = QT * 4
    VR: tl.constexpr = QT * 2
    cols = tl.arange(0, 128)
    base = (core * 2 + tile % 2) * (2 * ROWS + 1) * 128
    high = tl.load(Product + base + physical_rows[:, None] * 128 + cols[None, :]).to(
        tl.float32
    )
    low = tl.load(
        Product + base + (physical_rows[:, None] + ROWS) * 128 + cols[None, :]
    ).to(tl.float32)
    correction = tl.load(Product + base + 2 * ROWS * 128 + cols).to(tl.float32)
    product = tl.fma(
        low, low_scale[:, None], high - correction[None, :] * correction_factor
    )
    if N >= 512:
        rows = tl.arange(0, VR)
        extra_rows = rows
        extra_base = ((core * 2 + tile % 2) * 2 + sub) * VR * 128
        extra = tl.load(
            ExtraProduct + extra_base + extra_rows[:, None] * 128 + cols[None, :]
        ).to(tl.float32)
        product = tl.fma(extra, third_scale[:, None], product)
    coefficient = beta * value_scale
    return accumulator * alpha[:, None] + product * coefficient[:, None]


@triton.jit
def vector_grouped(
    QueryScale,
    KeyScale,
    ValueScale,
    Output,
    Workspace,
    Used,
    Cuq,
    N: tl.constexpr,
    META: tl.constexpr,
):
    QT: tl.constexpr = META[14]
    ROWS: tl.constexpr = QT * 4
    VR: tl.constexpr = QT * 2
    PM: tl.constexpr = triton.cdiv(2 * ROWS + 1, 16) * 16
    EM: tl.constexpr = triton.cdiv(VR, 16) * 16
    COUNT: tl.constexpr = VR * N
    FIXED: tl.constexpr = N >= 256 and QT >= 8
    with al.scope("vector"):
        core = tl.program_id(0).to(tl.int32)
        sub = al.sub_vec_id().to(tl.int32)
        heads: tl.constexpr = META[0]
        kvheads: tl.constexpr = META[1]
        blocks: tl.constexpr = META[9]
        groups: tl.constexpr = META[10]
        qgroups: tl.constexpr = META[16]
        causal: tl.constexpr = META[18]
        Score = Workspace
        address = Workspace.to(tl.uint64)
        Prob = (address + META[30]).to(tl.pointer_type(tl.int8))
        Product = (address + META[30] + META[31]).to(tl.pointer_type(tl.int32))
        if N >= 512:
            ExtraProb = (address + META[30] + META[31] + META[32]).to(
                tl.pointer_type(tl.int8)
            )
            ExtraProduct = (address + META[30] + META[31] + META[32] + META[36]).to(
                tl.pointer_type(tl.int32)
            )
            extra_padding = tl.arange(0, EM)
            word_cols = tl.arange(0, N // 4)
            ExtraWords = (address + META[30] + META[31] + META[32]).to(
                tl.pointer_type(tl.int32)
            )
            for ring in tl.static_range(2):
                tl.store(
                    ExtraWords
                    + ((core * 2 + ring) * 2 + sub) * EM * (N // 4)
                    + extra_padding[:, None] * (N // 4)
                    + word_cols[None, :],
                    0,
                )
        else:
            ExtraProduct = Product
        cols = tl.arange(0, N)
        rows = tl.arange(0, VR)
        query_indices = tl.arange(0, QT)
        query_rows = tl.interleave(query_indices, query_indices)
        head_rows = tl.interleave(
            tl.full((QT,), sub, tl.int32), tl.full((QT,), sub + 2, tl.int32)
        )
        physical_rows = rows * 2 + sub
        if sub == 0:
            words = tl.arange(0, N // 4)
            for ring in tl.static_range(2):
                tl.store(
                    Score
                    + META[30] // 4
                    + (core * 2 + ring) * PM * (N // 4)
                    + 2 * ROWS * (N // 4)
                    + words,
                    -2139062144,
                )
        scratch16 = tl.full((COUNT,), 0, tl.int16)
        scratch8 = tl.full((COUNT,), 0, tl.int8)
        for ordinal in range(tl.cdiv(groups, blocks)):
            group = core + ordinal * blocks
            if group < groups:
                batch = group // (qgroups * kvheads)
                kvhead = group % kvheads
                qstart = group // kvheads % qgroups * QT
                query_base = tl.load(Cuq + batch)
                query_length = tl.load(Cuq + batch + 1) - query_base
                if qstart < query_length:
                    real_queries = tl.minimum(QT, query_length - qstart)
                    nk = tl.load(Used + batch)
                    if causal:
                        visible = tl.minimum(
                            nk, tl.maximum(0, nk - query_length + qstart + real_queries)
                        )
                    else:
                        visible = nk
                    tiles = tl.cdiv(visible, N)
                    maximum = tl.full((VR,), -3.4e38, tl.float32)
                    denominator = tl.full((VR,), 0, tl.float32)
                    accumulator = tl.full((VR, 128), 0, tl.float32)
                    previous_alpha = tl.full((VR,), 1, tl.float32)
                    previous_beta = tl.full((VR,), 0, tl.float32)
                    previous_low_scale = tl.full((VR,), 0, tl.float32)
                    previous_third_scale = tl.full((VR,), 0, tl.float32)
                    previous_correction = 1.0
                    query_scale = tl.load(
                        QueryScale
                        + batch * META[20]
                        + (kvhead * 4 + head_rows) * META[21]
                        + qstart // 128 * META[22]
                    )
                    value_scale = tl.load(
                        ValueScale + batch * META[26] + kvhead * META[27], nk > 0, 0
                    )
                    previous_value_scale = value_scale
                    for tile in range(tiles):
                        al.sync_block_wait("cube", "vector", 2 + 6 * (tile % 2))
                        base = (core * 2 + tile % 2) * ROWS * N
                        if N == 128:
                            value_scale = tl.load(
                                ValueScale
                                + batch * META[26]
                                + kvhead * META[27]
                                + tile * META[28]
                            )
                        score = tl.load(
                            Score + base + physical_rows[:, None] * N + cols[None, :]
                        ).to(tl.float32)
                        if META[25] == 0:
                            key_scale = tl.load(
                                KeyScale + batch * META[23] + kvhead * META[24]
                            )
                            score = score * (
                                query_scale[:, None] * (key_scale * 0.08838834764831845)
                            )
                        else:
                            key_blocks = tile * (N // 128) + tl.arange(0, N // 128)
                            block_scale = tl.load(
                                KeyScale
                                + batch * META[23]
                                + kvhead * META[24]
                                + key_blocks * META[25],
                                key_blocks < tl.cdiv(nk, 128),
                                0,
                            )
                            key_scale = tl.broadcast_to(
                                block_scale[:, None], (N // 128, 128)
                            ).reshape((N,))
                            score = score * (
                                query_scale[:, None]
                                * (key_scale[None, :] * 0.08838834764831845)
                            )
                        if causal:
                            allowed = nk - query_length + qstart + query_rows + 1
                        else:
                            allowed = tl.full((VR,), nk, tl.int32)
                        valid_count = tl.where(
                            query_rows < real_queries,
                            tl.minimum(N, tl.maximum(0, allowed - tile * N)),
                            0,
                        )
                        score = tl.where(
                            cols[None, :] < valid_count[:, None], score, -3.4e38
                        )
                        local_maximum = tl.max(score, 1)
                        probability = tl.exp(score - local_maximum[:, None])
                        local_sum = tl.where(
                            valid_count > 0, tl.sum(probability, 1), 0.0
                        )
                        new_maximum = tl.maximum(maximum, local_maximum)
                        alpha = tl.exp(maximum - new_maximum)
                        beta = tl.where(
                            valid_count > 0, tl.exp(local_maximum - new_maximum), 0.0
                        )
                        denominator = denominator * alpha + local_sum * beta
                        maximum = new_maximum
                        if N >= 512:
                            covered = tl.minimum(N, visible - tile * N)
                            threshold = (
                                (covered - 1).to(tl.float32)
                                * 128.0
                                * tl.abs(value_scale)
                                / (64770.0 * 0.01)
                            )
                            adaptive = (
                                tl.sum(
                                    ((local_sum > 0.0) & (local_sum < threshold)).to(
                                        tl.int32
                                    ),
                                    0,
                                )
                                > 0
                            )
                        else:
                            adaptive = False
                        pbase = (core * 2 + tile % 2) * PM * N
                        if FIXED and not adaptive:
                            narrow = round_i16(
                                probability * 65024.0 - 32512.0, scratch16, COUNT
                            )
                            high = pack_i8(
                                ((narrow + 128).to(tl.int16) >> 8), scratch8, COUNT
                            )
                            tl.store(
                                Prob
                                + pbase
                                + physical_rows[:, None] * N
                                + cols[None, :],
                                high.reshape((VR, N)),
                            )
                            low = pack_i8((narrow << 8).to(tl.int16) >> 8, high, COUNT)
                            tl.store(
                                Prob
                                + pbase
                                + (physical_rows[:, None] + ROWS) * N
                                + cols[None, :],
                                low.reshape((VR, N)),
                            )
                            low_scale = tl.full((VR,), 1.0 / 256.0, tl.float32)
                            correction_factor = 127.0 / 128.0
                            beta = beta * (1.0 / 254.0)
                            residual = tl.full((VR, N), 0, tl.float32)
                            need_third = False
                        else:
                            scaled = probability * 255.0
                            high_round = round_i16(scaled, scratch16, COUNT)
                            high = pack_i8(high_round - 128, scratch8, COUNT)
                            tl.store(
                                Prob
                                + pbase
                                + physical_rows[:, None] * N
                                + cols[None, :],
                                high.reshape((VR, N)),
                            )
                            residual = scaled - high_round.reshape((VR, N)).to(
                                tl.float32
                            )
                            if adaptive:
                                span = tl.maximum(
                                    tl.max(tl.abs(residual), 1), 2.0**-119
                                )
                                low_scale = span / 127.0
                                low_round = round_i16(
                                    residual * (127.0 / span[:, None]),
                                    high_round,
                                    COUNT,
                                )
                                remaining = (
                                    residual
                                    - low_round.reshape((VR, N)).to(tl.float32)
                                    * low_scale[:, None]
                                )
                                error = (
                                    tl.sum(tl.abs(remaining), 1)
                                    * 128.0
                                    * tl.abs(value_scale)
                                    / tl.maximum(local_sum, 1.0)
                                    / 255.0
                                )
                                need_third = tl.sum((error > 0.01).to(tl.int32), 0) > 0
                                residual = remaining
                            else:
                                low_round = round_i16(
                                    residual * 254.0, high_round, COUNT
                                )
                                low_scale = tl.full((VR,), 1.0 / 254.0, tl.float32)
                                need_third = False
                            low = pack_i8(low_round, high, COUNT)
                            tl.store(
                                Prob
                                + pbase
                                + (physical_rows[:, None] + ROWS) * N
                                + cols[None, :],
                                low.reshape((VR, N)),
                            )
                            correction_factor = 1.0
                            beta = beta * (1.0 / 255.0)
                        if N >= 512:
                            extra_base = ((core * 2 + tile % 2) * 2 + sub) * EM * N
                            extra_rows = rows
                            if adaptive and need_third:
                                span = tl.maximum(
                                    tl.max(tl.abs(residual), 1), 2.0**-119
                                )
                                third_scale = span / 127.0
                                third_round = round_i16(
                                    residual * (127.0 / span[:, None]), scratch16, COUNT
                                )
                                third = pack_i8(third_round, scratch8, COUNT)
                                tl.store(
                                    ExtraProb
                                    + extra_base
                                    + extra_rows[:, None] * N
                                    + cols[None, :],
                                    third.reshape((VR, N)),
                                )
                            else:
                                third_scale = tl.full((VR,), 0, tl.float32)
                                tl.store(
                                    ExtraProb
                                    + extra_base
                                    + extra_rows[:, None] * N
                                    + cols[None, :],
                                    tl.full((VR, N), 0, tl.int8),
                                )
                            al.sync_block_set(
                                "vector",
                                "cube",
                                3 + 6 * (tile % 2),
                                sender_pipe=al.PIPE.PIPE_MTE3,
                                receiver_pipe=al.PIPE.PIPE_S,
                            )
                        else:
                            third_scale = tl.full((VR,), 0, tl.float32)
                            al.sync_block_set("vector", "cube", 3 + 6 * (tile % 2))
                        if tile > 0:
                            accumulator = update_grouped_output(
                                Product,
                                ExtraProduct,
                                accumulator,
                                previous_alpha,
                                previous_beta,
                                previous_value_scale,
                                previous_low_scale,
                                previous_third_scale,
                                previous_correction,
                                core,
                                sub,
                                physical_rows,
                                tile - 1,
                                QT,
                                N,
                            )
                        previous_alpha = alpha
                        previous_beta = beta
                        previous_low_scale = low_scale
                        previous_third_scale = third_scale
                        previous_correction = correction_factor
                        previous_value_scale = value_scale
                    if tiles > 0:
                        accumulator = update_grouped_output(
                            Product,
                            ExtraProduct,
                            accumulator,
                            previous_alpha,
                            previous_beta,
                            previous_value_scale,
                            previous_low_scale,
                            previous_third_scale,
                            previous_correction,
                            core,
                            sub,
                            physical_rows,
                            tiles - 1,
                            QT,
                            N,
                        )
                    normalized = accumulator / tl.maximum(denominator[:, None], 1.0)
                    output_cols = tl.arange(0, 128)
                    output_queries = tl.arange(0, QT)
                    output_heads = tl.arange(0, 2) * 2 + sub
                    output_offsets = (
                        (
                            query_base.to(tl.int64)
                            + qstart
                            + output_queries[:, None, None]
                        )
                        * heads
                        + kvhead * 4
                        + output_heads[None, :, None]
                    ) * 128 + output_cols[None, None, :]
                    output_values = normalized.reshape((QT, 2, 128)).to(tl.bfloat16)
                    tl.store(
                        Output + output_offsets,
                        output_values,
                        output_queries[:, None, None] < real_queries,
                    )
