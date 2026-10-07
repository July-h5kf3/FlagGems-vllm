# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.experimental.tle.language.dsa.ascend.custom_ops.registry import (  # noqa: F401
    cast_fp16_to_int8 as cast_fp16_to_int8,
)


@triton.jit
def update_hybrid_small_output(
    Product, accumulator, alpha, beta, value_scale, core, physical_rows, tile
):
    al.sync_block_wait("cube", "vector", 4 + 6 * (tile % 2))
    cols = tl.arange(0, 128)
    base = (core * 2 + tile % 2) * 17 * 128
    high = tl.load(Product + base + physical_rows[:, None] * 128 + cols[None, :]).to(
        tl.float32
    )
    low = tl.load(
        Product + base + (physical_rows[:, None] + 8) * 128 + cols[None, :]
    ).to(tl.float32)
    correction = tl.load(Product + base + 16 * 128 + cols).to(tl.float32)
    product = tl.fma(low, 1.0 / 254.0, high - correction[None, :])
    coefficient = beta * value_scale
    return accumulator * alpha[:, None] + product * coefficient[:, None]


@triton.jit
def vector_hybrid_small(
    QueryScale, KeyScale, ValueScale, Output, Workspace, Used, Cuq, META: tl.constexpr
):
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
        Prob = (Workspace.to(tl.uint64) + META[30]).to(tl.pointer_type(tl.int8))
        Product = (Workspace.to(tl.uint64) + META[30] + META[31]).to(
            tl.pointer_type(tl.int32)
        )
        cols = tl.arange(0, 256)
        rows = tl.arange(0, 4)
        query_rows = rows // 2
        head_rows = rows % 2 * 2 + sub
        physical_rows = rows * 2 + sub
        if sub == 0:
            correction_columns = tl.arange(0, 64)
            for ring in tl.static_range(2):
                tl.store(
                    Score
                    + META[30] // 4
                    + (core * 2 + ring) * 32 * 64
                    + 16 * 64
                    + correction_columns,
                    tl.full((64,), -2139062144, tl.int32),
                )
        else:
            pass
        scratch16 = tl.full((1024,), 0, tl.int16)
        scratch8 = tl.full((1024,), 0, tl.int8)
        for ordinal in range(tl.cdiv(groups, blocks)):
            group = core + ordinal * blocks
            if group < groups:
                batch = group // (qgroups * kvheads)
                kvhead = group % kvheads
                qstart = group // kvheads % qgroups * 2
                query_base = tl.load(Cuq + batch)
                query_length = tl.load(Cuq + batch + 1) - query_base
                if query_length <= 4 and qstart < query_length:
                    real_queries = tl.minimum(2, query_length - qstart)
                    nk = tl.load(Used + batch)
                    if causal:
                        visible = tl.minimum(
                            nk, tl.maximum(0, nk - query_length + qstart + real_queries)
                        )
                    else:
                        visible = nk
                    tiles = tl.cdiv(visible, 256)
                    maximum = tl.full((4,), -3.4e38, tl.float32)
                    denominator = tl.full((4,), 0, tl.float32)
                    accumulator = tl.full((4, 128), 0, tl.float32)
                    previous_alpha = tl.full((4,), 1, tl.float32)
                    previous_beta = tl.full((4,), 0, tl.float32)
                    query_scale = tl.load(
                        QueryScale
                        + batch * META[20]
                        + (kvhead * 4 + head_rows) * META[21]
                        + qstart // 128 * META[22]
                    )
                    value_scale = tl.load(
                        ValueScale + batch * META[26] + kvhead * META[27]
                    )
                    if META[25] == 0:
                        fixed_key_scale = tl.load(
                            KeyScale + batch * META[23] + kvhead * META[24]
                        )
                        fixed_coefficient = query_scale * (
                            fixed_key_scale * 0.08838834764831845
                        )
                    else:
                        pass
                    for tile in range(tiles):
                        al.sync_block_wait("cube", "vector", 2 + 6 * (tile % 2))
                        base = (core * 2 + tile % 2) * 8 * 256
                        score = tl.load(
                            Score + base + physical_rows[:, None] * 256 + cols[None, :]
                        ).to(tl.float32)
                        if META[25] == 0:
                            score = score * fixed_coefficient[:, None]
                        else:
                            key_blocks = tile * 2 + cols // 128
                            key_scale = tl.load(
                                KeyScale
                                + batch * META[23]
                                + kvhead * META[24]
                                + key_blocks * META[25],
                                key_blocks < tl.cdiv(nk, 128),
                                0,
                            )
                            score = score * (
                                query_scale[:, None]
                                * (key_scale[None, :] * 0.08838834764831845)
                            )
                        if causal:
                            allowed = nk - query_length + qstart + query_rows + 1
                        else:
                            allowed = tl.full((4,), nk, tl.int32)
                        valid_count = tl.where(
                            query_rows < real_queries,
                            tl.minimum(256, tl.maximum(0, allowed - tile * 256)),
                            0,
                        )
                        score = tl.where(
                            cols[None, :] < valid_count[:, None], score, -3.4e38
                        )
                        raw_maximum = tl.max(score, 1)
                        probability = tl.exp(score - raw_maximum[:, None])
                        local_maximum = raw_maximum
                        # Empty rows are excluded by beta=0 below.
                        local_sum = tl.sum(probability, 1)
                        new_maximum = tl.maximum(maximum, local_maximum)
                        alpha = tl.exp(maximum - new_maximum)
                        beta = tl.where(
                            valid_count > 0, tl.exp(local_maximum - new_maximum), 0.0
                        )
                        denominator = denominator * alpha + local_sum * beta
                        maximum = new_maximum
                        beta = beta * (1.0 / 255.0)
                        scaled = probability * 255.0
                        high = al.custom(
                            "cast_fp32_to_int16",
                            scaled.reshape((1024,)),
                            4,
                            1024,
                            out=scratch16,
                        )
                        high_rows = high.reshape((4, 256))
                        high_byte = al.custom(
                            "cast_fp16_to_int8",
                            (high_rows - 128).to(tl.float16).reshape((1024,)),
                            5,
                            1024,
                            out=scratch8,
                        )
                        high_stored = high_byte.reshape((4, 256))
                        pbase = (core * 2 + tile % 2) * 32 * 256
                        tl.store(
                            Prob + pbase + physical_rows[:, None] * 256 + cols[None, :],
                            high_stored,
                        )
                        residual = (scaled - high_rows.to(tl.float32)) * 254.0
                        scratch16 = al.custom(
                            "cast_fp32_to_int16",
                            residual.reshape((1024,)),
                            4,
                            1024,
                            out=high,
                        )
                        low_rows = scratch16
                        scratch8 = al.custom(
                            "cast_fp16_to_int8",
                            low_rows.to(tl.float16),
                            5,
                            1024,
                            out=high_byte,
                        )
                        low_stored = scratch8.reshape((4, 256))
                        tl.store(
                            Prob
                            + pbase
                            + (physical_rows[:, None] + 8) * 256
                            + cols[None, :],
                            low_stored,
                        )
                        al.sync_block_set("vector", "cube", 3 + 6 * (tile % 2))
                        if tile > 0:
                            accumulator = update_hybrid_small_output(
                                Product,
                                accumulator,
                                previous_alpha,
                                previous_beta,
                                value_scale,
                                core,
                                physical_rows,
                                tile - 1,
                            )
                        else:
                            pass
                        previous_alpha = alpha
                        previous_beta = beta
                    if tiles > 0:
                        accumulator = update_hybrid_small_output(
                            Product,
                            accumulator,
                            previous_alpha,
                            previous_beta,
                            value_scale,
                            core,
                            physical_rows,
                            tiles - 1,
                        )
                    else:
                        pass
                    normalized = accumulator / tl.maximum(denominator[:, None], 1.0)
                    output_cols = tl.arange(0, 128)
                    output_offsets = (
                        (query_base.to(tl.int64) + qstart + query_rows[:, None]) * heads
                        + kvhead * 4
                        + head_rows[:, None]
                    ) * 128 + output_cols[None, :]
                    tl.store(
                        Output + output_offsets,
                        normalized.to(tl.bfloat16),
                        query_rows[:, None] < real_queries,
                    )
                else:
                    pass
            else:
                pass
