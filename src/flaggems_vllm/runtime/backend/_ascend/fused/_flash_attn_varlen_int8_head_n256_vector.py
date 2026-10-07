# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.experimental.tle.language.dsa.ascend.custom_ops.registry import (  # noqa: F401
    cast_fp32_to_int16 as cast_fp32_to_int16,
)
from triton.language.extra.cann import libdevice


@triton.jit
def update_output(
    Product,
    ValueScale,
    accumulator,
    alpha0,
    alpha1,
    beta0,
    beta1,
    core,
    row_start,
    tile,
    batch,
    kvhead,
    real_rows,
    META: tl.constexpr,
):
    rows = row_start + tl.arange(0, 64)
    cols = tl.arange(0, 128)
    alpha = tl.full((64,), 0.0, tl.float32)
    alpha = tle.dsa.insert_slice(alpha, alpha0, [tl.full((), 0, tl.int32)], [32], [1])
    alpha = tle.dsa.insert_slice(alpha, alpha1, [tl.full((), 32, tl.int32)], [32], [1])
    beta = tl.full((64,), 0.0, tl.float32)
    beta = tle.dsa.insert_slice(beta, beta0, [tl.full((), 0, tl.int32)], [32], [1])
    beta = tle.dsa.insert_slice(beta, beta1, [tl.full((), 32, tl.int32)], [32], [1])
    base = (core * 2 + tile % 2) * 257 * 128
    high = tl.load(Product + base + rows[:, None] * 128 + cols[None, :]).to(tl.float32)
    correction = tl.load(Product + base + 128 * 128 + cols).to(tl.float32)
    low = tl.load(Product + base + 129 * 128 + rows[:, None] * 128 + cols[None, :]).to(
        tl.float32
    )
    value_scale = tl.load(
        ValueScale + batch * META[26] + kvhead * META[27] + tile * META[28]
    )
    product = tl.fma(low, 1.0 / 256.0, high - correction[None, :] * (127.0 / 128.0))
    coefficient = beta * value_scale
    return accumulator * alpha[:, None] + product * coefficient[:, None]


@triton.jit
def compute_probability(
    Score,
    Prob,
    KeyScale,
    query_scale,
    maximum,
    denominator,
    core,
    row_start,
    tile,
    batch,
    kvhead,
    nk,
    query_length,
    qstart,
    real_rows,
    scratch16,
    scratch8,
    CAUSAL: tl.constexpr,
    META: tl.constexpr,
):
    tl.static_assert(META[28] == 0)
    rows = row_start + tl.arange(0, 32)
    cols = tl.arange(0, 256)
    base = (core * 2 + tile % 2) * 128 * 256
    indices = tile * 256 + cols
    score = tl.load(Score + base + rows[:, None] * 256 + cols[None, :]).to(tl.float32)
    if META[25] == 0:
        key_scale = tl.load(KeyScale + batch * META[23] + kvhead * META[24])
        score = score * (query_scale * (key_scale * 0.08838834764831845))
    else:
        key_blocks = tile * 2 + tl.arange(0, 2)
        key_scale = tl.load(
            KeyScale + batch * META[23] + kvhead * META[24] + key_blocks * META[25],
            key_blocks < tl.cdiv(nk, 128),
            0,
        )
        key_columns = tl.broadcast_to(key_scale[:, None], (2, 128)).reshape((256,))
        score = score * (query_scale * (key_columns[None, :] * 0.08838834764831845))
    if CAUSAL:
        allowed = nk - query_length + qstart + rows + 1
    else:
        allowed = nk + tl.full((32,), 0, tl.int32)
    valid = indices[None, :] < allowed[:, None]
    score = tl.where(valid, score, -float("inf"))
    local_max = tl.max(score, 1)
    has_values = allowed > tile * 256
    local_max = tl.where(has_values, local_max, -3.4e38)
    probability = tl.exp(score - local_max[:, None])
    local_sum = tl.sum(probability, 1)
    new_max = tl.maximum(maximum, local_max)
    alpha = tl.exp(maximum - new_max)
    beta = tl.where(has_values, tl.exp(local_max - new_max), 0.0)
    denominator = denominator * alpha + local_sum * beta
    maximum = new_max
    beta = beta * (1.0 / 254.0)
    encoded = probability * 65024.0 - 32512.0
    scratch16 = al.custom(
        "cast_fp32_to_int16", tl.reshape(encoded, (8192,)), 4, 8192, out=scratch16
    )
    high = (scratch16 + 128) >> 8
    pbase = (core * 2 + tile % 2) * 272 * 256
    high_byte = al.custom(
        "cast_fp16_to_int8", high.to(tl.float16), 5, 8192, out=scratch8
    )
    tl.store(
        Prob + pbase + rows[:, None] * 256 + cols[None, :], high_byte.reshape((32, 256))
    )
    low = scratch16 - (high << 8)
    scratch8 = al.custom(
        "cast_fp16_to_int8", low.to(tl.float16), 5, 8192, out=high_byte
    )
    tl.store(
        Prob + pbase + 144 * 256 + rows[:, None] * 256 + cols[None, :],
        scratch8.reshape((32, 256)),
    )
    return maximum, denominator, alpha, beta, scratch16, scratch8


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
    tl.static_assert(N == 256)
    with al.scope("vector"):
        core = tl.program_id(0)
        sub = al.sub_vec_id().to(tl.int32)
        heads: tl.constexpr = META[0]
        kvheads: tl.constexpr = META[1]
        blocks: tl.constexpr = META[9]
        groups: tl.constexpr = META[10]
        qlen: tl.constexpr = META[15]
        qgroups: tl.constexpr = META[16]
        causal: tl.constexpr = META[18]
        Score = Workspace
        Prob = (Workspace.to(tl.uint64) + META[30]).to(tl.pointer_type(tl.int8))
        Product = Workspace + (META[30] + META[31]) // 4
        cols = tl.arange(0, 128)
        if sub == 0:
            padding_rows = tl.arange(0, 16)
            padding_words = tl.arange(0, 64)
            padding = tl.where(padding_rows[:, None] == 0, -2139062144, 0) + tl.full(
                (16, 64), 0, tl.int32
            )
            for ring in tl.static_range(2):
                tl.store(
                    Workspace
                    + META[30] // 4
                    + (core * 2 + ring) * 272 * 64
                    + (128 + padding_rows[:, None]) * 64
                    + padding_words[None, :],
                    padding,
                )
        else:
            pass
        scratch16 = tl.full((8192,), 0, tl.int16)
        scratch8 = tl.full((8192,), 0, tl.int8)
        for ordinal in range(tl.cdiv(groups, blocks)):
            group = core + ordinal * blocks
            if group < groups:
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
                    twice_visible = 2 * nk.to(tl.int64) - (qlen - 1 if causal else 0)
                    active_group = twice_visible < (6144 if causal else 4096)
                else:
                    pass
                real_rows = tl.minimum(128, query_length - qstart)
                if active_group:
                    if causal:
                        visible = tl.minimum(
                            nk, tl.maximum(0, nk - query_length + qstart + real_rows)
                        )
                    else:
                        visible = nk
                    tiles = tl.cdiv(visible, 256)
                    maximum0 = tl.full((32,), -3.4e38, tl.float32)
                    denominator0 = tl.full((32,), 0, tl.float32)
                    accumulator = tl.full((64, 128), 0, tl.float32)
                    previous_alpha0 = tl.full((32,), 1, tl.float32)
                    previous_beta0 = tl.full((32,), 0, tl.float32)
                    maximum1 = tl.full((32,), -3.4e38, tl.float32)
                    denominator1 = tl.full((32,), 0, tl.float32)
                    previous_alpha1 = tl.full((32,), 1, tl.float32)
                    previous_beta1 = tl.full((32,), 0, tl.float32)
                    query_scale = tl.load(
                        QueryScale
                        + batch * META[20]
                        + head * META[21]
                        + (qstart // 128) * META[22]
                    )
                    for tile in range(tiles):
                        al.sync_block_wait("cube", "vector", 2 + 6 * (tile % 2))
                        maximum0, denominator0, alpha0, beta0, scratch16, scratch8 = (
                            compute_probability(
                                Score,
                                Prob,
                                KeyScale,
                                query_scale,
                                maximum0,
                                denominator0,
                                core,
                                sub * 64 + 0,
                                tile,
                                batch,
                                kvhead,
                                nk,
                                query_length,
                                qstart,
                                real_rows,
                                scratch16,
                                scratch8,
                                causal,
                                META,
                            )
                        )
                        maximum1, denominator1, alpha1, beta1, scratch16, scratch8 = (
                            compute_probability(
                                Score,
                                Prob,
                                KeyScale,
                                query_scale,
                                maximum1,
                                denominator1,
                                core,
                                sub * 64 + 32,
                                tile,
                                batch,
                                kvhead,
                                nk,
                                query_length,
                                qstart,
                                real_rows,
                                scratch16,
                                scratch8,
                                causal,
                                META,
                            )
                        )
                        al.sync_block_set("vector", "cube", 3 + 6 * (tile % 2))
                        if tile > 0:
                            al.sync_block_wait(
                                "cube", "vector", 4 + 6 * ((tile - 1) % 2)
                            )
                            accumulator = update_output(
                                Product,
                                ValueScale,
                                accumulator,
                                previous_alpha0,
                                previous_alpha1,
                                previous_beta0,
                                previous_beta1,
                                core,
                                sub * 64,
                                tile - 1,
                                batch,
                                kvhead,
                                real_rows,
                                META,
                            )
                        else:
                            pass
                        previous_alpha0 = alpha0
                        previous_beta0 = beta0
                        previous_alpha1 = alpha1
                        previous_beta1 = beta1
                    if tiles > 0:
                        al.sync_block_wait("cube", "vector", 4 + 6 * ((tiles - 1) % 2))
                        accumulator = update_output(
                            Product,
                            ValueScale,
                            accumulator,
                            previous_alpha0,
                            previous_alpha1,
                            previous_beta0,
                            previous_beta1,
                            core,
                            sub * 64,
                            tiles - 1,
                            batch,
                            kvhead,
                            real_rows,
                            META,
                        )
                    else:
                        pass
                    denominator = tl.full((64,), 0.0, tl.float32)
                    denominator = tle.dsa.insert_slice(
                        denominator, denominator0, [tl.full((), 0, tl.int32)], [32], [1]
                    )
                    denominator = tle.dsa.insert_slice(
                        denominator,
                        denominator1,
                        [tl.full((), 32, tl.int32)],
                        [32],
                        [1],
                    )
                    normalized = (
                        accumulator
                        * libdevice.reciprocal(tl.maximum(denominator, 1.0))[:, None]
                    )
                    rows = sub * 64 + tl.arange(0, 64)
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
