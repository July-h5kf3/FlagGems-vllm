# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.language.extra.cann import libdevice


@triton.jit
def update_output_aligned(
    Product,
    ValueScale,
    accumulator,
    alpha,
    beta,
    core,
    sub,
    tile,
    batch,
    kvhead,
    real_rows,
    META: tl.constexpr,
):
    al.sync_block_wait("cube", "vector", 4 + 6 * (tile % 2))
    rows = sub * 64 + tl.arange(0, 64)
    cols = tl.arange(0, 128)
    base = (core * 2 + tile % 2) * 257 * 128
    high = tl.load(Product + base + rows[:, None] * 128 + cols[None, :]).to(tl.float32)
    correction = tl.load(Product + base + 128 * 128 + cols).to(tl.float32)
    low = tl.load(Product + base + 129 * 128 + rows[:, None] * 128 + cols[None, :]).to(
        tl.float32
    )
    value_scale = tl.load(
        ValueScale + batch * META[26] + kvhead * META[27] + tile * META[28]
    )
    product = tl.fma(low, 1.0 / 254.0, high - correction[None, :])
    coefficient = beta * value_scale
    return accumulator * alpha[:, None] + product * coefficient[:, None]


@triton.jit
def vector_aligned(
    QueryScale,
    KeyScale,
    ValueScale,
    Output,
    Workspace,
    WorkspaceI32,
    Used,
    Cuq,
    META: tl.constexpr,
):
    tl.static_assert(META[15] % 128 == 0)
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
        row_offsets = tl.arange(0, 64)
        if sub == 0:
            padding_rows = tl.arange(0, 16)
            padding_words = tl.arange(0, 32)
            padding = tl.where(padding_rows[:, None] == 0, -2139062144, 0) + tl.full(
                (16, 32), 0, tl.int32
            )
            for ring in tl.static_range(2):
                tl.store(
                    Score
                    + META[30] // 4
                    + (core * 2 + ring) * 272 * 32
                    + (128 + padding_rows[:, None]) * 32
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
                real_rows = 128
                rows = sub * 64 + row_offsets
                nk = tl.load(Used + batch)
                if causal:
                    visible = tl.minimum(
                        nk, tl.maximum(0, nk - qlen + qstart + real_rows)
                    )
                else:
                    visible = nk
                tiles = tl.cdiv(visible, 128)
                maximum = tl.full((64,), -3.4e38, tl.float32)
                denominator = tl.full((64,), 0, tl.float32)
                accumulator = tl.full((64, 128), 0, tl.float32)
                previous_alpha = tl.full((64,), 1, tl.float32)
                previous_beta = tl.full((64,), 0, tl.float32)
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
                    base = (core * 2 + tile % 2) * 128 * 128
                    indices = tile * 128 + cols
                    score = tl.load(
                        Score + base + rows[:, None] * 128 + cols[None, :]
                    ).to(tl.float32)
                    key_scale = tl.load(
                        KeyScale
                        + batch * META[23]
                        + kvhead * META[24]
                        + tile * META[25]
                    )
                    score = score * (query_scale * key_scale)
                    if causal:
                        allowed = nk - qlen + qstart + rows + 1
                    else:
                        allowed = tl.full((64,), nk, tl.int32)
                    valid = indices[None, :] < allowed[:, None]
                    score = tl.where(valid, score, -float("inf"))
                    local_max = tl.max(score, 1)
                    if causal:
                        has_values = allowed > tile * 128
                        local_max = tl.where(has_values, local_max, -3.4e38)
                    else:
                        pass
                    probability = tl.exp(score - local_max[:, None])
                    local_sum = tl.sum(probability, 1)
                    new_max = tl.maximum(maximum, local_max)
                    alpha = tl.exp(maximum - new_max)
                    beta = tl.exp(local_max - new_max)
                    if causal:
                        beta = tl.where(has_values, beta, 0.0)
                    else:
                        pass
                    denominator = denominator * alpha + local_sum * beta
                    maximum = new_max
                    beta = beta * (1.0 / 255.0)
                    scaled = probability * 255.0
                    high = al.custom(
                        "cast_fp32_to_int16",
                        tl.reshape(scaled, (8192,)),
                        4,
                        8192,
                        out=scratch16,
                    )
                    high_rows = high.reshape((64, 128))
                    pbase = (core * 2 + tile % 2) * 272 * 128
                    high_byte = al.custom(
                        "cast_fp16_to_int8",
                        tl.reshape((high_rows - 128).to(tl.float16), (8192,)),
                        5,
                        8192,
                        out=scratch8,
                    )
                    tl.store(
                        Prob + pbase + rows[:, None] * 128 + cols[None, :],
                        high_byte.reshape((64, 128)),
                    )
                    residual = (scaled - high_rows.to(tl.float32)) * 254.0
                    scratch16 = al.custom(
                        "cast_fp32_to_int16",
                        tl.reshape(residual, (8192,)),
                        4,
                        8192,
                        out=high,
                    )
                    scratch8 = al.custom(
                        "cast_fp16_to_int8",
                        scratch16.to(tl.float16),
                        5,
                        8192,
                        out=high_byte,
                    )
                    tl.store(
                        Prob + pbase + 144 * 128 + rows[:, None] * 128 + cols[None, :],
                        scratch8.reshape((64, 128)),
                    )
                    al.sync_block_set("vector", "cube", 3 + 6 * (tile % 2))
                    if tile > 0:
                        accumulator = update_output_aligned(
                            Product,
                            ValueScale,
                            accumulator,
                            previous_alpha,
                            previous_beta,
                            core,
                            sub,
                            tile - 1,
                            batch,
                            kvhead,
                            real_rows,
                            META,
                        )
                    else:
                        pass
                    previous_alpha = alpha
                    previous_beta = beta
                if tiles > 0:
                    accumulator = update_output_aligned(
                        Product,
                        ValueScale,
                        accumulator,
                        previous_alpha,
                        previous_beta,
                        core,
                        sub,
                        tiles - 1,
                        batch,
                        kvhead,
                        real_rows,
                        META,
                    )
                else:
                    pass
                normalized = accumulator / tl.maximum(denominator[:, None], 1.0)
                output_offsets = (
                    (batch * qlen + qstart + rows[:, None]) * heads + head
                ) * 128 + cols[None, :]
                tl.store(Output + output_offsets, normalized.to(tl.bfloat16))
            else:
                pass


@triton.jit
def update_output_generic(
    Product,
    ValueScale,
    accumulator,
    alpha,
    beta,
    core,
    sub,
    tile,
    batch,
    kvhead,
    real_rows,
    META: tl.constexpr,
):
    al.sync_block_wait("cube", "vector", 4 + 6 * (tile % 2))
    rows = sub * 64 + tl.arange(0, 64)
    cols = tl.arange(0, 128)
    base = (core * 2 + tile % 2) * 257 * 128
    high = tl.load(Product + base + rows[:, None] * 128 + cols[None, :]).to(tl.float32)
    correction = tl.load(Product + base + 128 * 128 + cols).to(tl.float32)
    low = tl.load(Product + base + 129 * 128 + rows[:, None] * 128 + cols[None, :]).to(
        tl.float32
    )
    value_scale = tl.load(
        ValueScale + batch * META[26] + kvhead * META[27] + tile * META[28]
    )
    product = tl.fma(low, 1.0 / 254.0, high - correction[None, :])
    coefficient = beta * value_scale
    return accumulator * alpha[:, None] + product * coefficient[:, None]


# Complete rows are allocated in score/PV workspaces. Padding rows remain
# independent in the rowwise math and Cube products; only output writes need
# query-tail masks. This permits 64-row processing without masked-load scratch.
@triton.jit
def vector_generic(
    QueryScale,
    KeyScale,
    ValueScale,
    Output,
    Workspace,
    WorkspaceI32,
    Used,
    Cuq,
    META: tl.constexpr,
):
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
        row_offsets = tl.arange(0, 64)
        if sub == 0:
            padding_words = tl.arange(0, 32)
            # Only row 128 contributes the signed-INT8 correction. The following
            # 15 M-alignment rows are never copied back by Cube.
            correction = tl.full((32,), -2139062144, tl.int32)
            for ring in tl.static_range(2):
                tl.store(
                    Score
                    + META[30] // 4
                    + (core * 2 + ring) * 272 * 32
                    + 128 * 32
                    + padding_words,
                    correction,
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
                real_rows = tl.minimum(128, qlen - qstart)
                rows = sub * 64 + row_offsets
                nk = tl.load(Used + batch)
                if causal:
                    visible = tl.minimum(
                        nk, tl.maximum(0, nk - qlen + qstart + real_rows)
                    )
                else:
                    visible = nk
                tiles = tl.cdiv(visible, 128)
                maximum = tl.full((64,), -3.4e38, tl.float32)
                denominator = tl.full((64,), 0, tl.float32)
                accumulator = tl.full((64, 128), 0, tl.float32)
                previous_alpha = tl.full((64,), 1, tl.float32)
                previous_beta = tl.full((64,), 0, tl.float32)
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
                    base = (core * 2 + tile % 2) * 128 * 128
                    indices = tile * 128 + cols
                    score = tl.load(
                        Score + base + rows[:, None] * 128 + cols[None, :]
                    ).to(tl.float32)
                    key_scale = tl.load(
                        KeyScale
                        + batch * META[23]
                        + kvhead * META[24]
                        + tile * META[25]
                    )
                    score = score * (query_scale * key_scale)
                    if causal:
                        allowed = nk - qlen + qstart + rows + 1
                    else:
                        allowed = tl.full((64,), nk, tl.int32)
                    valid = indices[None, :] < allowed[:, None]
                    score = tl.where(valid, score, -float("inf"))
                    local_max = tl.max(score, 1)
                    if causal:
                        has_values = allowed > tile * 128
                        local_max = tl.where(has_values, local_max, -3.4e38)
                    else:
                        pass
                    probability = tl.exp(score - local_max[:, None])
                    local_sum = tl.sum(probability, 1)
                    new_max = tl.maximum(maximum, local_max)
                    alpha = tl.exp(maximum - new_max)
                    beta = tl.exp(local_max - new_max)
                    if causal:
                        beta = tl.where(has_values, beta, 0.0)
                    else:
                        pass
                    denominator = denominator * alpha + local_sum * beta
                    maximum = new_max
                    beta = beta * (1.0 / 255.0)
                    scaled = probability * 255.0
                    high = al.custom(
                        "cast_fp32_to_int16",
                        tl.reshape(scaled, (8192,)),
                        4,
                        8192,
                        out=scratch16,
                    )
                    high_rows = high.reshape((64, 128))
                    pbase = (core * 2 + tile % 2) * 272 * 128
                    high_byte = al.custom(
                        "cast_fp16_to_int8",
                        tl.reshape((high_rows - 128).to(tl.float16), (8192,)),
                        5,
                        8192,
                        out=scratch8,
                    )
                    tl.store(
                        Prob + pbase + rows[:, None] * 128 + cols[None, :],
                        high_byte.reshape((64, 128)),
                    )
                    residual = (scaled - high_rows.to(tl.float32)) * 254.0
                    scratch16 = al.custom(
                        "cast_fp32_to_int16",
                        tl.reshape(residual, (8192,)),
                        4,
                        8192,
                        out=high,
                    )
                    scratch8 = al.custom(
                        "cast_fp16_to_int8",
                        scratch16.to(tl.float16),
                        5,
                        8192,
                        out=high_byte,
                    )
                    tl.store(
                        Prob + pbase + 144 * 128 + rows[:, None] * 128 + cols[None, :],
                        scratch8.reshape((64, 128)),
                    )
                    al.sync_block_set("vector", "cube", 3 + 6 * (tile % 2))
                    if tile > 0:
                        accumulator = update_output_generic(
                            Product,
                            ValueScale,
                            accumulator,
                            previous_alpha,
                            previous_beta,
                            core,
                            sub,
                            tile - 1,
                            batch,
                            kvhead,
                            real_rows,
                            META,
                        )
                    else:
                        pass
                    previous_alpha = alpha
                    previous_beta = beta
                if tiles > 0:
                    accumulator = update_output_generic(
                        Product,
                        ValueScale,
                        accumulator,
                        previous_alpha,
                        previous_beta,
                        core,
                        sub,
                        tiles - 1,
                        batch,
                        kvhead,
                        real_rows,
                        META,
                    )
                else:
                    pass
                if tiles > 0:
                    # Keep reciprocal at 64 elements before broadcasting.
                    inverse = libdevice.reciprocal(tl.maximum(denominator, 1.0))
                else:
                    inverse = tl.full((64,), 0.0, tl.float32)
                normalized = accumulator * inverse[:, None]
                output_offsets = (
                    (batch * qlen + qstart + rows[:, None]) * heads + head
                ) * 128 + cols[None, :]
                if META[15] % 128 == 0:
                    tl.store(Output + output_offsets, normalized.to(tl.bfloat16))
                else:
                    tl.store(
                        Output + output_offsets,
                        normalized.to(tl.bfloat16),
                        rows[:, None] < real_rows,
                    )
            else:
                pass
