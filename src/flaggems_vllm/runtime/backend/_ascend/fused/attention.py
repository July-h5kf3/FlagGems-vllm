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


import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.experimental.tle.language.dsa.ascend import custom_ops  # noqa: F401
from triton.language.extra.cann import libdevice

from flaggems_vllm.ops.flash_kernel import (
    apply_alibi,
    apply_mask,
    apply_softcap,
    softmax_rescale,
    virtual_to_cache_offset,
)
from flaggems_vllm.runtime import torch_device_fn


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


@triton.jit
def update_packed_output(
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
    rows = tl.arange(0, 8)
    cols = tl.arange(0, 128)
    base = (core * 2 + tile % 2) * 36 * 128
    source_rows = tl.arange(0, 32)
    original_base = base + sub * 18 * 128
    products = tl.load(
        Product + original_base + source_rows[:, None] * 128 + cols[None, :],
        source_rows[:, None] < 18,
        0,
    ).to(tl.float32)
    high0 = tle.dsa.extract_slice(
        products, [tl.full((), 0, tl.int32), tl.full((), 0, tl.int32)], [4, 128], [1, 1]
    )
    high1 = tle.dsa.extract_slice(
        products, [tl.full((), 9, tl.int32), tl.full((), 0, tl.int32)], [4, 128], [1, 1]
    )
    low0 = tle.dsa.extract_slice(
        products, [tl.full((), 4, tl.int32), tl.full((), 0, tl.int32)], [4, 128], [1, 1]
    )
    low1 = tle.dsa.extract_slice(
        products,
        [tl.full((), 13, tl.int32), tl.full((), 0, tl.int32)],
        [4, 128],
        [1, 1],
    )
    correction0 = tle.dsa.extract_slice(
        products, [tl.full((), 8, tl.int32), tl.full((), 0, tl.int32)], [1, 128], [1, 1]
    ).reshape((128,))
    correction1 = tle.dsa.extract_slice(
        products,
        [tl.full((), 17, tl.int32), tl.full((), 0, tl.int32)],
        [1, 128],
        [1, 1],
    ).reshape((128,))
    high = tl.full((8, 128), 0.0, tl.float32)
    high = tle.dsa.insert_slice(
        high,
        high0,
        [tl.full((), 0, tl.int32), tl.full((), 0, tl.int32)],
        [4, 128],
        [1, 1],
    )
    high = tle.dsa.insert_slice(
        high,
        high1,
        [tl.full((), 4, tl.int32), tl.full((), 0, tl.int32)],
        [4, 128],
        [1, 1],
    )
    low = tl.full((8, 128), 0.0, tl.float32)
    low = tle.dsa.insert_slice(
        low,
        low0,
        [tl.full((), 0, tl.int32), tl.full((), 0, tl.int32)],
        [4, 128],
        [1, 1],
    )
    low = tle.dsa.insert_slice(
        low,
        low1,
        [tl.full((), 4, tl.int32), tl.full((), 0, tl.int32)],
        [4, 128],
        [1, 1],
    )
    correction = tl.where(rows[:, None] < 4, correction0[None, :], correction1[None, :])
    product = tl.fma(low, low_scale[:, None], high - correction * correction_factor)
    extra_base = ((core * 2 + tile % 2) * 16 + sub * 8) * 128
    extra = tl.load(ExtraProduct + extra_base + rows[:, None] * 128 + cols[None, :]).to(
        tl.float32
    )
    product = tl.fma(extra, third_scale[:, None], product)
    coefficient = beta * value_scale
    return accumulator * alpha[:, None] + product * coefficient[:, None]


@triton.jit
def vector_packed(
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
    QT: tl.constexpr = 1
    ROWS: tl.constexpr = 16
    VR: tl.constexpr = 8
    PM: tl.constexpr = 64
    EM: tl.constexpr = 16
    COUNT: tl.constexpr = VR * N
    FIXED: tl.constexpr = False
    tl.static_assert(N == 512)
    with al.scope("vector"):
        core = tl.program_id(0).to(tl.int32)
        sub = al.sub_vec_id().to(tl.int32)
        heads: tl.constexpr = META[0]
        kvheads: tl.constexpr = META[1]
        blocks: tl.constexpr = META[9]
        groups: tl.constexpr = META[10]
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
            extra_padding = tl.arange(0, 2 * EM)
            word_cols = tl.arange(0, N // 4)
            ExtraWords = (address + META[30] + META[31] + META[32]).to(
                tl.pointer_type(tl.int32)
            )
            for ring in tl.static_range(2):
                tl.store(
                    ExtraWords
                    + ((core * 2 + ring) * 4 + sub * 2) * EM * (N // 4)
                    + extra_padding[:, None] * (N // 4)
                    + word_cols[None, :],
                    0,
                )
        else:
            ExtraProduct = Product
        cols = tl.arange(0, N)
        rows = tl.arange(0, VR)
        kv_rows = rows // 4
        query_rows = tl.full((8,), 0, tl.int32)
        head_rows = sub * 8 + rows
        physical_rows = head_rows
        packed_prob_rows = (sub * 2 + kv_rows) * 16 + rows % 4
        if sub == 0:
            words = tl.arange(0, N // 4)
            padding_rows = tl.arange(0, 8)
            kv_indices = tl.arange(0, 4)
            padding = tl.where(padding_rows == 0, -2139062144, 0)
            for ring in tl.static_range(2):
                tl.store(
                    Workspace
                    + META[30] // 4
                    + (
                        (core * 2 + ring) * 64
                        + kv_indices[:, None, None] * 16
                        + 8
                        + padding_rows[None, :, None]
                    )
                    * (N // 4)
                    + words[None, None, :],
                    tl.broadcast_to(padding[None, :, None], (4, 8, N // 4)),
                )
        else:
            pass
        scratch16 = tl.full((COUNT,), 0, tl.int16)
        scratch8 = tl.full((COUNT,), 0, tl.int8)
        for ordinal in range(tl.cdiv(groups, blocks)):
            group = core + ordinal * blocks
            if group < groups:
                batch = group // (kvheads // 4)
                kvhead = (group % (kvheads // 4)) * 4
                qstart = 0
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
                    )
                    value_scale0 = tl.load(
                        ValueScale + batch * META[26] + (kvhead + sub * 2) * META[27]
                    )
                    value_scale1 = tl.load(
                        ValueScale
                        + batch * META[26]
                        + (kvhead + sub * 2 + 1) * META[27]
                    )
                    value_scale = tl.where(rows < 4, value_scale0, value_scale1)
                    magnitude = tl.max(tl.abs(value_scale), 0)
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
                            key_scale0 = tl.load(
                                KeyScale
                                + batch * META[23]
                                + (kvhead + sub * 2) * META[24]
                            )
                            key_scale1 = tl.load(
                                KeyScale
                                + batch * META[23]
                                + (kvhead + sub * 2 + 1) * META[24]
                            )
                            key_scale = tl.where(rows < 4, key_scale0, key_scale1)
                            score = score * (
                                query_scale[:, None]
                                * (key_scale[:, None] * 0.08838834764831845)
                            )
                        else:
                            key_blocks = tile * (N // 128) + tl.arange(0, N // 128)
                            block_scale0 = tl.load(
                                KeyScale
                                + batch * META[23]
                                + (kvhead + sub * 2) * META[24]
                                + key_blocks * META[25],
                                key_blocks < tl.cdiv(nk, 128),
                                0,
                            )
                            block_scale1 = tl.load(
                                KeyScale
                                + batch * META[23]
                                + (kvhead + sub * 2 + 1) * META[24]
                                + key_blocks * META[25],
                                key_blocks < tl.cdiv(nk, 128),
                                0,
                            )
                            scale0 = tl.broadcast_to(
                                block_scale0[:, None], (N // 128, 128)
                            ).reshape((N,))
                            scale1 = tl.broadcast_to(
                                block_scale1[:, None], (N // 128, 128)
                            ).reshape((N,))
                            key_scale = tl.where(
                                rows[:, None] < 4, scale0[None, :], scale1[None, :]
                            )
                            score = score * (
                                query_scale[:, None] * (key_scale * 0.08838834764831845)
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
                                * magnitude
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
                                + packed_prob_rows[:, None] * N
                                + cols[None, :],
                                high.reshape((VR, N)),
                            )
                            low = pack_i8((narrow << 8).to(tl.int16) >> 8, high, COUNT)
                            tl.store(
                                Prob
                                + pbase
                                + (packed_prob_rows[:, None] + 4) * N
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
                                + packed_prob_rows[:, None] * N
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
                                    * magnitude
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
                                + (packed_prob_rows[:, None] + 4) * N
                                + cols[None, :],
                                low.reshape((VR, N)),
                            )
                            correction_factor = 1.0
                            beta = beta * (1.0 / 255.0)
                        if N >= 512:
                            extra_base = (core * 2 + tile % 2) * 4 * EM * N
                            extra_rows = (sub * 2 + kv_rows) * EM + rows % 4
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
                            accumulator = update_packed_output(
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
                        accumulator = update_packed_output(
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
                    normalized = accumulator / tl.maximum(denominator, 1.0)[:, None]
                    output_cols = tl.arange(0, 128)
                    output_heads = tl.arange(0, 8)
                    output_offsets = (
                        query_base.to(tl.int64) * heads
                        + kvhead * 4
                        + sub * 8
                        + output_heads[:, None]
                    ) * 128 + output_cols[None, :]
                    tl.store(Output + output_offsets, normalized.to(tl.bfloat16))
                else:
                    pass
            else:
                pass


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


@triton.jit
def update_hybrid_output(
    Product,
    accumulator,
    alpha,
    beta,
    value_scale,
    core,
    physical_rows,
    tile,
    ROWS: tl.constexpr,
    FIXED: tl.constexpr,
):
    al.sync_block_wait("cube", "vector", 4 + 6 * (tile % 2))
    cols = tl.arange(0, 128)
    base = (core * 2 + tile % 2) * (2 * ROWS + 1) * 128
    high = tl.load(Product + base + physical_rows[:, None] * 128 + cols[None, :]).to(
        tl.float32
    )
    low = tl.load(
        Product + base + (physical_rows[:, None] + ROWS) * 128 + cols[None, :]
    ).to(tl.float32)
    correction = tl.load(Product + base + 2 * ROWS * 128 + cols).to(tl.float32)
    if FIXED:
        product = tl.fma(low, 1.0 / 256.0, high - correction[None, :] * (127.0 / 128.0))
    else:
        product = tl.fma(low, 1.0 / 254.0, high - correction[None, :])
    coefficient = beta * value_scale
    return accumulator * alpha[:, None] + product * coefficient[:, None]


@triton.jit
def vector_hybrid_grouped(
    QueryScale,
    KeyScale,
    ValueScale,
    Output,
    Workspace,
    Used,
    Cuq,
    META: tl.constexpr,
    SELECTOR: tl.constexpr,
):
    QT: tl.constexpr = META[14]
    ROWS: tl.constexpr = QT * 4
    VR: tl.constexpr = QT * 2
    PM: tl.constexpr = triton.cdiv(2 * ROWS + 1, 16) * 16
    COUNT: tl.constexpr = VR * 256
    FIXED: tl.constexpr = QT >= 8
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
        rows = tl.arange(0, VR)
        query_rows = rows // 2
        head_rows = rows % 2 * 2 + sub
        physical_rows = rows * 2 + sub
        if sub == 0:
            correction_columns = tl.arange(0, 64)
            for ring in tl.static_range(2):
                tl.store(
                    Score
                    + META[30] // 4
                    + (core * 2 + ring) * PM * 64
                    + 2 * ROWS * 64
                    + correction_columns,
                    tl.full((64,), -2139062144, tl.int32),
                )
        else:
            pass
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
                if SELECTOR == 1:
                    selected = query_length <= 4
                else:
                    selected = query_length > 4
                if selected and qstart < query_length:
                    real_queries = tl.minimum(QT, query_length - qstart)
                    nk = tl.load(Used + batch)
                    if causal:
                        visible = tl.minimum(
                            nk, tl.maximum(0, nk - query_length + qstart + real_queries)
                        )
                    else:
                        visible = nk
                    tiles = tl.cdiv(visible, 256)
                    maximum = tl.full((VR,), -3.4e38, tl.float32)
                    denominator = tl.full((VR,), 0, tl.float32)
                    accumulator = tl.full((VR, 128), 0, tl.float32)
                    previous_alpha = tl.full((VR,), 1, tl.float32)
                    previous_beta = tl.full((VR,), 0, tl.float32)
                    query_scale0 = tl.load(
                        QueryScale
                        + batch * META[20]
                        + (kvhead * 4 + sub) * META[21]
                        + qstart // 128 * META[22]
                    )
                    query_scale1 = tl.load(
                        QueryScale
                        + batch * META[20]
                        + (kvhead * 4 + sub + 2) * META[21]
                        + qstart // 128 * META[22]
                    )
                    query_scale = tl.where(rows % 2 == 0, query_scale0, query_scale1)
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
                        base = (core * 2 + tile % 2) * ROWS * 256
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
                            allowed = tl.full((VR,), nk, tl.int32)
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
                        if FIXED:
                            beta = beta * (1.0 / 254.0)
                            encoded = probability * 65024.0 - 32512.0
                            narrow = al.custom(
                                "cast_fp32_to_int16",
                                encoded.reshape((COUNT,)),
                                4,
                                COUNT,
                                out=scratch16,
                            )
                            high_bits = (((narrow + 128).to(tl.int16)) >> 8).to(
                                tl.float16
                            )
                            high_byte = al.custom(
                                "cast_fp16_to_int8", high_bits, 5, COUNT, out=scratch8
                            )
                            pbase = (core * 2 + tile % 2) * PM * 256
                            tl.store(
                                Prob
                                + pbase
                                + physical_rows[:, None] * 256
                                + cols[None, :],
                                high_byte.reshape((VR, 256)),
                            )
                            low_bits = ((narrow << 8).to(tl.int16) >> 8).to(tl.float16)
                            scratch8 = al.custom(
                                "cast_fp16_to_int8", low_bits, 5, COUNT, out=high_byte
                            )
                            tl.store(
                                Prob
                                + pbase
                                + (physical_rows[:, None] + ROWS) * 256
                                + cols[None, :],
                                scratch8.reshape((VR, 256)),
                            )
                            scratch16 = narrow
                        else:
                            beta = beta * (1.0 / 255.0)
                            scaled = probability * 255.0
                            high = al.custom(
                                "cast_fp32_to_int16",
                                scaled.reshape((COUNT,)),
                                4,
                                COUNT,
                                out=scratch16,
                            )
                            high_rows = high.reshape((VR, 256))
                            high_byte = al.custom(
                                "cast_fp16_to_int8",
                                (high_rows - 128).to(tl.float16).reshape((COUNT,)),
                                5,
                                COUNT,
                                out=scratch8,
                            )
                            high_stored = high_byte.reshape((VR, 256))
                            pbase = (core * 2 + tile % 2) * PM * 256
                            tl.store(
                                Prob
                                + pbase
                                + physical_rows[:, None] * 256
                                + cols[None, :],
                                high_stored,
                            )
                            residual = (scaled - high_rows.to(tl.float32)) * 254.0
                            scratch16 = al.custom(
                                "cast_fp32_to_int16",
                                residual.reshape((COUNT,)),
                                4,
                                COUNT,
                                out=high,
                            )
                            low_rows = scratch16
                            scratch8 = al.custom(
                                "cast_fp16_to_int8",
                                low_rows.to(tl.float16),
                                5,
                                COUNT,
                                out=high_byte,
                            )
                            low_stored = scratch8.reshape((VR, 256))
                            tl.store(
                                Prob
                                + pbase
                                + (physical_rows[:, None] + ROWS) * 256
                                + cols[None, :],
                                low_stored,
                            )
                        al.sync_block_set("vector", "cube", 3 + 6 * (tile % 2))
                        if tile > 0:
                            accumulator = update_hybrid_output(
                                Product,
                                accumulator,
                                previous_alpha,
                                previous_beta,
                                value_scale,
                                core,
                                physical_rows,
                                tile - 1,
                                ROWS,
                                FIXED,
                            )
                        else:
                            pass
                        previous_alpha = alpha
                        previous_beta = beta
                    if tiles > 0:
                        accumulator = update_hybrid_output(
                            Product,
                            accumulator,
                            previous_alpha,
                            previous_beta,
                            value_scale,
                            core,
                            physical_rows,
                            tiles - 1,
                            ROWS,
                            FIXED,
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


@triton.jit
def update_head_output(
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
                        accumulator = update_head_output(
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
                    accumulator = update_head_output(
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
                        accumulator = update_head_output(
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
                    accumulator = update_head_output(
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


@triton.jit
def update_head_n256_output(
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
def head_n256_probability(
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
def vector_head_n256(
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
                            head_n256_probability(
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
                            head_n256_probability(
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
                            accumulator = update_head_n256_output(
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
                        accumulator = update_head_n256_output(
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


@triton.jit
def merge_head_half_state(first, second):
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
        merge_head_half_state(maximum0, maximum1),
        merge_head_half_state(denominator0, denominator1),
        merge_head_half_state(alpha0, alpha1),
        merge_head_half_state(beta0, beta1),
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
def vector_head_n512(
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


@triton.jit
def softmax_replay_half(
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
        softmax_replay_half(
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
def prepare_replay_slice(
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
        merge_head_half_state(maximum0, maximum1),
        merge_head_half_state(denominator0, denominator1),
        merge_head_half_state(alpha0, alpha1),
        merge_head_half_state(beta0, beta1),
        merge_head_half_state(low0, low1),
        merge_head_half_state(third0, third1),
        merge_head_half_state(correction0, correction1),
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
def vector_replay(
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
                            ) = prepare_replay_slice(
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


MTE2 = tl.constexpr(tle.dsa.TilePipe.MTE2.value)


MTE1 = tl.constexpr(tle.dsa.TilePipe.MTE1.value)


M = tl.constexpr(tle.dsa.TilePipe.M.value)


FIX = tl.constexpr(tle.dsa.TilePipe.FIX.value)


@triton.jit
def local_sync(PRODUCER: tl.constexpr, CONSUMER: tl.constexpr):
    tle.dsa.tile_set_flag(PRODUCER, CONSUMER, 0)
    tle.dsa.tile_wait_flag(PRODUCER, CONSUMER, 0)


@triton.jit
def load_matrix_a(
    src_address,
    dst_address,
    ROWS,
    CHANNELS: tl.constexpr,
    K,
    k_start,
    RESET_MATRIX: tl.constexpr = 1,
    RESET_PADDING: tl.constexpr = 0,
    load_rows=None,
):
    matrix_rows = ROWS if load_rows is None else load_rows
    al.custom(
        "cube_load3d_a_into",
        src_address,
        [0, 0, 0, 255],
        1,
        ROWS,
        CHANNELS,
        K,
        matrix_rows,
        k_start,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        RESET_MATRIX,
        RESET_PADDING,
        dst_address,
    )


@triton.jit
def copy_page_run(
    SRC,
    physical,
    count,
    address,
    head,
    PS: tl.constexpr,
    RS: tl.constexpr,
    C: tl.constexpr,
    nz_rows,
    FEATURES: tl.constexpr = 128,
):
    ptr = SRC.to(tl.uint64) + (physical.to(tl.uint32) * PS + head * 128).to(tl.uint64)
    al.custom(
        "cube_nd2nz_i8", address, ptr, 1, count * 16, FEATURES, 0, RS, nz_rows, 1, 0
    )


@triton.jit
def copy_paged_tile(
    SRC,
    TABLE,
    batch,
    head,
    tile,
    nk,
    base_address,
    TS: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    PADDED_ROWS: tl.constexpr = False,
    FEATURES: tl.constexpr = 128,
):
    pages = (nk + 15) >> 4
    for part in tl.static_range(N // C):
        start = tile * (N // 16) + part * (C // 16)
        count = tl.minimum(C // 16, tl.maximum(0, pages - start))
        if count > 0:
            table_base = TABLE + (batch * TS + start).to(tl.uint64)
            first = tl.load(table_base)
            run = 1
            written = 0
            for page in range(1, count):
                following = tl.load(table_base + page)
                if following == first + run:
                    run += 1
                else:
                    copy_page_run(
                        SRC,
                        first,
                        run,
                        base_address + part * C * FEATURES + written * 16 * 32,
                        head,
                        PAGE_STRIDE,
                        ROW_STRIDE,
                        C,
                        C if PADDED_ROWS else ((count + 1) >> 1) * 32,
                        FEATURES,
                    )
                    written += run
                    first = following
                    run = 1
            copy_page_run(
                SRC,
                first,
                run,
                base_address + part * C * FEATURES + written * 16 * 32,
                head,
                PAGE_STRIDE,
                ROW_STRIDE,
                C,
                C if PADDED_ROWS else ((count + 1) >> 1) * 32,
                FEATURES,
            )
        else:
            pass


@triton.jit
def compute_score(
    KEY,
    TABLE,
    SCORE,
    core,
    tile,
    batch,
    head,
    nk,
    real_queries,
    L1_KEY: tl.constexpr,
    TS: tl.constexpr,
    PS: tl.constexpr,
    RS: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    SCORE_ROWS: tl.constexpr = 128,
    FIXED_KEYS: tl.constexpr = False,
):
    copy_paged_tile(
        KEY, TABLE, batch, head, tile, nk, L1_KEY, TS, PS, RS, N, C, FIXED_KEYS
    )
    local_sync(MTE2, MTE1)
    for part in tl.static_range(N // C):
        valid_keys = C if FIXED_KEYS else tl.minimum(C, nk - tile * N - part * C)
        if valid_keys > 0:
            key_rows = (valid_keys + 31) & -32
            al.custom(
                "cube_load2d_b_into",
                L1_KEY + part * C * 128,
                0,
                key_rows * 128 // 512,
                1,
                0,
                0,
                0,
                0,
                0,
            )
            local_sync(MTE1, MTE2)
            local_sync(MTE1, M)
            tle.dsa.tile_wait_flag(FIX, M, 0)
            q_rows = (real_queries + 15) & -16
            al.custom("cube_mmad_into", 0, 0, q_rows, 128, key_rows, 3, 0, 0, 1, 0)
            local_sync(M, MTE1)
            local_sync(M, FIX)
            score_ptr = SCORE + ((core * 2 + tile % 2) * SCORE_ROWS * N + part * C) * 4
            al.custom(
                "cube_copy_l0c2gm_i32",
                score_ptr,
                0,
                key_rows,
                real_queries,
                N,
                q_rows,
                3,
                0,
                0,
                0,
                1,
            )
            tle.dsa.tile_set_flag(FIX, M, 0)
        else:
            pass
    al.sync_block_set("cube", "vector", 2 + 6 * (tile % 2))


@triton.jit
def probability_product(
    PROB,
    PRODUCT,
    prob_offset,
    product_offset,
    L1_PROB: tl.constexpr,
    L0_PROB: tl.constexpr,
    ROWS: tl.constexpr,
    STORE_ROWS: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    L1_VALUE: tl.constexpr = 0,
    LOAD_VALUE: tl.constexpr = False,
    real_rows=None,
    real_store=None,
    active_keys=None,
    STREAM_VALUE: tl.constexpr = False,
    VALUE_NZ_ROWS: tl.constexpr = None,
    EXTRA_PLANES: tl.constexpr = 2,
    EXTRA_ROWS: tl.constexpr = 0,
    EXTRA_STORE_ROWS: tl.constexpr = 0,
    EXTRA_PROB=None,
    EXTRA_PRODUCT=None,
    extra_prob_offset=0,
    extra_product_offset=0,
):
    key_count = N if active_keys is None else active_keys
    active_rows = ROWS if real_rows is None else real_rows
    stored_rows = STORE_ROWS if real_store is None else real_store
    matrix_rows = active_rows + EXTRA_PLANES * EXTRA_ROWS
    ptr = PROB + prob_offset
    al.custom("cube_nd2nz_i8", L1_PROB, ptr, 1, active_rows, N, 0, N, matrix_rows, 1, 0)
    if EXTRA_ROWS:
        for half in tl.static_range(EXTRA_PLANES):
            al.custom(
                "cube_nd2nz_i8",
                L1_PROB + (active_rows + half * EXTRA_ROWS) * 32,
                EXTRA_PROB + extra_prob_offset + half * EXTRA_ROWS * N,
                1,
                EXTRA_ROWS,
                N,
                0,
                N,
                matrix_rows,
                1,
                0,
            )
    else:
        pass
    local_sync(MTE2, MTE1)
    for part in tl.static_range(N // C):
        if part * C < key_count:
            key_rows = tl.minimum(C, ((key_count - part * C + 31) & -32))
            load_matrix_a(
                L1_PROB,
                L0_PROB,
                matrix_rows,
                N,
                key_rows,
                part * C,
                1 if part == 0 else 0,
                0,
            )
            if LOAD_VALUE and (part == 0 or STREAM_VALUE):
                for value_index in tl.static_range(1 if STREAM_VALUE else N // C):
                    value_part = part if STREAM_VALUE else value_index
                    value_rows = tl.minimum(
                        C, ((key_count - value_part * C + 31) & -32)
                    )
                    for section in tl.static_range(C // 32):
                        if section * 32 < value_rows:
                            al.custom(
                                "cube_load_transpose_b_into",
                                L1_VALUE + value_part * C * 128 + section * 32 * 32,
                                0,
                                4,
                                (
                                    value_rows // 32
                                    if VALUE_NZ_ROWS is None
                                    else VALUE_NZ_ROWS // 32
                                ),
                                1,
                                0,
                                0,
                                (0 if STREAM_VALUE else value_part * C * 128)
                                + section * 32 * 128,
                            )
                        else:
                            pass
            else:
                pass
            local_sync(MTE1, MTE2)
            local_sync(MTE1, M)
            if part == 0:
                tle.dsa.tile_wait_flag(FIX, M, 0)
            else:
                pass
            al.custom(
                "cube_mmad_into",
                L0_PROB,
                0 if STREAM_VALUE else part * C * 128,
                matrix_rows,
                key_rows,
                128,
                0,
                0,
                0,
                1 if part == 0 else 0,
                0,
            )
            local_sync(M, MTE1)
        else:
            pass
    local_sync(M, FIX)
    output = PRODUCT + product_offset * 4
    al.custom(
        "cube_copy_l0c2gm_i32",
        output,
        0,
        128,
        stored_rows,
        128,
        matrix_rows,
        0,
        0,
        0,
        0,
        1,
    )
    if EXTRA_ROWS:
        for half in tl.static_range(EXTRA_PLANES):
            extra_output = (
                EXTRA_PRODUCT
                + (extra_product_offset + half * EXTRA_STORE_ROWS * 128) * 4
            )
            al.custom(
                "cube_copy_l0c2gm_i32",
                extra_output,
                (active_rows + half * EXTRA_ROWS) * 16 * 4,
                128,
                EXTRA_STORE_ROWS,
                128,
                matrix_rows,
                0,
                0,
                0,
                0,
                1,
            )
    else:
        pass
    tle.dsa.tile_set_flag(FIX, M, 0)


@triton.jit
def launch_cube_tle(
    Query,
    Key,
    Value,
    Table,
    Used,
    QueryScale,
    KeyScale,
    ValueScale,
    Output,
    Workspace,
    Cuq,
    WorkspaceI32,
    BUNDLE_KEY: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    META: tl.constexpr,
):
    tl.static_assert(MODE == 3 or MODE == 4 or MODE == 5)
    C: tl.constexpr = 128 if N == 128 else 256
    L1_KEY: tl.constexpr = 128 * 128
    L1_VALUE: tl.constexpr = L1_KEY + N * 128
    L1_PROB: tl.constexpr = L1_VALUE + N * 128
    L0_PROB: tl.constexpr = 128 * 128
    tl.static_assert(L1_PROB + 144 * N <= 512 * 1024)
    tl.static_assert(L0_PROB + 144 * C <= 64 * 1024)
    tl.static_assert(N * 128 <= 64 * 1024)
    tl.static_assert(128 * C * 4 <= 128 * 1024)
    tl.static_assert(144 * 128 * 4 <= 128 * 1024)
    core = tl.program_id(0).to(tl.uint32)
    with al.scope(core_mode="cube"):
        HQ: tl.constexpr = META[0]
        HK: tl.constexpr = META[1]
        TS: tl.constexpr = META[4]
        PS: tl.constexpr = META[5]
        RS: tl.constexpr = META[6]
        BLOCKS: tl.constexpr = META[9]
        GROUPS: tl.constexpr = META[10]
        QLEN: tl.constexpr = META[15]
        QGROUPS: tl.constexpr = META[16]
        CAUSAL: tl.constexpr = META[18]
        SB: tl.constexpr = META[30]
        PB: tl.constexpr = META[31]
        VB: tl.constexpr = META[32]
        SCORE = Workspace.to(tl.uint64)
        PROB = SCORE + SB
        PRODUCT = SCORE + SB + PB
        EXTRA_P = SCORE + SB + PB + VB
        EXTRA_PRODUCT = SCORE + SB + PB + VB + META[36]
        if N == 512:
            CONTROL = WorkspaceI32 + (SB + PB + VB + META[36] + META[37]) // 4
        else:
            pass
        al.custom("cube_set_l0c_copy_params", 1, 0, 0)
        tle.dsa.tile_set_flag(FIX, M, 0)
        for ordinal in range(tl.cdiv(GROUPS, BLOCKS)):
            group = core + ordinal * BLOCKS
            if group < GROUPS:
                batch = group // (QGROUPS * HQ)
                head = group % HQ
                kvhead = head // (HQ // HK)
                qstart = (group // HQ % QGROUPS * 128).to(tl.int32)
                if MODE == 4:
                    query_base = tl.load(Cuq + batch).to(tl.uint32)
                    query_length = (
                        tl.load(Cuq + batch + 1).to(tl.uint32) - query_base
                    ).to(tl.int32)
                    active_group = (query_length > 4) & (qstart < query_length)
                else:
                    query_base = batch * QLEN
                    query_length = QLEN
                    active_group = True
                if MODE != 4 and QLEN % 128 == 0:
                    real_queries = 128
                else:
                    real_queries = tl.minimum(128, query_length - qstart)
                nk = tl.load(Used + batch)
                if MODE == 5:
                    twice_visible = 2 * nk.to(tl.int64) - (QLEN - 1 if CAUSAL else 0)
                    use_wide = twice_visible >= (6144 if CAUSAL else 4096)
                    active_group = use_wide == (N == 512)
                else:
                    pass
                if active_group:
                    if CAUSAL:
                        visible = tl.minimum(
                            nk, tl.maximum(0, nk - query_length + qstart + real_queries)
                        )
                    else:
                        visible = nk
                    tiles = tl.cdiv(visible, N)
                    if tiles > 0:
                        qp = Query.to(tl.uint64) + (
                            ((query_base + qstart.to(tl.uint32)) * HQ + head) * 128
                        ).to(tl.uint64)
                        al.custom(
                            "cube_nd2nz_i8",
                            0,
                            qp,
                            1,
                            real_queries,
                            128,
                            0,
                            HQ * 128,
                            128,
                            1,
                            0,
                        )
                        local_sync(MTE2, MTE1)
                        load_matrix_a(
                            0,
                            0,
                            128,
                            128,
                            128,
                            0,
                            1,
                            1,
                            load_rows=(real_queries + 15) & -16,
                        )
                        local_sync(MTE1, MTE2)
                        local_sync(MTE1, M)
                        compute_score(
                            Key,
                            Table,
                            SCORE,
                            core,
                            0,
                            batch,
                            kvhead,
                            nk,
                            real_queries,
                            L1_KEY,
                            TS,
                            PS,
                            RS,
                            N,
                            C,
                        )
                        for tile in range(tiles):
                            if tile + 1 < tiles:
                                compute_score(
                                    Key,
                                    Table,
                                    SCORE,
                                    core,
                                    tile + 1,
                                    batch,
                                    kvhead,
                                    nk,
                                    real_queries,
                                    L1_KEY,
                                    TS,
                                    PS,
                                    RS,
                                    N,
                                    C,
                                )
                            else:
                                pass
                            copy_paged_tile(
                                Value,
                                Table,
                                batch,
                                kvhead,
                                tile,
                                nk,
                                L1_VALUE,
                                TS,
                                PS,
                                RS,
                                N,
                                C,
                            )
                            al.sync_block_wait("vector", "cube", 3 + 6 * (tile % 2))
                            tile_keys = tl.minimum(N, nk - tile * N)
                            ring = core * 2 + tile % 2
                            if real_queries < 128:
                                aligned_rows = (real_queries + 15) & -16
                                probability_product(
                                    PROB,
                                    PRODUCT,
                                    ring * 272 * N,
                                    ring * 257 * 128,
                                    L1_PROB,
                                    L0_PROB,
                                    128,
                                    128,
                                    N,
                                    C,
                                    L1_VALUE,
                                    True,
                                    aligned_rows,
                                    real_queries,
                                    active_keys=tile_keys,
                                )
                                probability_product(
                                    PROB,
                                    PRODUCT,
                                    ring * 272 * N + 128 * N,
                                    ring * 257 * 128 + 128 * 128,
                                    L1_PROB,
                                    L0_PROB,
                                    16,
                                    1,
                                    N,
                                    C,
                                    active_keys=tile_keys,
                                )
                                probability_product(
                                    PROB,
                                    PRODUCT,
                                    ring * 272 * N + 144 * N,
                                    ring * 257 * 128 + 129 * 128,
                                    L1_PROB,
                                    L0_PROB,
                                    128,
                                    128,
                                    N,
                                    C,
                                    0,
                                    False,
                                    aligned_rows,
                                    real_queries,
                                    active_keys=tile_keys,
                                )
                            else:
                                probability_product(
                                    PROB,
                                    PRODUCT,
                                    ring * 272 * N,
                                    ring * 257 * 128,
                                    L1_PROB,
                                    L0_PROB,
                                    144,
                                    129,
                                    N,
                                    C,
                                    L1_VALUE,
                                    True,
                                    active_keys=tile_keys,
                                )
                                probability_product(
                                    PROB,
                                    PRODUCT,
                                    ring * 272 * N + 144 * N,
                                    ring * 257 * 128 + 129 * 128,
                                    L1_PROB,
                                    L0_PROB,
                                    128,
                                    128,
                                    N,
                                    C,
                                    active_keys=tile_keys,
                                )
                            if N == 512:
                                mask = tl.load(CONTROL + ring * 16, volatile=True) | (
                                    tl.load(CONTROL + ring * 16 + 8, volatile=True) << 2
                                )
                                for chunk in range(4):
                                    if mask & (1 << chunk):
                                        probability_product(
                                            EXTRA_P,
                                            EXTRA_PRODUCT,
                                            ring * 128 * N + chunk * 32 * N,
                                            ring * 128 * 128 + chunk * 32 * 128,
                                            L1_PROB,
                                            L0_PROB,
                                            32,
                                            32,
                                            N,
                                            C,
                                            active_keys=tile_keys,
                                        )
                                    else:
                                        pass
                            else:
                                pass
                            al.sync_block_set("cube", "vector", 4 + 6 * (tile % 2))
                    else:
                        pass
                else:
                    pass
            else:
                pass
        tle.dsa.tile_wait_flag(FIX, M, 0)
        tl.debug_barrier()
    if N == 128 and MODE == 3:
        if META[15] % 128 == 0 and META[10] <= META[9]:
            vector_aligned(
                QueryScale,
                KeyScale,
                ValueScale,
                Output,
                Workspace,
                WorkspaceI32,
                Used,
                Cuq,
                META,
            )
        else:
            vector_generic(
                QueryScale,
                KeyScale,
                ValueScale,
                Output,
                Workspace,
                WorkspaceI32,
                Used,
                Cuq,
                META,
            )
    elif N == 256:
        vector_head_n256(
            QueryScale,
            KeyScale,
            ValueScale,
            Output,
            WorkspaceI32,
            WorkspaceI32,
            Used,
            Cuq,
            N,
            MODE,
            META,
        )
    elif N == 512:
        vector_head_n512(
            QueryScale,
            KeyScale,
            ValueScale,
            Output,
            WorkspaceI32,
            WorkspaceI32,
            Used,
            Cuq,
            N,
            MODE,
            META,
        )
    else:
        tl.static_assert(False, "Unsupported head geometry")


@triton.jit
def grouped_cube(
    Query,
    Key,
    Value,
    Table,
    Used,
    Workspace,
    Cuq,
    N: tl.constexpr,
    META: tl.constexpr,
    SELECTOR: tl.constexpr,
):
    tl.static_assert(N == 128 or N == 256 or N == 512 or N == 1024)
    C: tl.constexpr = 512 if N >= 512 else N
    EXTRA_ROWS: tl.constexpr = tl.cdiv(META[14] * 2, 16) * 16
    QUERY_TILE: tl.constexpr = META[14]
    ROWS: tl.constexpr = QUERY_TILE * 4
    MATRIX_ROWS: tl.constexpr = tl.cdiv(ROWS, 16) * 16
    PRODUCT_ROWS: tl.constexpr = 2 * ROWS + 1
    PRODUCT_MATRIX_ROWS: tl.constexpr = tl.cdiv(PRODUCT_ROWS, 16) * 16
    LIVE_PROB_ROWS: tl.constexpr = PRODUCT_MATRIX_ROWS + (
        2 * EXTRA_ROWS if N >= 512 else 0
    )
    L1_KEY: tl.constexpr = MATRIX_ROWS * 128
    L1_VALUE: tl.constexpr = L1_KEY + N * 128
    L1_PROB: tl.constexpr = L1_VALUE + N * 128
    L0_PROB: tl.constexpr = MATRIX_ROWS * 128
    tl.static_assert(L1_PROB + LIVE_PROB_ROWS * N <= 512 * 1024)
    tl.static_assert(L0_PROB + LIVE_PROB_ROWS * C <= 64 * 1024)
    tl.static_assert(C * 128 <= 64 * 1024)
    tl.static_assert(MATRIX_ROWS * C * 4 <= 128 * 1024)
    tl.static_assert(LIVE_PROB_ROWS * 128 * 4 <= 128 * 1024)
    core = tl.program_id(0).to(tl.uint32)
    with al.scope("cube"):
        HQ: tl.constexpr = META[0]
        HK: tl.constexpr = META[1]
        TS: tl.constexpr = META[4]
        PS: tl.constexpr = META[5]
        RS: tl.constexpr = META[6]
        BLOCKS: tl.constexpr = META[9]
        GROUPS: tl.constexpr = META[10]
        QGROUPS: tl.constexpr = META[16]
        CAUSAL: tl.constexpr = META[18]
        SCORE = Workspace.to(tl.uint64)
        PROB = SCORE + META[30]
        PRODUCT = PROB + META[31]
        if N >= 512:
            EXTRA_PROB = PRODUCT + META[32]
            EXTRA_PRODUCT = EXTRA_PROB + META[36]
        else:
            pass
        al.custom("cube_set_l0c_copy_params", 1, 0, 0)
        tle.dsa.tile_set_flag(FIX, M, 0)
        for ordinal in range(tl.cdiv(GROUPS, BLOCKS)):
            group = core + ordinal * BLOCKS
            if group < GROUPS:
                batch = group // (QGROUPS * HK)
                kvhead = group % HK
                qstart = (group // HK % QGROUPS * QUERY_TILE).to(tl.int32)
                query_base = tl.load(Cuq + batch).to(tl.uint32)
                query_length = (tl.load(Cuq + batch + 1).to(tl.uint32) - query_base).to(
                    tl.int32
                )
                if SELECTOR == 1:
                    selected = query_length <= 4
                elif SELECTOR == 2:
                    selected = query_length > 4
                else:
                    selected = True
                if selected and qstart < query_length:
                    real_queries = tl.minimum(QUERY_TILE, query_length - qstart)
                    nk = tl.load(Used + batch)
                    if CAUSAL:
                        visible = tl.minimum(
                            nk, tl.maximum(0, nk - query_length + qstart + real_queries)
                        )
                    else:
                        visible = nk
                    tiles = tl.cdiv(visible, N)
                    if tiles > 0:
                        for row in tl.static_range(QUERY_TILE):
                            safe_row = tl.where(row < real_queries, row, 0)
                            query_ptr = Query.to(tl.uint64) + (
                                (
                                    (query_base + qstart.to(tl.uint32) + safe_row) * HQ
                                    + kvhead * 4
                                )
                                * 128
                            ).to(tl.uint64)
                            al.custom(
                                "cube_nd2nz_i8",
                                row * 4 * 32,
                                query_ptr,
                                1,
                                4,
                                128,
                                0,
                                128,
                                MATRIX_ROWS,
                                1,
                                0,
                            )
                        local_sync(MTE2, MTE1)
                        load_matrix_a(0, 0, MATRIX_ROWS, 128, 128, 0)
                        local_sync(MTE1, MTE2)
                        local_sync(MTE1, M)
                        compute_score(
                            Key,
                            Table,
                            SCORE,
                            core,
                            0,
                            batch,
                            kvhead,
                            nk,
                            ROWS,
                            L1_KEY,
                            TS,
                            PS,
                            RS,
                            N,
                            C,
                            ROWS,
                            True,
                        )
                        for tile in range(tiles):
                            if tile + 1 < tiles:
                                compute_score(
                                    Key,
                                    Table,
                                    SCORE,
                                    core,
                                    tile + 1,
                                    batch,
                                    kvhead,
                                    nk,
                                    ROWS,
                                    L1_KEY,
                                    TS,
                                    PS,
                                    RS,
                                    N,
                                    C,
                                    ROWS,
                                    True,
                                )
                            else:
                                pass
                            copy_paged_tile(
                                Value,
                                Table,
                                batch,
                                kvhead,
                                tile,
                                nk,
                                L1_VALUE,
                                TS,
                                PS,
                                RS,
                                N,
                                C,
                                True,
                            )
                            if N >= 512:
                                al.sync_block_wait(
                                    "vector",
                                    "cube",
                                    3 + 6 * (tile % 2),
                                    sender_pipe=al.PIPE.PIPE_MTE3,
                                    receiver_pipe=al.PIPE.PIPE_S,
                                )
                            else:
                                al.sync_block_wait("vector", "cube", 3 + 6 * (tile % 2))
                            probability_product(
                                PROB,
                                PRODUCT,
                                (core * 2 + tile % 2) * PRODUCT_MATRIX_ROWS * N,
                                (core * 2 + tile % 2) * PRODUCT_ROWS * 128,
                                L1_PROB,
                                L0_PROB,
                                PRODUCT_MATRIX_ROWS,
                                PRODUCT_ROWS,
                                N,
                                C,
                                L1_VALUE,
                                True,
                                active_keys=tl.minimum(N, nk - tile * N),
                                STREAM_VALUE=N == 1024,
                                VALUE_NZ_ROWS=C,
                                EXTRA_ROWS=EXTRA_ROWS if N >= 512 else 0,
                                EXTRA_STORE_ROWS=ROWS // 2 if N >= 512 else 0,
                                EXTRA_PROB=EXTRA_PROB if N >= 512 else None,
                                EXTRA_PRODUCT=EXTRA_PRODUCT if N >= 512 else None,
                                extra_prob_offset=(core * 2 + tile % 2)
                                * 2
                                * EXTRA_ROWS
                                * N,
                                extra_product_offset=(core * 2 + tile % 2) * ROWS * 128,
                            )
                            al.sync_block_set("cube", "vector", 4 + 6 * (tile % 2))
                    else:
                        pass
                else:
                    pass
            else:
                pass
        tle.dsa.tile_wait_flag(FIX, M, 0)
        tl.debug_barrier()


@triton.jit
def launch_hybrid_small_tle(
    Query,
    Key,
    Value,
    Table,
    Used,
    QueryScale,
    KeyScale,
    ValueScale,
    Output,
    Workspace,
    Cuq,
    BUNDLE_KEY: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    META: tl.constexpr,
):
    tl.static_assert(MODE == 1 and N == 256 and META[14] == 2)
    grouped_cube(Query, Key, Value, Table, Used, Workspace, Cuq, N, META, 1)
    vector_hybrid_small(
        QueryScale, KeyScale, ValueScale, Output, Workspace, Used, Cuq, META
    )


@triton.jit
def launch_hybrid_large_tle(
    Query,
    Key,
    Value,
    Table,
    Used,
    QueryScale,
    KeyScale,
    ValueScale,
    Output,
    Workspace,
    Cuq,
    BUNDLE_KEY: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    META: tl.constexpr,
):
    tl.static_assert(N == 256 and MODE == 2 and META[14] == 16)
    grouped_cube(Query, Key, Value, Table, Used, Workspace, Cuq, N, META, 2)
    vector_hybrid_grouped(
        QueryScale, KeyScale, ValueScale, Output, Workspace, Used, Cuq, META, 2
    )


@triton.jit
def launch_grouped_tle(
    Query,
    Key,
    Value,
    Table,
    Used,
    QueryScale,
    KeyScale,
    ValueScale,
    Output,
    Workspace,
    Cuq,
    BUNDLE_KEY: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    META: tl.constexpr,
):
    tl.static_assert(MODE == 0)
    if META[4] == 0 or META[7] == 0 or META[8] == 0:
        batch: tl.constexpr = META[10] // (META[16] * META[1])
        total = tl.load(Cuq + batch).to(tl.int64) * META[0] * 128
        offsets = tl.arange(0, 256)
        for start in range(tl.program_id(0) * 256, total, tl.num_programs(0) * 256):
            tl.store(Output + start + offsets, 0, start + offsets < total)
    else:
        grouped_cube(Query, Key, Value, Table, Used, Workspace, Cuq, N, META, 0)
        vector_grouped(
            QueryScale, KeyScale, ValueScale, Output, Workspace, Used, Cuq, N, META
        )


@triton.jit
def packed_score(
    Key,
    Table,
    Score,
    core,
    tile,
    batch,
    kvhead,
    nk,
    TS: tl.constexpr,
    PS: tl.constexpr,
    RS: tl.constexpr,
):
    copy_paged_tile(
        Key, Table, batch, kvhead, tile, nk, 8192, TS, PS, RS, 512, 512, True, 512
    )
    local_sync(MTE2, MTE1)
    for head in tl.static_range(4):
        load_matrix_a(head * 2048, 0, 16, 128, 128, 0)
        al.custom(
            "cube_load2d_b_into",
            8192 + head * 512 * 128,
            0,
            512 * 128 // 512,
            1,
            0,
            0,
            0,
            0,
            0,
        )
        local_sync(MTE1, MTE2)
        local_sync(MTE1, M)
        tle.dsa.tile_wait_flag(FIX, M, 0)
        al.custom("cube_mmad_into", 0, 0, 16, 128, 512, 0, 0, 0, 1, 0)
        local_sync(M, MTE1)
        local_sync(M, FIX)
        destination = Score + ((core * 2 + tile % 2) * 16 * 512 + head * 4 * 512) * 4
        al.custom(
            "cube_copy_l0c2gm_i32", destination, 0, 512, 4, 512, 16, 0, 0, 0, 0, 1
        )
        tle.dsa.tile_set_flag(FIX, M, 0)
    al.sync_block_set("cube", "vector", 2 + 6 * (tile % 2))


@triton.jit
def packed_cube(Query, Key, Value, Table, Used, Workspace, Cuq, META: tl.constexpr):
    tl.static_assert(270336 + 32 * 512 <= 512 * 1024)
    tl.static_assert(2048 + 32 * 512 <= 64 * 1024)
    tl.static_assert(512 * 128 <= 64 * 1024)
    tl.static_assert(16 * 512 * 4 <= 128 * 1024)
    core = tl.program_id(0).to(tl.uint32)
    with al.scope("cube"):
        HQ: tl.constexpr = META[0]
        HK: tl.constexpr = META[1]
        TS: tl.constexpr = META[4]
        PS: tl.constexpr = META[5]
        RS: tl.constexpr = META[6]
        BLOCKS: tl.constexpr = META[9]
        GROUPS: tl.constexpr = META[10]
        SCORE = Workspace.to(tl.uint64)
        PROB = SCORE + META[30]
        PRODUCT = PROB + META[31]
        EXTRA_PROB = PRODUCT + META[32]
        EXTRA_PRODUCT = EXTRA_PROB + META[36]
        al.custom("cube_set_l0c_copy_params", 1, 0, 0)
        tle.dsa.tile_set_flag(FIX, M, 0)
        for ordinal in range(tl.cdiv(GROUPS, BLOCKS)):
            group = core + ordinal * BLOCKS
            if group < GROUPS:
                batch = group // (HK // 4)
                kvhead = (group % (HK // 4)) * 4
                query_base = tl.load(Cuq + batch).to(tl.uint32)
                nk = tl.load(Used + batch)
                tiles = tl.cdiv(nk, 512)
                if tiles > 0:
                    for head in tl.static_range(4):
                        query_ptr = Query.to(tl.uint64) + (
                            (query_base * HQ + kvhead * 4 + head * 4) * 128
                        ).to(tl.uint64)
                        al.custom(
                            "cube_nd2nz_i8",
                            head * 2048,
                            query_ptr,
                            1,
                            4,
                            128,
                            0,
                            128,
                            16,
                            1,
                            0,
                        )
                    local_sync(MTE2, MTE1)
                    packed_score(
                        Key, Table, SCORE, core, 0, batch, kvhead, nk, TS, PS, RS
                    )
                    for tile in range(tiles):
                        if tile + 1 < tiles:
                            packed_score(
                                Key,
                                Table,
                                SCORE,
                                core,
                                tile + 1,
                                batch,
                                kvhead,
                                nk,
                                TS,
                                PS,
                                RS,
                            )
                        else:
                            pass
                        copy_paged_tile(
                            Value,
                            Table,
                            batch,
                            kvhead,
                            tile,
                            nk,
                            8192,
                            TS,
                            PS,
                            RS,
                            512,
                            512,
                            True,
                            512,
                        )
                        al.sync_block_wait("vector", "cube", 3 + 6 * (tile % 2))
                        ring = core * 2 + tile % 2
                        for head in tl.static_range(4):
                            probability_product(
                                PROB,
                                PRODUCT,
                                (ring * 64 + head * 16) * 512,
                                (ring * 36 + head * 9) * 128,
                                270336,
                                2048,
                                16,
                                9,
                                512,
                                512,
                                8192 + head * 512 * 128,
                                True,
                                active_keys=tl.minimum(512, nk - tile * 512),
                                VALUE_NZ_ROWS=512,
                                EXTRA_PLANES=1,
                                EXTRA_ROWS=16,
                                EXTRA_STORE_ROWS=4,
                                EXTRA_PROB=EXTRA_PROB,
                                EXTRA_PRODUCT=EXTRA_PRODUCT,
                                extra_prob_offset=(ring * 4 + head) * 16 * 512,
                                extra_product_offset=(ring * 4 + head) * 4 * 128,
                            )
                        al.sync_block_set("cube", "vector", 4 + 6 * (tile % 2))
                else:
                    pass
            else:
                pass
        tle.dsa.tile_wait_flag(FIX, M, 0)
        tl.debug_barrier()


@triton.jit
def launch_packed_tle(
    Query,
    Key,
    Value,
    Table,
    Used,
    QueryScale,
    KeyScale,
    ValueScale,
    Output,
    Workspace,
    Cuq,
    BUNDLE_KEY: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    META: tl.constexpr,
):
    tl.static_assert(MODE == 6 and N == 512)
    packed_cube(Query, Key, Value, Table, Used, Workspace, Cuq, META)
    vector_packed(
        QueryScale, KeyScale, ValueScale, Output, Workspace, Used, Cuq, N, META
    )


@triton.jit
def launch_replay_tle(
    Query,
    Key,
    Value,
    Table,
    Used,
    QueryScale,
    KeyScale,
    ValueScale,
    Output,
    Workspace,
    Cuq,
    WorkspaceI32,
    BUNDLE_KEY: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    META: tl.constexpr,
):
    tl.static_assert(MODE == 3 or MODE == 4 or MODE == 5)
    Q: tl.constexpr = META[14]
    tl.static_assert(Q == 32 or Q == 64)
    tl.static_assert(N == 256 or N == 512)
    C: tl.constexpr = 256
    L1_KEY: tl.constexpr = Q * 128
    L1_VALUE: tl.constexpr = L1_KEY + N * 128
    L1_PROB: tl.constexpr = L1_VALUE + N * 128
    L0_PROB: tl.constexpr = Q * 128
    P_ROWS: tl.constexpr = 2 * Q + 16
    EXTRA: tl.constexpr = Q if N == 512 else 0
    tl.static_assert(L1_PROB + (P_ROWS + EXTRA) * N <= 512 * 1024)
    tl.static_assert(L0_PROB + (P_ROWS + EXTRA) * C <= 64 * 1024)
    tl.static_assert((P_ROWS + EXTRA) * 128 * 4 <= 128 * 1024)
    core = tl.program_id(0).to(tl.uint32)
    with al.scope(core_mode="cube"):
        HQ: tl.constexpr = META[0]
        HK: tl.constexpr = META[1]
        TS: tl.constexpr = META[4]
        PS: tl.constexpr = META[5]
        RS: tl.constexpr = META[6]
        BLOCKS: tl.constexpr = META[9]
        GROUPS: tl.constexpr = META[10]
        QLEN: tl.constexpr = META[15]
        QGROUPS: tl.constexpr = META[16]
        CAUSAL: tl.constexpr = META[18]
        Flags = WorkspaceI32 + META[38]
        SCORE = Workspace.to(tl.uint64)
        PROB = SCORE + META[30]
        PRODUCT = PROB + META[31]
        EXTRA_P = PRODUCT + META[32]
        EXTRA_PRODUCT = EXTRA_P + META[36]
        al.custom("cube_set_l0c_copy_params", 1, 0, 0)
        tle.dsa.tile_set_flag(FIX, M, 0)
        for ordinal in range(tl.cdiv(GROUPS, BLOCKS)):
            group = core + ordinal * BLOCKS
            if group < GROUPS:
                batch = group // (QGROUPS * HQ)
                head = group % HQ
                kvhead = head // (HQ // HK)
                qstart = (group // HQ % QGROUPS * Q).to(tl.int32)
                if MODE == 4:
                    query_base = tl.load(Cuq + batch).to(tl.uint32)
                    query_length = (
                        tl.load(Cuq + batch + 1).to(tl.uint32) - query_base
                    ).to(tl.int32)
                    active_group = (query_length > 4) & (qstart < query_length)
                else:
                    query_base = batch * QLEN
                    query_length = QLEN
                    active_group = True
                real_queries = tl.minimum(Q, query_length - qstart)
                nk = tl.load(Used + batch)
                if MODE == 5:
                    twice_visible = 2 * nk.to(tl.int64) - (QLEN - 1 if CAUSAL else 0)
                    active_group = (twice_visible >= (6144 if CAUSAL else 4096)) == (
                        N == 512
                    )
                fast_group = (batch * META[39] + qstart // 128) * HQ + head
                replay = tl.load(Flags + fast_group * 2, volatile=True) | tl.load(
                    Flags + fast_group * 2 + 1, volatile=True
                )
                active_group = active_group & (replay != 0)
                if active_group:
                    if CAUSAL:
                        visible = tl.minimum(
                            nk, tl.maximum(0, nk - query_length + qstart + real_queries)
                        )
                    else:
                        visible = nk
                    tiles = tl.cdiv(visible, N)
                    if tiles > 0:
                        qp = Query.to(tl.uint64) + (
                            ((query_base + qstart.to(tl.uint32)) * HQ + head) * 128
                        ).to(tl.uint64)
                        al.custom(
                            "cube_nd2nz_i8",
                            0,
                            qp,
                            1,
                            real_queries,
                            128,
                            0,
                            HQ * 128,
                            Q,
                            1,
                            0,
                        )
                        local_sync(MTE2, MTE1)
                        load_matrix_a(
                            0,
                            0,
                            Q,
                            128,
                            128,
                            0,
                            1,
                            1,
                            load_rows=(real_queries + 15) & -16,
                        )
                        local_sync(MTE1, MTE2)
                        local_sync(MTE1, M)
                        compute_score(
                            Key,
                            Table,
                            SCORE,
                            core,
                            0,
                            batch,
                            kvhead,
                            nk,
                            real_queries,
                            L1_KEY,
                            TS,
                            PS,
                            RS,
                            N,
                            C,
                            Q,
                        )
                        for tile in range(tiles):
                            if tile + 1 < tiles:
                                compute_score(
                                    Key,
                                    Table,
                                    SCORE,
                                    core,
                                    tile + 1,
                                    batch,
                                    kvhead,
                                    nk,
                                    real_queries,
                                    L1_KEY,
                                    TS,
                                    PS,
                                    RS,
                                    N,
                                    C,
                                    Q,
                                )
                            copy_paged_tile(
                                Value,
                                Table,
                                batch,
                                kvhead,
                                tile,
                                nk,
                                L1_VALUE,
                                TS,
                                PS,
                                RS,
                                N,
                                C,
                            )
                            al.sync_block_wait(
                                "vector",
                                "cube",
                                3 + 6 * (tile % 2),
                                sender_pipe=al.PIPE.PIPE_MTE3,
                                receiver_pipe=al.PIPE.PIPE_S,
                            )
                            ring = core * 2 + tile % 2
                            probability_product(
                                PROB,
                                PRODUCT,
                                ring * P_ROWS * N,
                                ring * (2 * Q + 1) * 128,
                                L1_PROB,
                                L0_PROB,
                                P_ROWS,
                                2 * Q + 1,
                                N,
                                C,
                                L1_VALUE,
                                True,
                                active_keys=tl.minimum(N, nk - tile * N),
                                EXTRA_PLANES=1,
                                EXTRA_ROWS=EXTRA,
                                EXTRA_STORE_ROWS=EXTRA,
                                EXTRA_PROB=EXTRA_P if N == 512 else None,
                                EXTRA_PRODUCT=EXTRA_PRODUCT if N == 512 else None,
                                extra_prob_offset=ring * Q * N,
                                extra_product_offset=ring * Q * 128,
                            )
                            al.sync_block_set("cube", "vector", 4 + 6 * (tile % 2))
        tle.dsa.tile_wait_flag(FIX, M, 0)
        tl.debug_barrier()
    vector_replay(
        QueryScale,
        KeyScale,
        ValueScale,
        Output,
        WorkspaceI32,
        WorkspaceI32,
        Used,
        Cuq,
        N,
        MODE,
        META,
    )


LOG2E = tl.constexpr(1.4426950408889634)


LN2 = tl.constexpr(0.6931471805599453)


PROB_QUANT_LEVELS = tl.constexpr(255)


DESCALE_BLOCK = tl.constexpr(128)


BLOCK_M = tl.constexpr(64)


BLOCK_N = tl.constexpr(64)


LAUNCH_CHUNK = 2048


@triton.jit
def _flash_int8_fwd(
    Q,
    K,
    V,
    O,
    LSE,
    PART_OUT,
    PART_MAX,
    PART_DENOM,
    CUQ,
    CUK,
    USED,
    TABLE,
    QS,
    KS,
    VS,
    ALIBI,
    task_begin,
    batch_size,
    heads,
    group,
    max_blocks,
    sq,
    hq,
    sk,
    hk,
    pk,
    sv,
    hv,
    pv,
    so,
    ho,
    qs0,
    qs1,
    qs2,
    ks0,
    ks1,
    ks2,
    vs0,
    vs1,
    vs2,
    table_stride,
    alibi_batch_stride,
    alibi_head_stride,
    total_q,
    scale,
    causal: tl.constexpr,
    left: tl.constexpr,
    right: tl.constexpr,
    softcap: tl.constexpr,
    has_alibi: tl.constexpr,
    write_lse: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    SPLITS: tl.constexpr,
):
    task = task_begin + tl.program_id(0)
    # heads is the KV head count. Query heads that share it occupy the M tile.
    tile_part = task // (batch_size * heads)
    part = tile_part % SPLITS
    tile = tile_part // SPLITS
    rem = task % (batch_size * heads)
    batch = rem // heads
    kv_head = rem % heads
    lane = tl.arange(0, BLOCK_M)
    rows_per_tile = BLOCK_M // group
    q_row = tile * rows_per_tile + lane // group
    q_head = kv_head * group + lane % group
    q_start_i32 = tl.load(CUQ + batch)
    q_end_i32 = tl.load(CUQ + batch + 1)
    q_start = q_start_i32.to(tl.int64)
    nq = q_end_i32 - q_start_i32
    if PAGED:
        k_start = tl.full((), 0, tl.int64)
        nk = tl.load(USED + batch)
    else:
        k_start_i32 = tl.load(CUK + batch)
        k_end_i32 = tl.load(CUK + batch + 1)
        k_start = k_start_i32.to(tl.int64)
        nk = k_end_i32 - k_start_i32
    # Unused lanes appear when group does not divide the M tile.
    row_ok = (lane < rows_per_tile * group) & (q_row < nq)
    rows_safe = tl.where(row_ok, q_row, 0).to(tl.int64)
    d = tl.arange(0, D)
    if tile * rows_per_tile < nq:
        q = tl.load(
            Q + (q_start + rows_safe[:, None]) * sq + q_head[:, None] * hq + d[None, :],
            row_ok[:, None],
            0,
        )
        q_scale = tl.load(
            QS + batch * qs0 + q_head * qs1 + (q_row // DESCALE_BLOCK) * qs2,
            row_ok,
            0,
        )
        maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        denom = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        slope = tl.zeros((BLOCK_M,), dtype=tl.float32)
        if has_alibi:
            slope = tl.load(
                ALIBI + batch * alibi_batch_stride + q_head * alibi_head_stride
            )
        first = 0
        end = nk
        if left >= 0:
            first = tl.maximum(0, tile * rows_per_tile + nk - nq - left) // BLOCK_N
        if causal:
            end = tl.minimum(end, (tile + 1) * rows_per_tile + nk - nq)
        if right >= 0:
            end = tl.minimum(end, (tile + 1) * rows_per_tile + nk - nq + right)
        part_blocks = (max_blocks + SPLITS - 1) // SPLITS
        for start in tl.range(part * part_blocks, (part + 1) * part_blocks):
            if (start >= first) & (start < max_blocks) & (start * BLOCK_N < end):
                n = start * BLOCK_N + tl.arange(0, BLOCK_N)
                n_ok = n < nk
                n_safe = tl.where(n_ok, n, 0).to(tl.int64)
                if PAGED:
                    k_row = virtual_to_cache_offset(
                        n_safe,
                        nk,
                        TABLE + batch * table_stride,
                        PAGE,
                        sk,
                        pk,
                        boundary_check=True,
                    )
                    v_row = virtual_to_cache_offset(
                        n_safe,
                        nk,
                        TABLE + batch * table_stride,
                        PAGE,
                        sv,
                        pv,
                        boundary_check=True,
                    )
                else:
                    k_row = (k_start + n_safe) * sk
                    v_row = (k_start + n_safe) * sv
                if PAGED and PAGE == BLOCK_N:
                    page_id = tl.load(
                        TABLE + batch * table_stride + start,
                        start * PAGE < nk,
                        0,
                    ).to(tl.int64)
                    k_page = tl.make_block_ptr(
                        base=K + page_id * pk + kv_head * hk,
                        shape=(PAGE, D),
                        strides=(sk, 1),
                        offsets=(0, 0),
                        block_shape=(BLOCK_N, D),
                        order=(1, 0),
                    )
                    k = tl.trans(tl.load(k_page))
                elif PAGED and PAGE == 16:
                    k_rows = tl.full((BLOCK_N, D), 0, tl.int8)
                    first_page = start * (BLOCK_N // PAGE)
                    for page_in_tile in tl.static_range(0, BLOCK_N // PAGE):
                        page_number = first_page + page_in_tile
                        physical_page = tl.load(
                            TABLE + batch * table_stride + page_number,
                            page_number < (nk + PAGE - 1) // PAGE,
                            0,
                        ).to(tl.int64)
                        k_page = tl.make_block_ptr(
                            base=K + physical_page * pk + kv_head * hk,
                            shape=(PAGE, D),
                            strides=(sk, 1),
                            offsets=(0, 0),
                            block_shape=(PAGE, D),
                            order=(1, 0),
                        )
                        k_rows = tle.dsa.insert_slice(
                            k_rows,
                            tl.load(k_page),
                            offsets=[page_in_tile * PAGE, 0],
                            sizes=[PAGE, D],
                            strides=[1, 1],
                        )
                    k = tl.trans(k_rows)
                elif not PAGED and (start + 1) * BLOCK_N <= nk:
                    k_block_ptr = tl.make_block_ptr(
                        base=K + k_start * sk + kv_head * hk,
                        shape=(nk, D),
                        strides=(sk, 1),
                        offsets=(start * BLOCK_N, 0),
                        block_shape=(BLOCK_N, D),
                        order=(1, 0),
                    )
                    k = tl.trans(tl.load(k_block_ptr))
                else:
                    k = tl.load(
                        K + k_row[None, :] + kv_head * hk + d[:, None],
                        n_ok[None, :],
                        0,
                    )
                descale_block = start * BLOCK_N // DESCALE_BLOCK
                ks = tl.load(KS + batch * ks0 + kv_head * ks1 + descale_block * ks2)
                vs = tl.load(VS + batch * vs0 + kv_head * vs1 + descale_block * vs2)
                scores = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
                scores = scores * (q_scale * ks * scale)[:, None]
                if softcap > 0:
                    scores = softcap * apply_softcap(
                        scores, 1.0 / softcap, is_softcap=True
                    )
                else:
                    scores = apply_softcap(scores, 1.0, is_softcap=False)
                scores = apply_alibi(
                    scores,
                    n,
                    q_row,
                    nq,
                    nk,
                    is_causal=False,
                    is_alibi=has_alibi,
                    alibi_slope=slope[:, None],
                )
                scores = apply_mask(
                    scores,
                    n,
                    q_row,
                    nq,
                    nk,
                    nk if left < 0 else left,
                    (
                        (0 if causal else nq)
                        if right < 0
                        else (tl.minimum(right, 0) if causal else right)
                    ),
                    is_even_mn=False,
                    is_causal=causal,
                    is_local=left >= 0 or right >= 0,
                )
                scores = tl.where(row_ok[:, None], scores * LOG2E, float("-inf"))
                tile_max = tl.max(scores, 1)
                acc, p, maximum, denom = softmax_rescale(
                    acc,
                    scores,
                    maximum,
                    denom,
                    softmax_scale_log2e=1.0,
                    is_border=True,
                    use_tile_max=True,
                )
                safe_max = tl.where(maximum == float("-inf"), 0, maximum)
                beta = tl.exp2(tile_max - safe_max)
                p_scale = beta * (1.0 / PROB_QUANT_LEVELS)
                p_int8 = (tl.floor(p * PROB_QUANT_LEVELS + 0.5) - 128).to(tl.int8)
                v = tl.load(
                    V + v_row[:, None] + kv_head * hv + d[None, :],
                    n_ok[:, None],
                    0,
                )
                # p_int8 = level - 128, and masked V lanes are zero, so
                # 128 * sum(V) restores the nonnegative levels.
                partial = tl.dot(p_int8, v, out_dtype=tl.int32).to(tl.float32)
                partial += 128.0 * tl.sum(v.to(tl.int32), 0).to(tl.float32)
                acc = acc + partial * (p_scale * vs)[:, None]
        if SPLITS > 1:
            part_row = task * BLOCK_M + lane
            tl.store(PART_OUT + part_row[:, None] * D + d[None, :], acc)
            tl.store(PART_MAX + part_row, maximum)
            tl.store(PART_DENOM + part_row, denom)
        else:
            result = acc / tl.where(denom > 0, denom, 1)[:, None]
            tl.store(
                O
                + (q_start + rows_safe[:, None]) * so
                + q_head[:, None] * ho
                + d[None, :],
                result.to(OUT_DTYPE),
                row_ok[:, None],
            )
            if write_lse:
                lse = tl.where(denom > 0, maximum * LN2 + tl.log(denom), float("inf"))
                tl.store(LSE + q_head * total_q + q_start + rows_safe, lse, row_ok)


@triton.jit
def _flash_int8_merge(
    O,
    LSE,
    PART_OUT,
    PART_MAX,
    PART_DENOM,
    CUQ,
    batch_size,
    heads,
    group,
    total_q,
    so,
    ho,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLITS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    task = tl.program_id(0)
    tile = task // (batch_size * heads)
    rem = task % (batch_size * heads)
    batch = rem // heads
    kv_head = rem % heads
    lane = tl.arange(0, BLOCK_M)
    d = tl.arange(0, D)
    q_start = tl.load(CUQ + batch).to(tl.int64)
    nq = tl.load(CUQ + batch + 1) - q_start
    q_row = tile * (BLOCK_M // group) + lane // group
    q_head = kv_head * group + lane % group
    row_ok = (lane < (BLOCK_M // group) * group) & (q_row < nq)
    rows_safe = tl.where(row_ok, q_row, 0).to(tl.int64)
    if tile * (BLOCK_M // group) < nq:
        maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        denom = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)
        for part in tl.static_range(0, SPLITS):
            part_row = (
                (tile * SPLITS + part) * batch_size * heads + rem
            ) * BLOCK_M + lane
            part_m = tl.load(PART_MAX + part_row)
            part_d = tl.load(PART_DENOM + part_row)
            part_a = tl.load(PART_OUT + part_row[:, None] * D + d[None, :])
            new_max = tl.maximum(maximum, part_m)
            safe_max = tl.where(new_max == float("-inf"), 0, new_max)
            alpha = tl.exp2(maximum - safe_max)
            beta = tl.exp2(part_m - safe_max)
            denom = denom * alpha + part_d * beta
            acc = acc * alpha[:, None] + part_a * beta[:, None]
            maximum = new_max
        result = acc / tl.where(denom > 0, denom, 1)[:, None]
        tl.store(
            O + (q_start + rows_safe[:, None]) * so + q_head[:, None] * ho + d[None, :],
            result.to(OUT_DTYPE),
            row_ok[:, None],
        )
        if WRITE_LSE:
            lse = tl.where(denom > 0, maximum * LN2 + tl.log(denom), float("inf"))
            tl.store(LSE + q_head * total_q + q_start + rows_safe, lse, row_ok)


def _check_descale(name, scale, batch, nheads, length, device):
    if scale is None or scale.dtype != torch.float32 or scale.ndim != 3:
        raise TypeError(f"{name} must be a rank-3 float32 tensor")
    if scale.device != device:
        raise ValueError(f"{name} must be on the same device as Q")
    if scale.shape[0] != batch or scale.shape[1] != nheads:
        raise ValueError(f"{name} must have shape [batch, heads, blocks]")
    if length > 0:
        needed = (length + DESCALE_BLOCK.value - 1) // DESCALE_BLOCK.value
        if scale.stride(-1) != 0 and scale.shape[-1] < needed:
            raise ValueError(f"{name} covers {scale.shape[-1]} blocks, need {needed}")


def flash_attn_varlen_func_w8a8_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    q_v: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[Tuple[int, int]] = None,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    scheduler_metadata: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
    fa_version: int = 2,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Ascend INT8 variable-length FlashAttention-2, inference forward only.

    Q is [total_q, heads, D] and K/V are packed [total_k, kv_heads, D] or paged
    [pages, page_size, kv_heads, D]. D is 64 or 128. Q/K/V are INT8. FP32
    descales are [batch, heads, ceil(length / 128)] and are indexed by logical
    position. QK is an INT8 dot with INT32 accumulation. Softmax stays in FP32.
    The general path uses 256 probability levels and an INT8 PV dot with a
    -128 zero point. The mixed Cube/Vector path adds a signed INT8 probability
    residual to improve accuracy. Output is BF16 unless out is FP16 or BF16.
    """
    if (
        dropout_p == 0
        and not return_attn_probs
        and q_v is None
        and scheduler_metadata is None
        and s_aux is None
        and num_splits == 0
        and cp_world_size == 1
        and cp_rank == 0
        and cp_tot_seqused_k is None
        and fa_version == 2
        and not return_softmax_lse
        and alibi_slopes is None
        and softcap == 0
        and (
            softmax_scale is None
            or (isinstance(softmax_scale, (int, float)) and softmax_scale == 128**-0.5)
        )
        and (
            window_size is None
            or (
                isinstance(window_size, (tuple, list))
                and len(window_size) == 2
                and all(
                    isinstance(bound, (int, float)) and bound < 0
                    for bound in window_size
                )
            )
        )
    ):
        inline_out = try_run(
            q,
            k,
            v,
            max_seqlen_q,
            cu_seqlens_q,
            max_seqlen_k,
            block_table,
            seqused_k,
            q_descale,
            k_descale,
            v_descale,
            out,
            causal,
        )
        if inline_out is not None:
            return inline_out
    if dropout_p != 0 or return_attn_probs or q_v is not None:
        raise NotImplementedError(
            "Only attention inference without dropout is supported"
        )
    if scheduler_metadata is not None or s_aux is not None or num_splits != 0:
        raise NotImplementedError(
            "Scheduler, auxiliary and split-KV modes are unsupported"
        )
    if cp_world_size != 1 or cp_rank != 0 or cp_tot_seqused_k is not None:
        raise NotImplementedError("Context parallel attention is unsupported")
    if fa_version != 2:
        raise NotImplementedError("Only FA2 is implemented")
    from flaggems_vllm.ops.flash_api import CHECK_DEVICE

    CHECK_DEVICE(q)
    CHECK_DEVICE(k)
    CHECK_DEVICE(v)
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
        raise NotImplementedError("Q/K/V must use INT8")
    if q.ndim != 3:
        raise ValueError("Q must have shape [total_q, heads, D]")
    if q.shape[-1] not in (64, 128):
        raise NotImplementedError("Only head dimensions 64 and 128 are supported")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise NotImplementedError("The last dimension of Q/K/V must be contiguous")
    paged = block_table is not None
    if paged and (k.ndim != 4 or v.ndim != 4):
        raise ValueError("Paged K/V must have shape [pages, page_size, kv_heads, D]")
    if not paged and (k.ndim != 3 or v.ndim != 3):
        raise ValueError("Packed K/V must have shape [total_k, kv_heads, D]")
    if seqused_k is not None and not paged:
        raise NotImplementedError("seqused_k requires paged KV")
    if not paged and cu_seqlens_k is None:
        raise ValueError("Packed KV requires cu_seqlens_k")
    if paged and (seqused_k is None or block_table.ndim != 2):
        raise ValueError("Paged KV requires seqused_k and a rank-2 block_table")

    total, heads, dim = q.shape
    kv_heads = k.shape[-2]
    if kv_heads == 0 or heads % kv_heads != 0:
        raise ValueError("Query heads must be a positive multiple of KV heads")
    batch = cu_seqlens_q.shape[0] - 1
    if batch < 1:
        raise ValueError("cu_seqlens_q must contain at least one sequence")
    group = heads // kv_heads
    page = k.shape[1] if paged else 1
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    elif out.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("out must be FP16 or BF16")
    elif out.shape != q.shape or out.device != q.device or out.stride(-1) != 1:
        raise ValueError(
            "out must match Q shape and device, with a contiguous last dim"
        )
    lse = (
        torch.empty((heads, total), dtype=torch.float32, device=q.device)
        if return_softmax_lse
        else None
    )
    if total == 0:
        return (out, lse) if return_softmax_lse else out
    _check_descale("q_descale", q_descale, batch, heads, max_seqlen_q, q.device)
    _check_descale("k_descale", k_descale, batch, kv_heads, max_seqlen_k, q.device)
    _check_descale("v_descale", v_descale, batch, kv_heads, max_seqlen_k, q.device)

    left, right = (-1, -1) if window_size is None else window_size
    if group > BLOCK_M.value:
        raise NotImplementedError("KV group larger than the M tile is unsupported")
    block_m = min(BLOCK_M.value, triton.next_power_of_2(max(16, group * max_seqlen_q)))
    # Shared attention helpers need additional UB scratch on the general path.
    block_m = min(block_m, 32) if group <= 32 else block_m
    query_tile = block_m // group
    num_q_tiles = triton.cdiv(max_seqlen_q, query_tile)
    num_tasks = num_q_tiles * batch * kv_heads
    # A split is useful for long decode, where the unsplit grid has too few
    # independent query tiles. Larger N only fits the small M decode tile.
    splits = 4 if max_seqlen_q == 1 and max_seqlen_k >= 512 else 1
    block_n = (
        128
        if splits > 1 and paged and page == 16 and block_m == 16 and dim == 128
        else BLOCK_N.value
    )
    block_n = 32 if block_m > 32 else block_n
    max_blocks = triton.cdiv(max_seqlen_k, block_n)
    alibi_batch_stride = 0
    alibi_head_stride = 0
    alibi_ptr = q_descale
    if alibi_slopes is not None:
        alibi_ptr = alibi_slopes
        if alibi_slopes.ndim == 1:
            alibi_head_stride = alibi_slopes.stride(0)
        else:
            alibi_batch_stride = alibi_slopes.stride(0)
            alibi_head_stride = alibi_slopes.stride(1)
    lse_ptr = (
        lse if lse is not None else torch.empty(1, dtype=torch.float32, device=q.device)
    )
    out_dtype = tl.bfloat16 if out.dtype == torch.bfloat16 else tl.float16
    scale = dim**-0.5 if softmax_scale is None else softmax_scale
    k_token_stride = k.stride(-3)
    v_token_stride = v.stride(-3)
    k_page_stride = k.stride(0) if paged else 0
    v_page_stride = v.stride(0) if paged else 0
    # Split long decode sequences across cores, then merge online-softmax states.
    part_out = (
        torch.empty(
            (num_tasks * splits * block_m, dim), dtype=torch.float32, device=q.device
        )
        if splits > 1
        else out
    )
    part_max = (
        torch.empty(num_tasks * splits * block_m, dtype=torch.float32, device=q.device)
        if splits > 1
        else out
    )
    part_denom = (
        torch.empty(num_tasks * splits * block_m, dtype=torch.float32, device=q.device)
        if splits > 1
        else out
    )
    with torch_device_fn.device(q.device):
        task_begin = 0
        while task_begin < num_tasks * splits:
            count = min(LAUNCH_CHUNK, num_tasks * splits - task_begin)
            _flash_int8_fwd[(count,)](
                q,
                k,
                v,
                out,
                lse_ptr,
                part_out,
                part_max,
                part_denom,
                cu_seqlens_q,
                cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q,
                seqused_k if seqused_k is not None else cu_seqlens_q,
                block_table if block_table is not None else cu_seqlens_q,
                q_descale,
                k_descale,
                v_descale,
                alibi_ptr,
                task_begin,
                batch,
                kv_heads,
                group,
                max_blocks,
                q.stride(0),
                q.stride(1),
                k_token_stride,
                k.stride(-2),
                k_page_stride,
                v_token_stride,
                v.stride(-2),
                v_page_stride,
                out.stride(0),
                out.stride(1),
                q_descale.stride(0),
                q_descale.stride(1),
                q_descale.stride(2),
                k_descale.stride(0),
                k_descale.stride(1),
                k_descale.stride(2),
                v_descale.stride(0),
                v_descale.stride(1),
                v_descale.stride(2),
                block_table.stride(0) if paged else 0,
                alibi_batch_stride,
                alibi_head_stride,
                total,
                scale,
                causal,
                left,
                right,
                softcap,
                alibi_slopes is not None,
                return_softmax_lse,
                D=dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                PAGE=page,
                PAGED=paged,
                OUT_DTYPE=out_dtype,
                SPLITS=splits,
            )
            task_begin += count
        if splits > 1:
            _flash_int8_merge[(num_tasks,)](
                out,
                lse_ptr,
                part_out,
                part_max,
                part_denom,
                cu_seqlens_q,
                batch,
                kv_heads,
                group,
                total,
                out.stride(0),
                out.stride(1),
                D=dim,
                BLOCK_M=block_m,
                SPLITS=splits,
                OUT_DTYPE=out_dtype,
                WRITE_LSE=return_softmax_lse,
            )
    return (out, lse) if return_softmax_lse else out


_HEAD_CAUSAL_MIN_CONTEXT = 3072


_HEAD_UNMASKED_MIN_CONTEXT = 2048


_PACKED_DECODE_MIN_GROUPS = 16


_PACKED_DECODE_MIN_KV = 1024


TILING_FIELDS = (
    "heads",
    "kvHeads",
    "tasks",
    "vectorBlocks",
    "tableStride",
    "pageStride",
    "rowStride",
    "ksBlocks",
    "vsBlocks",
    "cubeBlocks",
    "totalGroups",
    "groupSize",
    "headGroupSize",
    "groupSplits",
    "queryLen",
    "originalQueryLen",
    "queryGroups",
    "hasCuq",
    "causal",
    "qsBlocks",
    "qsStrideB",
    "qsStrideH",
    "qsStrideBlock",
    "ksStrideB",
    "ksStrideH",
    "ksStrideBlock",
    "vsStrideB",
    "vsStrideH",
    "vsStrideBlock",
    "nTiles",
    "scoreBytes",
    "probBytes",
    "pvBytes",
    "alphaBytes",
    "vNzBytes",
    "separateCorrection",
    "corrProbBytes",
    "corrPvBytes",
)


@lru_cache(maxsize=None)
def _bitcode_key(path):
    # File contents, not only the Python registration, affect generated code.
    # The bundle is immutable for the lifetime of the process.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def _supported_device(device):
    # The mixed core mapping have been validated for this physical SKU.
    with torch_device_fn.device(device):
        return "910B4" in torch_device_fn.get_device_name(device)


def _scale_supported(scale, batch, heads, length, device):
    if not isinstance(scale, torch.Tensor):
        return False
    if (
        scale.dtype != torch.float32
        or scale.ndim != 3
        or scale.device != device
        or scale.shape[:2] != (batch, heads)
    ):
        return False
    if length > 0 and (
        scale.shape[2] == 0
        or (scale.stride(2) != 0 and scale.shape[2] < triton.cdiv(length, 128))
    ):
        return False
    # The device tiling uses uint32 strides and offsets. Non-contiguous and
    # broadcast scale views are supported, but their reachable offset must fit.
    if any(stride < 0 or stride >= 2**32 for stride in scale.stride()):
        return False
    return (
        sum(
            max(size - 1, 0) * stride
            for size, stride in zip(scale.shape, scale.stride())
        )
        < 2**32
    )


def _eligible(q, k, v, cuq, table, used, qs, ks, vs, out, maxq, maxk):
    if not all(isinstance(x, torch.Tensor) for x in (q, k, v, cuq, table, used)):
        return False
    if (
        q.device.type != "npu"
        or q.ndim != 3
        or q.shape[2] != 128
        or k.ndim != 4
        or k.shape[1] != 16
        or k.shape[3] != 128
        or v.shape != k.shape
        or table.ndim != 2
        or q.shape[0] == 0
    ):
        return False
    if (
        not isinstance(maxq, int)
        or not isinstance(maxk, int)
        or not 0 < maxq < 2**32
        or not 0 <= maxk < 2**32
    ):
        return False
    batch = table.shape[0]
    if (
        batch < 1
        or cuq.shape != (batch + 1,)
        or used.shape != (batch,)
        or table.shape[1] < triton.cdiv(maxk, 16)
        or q.shape[0] > batch * maxq
    ):
        return False
    heads, kv_heads = q.shape[1], k.shape[2]
    if heads < 1 or kv_heads < 1 or heads % kv_heads:
        return False
    single_head = q.shape[0] == batch * maxq and maxq >= 128
    # Grouped kernels physically pack four Q heads per KV head. The long
    # uniform single-head kernels instead map queryHead // group on device.
    if heads // kv_heads != 4 and not single_head:
        return False
    # C220 Nd2NzParams.srcDValue is uint16. Grouped Q copies individual rows
    # with source stride D; the single-head path copies across token rows.
    if k.stride(1) >= 2**16 or (single_head and q.stride(0) >= 2**16):
        return False
    for tensor in (q, k, v):
        if (
            tensor.dtype != torch.int8
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            return False
    for tensor in (cuq, table, used):
        if (
            tensor.dtype != torch.int32
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            return False
    if max(x.numel() for x in (q, k, v, cuq, table, used)) >= 2**32:
        return False
    if out is not None and (
        not isinstance(out, torch.Tensor)
        or out.dtype != torch.bfloat16
        or out.shape != q.shape
        or out.device != q.device
        or not out.is_contiguous()
    ):
        return False
    for scale, heads, length in (
        (qs, heads, maxq),
        (ks, kv_heads, maxk),
        (vs, kv_heads, maxk),
    ):
        if not _scale_supported(scale, batch, heads, length, q.device):
            return False
    return True


def _metadata(
    fields,
    q,
    k,
    table,
    qs,
    ks,
    vs,
    maxq,
    causal,
    width,
    tile,
    head=False,
    small=False,
    packed=False,
):
    batch = table.shape[0]
    rows = tile if head else tile * 4
    pv_rows = 2 * rows + 1
    pv_m = triton.cdiv(pv_rows, 16) * 16
    query_groups = 2 if small else triton.cdiv(maxq, tile)
    heads, kv_heads = q.shape[1], k.shape[2]
    groups = batch * query_groups * (heads if head else kv_heads)
    blocks = min(20, groups)
    values = dict.fromkeys(fields, 0)
    values.update(
        heads=heads,
        kvHeads=kv_heads,
        tasks=batch * query_groups * tile * heads,
        hasCuq=1,
        vectorBlocks=2 * blocks,
        tableStride=table.shape[1],
        pageStride=k.stride(0),
        rowStride=k.stride(1),
        cubeBlocks=blocks,
        totalGroups=groups,
        groupSize=4,
        headGroupSize=4,
        groupSplits=1,
        queryLen=tile,
        originalQueryLen=maxq,
        queryGroups=query_groups,
        causal=int(causal),
        qsBlocks=qs.shape[2],
        ksBlocks=ks.shape[2],
        vsBlocks=vs.shape[2],
        qsStrideB=qs.stride(0),
        qsStrideH=qs.stride(1),
        qsStrideBlock=qs.stride(2),
        ksStrideB=ks.stride(0),
        ksStrideH=ks.stride(1),
        ksStrideBlock=ks.stride(2),
        vsStrideB=vs.stride(0),
        vsStrideH=vs.stride(1),
        vsStrideBlock=vs.stride(2),
        nTiles=triton.cdiv(table.shape[1], 8),
        scoreBytes=blocks * 2 * rows * width * 4,
        probBytes=blocks * 2 * pv_m * width,
        pvBytes=blocks * 2 * pv_rows * 128 * 4,
    )
    if head:
        # Q128 uses separate high/low PV operations; smaller grouped tiles
        # concatenate both planes and their correction row in one PV matrix.
        high_m = triton.cdiv(rows + 1, 16) * 16
        values["probBytes"] = blocks * 2 * (high_m + rows) * width
        values["pvBytes"] = blocks * 2 * (2 * rows + 1) * 128 * 4
    if width >= 512:
        extra_rows = triton.cdiv(rows // 2, 16) * 16
        values["corrProbBytes"] = blocks * 2 * 2 * extra_rows * width
        values["corrPvBytes"] = blocks * 2 * rows * 128 * 4
    if packed:
        groups = batch * (kv_heads // 4)
        blocks = min(20, groups)
        values.update(
            totalGroups=groups,
            cubeBlocks=blocks,
            vectorBlocks=2 * blocks,
            headGroupSize=16,
            scoreBytes=blocks * 2 * 16 * width * 4,
            probBytes=blocks * 2 * 64 * width,
            pvBytes=blocks * 2 * 36 * 128 * 4,
            corrProbBytes=blocks * 2 * 4 * 16 * width,
            corrPvBytes=blocks * 2 * 4 * 4 * 128 * 4,
        )
    if any(not 0 <= value < 2**32 for value in values.values()):
        return None
    workspace_bytes = (
        values["scoreBytes"]
        + values["probBytes"]
        + values["pvBytes"]
        + values["corrProbBytes"]
        + values["corrPvBytes"]
    )
    workspace_bytes += blocks * 2 * 64 if width >= 512 else 0
    return tuple(values[name] for name in fields), blocks, workspace_bytes


def try_run(q, k, v, maxq, cuq, maxk, table, used, qs, ks, vs, out, causal):
    """Return None for unsupported scope, otherwise return the public output.

    CUQ/used/table *contents* follow the same validity contract as the public
    operator; this path never copies sequence metadata to the CPU to inspect it.
    """
    if not _eligible(q, k, v, cuq, table, used, qs, ks, vs, out, maxq, maxk):
        return None
    if not _supported_device(q.device):
        return None
    width = 256 if vs.stride(2) == 0 or vs.shape[2] == 1 else 128
    uniform = q.shape[0] == table.shape[0] * maxq
    minimum_wide_k = 256 if maxq <= 4 else 512
    width = (
        (1024 if maxk >= 2048 else 512)
        if uniform and maxq <= 8 and maxk > minimum_wide_k and width == 256
        else width
    )
    estimated_visible_keys = maxk - (maxq - 1) / 2 if causal else maxk
    if (
        uniform
        and maxq >= 128
        and width == 256
        and estimated_visible_keys
        >= (_HEAD_CAUSAL_MIN_CONTEXT if causal else _HEAD_UNMASKED_MIN_CONTEXT)
    ):
        width = 512
    # Q128 tails can regress small mixed batches (for example Q=[129, 5]).
    # Restrict the larger tile to long-prefill mixtures validated in benchmarks.
    mixed_head = not uniform and maxq >= 1024 and width == 256
    # Unlike grouped copies, single-head Nd2Nz uses the inter-token Q stride.
    if mixed_head and q.stride(0) >= 2**16:
        return None
    head = (uniform and maxq >= 128) or mixed_head
    partitioned_head = head and uniform and width == 512 and table.shape[0] > 1
    packed_decode = (
        uniform
        and maxq == 1
        and maxk >= _PACKED_DECODE_MIN_KV
        and q.shape[1] == 4 * k.shape[2]
        and k.shape[2] % 4 == 0
        and table.shape[0] * (k.shape[2] // 4) >= _PACKED_DECODE_MIN_GROUPS
        and (vs.stride(2) == 0 or vs.shape[2] == 1)
    )
    if packed_decode:
        width = 512
    if head:
        tile = 128
    elif width >= 512 and not packed_decode:
        tile = 4
    else:
        tile = min(16, triton.next_power_of_2(maxq))
    prepared = _metadata(
        TILING_FIELDS,
        q,
        k,
        table,
        qs,
        ks,
        vs,
        maxq,
        causal,
        width,
        tile,
        head=head,
        packed=packed_decode,
    )
    if prepared is None:
        return None
    metadata, blocks, workspace_bytes = prepared
    mixed = mixed_head or (not uniform and tile == 16 and width == 256)
    small = None
    if mixed or partitioned_head:
        small = _metadata(
            TILING_FIELDS,
            q,
            k,
            table,
            qs,
            ks,
            vs,
            maxq,
            causal,
            256 if partitioned_head else width,
            128 if partitioned_head else 2,
            head=partitioned_head,
            small=not partitioned_head,
        )
        if small is None:
            return None
        workspace_bytes = max(workspace_bytes, small[2])
    if head and width == 512:
        flag_offset = triton.cdiv(workspace_bytes, 64) * 16
        metadata = (*metadata, flag_offset)
        workspace_bytes = flag_offset * 4 + metadata[10] * 2 * 4
    from triton.experimental.tle.language.dsa.ascend.custom_ops.registry import (
        CUSTOM_OPS_BITCODE,
    )

    ordinary_head = head and not (partitioned_head or mixed)
    cube_bundle_key = (_bitcode_key(CUSTOM_OPS_BITCODE),)
    small_bundle_key = cube_bundle_key
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    args = (q, k, v, table, used, qs, ks, vs, out, workspace.view(torch.int32), cuq)

    def launch_cube(launch_blocks, launch_width, mode, launch_metadata):
        if launch_width == 128:
            # Uniform queries are required by the N128 Vector path.
            launch_args = (*args[:9], workspace.view(torch.int32), None)
        else:
            launch_args = args
        workspace_i32 = workspace.view(torch.int32)
        if mode == 0:
            launch_kernel = launch_grouped_tle
            launch_args = (
                *args[:9],
                workspace.view(torch.int32),
                cuq,
                cube_bundle_key,
                launch_width,
                mode,
                launch_metadata,
            )
        elif mode == 6:
            launch_kernel = launch_packed_tle
            launch_args = (
                *args[:9],
                workspace.view(torch.int32),
                cuq,
                cube_bundle_key,
                launch_width,
                mode,
                launch_metadata,
            )
        elif mode == 2:
            launch_kernel = launch_hybrid_large_tle
            launch_args = (*args, cube_bundle_key, launch_width, mode, launch_metadata)
        elif mode == 1:
            launch_kernel = launch_hybrid_small_tle
            launch_args = (
                *args[:9],
                workspace.view(torch.int32),
                cuq,
                small_bundle_key,
                launch_width,
                mode,
                launch_metadata,
            )
        else:
            launch_kernel = launch_cube_tle
            launch_args = (
                *launch_args,
                workspace_i32,
                cube_bundle_key,
                launch_width,
                mode,
                launch_metadata,
            )
        launch_kernel[(launch_blocks,)](
            *launch_args,
            disable_auto_inject_block_sync=True,
            multibuffer=False,
            enable_legacy_insert_load_store_for_mix_cv=mode == 1,
            enable_auto_bind_sub_block=True,
            disable_fma=launch_width == 128 or mode == 6,
            enable_ubuf_saving=(
                (
                    mode in (0, 2, 3, 4, 5, 6)
                    and launch_width != 128
                    and not (mode == 0 and launch_width == 1024)
                )
                or (
                    launch_width == 128
                    and not (
                        launch_metadata[15] % 128 == 0
                        and launch_metadata[10] <= launch_blocks
                    )
                )
            ),
        )

        if mode in (3, 4, 5) and launch_width == 512:

            replay_metadata = list(launch_metadata[:38])
            replay_metadata[14] = 32
            replay_metadata[16] = triton.cdiv(replay_metadata[15], 32)
            replay_metadata[10] = (
                replay_metadata[10] // launch_metadata[16] * replay_metadata[16]
            )
            replay_metadata[9] = min(20, replay_metadata[10])
            replay_metadata[3] = 2 * replay_metadata[9]
            replay_metadata[2] = replay_metadata[10] * 32
            rings = 2 * replay_metadata[9]
            replay_metadata[30] = rings * 32 * 512 * 4
            replay_metadata[31] = rings * 80 * 512
            replay_metadata[32] = rings * 65 * 128 * 4
            replay_metadata[36] = rings * 32 * 512
            replay_metadata[37] = rings * 32 * 128 * 4
            replay_metadata = (
                *replay_metadata,
                launch_metadata[38],
                launch_metadata[16],
            )
            launch_replay_tle[(replay_metadata[9],)](
                *launch_args[:-1],
                replay_metadata,
                disable_auto_inject_block_sync=True,
                multibuffer=False,
                enable_legacy_insert_load_store_for_mix_cv=False,
                enable_auto_bind_sub_block=True,
                disable_fma=False,
                enable_ubuf_saving=True,
            )

    with torch_device_fn.device(q.device):
        if partitioned_head:
            small_metadata, small_blocks, _ = small
            launch_cube(small_blocks, 256, 5, small_metadata)
            launch_cube(blocks, 512, 5, metadata)
        elif mixed:
            small_metadata, small_blocks, _ = small
            launch_cube(small_blocks, width, 1, small_metadata)
            if mixed_head:
                launch_cube(blocks, width, 4, metadata)
            else:
                launch_cube(blocks, width, 2, metadata)
        elif ordinary_head:
            launch_cube(blocks, width, 3, metadata)
        else:
            if packed_decode:
                launch_cube(blocks, width, 6, metadata)
            else:
                launch_cube(blocks, width, 0, metadata)
    return out
