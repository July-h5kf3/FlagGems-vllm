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
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def dense_int8_probability_merge(
    Stats,
    Metadata,
    PS,
    LSE,
    Lengths,
    SplitFlag,
    LS: tl.constexpr,
    B: tl.constexpr,
    LENGTH_BLOCK: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    QUANTIZED_CACHE: tl.constexpr,
    SORT_BATCHES: tl.constexpr = False,
):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    if QUANTIZED_CACHE:
        positions = tl.arange(0, SPLITS * 4)
        row_batches = rows // (H * SQ)
        row_heads = rows % (H * SQ)
        offsets = (
            (
                row_batches[:, None].to(tl.int64) * SPLITS * 3
                + (positions[None, :] // 4) * 3
            )
            * 4
            + positions[None, :] % 4
        ) * H * SQ + row_heads[:, None]
        maxima = tl.load(Stats + offsets, rows[:, None] < ROWS, -float("inf"))
        weighted_maxima = tl.load(
            Stats + offsets + 4 * H * SQ, rows[:, None] < ROWS, 0.0
        )
        local_denominators = tl.load(
            Stats + offsets + 8 * H * SQ, rows[:, None] < ROWS, 0.0
        )
        maximum = tl.max(maxima, 1)
        safe = tl.where(maximum == -float("inf"), 0.0, maximum)
        weights = tl.exp2(maxima - safe[:, None])
        denominator = tl.sum(local_denominators * weights, 1)
        scale = tl.maximum(tl.max(weighted_maxima * weights, 1), 1.0e-30)
        output_scale = scale / tl.where(denominator > 0, denominator, 1.0)
        logsum = tl.where(
            denominator > 0,
            maximum * 0.6931471805599453 + tl.log(denominator),
            float("inf"),
        )
        tl.store(
            LSE + (row_batches * H + row_heads % H) * SQ + row_heads // H,
            logsum,
            rows < ROWS,
        )
    else:
        splits = tl.arange(0, SPLITS)
        offsets = (rows[:, None] * SPLITS + splits[None, :]) * 2
        maxima = tl.load(Stats + offsets, rows[:, None] < ROWS, -float("inf"))
        weighted_maxima = tl.load(Stats + offsets + 1, rows[:, None] < ROWS, 0.0)
        maximum = tl.max(maxima, 1)
        safe = tl.where(maximum == -float("inf"), 0.0, maximum)
        weights = tl.exp2(maxima - safe[:, None])
        weighted = tl.max(weighted_maxima * weights, 1)
        scale = tl.maximum(weighted * (1.0 / 16129.0), 1.0e-30)
        output_scale = scale
    tl.store(Metadata + rows * 2, safe, rows < ROWS)
    tl.store(Metadata + rows * 2 + 1, 1.0 / scale, rows < ROWS)
    tl.store(PS + rows, output_scale, rows < ROWS)
    if tl.program_id(0) == 0:
        if ADAPTIVE or SORT_BATCHES:
            batches = tl.arange(0, LENGTH_BLOCK)
            lengths = tl.load(Lengths + batches * LS, batches < B, 0)
            maximum_length = tl.max(lengths, 0)
            minimum_length = tl.min(tl.where(batches < B, lengths, 2147483647), 0)
        if ADAPTIVE:
            tl.store(SplitFlag, maximum_length != minimum_length)
        else:
            tl.store(SplitFlag, 0)
        if SORT_BATCHES:
            if maximum_length != minimum_length:
                keys = tl.where(batches < B, lengths * LENGTH_BLOCK + batches, -1)
                ordered = tl.sort(keys, descending=True) % LENGTH_BLOCK
            else:
                ordered = batches
            tl.store(SplitFlag + 1 + batches, ordered, batches < B)


@libentry()
@gluon.jit
def dense_int8_value_merge(
    Partial,
    PartialD,
    Metadata,
    Scale,
    Output,
    LSE,
    SplitFlag,
    B: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
    SPLITS: gl.constexpr,
    QUANTIZED_CACHE: gl.constexpr = False,
    MERGE_HEADS: gl.constexpr = 4,
    UNIFORM_SPLITS: gl.constexpr = 1,
):
    if UNIFORM_SPLITS > 1 or gl.load(SplitFlag) != 0:
        parent: gl.constexpr = gl.BlockedLayout(
            [1, 1, 4], [1, 1, 64], [1, 2, 2], [2, 1, 0]
        )
        meta: gl.constexpr = gl.SliceLayout(2, parent)
        outlayout: gl.constexpr = gl.SliceLayout(0, parent)
        batch = gl.program_id(0)
        hs = gl.program_id(1) * MERGE_HEADS
        splits = gl.arange(0, SPLITS, layout=gl.SliceLayout(1, meta))
        heads = hs + gl.arange(0, MERGE_HEADS, layout=gl.SliceLayout(0, meta))
        oh = hs + gl.arange(0, MERGE_HEADS, layout=gl.SliceLayout(1, outlayout))
        dims = gl.arange(0, 512, layout=gl.SliceLayout(0, outlayout))
        s2 = gl.expand_dims(splits, 1)
        h2 = gl.expand_dims(heads, 0)
        s3 = gl.expand_dims(s2, 2)
        h3 = gl.expand_dims(h2, 2)
        d3 = gl.expand_dims(gl.expand_dims(dims, 0), 0)
        # Native PV writes only active partitions.
        split_count = SPLITS
        if UNIFORM_SPLITS > 1 and UNIFORM_SPLITS < SPLITS:
            split_count = gl.where(gl.load(SplitFlag) != 0, SPLITS, UNIFORM_SPLITS)
        values = gl.load(
            Partial + ((s3.to(gl.int64) * B + batch) * H * SQ + h3) * 512 + d3,
            (h3 < H * SQ) & (s3 < split_count),
            0.0,
        )
        scale = gl.load(Scale + batch * H * SQ + oh, oh < H * SQ, 0.0)
        if QUANTIZED_CACHE:
            output = gl.sum(values, 0) * gl.expand_dims(scale, 1)
        else:
            denominator = gl.sum(
                gl.load(
                    PartialD + (s2 * B + batch) * H * SQ + h2,
                    (h2 < H * SQ) & (s2 < split_count),
                    0.0,
                ),
                0,
            )
            denom = gl.convert_layout(
                gl.where(denominator > 0, denominator, 1.0),
                gl.SliceLayout(1, outlayout),
            )
            output = gl.sum(values, 0) * gl.expand_dims(scale / denom, 1)
        gl.store(
            Output
            + (batch.to(gl.int64) * H * SQ + gl.expand_dims(oh, 1)) * 512
            + gl.expand_dims(dims, 0),
            output,
            gl.expand_dims(oh < H * SQ, 1),
        )
        if not QUANTIZED_CACHE:
            maximum = gl.load(Metadata + (batch * H * SQ + oh) * 2, oh < H * SQ, 0.0)
            denominator = gl.convert_layout(denominator, gl.SliceLayout(1, outlayout))
            logsum = gl.where(
                denominator > 0,
                maximum * 0.6931471805599453 + gl.log(denominator),
                float("inf"),
            )
            gl.store(LSE + (batch * H + oh % H) * SQ + oh // H, logsum, oh < H * SQ)


@libentry()
@triton.jit
def dense_int8_batch_order(
    Lengths, Order, LS: tl.constexpr, B: tl.constexpr, BLOCK: tl.constexpr
):
    batches = tl.arange(0, BLOCK)
    lengths = tl.load(Lengths + batches * LS, batches < B, 0)
    maximum = tl.max(lengths, 0)
    minimum = tl.min(tl.where(batches < B, lengths, 2147483647), 0)
    if maximum != minimum:
        keys = tl.where(batches < B, lengths * BLOCK + batches, -1)
        ordered = tl.sort(keys, descending=True) % BLOCK
    else:
        ordered = batches
    tl.store(Order + batches, ordered, batches < B)
