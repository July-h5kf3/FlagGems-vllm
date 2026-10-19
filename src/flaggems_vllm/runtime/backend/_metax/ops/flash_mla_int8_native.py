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

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.metax import MacaMmaLayout, async_copy, mma

from flaggems_vllm.runtime.backend._metax.ops.flash_mla_int8_gluon import (
    pack_probability_terms,
)


@gluon.jit
def dense_int8_values_prefetch(
    KV,
    Table,
    Lengths,
    Scores,
    CacheMetadata,
    Metadata,
    PS,
    Output,
    LSE,
    Partial,
    PartialD,
    SplitFlag,
    KV_STRIDES: gl.constexpr,
    TS: gl.constexpr,
    LS: gl.constexpr,
    B: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
    CAP: gl.constexpr,
    BH: gl.constexpr = 64,
    SPLITS: gl.constexpr = 1,
    ADAPTIVE: gl.constexpr = False,
    QUANTIZED_CACHE: gl.constexpr = True,
    SORT_BATCHES: gl.constexpr = False,
    UNIFORM_SPLITS: gl.constexpr = 1,
    STAGED: gl.constexpr = False,
):
    gl.static_assert(QUANTIZED_CACHE and KV.dtype.element_ty == gl.int8)
    value_tile, head_tile, job = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    if ADAPTIVE:
        split, batch = job // B, job % B
    else:
        split: gl.constexpr = 0
        batch = job
    if SORT_BATCHES:
        batch = gl.load(SplitFlag + 1 + batch)
    matrix: gl.constexpr = MacaMmaLayout([4, 1], [4, 1, 16])
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 0)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 0)
    copy_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    probability_layout: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])
    tokens = gl.arange(0, 64, layout=gl.SliceLayout(1, copy_layout))
    dims = value_tile * 256 + gl.arange(0, 256, layout=gl.SliceLayout(0, copy_layout))
    p_tokens = gl.arange(0, 64, layout=gl.SliceLayout(1, probability_layout))
    p_heads = head_tile * BH + gl.arange(
        0, BH, layout=gl.SliceLayout(0, probability_layout)
    )
    rows = batch * H * SQ + p_heads
    maximum = gl.load(Metadata + rows * 2, p_heads < H * SQ, 0.0)
    inverse_scale = gl.load(Metadata + rows * 2 + 1, p_heads < H * SQ, 0.0)
    length = gl.load(Lengths + batch * LS)
    if ADAPTIVE:
        split_count = gl.where(gl.load(SplitFlag) != 0, SPLITS, UNIFORM_SPLITS)
    else:
        split_count: gl.constexpr = 1
    if split < split_count:
        high_accumulator = gl.full((256, BH), 0, gl.int32, matrix)
        low_accumulator = gl.full((256, BH), 0, gl.int32, matrix)
        buffer = gl.allocate_shared_memory(
            gl.int8, (64, 256), gl.SwizzledSharedLayout(64, 16, 2, [1, 0])
        )
        probability_buffer = gl.allocate_shared_memory(
            gl.int8, (64, 2 * BH), gl.SwizzledSharedLayout(16, 2, 4, [0, 1])
        )
        high_buffer = probability_buffer.slice(0, BH, dim=1)
        low_buffer = probability_buffer.slice(BH, BH, dim=1)
        probability_words = probability_buffer._reinterpret(
            gl.int32, (16, 2 * BH), gl.SwizzledSharedLayout(4, 2, 4, [0, 1])
        )
        if STAGED:
            qbuffer = probability_buffer._reinterpret(
                gl.uint16, (64, BH), gl.SwizzledSharedLayout(8, 1, 1, [0, 1])
            )
            meta_buffer = gl.allocate_shared_memory(
                gl.float32, (2 * BH, 4), gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
            )
            meta_copy: gl.constexpr = gl.BlockedLayout([1, 2], [32, 2], [4, 1], [1, 0])
            meta_rows = gl.arange(0, 2 * BH, layout=gl.SliceLayout(1, meta_copy))
            meta_cols = gl.arange(0, 4, layout=gl.SliceLayout(0, meta_copy))
            meta_heads = head_tile * BH + meta_rows % BH
            dummy = gl.reshape(
                gl.permute(
                    gl.full((64, BH), 0.0, gl.float32, probability_layout), (1, 0)
                ),
                (BH, 4, 16),
            )
            meta_layout: gl.constexpr = gl.SliceLayout(2, dummy.type.layout)
        pages = gl.cdiv(length, 64)
        begin, end = pages * split // split_count, pages * (split + 1) // split_count
        if begin < end:
            page = gl.load(Table + batch * TS + begin).to(gl.int64)
            async_copy.async_copy_global_to_shared(
                buffer,
                KV
                + page * KV_STRIDES[0]
                + tokens[:, None] * KV_STRIDES[1]
                + dims[None, :],
                None if STAGED else begin * 64 + tokens[:, None] < length,
            )
            if STAGED:
                score_offsets = (
                    (batch.to(gl.int64) * CAP + begin * 64) * H * SQ
                    + p_heads[None, :] * 64
                    + p_tokens[:, None]
                )
                async_copy.async_copy_global_to_shared(
                    qbuffer,
                    Scores + score_offsets,
                    None if (H * SQ) % BH == 0 else p_heads[None, :] < H * SQ,
                )
                meta_offsets = (
                    (
                        batch.to(gl.int64) * (CAP // 64) * 2
                        + begin * 2
                        + meta_rows[:, None] // BH
                    )
                    * H
                    * SQ
                    + meta_heads[:, None]
                ) * 4 + meta_cols[None, :]
                async_copy.async_copy_global_to_shared(
                    meta_buffer,
                    CacheMetadata + meta_offsets,
                    None if (H * SQ) % BH == 0 else meta_heads[:, None] < H * SQ,
                )
            async_copy.commit_group()
        for page_index in range(begin, end):
            if STAGED:
                async_copy.wait_all()
                async_copy.block_barrier()
                local_integer = qbuffer.load(probability_layout)
                local_maximum = meta_buffer.slice(0, BH, dim=0).load(meta_layout)
                local_scale = meta_buffer.slice(BH, BH, dim=0).load(meta_layout)
                expanded_maximum, _ = gl.broadcast(local_maximum[:, :, None], dummy)
                expanded_scale, _ = gl.broadcast(local_scale[:, :, None], dummy)
                local_maximum = gl.convert_layout(
                    gl.permute(gl.reshape(expanded_maximum, (BH, 64)), (1, 0)),
                    probability_layout,
                )
                local_scale = gl.convert_layout(
                    gl.permute(gl.reshape(expanded_scale, (BH, 64)), (1, 0)),
                    probability_layout,
                )
            else:
                score_offsets = (
                    (batch.to(gl.int64) * CAP + page_index * 64) * H * SQ
                    + p_heads[None, :] * 64
                    + p_tokens[:, None]
                )
                local_integer = gl.load(
                    Scores + score_offsets, p_heads[None, :] < H * SQ, 0
                )
                groups = p_tokens // 16
                meta_offsets = (
                    (batch.to(gl.int64) * (CAP // 64) * 2 + page_index * 2) * H * SQ
                    + p_heads[None, :]
                ) * 4 + groups[:, None]
                local_maximum = gl.load(
                    CacheMetadata + meta_offsets, p_heads[None, :] < H * SQ, 0.0
                )
                local_scale = gl.load(
                    CacheMetadata + meta_offsets + 4 * H * SQ,
                    p_heads[None, :] < H * SQ,
                    0.0,
                )
            factor = (
                local_scale
                * gl.exp2(local_maximum - maximum[None, :])
                * inverse_scale[None, :]
            )
            biased = (local_integer.to(gl.uint32) | 0x4B000000).to(
                gl.float32, bitcast=True
            )
            normalized = gl.fma(biased, factor, -8388608.0 * factor)
            integer = (normalized + 8388736.0).to(gl.int32, bitcast=True)
            packed_high, packed_low = pack_probability_terms(integer, FLOAT_BITS=True)
            if STAGED:
                async_copy.block_barrier()
            joined_words = gl.reshape(
                gl.permute(gl.join(packed_high, packed_low), (0, 2, 1)), (16, 2 * BH)
            )
            probability_words.store(joined_words)
            if not STAGED:
                async_copy.wait_all()
            async_copy.block_barrier()
            value = buffer.permute((1, 0)).load(lhs)
            has_next = page_index + 1 < end
            if STAGED:
                next_page = gl.load(
                    Table + batch * TS + page_index + 1, has_next, 0
                ).to(gl.int64)
                high_accumulator = mma(value, high_buffer.load(rhs), high_accumulator)
                high_accumulator = gl.inline_asm_elementwise(
                    "",
                    constraints="=r,0,~{memory}",
                    args=[high_accumulator],
                    dtype=gl.int32,
                    is_pure=False,
                    pack=1,
                )
                low_value = low_buffer.load(rhs)
                async_copy.block_barrier()
                if has_next:
                    # Cached padding probabilities are zero; all INT8 physical rows are safe to read.
                    async_copy.async_copy_global_to_shared(
                        buffer,
                        KV
                        + next_page * KV_STRIDES[0]
                        + tokens[:, None] * KV_STRIDES[1]
                        + dims[None, :],
                    )
                    score_offsets = (
                        (batch.to(gl.int64) * CAP + (page_index + 1) * 64) * H * SQ
                        + p_heads[None, :] * 64
                        + p_tokens[:, None]
                    )
                    async_copy.async_copy_global_to_shared(
                        qbuffer,
                        Scores + score_offsets,
                        None if (H * SQ) % BH == 0 else p_heads[None, :] < H * SQ,
                    )
                    meta_offsets = (
                        (
                            batch.to(gl.int64) * (CAP // 64) * 2
                            + (page_index + 1) * 2
                            + meta_rows[:, None] // BH
                        )
                        * H
                        * SQ
                        + meta_heads[:, None]
                    ) * 4 + meta_cols[None, :]
                    async_copy.async_copy_global_to_shared(
                        meta_buffer,
                        CacheMetadata + meta_offsets,
                        None if (H * SQ) % BH == 0 else meta_heads[:, None] < H * SQ,
                    )
                    async_copy.commit_group()
                low_accumulator = gl.inline_asm_elementwise(
                    "",
                    constraints="=r,0,~{memory}",
                    args=[low_accumulator],
                    dtype=gl.int32,
                    is_pure=False,
                    pack=1,
                )
                low_accumulator = mma(value, low_value, low_accumulator)
            else:
                async_copy.block_barrier()
                if has_next:
                    next_page = gl.load(Table + batch * TS + page_index + 1).to(
                        gl.int64
                    )
                    async_copy.async_copy_global_to_shared(
                        buffer,
                        KV
                        + next_page * KV_STRIDES[0]
                        + tokens[:, None] * KV_STRIDES[1]
                        + dims[None, :],
                        (page_index + 1) * 64 + tokens[:, None] < length,
                    )
                    async_copy.commit_group()
                high_accumulator = mma(value, high_buffer.load(rhs), high_accumulator)
                low_accumulator = mma(value, low_buffer.load(rhs), low_accumulator)
                async_copy.block_barrier()
        heads = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(0, matrix))
        out_dims = value_tile * 256 + gl.arange(
            0, 256, layout=gl.SliceLayout(1, matrix)
        )
        raw = high_accumulator.to(gl.float32) * 256.0 + low_accumulator.to(gl.float32)
        if split_count == 1:
            scale = gl.load(PS + batch * H * SQ + heads, heads < H * SQ, 0.0)
            output = raw * scale[None, :]
            store_layout: gl.constexpr = gl.BlockedLayout(
                [8, 1], [16, 4], [1, 4], [0, 1]
            )
            store_dims = value_tile * 256 + gl.arange(
                0, 256, layout=gl.SliceLayout(1, store_layout)
            )
            store_heads = head_tile * BH + gl.arange(
                0, BH, layout=gl.SliceLayout(0, store_layout)
            )
            output = gl.convert_layout(output.to(Output.dtype.element_ty), store_layout)
            gl.store(
                Output
                + (batch * H * SQ + store_heads[None, :]) * 512
                + store_dims[:, None],
                output,
                store_heads[None, :] < H * SQ,
            )
        else:
            offset = ((split * B + batch) * H * SQ + heads[None, :]).to(
                gl.int64
            ) * 512 + out_dims[:, None]
            gl.store(Partial + offset, raw, heads[None, :] < H * SQ)
