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


from dataclasses import replace
from functools import wraps
from typing import Optional, Tuple, Union

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl

from flaggems_vllm.ops.attention import (
    flash_attn_varlen_func as upstream_flash_attn_varlen_func,
)
from flaggems_vllm.ops.flash_kernel import (
    apply_alibi,
    apply_mask,
    apply_softcap,
    virtual_to_cache_offset,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._ascend.fused.attention_decode import (
    decode_attention,
)
from flaggems_vllm.runtime.backend._ascend.fused.attention_native import (
    AttentionConfig,
    native_tle_attention,
)
from flaggems_vllm.runtime.backend._ascend.fused.attention_prefill import (
    prefill_attention,
)

LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)
DESCALE_BLOCK = tl.constexpr(128)
BLOCK_M = tl.constexpr(64)
BLOCK_N = tl.constexpr(128)
LAUNCH_CHUNK = 2048


@triton.jit
def build_worklist(
    Cuq,
    Used,
    QueryBatch,
    QueryTile,
    KeyBatch,
    KeyPage,
    Counts,
    B: tl.constexpr,
    Q_ROWS: tl.constexpr,
    N: tl.constexpr,
    PAGE: tl.constexpr,
    BB: tl.constexpr,
):
    sequences = tl.arange(0, BB)
    begin = tl.load(Cuq + sequences, sequences < B, 0)
    end = tl.load(Cuq + sequences + 1, sequences < B, 0)
    query_counts = tl.cdiv(end - begin, Q_ROWS)
    key_lengths = tl.load(Used + sequences, sequences < B, 0)
    key_counts = tl.cdiv(key_lengths, N) * (N // PAGE)
    if tl.program_id(0) == 0:
        tl.store(Counts, tl.sum(query_counts, 0))
        tl.store(Counts + 1, tl.sum(key_counts, 0))
    else:
        pass
    for sequence in range(tl.program_id(0), B, 40):
        query_offset = tl.sum(tl.where(sequences < sequence, query_counts, 0), 0)
        key_offset = tl.sum(tl.where(sequences < sequence, key_counts, 0), 0)
        query_count = tl.cdiv(
            tl.load(Cuq + sequence + 1) - tl.load(Cuq + sequence), Q_ROWS
        )
        key_count = tl.cdiv(tl.load(Used + sequence), N) * (N // PAGE)
        for start in range(0, query_count, 256):
            index = start + tl.arange(0, 256)
            tl.store(QueryBatch + query_offset + index, sequence, index < query_count)
            tl.store(QueryTile + query_offset + index, index, index < query_count)
        for start in range(0, key_count, 256):
            index = start + tl.arange(0, 256)
            tl.store(KeyBatch + key_offset + index, sequence, index < key_count)
            tl.store(KeyPage + key_offset + index, index, index < key_count)


@triton.jit
def pack_queries(
    Q,
    Packed,
    Cuq,
    QueryBatch,
    QueryTile,
    Counts,
    Q_ROWS: tl.constexpr,
    batch: tl.constexpr,
    heads: tl.constexpr,
    capacity: tl.constexpr,
    group: tl.constexpr,
    sq: tl.constexpr,
    D: tl.constexpr,
    HAS_WORKLIST: tl.constexpr,
    UNIFORM_Q_LENGTH: tl.constexpr = 0,
):
    BLOCK: tl.constexpr = 16 if Q_ROWS >= 16 else Q_ROWS
    tiles: tl.constexpr = capacity // 16
    if HAS_WORKLIST:
        count = tl.load(Counts) * heads
    else:
        count = batch * heads * tiles
    for task in range(tl.program_id(0), count, 40):
        if HAS_WORKLIST:
            head = task % heads
            sequence = tl.load(QueryBatch + task // heads)
            tile = tl.load(QueryTile + task // heads)
        else:
            head = task // tiles % heads
            sequence = task // (tiles * heads)
            tile = task % tiles
        for piece in tl.static_range(0, Q_ROWS, BLOCK):
            row = tile * Q_ROWS + piece
            if UNIFORM_Q_LENGTH > 0:
                begin = (sequence * UNIFORM_Q_LENGTH).to(tl.int64)
            else:
                begin = tl.load(Cuq + sequence).to(tl.int64)
            if UNIFORM_Q_LENGTH > 0:
                length = UNIFORM_Q_LENGTH
            else:
                length = tl.load(Cuq + sequence + 1) - begin
            source = tl.make_block_ptr(
                Q + begin * sq + head * group * D,
                (length, group * D),
                (sq, 1),
                (row, 0),
                (BLOCK, group * D),
                (1, 0),
            )
            target = tl.make_block_ptr(
                Packed + (sequence * heads + head) * capacity * group * D,
                (capacity, group * D),
                (group * D, 1),
                (row, 0),
                (BLOCK, group * D),
                (1, 0),
            )
            tl.store(
                target,
                (
                    tl.load(source)
                    if UNIFORM_Q_LENGTH > 0 and UNIFORM_Q_LENGTH == capacity
                    else tl.load(source, boundary_check=(0,), padding_option="zero")
                ),
                boundary_check=(0,),
            )


@triton.jit
def unpack_output(
    Packed,
    PackedLSE,
    Output,
    LSE,
    Cuq,
    QueryBatch,
    QueryTile,
    Counts,
    Q_ROWS: tl.constexpr,
    B: tl.constexpr,
    HK: tl.constexpr,
    CAP: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    STRIDE: tl.constexpr,
    TOTAL: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    HAS_WORKLIST: tl.constexpr,
    UNIFORM_Q_LENGTH: tl.constexpr = 0,
):
    BLOCK: tl.constexpr = 16 if Q_ROWS >= 16 else Q_ROWS
    tiles: tl.constexpr = CAP // 16
    if HAS_WORKLIST:
        count = tl.load(Counts) * HK
    else:
        count = B * HK * tiles
    for task in range(tl.program_id(0), count, 40):
        if HAS_WORKLIST:
            head = task % HK
            batch = tl.load(QueryBatch + task // HK)
            tile = tl.load(QueryTile + task // HK)
        else:
            head = task // tiles % HK
            batch = task // (tiles * HK)
            tile = task % tiles
        for piece in tl.static_range(0, Q_ROWS, BLOCK):
            row = tile * Q_ROWS + piece
            if UNIFORM_Q_LENGTH > 0:
                begin = batch * UNIFORM_Q_LENGTH
            else:
                begin = tl.load(Cuq + batch)
            if UNIFORM_Q_LENGTH > 0:
                length = UNIFORM_Q_LENGTH
            else:
                length = tl.load(Cuq + batch + 1) - begin
            source = tl.make_block_ptr(
                Packed + (batch * HK + head) * CAP * G * D,
                (CAP, G * D),
                (G * D, 1),
                (row, 0),
                (BLOCK, G * D),
                (1, 0),
            )
            target = tl.make_block_ptr(
                Output + begin.to(tl.int64) * STRIDE + head * G * D,
                (length, G * D),
                (STRIDE, 1),
                (row, 0),
                (BLOCK, G * D),
                (1, 0),
            )
            if (
                UNIFORM_Q_LENGTH > 0
                and UNIFORM_Q_LENGTH == CAP
                or row + BLOCK <= length
            ):
                tl.store(target, tl.load(source))
            else:
                tl.store(target, tl.load(source), boundary_check=(0,))
            if WRITE_LSE:
                source_lse = tl.make_block_ptr(
                    PackedLSE + (batch * HK + head) * CAP * G,
                    (CAP, G),
                    (G, 1),
                    (row, 0),
                    (BLOCK, G),
                    (1, 0),
                )
                target_lse = tl.make_block_ptr(
                    LSE + head * G * TOTAL + begin.to(tl.int64),
                    (G, length),
                    (TOTAL, 1),
                    (0, row),
                    (G, BLOCK),
                    (1, 0),
                )
                tl.store(target_lse, tl.trans(tl.load(source_lse)), boundary_check=(1,))
            else:
                pass


@triton.jit
def pack_key_value(
    K,
    V,
    PackedK,
    PackedV,
    Table,
    Used,
    KeyBatch,
    KeyPage,
    Counts,
    batch: tl.constexpr,
    heads: tl.constexpr,
    capacity: tl.constexpr,
    page: tl.constexpr,
    table_stride: tl.constexpr,
    sk: tl.constexpr,
    pk: tl.constexpr,
    sv: tl.constexpr,
    pv: tl.constexpr,
    D: tl.constexpr,
    HAS_WORKLIST: tl.constexpr,
    KEY_MAJOR: tl.constexpr = False,
):
    pages: tl.constexpr = capacity // page
    width: tl.constexpr = heads * D
    if HAS_WORKLIST:
        count = tl.load(Counts + 1)
    else:
        count = batch * pages
    for task in range(tl.program_id(0), count, 40):
        if HAS_WORKLIST:
            logical_page = tl.load(KeyPage + task)
            sequence = tl.load(KeyBatch + task)
        else:
            logical_page = task % pages
            sequence = task // pages
        length = tl.load(Used + sequence)
        physical_page = tl.load(
            Table + sequence * table_stride + logical_page,
            logical_page * page < length,
            0,
        ).to(tl.int64)
        key = tl.make_block_ptr(
            K + physical_page * pk,
            (page, width),
            (sk, 1),
            (0, 0),
            (page, width),
            (1, 0),
        )
        value = tl.make_block_ptr(
            V + physical_page * pv,
            (page, heads, D),
            (sv, D, 1),
            (0, 0, 0),
            (page, heads, D),
            (2, 1, 0),
        )
        key_target = tl.make_block_ptr(
            PackedK + sequence * capacity * width,
            (capacity, width),
            (width, 1),
            (logical_page * page, 0),
            (page, width),
            (1, 0),
        )
        value_target = tl.make_block_ptr(
            PackedV + sequence * heads * capacity * D,
            (heads, capacity, D),
            (capacity * D, D, 1),
            (0, logical_page * page, 0),
            (heads, page, D),
            (2, 1, 0),
        )
        if KEY_MAJOR:
            key3 = tl.make_block_ptr(
                K + physical_page * pk,
                (page, heads, D),
                (sk, D, 1),
                (0, 0, 0),
                (page, heads, D),
                (2, 1, 0),
            )
            values3 = tl.load(key3)
            target3 = tl.make_block_ptr(
                PackedK + sequence * heads * capacity * D,
                (heads, capacity, D),
                (capacity * D, D, 1),
                (0, logical_page * page, 0),
                (heads, page, D),
                (2, 1, 0),
            )
            tl.store(target3, tl.permute(values3, (1, 0, 2)))
        else:
            tl.store(key_target, tl.load(key))
        tl.store(value_target, tl.permute(tl.load(value).to(tl.float16), (1, 0, 2)))


@triton.jit
def pack_inputs(
    Q,
    K,
    V,
    Query,
    Key,
    Value,
    Cuq,
    Used,
    Table,
    QBatch,
    QTile,
    Counts,
    KBatch,
    KPage,
    Q_ROWS: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    QCAP: tl.constexpr,
    G: tl.constexpr,
    KCAP: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    PK: tl.constexpr,
    SV: tl.constexpr,
    PV: tl.constexpr,
    D: tl.constexpr,
    HAS_WORKLIST: tl.constexpr,
    CACHE_REPEAT: tl.constexpr = 1,
    UNIFORM_Q_LENGTH: tl.constexpr = 0,
    UnsafeCount=None,
    INIT_UNSAFE: tl.constexpr = False,
    DIRECT_QUERY: tl.constexpr = False,
    KEY_MAJOR: tl.constexpr = False,
):
    if INIT_UNSAFE:
        if tl.program_id(0) == 0:
            tl.store(UnsafeCount, 0)
        else:
            pass
    else:
        pass
    if not DIRECT_QUERY:
        pack_queries(
            Q,
            Query,
            Cuq,
            QBatch,
            QTile,
            Counts,
            Q_ROWS,
            B,
            H,
            QCAP,
            G,
            SQ,
            D,
            HAS_WORKLIST,
            UNIFORM_Q_LENGTH,
        )
    else:
        pass
    pack_key_value(
        K,
        V,
        Key,
        Value,
        Table,
        Used,
        KBatch,
        KPage,
        Counts,
        B,
        H // CACHE_REPEAT,
        KCAP,
        PAGE,
        TABLE_STRIDE,
        SK,
        PK,
        SV,
        PV,
        D,
        HAS_WORKLIST,
        KEY_MAJOR,
    )


@triton.jit
def flash_int8_fwd(
    Q,
    K,
    V,
    Output,
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
    group: tl.constexpr,
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
    PACKED_NATIVE: tl.constexpr,
    CAPACITY: tl.constexpr,
    CAPACITY_Q: tl.constexpr,
):
    task = task_begin + tl.program_id(0)
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
    if PAGED or PACKED_NATIVE:
        k_start = tl.full((), 0, tl.int64)
        nk = tl.load(USED + batch)
    else:
        k_start_i32 = tl.load(CUK + batch)
        k_end_i32 = tl.load(CUK + batch + 1)
        k_start = k_start_i32.to(tl.int64)
        nk = k_end_i32 - k_start_i32
    row_ok = (lane < rows_per_tile * group) & (q_row < nq)
    rows_safe = tl.where(row_ok, q_row, 0).to(tl.int64)
    d = tl.arange(0, D)
    if tile * rows_per_tile < nq:
        if PACKED_NATIVE:
            query_ptr = tl.make_block_ptr(
                Q + (batch * heads + kv_head) * CAPACITY_Q * group * D,
                (CAPACITY_Q * group, D),
                (D, 1),
                (tile * BLOCK_M, 0),
                (BLOCK_M, D),
                (1, 0),
            )
            q = tl.load(query_ptr)
        else:
            q = tl.load(
                Q
                + (q_start + rows_safe[:, None]) * sq
                + q_head[:, None] * hq
                + d[None, :],
                row_ok[:, None],
                0,
            )
        q_scale = tl.load(
            QS + batch * qs0 + q_head * qs1 + q_row // DESCALE_BLOCK * qs2, row_ok, 0
        )
        maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        denom = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        slope = tl.zeros((BLOCK_M,), dtype=tl.float32)
        if has_alibi:
            slope = tl.load(
                ALIBI + batch * alibi_batch_stride + q_head * alibi_head_stride
            )
        else:
            pass
        first = 0
        end = nk
        if left >= 0:
            first = tl.maximum(0, tile * rows_per_tile + nk - nq - left) // BLOCK_N
        else:
            pass
        if causal:
            end = tl.minimum(end, (tile + 1) * rows_per_tile + nk - nq)
        else:
            pass
        if right >= 0:
            end = tl.minimum(end, (tile + 1) * rows_per_tile + nk - nq + right)
        else:
            pass
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
                if PACKED_NATIVE:
                    packed_base = (batch * CAPACITY * heads + kv_head) * D
                    key_ptr = tl.make_block_ptr(
                        K + packed_base,
                        (CAPACITY, D),
                        (heads * D, 1),
                        (start * BLOCK_N, 0),
                        (BLOCK_N, D),
                        (1, 0),
                    )
                    k = tl.trans(tl.load(key_ptr))
                elif PAGED and PAGE == BLOCK_N:
                    page_id = tl.load(
                        TABLE + batch * table_stride + start, start * PAGE < nk, 0
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
                        K + k_row[None, :] + kv_head * hk + d[:, None], n_ok[None, :], 0
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
                        else tl.minimum(right, 0) if causal else right
                    ),
                    is_even_mn=False,
                    is_causal=causal,
                    is_local=left >= 0 or right >= 0,
                )
                scores = tl.where(row_ok[:, None], scores * LOG2E, float("-inf"))
                tile_maximum = tl.max(scores, 1)
                difference = maximum - tile_maximum
                alpha = tl.where(
                    maximum == float("-inf"), 0, tl.exp2(tl.minimum(difference, 0))
                )
                beta = tl.where(
                    tile_maximum == float("-inf"),
                    0,
                    tl.exp2(tl.minimum(-difference, 0)),
                )
                safe_tile_maximum = tl.where(
                    tile_maximum == float("-inf"), 0, tile_maximum
                )
                p = tl.exp2(scores - safe_tile_maximum[:, None])
                denom = denom * alpha + tl.sum(p, 1) * beta
                if PACKED_NATIVE:
                    value_ptr = tl.make_block_ptr(
                        V + (batch * heads + kv_head) * CAPACITY * D,
                        (CAPACITY, D),
                        (D, 1),
                        (start * BLOCK_N, 0),
                        (BLOCK_N, D),
                        (1, 0),
                    )
                    v = tl.load(value_ptr)
                else:
                    v = tl.load(
                        V + v_row[:, None] + kv_head * hv + d[None, :], n_ok[:, None], 0
                    ).to(tl.float16)
                probability = p.to(tl.float16)
                partial = tl.dot(probability, v, out_dtype=tl.float32)
                if vs > 0.25:
                    residual = (p - probability.to(tl.float32)).to(tl.float16)
                    partial += tl.dot(residual, v, out_dtype=tl.float32)
                else:
                    pass
                acc = acc * alpha[:, None] + partial * (vs * beta)[:, None]
                maximum = tl.maximum(maximum, tile_maximum)
            else:
                pass
        if SPLITS > 1:
            part_row = task * BLOCK_M + lane
            tl.store(PART_OUT + part_row[:, None] * D + d[None, :], acc)
            tl.store(PART_MAX + part_row, maximum)
            tl.store(PART_DENOM + part_row, denom)
        else:
            result = acc / tl.where(denom > 0, denom, 1)[:, None]
            if PACKED_NATIVE:
                output_ptr = tl.make_block_ptr(
                    Output + q_start * so + kv_head * group * D,
                    (nq, group * D),
                    (so, 1),
                    (tile * rows_per_tile, 0),
                    (BLOCK_M // group, group * D),
                    (1, 0),
                )
                tl.store(
                    output_ptr,
                    result.reshape((BLOCK_M // group, group * D)).to(OUT_DTYPE),
                    boundary_check=(0,),
                )
            else:
                tl.store(
                    Output
                    + (q_start + rows_safe[:, None]) * so
                    + q_head[:, None] * ho
                    + d[None, :],
                    result.to(OUT_DTYPE),
                    row_ok[:, None],
                )
            if write_lse:
                lse = tl.where(denom > 0, maximum * LN2 + tl.log(denom), float("inf"))
                tl.store(LSE + q_head * total_q + q_start + rows_safe, lse, row_ok)
            else:
                pass
    else:
        pass


@triton.jit
def flash_int8_merge(
    Output,
    LSE,
    PART_OUT,
    PART_MAX,
    PART_DENOM,
    CUQ,
    batch_size,
    heads,
    group: tl.constexpr,
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
    row_ok = (lane < BLOCK_M // group * group) & (q_row < nq)
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
            Output
            + (q_start + rows_safe[:, None]) * so
            + q_head[:, None] * ho
            + d[None, :],
            result.to(OUT_DTYPE),
            row_ok[:, None],
        )
        if WRITE_LSE:
            lse = tl.where(denom > 0, maximum * LN2 + tl.log(denom), float("inf"))
            tl.store(LSE + q_head * total_q + q_start + rows_safe, lse, row_ok)
        else:
            pass
    else:
        pass


def check_descale(
    name: str,
    scale: Optional[torch.Tensor],
    batch: int,
    nheads: int,
    length: int,
    device: torch.device,
) -> None:
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
    else:
        pass


@wraps(upstream_flash_attn_varlen_func, assigned=(), updated=())
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
    """Ascend INT8 FA2 with native Triton QK, FP32 softmax and FP16 PV."""
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
    if seqused_k is not None and (not paged):
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
    check_descale("q_descale", q_descale, batch, heads, max_seqlen_q, q.device)
    check_descale("k_descale", k_descale, batch, kv_heads, max_seqlen_k, q.device)
    check_descale("v_descale", v_descale, batch, kv_heads, max_seqlen_k, q.device)
    left, right = (-1, -1) if window_size is None else window_size
    packed_native = (
        paged
        and max_seqlen_k > 0
        and (page == 16)
        and (group <= 16)
        and (group & group - 1 == 0)
        and (q.stride(-2) == dim)
        and (k.stride(-2) == dim)
        and (v.stride(-2) == dim)
        and (out.stride(-2) == dim)
    )
    is_prefill_fast = (
        packed_native
        and max_seqlen_q > 8
        and (q_descale.stride(2) == 0 or max_seqlen_q <= 128)
        and (k_descale.stride(2) == 0 or max_seqlen_k <= 128)
        and (v_descale.stride(2) == 0 or max_seqlen_k <= 128)
        and (softcap == 0)
        and (alibi_slopes is None)
        and ((left, right) == (-1, -1))
    )
    uniform_q_length = (
        max_seqlen_q if is_prefill_fast and total == batch * max_seqlen_q else 0
    )
    is_query_head_layout = (
        is_prefill_fast
        and uniform_q_length >= 512
        and (max_seqlen_k >= 512)
        and (uniform_q_length & uniform_q_length - 1 == 0)
        and (max_seqlen_k & max_seqlen_k - 1 == 0)
        and causal
        and (dim == 128)
        and (not return_softmax_lse)
    )
    use_direct_output = (
        is_query_head_layout
        and (not return_softmax_lse)
        and (dim == 128)
        and (uniform_q_length >= 512)
        and (uniform_q_length % 128 == 0)
    )
    kv_head_repeat = group if is_query_head_layout else 1
    if is_query_head_layout:
        kv_heads = heads
        group = 1
    else:
        pass
    tile_n = (
        256
        if packed_native and k_descale.stride(2) == 0 and (v_descale.stride(2) == 0)
        else 128
    )
    capacity = triton.cdiv(max_seqlen_k, tile_n) * tile_n
    use_tle = packed_native and max_seqlen_q > 0
    tle_rows = (
        128
        if is_prefill_fast
        else (
            256
            if max_seqlen_q > 8
            else max(16, 2 * group, triton.next_power_of_2(group * max_seqlen_q))
        )
    )
    query_alignment = max(16, tle_rows // group) if use_tle else 16
    query_capacity = triton.cdiv(max_seqlen_q, query_alignment) * query_alignment
    if group > BLOCK_M.value:
        raise NotImplementedError("KV group larger than the M tile is unsupported")
    block_m = min(BLOCK_M.value, triton.next_power_of_2(max(16, group * max_seqlen_q)))
    block_m = min(block_m, 32) if group <= 32 else block_m
    query_tile = block_m // group
    num_q_tiles = triton.cdiv(max_seqlen_q, query_tile)
    num_tasks = num_q_tiles * batch * kv_heads
    splits = 4 if not use_tle and max_seqlen_q == 1 and (max_seqlen_k >= 512) else 1
    block_n = 128 if packed_native else BLOCK_N.value if block_m <= 32 else 32
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
    else:
        pass
    lse_ptr = (
        lse if lse is not None else torch.empty(1, dtype=torch.float32, device=q.device)
    )
    out_dtype = tl.bfloat16 if out.dtype == torch.bfloat16 else tl.float16
    scale = dim ** (-0.5) if softmax_scale is None else softmax_scale
    k_token_stride = k.stride(-3)
    v_token_stride = v.stride(-3)
    k_page_stride = k.stride(0) if paged else 0
    v_page_stride = v.stride(0) if paged else 0
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
    decode_cap = triton.next_power_of_2(max(1, max_seqlen_q))
    decode_pack = 4 if decode_cap == 1 else 2 if decode_cap <= 4 else 1
    is_decode_fast = (
        packed_native
        and 0 < max_seqlen_q <= 8
        and (group == 4)
        and (kv_heads % decode_pack == 0)
        and (k_token_stride == kv_heads * dim)
        and (v_token_stride == kv_heads * dim)
        and (k_page_stride == page * kv_heads * dim)
        and (v_page_stride == page * kv_heads * dim)
    )
    with torch_device_fn.device(q.device):
        if is_decode_fast:
            decode_attention(
                q,
                k,
                v,
                out,
                lse_ptr,
                cu_seqlens_q,
                seqused_k,
                block_table,
                q_descale,
                k_descale,
                v_descale,
                max_seqlen_q,
                max_seqlen_k,
                scale,
                causal,
                left,
                right,
                softcap,
                alibi_slopes,
                return_softmax_lse,
            )
            return (out, lse) if return_softmax_lse else out
        else:
            pass
        if packed_native:
            query = (
                torch.empty(
                    (batch, kv_heads, query_capacity, group, dim),
                    dtype=torch.int8,
                    device=q.device,
                )
                if not is_query_head_layout
                else q
            )
            key_major_enabled = (
                use_direct_output
                and batch == 1
                and (group == 1)
                and (kv_heads == 32)
                and (kv_head_repeat == 4)
                and (query_capacity == 2048)
            )
            key = torch.empty(
                (
                    (batch, kv_heads // kv_head_repeat, capacity, dim)
                    if key_major_enabled
                    else (batch, capacity, kv_heads // kv_head_repeat, dim)
                ),
                dtype=torch.int8,
                device=q.device,
            )
            value = torch.empty(
                (batch, kv_heads // kv_head_repeat, capacity, dim),
                dtype=torch.float16,
                device=q.device,
            )
            has_worklist = (
                batch > 1 and batch <= 128 and (total * 2 < batch * max_seqlen_q)
            )
            query_batch = query_tile = work_counts = key_batch = key_page = cu_seqlens_q
            if has_worklist:
                work_capacity = triton.cdiv(total, tle_rows // group) + batch
                query_batch = torch.empty(
                    work_capacity, dtype=torch.int32, device=q.device
                )
                query_tile = torch.empty(
                    work_capacity, dtype=torch.int32, device=q.device
                )
                key_batch = torch.empty(
                    batch * (capacity // page), dtype=torch.int32, device=q.device
                )
                key_page = torch.empty(
                    batch * (capacity // page), dtype=torch.int32, device=q.device
                )
                work_counts = torch.empty(2, dtype=torch.int32, device=q.device)
                build_worklist[40,](
                    cu_seqlens_q,
                    seqused_k,
                    query_batch,
                    query_tile,
                    key_batch,
                    key_page,
                    work_counts,
                    batch,
                    tle_rows // group,
                    tile_n,
                    page,
                    triton.next_power_of_2(batch),
                )
            else:
                pass
            unsafe_count = (
                torch.empty(1, dtype=torch.int32, device=q.device)
                if is_prefill_fast
                else cu_seqlens_q
            )
            pack_inputs[40,](
                q,
                k,
                v,
                query,
                key,
                value,
                cu_seqlens_q,
                seqused_k,
                block_table,
                query_batch,
                query_tile,
                work_counts,
                key_batch,
                key_page,
                tle_rows // group if has_worklist else 16,
                batch,
                kv_heads,
                query_capacity,
                group,
                capacity,
                page,
                block_table.stride(0),
                q.stride(0),
                k_token_stride,
                k_page_stride,
                v_token_stride,
                v_page_stride,
                dim,
                has_worklist,
                kv_head_repeat,
                uniform_q_length,
                UnsafeCount=unsafe_count,
                INIT_UNSAFE=is_prefill_fast,
                DIRECT_QUERY=is_query_head_layout,
                KEY_MAJOR=key_major_enabled,
            )
        else:
            query, key, value = (q, k, v)
        if use_tle:
            query_tiles = triton.cdiv(max_seqlen_q, tle_rows // group)
            core_count = min(20, query_tiles * batch * kv_heads)
            score = torch.empty(
                (core_count * (2 * tle_rows * tile_n),),
                dtype=torch.int32,
                device=q.device,
            )
            probability = torch.empty(
                (core_count * (4 * tle_rows * tile_n),),
                dtype=torch.float16,
                device=q.device,
            )
            product = torch.empty(
                (
                    (core_count * tle_rows * dim,)
                    if is_prefill_fast
                    else (core_count, 2, tle_rows, dim)
                ),
                dtype=torch.float32,
                device=q.device,
            )
            task_offset, tile_shift = (
                (
                    (72, 14)
                    if query_tiles == 16
                    else (80, 8) if query_tiles == 32 else (0, 0)
                )
                if is_query_head_layout and batch == 1 and (kv_heads == 32)
                else (0, 0)
            )
            configuration = AttentionConfig(
                batch=batch,
                kv_heads=kv_heads,
                group=group,
                dim=dim,
                query_capacity=query_capacity,
                key_capacity=capacity,
                query_tiles=query_tiles,
                total_query=total,
                output_stride=out.stride(0),
                q_scale_strides=q_descale.stride(),
                k_scale_strides=k_descale.stride(),
                v_scale_strides=v_descale.stride(),
                causal=causal,
                write_lse=return_softmax_lse,
                scale=scale,
                block_m=tle_rows,
                block_n=tile_n,
                cores=core_count,
                left=left,
                right=right,
                softcap=softcap,
                has_alibi=alibi_slopes is not None,
                alibi_batch_stride=alibi_batch_stride,
                alibi_head_stride=alibi_head_stride,
                has_worklist=has_worklist,
                direct_output=use_direct_output,
                kv_head_repeat=kv_head_repeat,
                uniform_q_length=uniform_q_length,
                vector_elements=16384 if use_direct_output else 8192,
                query_stride=q.stride(0) if is_query_head_layout else 0,
                task_offset=task_offset,
                tile_shift=tile_shift,
                key_head_major=key_major_enabled,
            )
            native_output = (
                out
                if use_direct_output
                else torch.empty(
                    (batch, kv_heads, query_capacity * group, dim),
                    dtype=out.dtype,
                    device=q.device,
                )
            )
            native_lse = (
                lse_ptr
                if use_direct_output
                else (
                    torch.empty(
                        (batch, kv_heads, query_capacity * group),
                        dtype=torch.float32,
                        device=q.device,
                    )
                    if return_softmax_lse
                    else lse_ptr
                )
            )
            native_arguments = (
                query,
                key,
                value,
                native_output,
                native_lse,
                cu_seqlens_q,
                seqused_k,
                query_batch,
                query_tile,
                work_counts,
                q_descale,
                k_descale,
                v_descale,
                alibi_ptr,
                score,
                probability,
                product,
            )
            launch_options = dict(
                enable_reduce_sum_strength=True,
                multibuffer=False,
                enable_auto_bind_sub_block=False,
                disable_auto_inject_block_sync=True,
            )
            if key_major_enabled:
                launch_options["disable_auto_cv_work_space_manage"] = True
            if is_prefill_fast:
                flags = torch.empty(
                    (query_tiles * batch * kv_heads, 256),
                    dtype=torch.int32,
                    device=q.device,
                )
                prefill_attention[core_count,](
                    *native_arguments,
                    flags,
                    configuration,
                    **launch_options,
                    UnsafeCount=unsafe_count,
                )
                fallback_configuration = replace(
                    configuration, block_n=128, filter_flags=True, aggregate_flags=True
                )
                fallback_product = torch.empty(
                    (core_count, 2, tle_rows, dim), dtype=torch.float32, device=q.device
                )
                fallback_arguments = (*native_arguments[:-1], fallback_product)
                native_tle_attention[core_count,](
                    *fallback_arguments,
                    fallback_configuration,
                    Flags=flags,
                    **launch_options,
                    UnsafeCount=unsafe_count,
                )
            else:
                native_tle_attention[core_count,](
                    *native_arguments, configuration, **launch_options
                )
            if not use_direct_output:
                unpack_output[40,](
                    native_output,
                    native_lse,
                    out,
                    lse_ptr,
                    cu_seqlens_q,
                    query_batch,
                    query_tile,
                    work_counts,
                    tle_rows // group if has_worklist else 16,
                    batch,
                    kv_heads,
                    query_capacity,
                    group,
                    dim,
                    out.stride(0),
                    total,
                    return_softmax_lse,
                    HAS_WORKLIST=has_worklist,
                    UNIFORM_Q_LENGTH=uniform_q_length,
                )
            else:
                pass
            return (out, lse) if return_softmax_lse else out
        task_begin = 0
        while task_begin < num_tasks * splits:
            count = min(LAUNCH_CHUNK, num_tasks * splits - task_begin)
            flash_int8_fwd[count,](
                query,
                key,
                value,
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
                PACKED_NATIVE=packed_native,
                CAPACITY=capacity,
                CAPACITY_Q=query_capacity,
            )
            task_begin += count
        if splits > 1:
            flash_int8_merge[num_tasks,](
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
        else:
            pass
    return (out, lse) if return_softmax_lse else out


@wraps(upstream_flash_attn_varlen_func)
def flash_attn_varlen_func(q, *args, **kwargs):
    if q.dtype == torch.int8:
        return flash_attn_varlen_func_w8a8_int8(q, *args, **kwargs)
    return upstream_flash_attn_varlen_func(q, *args, **kwargs)
