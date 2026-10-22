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


from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.ops.flash_kernel import apply_alibi, apply_softcap
from flaggems_vllm.runtime.backend._ascend.fused.attention_native import (
    AttentionConfig,
    stage_sync,
    ub_zero,
)

PIPE = al.PIPE


@dataclass(frozen=True)
class DecodeConfig(AttentionConfig):
    head_pack: int = 4
    query_cap: int = 1
    probability_digits: int = 3
    query_stride: int = 0
    key_stride: int = 0
    key_page_stride: int = 0
    value_stride: int = 0
    value_page_stride: int = 0
    table_stride: int = 0
    scale_blocks: int = 0
    table_blocks: int = 0


@triton.jit
def pack_queries_diagonal(
    Query,
    Packed,
    Cuq,
    B: tl.constexpr,
    HKG: tl.constexpr,
    D: tl.constexpr,
    SQ: tl.constexpr,
    QC: tl.constexpr,
    PACK: tl.constexpr,
):
    M: tl.constexpr = QC * 4 * PACK
    DP: tl.constexpr = D * PACK
    for task in range(tl.program_id(0), B * HKG, 40):
        batch = task // HKG
        head_group = task % HKG
        begin = tl.load(Cuq + batch).to(tl.int64)
        length = tl.load(Cuq + batch + 1) - begin
        if length > 0:
            if PACK == 1:
                source = tl.make_block_ptr(
                    Query + begin * SQ + head_group * 4 * D,
                    (length, 4 * D),
                    (SQ, 1),
                    (0, 0),
                    (QC, 4 * D),
                    (1, 0),
                )
                target = tl.make_block_ptr(
                    Packed + task * M * DP,
                    (QC, 4 * D),
                    (4 * D, 1),
                    (0, 0),
                    (QC, 4 * D),
                    (1, 0),
                )
                tl.store(
                    target, tl.load(source, boundary_check=(0,), padding_option="zero")
                )
            else:
                target = tl.make_block_ptr(
                    Packed + task * M * DP, (M, DP), (DP, 1), (0, 0), (M, DP), (1, 0)
                )
                tl.store(target, tl.zeros((M, DP), tl.int8))
                for local_head in range(PACK):
                    for query in range(QC):
                        if query < length:
                            source = tl.make_block_ptr(
                                Query
                                + (begin + query) * SQ
                                + (head_group * PACK + local_head) * 4 * D,
                                (4, D),
                                (D, 1),
                                (0, 0),
                                (4, D),
                                (1, 0),
                            )
                            diagonal = tl.make_block_ptr(
                                Packed + task * M * DP,
                                (M, DP),
                                (DP, 1),
                                ((local_head * QC + query) * 4, local_head * D),
                                (4, D),
                                (1, 0),
                            )
                            tl.store(diagonal, tl.load(source))
                        else:
                            pass
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
    Meta,
    PAGES: tl.constexpr,
    ATTENTION_N: tl.constexpr,
):
    W: tl.constexpr = heads * D
    pages = tl.arange(0, PAGES)
    for sequence in range(tl.program_id(0), batch, 40):
        length = tl.load(Used + sequence)
        first = tl.load(Table + sequence * table_stride, length > 0, 0).to(tl.int64)
        physical = tl.load(
            Table + sequence * table_stride + pages, pages * page < length, 0
        ).to(tl.int64)
        contiguous = (
            tl.sum(
                tl.where(
                    pages * page < length, (physical == first + pages).to(tl.int32), 1
                ),
                0,
            )
            == PAGES
        )
        contiguous &= length % ATTENTION_N == 0
        tl.store(Meta + sequence, tl.where(contiguous, first, -1).to(tl.int32))
        if not contiguous:
            for logical in range(tl.cdiv(length, ATTENTION_N) * (ATTENTION_N // page)):
                source_page = tl.load(
                    Table + sequence * table_stride + logical,
                    logical * page < length,
                    0,
                ).to(tl.int64)
                source_k = tl.make_block_ptr(
                    K + source_page * pk, (page, W), (sk, 1), (0, 0), (page, W), (1, 0)
                )
                target_k = tl.make_block_ptr(
                    PackedK + sequence * capacity * W,
                    (capacity, W),
                    (W, 1),
                    (logical * page, 0),
                    (page, W),
                    (1, 0),
                )
                tl.store(target_k, tl.load(source_k))
                source_v = tl.make_block_ptr(
                    V + source_page * pv, (page, W), (sv, 1), (0, 0), (page, W), (1, 0)
                )
                target_v = tl.make_block_ptr(
                    PackedV + sequence * capacity * W,
                    (capacity, W),
                    (W, 1),
                    (logical * page, 0),
                    (page, W),
                    (1, 0),
                )
                tl.store(target_v, tl.load(source_v))
        else:
            pass


@triton.jit
def prepare_decode_inputs(
    Q, Packed, K, V, Key, Value, Cuq, Used, Table, VS, Flags, Meta, CONFIG: tl.constexpr
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    B: tl.constexpr = CONFIG.batch
    HKG: tl.constexpr = CONFIG.kv_heads // CONFIG.head_pack
    D: tl.constexpr = CONFIG.dim
    QC: tl.constexpr = CONFIG.query_cap
    PACK: tl.constexpr = CONFIG.head_pack
    SQ: tl.constexpr = CONFIG.query_stride
    SVB: tl.constexpr = CONFIG.v_scale_strides[0]
    SVH: tl.constexpr = CONFIG.v_scale_strides[1]
    SVK: tl.constexpr = CONFIG.v_scale_strides[2]
    SCALE_PAD: tl.constexpr = CONFIG.scale_blocks
    N: tl.constexpr = CONFIG.block_n
    pack_queries_diagonal(Q, Packed, Cuq, B, HKG, D, SQ, QC, PACK)
    scale_blocks = tl.arange(0, SCALE_PAD)
    for task in range(tl.program_id(0), B * HKG, 40):
        batch = task // HKG
        head_group = task % HKG
        nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        if nq > 0:
            nk = tl.load(Used + batch)
            heads = head_group * PACK + tl.arange(0, PACK)
            if SVK == 0:
                values = tl.load(VS + batch * SVB + heads * SVH)
                maximum = tl.max(tl.abs(values), 0)
            else:
                values = tl.load(
                    VS
                    + batch * SVB
                    + heads[:, None] * SVH
                    + scale_blocks[None, :] * SVK,
                    scale_blocks[None, :] < tl.cdiv(nk, 128),
                    0,
                )
                maximum = tl.max(tl.max(tl.abs(values), 1), 0)
            rounding = (
                8.0 * (tl.cdiv(nk, N) + 4) * 5.960464477539063e-08 * (1.0 + 0.008 * nk)
            )
            if DIGITS == 2:
                bound = 128.0 * maximum * rounding
                index = tl.arange(0, 256)
                payload = tl.where(
                    index & 127 == 1,
                    maximum.to(tl.int32, bitcast=True),
                    bound.to(tl.int32, bitcast=True),
                )
                target = tl.make_block_ptr(
                    Flags + task * 256, (256,), (1,), (0,), (256,), (0,)
                )
                tl.store(target, payload)
            else:
                quantization = nk * (1.0 / (2.0 * 2048383.0) + 1e-09)
                bound = 128.0 * maximum * (quantization + rounding)
                tl.store(Flags + task, (~(bound <= 0.01)).to(tl.int32))
        elif DIGITS == 2:
            target = tl.make_block_ptr(
                Flags + task * 256, (256,), (1,), (0,), (256,), (0,)
            )
            tl.store(target, tl.zeros((256,), tl.int32))
        else:
            tl.store(Flags + task, 0)
    pack_key_value(
        K,
        V,
        Key,
        Value,
        Table,
        Used,
        B,
        CONFIG.kv_heads,
        CONFIG.key_capacity,
        16,
        CONFIG.table_stride,
        CONFIG.key_stride,
        CONFIG.key_page_stride,
        CONFIG.value_stride,
        CONFIG.value_page_stride,
        D,
        Meta,
        CONFIG.table_blocks,
        N,
    )


@triton.jit
def cube_score(Query, Key, Score, column_tile, nk, CONFIG: tl.constexpr):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    DP: tl.constexpr = CONFIG.dim * CONFIG.head_pack
    CN0: tl.constexpr = N if N <= 32768 // M else 32768 // M
    CN: tl.constexpr = CN0 if CN0 <= 131072 // DP else 131072 // DP
    query_ptr = tl.make_block_ptr(Query, (M, DP), (DP, 1), (0, 0), (M, DP), (1, 0))
    query = tl.load(query_ptr)
    for column in range(0, N, CN):
        key_ptr = tl.make_block_ptr(
            Key,
            (CONFIG.key_capacity, DP),
            (CONFIG.kv_heads * CONFIG.dim, 1),
            (column_tile * N + column, 0),
            (CN, DP),
            (1, 0),
        )
        score = tl.dot(query, tl.trans(tl.load(key_ptr)), out_dtype=tl.int32)
        target = tl.make_block_ptr(Score, (M, N), (N, 1), (0, column), (M, CN), (1, 0))
        tl.store(target, score)


@triton.jit
def cube_product(
    Probability,
    Value,
    Product,
    column_tile,
    CONFIG: tl.constexpr,
    RESIDUAL: tl.constexpr,
    nk,
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    PLANES: tl.constexpr = 2 if DIGITS == 2 else 4
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    DP: tl.constexpr = CONFIG.dim * CONFIG.head_pack
    KN: tl.constexpr = (
        256
        if DIGITS == 2 and CONFIG.query_cap == 1 and (N >= 256)
        else 512 if N >= 512 else N
    )
    accumulator = tl.zeros((PLANES * M, DP), tl.int32)
    for column in range(0, N, KN):
        vp = tl.make_block_ptr(
            Value,
            (CONFIG.key_capacity, DP),
            (CONFIG.kv_heads * CONFIG.dim, 1),
            (column_tile * N + column, 0),
            (KN, DP),
            (1, 0),
        )
        pp = tl.make_block_ptr(
            Probability, (DIGITS * M, N), (N, 1), (0, column), (PLANES * M, KN), (1, 0)
        )
        if DIGITS == 2:
            p = tl.load(pp)
        else:
            p = tl.load(pp, boundary_check=(0,), padding_option="zero")
        accumulator = tl.dot(p, tl.load(vp), accumulator, out_dtype=tl.int32)
    target = tl.make_block_ptr(
        Product, (PLANES * M, DP), (DP, 1), (0, 0), (PLANES * M, DP), (1, 0)
    )
    tl.store(target, accumulator)


@triton.jit
def cube_task(
    query,
    key_base,
    value_base,
    Score,
    Probability,
    Product,
    core,
    nq,
    nk,
    tile,
    CONFIG: tl.constexpr,
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    PLANES: tl.constexpr = 2 if CONFIG.probability_digits == 2 else 4
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim * CONFIG.head_pack
    Q_ROWS: tl.constexpr = M // CONFIG.group
    end = nk
    if CONFIG.causal:
        end = tl.minimum(nk, (tile + 1) * Q_ROWS + nk - nq)
    else:
        pass
    count = tl.cdiv(tl.maximum(end, 0), N)
    if count > 0:
        cube_score(query, key_base, Score + core * 2 * M * N, 0, nk, CONFIG)
        stage_sync(0, 0, False)
        for column_tile in tl.range(0, count, disable_licm=True):
            slot = column_tile % 2
            if column_tile + 1 < count:
                next_slot = (column_tile + 1) % 2
                cube_score(
                    query,
                    key_base,
                    Score + (core * 2 + next_slot) * M * N,
                    column_tile + 1,
                    nk,
                    CONFIG,
                )
                stage_sync(next_slot, 0, False)
            else:
                pass
            stage_sync(slot, 1, True)
            probability_base = Probability + (core * 2 + slot) * DIGITS * M * N
            product_base = Product + (core * 2 + slot) * PLANES * M * D
            cube_product(
                probability_base,
                value_base,
                product_base,
                column_tile,
                CONFIG,
                False,
                nk,
            )
            stage_sync(slot, 2, False)
    else:
        pass


@triton.jit
def cube_main(
    Query,
    Key,
    Value,
    Cuq,
    Used,
    RawK,
    RawV,
    Meta,
    ValueScale,
    Score,
    Probability,
    Product,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    D: tl.constexpr = CONFIG.dim
    PACK: tl.constexpr = CONFIG.head_pack
    HK: tl.constexpr = CONFIG.kv_heads
    HKG: tl.constexpr = HK // PACK
    core = tl.program_id(0)
    for task in tl.range(core, CONFIG.batch * HKG, CONFIG.cores, disable_licm=True):
        batch = task // HKG
        head = task % HKG
        nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        nk = tl.load(Used + batch)
        if nq > 0:
            query = Query + (batch * HKG + head) * M * D * PACK
            first = tl.load(Meta + batch).to(tl.int64)
            if first >= 0:
                cube_task(
                    query,
                    RawK + (first * 16 * HK + head * PACK) * D,
                    RawV + (first * 16 * HK + head * PACK) * D,
                    Score,
                    Probability,
                    Product,
                    core,
                    nq,
                    nk,
                    0,
                    CONFIG,
                )
            else:
                cube_task(
                    query,
                    Key + (batch * CONFIG.key_capacity * HK + head * PACK) * D,
                    Value + (batch * CONFIG.key_capacity * HK + head * PACK) * D,
                    Score,
                    Probability,
                    Product,
                    core,
                    nq,
                    nk,
                    0,
                    CONFIG,
                )
        else:
            pass
    al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_S)


@triton.jit
def softmax_block(
    Score,
    Probability,
    QueryScale,
    KeyScale,
    ValueScale,
    Alibi,
    MaxState,
    SumState,
    AlphaState,
    batch,
    head,
    tile,
    nq,
    nk,
    column_tile,
    sub,
    CONFIG: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    VR: tl.constexpr = M // 2
    CHUNK: tl.constexpr = CONFIG.block_m // 2 if CONFIG.query_cap > 1 else CONFIG.group
    PAD: tl.constexpr = 1 if CONFIG.query_cap > 1 else 2
    G: tl.constexpr = CONFIG.group
    QSHIFT: tl.constexpr = (
        0
        if CONFIG.query_cap == 1
        else 1 if CONFIG.query_cap == 2 else 2 if CONFIG.query_cap == 4 else 3
    )
    columns = column_tile * N + tl.arange(0, N)
    descale_block = column_tile * (N // 128)
    slot = column_tile % 2
    for piece in tl.range(0, VR, CHUNK, loop_unroll_factor=1, disable_licm=True):
        query_rows = sub * VR + piece + tl.arange(0, CHUNK) >> 2 & CONFIG.query_cap - 1
        kv_head = head * CONFIG.head_pack + (sub * VR + piece >> QSHIFT + 2)
        query_heads = kv_head * G + (sub * VR + piece + tl.arange(0, CHUNK) & 3)
        valid_rows = query_rows < nq
        query_scale = tl.load(
            QueryScale
            + batch * CONFIG.q_scale_strides[0]
            + query_heads * CONFIG.q_scale_strides[1],
            valid_rows,
            0,
        )
        sp = tl.make_block_ptr(
            Score, (M, N), (N, 1), (sub * VR + piece, 0), (CHUNK, N), (1, 0)
        )
        scores = tl.load(sp).to(tl.float32)
        key_scale = tl.load(
            KeyScale
            + batch * CONFIG.k_scale_strides[0]
            + kv_head * CONFIG.k_scale_strides[1]
            + descale_block * CONFIG.k_scale_strides[2]
        )
        scores *= (query_scale * key_scale * CONFIG.scale)[:, None]
        if CONFIG.softcap > 0:
            scores = CONFIG.softcap * apply_softcap(
                scores, 1.0 / CONFIG.softcap, is_softcap=True
            )
        else:
            pass
        if CONFIG.has_alibi:
            slope = tl.load(
                Alibi
                + batch * CONFIG.alibi_batch_stride
                + query_heads * CONFIG.alibi_head_stride
            )
            scores = apply_alibi(
                scores,
                columns,
                query_rows,
                nq,
                nk,
                is_causal=False,
                is_alibi=True,
                alibi_slope=slope[:, None],
            )
        else:
            pass
        valid = valid_rows[:, None] & (columns[None, :] < nk)
        aligned_rows = query_rows + nk - nq
        if CONFIG.causal:
            valid &= columns[None, :] <= aligned_rows[:, None]
        else:
            pass
        if CONFIG.left >= 0:
            valid &= columns[None, :] >= aligned_rows[:, None] - CONFIG.left
        else:
            pass
        if CONFIG.right >= 0:
            valid &= columns[None, :] <= aligned_rows[:, None] + CONFIG.right
        else:
            pass
        scores *= 1.4426950408889634
        scores = tl.where(valid, scores, float("-inf"))
        max_slot = tle.dsa.subview(MaxState, [piece * PAD], [CHUNK], [1])
        sum_slot = tle.dsa.subview(SumState, [piece * PAD], [CHUNK], [1])
        alpha_slot = tle.dsa.subview(
            AlphaState, [slot * VR * PAD + piece * PAD], [CHUNK], [1]
        )
        maximum = tle.dsa.to_tensor(max_slot, writable=False)
        denominator = tle.dsa.to_tensor(sum_slot, writable=False)
        block_maximum = tl.max(scores, 1)
        new_maximum = tl.maximum(maximum, block_maximum)
        safe_maximum = tl.where(new_maximum == float("-inf"), 0, new_maximum)
        alpha = tl.exp2(maximum - safe_maximum)
        if DIGITS == 2:
            safe_block_maximum = tl.where(
                block_maximum == float("-inf"), 0.0, block_maximum
            )
            beta = tl.exp2(block_maximum - safe_maximum)
            probability = tl.exp2(scores - safe_block_maximum[:, None])
            new_denominator = denominator * alpha + beta * tl.sum(probability, 1)
        else:
            probability = tl.exp2(scores - safe_maximum[:, None])
            new_denominator = denominator * alpha + tl.sum(probability, 1)
        tle.dsa.to_buffer(alpha, bind_buffer=alpha_slot)
        if DIGITS == 2:
            beta_slot = tle.dsa.subview(
                AlphaState, [3 * VR * PAD + slot * VR * PAD + piece * PAD], [CHUNK], [1]
            )
            tle.dsa.to_buffer(beta, bind_buffer=beta_slot)
        else:
            pass
        tle.dsa.to_buffer(new_maximum, bind_buffer=max_slot)
        tle.dsa.to_buffer(new_denominator, bind_buffer=sum_slot)
        remainder = probability
        for digit in tl.static_range(DIGITS):
            integer = tl.floor(remainder * 127.0 ** (digit + 1) + 0.5)
            quantized = al.cast(integer, tl.int8, overflow_mode="saturate")
            pp = tl.make_block_ptr(
                Probability + digit * M * N,
                (M, N),
                (N, 1),
                (sub * VR + piece, 0),
                (CHUNK, N),
                (1, 0),
            )
            tl.store(pp, quantized)
            remainder -= integer * (1.0 / 127.0 ** (digit + 1))
        if DIGITS == 2:
            error_slot = tle.dsa.subview(
                AlphaState, [2 * VR * PAD + piece * PAD], [CHUNK], [1]
            )
            previous_error = tle.dsa.to_tensor(error_slot, writable=False)
            value_scale = tl.load(
                ValueScale
                + batch * CONFIG.v_scale_strides[0]
                + kv_head * CONFIG.v_scale_strides[1]
                + descale_block * CONFIG.v_scale_strides[2]
            )
            block_error = (
                128.0 * tl.abs(value_scale) * tl.sum(tl.abs(remainder) + 2e-08, 1)
            )
            tle.dsa.to_buffer(
                previous_error * alpha + block_error * beta, bind_buffer=error_slot
            )
        else:
            pass


@triton.jit
def prepare_probability(
    Score,
    Probability,
    QueryScale,
    KeyScale,
    ValueScale,
    Alibi,
    MaxState,
    SumState,
    AlphaState,
    batch,
    head,
    tile,
    nq,
    nk,
    column_tile,
    sub,
    CONFIG: tl.constexpr,
):
    softmax_block(
        Score,
        Probability,
        QueryScale,
        KeyScale,
        ValueScale,
        Alibi,
        MaxState,
        SumState,
        AlphaState,
        batch,
        head,
        tile,
        nq,
        nk,
        column_tile,
        sub,
        CONFIG,
        False,
    )


@triton.jit
def accumulate_product(
    Product,
    ValueScale,
    AccState,
    AlphaState,
    batch,
    head,
    column_tile,
    sub,
    CONFIG: tl.constexpr,
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    DP: tl.constexpr = D * CONFIG.head_pack
    VR: tl.constexpr = M // 2
    G: tl.constexpr = CONFIG.group
    QSHIFT: tl.constexpr = (
        0
        if CONFIG.query_cap == 1
        else 1 if CONFIG.query_cap == 2 else 2 if CONFIG.query_cap == 4 else 3
    )
    PAD: tl.constexpr = 1 if CONFIG.query_cap > 1 else 2
    C: tl.constexpr = VR if CONFIG.query_cap > 1 else G
    slot = column_tile % 2
    for piece in range(0, VR, C):
        local_head = sub * VR + piece >> QSHIFT + 2
        kv_head = head * CONFIG.head_pack + local_head
        value_scale = tl.load(
            ValueScale
            + batch * CONFIG.v_scale_strides[0]
            + kv_head * CONFIG.v_scale_strides[1]
            + column_tile * (N // 128) * CONFIG.v_scale_strides[2]
        )
        acc_slot = tle.dsa.subview(AccState, [piece, 0], [C, D], [1, 1])
        alpha_slot = tle.dsa.subview(
            AlphaState, [slot * VR * PAD + piece * PAD], [C], [1]
        )
        accumulator = tle.dsa.to_tensor(acc_slot, writable=False)
        alpha = tle.dsa.to_tensor(alpha_slot, writable=False)
        if DIGITS == 2:
            beta_slot = tle.dsa.subview(
                AlphaState, [3 * VR * PAD + slot * VR * PAD + piece * PAD], [C], [1]
            )
            beta = tle.dsa.to_tensor(beta_slot, writable=False)
        else:
            pass
        product = tl.zeros((C, D), tl.float32)
        for digit in tl.static_range(DIGITS):
            source = tl.make_block_ptr(
                Product + digit * M * DP,
                (M, DP),
                (DP, 1),
                (sub * VR + piece, local_head * D),
                (C, D),
                (1, 0),
            )
            product += tl.load(source).to(tl.float32) * (1.0 / 127.0 ** (digit + 1))
        if DIGITS == 2:
            tle.dsa.to_buffer(
                accumulator * alpha[:, None] + product * (value_scale * beta)[:, None],
                bind_buffer=acc_slot,
            )
        else:
            tle.dsa.to_buffer(
                accumulator * alpha[:, None] + product * value_scale,
                bind_buffer=acc_slot,
            )


@triton.jit
def write_single_output(
    Output,
    LSE,
    Cuq,
    AccState,
    MaxState,
    SumState,
    batch,
    head,
    sub,
    CONFIG: tl.constexpr,
    Flags,
    nk,
    task,
    ErrorState,
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    G: tl.constexpr = CONFIG.group
    D: tl.constexpr = CONFIG.dim
    VR: tl.constexpr = CONFIG.block_m // 2
    QSHIFT: tl.constexpr = (
        0
        if CONFIG.query_cap == 1
        else 1 if CONFIG.query_cap == 2 else 2 if CONFIG.query_cap == 4 else 3
    )
    PAD: tl.constexpr = 1 if CONFIG.query_cap > 1 else 2
    begin = tl.load(Cuq + batch).to(tl.int64)
    nq = tl.load(Cuq + batch + 1) - begin
    if DIGITS == 2:
        metadata = Flags + task * 256 + sub * 128
        rounding_bound = tl.load(metadata).to(tl.float32, bitcast=True)
        unsafe = tl.full((), 0, tl.int32)
    else:
        pass
    for piece in range(0, VR, G):
        query_index = sub * VR + piece >> 2 & CONFIG.query_cap - 1
        local_head = sub * VR + piece >> QSHIFT + 2
        query_head = (head * CONFIG.head_pack + local_head) * G
        if query_index < nq:
            acc_slot = tle.dsa.subview(AccState, [piece, 0], [G, D], [1, 1])
            sum_slot = tle.dsa.subview(SumState, [piece * PAD], [G], [1])
            accumulator = tle.dsa.to_tensor(acc_slot, writable=False)
            denominator = tle.dsa.to_tensor(sum_slot, writable=False)
            if DIGITS == 2:
                error_slot = tle.dsa.subview(
                    ErrorState, [2 * VR * PAD + piece * PAD], [G], [1]
                )
                quantization_bound = tle.dsa.to_tensor(
                    error_slot, writable=False
                ) / tl.where(denominator > 0, denominator, 1)
                bound = tl.where(
                    denominator > 0, quantization_bound + rounding_bound, 0.0
                )
                unsafe = tl.maximum(unsafe, tl.max((~(bound <= 0.01)).to(tl.int32), 0))
            else:
                pass
            output = accumulator / tl.where(denominator > 0, denominator, 1)[:, None]
            target = tl.make_block_ptr(
                Output + (begin + query_index) * CONFIG.output_stride + query_head * D,
                (G, D),
                (D, 1),
                (0, 0),
                (G, D),
                (1, 0),
            )
            tl.store(target, output.to(Output.dtype.element_ty))
            if CONFIG.write_lse:
                max_slot = tle.dsa.subview(MaxState, [piece * PAD], [G], [1])
                maximum = tle.dsa.to_tensor(max_slot, writable=False)
                lse = tl.where(
                    denominator > 0,
                    maximum * 0.6931471805599453 + tl.log(denominator),
                    float("inf"),
                )
                tl.store(
                    LSE
                    + (query_head + tl.arange(0, G)) * CONFIG.total_query
                    + begin
                    + query_index,
                    lse,
                )
            else:
                pass
        else:
            pass
    if DIGITS == 2:
        flag_target = tl.make_block_ptr(metadata, (128,), (1,), (0,), (128,), (0,))
        tl.store(flag_target, tl.full((128,), unsafe, tl.int32))
    else:
        pass


@triton.jit
def vector_main(
    Output,
    LSE,
    Cuq,
    Used,
    QueryScale,
    KeyScale,
    ValueScale,
    Alibi,
    Score,
    Probability,
    Product,
    Flags,
    CONFIG: tl.constexpr,
):
    DIGITS: tl.constexpr = CONFIG.probability_digits
    PLANES: tl.constexpr = 2 if CONFIG.probability_digits == 2 else 4
    SLOTS: tl.constexpr = 5 if DIGITS == 2 else 2
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    VR: tl.constexpr = M // 2
    PAD: tl.constexpr = 1 if CONFIG.query_cap > 1 else 2
    G: tl.constexpr = CONFIG.group
    B: tl.constexpr = CONFIG.batch
    HK: tl.constexpr = CONFIG.kv_heads // CONFIG.head_pack
    core = tl.program_id(0)
    sub = al.sub_vec_id().to(tl.int32)
    max_state = tle.dsa.alloc([VR * PAD], tl.float32, tle.dsa.ascend.UB)
    sum_state = tle.dsa.alloc([VR * PAD], tl.float32, tle.dsa.ascend.UB)
    alpha_state = tle.dsa.alloc([SLOTS * VR * PAD], tl.float32, tle.dsa.ascend.UB)
    acc_state = tle.dsa.alloc([VR, D], tl.float32, tle.dsa.ascend.UB)
    for task in tl.range(
        core, CONFIG.query_tiles * B * HK, CONFIG.cores, disable_licm=True
    ):
        tile = task // (B * HK)
        batch = task // HK % B
        head = task % HK
        nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        nk = tl.load(Used + batch)
        if tile * (M // G) < nq:
            zero = ub_zero()
            negative_infinity = (zero.to(tl.uint32, bitcast=True) | 0xFF800000).to(
                tl.float32, bitcast=True
            )
            tle.dsa.to_buffer(
                tl.full((VR * PAD,), negative_infinity, tl.float32),
                bind_buffer=max_state,
            )
            tle.dsa.to_buffer(
                tl.full((VR * PAD,), zero, tl.float32), bind_buffer=sum_state
            )
            tle.dsa.to_buffer(tl.full((VR, D), zero, tl.float32), bind_buffer=acc_state)
            if DIGITS == 2:
                error_state = tle.dsa.subview(
                    alpha_state, [2 * VR * PAD], [VR * PAD], [1]
                )
                tle.dsa.to_buffer(
                    tl.full((VR * PAD,), zero, tl.float32), bind_buffer=error_state
                )
            else:
                pass
            end = nk
            if CONFIG.causal:
                end = tl.minimum(nk, (tile + 1) * (M // G) + nk - nq)
            else:
                pass
            count = tl.cdiv(tl.maximum(end, 0), N)
            if count > 0:
                stage_sync(0, 0, True)
                prepare_probability(
                    Score + core * 2 * M * N,
                    Probability + core * 2 * DIGITS * M * N,
                    QueryScale,
                    KeyScale,
                    ValueScale,
                    Alibi,
                    max_state,
                    sum_state,
                    alpha_state,
                    batch,
                    head,
                    tile,
                    nq,
                    nk,
                    0,
                    sub,
                    CONFIG,
                )
                stage_sync(0, 1, False)
                for column_tile in tl.range(0, count, disable_licm=True):
                    slot = column_tile % 2
                    if column_tile + 1 < count:
                        next_slot = (column_tile + 1) % 2
                        stage_sync(next_slot, 0, True)
                        prepare_probability(
                            Score + (core * 2 + next_slot) * M * N,
                            Probability + (core * 2 + next_slot) * DIGITS * M * N,
                            QueryScale,
                            KeyScale,
                            ValueScale,
                            Alibi,
                            max_state,
                            sum_state,
                            alpha_state,
                            batch,
                            head,
                            tile,
                            nq,
                            nk,
                            column_tile + 1,
                            sub,
                            CONFIG,
                        )
                        stage_sync(next_slot, 1, False)
                    else:
                        pass
                    stage_sync(slot, 2, True)
                    accumulate_product(
                        Product + (core * 2 + slot) * PLANES * M * D * CONFIG.head_pack,
                        ValueScale,
                        acc_state,
                        alpha_state,
                        batch,
                        head,
                        column_tile,
                        sub,
                        CONFIG,
                    )
            else:
                pass
            write_single_output(
                Output,
                LSE,
                Cuq,
                acc_state,
                max_state,
                sum_state,
                batch,
                head,
                sub,
                CONFIG,
                Flags,
                nk,
                task,
                alpha_state,
            )
        else:
            pass
    al.sync_block_set("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_S)


@triton.jit
def replay_ready(WAIT: tl.constexpr):
    if WAIT:
        al.sync_block_wait("cube", "vector", 0, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
    else:
        al.sync_block_set("cube", "vector", 0, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)


@triton.jit
def replay_consumed(WAIT: tl.constexpr):
    if WAIT:
        al.sync_block_wait("vector", "cube", 1, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)
    else:
        al.sync_block_set("vector", "cube", 1, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)


@triton.jit
def replay_cube_task(Query, Key, Score, nk, CONFIG: tl.constexpr):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    for column in range(tl.cdiv(nk, N)):
        cube_score(Query, Key, Score + column * M * N, column, nk, CONFIG)
    replay_ready(False)
    replay_consumed(True)


@triton.jit
def replay_cube_main(
    Query, Key, RawK, Meta, Cuq, Used, Flags, Score, CONFIG: tl.constexpr
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    PACK: tl.constexpr = CONFIG.head_pack
    HK: tl.constexpr = CONFIG.kv_heads
    HKG: tl.constexpr = HK // PACK
    BLOCKS: tl.constexpr = tl.cdiv(CONFIG.key_capacity, N)
    core = tl.program_id(0)
    for task in tl.range(core, CONFIG.batch * HKG, CONFIG.cores, disable_licm=True):
        batch = task // HKG
        head = task % HKG
        if CONFIG.probability_digits == 2:
            flag = tl.load(Flags + task * 256, volatile=True) | tl.load(
                Flags + task * 256 + 128, volatile=True
            )
        else:
            flag = tl.load(Flags + task, volatile=True)
        if flag != 0:
            nk = tl.load(Used + batch)
            first = tl.load(Meta + batch).to(tl.int64)
            query = Query + task * M * D * PACK
            target = Score + core * BLOCKS * M * N
            if first >= 0:
                replay_cube_task(
                    query,
                    RawK + (first * 16 * HK + head * PACK) * D,
                    target,
                    nk,
                    CONFIG,
                )
            else:
                replay_cube_task(
                    query,
                    Key + (batch * CONFIG.key_capacity * HK + head * PACK) * D,
                    target,
                    nk,
                    CONFIG,
                )
        else:
            pass
    al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)


@triton.jit
def replay_score_values(
    Score,
    QS,
    KS,
    Alibi,
    batch,
    query_head,
    kv_head,
    query,
    nq,
    nk,
    column,
    offset,
    logical_row,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    columns = column * N + offset + tl.arange(0, 64)
    values = tl.load(
        Score + column * M * N + logical_row * N + offset + tl.arange(0, 64)
    ).to(tl.float32)
    qs = tl.load(
        QS + batch * CONFIG.q_scale_strides[0] + query_head * CONFIG.q_scale_strides[1]
    )
    ks = tl.load(
        KS
        + batch * CONFIG.k_scale_strides[0]
        + kv_head * CONFIG.k_scale_strides[1]
        + column * (N // 128) * CONFIG.k_scale_strides[2]
    )
    values *= qs * ks * CONFIG.scale
    if CONFIG.softcap > 0:
        values = CONFIG.softcap * apply_softcap(
            values, 1.0 / CONFIG.softcap, is_softcap=True
        )
    else:
        pass
    if CONFIG.has_alibi:
        slope = tl.load(
            Alibi
            + batch * CONFIG.alibi_batch_stride
            + query_head * CONFIG.alibi_head_stride
        )
        values2 = apply_alibi(
            values[None, :],
            columns,
            tl.full((1,), query, tl.int32),
            nq,
            nk,
            is_causal=False,
            is_alibi=True,
            alibi_slope=tl.full((1, 1), slope, tl.float32),
        )
        values = tl.reshape(values2, (64,))
    else:
        pass
    valid = columns < nk
    if CONFIG.causal:
        valid &= columns <= query + nk - nq
    else:
        pass
    if CONFIG.left >= 0:
        valid &= columns >= query + nk - nq - CONFIG.left
    else:
        pass
    if CONFIG.right >= 0:
        valid &= columns <= query + nk - nq + CONFIG.right
    else:
        pass
    return tl.where(valid, values * 1.4426950408889634, float("-inf"))


@triton.jit
def replay_rows(
    Score,
    Value,
    Output,
    QS,
    KS,
    VS,
    Alibi,
    begin,
    batch,
    head,
    sub,
    nq,
    nk,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    G: tl.constexpr = CONFIG.group
    VR: tl.constexpr = M // 2
    QSHIFT: tl.constexpr = (
        0
        if CONFIG.query_cap == 1
        else 1 if CONFIG.query_cap == 2 else 2 if CONFIG.query_cap == 4 else 3
    )
    for local_row in range(VR):
        logical_row = sub * VR + local_row
        query = logical_row >> 2 & CONFIG.query_cap - 1
        local_head = logical_row >> QSHIFT + 2
        kv_head = head * CONFIG.head_pack + local_head
        query_head = kv_head * G + (logical_row & 3)
        if query < nq:
            maximum = tl.full((), float("-inf"), tl.float32)
            for column in range(tl.cdiv(nk, N)):
                for offset in range(0, N, 64):
                    score = replay_score_values(
                        Score,
                        QS,
                        KS,
                        Alibi,
                        batch,
                        query_head,
                        kv_head,
                        query,
                        nq,
                        nk,
                        column,
                        offset,
                        logical_row,
                        CONFIG,
                    )
                    maximum = tl.maximum(maximum, tl.max(score, 0))
            safe_maximum = tl.where(maximum == float("-inf"), 0, maximum)
            numerator = tl.zeros((D,), tl.float32)
            denominator = tl.full((), 0.0, tl.float32)
            for column in range(tl.cdiv(nk, N)):
                scale = tl.load(
                    VS
                    + batch * CONFIG.v_scale_strides[0]
                    + kv_head * CONFIG.v_scale_strides[1]
                    + column * (N // 128) * CONFIG.v_scale_strides[2]
                )
                for offset in range(0, N, 64):
                    score = replay_score_values(
                        Score,
                        QS,
                        KS,
                        Alibi,
                        batch,
                        query_head,
                        kv_head,
                        query,
                        nq,
                        nk,
                        column,
                        offset,
                        logical_row,
                        CONFIG,
                    )
                    probability = tl.exp2(score - safe_maximum)
                    denominator += tl.sum(probability, 0)
                    vp = tl.make_block_ptr(
                        Value,
                        (CONFIG.key_capacity, CONFIG.head_pack * D),
                        (CONFIG.kv_heads * D, 1),
                        (column * N + offset, local_head * D),
                        (64, D),
                        (1, 0),
                    )
                    values = tl.load(vp).to(tl.float32)
                    numerator += tl.sum(probability[:, None] * values, 0) * scale
            output = numerator / tl.where(denominator > 0, denominator, 1)
            tl.store(
                Output
                + (begin + query) * CONFIG.output_stride
                + query_head * D
                + tl.arange(0, D),
                output.to(Output.dtype.element_ty),
            )
        else:
            pass


@triton.jit
def replay_vector_main(
    Value,
    RawV,
    Meta,
    Output,
    Cuq,
    Used,
    QS,
    KS,
    VS,
    Alibi,
    Flags,
    Score,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    PACK: tl.constexpr = CONFIG.head_pack
    HK: tl.constexpr = CONFIG.kv_heads
    HKG: tl.constexpr = HK // PACK
    BLOCKS: tl.constexpr = tl.cdiv(CONFIG.key_capacity, N)
    core = tl.program_id(0)
    sub = al.sub_vec_id().to(tl.int32)
    for task in tl.range(core, CONFIG.batch * HKG, CONFIG.cores, disable_licm=True):
        if CONFIG.probability_digits == 2:
            flag_pointer = tl.make_block_ptr(
                Flags + task * 256, (256,), (1,), (0,), (256,), (0,)
            )
            flag = tl.max(tl.load(flag_pointer), 0)
        else:
            flag = tl.load(Flags + task)
        if flag != 0:
            batch = task // HKG
            head = task % HKG
            begin = tl.load(Cuq + batch).to(tl.int64)
            nq = tl.load(Cuq + batch + 1) - begin
            nk = tl.load(Used + batch)
            first = tl.load(Meta + batch).to(tl.int64)
            replay_ready(True)
            if first >= 0:
                replay_rows(
                    Score + core * BLOCKS * M * N,
                    RawV + (first * 16 * HK + head * PACK) * D,
                    Output,
                    QS,
                    KS,
                    VS,
                    Alibi,
                    begin,
                    batch,
                    head,
                    sub,
                    nq,
                    nk,
                    CONFIG,
                )
            else:
                replay_rows(
                    Score + core * BLOCKS * M * N,
                    Value + (batch * CONFIG.key_capacity * HK + head * PACK) * D,
                    Output,
                    QS,
                    KS,
                    VS,
                    Alibi,
                    begin,
                    batch,
                    head,
                    sub,
                    nq,
                    nk,
                    CONFIG,
                )
            replay_consumed(False)
        else:
            pass
    al.sync_block_set("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)


@triton.jit
def decode_attention_kernel(
    Query,
    Key,
    Value,
    Output,
    LSE,
    Cuq,
    Used,
    RawK,
    RawV,
    Meta,
    QS,
    KS,
    VS,
    Alibi,
    Score,
    Probability,
    Product,
    Flags,
    ReplayScore,
    CONFIG: tl.constexpr,
):
    with al.scope(core_mode="cube"):
        cube_main(
            Query,
            Key,
            Value,
            Cuq,
            Used,
            RawK,
            RawV,
            Meta,
            VS,
            Score,
            Probability,
            Product,
            CONFIG,
        )
        if CONFIG.probability_digits == 2:
            al.sync_block_set("cube", "vector", 7, PIPE.PIPE_S, PIPE.PIPE_MTE2)
        else:
            pass
        replay_cube_main(Query, Key, RawK, Meta, Cuq, Used, Flags, ReplayScore, CONFIG)
    with al.scope(core_mode="vector"):
        vector_main(
            Output,
            LSE,
            Cuq,
            Used,
            QS,
            KS,
            VS,
            Alibi,
            Score,
            Probability,
            Product,
            Flags,
            CONFIG,
        )
        if CONFIG.probability_digits == 2:
            al.sync_block_wait("cube", "vector", 7, PIPE.PIPE_S, PIPE.PIPE_MTE2)
        else:
            pass
        replay_vector_main(
            Value,
            RawV,
            Meta,
            Output,
            Cuq,
            Used,
            QS,
            KS,
            VS,
            Alibi,
            Flags,
            ReplayScore,
            CONFIG,
        )


def decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    cuq: torch.Tensor,
    used: torch.Tensor,
    table: torch.Tensor,
    qs: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    max_query: int,
    max_key: int,
    scale: float,
    causal: bool,
    left: int,
    right: int,
    softcap: float,
    alibi: Optional[torch.Tensor],
    write_lse: bool,
) -> None:
    batch = cuq.shape[0] - 1
    heads = k.shape[-2]
    dim = q.shape[-1]
    query_cap = triton.next_power_of_2(max_query)
    pack = 4 if query_cap == 1 else 2 if query_cap <= 4 else 1
    digits = 2 if max_query == 1 and max_key > 1024 else 3
    planes = 2 if digits == 2 else 4
    rows = query_cap * 4 * pack
    columns = (
        (1024 if query_cap == 1 else 512)
        if ks.stride(2) == 0 and vs.stride(2) == 0
        else 128
    )
    capacity = triton.cdiv(max_key, columns) * columns
    groups = heads // pack
    cores = min(20, batch * groups)
    query = torch.empty(
        (batch, groups, rows, dim * pack), dtype=torch.int8, device=q.device
    )
    key = torch.empty((batch, capacity, heads, dim), dtype=torch.int8, device=q.device)
    value = torch.empty(
        (batch, capacity, heads, dim), dtype=torch.int8, device=q.device
    )
    metadata = torch.empty(batch, dtype=torch.int32, device=q.device)
    flag_words = 256 if digits == 2 else 1
    flags = torch.empty(batch * groups * flag_words, dtype=torch.int32, device=q.device)
    score = torch.empty((cores, 2, rows, columns), dtype=torch.int32, device=q.device)
    probability = torch.empty(
        (cores, 2, digits, rows, columns), dtype=torch.int8, device=q.device
    )
    product = torch.empty(
        (cores, 2, planes, rows, dim * pack), dtype=torch.int32, device=q.device
    )
    replay_score = torch.empty(
        (cores, triton.cdiv(capacity, columns), rows, columns),
        dtype=torch.int32,
        device=q.device,
    )
    alibi_batch_stride = 0
    alibi_head_stride = 0
    if alibi is not None:
        alibi_head_stride = alibi.stride(-1)
        if alibi.ndim == 2:
            alibi_batch_stride = alibi.stride(0)
        else:
            pass
    else:
        pass
    configuration = DecodeConfig(
        batch=batch,
        kv_heads=heads,
        group=4,
        dim=dim,
        query_capacity=16,
        key_capacity=capacity,
        query_tiles=1,
        total_query=q.shape[0],
        output_stride=output.stride(0),
        q_scale_strides=qs.stride(),
        k_scale_strides=ks.stride(),
        v_scale_strides=vs.stride(),
        causal=causal,
        write_lse=write_lse,
        scale=scale,
        block_m=rows,
        block_n=columns,
        cores=cores,
        left=left,
        right=right,
        softcap=softcap,
        has_alibi=alibi is not None,
        alibi_batch_stride=alibi_batch_stride,
        alibi_head_stride=alibi_head_stride,
        head_pack=pack,
        query_cap=query_cap,
        probability_digits=digits,
        query_stride=q.stride(0),
        key_stride=k.stride(-3),
        key_page_stride=k.stride(0),
        value_stride=v.stride(-3),
        value_page_stride=v.stride(0),
        table_stride=table.stride(0),
        scale_blocks=triton.next_power_of_2(vs.shape[2]),
        table_blocks=triton.next_power_of_2(table.shape[1]),
    )
    prepare_decode_inputs[40,](
        q, query, k, v, key, value, cuq, used, table, vs, flags, metadata, configuration
    )
    decode_attention_kernel[cores,](
        query,
        key,
        value,
        output,
        lse,
        cuq,
        used,
        k,
        v,
        metadata,
        qs,
        ks,
        vs,
        alibi if alibi is not None else qs,
        score,
        probability,
        product,
        flags,
        replay_score,
        configuration,
        multibuffer=digits == 2 and query_cap == 1,
        enable_auto_bind_sub_block=False,
        disable_auto_inject_block_sync=True,
    )
