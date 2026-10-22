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
from typing import Tuple

import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.ops.flash_kernel import apply_alibi, apply_softcap

PIPE = al.PIPE
RESIDUAL_SCALE_BITS = tl.constexpr(1048576000)


@triton.jit
def ub_zero():
    # Prevent loop hoisting of the initialization scalar.
    bits = tl.inline_asm_elementwise("MOV $0, 0", "=l", [], tl.int64, False, 1)
    return bits.to(tl.uint32).to(tl.float32, bitcast=True)


@dataclass(frozen=True)
class AttentionConfig:
    __triton_compile_time_value__ = True
    batch: int
    kv_heads: int
    group: int
    dim: int
    query_capacity: int
    key_capacity: int
    query_tiles: int
    total_query: int
    output_stride: int
    q_scale_strides: Tuple[int, int, int]
    k_scale_strides: Tuple[int, int, int]
    v_scale_strides: Tuple[int, int, int]
    causal: bool
    write_lse: bool
    scale: float
    block_m: int
    block_n: int
    cores: int
    left: int
    right: int
    softcap: float
    has_alibi: bool
    alibi_batch_stride: int
    alibi_head_stride: int
    has_worklist: bool = False
    direct_output: bool = False
    filter_flags: bool = False
    kv_head_repeat: int = 1
    uniform_q_length: int = 0
    vector_elements: int = 8192
    aggregate_flags: bool = False
    query_stride: int = 0
    task_offset: int = 0
    tile_shift: int = 0
    key_head_major: bool = False


@triton.jit
def stage_sync(slot, kind: tl.constexpr, WAIT: tl.constexpr):
    if WAIT:
        if kind == 1:
            if slot == 0:
                al.sync_block_wait("vector", "cube", 1, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
            else:
                al.sync_block_wait("vector", "cube", 4, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
        elif slot == 0:
            al.sync_block_wait("cube", "vector", kind, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
        else:
            al.sync_block_wait(
                "cube", "vector", kind + 3, PIPE.PIPE_FIX, PIPE.PIPE_MTE2
            )
    elif kind == 1:
        if slot == 0:
            al.sync_block_set("vector", "cube", 1, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
        else:
            al.sync_block_set("vector", "cube", 4, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
    elif slot == 0:
        al.sync_block_set("cube", "vector", kind, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
    else:
        al.sync_block_set("cube", "vector", kind + 3, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)


@triton.jit
def cube_score(Query, Key, Score, column_tile, CONFIG: tl.constexpr):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    CM: tl.constexpr = 128 if M >= 128 else M
    CN: tl.constexpr = N if N <= 32768 // CM else 32768 // CM
    for column in range(0, N, CN):
        kp = tl.make_block_ptr(
            Key,
            (CONFIG.key_capacity, D),
            (
                (
                    D
                    if CONFIG.key_head_major
                    else CONFIG.kv_heads // CONFIG.kv_head_repeat * D
                ),
                1,
            ),
            (column_tile * N + column, 0),
            (CN, D),
            (1, 0),
        )
        key = tl.trans(tl.load(kp))
        for row in range(0, M, CM):
            qp = tl.make_block_ptr(
                Query,
                (M, D),
                (CONFIG.query_stride if CONFIG.query_stride > 0 else D, 1),
                (row, 0),
                (CM, D),
                (1, 0),
            )
            score = tl.dot(tl.load(qp), key, out_dtype=tl.int32)
            sp = tl.make_block_ptr(
                Score, (M, N), (N, 1), (row, column), (CM, CN), (1, 0)
            )
            tl.store(sp, score)


@triton.jit
def cube_product(
    Probability,
    Value,
    Product,
    column_tile,
    CONFIG: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    CM: tl.constexpr = 128 if M >= 128 else M
    vp = tl.make_block_ptr(
        Value, (CONFIG.key_capacity, D), (D, 1), (column_tile * N, 0), (N, D), (1, 0)
    )
    value = tl.load(vp)
    for row in range(0, M, CM):
        pp = tl.make_block_ptr(Probability, (M, N), (N, 1), (row, 0), (CM, N), (1, 0))
        product = tl.dot(tl.load(pp), value, out_dtype=tl.float32)
        if RESIDUAL:
            rp = tl.make_block_ptr(
                Probability + M * N, (M, N), (N, 1), (row, 0), (CM, N), (1, 0)
            )
            product = tl.dot(tl.load(rp), value, product, out_dtype=tl.float32)
        else:
            pass
        op = tl.make_block_ptr(Product, (M, D), (D, 1), (row, 0), (CM, D), (1, 0))
        tl.store(op, product)


@triton.jit
def cube_main(
    Query,
    Key,
    Value,
    Cuq,
    Used,
    QueryBatch,
    QueryTile,
    Counts,
    ValueScale,
    Score,
    Probability,
    Product,
    CONFIG: tl.constexpr,
    Flags,
    UnsafeCount,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    G: tl.constexpr = CONFIG.group
    HK: tl.constexpr = CONFIG.kv_heads
    B: tl.constexpr = CONFIG.batch
    Q_ROWS: tl.constexpr = M // G
    core = tl.program_id(0)
    if CONFIG.has_worklist:
        work_count = tl.load(Counts)
    else:
        work_count = CONFIG.query_tiles * B
    if CONFIG.aggregate_flags:
        if tl.load(UnsafeCount) == 0:
            work_count = 0
        else:
            pass
    else:
        pass
    for task in tl.range(core, work_count * HK, CONFIG.cores, disable_licm=True):
        if CONFIG.has_worklist:
            tile = tl.load(QueryTile + task // HK)
            batch = tl.load(QueryBatch + task // HK)
        else:
            tile = task // (B * HK)
            batch = task // HK % B
        head = task % HK
        if CONFIG.uniform_q_length > 0:
            nq = CONFIG.uniform_q_length
        else:
            nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        nk = tl.load(Used + batch)
        if CONFIG.filter_flags:
            required = tl.load(Flags + task * 256) | tl.load(Flags + task * 256 + 128)
        else:
            required = 1
        if (tile * Q_ROWS < nq) & (required != 0):
            if CONFIG.query_stride > 0:
                query = (
                    Query
                    + (batch * CONFIG.uniform_q_length + tile * M) * CONFIG.query_stride
                    + head * D
                )
            else:
                query = (
                    Query
                    + ((batch * HK + head) * CONFIG.query_capacity * G + tile * M) * D
                )
            if CONFIG.key_head_major:
                key_base = (
                    Key
                    + (
                        batch * (HK // CONFIG.kv_head_repeat)
                        + head // CONFIG.kv_head_repeat
                    )
                    * CONFIG.key_capacity
                    * D
                )
            else:
                key_base = (
                    Key
                    + (
                        batch * CONFIG.key_capacity * (HK // CONFIG.kv_head_repeat)
                        + head // CONFIG.kv_head_repeat
                    )
                    * D
                )
            value_base = (
                Value
                + (
                    batch * (HK // CONFIG.kv_head_repeat)
                    + head // CONFIG.kv_head_repeat
                )
                * CONFIG.key_capacity
                * D
            )
            end = nk
            if CONFIG.causal:
                end = tl.minimum(nk, (tile + 1) * Q_ROWS + nk - nq)
            else:
                pass
            count = tl.cdiv(tl.maximum(end, 0), N)
            if count > 0:
                cube_score(query, key_base, Score + core * 2 * M * N, 0, CONFIG)
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
                            CONFIG,
                        )
                        stage_sync(next_slot, 0, False)
                    else:
                        pass
                    stage_sync(slot, 1, True)
                    descale_block = column_tile * (N // 128)
                    value_scale = tl.load(
                        ValueScale
                        + batch * CONFIG.v_scale_strides[0]
                        + head // CONFIG.kv_head_repeat * CONFIG.v_scale_strides[1]
                        + descale_block * CONFIG.v_scale_strides[2]
                    )
                    scale_bits = value_scale.to(tl.int32, bitcast=True) & 2147483647
                    probability_base = Probability + (core * 2 + slot) * 2 * M * N
                    product_base = Product + (core * 2 + slot) * M * D
                    if scale_bits > RESIDUAL_SCALE_BITS:
                        cube_product(
                            probability_base,
                            value_base,
                            product_base,
                            column_tile,
                            CONFIG,
                            True,
                        )
                    else:
                        cube_product(
                            probability_base,
                            value_base,
                            product_base,
                            column_tile,
                            CONFIG,
                            False,
                        )
                    stage_sync(slot, 2, False)
            else:
                pass
        else:
            pass
    al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)


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
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    VR: tl.constexpr = M // 2
    CHUNK: tl.constexpr = (
        16 if VR >= 16 and M <= 256 and (N <= 256) else 8 if VR >= 8 else VR
    )
    G: tl.constexpr = CONFIG.group
    columns = column_tile * N + tl.arange(0, N)
    descale_block = column_tile * (N // 128)
    key_scale = tl.load(
        KeyScale
        + batch * CONFIG.k_scale_strides[0]
        + head // CONFIG.kv_head_repeat * CONFIG.k_scale_strides[1]
        + descale_block * CONFIG.k_scale_strides[2]
    )
    slot = column_tile % 2
    for piece in tl.range(0, VR, CHUNK, loop_unroll_factor=1, disable_licm=True):
        rows = sub * VR + piece + tl.arange(0, CHUNK)
        query_rows = tile * (M // G) + rows // G
        query_heads = head * G + rows % G
        valid_rows = query_rows < nq
        query_scale = tl.load(
            QueryScale
            + batch * CONFIG.q_scale_strides[0]
            + query_heads * CONFIG.q_scale_strides[1]
            + query_rows // 128 * CONFIG.q_scale_strides[2],
            valid_rows,
            0,
        )
        sp = tl.make_block_ptr(
            Score, (M, N), (N, 1), (sub * VR + piece, 0), (CHUNK, N), (1, 0)
        )
        scores = tl.load(sp).to(tl.float32)
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
        max_slot = tle.dsa.subview(MaxState, [piece], [CHUNK], [1])
        sum_slot = tle.dsa.subview(SumState, [piece], [CHUNK], [1])
        alpha_slot = tle.dsa.subview(AlphaState, [slot * VR + piece], [CHUNK], [1])
        maximum = tle.dsa.to_tensor(max_slot, writable=False)
        denominator = tle.dsa.to_tensor(sum_slot, writable=False)
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(new_maximum == float("-inf"), 0, new_maximum)
        alpha = tl.exp2(maximum - safe_maximum)
        probability = tl.exp2(scores - safe_maximum[:, None])
        new_denominator = denominator * alpha + tl.sum(probability, 1)
        tle.dsa.to_buffer(alpha, bind_buffer=alpha_slot)
        tle.dsa.to_buffer(new_maximum, bind_buffer=max_slot)
        tle.dsa.to_buffer(new_denominator, bind_buffer=sum_slot)
        probability_half = probability.to(tl.float16)
        pp = tl.make_block_ptr(
            Probability, (M, N), (N, 1), (sub * VR + piece, 0), (CHUNK, N), (1, 0)
        )
        tl.store(pp, probability_half)
        if RESIDUAL:
            residual = (probability - probability_half.to(tl.float32)).to(tl.float16)
            rp = tl.make_block_ptr(
                Probability + M * N,
                (M, N),
                (N, 1),
                (sub * VR + piece, 0),
                (CHUNK, N),
                (1, 0),
            )
            tl.store(rp, residual)
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
    N: tl.constexpr = CONFIG.block_n
    value_scale = tl.load(
        ValueScale
        + batch * CONFIG.v_scale_strides[0]
        + head // CONFIG.kv_head_repeat * CONFIG.v_scale_strides[1]
        + column_tile * (N // 128) * CONFIG.v_scale_strides[2]
    )
    scale_bits = value_scale.to(tl.int32, bitcast=True) & 2147483647
    if scale_bits > RESIDUAL_SCALE_BITS:
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
            True,
        )
    else:
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
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    VR: tl.constexpr = M // 2
    CHUNK: tl.constexpr = (
        16 if VR >= 16 and M <= 256 and (N <= 256) else 8 if VR >= 8 else VR
    )
    value_scale = tl.load(
        ValueScale
        + batch * CONFIG.v_scale_strides[0]
        + head // CONFIG.kv_head_repeat * CONFIG.v_scale_strides[1]
        + column_tile * (N // 128) * CONFIG.v_scale_strides[2]
    )
    slot = column_tile % 2
    for piece in tl.range(0, VR, CHUNK, loop_unroll_factor=1, disable_licm=True):
        acc_slot = tle.dsa.subview(AccState, [piece, 0], [CHUNK, D], [1, 1])
        alpha_slot = tle.dsa.subview(AlphaState, [slot * VR + piece], [CHUNK], [1])
        accumulator = tle.dsa.to_tensor(acc_slot, writable=False)
        alpha = tle.dsa.to_tensor(alpha_slot, writable=False)
        product_ptr = tl.make_block_ptr(
            Product, (M, D), (D, 1), (sub * VR + piece, 0), (CHUNK, D), (1, 0)
        )
        product = tl.load(product_ptr)
        updated = accumulator * alpha[:, None] + product * value_scale
        tle.dsa.to_buffer(updated, bind_buffer=acc_slot)


@triton.jit
def write_output(
    Output,
    LSE,
    Cuq,
    AccState,
    MaxState,
    SumState,
    batch,
    head,
    tile,
    nq,
    sub,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    G: tl.constexpr = CONFIG.group
    VR: tl.constexpr = M // 2
    CHUNK: tl.constexpr = (
        16 if VR >= 16 and M <= 256 and (N <= 256) else 8 if VR >= 8 else VR
    )
    output_base = (batch * CONFIG.kv_heads + head) * CONFIG.query_capacity * G
    for piece in tl.range(0, VR, CHUNK, loop_unroll_factor=1, disable_licm=True):
        acc_slot = tle.dsa.subview(AccState, [piece, 0], [CHUNK, D], [1, 1])
        sum_slot = tle.dsa.subview(SumState, [piece], [CHUNK], [1])
        accumulator = tle.dsa.to_tensor(acc_slot, writable=False)
        denominator = tle.dsa.to_tensor(sum_slot, writable=False)
        output = accumulator / tl.where(denominator > 0, denominator, 1)[:, None]
        row = tile * M + sub * VR + piece
        if CONFIG.direct_output:
            if CONFIG.uniform_q_length > 0:
                begin = batch * CONFIG.uniform_q_length
            else:
                begin = tl.load(Cuq + batch)
            row = tile * M + sub * VR + piece
            target = tl.make_block_ptr(
                Output + begin * CONFIG.output_stride + head * G * D,
                (nq, G * D),
                (CONFIG.output_stride, 1),
                (row // G, 0),
                (CHUNK // G, G * D),
                (1, 0),
            )
            if (
                CONFIG.uniform_q_length > 0
                and CONFIG.uniform_q_length == CONFIG.query_capacity
            ):
                tl.store(
                    target,
                    output.to(Output.dtype.element_ty).reshape((CHUNK // G, G * D)),
                )
            elif row // G + CHUNK // G <= nq:
                tl.store(
                    target,
                    output.to(Output.dtype.element_ty).reshape((CHUNK // G, G * D)),
                )
            else:
                tl.store(
                    target,
                    output.to(Output.dtype.element_ty).reshape((CHUNK // G, G * D)),
                    boundary_check=(0,),
                )
        else:
            target = tl.make_block_ptr(
                Output + output_base * D,
                (CONFIG.query_capacity * G, D),
                (D, 1),
                (row, 0),
                (CHUNK, D),
                (1, 0),
            )
            tl.store(target, output.to(Output.dtype.element_ty))
        if CONFIG.write_lse:
            max_slot = tle.dsa.subview(MaxState, [piece], [CHUNK], [1])
            maximum = tle.dsa.to_tensor(max_slot, writable=False)
            lse = tl.where(
                denominator > 0,
                maximum * 0.6931471805599453 + tl.log(denominator),
                float("inf"),
            )
            if CONFIG.direct_output:
                lane = tl.arange(0, CHUNK)
                query_row = (row + lane) // G
                query_head = head * G + (row + lane) % G
                if (
                    CONFIG.uniform_q_length > 0
                    and CONFIG.uniform_q_length == CONFIG.query_capacity
                ):
                    tl.store(
                        LSE + query_head * CONFIG.total_query + begin + query_row, lse
                    )
                else:
                    tl.store(
                        LSE + query_head * CONFIG.total_query + begin + query_row,
                        lse,
                        query_row < nq,
                    )
            else:
                lse_ptr = tl.make_block_ptr(
                    LSE + output_base,
                    (CONFIG.query_capacity * G,),
                    (1,),
                    (row,),
                    (CHUNK,),
                    (0,),
                )
                tl.store(lse_ptr, lse)
        else:
            pass


@triton.jit
def vector_main(
    Output,
    LSE,
    Cuq,
    Used,
    QueryBatch,
    QueryTile,
    Counts,
    QueryScale,
    KeyScale,
    ValueScale,
    Alibi,
    Score,
    Probability,
    Product,
    CONFIG: tl.constexpr,
    Flags,
    UnsafeCount,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    VR: tl.constexpr = M // 2
    G: tl.constexpr = CONFIG.group
    B: tl.constexpr = CONFIG.batch
    HK: tl.constexpr = CONFIG.kv_heads
    core = tl.program_id(0)
    sub = al.sub_vec_id().to(tl.int32)
    max_state = tle.dsa.alloc([VR], tl.float32, tle.dsa.ascend.UB)
    sum_state = tle.dsa.alloc([VR], tl.float32, tle.dsa.ascend.UB)
    alpha_state = tle.dsa.alloc([2 * VR], tl.float32, tle.dsa.ascend.UB)
    acc_state = tle.dsa.alloc([VR, D], tl.float32, tle.dsa.ascend.UB)
    if CONFIG.has_worklist:
        work_count = tl.load(Counts)
    else:
        work_count = CONFIG.query_tiles * B
    if CONFIG.aggregate_flags:
        if tl.load(UnsafeCount) == 0:
            work_count = 0
        else:
            pass
    else:
        pass
    for task in tl.range(core, work_count * HK, CONFIG.cores, disable_licm=True):
        if CONFIG.has_worklist:
            tile = tl.load(QueryTile + task // HK)
            batch = tl.load(QueryBatch + task // HK)
        else:
            tile = task // (B * HK)
            batch = task // HK % B
        head = task % HK
        if CONFIG.uniform_q_length > 0:
            nq = CONFIG.uniform_q_length
        else:
            nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        nk = tl.load(Used + batch)
        if CONFIG.filter_flags:
            required = tl.load(Flags + task * 256) | tl.load(Flags + task * 256 + 128)
        else:
            required = 1
        if (tile * (M // G) < nq) & (required != 0):
            zero = ub_zero()
            negative_infinity = (zero.to(tl.uint32, bitcast=True) | 0xFF800000).to(
                tl.float32, bitcast=True
            )
            tle.dsa.to_buffer(
                tl.full((VR,), negative_infinity, tl.float32), bind_buffer=max_state
            )
            tle.dsa.to_buffer(tl.full((VR,), zero, tl.float32), bind_buffer=sum_state)
            tle.dsa.to_buffer(tl.full((VR, D), zero, tl.float32), bind_buffer=acc_state)
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
                    Probability + core * 4 * M * N,
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
                            Probability + (core * 2 + next_slot) * 2 * M * N,
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
                        Product + (core * 2 + slot) * M * D,
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
            write_output(
                Output,
                LSE,
                Cuq,
                acc_state,
                max_state,
                sum_state,
                batch,
                head,
                tile,
                nq,
                sub,
                CONFIG,
            )
        else:
            pass
    al.sync_block_set("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)


@triton.jit
def native_tle_attention(
    Query,
    Key,
    Value,
    Output,
    LSE,
    Cuq,
    Used,
    QueryBatch,
    QueryTile,
    Counts,
    QueryScale,
    KeyScale,
    ValueScale,
    Alibi,
    Score,
    Probability,
    Product,
    CONFIG: tl.constexpr,
    Flags=None,
    UnsafeCount=None,
):
    with al.scope(core_mode="cube"):
        cube_main(
            Query,
            Key,
            Value,
            Cuq,
            Used,
            QueryBatch,
            QueryTile,
            Counts,
            ValueScale,
            Score,
            Probability,
            Product,
            CONFIG,
            Flags,
            UnsafeCount,
        )
    with al.scope(core_mode="vector"):
        vector_main(
            Output,
            LSE,
            Cuq,
            Used,
            QueryBatch,
            QueryTile,
            Counts,
            QueryScale,
            KeyScale,
            ValueScale,
            Alibi,
            Score,
            Probability,
            Product,
            CONFIG,
            Flags,
            UnsafeCount,
        )
