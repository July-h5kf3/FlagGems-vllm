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
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.runtime.backend._ascend.fused.attention_native import (
    RESIDUAL_SCALE_BITS,
    stage_sync,
    ub_zero,
)

PIPE = al.PIPE


@triton.jit
def cube_score(query, Key, Score, column, CONFIG: tl.constexpr):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    CN: tl.constexpr = 128 if N >= 128 else N
    for piece in range(0, N, CN):
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
            (column * N + piece, 0),
            (CN, D),
            (1, 0),
        )
        score = tl.dot(query, tl.trans(tl.load(kp)), out_dtype=tl.int32)
        sp = tl.make_block_ptr(Score, (M, N), (N, 1), (0, piece), (M, CN), (1, 0))
        tl.store(sp, score)


@triton.jit
def cube_loop(
    Query,
    Key,
    Value,
    Score,
    Probability,
    Product,
    Flags,
    core,
    count,
    CONFIG: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    qp = tl.make_block_ptr(
        Query,
        (M, D),
        (CONFIG.query_stride if CONFIG.query_stride > 0 else D, 1),
        (0, 0),
        (M, D),
        (1, 0),
    )
    query = tl.load(qp)
    accumulator = tl.zeros((M, D), tl.float32)
    cube_score(query, Key, Score + core * (2 * M * N) + 0 * M * N, 0, CONFIG)
    stage_sync(0, 0, False)
    for column in tl.range(0, count, disable_licm=True):
        slot = column & 1
        if column + 1 < count:
            cube_score(
                query,
                Key,
                Score + core * (2 * M * N) + (column + 1 & 1) * M * N,
                column + 1,
                CONFIG,
            )
            stage_sync(column + 1 & 1, 0, False)
        else:
            pass
        if N <= 256:
            value_ptr = tl.make_block_ptr(
                Value, (CONFIG.key_capacity, D), (D, 1), (column * N, 0), (N, D), (1, 0)
            )
            prefetched_value = tl.load(value_ptr)
        else:
            pass
        stage_sync(slot, 1, True)
        KN: tl.constexpr = 256 if N >= 256 else N
        for piece in range(0, N, KN):
            pp = tl.make_block_ptr(
                Probability + core * (4 * M * N) + slot * 2 * M * N,
                (M, N),
                (N, 1),
                (0, piece),
                (M, KN),
                (1, 0),
            )
            vp = tl.make_block_ptr(
                Value,
                (CONFIG.key_capacity, D),
                (D, 1),
                (column * N + piece, 0),
                (KN, D),
                (1, 0),
            )
            values = prefetched_value if N <= 256 else tl.load(vp)
            accumulator = tl.dot(tl.load(pp), values, accumulator, out_dtype=tl.float32)
            if RESIDUAL:
                rp = tl.make_block_ptr(
                    Probability + core * (4 * M * N) + slot * 2 * M * N + M * N,
                    (M, N),
                    (N, 1),
                    (0, piece),
                    (M, KN),
                    (1, 0),
                )
                accumulator = tl.dot(
                    tl.load(rp), values, accumulator, out_dtype=tl.float32
                )
            else:
                pass
    target = tl.make_block_ptr(
        Product + core * (M * D), (M, D), (D, 1), (0, 0), (M, D), (1, 0)
    )
    tl.store(target, accumulator)
    al.sync_block_set("cube", "vector", 7, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)


@triton.jit
def cube_main(
    Query,
    Key,
    Value,
    Cuq,
    Used,
    QBatch,
    QTile,
    Counts,
    VS,
    Score,
    Probability,
    Product,
    Flags,
    CONFIG: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    HK: tl.constexpr = CONFIG.kv_heads
    B: tl.constexpr = CONFIG.batch
    G: tl.constexpr = CONFIG.group
    core = tl.program_id(0)
    if CONFIG.has_worklist:
        work_count = tl.load(Counts)
    else:
        work_count = CONFIG.query_tiles * B
    for linear_task in tl.range(core, work_count * HK, CONFIG.cores, disable_licm=True):
        if CONFIG.task_offset > 0:
            shifted = (linear_task + CONFIG.task_offset) % (CONFIG.query_tiles * B * HK)
            shifted_head = shifted % HK
            shifted_tile = (
                shifted // (B * HK)
                + shifted_head // CONFIG.kv_head_repeat * CONFIG.tile_shift
            ) % CONFIG.query_tiles
            task = (shifted_tile * B + shifted // HK % B) * HK + shifted_head
        else:
            task = linear_task
        if CONFIG.has_worklist:
            batch = tl.load(QBatch + task // HK)
            tile = tl.load(QTile + task // HK)
        else:
            batch = task // HK % B
            tile = task // (B * HK)
        head = task % HK
        if CONFIG.uniform_q_length > 0:
            nq = CONFIG.uniform_q_length
        else:
            nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        nk = tl.load(Used + batch)
        if tile * (M // G) < nq:
            end = nk
            if CONFIG.causal:
                end = tl.minimum(nk, (tile + 1) * (M // G) + nk - nq)
            else:
                pass
            count = tl.cdiv(tl.maximum(end, 0), N)
            if count > 0:
                if CONFIG.query_stride > 0:
                    query = (
                        Query
                        + (batch * CONFIG.uniform_q_length + tile * M)
                        * CONFIG.query_stride
                        + head * D
                    )
                else:
                    query = (
                        Query
                        + ((batch * HK + head) * CONFIG.query_capacity * G + tile * M)
                        * D
                    )
                if CONFIG.key_head_major:
                    key = (
                        Key
                        + (
                            batch * (HK // CONFIG.kv_head_repeat)
                            + head // CONFIG.kv_head_repeat
                        )
                        * CONFIG.key_capacity
                        * D
                    )
                else:
                    key = (
                        Key
                        + (
                            batch * CONFIG.key_capacity * (HK // CONFIG.kv_head_repeat)
                            + head // CONFIG.kv_head_repeat
                        )
                        * D
                    )
                value = (
                    Value
                    + (
                        batch * (HK // CONFIG.kv_head_repeat)
                        + head // CONFIG.kv_head_repeat
                    )
                    * CONFIG.key_capacity
                    * D
                )
                flag = Flags + task * 256
                vs = tl.load(
                    VS
                    + batch * CONFIG.v_scale_strides[0]
                    + head // CONFIG.kv_head_repeat * CONFIG.v_scale_strides[1]
                )
                if not vs.to(tl.int32, bitcast=True) & 2147483647 > RESIDUAL_SCALE_BITS:
                    cube_loop(
                        query,
                        key,
                        value,
                        Score,
                        Probability,
                        Product,
                        flag,
                        core,
                        count,
                        CONFIG,
                        False,
                    )
                else:
                    pass
            else:
                pass
        else:
            pass
    al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)


@triton.jit
def prepare_block(
    Score,
    Probability,
    Flags,
    Coefficients,
    Denominator,
    batch,
    head,
    tile,
    nq,
    nk,
    column,
    sub,
    CONFIG: tl.constexpr,
    RESIDUAL: tl.constexpr,
    MASKED: tl.constexpr,
    FIRST: tl.constexpr,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    VR: tl.constexpr = M // 2
    C: tl.constexpr = (
        CONFIG.vector_elements // N if VR >= CONFIG.vector_elements // N else VR
    )
    any_update = tl.full((), 0, tl.int32)
    for piece in tl.range(0, VR, C, disable_licm=True, loop_unroll_factor=1):
        rows = sub * VR + piece + tl.arange(0, C)
        qp = Coefficients
        sp = tl.make_block_ptr(
            Score, (M, N), (N, 1), (sub * VR + piece, 0), (C, N), (1, 0)
        )
        if MASKED:
            qr = tile * (M // CONFIG.group) + rows // CONFIG.group
            kc = column * N + tl.arange(0, N)
            valid = (qr[:, None] < nq) & (kc[None, :] < nk)
            if CONFIG.causal:
                valid &= kc[None, :] <= qr[:, None] + nk - nq
            else:
                pass
        else:
            pass
        if CONFIG.group == 1:
            scores = tl.load(sp).to(tl.float32) * qp
        else:
            scores = tl.load(sp).to(tl.float32) * qp[:, None]
        if MASKED:
            scores = tl.where(valid, scores, float("-inf"))
        else:
            pass
        ds = tle.dsa.subview(Denominator, [piece], [C], [1])
        denominator = tle.dsa.to_tensor(ds, writable=False)
        # A fixed reference avoids per-block rescaling; unsafe rows are recomputed.
        reference = tl.full((C,), -2.772588722239781, tl.float32)
        safe = tl.where(reference == float("-inf"), 0, reference)
        p = tl.exp(scores - safe[:, None])
        block_sum = tl.sum(p, 1)
        tle.dsa.to_buffer(denominator + block_sum, bind_buffer=ds)
        half = p.to(tl.float16)
        pp = tl.make_block_ptr(
            Probability, (M, N), (N, 1), (sub * VR + piece, 0), (C, N), (1, 0)
        )
        tl.store(pp, half)
        if RESIDUAL:
            rp = tl.make_block_ptr(
                Probability + M * N,
                (M, N),
                (N, 1),
                (sub * VR + piece, 0),
                (C, N),
                (1, 0),
            )
            tl.store(rp, (p - half.to(tl.float32)).to(tl.float16))
        else:
            pass
    return any_update


@triton.jit
def prepare_selected(
    Score,
    Probability,
    Flags,
    Coefficients,
    Denominator,
    batch,
    head,
    tile,
    nq,
    nk,
    column,
    sub,
    bits,
    CONFIG: tl.constexpr,
    FIRST: tl.constexpr,
):
    query_rows: tl.constexpr = CONFIG.block_m // CONFIG.group
    half_rows: tl.constexpr = query_rows // 2
    rows_full = tile * query_rows + (sub + 1) * half_rows <= nq
    columns_full = (column + 1) * CONFIG.block_n <= nk
    if CONFIG.causal:
        columns_full &= (
            column + 1
        ) * CONFIG.block_n <= tile * query_rows + sub * half_rows + nk - nq + 1
    else:
        pass
    is_full = rows_full & columns_full
    if is_full:
        required = prepare_block(
            Score,
            Probability,
            Flags,
            Coefficients,
            Denominator,
            batch,
            head,
            tile,
            nq,
            nk,
            column,
            sub,
            CONFIG,
            False,
            False,
            FIRST,
        )
    else:
        required = prepare_block(
            Score,
            Probability,
            Flags,
            Coefficients,
            Denominator,
            batch,
            head,
            tile,
            nq,
            nk,
            column,
            sub,
            CONFIG,
            False,
            True,
            FIRST,
        )
    return required


@triton.jit
def vector_main(
    Output,
    LSE,
    Cuq,
    Used,
    QBatch,
    QTile,
    Counts,
    QS,
    KS,
    VS,
    Score,
    Probability,
    Product,
    Flags,
    CONFIG: tl.constexpr,
    UnsafeCount,
):
    M: tl.constexpr = CONFIG.block_m
    N: tl.constexpr = CONFIG.block_n
    D: tl.constexpr = CONFIG.dim
    G: tl.constexpr = CONFIG.group
    HK: tl.constexpr = CONFIG.kv_heads
    B: tl.constexpr = CONFIG.batch
    VR: tl.constexpr = M // 2
    C: tl.constexpr = (
        CONFIG.vector_elements // N if VR >= CONFIG.vector_elements // N else VR
    )
    core = tl.program_id(0)
    sub = al.sub_vec_id().to(tl.int32)
    denominator = tle.dsa.alloc([VR], tl.float32, tle.dsa.ascend.UB)
    if CONFIG.has_worklist:
        work_count = tl.load(Counts)
    else:
        work_count = CONFIG.query_tiles * B
    for linear_task in tl.range(core, work_count * HK, CONFIG.cores, disable_licm=True):
        if CONFIG.task_offset > 0:
            shifted = (linear_task + CONFIG.task_offset) % (CONFIG.query_tiles * B * HK)
            shifted_head = shifted % HK
            shifted_tile = (
                shifted // (B * HK)
                + shifted_head // CONFIG.kv_head_repeat * CONFIG.tile_shift
            ) % CONFIG.query_tiles
            task = (shifted_tile * B + shifted // HK % B) * HK + shifted_head
        else:
            task = linear_task
        if CONFIG.has_worklist:
            batch = tl.load(QBatch + task // HK)
            tile = tl.load(QTile + task // HK)
        else:
            batch = task // HK % B
            tile = task // (B * HK)
        head = task % HK
        if CONFIG.uniform_q_length > 0:
            nq = CONFIG.uniform_q_length
        else:
            nq = tl.load(Cuq + batch + 1) - tl.load(Cuq + batch)
        nk = tl.load(Used + batch)
        if tile * (M // G) < nq:
            vs = tl.load(
                VS
                + batch * CONFIG.v_scale_strides[0]
                + head // CONFIG.kv_head_repeat * CONFIG.v_scale_strides[1]
            )
            bits = vs.to(tl.int32, bitcast=True) & 2147483647
            if bits <= RESIDUAL_SCALE_BITS:
                zero = ub_zero()
                tle.dsa.to_buffer(
                    tl.full((VR,), zero, tl.float32), bind_buffer=denominator
                )
                if G == 1:
                    qs = tl.load(
                        QS
                        + batch * CONFIG.q_scale_strides[0]
                        + head * CONFIG.q_scale_strides[1]
                    )
                else:
                    qheads = head * G + (sub * VR + tl.arange(0, C) & G - 1)
                    qs = tl.load(
                        QS
                        + batch * CONFIG.q_scale_strides[0]
                        + qheads * CONFIG.q_scale_strides[1]
                    )
                ks = tl.load(
                    KS
                    + batch * CONFIG.k_scale_strides[0]
                    + head // CONFIG.kv_head_repeat * CONFIG.k_scale_strides[1]
                )
                coefficients = qs * ks * CONFIG.scale
                end = nk
                if CONFIG.causal:
                    end = tl.minimum(nk, (tile + 1) * (M // G) + nk - nq)
                else:
                    pass
                count = tl.cdiv(tl.maximum(end, 0), N)
                flag = Flags + task * 256
                if count > 0:
                    stage_sync(0, 0, True)
                    score = Score + core * (2 * M * N) + 0 * M * N
                    probability = Probability + core * (4 * M * N) + 0 * 2 * M * N
                    prepare_selected(
                        score,
                        probability,
                        flag,
                        coefficients,
                        denominator,
                        batch,
                        head,
                        tile,
                        nq,
                        nk,
                        0,
                        sub,
                        bits,
                        CONFIG,
                        True,
                    )
                    stage_sync(0, 1, False)
                    for column in tl.range(1, count, disable_licm=True):
                        slot = column & 1
                        stage_sync(slot, 0, True)
                        score = Score + core * (2 * M * N) + slot * M * N
                        probability = (
                            Probability + core * (4 * M * N) + slot * 2 * M * N
                        )
                        prepare_selected(
                            score,
                            probability,
                            flag,
                            coefficients,
                            denominator,
                            batch,
                            head,
                            tile,
                            nq,
                            nk,
                            column,
                            sub,
                            bits,
                            CONFIG,
                            False,
                        )
                        stage_sync(slot, 1, False)
                    al.sync_block_wait(
                        "cube", "vector", 7, PIPE.PIPE_FIX, PIPE.PIPE_MTE2
                    )
                else:
                    pass
                output_requires = tl.zeros((16,), tl.int32)
                for piece in range(0, VR, 16):
                    ds = tle.dsa.subview(denominator, [piece], [16], [1])
                    sums = tle.dsa.to_tensor(ds, writable=False)
                    if count > 0:
                        source = tl.make_block_ptr(
                            Product + core * (M * D),
                            (M, D),
                            (D, 1),
                            (sub * VR + piece, 0),
                            (16, D),
                            (1, 0),
                        )
                        numerator = tl.load(source)
                    else:
                        numerator = tl.zeros((16, D), tl.float32)
                    # Any nonfinite FP16 P makes every PV column nonfinite for finite V.
                    sample = tl.load(
                        Product
                        + core * (M * D)
                        + (sub * VR + piece + tl.arange(0, 16)) * D
                    )
                    query_row = (
                        tile * (M // G) + (sub * VR + piece + tl.arange(0, 16)) // G
                    )
                    has_keys = (query_row < nq) & (nk > 0)
                    if CONFIG.causal:
                        has_keys &= query_row + nk - nq >= 0
                    else:
                        pass
                    output_requires = output_requires | (
                        has_keys & ((sums < 1.0) | ~(tl.abs(sample) < float("inf")))
                    ).to(tl.int32)
                    output = numerator / tl.where(sums > 0, sums, 1)[:, None] * vs
                    base = (batch * HK + head) * CONFIG.query_capacity * G
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
                            (16 // G, G * D),
                            (1, 0),
                        )
                        if (
                            CONFIG.uniform_q_length > 0
                            and CONFIG.uniform_q_length == CONFIG.query_capacity
                        ):
                            tl.store(
                                target,
                                output.to(Output.dtype.element_ty).reshape(
                                    (16 // G, G * D)
                                ),
                            )
                        elif row // G + 16 // G <= nq:
                            tl.store(
                                target,
                                output.to(Output.dtype.element_ty).reshape(
                                    (16 // G, G * D)
                                ),
                            )
                        else:
                            tl.store(
                                target,
                                output.to(Output.dtype.element_ty).reshape(
                                    (16 // G, G * D)
                                ),
                                boundary_check=(0,),
                            )
                    else:
                        row = tile * M + sub * VR + piece
                        base = (batch * HK + head) * CONFIG.query_capacity * G
                        target = tl.make_block_ptr(
                            Output + base * D,
                            (CONFIG.query_capacity * G, D),
                            (D, 1),
                            (row, 0),
                            (16, D),
                            (1, 0),
                        )
                        tl.store(target, output.to(Output.dtype.element_ty))
                    if CONFIG.write_lse:
                        maxima = tl.full((16,), -2.772588722239781, tl.float32)
                        lse = tl.where(sums > 0, maxima + tl.log(sums), float("inf"))
                        if CONFIG.direct_output:
                            lane = tl.arange(0, 16)
                            query_row = (row + lane) // G
                            query_head = head * G + (row + lane) % G
                            if (
                                CONFIG.uniform_q_length > 0
                                and CONFIG.uniform_q_length == CONFIG.query_capacity
                            ):
                                tl.store(
                                    LSE
                                    + query_head * CONFIG.total_query
                                    + begin
                                    + query_row,
                                    lse,
                                )
                            else:
                                tl.store(
                                    LSE
                                    + query_head * CONFIG.total_query
                                    + begin
                                    + query_row,
                                    lse,
                                    query_row < nq,
                                )
                        else:
                            lp = tl.make_block_ptr(
                                LSE + base,
                                (CONFIG.query_capacity * G,),
                                (1,),
                                (row,),
                                (16,),
                                (0,),
                            )
                            tl.store(lp, lse)
                    else:
                        pass
                task_unsafe = tl.max(output_requires, 0)
                tl.store(flag + sub * 128, task_unsafe)
                # Only zero/nonzero is used by the subsequent native recomputation kernel.
                if task_unsafe != 0:
                    tl.atomic_add(UnsafeCount, 1)
                else:
                    pass
            else:
                task_unsafe = 1
                tl.store(Flags + task * 256 + sub * 128, task_unsafe)
                # Only zero/nonzero is used by the subsequent native recomputation kernel.
                if task_unsafe != 0:
                    tl.atomic_add(UnsafeCount, 1)
                else:
                    pass
        else:
            task_unsafe = 0
            tl.store(Flags + task * 256 + sub * 128, task_unsafe)
            if task_unsafe != 0:
                tl.atomic_add(UnsafeCount, 1)
            else:
                pass
    al.sync_block_set("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)


@triton.jit
def prefill_attention(
    Query,
    Key,
    Value,
    Output,
    LSE,
    Cuq,
    Used,
    QBatch,
    QTile,
    Counts,
    QS,
    KS,
    VS,
    Alibi,
    Score,
    Probability,
    Product,
    Flags,
    CONFIG: tl.constexpr,
    UnsafeCount,
):
    with al.scope(core_mode="cube"):
        cube_main(
            Query,
            Key,
            Value,
            Cuq,
            Used,
            QBatch,
            QTile,
            Counts,
            VS,
            Score,
            Probability,
            Product,
            Flags,
            CONFIG,
        )
    with al.scope(core_mode="vector"):
        vector_main(
            Output,
            LSE,
            Cuq,
            Used,
            QBatch,
            QTile,
            Counts,
            QS,
            KS,
            VS,
            Score,
            Probability,
            Product,
            Flags,
            CONFIG,
            UnsafeCount,
        )
