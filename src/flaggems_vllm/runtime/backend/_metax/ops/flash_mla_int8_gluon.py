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
from triton.experimental.gluon.language._core import builtin
from triton.experimental.gluon.language.metax import MacaMmaLayout, async_copy, mma

from flaggems_vllm.utils import libentry


@builtin
def native_byte_permute(first, second, selector, _semantic=None):
    # Native selectors 0..3 refer to the second operand.
    result = _semantic.builder.create_extern_elementwise(
        "",
        "",
        "llvm.mxc.byte.perm",
        [second.handle, first.handle, selector.handle],
        first.type.to_ir(_semantic.builder),
        True,
    )
    return gl.tensor(result, first.type)


@gluon.jit
def pack_probability_terms(integer):
    tokens: gl.constexpr = integer.shape[0]
    heads: gl.constexpr = integer.shape[1]
    grouped = gl.reshape(gl.permute(integer, (1, 0)), (heads, tokens // 4, 2, 2))
    even, odd = gl.split(grouped)
    first, third = gl.split(even)
    second, fourth = gl.split(odd)
    halves = gl.full(first.shape, 0x05040100, gl.int32, first.type.layout)
    select_bytes = gl.full(first.shape, 0x06040200, gl.int32, first.type.layout)
    first_pair = native_byte_permute(first, second, halves)
    second_pair = native_byte_permute(third, fourth, halves)
    # Quantized values fit 14 bits, so rounding cannot carry between 16-bit fields.
    high = native_byte_permute(
        (first_pair + 0x00400040) >> 7, (second_pair + 0x00400040) >> 7, select_bytes
    )
    low_bytes = native_byte_permute(first_pair, second_pair, select_bytes)
    low = low_bytes ^ ((high & 0x01010101) << 7)
    return gl.permute(high, (1, 0)), gl.permute(low.to(gl.int32), (1, 0))


@gluon.jit
def dense_int8_scores_resident(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Scores,
    LogicalKS,
    MaxStats,
    Q_STRIDES: gl.constexpr,
    QR_STRIDES: gl.constexpr,
    KV_STRIDES: gl.constexpr,
    KR_STRIDES: gl.constexpr,
    QS_STRIDES: gl.constexpr,
    KS_STRIDES: gl.constexpr,
    TS: gl.constexpr,
    LS: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
    CAP: gl.constexpr,
    SCALE: gl.constexpr,
    CAUSAL: gl.constexpr,
    BH: gl.constexpr = 64,
    SPLITS: gl.constexpr = 4,
):
    if BH >= 64:
        tn: gl.constexpr = 4
        matrix: gl.constexpr = MacaMmaLayout([4, 1], [1, 4, 16])
    elif BH == 32:
        tn: gl.constexpr = 2
        matrix: gl.constexpr = MacaMmaLayout([2, 2], [1, 2, 16])
    else:
        tn: gl.constexpr = 1
        matrix: gl.constexpr = MacaMmaLayout([1, 4], [1, 1, 16])
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 0)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 0)
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    rlayout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    split, head_tile, batch = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    qheads = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, layout))
    qdims = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))
    qptr = (
        Q
        + batch * Q_STRIDES[0]
        + (qheads // H)[:, None] * Q_STRIDES[1]
        + (qheads % H)[:, None] * Q_STRIDES[2]
        + qdims[None, :]
    )
    q0 = gl.convert_layout(gl.load(qptr, qheads[:, None] < H * SQ, 0), lhs)
    q1 = gl.convert_layout(gl.load(qptr + 256, qheads[:, None] < H * SQ, 0), lhs)
    rheads = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, rlayout))
    rdims_q = gl.arange(0, 64, layout=gl.SliceLayout(0, rlayout))
    qr = gl.convert_layout(
        gl.load(
            QR
            + batch * QR_STRIDES[0]
            + (rheads // H)[:, None] * QR_STRIDES[1]
            + (rheads % H)[:, None] * QR_STRIDES[2]
            + rdims_q[None, :],
            rheads[:, None] < H * SQ,
            0,
        ),
        lhs,
    )
    tokens = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    dims = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))
    rtokens = gl.arange(0, 64, layout=gl.SliceLayout(1, rlayout))
    rdims = gl.arange(0, 64, layout=gl.SliceLayout(0, rlayout))
    heads = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, matrix))
    output_tokens = gl.arange(0, 64, layout=gl.SliceLayout(0, matrix))
    qscale = gl.load(
        QS
        + batch * QS_STRIDES[0]
        + (heads // H) * QS_STRIDES[1]
        + (heads % H) * QS_STRIDES[2],
        heads < H * SQ,
        0,
    )
    length = gl.load(Lengths + batch * LS)
    kbuf = gl.allocate_shared_memory(
        gl.int8, (64, 256), gl.SwizzledSharedLayout(16, tn, 8, [1, 0])
    )
    krbuf = gl.allocate_shared_memory(
        KR.dtype.element_ty, (64, 64), gl.SwizzledSharedLayout(16, tn, 4, [1, 0])
    )
    if BH >= 64:
        local_n: gl.constexpr = 4
        local_warps: gl.constexpr = [4, 1, 1]
    elif BH == 32:
        local_n: gl.constexpr = 2
        local_warps: gl.constexpr = [2, 2, 1]
    else:
        local_n: gl.constexpr = 1
        local_warps: gl.constexpr = [1, 4, 1]
    score_blocked: gl.constexpr = gl.BlockedLayout(
        [4, local_n], [4, 16], local_warps[:2], [1, 0]
    )
    local_parent: gl.constexpr = gl.BlockedLayout(
        [4, 1, local_n], [4, 16, 1], local_warps, [2, 1, 0]
    )
    lane_layout: gl.constexpr = gl.SliceLayout(2, local_parent)
    maximum_matrix = gl.full(
        (BH, 64 // local_n), -float("inf"), gl.float32, lane_layout
    )
    # Keep log weights relative to the current maximum so large scores cannot
    # absorb the contribution from log2(KS).
    weighted_matrix = gl.full(
        (BH, 64 // local_n), -float("inf"), gl.float32, lane_layout
    )
    pages = gl.cdiv(length, 64)
    for page_index in range(pages * split // SPLITS, pages * (split + 1) // SPLITS):
        page = gl.load(Table + batch * TS + page_index, page_index * 64 < length, 0).to(
            gl.int64
        )
        content = gl.full((BH, 64), 0, gl.int32, matrix)
        for chunk in gl.static_range(2):
            if chunk == 0:
                query = q0
            else:
                query = q1
            async_copy.async_copy_global_to_shared(
                kbuf,
                KV
                + page * KV_STRIDES[0]
                + tokens[:, None] * KV_STRIDES[1]
                + (chunk * 256 + dims)[None, :],
                page_index * 64 + tokens[:, None] < length,
            )
            async_copy.commit_group()
            async_copy.wait_group(0)
            content = mma(query, kbuf.permute((1, 0)).load(rhs), content)
            gl.thread_barrier()
        query_rope = qr
        async_copy.async_copy_global_to_shared(
            krbuf,
            KR
            + page * KR_STRIDES[0]
            + rtokens[:, None] * KR_STRIDES[1]
            + rdims[None, :],
            page_index * 64 + rtokens[:, None] < length,
        )
        async_copy.commit_group()
        async_copy.wait_group(0)
        rope = mma(
            query_rope,
            krbuf.permute((1, 0)).load(rhs),
            gl.full((BH, 64), 0.0, gl.float32, matrix),
        )
        raw = content.to(gl.float32) + rope
        gl.thread_barrier()
        kscale = gl.load(
            KS + page * KS_STRIDES[0] + output_tokens * KS_STRIDES[1],
            page_index * 64 + output_tokens < length,
            0,
        )
        scores = raw * qscale[:, None] * kscale[None, :] * (SCALE * 1.4426950408889634)
        bound = gl.full((BH,), length, gl.int32, gl.SliceLayout(1, matrix))
        if CAUSAL:
            bound = gl.maximum(length - SQ + heads // H + 1, 0)
        scores = gl.where(
            page_index * 64 + output_tokens[None, :] < bound[:, None],
            scores,
            -float("inf"),
        )
        local_scores = gl.reshape(
            gl.convert_layout(scores, score_blocked), (BH, 64 // local_n, local_n)
        )
        local_scores = gl.convert_layout(local_scores, local_parent)
        expanded_ks, _ = gl.broadcast(gl.log2(kscale)[None, :], scores)
        local_ks = gl.reshape(
            gl.convert_layout(expanded_ks, score_blocked), (BH, 64 // local_n, local_n)
        )
        local_ks = gl.convert_layout(local_ks, local_parent)
        local_maximum = gl.max(local_scores, 2)
        safe_local = gl.where(local_maximum == -float("inf"), 0.0, local_maximum)
        local_weighted = gl.max(local_scores - safe_local[:, :, None] + local_ks, 2)
        next_maximum = gl.maximum(maximum_matrix, local_maximum)
        safe = gl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        alpha = maximum_matrix - safe
        beta = local_maximum - safe
        weighted_matrix = gl.maximum(weighted_matrix + alpha, local_weighted + beta)
        maximum_matrix = next_maximum
        offset = (
            (batch.to(gl.int64) * CAP + page_index * 64) * H * SQ
            + heads[:, None] * 64
            + output_tokens[None, :]
        )
        gl.store(Scores + offset, scores, heads[:, None] < H * SQ)
        if head_tile == 0:
            gl.store(LogicalKS + batch * CAP + page_index * 64 + output_tokens, kscale)
    maxima = gl.max(maximum_matrix, 1)
    safe = gl.where(maxima == -float("inf"), 0.0, maxima)
    weighted_maxima = gl.max(weighted_matrix + (maximum_matrix - safe[:, None]), 1)
    maxima = gl.convert_layout(maxima, heads.type.layout)
    weighted_maxima = gl.convert_layout(weighted_maxima, heads.type.layout)
    stats_offset = ((batch * H * SQ + heads) * SPLITS + split) * 2
    gl.store(MaxStats + stats_offset, maxima, heads < H * SQ)
    gl.store(MaxStats + stats_offset + 1, weighted_maxima, heads < H * SQ)


@gluon.jit
def score_body(
    Q,
    QR,
    KV,
    KR,
    Page,
    Batch,
    HeadTile,
    Length,
    PageIndex,
    Q_STRIDES: gl.constexpr,
    QR_STRIDES: gl.constexpr,
    KV_STRIDES: gl.constexpr,
    KR_STRIDES: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
    BH: gl.constexpr,
    FULL: gl.constexpr,
):
    if BH == 64:
        tm: gl.constexpr = 2
        tn: gl.constexpr = 2
        warps: gl.constexpr = [2, 2]
    elif BH == 32:
        tm: gl.constexpr = 1
        tn: gl.constexpr = 2
        warps: gl.constexpr = [2, 2]
    else:
        tm: gl.constexpr = 1
        tn: gl.constexpr = 1
        warps: gl.constexpr = [1, 4]
    matrix: gl.constexpr = MacaMmaLayout(warps, [tm, tn, 16])
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 0)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 0)
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    qlayout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    rows = HeadTile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, qlayout))
    qdims = gl.arange(0, 256, layout=gl.SliceLayout(0, qlayout))
    tokens = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    dims = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))
    qbuf = gl.allocate_shared_memory(
        gl.int8, (BH, 256), gl.SwizzledSharedLayout(16, tm, 8, [1, 0])
    )
    kbuf = gl.allocate_shared_memory(
        gl.int8, (64, 256), gl.SwizzledSharedLayout(16, tn, 8, [1, 0])
    )
    content = gl.full((BH, 64), 0, gl.int32, matrix)
    qmask = rows[:, None] < H * SQ
    if H * SQ % BH == 0:
        qmask = gl.full((BH, 1), True, gl.int1, qlayout)
    kmask = PageIndex * 64 + tokens[:, None] < Length
    if FULL:
        kmask = gl.full((64, 1), True, gl.int1, layout)
    for chunk in range(2):
        feature = chunk * 256 + dims
        async_copy.async_copy_global_to_shared(
            qbuf,
            Q
            + Batch * Q_STRIDES[0]
            + (rows // H)[:, None] * Q_STRIDES[1]
            + (rows % H)[:, None] * Q_STRIDES[2]
            + (chunk * 256 + qdims)[None, :],
            qmask,
        )
        async_copy.async_copy_global_to_shared(
            kbuf,
            KV
            + Page * KV_STRIDES[0]
            + tokens[:, None] * KV_STRIDES[1]
            + feature[None, :],
            kmask,
        )
        async_copy.commit_group()
        async_copy.wait_group(0)
        gl.thread_barrier()
        q = qbuf.load(lhs)
        key = kbuf.permute((1, 0)).load(rhs)
        content = mma(q, key, content)
        gl.thread_barrier()
    rlayout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    qrlayout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    rrows = HeadTile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, qrlayout))
    qrdims = gl.arange(0, 64, layout=gl.SliceLayout(0, qrlayout))
    rtokens = gl.arange(0, 64, layout=gl.SliceLayout(1, rlayout))
    rdims = gl.arange(0, 64, layout=gl.SliceLayout(0, rlayout))
    qrbuf = gl.allocate_shared_memory(
        QR.dtype.element_ty, (BH, 64), gl.SwizzledSharedLayout(16, tm, 4, [1, 0])
    )
    krbuf = gl.allocate_shared_memory(
        KR.dtype.element_ty, (64, 64), gl.SwizzledSharedLayout(16, tn, 4, [1, 0])
    )
    rqmask = rrows[:, None] < H * SQ
    if H * SQ % BH == 0:
        rqmask = gl.full((BH, 1), True, gl.int1, qrlayout)
    rkmask = PageIndex * 64 + rtokens[:, None] < Length
    if FULL:
        rkmask = gl.full((64, 1), True, gl.int1, rlayout)
    async_copy.async_copy_global_to_shared(
        qrbuf,
        QR
        + Batch * QR_STRIDES[0]
        + (rrows // H)[:, None] * QR_STRIDES[1]
        + (rrows % H)[:, None] * QR_STRIDES[2]
        + qrdims[None, :],
        rqmask,
    )
    async_copy.async_copy_global_to_shared(
        krbuf,
        KR + Page * KR_STRIDES[0] + rtokens[:, None] * KR_STRIDES[1] + rdims[None, :],
        rkmask,
    )
    async_copy.commit_group()
    async_copy.wait_group(0)
    gl.thread_barrier()
    rope = mma(
        qrbuf.load(lhs),
        krbuf.permute((1, 0)).load(rhs),
        gl.full((BH, 64), 0.0, gl.float32, matrix),
    )
    return content.to(gl.float32) + rope


@gluon.jit
def dense_int8_scores_paged(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Scores,
    LogicalKS,
    Q_STRIDES: gl.constexpr,
    QR_STRIDES: gl.constexpr,
    KV_STRIDES: gl.constexpr,
    KR_STRIDES: gl.constexpr,
    QS_STRIDES: gl.constexpr,
    KS_STRIDES: gl.constexpr,
    TS: gl.constexpr,
    LS: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
    CAP: gl.constexpr,
    SCALE: gl.constexpr,
    CAUSAL: gl.constexpr,
    BH: gl.constexpr = 16,
):
    if BH == 64:
        matrix: gl.constexpr = MacaMmaLayout([2, 2], [2, 2, 16])
    elif BH == 32:
        matrix: gl.constexpr = MacaMmaLayout([2, 2], [1, 2, 16])
    else:
        matrix: gl.constexpr = MacaMmaLayout([1, 4], [1, 1, 16])
    page_index, head_tile, batch = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    heads = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, matrix))
    tokens = gl.arange(0, 64, layout=gl.SliceLayout(0, matrix))
    length = gl.load(Lengths + batch * LS)
    page = gl.load(Table + batch * TS + page_index, page_index * 64 < length, 0).to(
        gl.int64
    )
    if page_index * 64 < length:
        if page_index * 64 + 64 <= length:
            raw = score_body(
                Q,
                QR,
                KV,
                KR,
                page,
                batch,
                head_tile,
                length,
                page_index,
                Q_STRIDES,
                QR_STRIDES,
                KV_STRIDES,
                KR_STRIDES,
                H,
                SQ,
                BH,
                True,
            )
        else:
            raw = score_body(
                Q,
                QR,
                KV,
                KR,
                page,
                batch,
                head_tile,
                length,
                page_index,
                Q_STRIDES,
                QR_STRIDES,
                KV_STRIDES,
                KR_STRIDES,
                H,
                SQ,
                BH,
                False,
            )
        qs = gl.load(
            QS
            + batch * QS_STRIDES[0]
            + (heads // H) * QS_STRIDES[1]
            + (heads % H) * QS_STRIDES[2],
            heads < H * SQ,
            0,
        )
        ks = gl.load(
            KS + page * KS_STRIDES[0] + tokens * KS_STRIDES[1],
            page_index * 64 + tokens < length,
            0,
        )
        scores = raw * qs[:, None] * ks[None, :] * (SCALE * 1.4426950408889634)
        bound = gl.full((BH,), length, gl.int32, gl.SliceLayout(1, matrix))
        if CAUSAL:
            bound = gl.maximum(length - SQ + heads // H + 1, 0)
        scores = gl.where(
            page_index * 64 + tokens[None, :] < bound[:, None], scores, -float("inf")
        )
        offset = (
            (batch * H * SQ + heads[:, None]).to(gl.int64) * CAP
            + page_index * 64
            + tokens[None, :]
        )
        gl.store(Scores + offset, scores, heads[:, None] < H * SQ)
        if head_tile == 0:
            gl.store(LogicalKS + batch * CAP + page_index * 64 + tokens, ks)
    else:
        pass


@gluon.jit
def dense_int8_scores_grouped(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Scores,
    LogicalKS,
    MaxStats,
    Q_STRIDES: gl.constexpr,
    QR_STRIDES: gl.constexpr,
    KV_STRIDES: gl.constexpr,
    KR_STRIDES: gl.constexpr,
    QS_STRIDES: gl.constexpr,
    KS_STRIDES: gl.constexpr,
    TS: gl.constexpr,
    LS: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
    CAP: gl.constexpr,
    SCALE: gl.constexpr,
    CAUSAL: gl.constexpr,
    BH: gl.constexpr = 16,
    SPLITS: gl.constexpr = 256,
    WITH_STATS: gl.constexpr = True,
):
    group, head_tile, batch = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    length = gl.load(Lengths + batch * LS)
    if group * 128 < length:
        if BH >= 64:
            tm: gl.constexpr = 2
            tn: gl.constexpr = 4
            matrix: gl.constexpr = MacaMmaLayout([2, 2], [2, 4, 16])
        elif BH == 32:
            tm: gl.constexpr = 1
            tn: gl.constexpr = 4
            matrix: gl.constexpr = MacaMmaLayout([2, 2], [1, 4, 16])
        else:
            tm: gl.constexpr = 1
            tn: gl.constexpr = 2
            matrix: gl.constexpr = MacaMmaLayout([1, 4], [1, 2, 16])
        lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 0)
        rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 0)
        layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
        rows = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, layout))
        tokens = gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
        dims = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
        page0 = gl.load(Table + batch * TS + group * 2).to(gl.int64)
        page1 = gl.load(
            Table + batch * TS + group * 2 + 1, group * 128 + 64 < length, 0
        ).to(gl.int64)
        pages = gl.where(tokens < 64, page0, page1)
        qbuf = gl.allocate_shared_memory(
            gl.int8, (BH, 128), gl.SwizzledSharedLayout(16, tm, 8, [1, 0])
        )
        kbuf = gl.allocate_shared_memory(
            gl.int8, (128, 128), gl.SwizzledSharedLayout(16, tn, 8, [1, 0])
        )
        content = gl.full((BH, 128), 0, gl.int32, matrix)
        for chunk in gl.static_range(4):
            async_copy.async_copy_global_to_shared(
                qbuf,
                Q
                + batch * Q_STRIDES[0]
                + (rows // H)[:, None] * Q_STRIDES[1]
                + (rows % H)[:, None] * Q_STRIDES[2]
                + (chunk * 128 + dims)[None, :],
                rows[:, None] < H * SQ,
            )
            async_copy.async_copy_global_to_shared(
                kbuf,
                KV
                + pages[:, None] * KV_STRIDES[0]
                + (tokens % 64)[:, None] * KV_STRIDES[1]
                + (chunk * 128 + dims)[None, :],
                group * 128 + tokens[:, None] < length,
            )
            async_copy.commit_group()
            async_copy.wait_group(0)
            content = mma(qbuf.load(lhs), kbuf.permute((1, 0)).load(rhs), content)
            gl.thread_barrier()
        rlayout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
        rrows = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, rlayout))
        rtokens = gl.arange(0, 128, layout=gl.SliceLayout(1, rlayout))
        rdims = gl.arange(0, 64, layout=gl.SliceLayout(0, rlayout))
        rpages = gl.where(rtokens < 64, page0, page1)
        qrbuf = qbuf._reinterpret(
            QR.dtype.element_ty, (BH, 64), gl.SwizzledSharedLayout(16, tm, 4, [1, 0])
        )
        krbuf = kbuf._reinterpret(
            KR.dtype.element_ty, (128, 64), gl.SwizzledSharedLayout(16, tn, 4, [1, 0])
        )
        async_copy.async_copy_global_to_shared(
            qrbuf,
            QR
            + batch * QR_STRIDES[0]
            + (rrows // H)[:, None] * QR_STRIDES[1]
            + (rrows % H)[:, None] * QR_STRIDES[2]
            + rdims[None, :],
            rrows[:, None] < H * SQ,
        )
        async_copy.async_copy_global_to_shared(
            krbuf,
            KR
            + rpages[:, None] * KR_STRIDES[0]
            + (rtokens % 64)[:, None] * KR_STRIDES[1]
            + rdims[None, :],
            group * 128 + rtokens[:, None] < length,
        )
        async_copy.commit_group()
        async_copy.wait_group(0)
        rope = mma(
            qrbuf.load(lhs),
            krbuf.permute((1, 0)).load(rhs),
            gl.full((BH, 128), 0.0, gl.float32, matrix),
        )
        heads = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, matrix))
        out_tokens = gl.arange(0, 128, layout=gl.SliceLayout(0, matrix))
        out_pages = gl.where(out_tokens < 64, page0, page1)
        qs = gl.load(
            QS
            + batch * QS_STRIDES[0]
            + (heads // H) * QS_STRIDES[1]
            + (heads % H) * QS_STRIDES[2],
            heads < H * SQ,
            0.0,
        )
        ks = gl.load(
            KS + out_pages * KS_STRIDES[0] + (out_tokens % 64) * KS_STRIDES[1],
            group * 128 + out_tokens < length,
            0.0,
        )
        scores = (
            (content.to(gl.float32) + rope)
            * qs[:, None]
            * ks[None, :]
            * (SCALE * 1.4426950408889634)
        )
        bound = gl.full((BH,), length, gl.int32, gl.SliceLayout(1, matrix))
        if CAUSAL:
            bound = gl.maximum(length - SQ + heads // H + 1, 0)
        scores = gl.where(
            group * 128 + out_tokens[None, :] < bound[:, None], scores, -float("inf")
        )
        if WITH_STATS:
            maxima = gl.max(scores, 1)
            safe = gl.where(maxima == -float("inf"), 0.0, maxima)
            weighted = gl.max(gl.exp2(scores - safe[:, None]) * ks[None, :], 1)
            stats_offset = ((batch * H * SQ + heads) * SPLITS + group) * 2
            gl.store(MaxStats + stats_offset, maxima, heads < H * SQ)
            gl.store(MaxStats + stats_offset + 1, weighted, heads < H * SQ)
        offsets = (
            (batch * H * SQ + heads[:, None]).to(gl.int64) * CAP
            + group * 128
            + out_tokens[None, :]
        )
        gl.store(
            Scores + offsets,
            scores,
            (heads[:, None] < H * SQ) & (group * 128 + out_tokens[None, :] < CAP),
        )
        if head_tile == 0:
            gl.store(
                LogicalKS + batch * CAP + group * 128 + out_tokens,
                ks,
                group * 128 + out_tokens < CAP,
            )
    elif WITH_STATS:
        empty_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
        empty_heads = head_tile * BH + gl.arange(0, BH, layout=empty_layout)
        empty_offset = ((batch * H * SQ + empty_heads) * SPLITS + group) * 2
        gl.store(MaxStats + empty_offset, -float("inf"), empty_heads < H * SQ)
        gl.store(MaxStats + empty_offset + 1, 0.0, empty_heads < H * SQ)
    else:
        pass


@triton.jit
def probability_row(
    Scores,
    LogicalKS,
    PH,
    PL,
    PS,
    LSE,
    H: tl.constexpr,
    SQ: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    row,
    length,
):
    aligned_length = tl.cdiv(length, 16) * 16
    token = tl.arange(0, BLOCK)
    scores = tl.load(
        Scores + row.to(tl.int64) * CAP + token, token < aligned_length, -float("inf")
    )
    maximum = tl.max(scores, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    probabilities = tl.exp2(scores - safe_maximum)
    denominator = tl.sum(probabilities, 0)
    ks = tl.load(
        LogicalKS + (row // (H * SQ)).to(tl.int64) * CAP + token,
        token < aligned_length,
        0,
    )
    scaled = probabilities * ks
    scale = tl.maximum(tl.max(scaled, 0) / 16129.0, 1.0e-30)
    normalized = scaled / scale
    quantized = (normalized + 0.5).to(tl.int32)
    high = (normalized * (1.0 / 127.0) + 0.5).to(tl.int32)
    low = quantized - high * 127
    tl.store(
        PH + row.to(tl.int64) * CAP + token, high.to(tl.int8), token < aligned_length
    )
    tl.store(
        PL + row.to(tl.int64) * CAP + token, low.to(tl.int8), token < aligned_length
    )
    tl.store(PS + row, scale / tl.where(denominator > 0, denominator, 1.0))
    batch = row // (H * SQ)
    head = row % (H * SQ)
    logsum = tl.where(
        denominator > 0,
        maximum * 0.6931471805599453 + tl.log(denominator),
        float("inf"),
    )
    tl.store(LSE + (batch * H + head % H) * SQ + head // H, logsum)


@libentry()
@triton.jit
def dense_int8_probabilities(
    Scores,
    LogicalKS,
    PH,
    PL,
    PS,
    LSE,
    H: tl.constexpr,
    SQ: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    Lengths,
    LS: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(Lengths + (row // (H * SQ)) * LS)
    if CAP <= 4096:
        probability_row(
            Scores, LogicalKS, PH, PL, PS, LSE, H, SQ, CAP, BLOCK, row, length
        )
    elif length <= 4096:
        probability_row(
            Scores, LogicalKS, PH, PL, PS, LSE, H, SQ, CAP, 4096, row, length
        )
    elif CAP <= 8192:
        probability_row(
            Scores, LogicalKS, PH, PL, PS, LSE, H, SQ, CAP, BLOCK, row, length
        )
    elif length <= 8192:
        probability_row(
            Scores, LogicalKS, PH, PL, PS, LSE, H, SQ, CAP, 8192, row, length
        )
    else:
        probability_row(
            Scores, LogicalKS, PH, PL, PS, LSE, H, SQ, CAP, BLOCK, row, length
        )


@libentry()
@triton.jit
def dense_int8_probability_merge(
    Stats,
    Metadata,
    PS,
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
    LOG_STATS: tl.constexpr,
):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    splits = tl.arange(0, SPLITS)
    offsets = (rows[:, None] * SPLITS + splits[None, :]) * 2
    maxima = tl.load(Stats + offsets, rows[:, None] < ROWS, -float("inf"))
    weighted_maxima = tl.load(Stats + offsets + 1, rows[:, None] < ROWS, 0.0)
    maximum = tl.max(maxima, 1)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    if LOG_STATS:
        weighted = tl.max(weighted_maxima + (maxima - safe[:, None]), 1)
        scale = tl.maximum(tl.exp2(weighted - 13.977369373544331), 1.0e-30)
    else:
        weights = tl.exp2(maxima - safe[:, None])
        weighted = tl.max(weighted_maxima * weights, 1)
        scale = tl.maximum(weighted * (1.0 / 16129.0), 1.0e-30)
    tl.store(Metadata + rows * 2, safe, rows < ROWS)
    tl.store(Metadata + rows * 2 + 1, 1.0 / scale, rows < ROWS)
    tl.store(PS + rows, scale, rows < ROWS)
    if ADAPTIVE and tl.program_id(0) == 0:
        batch = tl.arange(0, LENGTH_BLOCK)
        lengths = tl.load(Lengths + batch * LS, batch < B, 0)
        maximum_length = tl.max(lengths, 0)
        minimum_length = tl.min(tl.where(batch < B, lengths, 2147483647), 0)
        tl.store(SplitFlag, maximum_length != minimum_length)


@gluon.jit
def dense_int8_values_prequantized(
    KV,
    Table,
    Lengths,
    PH,
    PL,
    PS,
    Output,
    KV_STRIDES: gl.constexpr,
    TS: gl.constexpr,
    LS: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
    CAP: gl.constexpr,
    BH: gl.constexpr = 32,
):
    value_tile, head_tile, batch = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    matrix: gl.constexpr = MacaMmaLayout([4, 1], [4, 1, 16])
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 0)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 0)
    load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    tokens = gl.arange(0, 64, layout=gl.SliceLayout(1, load_layout))
    dims = value_tile * 256 + gl.arange(0, 256, layout=gl.SliceLayout(0, load_layout))
    probability_layout: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])
    p_tokens = gl.arange(0, 64, layout=gl.SliceLayout(1, probability_layout))
    packed_heads = gl.arange(0, 2 * BH, layout=gl.SliceLayout(0, probability_layout))
    heads = head_tile * BH + packed_heads % BH
    length = gl.load(Lengths + batch * LS)
    accumulator = gl.full((256, 2 * BH), 0, gl.int32, matrix)
    buffer = gl.allocate_shared_memory(
        gl.int8, (64, 256), gl.SwizzledSharedLayout(64, 16, 2, [1, 0])
    )
    p_buffer = gl.allocate_shared_memory(
        gl.int8, (64, 2 * BH), gl.SwizzledSharedLayout(16, 2, 4, [0, 1])
    )
    for page_index in range(gl.cdiv(length, 64)):
        page = gl.load(Table + batch * TS + page_index).to(gl.int64)
        async_copy.async_copy_global_to_shared(
            buffer,
            KV + page * KV_STRIDES[0] + tokens[:, None] * KV_STRIDES[1] + dims[None, :],
            page_index * 64 + tokens[:, None] < length,
        )
        logical = page_index * 64 + p_tokens
        offsets = (batch * H * SQ + heads[None, :]) * CAP + logical[:, None]
        pointer = gl.where(packed_heads[None, :] < BH, PH + offsets, PL + offsets)
        if page_index * 64 + 64 <= length:
            async_copy.async_copy_global_to_shared(
                p_buffer, pointer, heads[None, :] < H * SQ
            )
        else:
            probability = gl.load(
                pointer, (heads[None, :] < H * SQ) & (logical[:, None] < length), 0
            )
            p_buffer.store(probability)
        async_copy.commit_group()
        async_copy.wait_group(0)
        gl.thread_barrier()
        probability = p_buffer.load(rhs)
        value = buffer.permute((1, 0)).load(lhs)
        accumulator = mma(value, probability, accumulator)
        gl.thread_barrier()
    high, low = gl.split(
        gl.permute(gl.reshape(accumulator.to(gl.float32), (256, 2, BH)), (0, 2, 1))
    )
    out_layout: gl.constexpr = high.type.layout
    source_heads = head_tile * BH + gl.arange(
        0, BH, layout=gl.SliceLayout(0, out_layout)
    )
    scale = gl.load(PS + batch * H * SQ + source_heads, source_heads < H * SQ, 0)
    output = (high * 127.0 + low) * scale[None, :]
    store_layout: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 4], [0, 1])
    out_dims = value_tile * 256 + gl.arange(
        0, 256, layout=gl.SliceLayout(1, store_layout)
    )
    out_heads = head_tile * BH + gl.arange(
        0, BH, layout=gl.SliceLayout(0, store_layout)
    )
    pointer = Output + (batch * H * SQ + out_heads[None, :]) * 512 + out_dims[:, None]
    output = gl.convert_layout(output.to(Output.dtype.element_ty), pointer.type.layout)
    gl.store(pointer, output, out_heads[None, :] < H * SQ)


@gluon.jit
def dense_int8_values(
    KV,
    Table,
    Lengths,
    Scores,
    LogicalKS,
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
    BH: gl.constexpr = 32,
    SPLITS: gl.constexpr = 4,
    ADAPTIVE: gl.constexpr = True,
    PAGE_TILED: gl.constexpr = False,
):
    value_tile, head_tile, job = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    if ADAPTIVE:
        split = job // B
        batch = job % B
    else:
        split: gl.constexpr = 0
        batch = job
    matrix: gl.constexpr = MacaMmaLayout([4, 1], [4, 1, 16])
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 0)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 0)
    load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    tokens = gl.arange(0, 64, layout=gl.SliceLayout(1, load_layout))
    dims = value_tile * 256 + gl.arange(0, 256, layout=gl.SliceLayout(0, load_layout))
    probability_layout: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])
    p_tokens = gl.arange(0, 64, layout=gl.SliceLayout(1, probability_layout))
    p_heads = head_tile * BH + gl.arange(
        0, BH, layout=gl.SliceLayout(0, probability_layout)
    )
    rows = batch * H * SQ + p_heads
    maximum = gl.load(Metadata + rows * 2, p_heads < H * SQ, 0.0)
    inverse_scale = gl.load(Metadata + rows * 2 + 1, p_heads < H * SQ, 0.0)
    denominator_matrix = gl.full((64, BH), 0.0, gl.float32, probability_layout)
    length = gl.load(Lengths + batch * LS)
    if ADAPTIVE:
        split_count = gl.where(gl.load(SplitFlag) != 0, SPLITS, 1)
    else:
        split_count: gl.constexpr = 1
    if split < split_count:
        high_accumulator = gl.full((256, BH), 0, gl.int32, matrix)
        low_accumulator = gl.full((256, BH), 0, gl.int32, matrix)
        buffer = gl.allocate_shared_memory(
            gl.int8, (64, 256), gl.SwizzledSharedLayout(64, 16, 2, [1, 0])
        )
        high_buffer = gl.allocate_shared_memory(
            gl.int8, (64, BH), gl.SwizzledSharedLayout(16, 2, 4, [0, 1])
        )
        low_buffer = gl.allocate_shared_memory(
            gl.int8, (64, BH), gl.SwizzledSharedLayout(16, 2, 4, [0, 1])
        )
        pages = gl.cdiv(length, 64)
        for page_index in range(
            pages * split // split_count, pages * (split + 1) // split_count
        ):
            page = gl.load(Table + batch * TS + page_index).to(gl.int64)
            async_copy.async_copy_global_to_shared(
                buffer,
                KV
                + page * KV_STRIDES[0]
                + tokens[:, None] * KV_STRIDES[1]
                + dims[None, :],
                page_index * 64 + tokens[:, None] < length,
            )
            async_copy.commit_group()
            logical = page_index * 64 + p_tokens
            if PAGE_TILED:
                score_offsets = (
                    (batch.to(gl.int64) * CAP + page_index * 64) * H * SQ
                    + p_heads[None, :] * 64
                    + p_tokens[:, None]
                )
            else:
                score_offsets = rows[None, :].to(gl.int64) * CAP + logical[:, None]
            scores = gl.load(
                Scores + score_offsets,
                p_heads[None, :] < H * SQ,
                -float("inf"),
            )
            ks = gl.load(LogicalKS + batch * CAP + logical)
            probability = gl.exp2(scores - maximum[None, :])
            denominator_matrix += probability
            normalized = probability * ks[:, None] * inverse_scale[None, :]
            integer = (normalized + 0.5).to(gl.int32)
            packed_high, packed_low = pack_probability_terms(integer)
            high_words = high_buffer._reinterpret(
                gl.int32, (16, BH), gl.SwizzledSharedLayout(4, 2, 4, [0, 1])
            )
            low_words = low_buffer._reinterpret(
                gl.int32, (16, BH), gl.SwizzledSharedLayout(4, 2, 4, [0, 1])
            )
            high_words.store(packed_high)
            low_words.store(packed_low)
            async_copy.wait_group(0)
            gl.thread_barrier()
            value = buffer.permute((1, 0)).load(lhs)
            high_accumulator = mma(value, high_buffer.load(rhs), high_accumulator)
            low_accumulator = mma(value, low_buffer.load(rhs), low_accumulator)
            gl.thread_barrier()
        heads = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(0, matrix))
        out_dims = value_tile * 256 + gl.arange(
            0, 256, layout=gl.SliceLayout(1, matrix)
        )
        raw = high_accumulator.to(gl.float32) * 128.0 + low_accumulator.to(gl.float32)
        if split_count == 1:
            scale = gl.load(PS + batch * H * SQ + heads, heads < H * SQ, 0.0)
            denominator = gl.sum(denominator_matrix, 0)
            divisor = gl.convert_layout(
                gl.where(denominator > 0, denominator, 1.0), gl.SliceLayout(0, matrix)
            )
            output = raw * (scale / divisor)[None, :]
            store_layout: gl.constexpr = gl.BlockedLayout(
                [8, 1], [16, 4], [1, 4], [0, 1]
            )
            store_dims = value_tile * 256 + gl.arange(
                0, 256, layout=gl.SliceLayout(1, store_layout)
            )
            store_heads = head_tile * BH + gl.arange(
                0, BH, layout=gl.SliceLayout(0, store_layout)
            )
            output_pointer = (
                Output
                + (batch * H * SQ + store_heads[None, :]) * 512
                + store_dims[:, None]
            )
            output = gl.convert_layout(output.to(Output.dtype.element_ty), store_layout)
            gl.store(output_pointer, output, store_heads[None, :] < H * SQ)
            if value_tile == 0:
                lse = gl.where(
                    denominator > 0,
                    maximum * 0.6931471805599453 + gl.log(denominator),
                    float("inf"),
                )
                gl.store(
                    LSE + (batch * H + p_heads % H) * SQ + p_heads // H,
                    lse,
                    p_heads < H * SQ,
                )
        else:
            offset = ((split * B + batch) * H * SQ + heads[None, :]).to(
                gl.int64
            ) * 512 + out_dims[:, None]
            gl.store(Partial + offset, raw, heads[None, :] < H * SQ)
            if value_tile == 0:
                denominator = gl.sum(denominator_matrix, 0)
                gl.store(
                    PartialD + (split * B + batch) * H * SQ + p_heads,
                    denominator,
                    p_heads < H * SQ,
                )


@libentry()
@triton.jit
def dense_int8_value_merge(
    Partial,
    PartialD,
    Metadata,
    Scale,
    Output,
    LSE,
    SplitFlag,
    B: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    SPLITS: tl.constexpr,
):
    if tl.load(SplitFlag) != 0:
        batch, head = tl.program_id(0), tl.program_id(1)
        splits = tl.arange(0, SPLITS)
        dims = tl.arange(0, 512)
        values = tl.load(
            Partial
            + ((splits[:, None] * B + batch) * H * SQ + head) * 512
            + dims[None, :]
        )
        denominator = tl.sum(
            tl.load(PartialD + (splits * B + batch) * H * SQ + head), 0
        )
        scale = tl.load(Scale + batch * H * SQ + head)
        output = tl.sum(values, 0) * (
            scale / tl.where(denominator > 0, denominator, 1.0)
        )
        tl.store(Output + (batch * H * SQ + head) * 512 + dims, output)
        maximum = tl.load(Metadata + (batch * H * SQ + head) * 2)
        lse = tl.where(
            denominator > 0,
            maximum * 0.6931471805599453 + tl.log(denominator),
            float("inf"),
        )
        tl.store(LSE + (batch * H + head % H) * SQ + head // H, lse)


@gluon.jit
def dense_int8_mla_smallheads(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Partial,
    Stats,
    Output,
    LSE,
    Q_STRIDES: gl.constexpr,
    QR_STRIDES: gl.constexpr,
    KV_STRIDES: gl.constexpr,
    KR_STRIDES: gl.constexpr,
    QS_STRIDES: gl.constexpr,
    KS_STRIDES: gl.constexpr,
    TABLE_STRIDE: gl.constexpr,
    LENGTH_STRIDE: gl.constexpr,
    B: gl.constexpr,
    H: gl.constexpr,
    SPLITS: gl.constexpr,
    SCALE: gl.constexpr,
    SQ: gl.constexpr,
    CAUSAL: gl.constexpr,
    BH: gl.constexpr,
):
    gl.static_assert(BH == 16)
    W: gl.constexpr = 4
    qk: gl.constexpr = MacaMmaLayout([1, 4], [1, 1, 16], col_major=True)
    qlhs: gl.constexpr = gl.DotOperandLayout(0, qk, 0)
    qrhs: gl.constexpr = gl.DotOperandLayout(1, qk, 0)
    pv: gl.constexpr = MacaMmaLayout([1, 4], [1, 2, 16], col_major=True)
    plhs: gl.constexpr = gl.DotOperandLayout(0, pv, 0)
    vrhs: gl.constexpr = gl.DotOperandLayout(1, pv, 0)
    copy_layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [W, 1], [1, 0])
    rope_copy: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [W, 1], [1, 0])
    split_score: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 0, 1], [0, 0, 2]],
        [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 4], [0, 0, 8]],
        [[0, 1, 0], [0, 2, 0]],
        [],
        [BH, 4, 16],
    )
    all_partials: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [],
        [BH, 4],
    )
    _, head_tile, job = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    batch, split = job % B, job // B
    qcopy: gl.constexpr = gl.BlockedLayout([1, 16], [2, 32], [W, 1], [1, 0])
    qload_heads = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, qcopy))
    qload_dims = gl.arange(0, 512, layout=gl.SliceLayout(0, qcopy))
    rload_heads = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, rope_copy))
    rload_dims = gl.arange(0, 64, layout=gl.SliceLayout(0, rope_copy))
    qbuf = gl.allocate_shared_memory(
        gl.int8, (BH, 512), gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    )
    qrbuf = gl.allocate_shared_memory(
        QR.dtype.element_ty, (BH, 64), gl.SwizzledSharedLayout(16, 1, 4, [1, 0])
    )
    async_copy.async_copy_global_to_shared(
        qbuf,
        Q
        + batch * Q_STRIDES[0]
        + (qload_heads // H)[:, None] * Q_STRIDES[1]
        + (qload_heads % H)[:, None] * Q_STRIDES[2]
        + qload_dims[None, :],
        qload_heads[:, None] < H * SQ,
    )
    async_copy.async_copy_global_to_shared(
        qrbuf,
        QR
        + batch * QR_STRIDES[0]
        + (rload_heads // H)[:, None] * QR_STRIDES[1]
        + (rload_heads % H)[:, None] * QR_STRIDES[2]
        + rload_dims[None, :],
        rload_heads[:, None] < H * SQ,
    )
    async_copy.commit_group()
    tokens = gl.arange(0, 64, layout=gl.SliceLayout(1, copy_layout))
    dims = gl.arange(0, 128, layout=gl.SliceLayout(0, copy_layout))
    rtokens = gl.arange(0, 64, layout=gl.SliceLayout(1, rope_copy))
    rdims = gl.arange(0, 64, layout=gl.SliceLayout(0, rope_copy))
    heads = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, qk))
    out_tokens = gl.arange(0, 64, layout=gl.SliceLayout(0, qk))
    qs = gl.load(
        QS
        + batch * QS_STRIDES[0]
        + (heads // H) * QS_STRIDES[1]
        + (heads % H) * QS_STRIDES[2],
        heads < H * SQ,
        0.0,
    )
    length = gl.load(Lengths + batch * LENGTH_STRIDE)
    bound = gl.full((BH,), length, gl.int32, heads.type.layout)
    if CAUSAL:
        bound = gl.maximum(length - SQ + heads // H + 1, 0)
    # Retain four value fragments in registers so KR/stats/P share one 8KiB plane.
    plane0 = gl.allocate_shared_memory(
        gl.int8, (64, 128), gl.SwizzledSharedLayout(16, 2, 8, [1, 0])
    )
    krbuf = plane0._reinterpret(
        KR.dtype.element_ty, (64, 64), gl.SwizzledSharedLayout(16, 2, 4, [1, 0])
    )
    stats_full = plane0._reinterpret(
        gl.float32, (BH, 2048 // BH), gl.SwizzledSharedLayout(1, 1, 1, [0, 1])
    )
    pfull = plane0._reinterpret(
        gl.int8, (64, 128), gl.SwizzledSharedLayout(16, 2, 4, [0, 1])
    )
    pwords = pfull._reinterpret(
        gl.int32, (16, 128), gl.SwizzledSharedLayout(4, 2, 4, [0, 1])
    )
    p_high = pfull.slice(0, BH, dim=1)
    p_low = pfull.slice(BH, BH, dim=1)
    maximum = gl.full((BH,), -float("inf"), gl.float32, heads.type.layout)
    denominator = gl.full((BH,), 0.0, gl.float32, heads.type.layout)
    acc0 = gl.full((BH, 128), 0.0, gl.float32, pv)
    acc1 = gl.full((BH, 128), 0.0, gl.float32, pv)
    acc2 = gl.full((BH, 128), 0.0, gl.float32, pv)
    acc3 = gl.full((BH, 128), 0.0, gl.float32, pv)
    pages = gl.cdiv(length, 64)
    for page_index in range(pages * split // SPLITS, pages * (split + 1) // SPLITS):
        page = gl.load(Table + batch * TABLE_STRIDE + page_index).to(gl.int64)
        content = gl.full((BH, 64), 0, gl.int32, qk)
        for chunk in gl.static_range(4):
            buffer = plane0
            async_copy.async_copy_global_to_shared(
                buffer,
                KV
                + page * KV_STRIDES[0]
                + tokens[:, None] * KV_STRIDES[1]
                + (chunk * 128 + dims)[None, :],
                page_index * 64 + tokens[:, None] < length,
            )
            async_copy.commit_group()
            async_copy.wait_group(0)
            query = qbuf.slice(chunk * 128, 128, dim=1).load(qlhs)
            content = mma(query, buffer.permute((1, 0)).load(qrhs), content)
            retained = plane0.load(vrhs)
            if chunk == 0:
                retained0 = retained
            elif chunk == 1:
                retained1 = retained
            elif chunk == 2:
                retained2 = retained
            else:
                retained3 = retained
            gl.thread_barrier()
        async_copy.async_copy_global_to_shared(
            krbuf,
            KR
            + page * KR_STRIDES[0]
            + rtokens[:, None] * KR_STRIDES[1]
            + rdims[None, :],
            page_index * 64 + rtokens[:, None] < length,
        )
        async_copy.commit_group()
        query_rope = qrbuf.load(qlhs)
        async_copy.wait_group(0)
        rope = mma(
            query_rope,
            krbuf.permute((1, 0)).load(qrhs),
            gl.full((BH, 64), 0.0, gl.float32, qk),
        )
        gl.thread_barrier()
        ks = gl.load(
            KS + page * KS_STRIDES[0] + out_tokens * KS_STRIDES[1],
            page_index * 64 + out_tokens < length,
            0.0,
        )
        scores = (
            (content.to(gl.float32) + rope)
            * qs[:, None]
            * ks[None, :]
            * (SCALE * 1.4426950408889634)
        )
        scores = gl.where(
            (heads[:, None] < H * SQ)
            & (page_index * 64 + out_tokens[None, :] < bound[:, None]),
            scores,
            -float("inf"),
        )
        local_scores = gl.convert_layout(gl.reshape(scores, (BH, 4, 16)), split_score)
        stats_full.slice(0, 4, dim=1).store(gl.max(local_scores, 2))
        gl.thread_barrier()
        page_maximum = gl.max(stats_full.slice(0, 4, dim=1).load(all_partials), 1)
        page_maximum = gl.convert_layout(page_maximum, heads.type.layout)
        next_maximum = gl.maximum(maximum, page_maximum)
        safe = gl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        probability = gl.exp2(scores - safe[:, None])
        scaled = probability * ks[None, :]
        split_p = gl.convert_layout(gl.reshape(probability, (BH, 4, 16)), split_score)
        split_scaled = gl.convert_layout(gl.reshape(scaled, (BH, 4, 16)), split_score)
        partial_stats = gl.reshape(
            gl.permute(gl.join(gl.sum(split_p, 2), gl.max(split_scaled, 2)), (0, 2, 1)),
            (BH, 8),
        )
        stats_full.slice(8, 8, dim=1).store(partial_stats)
        gl.thread_barrier()
        page_sum = gl.convert_layout(
            gl.sum(stats_full.slice(8, 4, dim=1).load(all_partials), 1),
            heads.type.layout,
        )
        scale = gl.convert_layout(
            gl.max(stats_full.slice(12, 4, dim=1).load(all_partials), 1),
            heads.type.layout,
        )
        scale = gl.maximum(scale * (1.0 / 16129.0), 1.0e-30)
        alpha = gl.exp2(maximum - safe)
        next_denominator = denominator * alpha + page_sum
        inverse_d = 1.0 / gl.where(next_denominator > 0, next_denominator, 1.0)
        output_alpha = denominator * alpha * inverse_d
        output_scale = scale * inverse_d
        out_alpha = gl.convert_layout(output_alpha, gl.SliceLayout(1, pv))
        out_scale = gl.convert_layout(output_scale, gl.SliceLayout(1, pv))
        inverse_scale = 1.0 / scale
        integer = (scaled * inverse_scale[:, None] + 0.5).to(gl.int32)
        packed_high, packed_low = pack_probability_terms(gl.permute(integer, (1, 0)))
        gl.thread_barrier()
        joined_words = gl.reshape(
            gl.permute(gl.join(packed_high, packed_low), (0, 2, 1)), (16, 2 * BH)
        )
        pwords.slice(0, 2 * BH, dim=1).store(joined_words)
        gl.thread_barrier()
        for chunk in gl.static_range(4):
            if chunk == 0:
                value = retained0
                accumulator = acc0
            elif chunk == 1:
                value = retained1
                accumulator = acc1
            elif chunk == 2:
                value = retained2
                accumulator = acc2
            else:
                value = retained3
                accumulator = acc3
            high = mma(
                p_high.permute((1, 0)).load(plhs),
                value,
                gl.full((BH, 128), 0, gl.int32, pv),
            )
            # A 64-token INT8 dot scaled by 14-bit probabilities fits INT32.
            combined = mma(p_low.permute((1, 0)).load(plhs), value, high << 7)
            accumulator = (
                accumulator * out_alpha[:, None]
                + combined.to(gl.float32) * out_scale[:, None]
            )
            # Bound live temporary MMA results across the four value fragments.
            accumulator = gl.inline_asm_elementwise(
                "",
                constraints="=r,0",
                args=[accumulator.to(gl.int32, bitcast=True)],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            ).to(gl.float32, bitcast=True)
            if chunk == 0:
                acc0 = accumulator
            elif chunk == 1:
                acc1 = accumulator
            elif chunk == 2:
                acc2 = accumulator
            else:
                acc3 = accumulator
        maximum = next_maximum
        denominator = next_denominator
        gl.thread_barrier()
    async_copy.wait_group(0)
    out_heads = head_tile * BH + gl.arange(0, BH, layout=gl.SliceLayout(1, pv))
    local_dims = gl.arange(0, 128, layout=gl.SliceLayout(0, pv))
    for chunk in gl.static_range(4):
        if chunk == 0:
            accumulator = acc0
        elif chunk == 1:
            accumulator = acc1
        elif chunk == 2:
            accumulator = acc2
        else:
            accumulator = acc3
        out_dims = chunk * 128 + local_dims
        if SPLITS == 1:
            gl.store(
                Output
                + (batch * H * SQ + out_heads[:, None]) * 512
                + out_dims[None, :],
                accumulator,
                out_heads[:, None] < H * SQ,
            )
        else:
            gl.store(
                Partial
                + ((split * B + batch) * H * SQ + out_heads[:, None]) * 512
                + out_dims[None, :],
                accumulator,
                out_heads[:, None] < H * SQ,
            )
    if SPLITS == 1:
        lse = gl.where(
            denominator > 0,
            maximum * 0.6931471805599453 + gl.log(denominator),
            float("inf"),
        )
        gl.store(LSE + (batch * H + heads % H) * SQ + heads // H, lse, heads < H * SQ)
    else:
        gl.store(
            Stats + (split * 2 * B + batch) * H * SQ + heads, maximum, heads < H * SQ
        )
        gl.store(
            Stats + ((split * 2 + 1) * B + batch) * H * SQ + heads,
            denominator,
            heads < H * SQ,
        )
