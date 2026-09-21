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

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.language.core as tlc

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

HAS_TLE = has_triton_tle(3, 6, 0)
if HAS_TLE:
    import triton.experimental.tle.language as tle
    from triton.experimental.tle.language.gpu import types as tle_types

    from flaggems_vllm.ops.flash_mla_ckv_fp8_per_token import (
        _publish_p_fp8_sw64_cuda_native_coupled_stmatrix,
    )
else:
    tle = None
    tle_types = None


@tlc.builtin
def sparse_smem_subslice(buf, offsets, shape, _semantic=None):
    offsets = [int(tlc._unwrap_if_constexpr(value)) for value in offsets]
    shape = [int(tlc._unwrap_if_constexpr(value)) for value in shape]
    view_type = tle_types.buffered_tensor_type(
        buf.dtype,
        shape,
        buf.type.storage,
        buf.type.layout,
        _semantic,
        alloc_shape=buf.type.alloc_shape,
    )
    handle = _semantic.builder.create_memdesc_subslice(
        view_type.to_ir(_semantic.builder), buf.handle, offsets
    )
    return tle_types.buffered_tensor(
        handle,
        buf.dtype,
        shape,
        buf.type.storage,
        buf.type.layout,
        _semantic,
        alloc_shape=buf.type.alloc_shape,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("flash_mla_sparse_fwd_w8a8_fp8"),
    key=["B", "H", "TOPK", "SPLITS"],
)
@triton.jit
def sparse_fp8_kernel(
    Q,
    QRope,
    KV,
    KVRope,
    Indices,
    QScale,
    KVScale,
    Sink,
    Length,
    Output,
    LSE,
    Partial,
    Stats,
    stride_qb,
    stride_qh,
    stride_qrb,
    stride_qrh,
    stride_kvp,
    stride_kvt,
    stride_krp,
    stride_krt,
    stride_ib,
    stride_ik,
    stride_qsb,
    stride_qsh,
    stride_ksp,
    stride_kst,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    SM_SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    split = tl.program_id(2)
    dims = tl.arange(0, 256)
    keys = tl.arange(0, BLOCK_K)
    q_ptr = Q + batch * stride_qb + heads[:, None] * stride_qh + dims[None, :]
    query0 = tl.load(q_ptr, heads[:, None] < H, 0.0)
    query1 = tl.load(q_ptr + 256, heads[:, None] < H, 0.0)
    rope_dims = tl.arange(0, 64)
    query_rope = tl.load(
        QRope + batch * stride_qrb + heads[:, None] * stride_qrh + rope_dims[None, :],
        heads[:, None] < H,
        0,
    )
    query_scale = tl.load(
        QScale + batch * stride_qsb + heads * stride_qsh, heads < H, 0
    )
    maximum = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    denominator = tl.full((BLOCK_H,), 0, tl.float32)
    value0 = tl.full((BLOCK_H, 256), 0, tl.float32)
    value1 = tl.full((BLOCK_H, 256), 0, tl.float32)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    blocks_per_split: tl.constexpr = triton.cdiv(TOPK, BLOCK_K * SPLITS)
    start = split * blocks_per_split * BLOCK_K
    stop = tl.minimum(start + blocks_per_split * BLOCK_K, length)
    for base in range(start, stop, BLOCK_K):
        positions = base + keys
        ids = tl.load(
            Indices + batch * stride_ib + positions * stride_ik, positions < length, -1
        )
        valid = (positions < length) & (ids >= 0) & (ids < N)
        ids = tl.where(valid, ids, 0).to(tl.int64)
        pages = ids // 64
        tokens = ids % 64
        kv_ptr = (
            KV
            + pages[None, :] * stride_kvp
            + tokens[None, :] * stride_kvt
            + dims[:, None]
        )
        key0 = tl.load(kv_ptr, valid[None, :], 0.0)
        key1 = tl.load(kv_ptr + 256, valid[None, :], 0.0)
        kv_scale = tl.load(KVScale + pages * stride_ksp + tokens * stride_kst, valid, 0)
        logits = tl.dot(query0, key0, out_dtype=tl.float32)
        logits = tl.dot(query1, key1, logits)
        key_rope = tl.load(
            KVRope
            + pages[None, :] * stride_krp
            + tokens[None, :] * stride_krt
            + rope_dims[:, None],
            valid[None, :],
            0,
        )
        rope_logits = tl.dot(query_rope, key_rope, out_dtype=tl.float32)
        logits += rope_logits
        logits = logits * query_scale[:, None] * kv_scale[None, :] * SM_SCALE
        logits = tl.where(valid[None, :], logits, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(logits, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp(maximum - safe_maximum)
        probabilities = tl.exp(logits - safe_maximum[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        # Fold token-dependent V scales into P so PV still uses FP8 operands.
        weighted = probabilities * kv_scale[None, :]
        probability_scale = tl.max(weighted, 1) / 448.0
        probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
        probability_fp8 = (weighted / probability_scale[:, None]).to(tl.float8e4nv)
        contribution0 = tl.dot(probability_fp8, tl.trans(key0), out_dtype=tl.float32)
        contribution1 = tl.dot(probability_fp8, tl.trans(key1), out_dtype=tl.float32)
        value0 = (
            value0 * correction[:, None] + contribution0 * probability_scale[:, None]
        )
        value1 = (
            value1 * correction[:, None] + contribution1 * probability_scale[:, None]
        )
        maximum = next_maximum
    if SPLITS == 1:
        logsum = tl.where(denominator > 0, maximum + tl.log(denominator), float("inf"))
        inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
        if HAS_SINK:
            sink = tl.load(Sink + heads, heads < H, 0)
            inverse *= tl.where(
                sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
            )
        output_ptr = Output + (batch * H + heads[:, None]) * 512 + dims[None, :]
        tl.store(output_ptr, value0 * inverse[:, None], heads[:, None] < H)
        tl.store(output_ptr + 256, value1 * inverse[:, None], heads[:, None] < H)
        tl.store(LSE + batch * H + heads, logsum, heads < H)
    else:
        partial_ptr = (
            Partial
            + ((batch * SPLITS + split) * H + heads[:, None]) * 512
            + dims[None, :]
        )
        tl.store(partial_ptr, value0, heads[:, None] < H)
        tl.store(partial_ptr + 256, value1, heads[:, None] < H)
        stats_ptr = Stats + ((batch * SPLITS + split) * H + heads) * 2
        tl.store(stats_ptr, maximum, heads < H)
        tl.store(stats_ptr + 1, denominator, heads < H)


@triton.jit
def sparse_fp8_merge(
    Partial,
    Stats,
    Sink,
    Output,
    LSE,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // H
    head = row % H
    split = tl.arange(0, SPLITS)
    dims = tl.arange(0, 512)
    stats_ptr = Stats + ((batch * SPLITS + split) * H + head) * 2
    maximum = tl.load(stats_ptr)
    denominator = tl.load(stats_ptr + 1)
    global_maximum = tl.max(maximum, 0)
    safe_maximum = tl.where(global_maximum == -float("inf"), 0.0, global_maximum)
    weights = tl.exp(maximum - safe_maximum)
    total = tl.sum(denominator * weights, 0)
    logsum = tl.where(total > 0, global_maximum + tl.log(total), float("inf"))
    inverse = tl.where(total > 0, 1.0 / total, 0.0)
    if HAS_SINK:
        sink = tl.load(Sink + head)
        inverse *= tl.where(
            sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
        )
    partial = tl.load(
        Partial + ((batch * SPLITS + split[:, None]) * H + head) * 512 + dims[None, :]
    )
    output = tl.sum(partial * weights[:, None], 0) * inverse
    tl.store(Output + row * 512 + dims, output)
    tl.store(LSE + row, logsum)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_fp8_scores"),
    key=["B", "H", "TOPK"],
)
@triton.jit
def sparse_fp8_scores(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Length,
    Scores,
    stride_qb,
    stride_qh,
    stride_qrb,
    stride_qrh,
    stride_kp,
    stride_kt,
    stride_krp,
    stride_krt,
    stride_qsb,
    stride_qsh,
    stride_ksp,
    stride_kst,
    stride_ib,
    stride_ik,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    PAD_K: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch, head_block, key_block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    heads = head_block * 64 + tl.arange(0, 64)
    dims = tl.arange(0, 512)
    ropes = tl.arange(0, 64)
    keys = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    ids = tl.load(Indices + batch * stride_ib + keys * stride_ik, keys < length, -1)
    valid = (ids >= 0) & (ids < N) & (keys < length)
    ids = tl.where(valid, ids, 0).to(tl.int64)
    pages, slots = ids // 64, ids % 64
    query = tl.load(Q + batch * stride_qb + heads[:, None] * stride_qh + dims[None, :])
    key = tl.load(
        KV + pages[None, :] * stride_kp + slots[None, :] * stride_kt + dims[:, None],
        valid[None, :],
        0.0,
    )
    qr = tl.load(QR + batch * stride_qrb + heads[:, None] * stride_qrh + ropes[None, :])
    kr = tl.load(
        KR + pages[None, :] * stride_krp + slots[None, :] * stride_krt + ropes[:, None],
        valid[None, :],
        0.0,
    )
    content = tl.dot(query, key, out_dtype=tl.float32)
    rope = tl.dot(qr, kr, out_dtype=tl.float32)
    qs = tl.load(QS + batch * stride_qsb + heads * stride_qsh)
    ks = tl.load(KS + pages * stride_ksp + slots * stride_kst, valid, 0)
    score = (content + rope) * qs[:, None] * ks[None, :] * SCALE
    score = tl.where(valid[None, :], score, -float("inf"))
    tl.store(
        Scores + (batch * H + heads[:, None]) * PAD_K + keys[None, :],
        score,
        keys[None, :] < PAD_K,
    )


@triton.jit
def sparse_fp8_probabilities(
    Scores,
    Indices,
    KS,
    Sink,
    Probabilities,
    Factors,
    LSE,
    stride_ib,
    stride_ik,
    stride_ksp,
    stride_kst,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    PAD_K: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    keys = tl.arange(0, BLOCK)
    scores = tl.load(
        Scores + row[:, None] * PAD_K + keys[None, :],
        keys[None, :] < TOPK,
        -float("inf"),
    )
    maximum = tl.max(scores, 1)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    probabilities = tl.exp(scores - safe_maximum[:, None])
    denominator = tl.sum(probabilities, 1)
    batch = tl.program_id(0) * BLOCK_ROWS // H
    ids = tl.load(Indices + batch * stride_ib + keys * stride_ik, keys < TOPK, -1)
    valid = (ids >= 0) & (ids < N) & (keys < TOPK)
    ids = tl.where(valid, ids, 0).to(tl.int64)
    scales = tl.load(KS + ids // 64 * stride_ksp + ids % 64 * stride_kst, valid, 0.0)
    weighted = probabilities * scales[None, :]
    probability_scale = tl.max(weighted, 1) / 448.0
    probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
    tl.store(
        Probabilities + row[:, None] * PAD_K + keys[None, :],
        weighted / probability_scale[:, None],
        keys[None, :] < PAD_K,
    )
    logsum = tl.where(denominator > 0, maximum + tl.log(denominator), float("inf"))
    factor = tl.where(denominator > 0, probability_scale / denominator, 0.0)
    if HAS_SINK:
        sink = tl.load(Sink + row % H)
        factor *= tl.where(
            sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
        )
    tl.store(Factors + row, factor)
    tl.store(LSE + row, logsum)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_fp8_values"),
    key=["B", "H", "TOPK"],
)
@triton.jit
def sparse_fp8_values(
    Probabilities,
    Factors,
    KV,
    Indices,
    Output,
    stride_kp,
    stride_kt,
    stride_ib,
    stride_ik,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    PAD_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch, head_block, value_block = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    heads = head_block * 64 + tl.arange(0, 64)
    dims = value_block * BLOCK_D + tl.arange(0, BLOCK_D)
    keys = tl.arange(0, BLOCK_K)
    acc = tl.zeros((64, BLOCK_D), tl.float32)
    for start in range(triton.cdiv(TOPK, BLOCK_K)):
        positions = start * BLOCK_K + keys
        weights = tl.load(
            Probabilities + (batch * H + heads[:, None]) * PAD_K + positions[None, :],
            positions[None, :] < PAD_K,
            0.0,
        )
        ids = tl.load(
            Indices + batch * stride_ib + positions * stride_ik, positions < TOPK, -1
        )
        valid = (ids >= 0) & (ids < N) & (positions < TOPK)
        ids = tl.where(valid, ids, 0).to(tl.int64)
        values = tl.load(
            KV
            + ids[:, None] // 64 * stride_kp
            + ids[:, None] % 64 * stride_kt
            + dims[None, :],
            valid[:, None],
            0.0,
        )
        acc = tl.dot(weights, values, acc)
    factors = tl.load(Factors + batch * H + heads)
    tl.store(
        Output + (batch * H + heads[:, None]) * 512 + dims[None, :],
        acc * factors[:, None],
    )


@triton.jit
def sparse_fp8_transpose_values(
    s_src,
    s_dst,
    dst_row: tl.constexpr,
):
    """Transpose with coupled K permutation and bank-distributed output rows."""
    carrier = tl.arange(0, 256).to(tl.uint32)
    src_base = tle.gpu.local_ptr(s_src, (0, 0))
    dst_base = tle.gpu.local_ptr(s_dst, (dst_row, 0))
    return tl.inline_asm_elementwise(
        asm=(
            "{\n"
            ".reg .b32 tid, lane, warp, src_row, tmp, tmp2;\n"
            ".reg .b32 src_log, src_phys, src_addr0, src_addr1;\n"
            ".reg .b32 dst_row_r, dst_col, dst_log0, dst_log1;\n"
            ".reg .b32 dst_phys0, dst_phys1, dst_addr0, dst_addr1;\n"
            ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
            ".reg .b32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
            "mov.u32 tid, %tid.x;\n"
            "and.b32 tid, tid, 255;\n"
            "and.b32 lane, tid, 31;\n"
            "shr.u32 warp, tid, 5;\n"
            # The coupled P permutation cancels the source-row bit permutation.
            "mov.u32 src_row, lane;\n"
            "shl.b32 src_log, src_row, 7;\n"
            "shl.b32 tmp, warp, 4;\n"
            "add.u32 src_log, src_log, tmp;\n"
            "shr.u32 tmp, src_log, 7;\n"
            "and.b32 tmp, tmp, 7;\n"
            "shl.b32 tmp, tmp, 4;\n"
            "xor.b32 src_phys, src_log, tmp;\n"
            "add.u32 src_addr0, $2, src_phys;\n"
            "add.u32 src_addr1, src_addr0, 4096;\n"
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
            "{a0, a1, a2, a3}, [src_addr0];\n"
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
            "{b0, b1, b2, b3}, [src_addr1];\n"
            "prmt.b32 c0, a0, a1, 0x6420;\n"
            "prmt.b32 c1, a0, a1, 0x7531;\n"
            "prmt.b32 c2, a2, a3, 0x6420;\n"
            "prmt.b32 c3, a2, a3, 0x7531;\n"
            "prmt.b32 d0, b0, b1, 0x6420;\n"
            "prmt.b32 d1, b0, b1, 0x7531;\n"
            "prmt.b32 d2, b2, b3, 0x6420;\n"
            "prmt.b32 d3, b2, b3, 0x7531;\n"
            # Consecutive rows distribute each matrix store over all shared banks.
            "and.b32 dst_row_r, lane, 15;\n"
            "shl.b32 tmp, warp, 4;\n"
            "add.u32 dst_row_r, dst_row_r, tmp;\n"
            "shr.u32 dst_col, lane, 4;\n"
            "and.b32 dst_col, dst_col, 1;\n"
            "shl.b32 dst_col, dst_col, 4;\n"
            "shl.b32 dst_log0, dst_row_r, 6;\n"
            "add.u32 dst_log0, dst_log0, dst_col;\n"
            "add.u32 dst_log1, dst_log0, 32;\n"
            "shr.u32 tmp, dst_log0, 7;\n"
            "and.b32 tmp, tmp, 3;\n"
            "shl.b32 tmp, tmp, 4;\n"
            "xor.b32 dst_phys0, dst_log0, tmp;\n"
            "shr.u32 tmp2, dst_log1, 7;\n"
            "and.b32 tmp2, tmp2, 3;\n"
            "shl.b32 tmp2, tmp2, 4;\n"
            "xor.b32 dst_phys1, dst_log1, tmp2;\n"
            "add.u32 dst_addr0, $3, dst_phys0;\n"
            "add.u32 dst_addr1, $3, dst_phys1;\n"
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "[dst_addr0], {c0, c1, c2, c3};\n"
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "[dst_addr1], {d0, d1, d2, d3};\n"
            "mov.u32 $0, $1;\n"
            "}"
        ),
        constraints="=r,r,r,r",
        args=[carrier, src_base, dst_base],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@tlc.builtin
def sparse_named_barriers(count, threads, base, _semantic=None):
    barriers = tle.gpu.alloc_barriers(count, arrive_count=threads, _semantic=_semantic)
    # Lazy IDs are not unique across JIT helpers; reserve them before capture.
    base = int(tlc._unwrap_if_constexpr(base))
    barriers.named_base_id = base
    barriers.type.named_base_id = base
    return barriers


@triton.jit
def sparse_named_wait_pair(barriers, slot):
    # Separate branches keep barrier IDs constant through the combine pass.
    if slot == 0:
        tle.gpu.barrier_wait(barriers[0])
    else:
        pass
    if slot == 1:
        tle.gpu.barrier_wait(barriers[1])
    else:
        pass


@triton.jit
def sparse_named_arrive_pair(barriers, slot):
    # Separate branches keep barrier IDs constant through the combine pass.
    if slot == 0:
        tle.gpu.barrier_arrive(barriers[0])
    else:
        pass
    if slot == 1:
        tle.gpu.barrier_arrive(barriers[1])
    else:
        pass


@triton.jit
def sparse_fp8_producer(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Length,
    Sink,
    Output,
    LSE,
    qb,
    qh,
    qrb,
    qrh,
    kp,
    kt,
    krp,
    krt,
    qsb,
    qsh,
    ksp,
    kst,
    ib,
    ik,
    sq,
    sr,
    sk,
    skr,
    sv0,
    sv1,
    sp,
    alpha,
    scales,
    mask,
    factor,
    qfull,
    kfull,
    kempty,
    pfull,
    ofull,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
):
    batch = tl.program_id(0)
    heads = tl.program_id(1) * 64 + tl.arange(0, 64)
    dims = tl.arange(0, 512)
    ropes = tl.arange(0, 64)
    query = tl.load(Q + batch * qb + heads[:, None] * qh + dims[None, :])
    query_rope = tl.load(QR + batch * qrb + heads[:, None] * qrh + ropes[None, :])
    tl.store(tle.gpu.local_ptr(sq), query)
    tl.store(tle.gpu.local_ptr(sr), query_rope)
    tle.gpu.barrier_arrive(qfull[0])
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    for step in range(tl.cdiv(length, 64)):
        buf = step % 2
        if step >= 2:
            sparse_named_wait_pair(kempty, buf)
        else:
            pass
        positions = step * 64 + ropes
        ids = tl.load(Indices + batch * ib + positions * ik, positions < length, -1)
        valid = (positions < length) & (ids >= 0) & (ids < N)
        ids = tl.where(valid, ids, 0).to(tl.int64)
        (pages, slots) = (ids // 64, ids % 64)
        columns = tl.arange(0, 128)
        for group in tl.static_range(4):
            content = tl.load(
                KV
                + pages[:, None] * kp
                + slots[:, None] * kt
                + group * 128
                + columns[None, :],
                valid[:, None],
                0.0,
            )
            tl.store(
                tle.gpu.local_ptr(
                    sparse_smem_subslice(sk.slot(buf), [0, group * 128], [64, 128])
                ),
                content,
            )
        rope = tl.load(
            KR + pages[:, None] * krp + slots[:, None] * krt + ropes[None, :],
            valid[:, None],
            0.0,
        )
        tl.store(tle.gpu.local_ptr(skr.slot(buf)), rope)
        kv_scale = tl.load(KS + pages * ksp + slots * kst, valid, 0.0)
        tl.store(tle.gpu.local_ptr(scales.slot(buf)), kv_scale)
        tl.store(tle.gpu.local_ptr(mask.slot(buf)), tl.where(valid, 0.0, -float("inf")))
        # Matrix transpose avoids byte stores and their shared-memory bank conflicts.
        tl.debug_barrier()
        for group in tl.static_range(4):
            source = sparse_smem_subslice(sk.slot(buf), [0, group * 128], [64, 128])
            if group < 2:
                sparse_fp8_transpose_values(source, sv0.slot(buf), group * 128)
            else:
                sparse_fp8_transpose_values(source, sv1.slot(buf), (group - 2) * 128)
        tl.inline_asm_elementwise(
            "{ fence.proxy.async.shared::cta; mov.u32 $0, $1; }",
            constraints="=r,r",
            args=[tl.arange(0, 256)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        # Inline PTX stores are opaque to TLE's automatic publication barriers.
        tl.debug_barrier()
        sparse_named_arrive_pair(kfull, buf)


@triton.jit
def sparse_fp8_store_output(
    acc, inverse, scratch, Output, batch, heads, H: tl.constexpr, OFFSET: tl.constexpr
):
    dims = tl.arange(0, 256)
    rows = tl.arange(0, 64)
    # Retired K storage makes MMA output lanes write fully populated global sectors.
    scratch_ptr = tle.gpu.local_ptr(scratch, (0, 0)).to(tl.pointer_type(tl.bfloat16, 3))
    offsets = rows[:, None] * 256 + dims[None, :]
    # Undo the V row permutation as a view before the shared output exchange.
    acc = tl.reshape(
        tl.permute(tl.reshape(acc, (64, 16, 2, 8)), (0, 1, 3, 2)), (64, 256)
    )
    tl.store(scratch_ptr + offsets, (acc * inverse[:, None]).to(tl.bfloat16))
    tl.debug_barrier()
    value = tl.load(scratch_ptr + offsets)
    tl.store(
        Output + (batch * H + heads[:, None]) * 512 + OFFSET + dims[None, :], value
    )


@triton.jit
def sparse_fp8_consumer0(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Length,
    Sink,
    Output,
    LSE,
    qb,
    qh,
    qrb,
    qrh,
    kp,
    kt,
    krp,
    krt,
    qsb,
    qsh,
    ksp,
    kst,
    ib,
    ik,
    sq,
    sr,
    sk,
    skr,
    sv0,
    sv1,
    sp,
    alpha,
    scales,
    mask,
    factor,
    qfull,
    kfull,
    kempty,
    pfull,
    ofull,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    batch = tl.program_id(0)
    heads = tl.program_id(1) * 64 + tl.arange(0, 64)
    # Keep online softmax in base-2 units and fold row-constant scaling once.
    query_scale = tl.load(QS + batch * qsb + heads * qsh) * (SCALE * 1.4426950408889634)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    maximum = tl.full((64,), -float("inf"), tl.float32)
    denominator = tl.zeros((64,), tl.float32)
    acc0 = tl.zeros((64, 128), tl.float32)
    acc1 = tl.zeros((64, 128), tl.float32)
    previous_scale = tl.full((64,), 1.0, tl.float32)
    tle.gpu.barrier_wait(qfull[0])
    for step in range(tl.cdiv(length, 64)):
        buf = step % 2
        sparse_named_wait_pair(kfull, buf)
        logits = tle.gpu.wgmma(sq, sk.slot(buf), out_dtype=tl.float32, trans_b=True)
        logits = tle.gpu.wgmma(sr, skr.slot(buf), logits, trans_b=True)
        logits = tle.gpu.wgmma_wait(0, logits)
        kv_scale = tl.load(tle.gpu.local_ptr(scales.slot(buf)))
        add_mask = tl.load(tle.gpu.local_ptr(mask.slot(buf)))
        logits = logits * query_scale[:, None] * kv_scale[None, :] + add_mask[None, :]
        next_maximum = tl.maximum(maximum, tl.max(logits, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp2(maximum - safe_maximum)
        probabilities = tl.exp2(logits - safe_maximum[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        weighted = probabilities * kv_scale[None, :]
        probability_scale = tl.max(weighted, 1) / 448.0
        probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
        p = weighted / probability_scale[:, None]
        # P and V share the same K permutation, avoiding cross-lane P shuffles.
        _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(sp.slot(buf), p)
        # Keep PV in probability-scale units, requiring one rescale per tile.
        correction = correction * previous_scale / probability_scale
        tl.store(tle.gpu.local_ptr(alpha.slot(buf)), correction)
        sparse_named_arrive_pair(pfull, buf)
        acc0 *= correction[:, None]
        acc1 *= correction[:, None]
        acc0 = tle.gpu.wgmma(
            sp.slot(buf),
            sparse_smem_subslice(sv0.slot(buf), [0, 0], [128, 64]),
            acc0,
            trans_b=True,
        )
        acc1 = tle.gpu.wgmma(
            sp.slot(buf),
            sparse_smem_subslice(sv0.slot(buf), [128, 0], [128, 64]),
            acc1,
            trans_b=True,
        )
        acc0 = tle.gpu.wgmma_wait(0, acc0)
        acc1 = tle.gpu.wgmma_wait(0, acc1)
        previous_scale = probability_scale
        maximum = next_maximum
        sparse_named_arrive_pair(kempty, buf)
    logsum = tl.where(
        denominator > 0,
        (maximum + tl.log2(denominator)) * 0.6931471805599453,
        float("inf"),
    )
    inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
    if HAS_SINK:
        sink = tl.load(Sink + heads)
        inverse *= tl.where(
            sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
        )
    inverse *= previous_scale
    tl.store(tle.gpu.local_ptr(factor), inverse)
    tle.gpu.barrier_arrive(ofull[0], phaseIdx=0)
    acc = tl.reshape(tl.permute(tl.join(acc0, acc1), (0, 2, 1)), (64, 256))
    sparse_fp8_store_output(acc, inverse, sk.slot(0), Output, batch, heads, H, 0)
    tl.store(LSE + batch * H + heads, logsum)


@triton.jit
def sparse_fp8_consumer1(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Length,
    Sink,
    Output,
    LSE,
    sq,
    sr,
    sk,
    skr,
    sv0,
    sv1,
    sp,
    alpha,
    scales,
    mask,
    factor,
    qfull,
    kfull,
    kempty,
    pfull,
    ofull,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
):
    batch = tl.program_id(0)
    heads = tl.program_id(1) * 64 + tl.arange(0, 64)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    acc0 = tl.zeros((64, 128), tl.float32)
    acc1 = tl.zeros((64, 128), tl.float32)
    for step in range(tl.cdiv(length, 64)):
        buf = step % 2
        sparse_named_wait_pair(pfull, buf)
        correction = tl.load(tle.gpu.local_ptr(alpha.slot(buf)))
        acc0 *= correction[:, None]
        acc1 *= correction[:, None]
        acc0 = tle.gpu.wgmma(
            sp.slot(buf),
            sparse_smem_subslice(sv1.slot(buf), [0, 0], [128, 64]),
            acc0,
            trans_b=True,
        )
        acc1 = tle.gpu.wgmma(
            sp.slot(buf),
            sparse_smem_subslice(sv1.slot(buf), [128, 0], [128, 64]),
            acc1,
            trans_b=True,
        )
        acc0 = tle.gpu.wgmma_wait(0, acc0)
        acc1 = tle.gpu.wgmma_wait(0, acc1)
        sparse_named_arrive_pair(kempty, buf)
    tle.gpu.barrier_wait(ofull[0], phaseIdx=0)
    inverse = tl.load(tle.gpu.local_ptr(factor))
    acc = tl.reshape(tl.permute(tl.join(acc0, acc1), (0, 2, 1)), (64, 256))
    sparse_fp8_store_output(acc, inverse, sk.slot(1), Output, batch, heads, H, 256)


if HAS_TLE:

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("sparse_fp8_warp_specialized"),
        key=["B", "H", "TOPK"],
    )
    @triton.jit
    def sparse_fp8_warp_specialized(
        Q,
        QR,
        KV,
        KR,
        QS,
        KS,
        Indices,
        Length,
        Sink,
        Output,
        LSE,
        qb,
        qh,
        qrb,
        qrh,
        kp,
        kt,
        krp,
        krt,
        qsb,
        qsh,
        ksp,
        kst,
        ib,
        ik,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        TOPK: tl.constexpr,
        SCALE: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        HAS_SINK: tl.constexpr,
        PRODUCER_REGS: tl.constexpr,
    ):
        sq = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
        sr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
        sk = tle.gpu.alloc([2, 64, 512], tl.float8e4nv, scope=tle.gpu.smem)
        skr = tle.gpu.alloc([2, 64, 64], tl.bfloat16, scope=tle.gpu.smem)
        sv0 = tle.gpu.alloc([2, 256, 64], tl.float8e4nv, scope=tle.gpu.smem)
        sv1 = tle.gpu.alloc([2, 256, 64], tl.float8e4nv, scope=tle.gpu.smem)
        sp = tle.gpu.alloc([2, 64, 64], tl.float8e4nv, scope=tle.gpu.smem)
        alpha = tle.gpu.alloc(
            [2, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        scales = tle.gpu.alloc(
            [2, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        mask = tle.gpu.alloc(
            [2, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        factor = tle.gpu.alloc(
            [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        # TLE maps these virtual IDs to physical IDs outside the WS reserved set.
        qfull = sparse_named_barriers(1, 384, 16)
        kfull = sparse_named_barriers(2, 384, 17)
        kempty = sparse_named_barriers(2, 512, 19)
        pfull = sparse_named_barriers(2, 256, 21)
        ofull = tle.gpu.alloc_barriers(1, arrive_count=1)
        tle.gpu.warp_specialize(
            [
                (
                    sparse_fp8_consumer0,
                    (
                        Q,
                        QR,
                        KV,
                        KR,
                        QS,
                        KS,
                        Indices,
                        Length,
                        Sink,
                        Output,
                        LSE,
                        qb,
                        qh,
                        qrb,
                        qrh,
                        kp,
                        kt,
                        krp,
                        krt,
                        qsb,
                        qsh,
                        ksp,
                        kst,
                        ib,
                        ik,
                        sq,
                        sr,
                        sk,
                        skr,
                        sv0,
                        sv1,
                        sp,
                        alpha,
                        scales,
                        mask,
                        factor,
                        qfull,
                        kfull,
                        kempty,
                        pfull,
                        ofull,
                        H,
                        TOPK,
                        SCALE,
                        HAS_LENGTH,
                        HAS_SINK,
                    ),
                ),
                (
                    sparse_fp8_producer,
                    (
                        Q,
                        QR,
                        KV,
                        KR,
                        QS,
                        KS,
                        Indices,
                        Length,
                        Sink,
                        Output,
                        LSE,
                        qb,
                        qh,
                        qrb,
                        qrh,
                        kp,
                        kt,
                        krp,
                        krt,
                        qsb,
                        qsh,
                        ksp,
                        kst,
                        ib,
                        ik,
                        sq,
                        sr,
                        sk,
                        skr,
                        sv0,
                        sv1,
                        sp,
                        alpha,
                        scales,
                        mask,
                        factor,
                        qfull,
                        kfull,
                        kempty,
                        pfull,
                        ofull,
                        H,
                        N,
                        TOPK,
                        HAS_LENGTH,
                    ),
                ),
                (
                    sparse_fp8_consumer1,
                    (
                        Q,
                        QR,
                        KV,
                        KR,
                        QS,
                        KS,
                        Indices,
                        Length,
                        Sink,
                        Output,
                        LSE,
                        sq,
                        sr,
                        sk,
                        skr,
                        sv0,
                        sv1,
                        sp,
                        alpha,
                        scales,
                        mask,
                        factor,
                        qfull,
                        kfull,
                        kempty,
                        pfull,
                        ofull,
                        H,
                        TOPK,
                        HAS_LENGTH,
                    ),
                ),
            ],
            [8, 4],
            [PRODUCER_REGS, 168],
        )


def flash_mla_sparse_fwd_w8a8_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode using the separate per-token cache format of dense MLA.

    q_nope [B, 1, H, 512] and k_cache_lora [P, 64, 512] are FP8 e4m3fn;
    q_rope [B, 1, H, 64] and k_cache_rope [P, 64, 64] are BF16.
    Both NoPE and RoPE store values divided by the corresponding FP32 scale:
    q_scale [B, 1, H, 1] and k_scale [P, 64, 1]. This directly accepts
    quantize_q_ckv_per_token / quantize_k_ckv_per_token outputs from dense MLA.
    V is the dequantized 512-dimensional NoPE cache. H must be 64 or 128.

    indices [B, 1, topk] contains int32 physical token IDs (page * 64 + slot).
    Negative and out-of-range IDs are ignored. topk_length [B] optionally
    limits the number of entries read per request. attn_sink [H] affects
    output only. Inputs must be finite, with positive finite scales.

    Returns BF16 output [B, 1, H, 512] and natural-log FP32 LSE [B, H, 1].
    Empty attention produces zero output and +inf LSE. Forward only.
    The TLE path requires FlagTree's cross-dtype WGMMA support (PR #1001).
    """
    if q_nope.device.type != "cuda":
        raise NotImplementedError("FP8 sparse MLA requires NVIDIA Hopper CUDA")
    if torch.cuda.get_device_capability(q_nope.device)[0] != 9:
        raise NotImplementedError("FP8 sparse MLA is supported on Hopper")
    if q_nope.dtype != torch.float8_e4m3fn or k_cache_lora.dtype != torch.float8_e4m3fn:
        raise TypeError("NoPE tensors must have dtype float8_e4m3fn")
    if q_rope.dtype != torch.bfloat16 or k_cache_rope.dtype != torch.bfloat16:
        raise TypeError("RoPE tensors must have dtype bfloat16")
    if q_nope.ndim != 4 or k_cache_lora.ndim != 3 or indices.ndim != 3:
        raise ValueError("q_nope, cache and indices must have ranks 4, 3 and 3")
    batch, query_length, heads, dim = q_nope.shape
    pages = k_cache_lora.shape[0]
    topk = indices.shape[-1]
    if query_length != 1 or heads not in (64, 128) or dim != 512:
        raise NotImplementedError(
            "Requires one query, 64/128 heads and 512 NoPE dimensions"
        )
    if q_rope.shape != (batch, 1, heads, 64):
        raise ValueError("q_rope must have shape [batch, 1, heads, 64]")
    if k_cache_lora.shape != (pages, 64, 512) or k_cache_rope.shape != (pages, 64, 64):
        raise ValueError(
            "Caches must have page size 64 and NoPE/RoPE dimensions 512/64"
        )
    if indices.shape != (batch, 1, topk) or indices.dtype != torch.int32:
        raise ValueError("indices must be int32 [batch, 1, topk]")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("q_scale and k_scale must have dtype float32")
    if q_scale.shape != (batch, 1, heads, 1) or k_scale.shape != (pages, 64, 1):
        raise ValueError("Scale shapes must be [batch, 1, heads, 1] and [pages, 64, 1]")
    for tensor in (q_nope, q_rope, k_cache_lora, k_cache_rope):
        if tensor.stride(-1) != 1:
            raise ValueError("NoPE and RoPE must be contiguous in the last dimension")
    for tensor in (
        q_rope,
        k_cache_lora,
        k_cache_rope,
        indices,
        q_scale,
        k_scale,
        attn_sink,
        topk_length,
    ):
        if tensor is not None and tensor.device != q_nope.device:
            raise ValueError("All tensors must be on the same CUDA device")
    if attn_sink is not None and (
        attn_sink.shape != (heads,)
        or attn_sink.dtype != torch.float32
        or not attn_sink.is_contiguous()
    ):
        raise ValueError("attn_sink must be contiguous float32 [heads]")
    if topk_length is not None and (
        topk_length.shape != (batch,)
        or topk_length.dtype != torch.int32
        or not topk_length.is_contiguous()
    ):
        raise ValueError("topk_length must be contiguous int32 [batch]")
    output = torch.empty(
        (batch, 1, heads, 512), device=q_nope.device, dtype=torch.bfloat16
    )
    lse = torch.empty((batch, heads, 1), device=q_nope.device, dtype=torch.float32)
    if batch == 0:
        return output, lse
    softmax_scale = 576**-0.5 if softmax_scale is None else float(softmax_scale)
    if HAS_TLE and batch >= 16:
        sparse_fp8_warp_specialized[batch, heads // 64](
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            output,
            lse,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            topk_length is not None,
            attn_sink is not None,
        )
        return output, lse
    # Bound the row-wise softmax working set; longer lists use online softmax.
    if batch >= 16 and 128 <= topk <= 8192:
        padded_topk = triton.cdiv(topk, 128) * 128
        scores = torch.empty(
            (batch, heads, padded_topk), device=q_nope.device, dtype=torch.float32
        )
        probabilities = torch.empty(
            (batch, heads, padded_topk), device=q_nope.device, dtype=torch.float8_e4m3fn
        )
        factors = torch.empty((batch, heads), device=q_nope.device, dtype=torch.float32)
        sparse_fp8_scores[
            lambda meta: (batch, heads // 64, triton.cdiv(padded_topk, meta["BLOCK_K"]))
        ](
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            topk_length,
            scores,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            batch,
            heads,
            pages * 64,
            topk,
            padded_topk,
            softmax_scale,
            topk_length is not None,
        )
        reduction_block = triton.next_power_of_2(padded_topk)
        reduction_rows = min(8, max(1, 8192 // reduction_block))
        sparse_fp8_probabilities[(batch * heads // reduction_rows,)](
            scores,
            indices,
            k_scale,
            attn_sink,
            probabilities,
            factors,
            lse,
            indices.stride(0),
            indices.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            heads,
            pages * 64,
            topk,
            padded_topk,
            attn_sink is not None,
            reduction_block,
            reduction_rows,
            num_warps=4,
        )
        sparse_fp8_values[lambda meta: (batch, heads // 64, 512 // meta["BLOCK_D"])](
            probabilities,
            factors,
            k_cache_lora,
            indices,
            output,
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            indices.stride(0),
            indices.stride(2),
            batch,
            heads,
            pages * 64,
            topk,
            padded_topk,
        )
        return output, lse
    # Bound workspace and give small decode batches enough independent CTAs.
    desired_splits = triton.next_power_of_2(triton.cdiv(132, batch * (heads // 32)))
    splits = min(desired_splits, 32, max(1, triton.next_power_of_2(topk // 256)))
    if splits > 1:
        partial = torch.empty(
            (batch, splits, heads, 512), device=q_nope.device, dtype=torch.float32
        )
        stats = torch.empty(
            (batch, splits, heads, 2), device=q_nope.device, dtype=torch.float32
        )
    else:
        partial, stats = output, lse
    sparse_fp8_kernel[
        lambda meta: (batch, triton.cdiv(heads, meta["BLOCK_H"]), splits)
    ](
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        indices,
        q_scale,
        k_scale,
        attn_sink,
        topk_length,
        output,
        lse,
        partial,
        stats,
        q_nope.stride(0),
        q_nope.stride(2),
        q_rope.stride(0),
        q_rope.stride(2),
        k_cache_lora.stride(0),
        k_cache_lora.stride(1),
        k_cache_rope.stride(0),
        k_cache_rope.stride(1),
        indices.stride(0),
        indices.stride(2),
        q_scale.stride(0),
        q_scale.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        batch,
        heads,
        pages * 64,
        topk,
        softmax_scale,
        splits,
        attn_sink is not None,
        topk_length is not None,
    )
    if splits > 1:
        sparse_fp8_merge[(batch * heads,)](
            partial,
            stats,
            attn_sink,
            output,
            lse,
            heads,
            splits,
            attn_sink is not None,
            num_warps=4,
        )
    return output, lse
