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
import triton.language.core as tlc
from triton.experimental.tle.language.gpu import types as tle_types

from flaggems_vllm import runtime
from flaggems_vllm.ops.flash_mla import tle
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.mla_sparse_fp8_constants import (
    ACCUMULATOR_SCALE_FLOOR,
    QK_RECOMPUTE_THRESHOLD,
)
from flaggems_vllm.utils import libentry, libtuner


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


@tlc.builtin
def sparse_named_barriers(count, threads, base, _semantic=None):
    barriers = tle.gpu.alloc_barriers(count, arrive_count=threads, _semantic=_semantic)
    # Lazy IDs are not unique across JIT helpers; reserve them before capture.
    base = int(tlc._unwrap_if_constexpr(base))
    barriers.named_base_id = base
    barriers.type.named_base_id = base
    return barriers


@libentry()
@triton.jit
def sparse_fp8_empty(Output, LSE, ROWS: tl.constexpr):
    # Fixed write-only geometry; no attention tiling or reduction is performed.
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    tl.store(Output + offsets, 0.0, offsets < ROWS * 512)
    tl.store(LSE + offsets, float("inf"), offsets < ROWS)


@triton.jit
def sparse_fp8_accumulate(logits, contribution):
    # Keep FP32 additions separate from tensor-core accumulation.
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [logits, contribution],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sparse_fp8_repair"),
    key=["B", "H", "TOPK", "SPLITS"],
)
@triton.jit
def sparse_fp8_repair(
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
    RepairFlags,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    REPAIR_SPLITS: tl.constexpr = 1,
):
    batch = tl.program_id(0)
    flags = tl.load(
        RepairFlags
        + (batch * (H // 64) + tl.program_id(1)) * REPAIR_SPLITS
        + tl.arange(0, REPAIR_SPLITS)
    )
    if tl.max(flags, 0) == 0:
        return
    else:
        pass
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
    amplification = tl.max(tl.abs(query_scale * (SM_SCALE * 1.4426950408889634)), 0)
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
        if amplification * tl.max(kv_scale, 0) > QK_RECOMPUTE_THRESHOLD:
            logits = tl.zeros((BLOCK_H, BLOCK_K), tl.float32)
            pad = tl.arange(0, 32)
            for feature in range(512):
                query_column = tl.load(
                    Q + batch * stride_qb + heads * stride_qh + feature, heads < H, 0.0
                )
                key_column = tl.load(
                    KV + pages * stride_kvp + tokens * stride_kvt + feature, valid, 0.0
                )
                query_feature = tl.where(
                    pad[None, :] == 0, query_column[:, None], 0.0
                ).to(tl.float8e4nv)
                key_feature = tl.where(pad[:, None] == 0, key_column[None, :], 0.0).to(
                    tl.float8e4nv
                )
                contribution = tl.dot(query_feature, key_feature, out_dtype=tl.float32)
                logits = sparse_fp8_accumulate(logits, contribution)
        else:
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


@triton.jit
def sparse_fp8_checked_merge(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Length,
    Sink,
    Flags,
    Partial,
    Stats,
    Output,
    LSE,
    qb: tl.constexpr,
    qh: tl.constexpr,
    qrb: tl.constexpr,
    qrh: tl.constexpr,
    kp: tl.constexpr,
    kt: tl.constexpr,
    krp: tl.constexpr,
    krt: tl.constexpr,
    qsb: tl.constexpr,
    qsh: tl.constexpr,
    ksp: tl.constexpr,
    kst: tl.constexpr,
    ib: tl.constexpr,
    ik: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    row = tl.program_id(0)
    batch, head = row // H, row % H
    split = tl.arange(0, SPLITS)
    flags = tl.load(Flags + (batch * (H // 64) + head // 64) * SPLITS + split)
    dims = tl.arange(0, 512)
    if tl.max(flags, 0) != 0:
        ropes = tl.arange(0, 64)
        query = tl.load(Q + batch * qb + head * qh + dims).to(tl.float32)
        query_rope = tl.load(QR + batch * qrb + head * qrh + ropes).to(tl.float32)
        query_scale = tl.load(QS + batch * qsb + head * qsh)
        length = (
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
        )
        maximum = -float("inf")
        denominator = 0.0
        values = tl.zeros((512,), tl.float32)
        for position in range(length):
            token = tl.load(Indices + batch * ib + position * ik)
            if (token >= 0) & (token < N):
                physical_token = token.to(tl.uint64)
                page, slot = physical_token // 64, physical_token % 64
                key = tl.load(KV + page * kp + slot * kt + dims).to(tl.float32)
                key_rope = tl.load(KR + page * krp + slot * krt + ropes).to(tl.float32)
                key_scale = tl.load(KS + page * ksp + slot * kst)
                score = (
                    (tl.sum(query * key, 0) + tl.sum(query_rope * key_rope, 0))
                    * query_scale
                    * key_scale
                    * (SCALE * 1.4426950408889634)
                )
                new_maximum = tl.maximum(maximum, score)
                old_delta = tl.inline_asm_elementwise(
                    "sub.rn.f32 $0, $1, $2;",
                    "=f,f,f",
                    [maximum, new_maximum],
                    dtype=tl.float32,
                    is_pure=False,
                    pack=1,
                )
                score_delta = tl.inline_asm_elementwise(
                    "sub.rn.f32 $0, $1, $2;",
                    "=f,f,f",
                    [score, new_maximum],
                    dtype=tl.float32,
                    is_pure=False,
                    pack=1,
                )
                correction = tl.exp2(old_delta)
                probability = tl.exp2(score_delta)
                denominator = denominator * correction + probability
                values = values * correction + probability * (key * key_scale)
                maximum = new_maximum
            else:
                pass
        logsum = tl.where(
            denominator > 0,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
            float("inf"),
        )
        inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
        if HAS_SINK:
            sink = tl.load(Sink + head)
            inverse *= tl.where(
                sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
            )
        else:
            pass
        tl.store(Output + row * 512 + dims, values * inverse)
        tl.store(LSE + row, logsum)
    elif SPLITS > 1:
        stats_ptr = Stats + ((batch * SPLITS + split) * H + head) * 2
        maxima = tl.load(stats_ptr)
        denominators = tl.load(stats_ptr + 1)
        global_maximum = tl.max(maxima, 0)
        safe_maximum = tl.where(global_maximum == -float("inf"), 0.0, global_maximum)
        weights = tl.exp(maxima - safe_maximum)
        total = tl.sum(denominators * weights, 0)
        logsum = tl.where(total > 0, global_maximum + tl.log(total), float("inf"))
        inverse = tl.where(total > 0, 1.0 / total, 0.0)
        if HAS_SINK:
            sink = tl.load(Sink + head)
            inverse *= tl.where(
                sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
            )
        else:
            pass
        partial = tl.load(
            Partial
            + ((batch * SPLITS + split[:, None]) * H + head) * 512
            + dims[None, :]
        )
        output = tl.sum(partial * weights[:, None], 0) * inverse
        tl.store(Output + row * 512 + dims, output)
        tl.store(LSE + row, logsum)
    else:
        pass


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
def sparse_fp8_split_blocks(length, SPLITS: tl.constexpr):
    blocks = tl.cdiv(length, 64)
    if SPLITS == 1:
        first_block = 0
        split_blocks = blocks
    else:
        blocks_per_split = tl.cdiv(blocks, SPLITS)
        first_block = tl.program_id(2) * blocks_per_split
        split_blocks = tl.maximum(0, tl.minimum(blocks_per_split, blocks - first_block))
    return first_block, split_blocks


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
    SPLITS: tl.constexpr,
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
    first_block, split_blocks = sparse_fp8_split_blocks(length, SPLITS)
    for step in range(split_blocks):
        buf = step % 2
        if step >= 2:
            sparse_named_wait_pair(kempty, buf)
        else:
            pass
        positions = (first_block + step) * 64 + ropes
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
def sparse_fp8_publish_output(
    acc, inverse, Output, batch, head_block, H: tl.constexpr, OFFSET: tl.constexpr
):
    if Output.dtype.element_ty == tl.float32:
        rows = head_block * 64 + tl.arange(0, 64)
        columns = tl.arange(0, 128)
        columns = (columns // 16) * 16 + (columns % 8) * 2 + (columns % 16) // 8
        tl.store(
            Output + (batch * H + rows[:, None]) * 512 + OFFSET + columns[None, :],
            acc * inverse[:, None],
        )
        return
        # Pair permuted accumulator columns into adjacent BF16 output elements.
    base_u64 = (Output + (batch * H + head_block * 64) * 512 + OFFSET).to(tl.uint64)
    tl.inline_asm_elementwise(
        asm=(
            "{\n"
            ".reg .b32 tid, row, col, offset, a, b, c, d;\n"
            ".reg .b64 addr;\n"
            "mov.u32 tid, %tid.x;\n"
            "and.b32 tid, tid, 127;\n"
            "shr.u32 row, tid, 5;\n"
            "shl.b32 row, row, 4;\n"
            "and.b32 col, tid, 31;\n"
            "shr.u32 col, col, 2;\n"
            "add.u32 row, row, col;\n"
            "and.b32 col, tid, 3;\n"
            "shl.b32 col, col, 3;\n"
            "shl.b32 row, row, 10;\n"
            "add.u32 offset, row, col;\n"
            "cvt.u64.u32 addr, offset;\n"
            "add.u64 addr, addr, $128;\n"
            "cvt.rn.bf16x2.f32 a, $68, $64;\n"
            "cvt.rn.bf16x2.f32 b, $69, $65;\n"
            "cvt.rn.bf16x2.f32 c, $70, $66;\n"
            "cvt.rn.bf16x2.f32 d, $71, $67;\n"
            "st.global.v2.b32 [addr+0], {a, b};\n"
            "st.global.v2.b32 [addr+8192], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $76, $72;\n"
            "cvt.rn.bf16x2.f32 b, $77, $73;\n"
            "cvt.rn.bf16x2.f32 c, $78, $74;\n"
            "cvt.rn.bf16x2.f32 d, $79, $75;\n"
            "st.global.v2.b32 [addr+32], {a, b};\n"
            "st.global.v2.b32 [addr+8224], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $84, $80;\n"
            "cvt.rn.bf16x2.f32 b, $85, $81;\n"
            "cvt.rn.bf16x2.f32 c, $86, $82;\n"
            "cvt.rn.bf16x2.f32 d, $87, $83;\n"
            "st.global.v2.b32 [addr+64], {a, b};\n"
            "st.global.v2.b32 [addr+8256], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $92, $88;\n"
            "cvt.rn.bf16x2.f32 b, $93, $89;\n"
            "cvt.rn.bf16x2.f32 c, $94, $90;\n"
            "cvt.rn.bf16x2.f32 d, $95, $91;\n"
            "st.global.v2.b32 [addr+96], {a, b};\n"
            "st.global.v2.b32 [addr+8288], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $100, $96;\n"
            "cvt.rn.bf16x2.f32 b, $101, $97;\n"
            "cvt.rn.bf16x2.f32 c, $102, $98;\n"
            "cvt.rn.bf16x2.f32 d, $103, $99;\n"
            "st.global.v2.b32 [addr+128], {a, b};\n"
            "st.global.v2.b32 [addr+8320], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $108, $104;\n"
            "cvt.rn.bf16x2.f32 b, $109, $105;\n"
            "cvt.rn.bf16x2.f32 c, $110, $106;\n"
            "cvt.rn.bf16x2.f32 d, $111, $107;\n"
            "st.global.v2.b32 [addr+160], {a, b};\n"
            "st.global.v2.b32 [addr+8352], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $116, $112;\n"
            "cvt.rn.bf16x2.f32 b, $117, $113;\n"
            "cvt.rn.bf16x2.f32 c, $118, $114;\n"
            "cvt.rn.bf16x2.f32 d, $119, $115;\n"
            "st.global.v2.b32 [addr+192], {a, b};\n"
            "st.global.v2.b32 [addr+8384], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $124, $120;\n"
            "cvt.rn.bf16x2.f32 b, $125, $121;\n"
            "cvt.rn.bf16x2.f32 c, $126, $122;\n"
            "cvt.rn.bf16x2.f32 d, $127, $123;\n"
            "st.global.v2.b32 [addr+224], {a, b};\n"
            "st.global.v2.b32 [addr+8416], {c, d};\n"
            "mov.u32 $0, 0;\n"
            "}\n"
        ),
        constraints=(
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,"
            "l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,"
            "l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,"
            "l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l"
        ),
        args=[acc * inverse[:, None], base_u64],
        dtype=tl.uint32,
        is_pure=False,
        pack=64,
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
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    RepairFlags,
):
    batch = tl.program_id(0)
    heads = tl.program_id(1) * 64 + tl.arange(0, 64)
    # Keep online softmax in base-2 units and fold row-constant scaling once.
    query_scale = tl.load(QS + batch * qsb + heads * qsh) * (SCALE * 1.4426950408889634)
    amplification = tl.max(tl.abs(query_scale), 0)
    needs_repair = tl.full((), False, tl.int1)
    maximum_weight_scale = tl.zeros((64,), tl.float32)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    maximum = tl.full((64,), -float("inf"), tl.float32)
    denominator = tl.zeros((64,), tl.float32)
    acc0 = tl.zeros((64, 128), tl.float32)
    acc1 = tl.zeros((64, 128), tl.float32)
    previous_scale = tl.full((64,), 1.0, tl.float32)
    tle.gpu.barrier_wait(qfull[0])
    first_block, split_blocks = sparse_fp8_split_blocks(length, SPLITS)
    for step in range(split_blocks):
        buf = step % 2
        sparse_named_wait_pair(kfull, buf)
        logits = tle.gpu.wgmma(sq, sk.slot(buf), out_dtype=tl.float32, trans_b=True)
        logits = tle.gpu.wgmma(sr, skr.slot(buf), logits, trans_b=True)
        logits = tle.gpu.wgmma_wait(0, logits)
        kv_scale = tl.load(tle.gpu.local_ptr(scales.slot(buf)))
        needs_repair |= amplification * tl.max(kv_scale, 0) > QK_RECOMPUTE_THRESHOLD
        add_mask = tl.load(tle.gpu.local_ptr(mask.slot(buf)))
        logits = logits * query_scale[:, None] * kv_scale[None, :] + add_mask[None, :]
        next_maximum = tl.maximum(maximum, tl.max(logits, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp2(maximum - safe_maximum)
        probabilities = tl.exp2(logits - safe_maximum[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        weighted = probabilities * kv_scale[None, :]
        probability_scale = tl.max(weighted, 1) / 448.0
        maximum_weight_scale = tl.maximum(
            maximum_weight_scale * correction, probability_scale
        )
        probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
        needs_repair |= (
            tl.max(
                (maximum_weight_scale * ACCUMULATOR_SCALE_FLOOR > probability_scale).to(
                    tl.int32
                ),
                0,
            )
            != 0
        )
        p = weighted / probability_scale[:, None]
        # P and V share the same K permutation, avoiding cross-lane P shuffles.
        publish_p_fp8_sw64_coupled_stmatrix(sp.slot(buf), p)
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
    if SPLITS == 1:
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
    else:
        # Keep partial values in physical units; merge applies the sink only once.
        inverse = previous_scale
    output_batch = batch if SPLITS == 1 else batch * SPLITS + tl.program_id(2)
    tl.store(tle.gpu.local_ptr(factor), inverse)
    tle.gpu.barrier_arrive(ofull[0], phaseIdx=0)
    sparse_fp8_publish_output(
        acc0, inverse, Output, output_batch, tl.program_id(1), H, 0
    )
    sparse_fp8_publish_output(
        acc1, inverse, Output, output_batch, tl.program_id(1), H, 128
    )
    if SPLITS == 1:
        tl.store(LSE + batch * H + heads, logsum)
    else:
        stats = LSE + ((batch * SPLITS + tl.program_id(2)) * H + heads) * 2
        tl.store(stats, maximum * 0.6931471805599453)
        tl.store(stats + 1, denominator)
    tl.store(
        RepairFlags
        + (batch * (H // 64) + tl.program_id(1)) * SPLITS
        + tl.program_id(2),
        needs_repair.to(tl.int32),
    )


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
    SPLITS: tl.constexpr,
):
    batch = tl.program_id(0)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    acc0 = tl.zeros((64, 128), tl.float32)
    acc1 = tl.zeros((64, 128), tl.float32)
    first_block, split_blocks = sparse_fp8_split_blocks(length, SPLITS)
    for step in range(split_blocks):
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
    output_batch = batch if SPLITS == 1 else batch * SPLITS + tl.program_id(2)
    sparse_fp8_publish_output(
        acc0, inverse, Output, output_batch, tl.program_id(1), H, 256
    )
    sparse_fp8_publish_output(
        acc1, inverse, Output, output_batch, tl.program_id(1), H, 384
    )


@triton.jit
def publish_p_fp8_sw64_coupled_stmatrix(s_p, p):
    """CUDA-native P publication; V repack carries the matching K permutation."""
    base = tle.gpu.local_ptr(s_p, (0, 0))
    base_u32 = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[base],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return tl.inline_asm_elementwise(
        asm=(
            "{\n"
            ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15;\n"
            ".reg .b32 tid, warp_off, row_off, common, tmp, phys0, phys1, addr0, addr1;\n"
            ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
            "mov.u32 tid, %tid.x;\n"
            "and.b32 warp_off, tid, 96;\n"
            "shl.b32 warp_off, warp_off, 5;\n"
            "and.b32 row_off, tid, 15;\n"
            "shl.b32 row_off, row_off, 6;\n"
            "or.b32 common, warp_off, row_off;\n"
            "and.b32 tmp, tid, 16;\n"
            "or.b32 common, common, tmp;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h0, $33, $32;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h1, $35, $34;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h2, $37, $36;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h3, $39, $38;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h4, $41, $40;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h5, $43, $42;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h6, $45, $44;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h7, $47, $46;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h8, $49, $48;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h9, $51, $50;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h10, $53, $52;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h11, $55, $54;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h12, $57, $56;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h13, $59, $58;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h14, $61, $60;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h15, $63, $62;\n"
            "mov.b32 a0, {h0, h2};\n"
            "mov.b32 a1, {h1, h3};\n"
            "mov.b32 a2, {h4, h6};\n"
            "mov.b32 a3, {h5, h7};\n"
            "mov.b32 b0, {h8, h10};\n"
            "mov.b32 b1, {h9, h11};\n"
            "mov.b32 b2, {h12, h14};\n"
            "mov.b32 b3, {h13, h15};\n"
            "shr.u32 tmp, common, 7;\n"
            "and.b32 tmp, tmp, 3;\n"
            "shl.b32 tmp, tmp, 4;\n"
            "xor.b32 phys0, common, tmp;\n"
            "add.u32 common, common, 32;\n"
            "shr.u32 tmp, common, 7;\n"
            "and.b32 tmp, tmp, 3;\n"
            "shl.b32 tmp, tmp, 4;\n"
            "xor.b32 phys1, common, tmp;\n"
            "add.u32 addr0, $64, phys0;\n"
            "add.u32 addr1, $64, phys1;\n"
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr0], {a0, a1, a2, a3};\n"
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr1], {b0, b1, b2, b3};\n"
            "fence.proxy.async.shared::cta;\n"
            "mov.u32 $0, $64;\n"
            "}"
        ),
        constraints="=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",  # noqa: E501
        args=[p, base_u32],
        dtype=tl.uint32,
        is_pure=False,
        pack=32,
    )
