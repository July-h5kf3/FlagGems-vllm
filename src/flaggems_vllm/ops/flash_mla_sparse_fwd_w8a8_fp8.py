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
    beta,
    scales,
    mask,
    factor,
    qfull,
    kfull,
    kempty,
    pfull,
    pempty,
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
    tle.gpu.barrier_arrive(qfull[0], phaseIdx=0)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    for step in range(tl.cdiv(length, 64)):
        buf = step % 2
        phase = step // 2
        tle.gpu.barrier_wait(kempty[buf], phaseIdx=phase)
        positions = step * 64 + ropes
        ids = tl.load(Indices + batch * ib + positions * ik, positions < length, -1)
        valid = (positions < length) & (ids >= 0) & (ids < N)
        ids = tl.where(valid, ids, 0).to(tl.int64)
        (pages, slots) = (ids // 64, ids % 64)
        # Bound the producer's register footprint while transposing V.
        for group in tl.static_range(8):
            content = tl.load(
                KV
                + pages[:, None] * kp
                + slots[:, None] * kt
                + group * 64
                + ropes[None, :],
                valid[:, None],
                0.0,
            )
            content = tl.inline_asm_elementwise(
                "mov.b32 $0, $1;",
                "=r,r",
                [content],
                dtype=tl.float8e4nv,
                is_pure=True,
                pack=4,
            )
            tl.store(
                tle.gpu.local_ptr(
                    sparse_smem_subslice(sk.slot(buf), [0, group * 64], [64, 64])
                ),
                content,
            )
            if group < 4:
                value_view = sparse_smem_subslice(
                    sv0.slot(buf), [group * 64, 0], [64, 64]
                )
            else:
                value_view = sparse_smem_subslice(
                    sv1.slot(buf), [(group - 4) * 64, 0], [64, 64]
                )
            tl.store(tle.gpu.local_ptr(value_view), tl.trans(content))
        rope = tl.load(
            KR + pages[:, None] * krp + slots[:, None] * krt + ropes[None, :],
            valid[:, None],
            0.0,
        )
        tl.store(tle.gpu.local_ptr(skr.slot(buf)), rope)
        kv_scale = tl.load(KS + pages * ksp + slots * kst, valid, 0.0)
        tl.store(tle.gpu.local_ptr(scales.slot(buf)), kv_scale)
        tl.store(tle.gpu.local_ptr(mask.slot(buf)), tl.where(valid, 0.0, -float("inf")))
        tle.gpu.barrier_arrive(kfull[buf], phaseIdx=phase)


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
    beta,
    scales,
    mask,
    factor,
    qfull,
    kfull,
    kempty,
    pfull,
    pempty,
    ofull,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    batch = tl.program_id(0)
    heads = tl.program_id(1) * 64 + tl.arange(0, 64)
    dims = tl.arange(0, 256)
    query_scale = tl.load(QS + batch * qsb + heads * qsh)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    maximum = tl.full((64,), -float("inf"), tl.float32)
    denominator = tl.zeros((64,), tl.float32)
    acc = tl.zeros((64, 256), tl.float32)
    tle.gpu.barrier_wait(qfull[0], phaseIdx=0)
    for step in range(tl.cdiv(length, 64)):
        (buf, phase) = (step % 2, step // 2)
        tle.gpu.barrier_wait(kfull[buf], phaseIdx=phase)
        logits = tle.gpu.wgmma(sq, sk.slot(buf), out_dtype=tl.float32, trans_b=True)
        logits = tle.gpu.wgmma(sr, skr.slot(buf), logits, trans_b=True)
        logits = tle.gpu.wgmma_wait(0, logits)
        kv_scale = tl.load(tle.gpu.local_ptr(scales.slot(buf)))
        add_mask = tl.load(tle.gpu.local_ptr(mask.slot(buf)))
        logits = (
            logits * query_scale[:, None] * kv_scale[None, :] * SCALE
            + add_mask[None, :]
        )
        next_maximum = tl.maximum(maximum, tl.max(logits, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp(maximum - safe_maximum)
        probabilities = tl.exp(logits - safe_maximum[:, None])
        denominator = denominator * correction + tl.sum(probabilities, 1)
        weighted = probabilities * kv_scale[None, :]
        probability_scale = tl.max(weighted, 1) / 448.0
        probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
        p = (weighted / probability_scale[:, None]).to(tl.float8e4nv)
        tle.gpu.barrier_wait(pempty[buf], phaseIdx=phase)
        tl.store(tle.gpu.local_ptr(sp.slot(buf)), p)
        tl.store(tle.gpu.local_ptr(alpha.slot(buf)), correction)
        tl.store(tle.gpu.local_ptr(beta.slot(buf)), probability_scale)
        tle.gpu.barrier_arrive(pfull[buf], phaseIdx=phase)
        acc *= (correction / probability_scale)[:, None]
        acc = tle.gpu.wgmma(sp.slot(buf), sv0.slot(buf), acc, trans_b=True)
        acc = tle.gpu.wgmma_wait(0, acc)
        acc *= probability_scale[:, None]
        maximum = next_maximum
        tle.gpu.barrier_arrive(kempty[buf], phaseIdx=phase)
    logsum = tl.where(denominator > 0, maximum + tl.log(denominator), float("inf"))
    inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
    if HAS_SINK:
        sink = tl.load(Sink + heads)
        inverse *= tl.where(
            sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
        )
    tl.store(tle.gpu.local_ptr(factor), inverse)
    tle.gpu.barrier_arrive(ofull[0], phaseIdx=0)
    tl.store(
        Output + (batch * H + heads[:, None]) * 512 + dims[None, :],
        acc * inverse[:, None],
    )
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
    beta,
    scales,
    mask,
    factor,
    qfull,
    kfull,
    kempty,
    pfull,
    pempty,
    ofull,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
):
    batch = tl.program_id(0)
    heads = tl.program_id(1) * 64 + tl.arange(0, 64)
    dims = tl.arange(0, 256)
    length = (
        tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK) if HAS_LENGTH else TOPK
    )
    acc = tl.zeros((64, 256), tl.float32)
    for step in range(tl.cdiv(length, 64)):
        (buf, phase) = (step % 2, step // 2)
        tle.gpu.barrier_wait(kfull[buf], phaseIdx=phase)
        tle.gpu.barrier_wait(pfull[buf], phaseIdx=phase)
        correction = tl.load(tle.gpu.local_ptr(alpha.slot(buf)))
        probability_scale = tl.load(tle.gpu.local_ptr(beta.slot(buf)))
        acc *= (correction / probability_scale)[:, None]
        acc = tle.gpu.wgmma(sp.slot(buf), sv1.slot(buf), acc, trans_b=True)
        acc = tle.gpu.wgmma_wait(0, acc)
        acc *= probability_scale[:, None]
        tle.gpu.barrier_arrive(pempty[buf], phaseIdx=phase)
        tle.gpu.barrier_arrive(kempty[buf], phaseIdx=phase)
    tle.gpu.barrier_wait(ofull[0], phaseIdx=0)
    inverse = tl.load(tle.gpu.local_ptr(factor))
    tl.store(
        Output + (batch * H + heads[:, None]) * 512 + 256 + dims[None, :],
        acc * inverse[:, None],
    )


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
        C0_REGS: tl.constexpr,
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
        beta = tle.gpu.alloc(
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
        qfull = tle.gpu.alloc_barriers(1, arrive_count=1)
        kfull = tle.gpu.alloc_barriers(2, arrive_count=1)
        kempty = tle.gpu.alloc_barriers(2, arrive_count=2, init=tle.gpu.READY)
        pfull = tle.gpu.alloc_barriers(2, arrive_count=1)
        pempty = tle.gpu.alloc_barriers(2, arrive_count=1, init=tle.gpu.READY)
        ofull = tle.gpu.alloc_barriers(1, arrive_count=1)
        tle.gpu.warp_specialize(
            [
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
                        beta,
                        scales,
                        mask,
                        factor,
                        qfull,
                        kfull,
                        kempty,
                        pfull,
                        pempty,
                        ofull,
                        H,
                        N,
                        TOPK,
                        HAS_LENGTH,
                    ),
                ),
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
                        beta,
                        scales,
                        mask,
                        factor,
                        qfull,
                        kfull,
                        kempty,
                        pfull,
                        pempty,
                        ofull,
                        H,
                        TOPK,
                        SCALE,
                        HAS_LENGTH,
                        HAS_SINK,
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
                        beta,
                        scales,
                        mask,
                        factor,
                        qfull,
                        kfull,
                        kempty,
                        pfull,
                        pempty,
                        ofull,
                        H,
                        TOPK,
                        HAS_LENGTH,
                    ),
                ),
            ],
            [4, 4],
            [C0_REGS, 168],
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
