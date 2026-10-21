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

import math
from typing import Optional

import torch
import triton
import triton.experimental.gluon as g
import triton.experimental.gluon.language as gl
import triton.language as tl
from triton.experimental.gluon.language import (
    BlockedLayout,
    SliceLayout,
    SwizzledSharedLayout,
)
from triton.experimental.gluon.language.amd.cdna3 import buffer_load

from flaggems_vllm import runtime
from flaggems_vllm.ops.flash_mla import _get_num_sms
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.shape_utils import can_use_int32_index


@triton.jit
def accumulate_int8_pv(
    KVBase,
    Valid,
    PHigh,
    PLow,
    Rescale,
    PScale,
    Accumulator,
    OFFSET: tl.constexpr,
    BASE: tl.constexpr,
):
    # Two INT8 products preserve the softmax probabilities to about 14 bits.
    # Reload V in small tiles to bound live registers and shared-memory usage.
    dims = OFFSET + tl.arange(0, 128)
    value = tl.load(KVBase + dims[None, :], Valid[:, None], 0)
    high = tl.dot(tl.trans(value), tl.trans(PHigh), out_dtype=tl.int32)
    # At 64 tokens the combined INT32 magnitude is bounded by 133169152.
    combined = tl.dot(tl.trans(value), tl.trans(PLow), high * BASE, out_dtype=tl.int32)
    return Accumulator * Rescale[None, :] + combined.to(tl.float32) * (
        PScale[None, :] * (1.0 / BASE)
    )


def prune_mla_configs(configs, named_args, **kwargs):
    if named_args["H"] * named_args["SQ"] <= 32:
        return configs
    return [config for config in configs if config.kwargs["QK_BLOCK"] == 128]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dense_int8_mla_kernel"),
    key=["B", "H", "SQ", "SPLITS", "TABLE_STRIDE", "CAUSAL"],
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": prune_mla_configs},
)
@triton.jit
def dense_int8_mla_kernel(
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
    Q_STRIDES: tl.constexpr,
    QR_STRIDES: tl.constexpr,
    KV_STRIDES: tl.constexpr,
    KR_STRIDES: tl.constexpr,
    QS_STRIDES: tl.constexpr,
    KS_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    SQ: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr,
    QK_BLOCK: tl.constexpr,
):
    head_tile, batch, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    heads = head_tile * BH + tl.arange(0, BH)
    queries, qheads = heads // H, heads % H
    merged_heads = SQ * H
    PRELOAD_Q: tl.constexpr = SQ * H <= 32
    PV_BASE: tl.constexpr = 127 if PRELOAD_Q else 128
    content_dims = tl.arange(0, QK_BLOCK)
    rope_dims = tl.arange(0, 64)
    tokens = tl.arange(0, 64)
    q_base = (
        Q
        + batch * Q_STRIDES[0]
        + queries[:, None] * Q_STRIDES[1]
        + qheads[:, None] * Q_STRIDES[2]
    )
    qr = tl.load(
        QR
        + batch * QR_STRIDES[0]
        + queries[:, None] * QR_STRIDES[1]
        + qheads[:, None] * QR_STRIDES[2]
        + rope_dims[None, :],
        heads[:, None] < merged_heads,
        0,
    )
    qs = tl.load(
        QS + batch * QS_STRIDES[0] + queries * QS_STRIDES[1] + qheads * QS_STRIDES[2],
        heads < merged_heads,
        0,
    )
    query_tiles = ()
    if PRELOAD_Q:
        for chunk in tl.static_range(512 // QK_BLOCK):
            query_tile = tl.load(
                q_base + chunk * QK_BLOCK + content_dims[None, :],
                heads[:, None] < merged_heads,
                0,
            )
            query_tiles += (query_tile,)
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    query_length = tl.full((BH,), length, tl.int32)
    if CAUSAL:
        query_length = tl.maximum(length - SQ + queries + 1, 0)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.zeros((BH,), tl.float32)
    acc0 = tl.zeros((128, BH), tl.float32)
    acc1 = tl.zeros((128, BH), tl.float32)
    acc2 = tl.zeros((128, BH), tl.float32)
    acc3 = tl.zeros((128, BH), tl.float32)
    valid_pages = tl.cdiv(length, 64)
    first_page = valid_pages * split // SPLITS
    end_page = valid_pages * (split + 1) // SPLITS
    for page_idx in range(first_page, end_page):
        page = tl.load(Table + batch * TABLE_STRIDE + page_idx).to(tl.int64)
        valid = page_idx * 64 + tokens < length
        kv_base = KV + page * KV_STRIDES[0] + tokens[:, None] * KV_STRIDES[1]
        content_raw = tl.zeros((64, BH), tl.int32)
        if PRELOAD_Q:
            for chunk in tl.static_range(512 // QK_BLOCK):
                dims = chunk * QK_BLOCK + content_dims
                key = tl.load(kv_base + dims[None, :], valid[:, None], 0)
                content_raw = tl.dot(
                    key, tl.trans(query_tiles[chunk]), content_raw, out_dtype=tl.int32
                )
        else:
            # Keeping wide queries across PV spills the output accumulators.
            for chunk in range(4):
                dims = chunk * 128 + content_dims
                q = tl.load(q_base + dims[None, :], heads[:, None] < merged_heads, 0)
                key = tl.load(kv_base + dims[None, :], valid[:, None], 0)
                content_raw = tl.dot(key, tl.trans(q), content_raw, out_dtype=tl.int32)
        kr = tl.load(
            KR
            + page * KR_STRIDES[0]
            + tokens[:, None] * KR_STRIDES[1]
            + rope_dims[None, :],
            valid[:, None],
            0,
        )
        ks = tl.load(KS + page * KS_STRIDES[0] + tokens * KS_STRIDES[1], valid, 0)
        content_scores = tl.trans(content_raw).to(tl.float32)
        rope_scores = tl.trans(tl.dot(kr, tl.trans(qr), out_dtype=tl.float32))
        scores = (
            (content_scores + rope_scores)
            * qs[:, None]
            * ks[None, :]
            * (SCALE * 1.4426950408889634)
        )
        scores = tl.where(
            page_idx * 64 + tokens[None, :] < query_length[:, None],
            scores,
            -float("inf"),
        )
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        probabilities = tl.exp2(scores - safe_maximum[:, None])
        rescale = tl.exp2(maximum - safe_maximum)
        denominator = denominator * rescale + tl.sum(probabilities, 1)
        scaled = probabilities * ks[None, :]
        probability_scale = tl.maximum(tl.max(scaled, 1) / 127.0, 1.0e-30)
        scaled /= probability_scale[:, None]
        if PRELOAD_Q:
            # The small-head query-preload path benefits from float residuals.
            high = tl.floor(scaled + 0.5).to(tl.int8)
            low = tl.floor((scaled - high.to(tl.float32)) * 127.0 + 0.5).to(tl.int8)
        else:
            # scaled is in [0, 127]; the biased FP32 ULP is 1/128.
            quant = (scaled + 65536.0).to(tl.int32, bitcast=True) - 0x47800000
            high_raw = (quant + 64) >> 7
            high = high_raw.to(tl.int8)
            low = (quant - (high_raw << 7)).to(tl.int8)
        acc0 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc0, 0, PV_BASE
        )
        acc1 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc1, 128, PV_BASE
        )
        acc2 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc2, 256, PV_BASE
        )
        acc3 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc3, 384, PV_BASE
        )
        maximum = next_maximum
    out_dims = tl.arange(0, 128)
    if SPLITS == 1:
        inverse = 1.0 / tl.where(denominator > 0, denominator, 1.0)
        base = (
            Output + (batch * merged_heads + heads[None, :]) * 512 + out_dims[:, None]
        )
        tl.store(base, acc0 * inverse[None, :], heads[None, :] < merged_heads)
        tl.store(base + 128, acc1 * inverse[None, :], heads[None, :] < merged_heads)
        tl.store(base + 256, acc2 * inverse[None, :], heads[None, :] < merged_heads)
        tl.store(base + 384, acc3 * inverse[None, :], heads[None, :] < merged_heads)
        logsum = tl.where(
            denominator > 0,
            maximum * 0.6931471805599453 + tl.log(denominator),
            float("inf"),
        )
        tl.store(
            LSE + (batch * H + qheads) * SQ + queries, logsum, heads < merged_heads
        )
    else:
        base = (
            Partial
            + ((split * B + batch) * merged_heads + heads[None, :]) * 512
            + out_dims[:, None]
        )
        tl.store(base, acc0, heads[None, :] < merged_heads)
        tl.store(base + 128, acc1, heads[None, :] < merged_heads)
        tl.store(base + 256, acc2, heads[None, :] < merged_heads)
        tl.store(base + 384, acc3, heads[None, :] < merged_heads)
        tl.store(
            Stats + (split * 2 * B + batch) * merged_heads + heads,
            maximum,
            heads < merged_heads,
        )
        tl.store(
            Stats + ((split * 2 + 1) * B + batch) * merged_heads + heads,
            denominator,
            heads < merged_heads,
        )


@libentry()
@triton.jit
def dense_int8_mla_merge(
    Partial,
    Stats,
    Output,
    LSE,
    B: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    SPLITS: tl.constexpr,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, 512)
    merged_heads = H * SQ
    maxima = tl.load(Stats + (splits * 2 * B + batch) * merged_heads + head)
    maximum = tl.max(maxima, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp2(maxima - safe_maximum)
    sums = tl.load(Stats + ((splits * 2 + 1) * B + batch) * merged_heads + head)
    denominator = tl.sum(sums * weights, 0)
    values = tl.load(
        Partial
        + ((splits[:, None] * B + batch) * merged_heads + head) * 512
        + dims[None, :]
    )
    output = tl.sum(values * weights[:, None], 0) / tl.where(
        denominator > 0, denominator, 1.0
    )
    tl.store(Output + (batch * merged_heads + head) * 512 + dims, output)
    logsum = tl.where(
        denominator > 0,
        maximum * 0.6931471805599453 + tl.log(denominator),
        float("inf"),
    )
    tl.store(LSE + (batch * H + head % H) * SQ + head // H, logsum)


@libentry()
@triton.jit
def _staged_int8_mla_qk(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Packed,
    Stats,
    Q_STRIDES: tl.constexpr,
    QR_STRIDES: tl.constexpr,
    KV_STRIDES: tl.constexpr,
    KR_STRIDES: tl.constexpr,
    QS_STRIDES: tl.constexpr,
    KS_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr,
    QK_BLOCK: tl.constexpr,
    PAGES: tl.constexpr,
):
    ht, first_page, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    first_page *= PAGES
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    if first_page * 64 >= length:
        return
    heads = ht * BH + tl.arange(0, BH)
    queries, qheads = heads // H, heads % H
    mh: tl.constexpr = H * SQ
    tokens = tl.arange(0, 64)
    dims = tl.arange(0, QK_BLOCK)
    rd = tl.arange(0, 64)
    qb = (
        Q
        + batch * Q_STRIDES[0]
        + queries[:, None] * Q_STRIDES[1]
        + qheads[:, None] * Q_STRIDES[2]
    )
    # Reuse Q across cache pages without retaining PV output accumulators.
    if PAGES > 1:
        query_tiles = ()
        for chunk in tl.static_range(512 // QK_BLOCK):
            query = tl.load(
                qb + chunk * QK_BLOCK + dims[None, :], heads[:, None] < mh, 0
            )
            query_tiles += (query,)
        qr = tl.load(
            QR
            + batch * QR_STRIDES[0]
            + queries[:, None] * QR_STRIDES[1]
            + qheads[:, None] * QR_STRIDES[2]
            + rd[None, :],
            heads[:, None] < mh,
            0,
        )
        qs = tl.load(
            QS
            + batch * QS_STRIDES[0]
            + queries * QS_STRIDES[1]
            + qheads * QS_STRIDES[2],
            heads < mh,
            0,
        )
    if PAGES == 1:
        pages_in_group = 1
    else:
        pages_in_group = tl.minimum(PAGES, tl.cdiv(length, 64) - first_page)
    for page_offset in range(pages_in_group):
        page_idx = first_page + page_offset
        page = tl.load(Table + batch * TABLE_STRIDE + page_idx).to(tl.int64)
        valid = page_idx * 64 + tokens < length
        kb = KV + page * KV_STRIDES[0] + tokens[:, None] * KV_STRIDES[1]
        content = tl.zeros((BH, 64), tl.int32)
        if PAGES > 1:
            for chunk in tl.static_range(512 // QK_BLOCK):
                key = tl.load(kb + chunk * QK_BLOCK + dims[None, :], valid[:, None], 0)
                content = tl.dot(
                    query_tiles[chunk], tl.trans(key), content, out_dtype=tl.int32
                )
        else:
            for chunk in range(512 // QK_BLOCK):
                q = tl.load(
                    qb + chunk * QK_BLOCK + dims[None, :], heads[:, None] < mh, 0
                )
                k = tl.load(kb + chunk * QK_BLOCK + dims[None, :], valid[:, None], 0)
                content = tl.dot(q, tl.trans(k), content, out_dtype=tl.int32)
        if PAGES == 1:
            qr = tl.load(
                QR
                + batch * QR_STRIDES[0]
                + queries[:, None] * QR_STRIDES[1]
                + qheads[:, None] * QR_STRIDES[2]
                + rd[None, :],
                heads[:, None] < mh,
                0,
            )
        kr = tl.load(
            KR + page * KR_STRIDES[0] + tokens[:, None] * KR_STRIDES[1] + rd[None, :],
            valid[:, None],
            0,
        )
        if PAGES == 1:
            qs = tl.load(
                QS
                + batch * QS_STRIDES[0]
                + queries * QS_STRIDES[1]
                + qheads * QS_STRIDES[2],
                heads < mh,
                0,
            )
        ks = tl.load(KS + page * KS_STRIDES[0] + tokens * KS_STRIDES[1], valid, 0)
        score = (
            (content.to(tl.float32) + tl.dot(qr, tl.trans(kr), out_dtype=tl.float32))
            * qs[:, None]
            * ks[None, :]
            * (SCALE * 1.4426950408889634)
        )
        if CAUSAL:
            query_length = tl.maximum(length - SQ + queries + 1, 0)
        else:
            query_length = tl.full((BH,), length, tl.int32)
        score = tl.where(
            page_idx * 64 + tokens[None, :] < query_length[:, None],
            score,
            -float("inf"),
        )
        maximum = tl.max(score, 1)
        safe = tl.where(maximum == -float("inf"), 0.0, maximum)
        prob = tl.exp2(score - safe[:, None])
        denom = tl.sum(prob, 1)
        scaled = prob * ks[None, :]
        ps = tl.maximum(tl.max(scaled, 1) / 65535.0, 1.0e-30)
        normalized = scaled / ps[:, None]
        code = (normalized + 0.5).to(tl.int32)
        offset = ((batch * CAPACITY + page_idx) * mh + heads[:, None]) * 64 + tokens[
            None, :
        ]
        # Convert through INT32 before truncation: 65535 exceeds signed INT16.
        tl.store(Packed + offset, code.to(tl.int16), heads[:, None] < mh)
        stat_base = ((batch * CAPACITY + page_idx) * mh + heads) * 3
        tl.store(Stats + stat_base, maximum, heads < mh)
        tl.store(Stats + stat_base + 1, denom, heads < mh)
        tl.store(Stats + stat_base + 2, ps, heads < mh)


@libentry()
@triton.jit
def _staged_int8_mla_normalize(
    Stats,
    Weight,
    RowFactor,
    LSE,
    Lengths,
    CAPACITY: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    NP: tl.constexpr,
):
    head, batch = tl.program_id(0), tl.program_id(1)
    mh: tl.constexpr = H * SQ
    pages = tl.cdiv(tl.load(Lengths + batch * LENGTH_STRIDE), 64)
    idx = tl.arange(0, NP)
    base = ((batch * CAPACITY + idx) * mh + head) * 3
    m = tl.load(Stats + base, idx < pages, -float("inf"))
    d = tl.load(Stats + base + 1, idx < pages, 0.0)
    ps = tl.load(Stats + base + 2, idx < pages, 0.0)
    maximum = tl.max(m, 0)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    beta = tl.exp2(m - safe)
    denom = tl.sum(d * beta, 0)
    inv = 1.0 / tl.where(denom > 0, denom, 1.0)
    global_scale = tl.maximum(tl.max(ps * beta, 0) * (65535.0 / 127.0), 1.0e-30)
    factor = beta * ps / global_scale
    tl.store(Weight + (batch * CAPACITY + idx) * mh + head, factor, idx < pages)
    tl.store(RowFactor + batch * mh + head, global_scale * ((1.0 / 128.0) * inv))
    lse = tl.where(
        denom > 0, maximum * 0.6931471805599453 + tl.log(denom), float("inf")
    )
    tl.store(LSE + (batch * H + head % H) * SQ + head // H, lse)


@libentry()
@triton.jit
def _staged_int8_mla_requant(
    Packed,
    Weight,
    Lengths,
    CAPACITY: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    BH: tl.constexpr,
):
    ht, page, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    if page * 64 >= length:
        return
    mh: tl.constexpr = H * SQ
    heads = ht * BH + tl.arange(0, BH)
    tokens = tl.arange(0, 64)
    base = (batch * CAPACITY + page) * mh + heads
    offset = base[:, None] * 64 + tokens[None, :]
    raw = tl.load(Packed + offset, heads[:, None] < mh, 0).to(tl.int32) & 65535
    scale = tl.load(Weight + base, heads < mh, 0.0)
    normalized = tl.minimum(raw.to(tl.float32) * scale[:, None], 127.0)
    quant = (normalized + 65536.0).to(tl.int32, bitcast=True) - 0x47800000
    high = (quant + 64) >> 7
    low = quant - (high << 7)
    code = high | ((low & 255) << 8)
    tl.store(Packed + offset, code.to(tl.int16), heads[:, None] < mh)


@libentry()
@triton.jit
def _staged_int8_mla_pv(
    KV,
    Table,
    Lengths,
    Packed,
    RowFactor,
    Output,
    KV_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
):
    dt, ht, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    mh: tl.constexpr = H * SQ
    heads = ht * BH + tl.arange(0, BH)
    dims = dt * BD + tl.arange(0, BD)
    tokens = tl.arange(0, 64)
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    pages = tl.cdiv(length, 64)
    tl.static_assert(CAPACITY <= 2048)
    # At 131072 tokens, |high| <= 2130706432 and |low| <= 1073741824.
    # Keep the limbs separate until FP32 conversion to avoid INT32 overflow.
    hi = tl.zeros((BD, BH), tl.int32)
    lo = tl.zeros((BD, BH), tl.int32)
    for page_idx in range(pages):
        page = tl.load(Table + batch * TABLE_STRIDE + page_idx).to(tl.int64)
        valid = page_idx * 64 + tokens < length
        value = tl.load(
            KV + page * KV_STRIDES[0] + tokens[None, :] * KV_STRIDES[1] + dims[:, None],
            valid[None, :],
            0,
        )
        offset = ((batch * CAPACITY + page_idx) * mh + heads[None, :]) * 64 + tokens[
            :, None
        ]
        code = tl.load(Packed + offset, heads[None, :] < mh, 0)
        high = code.to(tl.int8)
        low = (code >> 8).to(tl.int8)
        hi = tl.dot(value, high, hi, out_dtype=tl.int32)
        lo = tl.dot(value, low, lo, out_dtype=tl.int32)
    total = hi.to(tl.float32) * 128.0 + lo.to(tl.float32)
    factor = tl.load(RowFactor + batch * mh + heads, heads < mh, 0.0)
    tl.store(
        Output + (batch * mh + heads[None, :]) * 512 + dims[:, None],
        total * factor[None, :],
        heads[None, :] < mh,
    )


@g.jit
def expand_mla_pv_word(value, N: gl.constexpr):
    R: gl.constexpr = BlockedLayout([1, 1, 1], [16, 1, 4], [4, 1, 1], [0, 2, 1])
    C: gl.constexpr = BlockedLayout([1, 1], [16, 4], [4, 1], [0, 1])
    value = gl.convert_layout(value, SliceLayout(1, R))
    result = value[:, None, :] + gl.full((64, N // 4, 4), 0, value.dtype, R)
    return gl.convert_layout(result.reshape((64, N)), C)


@g.jit
def interleave_mla_pv_bytes(first, second, selector):
    return gl.inline_asm_elementwise(
        "v_perm_b32 $0, $2, $1, $3",
        constraints="=v,v,v,v",
        args=[first, second, selector],
        dtype=gl.uint32,
        is_pure=True,
        pack=1,
    )


@g.jit
def accumulate_mla_pv_limbs(High, Low, Base, HE, HO, LE, LO):
    return gl.inline_asm_elementwise(
        """s_waitcnt lgkmcnt(0)
v_xor_b32 v63, 64, $192
ds_read_m32x16_b16_alt v[64:67], $192 offset:0
ds_read_m32x16_b16_alt v[68:71], v63 offset:0
ds_read_m32x16_b16_alt v[72:75], $192 offset:128
ds_read_m32x16_b16_alt v[76:79], v63 offset:128
ds_read_m32x16_b16_alt v[80:83], $192 offset:256
ds_read_m32x16_b16_alt v[84:87], v63 offset:256
ds_read_m32x16_b16_alt v[88:91], $192 offset:384
ds_read_m32x16_b16_alt v[92:95], v63 offset:384
s_waitcnt lgkmcnt(7)
s_nop 0
v_mmac_i32_16x16x32_i8 v[96:99], $128, v[64:65], v[96:99]
v_mmac_i32_16x16x32_i8 v[128:131], $128, v[66:67], v[128:131]
v_mmac_i32_16x16x32_i8 v[160:163], $160, v[64:65], v[160:163]
v_mmac_i32_16x16x32_i8 v[192:195], $160, v[66:67], v[192:195]
s_nop 0
s_waitcnt lgkmcnt(6)
s_nop 0
v_mmac_i32_16x16x32_i8 v[100:103], $128, v[68:69], v[100:103]
v_mmac_i32_16x16x32_i8 v[132:135], $128, v[70:71], v[132:135]
v_mmac_i32_16x16x32_i8 v[164:167], $160, v[68:69], v[164:167]
v_mmac_i32_16x16x32_i8 v[196:199], $160, v[70:71], v[196:199]
s_nop 0
s_waitcnt lgkmcnt(5)
s_nop 0
v_mmac_i32_16x16x32_i8 v[104:107], $128, v[72:73], v[104:107]
v_mmac_i32_16x16x32_i8 v[136:139], $128, v[74:75], v[136:139]
v_mmac_i32_16x16x32_i8 v[168:171], $160, v[72:73], v[168:171]
v_mmac_i32_16x16x32_i8 v[200:203], $160, v[74:75], v[200:203]
s_nop 0
s_waitcnt lgkmcnt(4)
s_nop 0
v_mmac_i32_16x16x32_i8 v[108:111], $128, v[76:77], v[108:111]
v_mmac_i32_16x16x32_i8 v[140:143], $128, v[78:79], v[140:143]
v_mmac_i32_16x16x32_i8 v[172:175], $160, v[76:77], v[172:175]
v_mmac_i32_16x16x32_i8 v[204:207], $160, v[78:79], v[204:207]
s_nop 0
s_waitcnt lgkmcnt(3)
s_nop 0
v_mmac_i32_16x16x32_i8 v[112:115], $128, v[80:81], v[112:115]
v_mmac_i32_16x16x32_i8 v[144:147], $128, v[82:83], v[144:147]
v_mmac_i32_16x16x32_i8 v[176:179], $160, v[80:81], v[176:179]
v_mmac_i32_16x16x32_i8 v[208:211], $160, v[82:83], v[208:211]
s_nop 0
s_waitcnt lgkmcnt(2)
s_nop 0
v_mmac_i32_16x16x32_i8 v[116:119], $128, v[84:85], v[116:119]
v_mmac_i32_16x16x32_i8 v[148:151], $128, v[86:87], v[148:151]
v_mmac_i32_16x16x32_i8 v[180:183], $160, v[84:85], v[180:183]
v_mmac_i32_16x16x32_i8 v[212:215], $160, v[86:87], v[212:215]
s_nop 0
s_waitcnt lgkmcnt(1)
s_nop 0
v_mmac_i32_16x16x32_i8 v[120:123], $128, v[88:89], v[120:123]
v_mmac_i32_16x16x32_i8 v[152:155], $128, v[90:91], v[152:155]
v_mmac_i32_16x16x32_i8 v[184:187], $160, v[88:89], v[184:187]
v_mmac_i32_16x16x32_i8 v[216:219], $160, v[90:91], v[216:219]
s_nop 0
s_waitcnt lgkmcnt(0)
s_nop 0
v_mmac_i32_16x16x32_i8 v[124:127], $128, v[92:93], v[124:127]
v_mmac_i32_16x16x32_i8 v[156:159], $128, v[94:95], v[156:159]
v_mmac_i32_16x16x32_i8 v[188:191], $160, v[92:93], v[188:191]
v_mmac_i32_16x16x32_i8 v[220:223], $160, v[94:95], v[220:223]
s_nop 0""",
        constraints=(
            "=&{v96},=&{v97},=&{v98},=&{v99},=&{v100},=&{v101},=&{v102},=&{v103},"
            "=&{v104},=&{v105},=&{v106},=&{v107},=&{v108},=&{v109},=&{v110},=&{v111},"
            "=&{v112},=&{v113},=&{v114},=&{v115},=&{v116},=&{v117},=&{v118},=&{v119},"
            "=&{v120},=&{v121},=&{v122},=&{v123},=&{v124},=&{v125},=&{v126},=&{v127},"
            "=&{v128},=&{v129},=&{v130},=&{v131},=&{v132},=&{v133},=&{v134},=&{v135},"
            "=&{v136},=&{v137},=&{v138},=&{v139},=&{v140},=&{v141},=&{v142},=&{v143},"
            "=&{v144},=&{v145},=&{v146},=&{v147},=&{v148},=&{v149},=&{v150},=&{v151},"
            "=&{v152},=&{v153},=&{v154},=&{v155},=&{v156},=&{v157},=&{v158},=&{v159},"
            "=&{v160},=&{v161},=&{v162},=&{v163},=&{v164},=&{v165},=&{v166},=&{v167},"
            "=&{v168},=&{v169},=&{v170},=&{v171},=&{v172},=&{v173},=&{v174},=&{v175},"
            "=&{v176},=&{v177},=&{v178},=&{v179},=&{v180},=&{v181},=&{v182},=&{v183},"
            "=&{v184},=&{v185},=&{v186},=&{v187},=&{v188},=&{v189},=&{v190},=&{v191},"
            "=&{v192},=&{v193},=&{v194},=&{v195},=&{v196},=&{v197},=&{v198},=&{v199},"
            "=&{v200},=&{v201},=&{v202},=&{v203},=&{v204},=&{v205},=&{v206},=&{v207},"
            "=&{v208},=&{v209},=&{v210},=&{v211},=&{v212},=&{v213},=&{v214},=&{v215},"
            "=&{v216},=&{v217},=&{v218},=&{v219},=&{v220},=&{v221},=&{v222},=&{v223},"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "0,1,2,3,4,5,6,7,"
            "8,9,10,11,12,13,14,15,"
            "16,17,18,19,20,21,22,23,"
            "24,25,26,27,28,29,30,31,"
            "32,33,34,35,36,37,38,39,"
            "40,41,42,43,44,45,46,47,"
            "48,49,50,51,52,53,54,55,"
            "56,57,58,59,60,61,62,63,"
            "64,65,66,67,68,69,70,71,"
            "72,73,74,75,76,77,78,79,"
            "80,81,82,83,84,85,86,87,"
            "88,89,90,91,92,93,94,95,"
            "96,97,98,99,100,101,102,103,"
            "104,105,106,107,108,109,110,111,"
            "112,113,114,115,116,117,118,119,"
            "120,121,122,123,124,125,126,127,"
            "~{v64},~{v65},~{v66},~{v67},~{v68},~{v69},~{v70},~{v71},"
            "~{v72},~{v73},~{v74},~{v75},~{v76},~{v77},~{v78},~{v79},"
            "~{v80},~{v81},~{v82},~{v83},~{v84},~{v85},~{v86},~{v87},"
            "~{v88},~{v89},~{v90},~{v91},~{v92},~{v93},~{v94},~{v95},"
            "~{v63},~{memory}"
        ),
        args=[High, Low, Base, HE, HO, LE, LO],
        dtype=(gl.int32, gl.int32, gl.int32, gl.int32),
        is_pure=False,
        pack=32,
    )


@g.jit
def native_int8_mla_pv(
    KV,
    Table,
    Lengths,
    Packed,
    RowFactor,
    Output,
    KV_STRIDES: gl.constexpr,
    TABLE_STRIDE: gl.constexpr,
    CAPACITY: gl.constexpr,
    LENGTH_STRIDE: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
):
    PAGE_GROUP: gl.constexpr = 2
    SHARED_BYTES: gl.constexpr = 16384 * PAGE_GROUP
    C: gl.constexpr = BlockedLayout([1, 1], [16, 4], [4, 1], [0, 1])
    W: gl.constexpr = BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    P3: gl.constexpr = BlockedLayout([1, 1, 2], [16, 4, 1], [4, 1, 1], [0, 2, 1])
    P2: gl.constexpr = SliceLayout(2, P3)
    dt, ht, batch = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    mh: gl.constexpr = H * SQ
    local_heads = gl.arange(0, 64, layout=SliceLayout(1, C))
    heads = ht * 64 + local_heads
    groups = gl.arange(0, 4, layout=SliceLayout(0, C))
    ph = ht * 64 + gl.arange(0, 64, layout=SliceLayout(1, P2))
    pg = gl.arange(0, 4, layout=SliceLayout(0, P2))
    pj = gl.arange(0, 2, layout=SliceLayout(0, SliceLayout(1, P3)))
    length = gl.load(Lengths + batch * LENGTH_STRIDE)
    pages = gl.cdiv(length, 64)
    shared = gl.allocate_shared_memory(
        gl.uint8, (SHARED_BYTES,), SwizzledSharedLayout(16, 1, 1, [0])
    )
    paired = shared._reinterpret(
        gl.uint32, (32 * PAGE_GROUP, 128), SwizzledSharedLayout(4, 1, 8, [1, 0])
    )
    tail = shared.slice(SHARED_BYTES - 1024, 1024, 0)._reinterpret(
        gl.uint32, (64, 4), SwizzledSharedLayout(1, 1, 1, [0, 1])
    )
    tail.store(gl.full((64, 4), 0, gl.uint32, C))
    gl.thread_barrier()
    zero = gl.inline_asm_elementwise(
        "v_and_b32 $0, $1, 0",
        constraints="=v,v,~{memory}",
        args=[tail.load(C)],
        dtype=gl.uint32,
        is_pure=False,
        pack=1,
    )
    # Each shared halfword holds one feature from two adjacent INT8 tokens.
    source_pair = (local_heads[:, None] & 15) // 4 + groups[None, :] * 4
    base_raw = (
        source_pair * 512
        + (((local_heads[:, None] & 3) * 8) ^ ((source_pair & 7) * 8)) * 2
        + zero
    )
    base = expand_mla_pv_word(base_raw, 128)
    pairs = gl.arange(0, 32, layout=SliceLayout(1, W))
    features = dt * 64 + gl.arange(0, 64, layout=SliceLayout(0, W))
    # Separate P14 limbs stay within INT32 at the 131072-token capacity limit.
    he = gl.full((64, 128), 0, gl.int32, C)
    ho = he
    le = he
    lo = he
    for group in tl.range(0, gl.cdiv(pages, PAGE_GROUP), loop_unroll_factor=1):
        highs = ()
        lows = ()
        gl.thread_barrier()
        for page_offset in gl.static_range(PAGE_GROUP):
            page_idx = group * PAGE_GROUP + page_offset
            page = gl.load(
                Table + batch * TABLE_STRIDE + page_idx, page_idx < pages, 0
            ).to(gl.int64)
            for half in gl.static_range(2):
                poffset = (
                    ph[:, None, None] * 16
                    + half * 8
                    + pg[None, :, None] * 2
                    + pj[None, None, :]
                )
                pwords = buffer_load(
                    Packed.to(tl.pointer_type(gl.uint64))
                    + (batch * CAPACITY + page_idx) * mh * 16,
                    poffset,
                    (ph[:, None, None] < mh) & (page_idx < pages),
                    0,
                )
                p0, p1 = gl.split(pwords)
                p00 = p0.to(gl.uint32)
                p01 = (p0 >> 32).to(gl.uint32)
                p10 = p1.to(gl.uint32)
                p11 = (p1 >> 32).to(gl.uint32)
                high0 = interleave_mla_pv_bytes(p00, p01, 0x06040200)
                high1 = interleave_mla_pv_bytes(p10, p11, 0x06040200)
                low0 = interleave_mla_pv_bytes(p00, p01, 0x07050301)
                low1 = interleave_mla_pv_bytes(p10, p11, 0x07050301)
                highs += (
                    expand_mla_pv_word(
                        high0.to(gl.uint64) | (high1.to(gl.uint64) << 32), 128
                    ),
                )
                lows += (
                    expand_mla_pv_word(
                        low0.to(gl.uint64) | (low1.to(gl.uint64) << 32), 128
                    ),
                )
            first_token = 2 * pairs
            first = buffer_load(
                KV.to(tl.pointer_type(gl.uint32)) + page * (KV_STRIDES[0] // 4),
                first_token[:, None] * (KV_STRIDES[1] // 4) + features[None, :],
                page_idx * 64 + first_token[:, None] < length,
                0,
            )
            second = buffer_load(
                KV.to(tl.pointer_type(gl.uint32)) + page * (KV_STRIDES[0] // 4),
                (first_token[:, None] + 1) * (KV_STRIDES[1] // 4) + features[None, :],
                page_idx * 64 + first_token[:, None] + 1 < length,
                0,
            )
            low_pair = interleave_mla_pv_bytes(first, second, 0x05010400)
            high_pair = interleave_mla_pv_bytes(first, second, 0x07030602)
            packed_pairs = gl.join(low_pair, high_pair).reshape((32, 128))
            paired.slice(page_offset * 32, 32, 0).store(packed_pairs)
        gl.thread_barrier()
        for half in gl.static_range(2 * PAGE_GROUP):
            he, ho, le, lo = accumulate_mla_pv_limbs(
                highs[half], lows[half], base + half * 8192, he, ho, le, lo
            )
    factor = buffer_load(RowFactor + batch * mh, heads, heads < mh, 0.0)
    even = (he.to(gl.float32) * 128.0 + le.to(gl.float32)) * factor[:, None]
    odd = (ho.to(gl.float32) * 128.0 + lo.to(gl.float32)) * factor[:, None]
    OUT: gl.constexpr = BlockedLayout([1, 2], [16, 4], [4, 1], [0, 1])
    result = gl.convert_layout(gl.join(even, odd).reshape((64, 256)), OUT)
    oh = ht * 64 + gl.arange(0, 64, layout=SliceLayout(1, OUT))
    of = dt * 256 + gl.arange(0, 256, layout=SliceLayout(0, OUT))
    gl.store(
        Output + (batch * mh + oh[:, None]) * 512 + of[None, :],
        result,
        oh[:, None] < mh,
    )


class FlashMLAInt8PreparedHandle:
    """Reuse decode workspace; callers must serialize calls on one stream."""

    def __init__(
        self,
        tensors: tuple[torch.Tensor, ...],
        splits: int,
        scale: float,
        causal: bool,
        automatic_splits: bool = False,
    ):
        self.tensors = tensors
        query, rope = tensors[:2]
        self.batch, self.queries, self.heads, _ = query.shape
        self.scale, self.causal = scale, causal
        self.output = torch.empty(
            (self.batch, self.queries, self.heads, 512),
            dtype=rope.dtype,
            device=query.device,
        )
        self.lse = torch.empty(
            (self.batch, self.heads, self.queries),
            dtype=torch.float32,
            device=query.device,
        )
        self.capacity = tensors[6].shape[1]
        merged_heads = self.queries * self.heads
        # Bound the staged probability and page-statistics workspace to 4 GiB.
        staged_bytes = (
            self.batch * merged_heads * ((64 * 2 + 4 * 4) * self.capacity + 4)
        )
        self.staged = (
            (
                (splits == 1 and merged_heads >= 64)
                or (automatic_splits and merged_heads == 32 and self.batch >= 128)
            )
            and 64 <= self.capacity <= 2048
            and staged_bytes <= 4 * 1024**3
        )
        self.splits = 1 if self.staged else splits
        if self.staged:
            self.packed = torch.empty(
                (self.batch, self.capacity, merged_heads, 64),
                dtype=torch.int16,
                device=query.device,
            )
            self.staged_stats = torch.empty(
                (self.batch, self.capacity, merged_heads, 3),
                dtype=torch.float32,
                device=query.device,
            )
            self.weight = torch.empty(
                (self.batch, self.capacity, merged_heads),
                dtype=torch.float32,
                device=query.device,
            )
            self.rowfactor = torch.empty(
                (self.batch, merged_heads),
                dtype=torch.float32,
                device=query.device,
            )
        else:
            workspace_splits = splits if splits > 1 else 0
            self.partial = torch.empty(
                (workspace_splits, self.batch, self.queries * self.heads, 512),
                dtype=torch.float32,
                device=query.device,
            )
            self.stats = torch.empty(
                (workspace_splits, 2, self.batch, self.queries * self.heads),
                dtype=torch.float32,
                device=query.device,
            )

        self.has_native_pv = False
        kv, table, lengths = tensors[2], tensors[6], tensors[7]
        if (
            self.staged
            and merged_heads >= 128
            and rope.dtype == torch.bfloat16
            and self.batch * triton.cdiv(merged_heads, 64) * 2
            >= _get_num_sms(query.device)
            and 63 * kv.stride(1) + 512 < 2**31
            and kv.stride(-1) == 1
            and kv.data_ptr() % 16 == 0
            and kv.stride(0) % 16 == 0
            and kv.stride(1) % 16 == 0
            and torch.cuda.get_device_properties(query.device).gcnArchName.startswith(
                "gfx936"
            )
        ):
            with torch_device_fn.device(query.device):
                binary = native_int8_mla_pv.warmup(
                    kv,
                    table,
                    lengths,
                    self.packed,
                    self.rowfactor,
                    self.output,
                    kv.stride(),
                    table.stride(0),
                    self.capacity,
                    lengths.stride(0),
                    self.heads,
                    self.queries,
                    grid=(2, triton.cdiv(merged_heads, 64), self.batch),
                    num_warps=4,
                    num_stages=1,
                )
            # A full-span allocation of exactly 32 KiB fixes the native LDS base.
            self.has_native_pv = (
                binary.metadata.shared == 32768 and binary.metadata.warp_size == 64
            )

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch == 0 or self.heads == 0:
            return self.output, self.lse
        query, rope, kv, kv_rope, qs, ks, table, lengths = self.tensors
        with torch_device_fn.device(query.device):
            if self.staged:
                merged_heads = self.queries * self.heads
                page_group = 4 if merged_heads >= 64 else 1
                qk_heads = 32 if merged_heads == 32 else 128
                qk_block = 512 if merged_heads == 32 else 128
                pv_heads = 32 if merged_heads == 32 else 64
                _staged_int8_mla_qk[
                    (
                        triton.cdiv(merged_heads, qk_heads),
                        triton.cdiv(self.capacity, page_group),
                        self.batch,
                    )
                ](
                    *self.tensors,
                    self.packed,
                    self.staged_stats,
                    query.stride(),
                    rope.stride(),
                    kv.stride(),
                    kv_rope.stride(),
                    qs.stride(),
                    ks.stride(),
                    table.stride(0),
                    self.capacity,
                    lengths.stride(0),
                    self.batch,
                    self.heads,
                    self.queries,
                    self.scale,
                    self.causal,
                    qk_heads,
                    qk_block,
                    page_group,
                    num_warps=4,
                    num_stages=1,
                )
                _staged_int8_mla_normalize[(merged_heads, self.batch)](
                    self.staged_stats,
                    self.weight,
                    self.rowfactor,
                    self.lse,
                    lengths,
                    self.capacity,
                    lengths.stride(0),
                    self.heads,
                    self.queries,
                    triton.next_power_of_2(self.capacity),
                    num_warps=4,
                )
                _staged_int8_mla_requant[
                    (triton.cdiv(merged_heads, 32), self.capacity, self.batch)
                ](
                    self.packed,
                    self.weight,
                    lengths,
                    self.capacity,
                    lengths.stride(0),
                    self.heads,
                    self.queries,
                    32,
                    num_warps=4,
                )
                if self.has_native_pv:
                    native_int8_mla_pv[(2, triton.cdiv(merged_heads, 64), self.batch)](
                        kv,
                        table,
                        lengths,
                        self.packed,
                        self.rowfactor,
                        self.output,
                        kv.stride(),
                        table.stride(0),
                        self.capacity,
                        lengths.stride(0),
                        self.heads,
                        self.queries,
                        num_warps=4,
                        num_stages=1,
                    )
                else:
                    _staged_int8_mla_pv[
                        (2, triton.cdiv(merged_heads, pv_heads), self.batch)
                    ](
                        kv,
                        table,
                        lengths,
                        self.packed,
                        self.rowfactor,
                        self.output,
                        kv.stride(),
                        table.stride(0),
                        self.capacity,
                        lengths.stride(0),
                        self.heads,
                        self.queries,
                        pv_heads,
                        256,
                        num_warps=4,
                        num_stages=1,
                    )
                return self.output, self.lse
            dense_int8_mla_kernel[
                lambda meta: (
                    triton.cdiv(self.queries * self.heads, meta["BH"]),
                    self.batch,
                    self.splits,
                )
            ](
                *self.tensors,
                self.partial,
                self.stats,
                self.output,
                self.lse,
                query.stride(),
                rope.stride(),
                kv.stride(),
                kv_rope.stride(),
                qs.stride(),
                ks.stride(),
                table.stride(0),
                lengths.stride(0),
                self.batch,
                self.heads,
                self.splits,
                self.scale,
                self.queries,
                self.causal,
            )
            if self.splits > 1:
                dense_int8_mla_merge[(self.batch, self.queries * self.heads)](
                    self.partial,
                    self.stats,
                    self.output,
                    self.lse,
                    self.batch,
                    self.heads,
                    self.queries,
                    self.splits,
                    num_warps=4,
                )
        return self.output, self.lse


def prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_lora: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int = 512,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    num_splits: Optional[int] = None,
) -> tuple[FlashMLAInt8PreparedHandle, tuple[torch.Tensor, torch.Tensor]]:
    """Decode 64-token pages; RoPE is divided by the content scale as in PR851.

    Active page IDs and lengths must be valid, scales finite and positive.
    Causal queries attend the aligned suffix of the valid cache.
    """
    if q_nope.ndim != 4 or q_nope.shape[1] not in (1, 2) or q_nope.shape[-1] != 512:
        raise NotImplementedError("requires Q [batch, 1 or 2, heads, 512]")
    if q_nope.device.type != "cuda":
        raise ValueError("requires Hygon device tensors")
    batch, queries, heads, _ = q_nope.shape
    if (
        k_lora.ndim not in (3, 4)
        or k_lora.shape[1] != 64
        or k_lora.shape[-1] != 512
        or (k_lora.ndim == 4 and k_lora.shape[2] != 1)
        or head_dim_v != 512
    ):
        raise NotImplementedError("requires KV [pages, 64, 512] and head_dim_v=512")
    expected_shapes = (
        (batch, queries, heads, 64),
        (*k_lora.shape[:-1], 64),
        (batch, queries, heads, 1),
        (*k_lora.shape[:-1], 1),
        (batch,),
    )
    for tensor, shape in zip(
        (q_rope, k_rope, q_scale, k_scale, cache_seqlens), expected_shapes
    ):
        if tensor.shape != shape:
            raise ValueError(f"expected shape {shape}, got {tuple(tensor.shape)}")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] == 0
    ):
        raise ValueError("block_table must have shape [batch, positive page capacity]")
    if q_nope.dtype != torch.int8 or k_lora.dtype != torch.int8:
        raise TypeError("Q and KV content must be INT8")
    if (
        q_rope.dtype not in (torch.float16, torch.bfloat16)
        or k_rope.dtype != q_rope.dtype
    ):
        raise TypeError("RoPE tensors must have the same FP16 or BF16 dtype")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("scales must be FP32")
    if block_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        raise TypeError("block_table and cache_seqlens must be INT32")
    tensors = (
        q_nope,
        q_rope,
        k_lora,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
    )
    for tensor in tensors:
        if tensor.device != q_nope.device or (
            tensor.numel() != 0 and tensor.stride(-1) != 1
        ):
            raise ValueError(
                "all tensors must share a device and contiguous last dimension"
            )
    if not all(
        can_use_int32_index(tensor)
        for tensor in (q_nope, q_rope, q_scale, block_table, cache_seqlens)
    ):
        raise NotImplementedError("query and metadata offsets must fit signed INT32")
    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    if softmax_scale is not None and not isinstance(softmax_scale, (int, float)):
        raise TypeError("softmax_scale must be a Python number")
    scale = 576**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if num_splits is None:
        splits = min(
            16,
            triton.next_power_of_2(
                max(
                    1,
                    triton.cdiv(
                        4 * _get_num_sms(q_nope.device),
                        max(1, batch * triton.cdiv(queries * heads, 16)),
                    ),
                )
            ),
        )
        splits = min(splits, max(1, triton.next_power_of_2(block_table.shape[1])))
    else:
        splits = num_splits
    if (
        isinstance(splits, bool)
        or not isinstance(splits, int)
        or splits not in (1, 2, 4, 8, 16)
    ):
        raise ValueError("num_splits must be one of 1, 2, 4, 8, 16")
    output_bytes = batch * queries * heads * 512 * q_rope.element_size()
    partial_bytes = (splits if splits > 1 else 0) * batch * queries * heads * 512 * 4
    if max(output_bytes, partial_bytes) > 2**31 - 1:
        raise NotImplementedError(
            "output and partial workspaces must fit signed INT32 indexing"
        )
    handle = FlashMLAInt8PreparedHandle(
        tensors, splits, scale, causal, automatic_splits=num_splits is None
    )
    return handle, handle()


def flash_mla_with_kvcache_fwd_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_lora: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int = 512,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    num_splits: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, outputs = prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
        q_nope,
        q_rope,
        k_lora,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
        softmax_scale=softmax_scale,
        causal=causal,
        num_splits=num_splits,
    )
    return outputs
