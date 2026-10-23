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
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store

from flaggems_vllm import runtime
from flaggems_vllm.ops.flash_mla import _get_num_sms
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._hygon.ops.flash_mla_int8_native import (
    accumulate_mla_pv_limbs,
    interleave_mla_pv_bytes,
    native_int8_mla_pv_wide,
    quantize_mla_p16_word,
)
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
        inverse_ps = 1.0 / ps
        normalized = scaled * inverse_ps[:, None]
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
@g.jit
def _staged_int8_mla_normalize(
    Stats,
    Weight,
    RowFactor,
    LSE,
    Lengths,
    CAPACITY: gl.constexpr,
    LENGTH_STRIDE: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
    NP: gl.constexpr,
):
    BH: gl.constexpr = 8 if NP <= 1024 else 1
    # Adjacent heads make page-statistic reads and weight writes coalesced.
    L: gl.constexpr = BlockedLayout([1, 1], [BH, 64 // BH], [1, 4], [0, 1])
    ht, batch = gl.program_id(0), gl.program_id(1)
    mh: gl.constexpr = H * SQ
    heads = ht * BH + gl.arange(0, BH, layout=SliceLayout(1, L))
    idx = gl.arange(0, NP, layout=SliceLayout(0, L))
    pages = gl.cdiv(gl.load(Lengths + batch * LENGTH_STRIDE), 64)
    valid = (idx[None, :] < pages) & (heads[:, None] < mh)
    base = ((batch * CAPACITY + idx[None, :]) * mh + heads[:, None]) * 3
    m = gl.load(Stats + base, valid, -float("inf"))
    d = gl.load(Stats + base + 1, valid, 0.0)
    ps = gl.load(Stats + base + 2, valid, 0.0)
    maximum = gl.max(m, 1)
    safe = gl.where(maximum == -float("inf"), 0.0, maximum)
    beta = gl.exp2(m - safe[:, None])
    denom = gl.sum(d * beta, 1)
    inv = 1.0 / gl.where(denom > 0.0, denom, 1.0)
    global_scale = gl.maximum(gl.max(ps * beta, 1) * (65535.0 / 127.0), 1.0e-30)
    factor = beta * ps / global_scale[:, None]
    gl.store(
        Weight + (batch * CAPACITY + idx[None, :]) * mh + heads[:, None], factor, valid
    )
    gl.store(
        RowFactor + batch * mh + heads, global_scale * ((1.0 / 128.0) * inv), heads < mh
    )
    lse = gl.where(
        denom > 0.0, maximum * 0.6931471805599453 + gl.log(denom), float("inf")
    )
    gl.store(LSE + (batch * H + heads % H) * SQ + heads // H, lse, heads < mh)


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


@triton.jit
def order_mla_batches(
    Lengths, Order, B: tl.constexpr, STRIDE: tl.constexpr, NB: tl.constexpr
):
    i = tl.arange(0, NB)
    length = tl.maximum(tl.load(Lengths + i * STRIDE, i < B, 0), 0).to(tl.int64)
    key = tl.where(i < B, length * NB + (NB - 1 - i), -1)
    ranked = tl.sort(key, descending=True)
    original = (NB - 1 - (ranked & (NB - 1))).to(tl.int32)
    tl.store(Order + i, original, i < B)


@g.jit
def expand_mla_pv_word(value, N: gl.constexpr):
    R: gl.constexpr = BlockedLayout([1, 1, 1], [16, 1, 4], [4, 1, 1], [0, 2, 1])
    C: gl.constexpr = BlockedLayout([1, 1], [16, 4], [4, 1], [0, 1])
    value = gl.convert_layout(value, SliceLayout(1, R))
    result = value[:, None, :] + gl.full((64, N // 4, 4), 0, value.dtype, R)
    return gl.convert_layout(result.reshape((64, N)), C)


@g.jit
def native_int8_mla_pv(
    KV,
    Table,
    Lengths,
    Packed,
    Weight,
    RowFactor,
    Output,
    KV_STRIDES: gl.constexpr,
    TABLE_STRIDE: gl.constexpr,
    CAPACITY: gl.constexpr,
    LENGTH_STRIDE: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
    BATCH: gl.constexpr = 0,
    PARTS: gl.constexpr = 1,
    Order=None,
):
    PAGE_GROUP: gl.constexpr = 2
    SHARED_BYTES: gl.constexpr = 16384 * PAGE_GROUP
    C: gl.constexpr = BlockedLayout([1, 1], [16, 4], [4, 1], [0, 1])
    W: gl.constexpr = BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    P3: gl.constexpr = BlockedLayout([1, 1, 2], [16, 4, 1], [4, 1, 1], [0, 2, 1])
    P2: gl.constexpr = SliceLayout(2, P3)
    dt, ht, batch = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    if Order is not None:
        batch = gl.load(Order + batch)
    if PARTS > 1:
        part = dt // 2
        dt = dt % 2
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
    if PARTS > 1:
        page_groups = gl.cdiv(pages, PAGE_GROUP)
        groups_per_part = gl.cdiv(page_groups, PARTS)
        first_group = part * groups_per_part
        last_group = gl.minimum(page_groups, first_group + groups_per_part)
    else:
        first_group = 0
        last_group = gl.cdiv(pages, PAGE_GROUP)
    for group in tl.range(first_group, last_group, loop_unroll_factor=1):
        highs = ()
        lows = ()
        gl.thread_barrier()
        for page_offset in gl.static_range(PAGE_GROUP):
            page_idx = group * PAGE_GROUP + page_offset
            page = gl.load(
                Table + batch * TABLE_STRIDE + page_idx, page_idx < pages, 0
            ).to(gl.int64)
            scale = buffer_load(
                Weight + (batch * CAPACITY + page_idx) * mh,
                ph,
                (ph < mh) & (page_idx < pages),
                0.0,
            )
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
                high0, low0 = quantize_mla_p16_word(p0, scale[:, None])
                high1, low1 = quantize_mla_p16_word(p1, scale[:, None])
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
    if PARTS == 1:
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
    else:
        # Empty partitions explicitly store zero limbs for graph replay.
        HTILES: gl.constexpr = (H * SQ + 63) // 64
        tile = (batch * HTILES + ht) * 2 + dt
        OUT: gl.constexpr = BlockedLayout([1, 2], [16, 4], [4, 1], [0, 1])
        oh = gl.arange(0, 64, layout=SliceLayout(1, OUT))
        read_col = gl.arange(0, 16, layout=SliceLayout(0, C))
        item = gl.arange(0, 32, layout=SliceLayout(0, OUT))
        for block in gl.static_range(8):
            indices = (block * 16 + read_col)[None, :] + gl.full(
                (64, 16), 0, gl.int32, C
            )
            hi = gl.gather(he, indices, 1)
            li = gl.gather(le, indices, 1)
            even = gl.convert_layout(gl.join(hi, li).reshape((64, 32)), OUT)
            offset = (
                (((tile * PARTS + part) * 2) * 128 + block * 16 + item[None, :] // 2)
                * 64
                + oh[:, None]
            ) * 2 + item[None, :] % 2
            buffer_store(even, Output, offset.to(gl.int32), ht * 64 + oh[:, None] < mh)
            ho_chunk = gl.gather(ho, indices, 1)
            lo_chunk = gl.gather(lo, indices, 1)
            odd = gl.convert_layout(gl.join(ho_chunk, lo_chunk).reshape((64, 32)), OUT)
            buffer_store(
                odd,
                Output,
                (offset + 128 * 64 * 2).to(gl.int32),
                ht * 64 + oh[:, None] < mh,
            )


@g.jit
def merge_mla_pv_integer(
    Partial,
    RowFactor,
    Output,
    BATCH: gl.constexpr,
    MH: gl.constexpr,
    PARTS: gl.constexpr,
    COLS: gl.constexpr,
):
    # Integer limb sums retain the full-row INT32 bounds before FP32 conversion.
    HTILES: gl.constexpr = (MH + 63) // 64
    L: gl.constexpr = BlockedLayout([1, 1], [16, 4], [4, 1], [0, 1])
    block, tile = gl.program_id(0), gl.program_id(1)
    batch = tile // (HTILES * 2)
    ht = (tile // 2) % HTILES
    dt = tile % 2
    local_head = gl.arange(0, 64, layout=SliceLayout(1, L))
    heads = ht * 64 + local_head
    col = block * COLS + gl.arange(0, COLS, layout=SliceLayout(0, L))
    factor = buffer_load(RowFactor + batch * MH, heads, heads < MH, 0.0)
    high = gl.full((64, COLS), 0, gl.int32, L)
    low = high
    for part in gl.static_range(PARTS):
        offset = (((tile * PARTS + part) * 2) * 128 + col[None, :]) * 64 + local_head[
            :, None
        ]
        word = buffer_load(
            Partial.to(tl.pointer_type(gl.uint64)),
            offset.to(gl.int32),
            heads[:, None] < MH,
            0,
        )
        high += (word & 0xFFFFFFFF).to(gl.uint32).to(gl.int32, bitcast=True)
        low += (word >> 32).to(gl.uint32).to(gl.int32, bitcast=True)
    even = (high.to(gl.float32) * 128.0 + low.to(gl.float32)) * factor[:, None]
    high = gl.full((64, COLS), 0, gl.int32, L)
    low = high
    for part in gl.static_range(PARTS):
        offset = (
            ((tile * PARTS + part) * 2 + 1) * 128 + col[None, :]
        ) * 64 + local_head[:, None]
        word = buffer_load(
            Partial.to(tl.pointer_type(gl.uint64)),
            offset.to(gl.int32),
            heads[:, None] < MH,
            0,
        )
        high += (word & 0xFFFFFFFF).to(gl.uint32).to(gl.int32, bitcast=True)
        low += (word >> 32).to(gl.uint32).to(gl.int32, bitcast=True)
    odd = (high.to(gl.float32) * 128.0 + low.to(gl.float32)) * factor[:, None]
    OUT: gl.constexpr = BlockedLayout([1, 2], [16, 4], [4, 1], [0, 1])
    value = gl.convert_layout(gl.join(even, odd).reshape((64, COLS * 2)), OUT)
    oh = ht * 64 + gl.arange(0, 64, layout=SliceLayout(1, OUT))
    of = (
        dt * 256 + block * COLS * 2 + gl.arange(0, COLS * 2, layout=SliceLayout(0, OUT))
    )
    gl.store(
        Output + (batch * MH + oh[:, None]) * 512 + of[None, :], value, oh[:, None] < MH
    )


@g.jit
def accumulate_small_head_pv_limbs(High, Low, Base, HE, HO, LE, LO):
    # Fixed registers retain both integer limbs without padding the head tile.
    return gl.inline_asm_elementwise(
        """s_waitcnt lgkmcnt(0)
v_xor_b32 v80, 64, $96
ds_read_m32x16_b16_alt v[0:3], $96 offset:0
ds_read_m32x16_b16_alt v[4:7], v80 offset:0
ds_read_m32x16_b16_alt v[8:11], $96 offset:128
ds_read_m32x16_b16_alt v[12:15], v80 offset:128
s_waitcnt lgkmcnt(3)
s_nop 0
v_mmac_i32_16x16x32_i8 v[16:19], $64, v[0:1], v[16:19]
v_mmac_i32_16x16x32_i8 v[32:35], $64, v[2:3], v[32:35]
v_mmac_i32_16x16x32_i8 v[48:51], $80, v[0:1], v[48:51]
v_mmac_i32_16x16x32_i8 v[64:67], $80, v[2:3], v[64:67]
s_nop 0
s_waitcnt lgkmcnt(2)
s_nop 0
v_mmac_i32_16x16x32_i8 v[20:23], $64, v[4:5], v[20:23]
v_mmac_i32_16x16x32_i8 v[36:39], $64, v[6:7], v[36:39]
v_mmac_i32_16x16x32_i8 v[52:55], $80, v[4:5], v[52:55]
v_mmac_i32_16x16x32_i8 v[68:71], $80, v[6:7], v[68:71]
s_nop 0
s_waitcnt lgkmcnt(1)
s_nop 0
v_mmac_i32_16x16x32_i8 v[24:27], $64, v[8:9], v[24:27]
v_mmac_i32_16x16x32_i8 v[40:43], $64, v[10:11], v[40:43]
v_mmac_i32_16x16x32_i8 v[56:59], $80, v[8:9], v[56:59]
v_mmac_i32_16x16x32_i8 v[72:75], $80, v[10:11], v[72:75]
s_nop 0
s_waitcnt lgkmcnt(0)
s_nop 0
v_mmac_i32_16x16x32_i8 v[28:31], $64, v[12:13], v[28:31]
v_mmac_i32_16x16x32_i8 v[44:47], $64, v[14:15], v[44:47]
v_mmac_i32_16x16x32_i8 v[60:63], $80, v[12:13], v[60:63]
v_mmac_i32_16x16x32_i8 v[76:79], $80, v[14:15], v[76:79]
s_nop 0""",
        constraints=(
            "=&{v16},=&{v17},=&{v18},=&{v19},=&{v20},=&{v21},=&{v22},=&{v23},"
            "=&{v24},=&{v25},=&{v26},=&{v27},=&{v28},=&{v29},=&{v30},=&{v31},"
            "=&{v32},=&{v33},=&{v34},=&{v35},=&{v36},=&{v37},=&{v38},=&{v39},"
            "=&{v40},=&{v41},=&{v42},=&{v43},=&{v44},=&{v45},=&{v46},=&{v47},"
            "=&{v48},=&{v49},=&{v50},=&{v51},=&{v52},=&{v53},=&{v54},=&{v55},"
            "=&{v56},=&{v57},=&{v58},=&{v59},=&{v60},=&{v61},=&{v62},=&{v63},"
            "=&{v64},=&{v65},=&{v66},=&{v67},=&{v68},=&{v69},=&{v70},=&{v71},"
            "=&{v72},=&{v73},=&{v74},=&{v75},=&{v76},=&{v77},=&{v78},=&{v79},"
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
            "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},"
            "~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},"
            "~{v80},~{memory}"
        ),
        args=[High, Low, Base, HE, HO, LE, LO],
        dtype=(gl.int32, gl.int32, gl.int32, gl.int32),
        is_pure=False,
        pack=16,
    )


@g.jit
def expand_small_head_pv_base(value):
    R: gl.constexpr = BlockedLayout(
        [1, 1, 1, 1], [16, 1, 1, 4], [2, 2, 1, 1], [0, 3, 2, 1]
    )
    C: gl.constexpr = BlockedLayout([1, 1, 1], [16, 1, 4], [2, 2, 1], [0, 2, 1])
    value = gl.convert_layout(value, SliceLayout(2, R))
    expanded = value[:, :, None, :] + gl.full((32, 2, 16, 4), 0, value.dtype, R)
    return gl.convert_layout(expanded.reshape((32, 2, 64)), C)


@g.jit
def expand_small_head_pv_probability(value):
    C: gl.constexpr = BlockedLayout([1, 1, 1], [16, 1, 4], [2, 2, 1], [0, 2, 1])
    value = gl.convert_layout(value, SliceLayout(1, C))
    paired = value[:, None, :] + gl.full((32, 2, 4), 0, value.dtype, C)
    return expand_small_head_pv_base(paired)


@g.jit
def small_head_int8_mla_pv(
    KV,
    Table,
    Lengths,
    Packed,
    Weight,
    RowFactor,
    Output,
    KV_STRIDES: gl.constexpr,
    TABLE_STRIDE: gl.constexpr,
    CAPACITY: gl.constexpr,
    LENGTH_STRIDE: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
    BATCH: gl.constexpr = 0,
    PARTS: gl.constexpr = 1,
    Order=None,
):
    C: gl.constexpr = BlockedLayout([1, 1, 1], [16, 1, 4], [2, 2, 1], [0, 2, 1])
    W: gl.constexpr = BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    P3: gl.constexpr = BlockedLayout([1, 1, 2], [16, 4, 1], [2, 2, 1], [0, 2, 1])
    P2: gl.constexpr = SliceLayout(2, P3)
    dt, ht, batch = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    if Order is not None:
        batch = gl.load(Order + batch)
    part = dt // 2
    dt = dt % 2
    mh: gl.constexpr = H * SQ
    local = gl.arange(0, 32, layout=SliceLayout(1, SliceLayout(2, C)))
    heads = ht * 32 + local
    groups = gl.arange(0, 4, layout=SliceLayout(0, SliceLayout(1, C)))
    halves = gl.arange(0, 2, layout=SliceLayout(0, SliceLayout(2, C)))
    ph = ht * 32 + gl.arange(0, 32, layout=SliceLayout(1, P2))
    pg = gl.arange(0, 4, layout=SliceLayout(0, P2))
    pj = gl.arange(0, 2, layout=SliceLayout(0, SliceLayout(1, P3)))
    length = gl.load(Lengths + batch * LENGTH_STRIDE)
    pages = gl.cdiv(length, 64)
    shared = gl.allocate_shared_memory(
        gl.uint8, (32768,), SwizzledSharedLayout(16, 1, 1, [0])
    )
    paired = shared._reinterpret(
        gl.uint32, (4, 32, 64), SwizzledSharedLayout(4, 1, 8, [2, 1, 0])
    )
    tail = shared.slice(32768 - 1024, 1024, 0)._reinterpret(
        gl.uint32, (32, 2, 4), SwizzledSharedLayout(1, 1, 1, [0, 2, 1])
    )
    tail.store(gl.full((32, 2, 4), 0, gl.uint32, C))
    gl.thread_barrier()
    zero = gl.inline_asm_elementwise(
        "v_and_b32 $0, $1, 0",
        constraints="=v,v,~{memory}",
        args=[tail.load(C)],
        dtype=gl.uint32,
        is_pure=False,
        pack=1,
    )
    source = (local[:, None, None] & 15) // 4 + groups[None, None, :] * 4
    raw = (
        source * 256
        + (((local[:, None, None] & 3) * 8) ^ ((source & 7) * 8)) * 2
        + halves[None, :, None] * 8192
        + zero
    )
    base = expand_small_head_pv_base(raw)
    pairs = gl.arange(0, 32, layout=SliceLayout(1, W))
    features = dt * 64 + gl.arange(0, 64, layout=SliceLayout(0, W))
    he = gl.full((32, 2, 64), 0, gl.int32, C)
    ho = he
    le = he
    lo = he
    groups_total = gl.cdiv(pages, 2)
    groups_per_part = gl.cdiv(groups_total, PARTS)
    first_group = part * groups_per_part
    last_group = gl.minimum(groups_total, first_group + groups_per_part)
    for group in tl.range(first_group, last_group, loop_unroll_factor=1):
        gl.thread_barrier()
        highs = ()
        lows = ()
        for page_offset in gl.static_range(2):
            page_idx = group * 2 + page_offset
            page = gl.load(
                Table + batch * TABLE_STRIDE + page_idx, page_idx < pages, 0
            ).to(gl.int64)
            scale = buffer_load(
                Weight + (batch * CAPACITY + page_idx) * mh,
                ph,
                (ph < mh) & (page_idx < pages),
                0.0,
            )
            for half in gl.static_range(2):
                offset = (
                    ph[:, None, None] * 16
                    + half * 8
                    + pg[None, :, None] * 2
                    + pj[None, None, :]
                )
                words = buffer_load(
                    Packed.to(tl.pointer_type(gl.uint64))
                    + (batch * CAPACITY + page_idx) * mh * 16,
                    offset,
                    (ph[:, None, None] < mh) & (page_idx < pages),
                    0,
                )
                p0, p1 = gl.split(words)
                h0, l0 = quantize_mla_p16_word(p0, scale[:, None])
                h1, l1 = quantize_mla_p16_word(p1, scale[:, None])
                highs += (
                    expand_small_head_pv_probability(
                        h0.to(gl.uint64) | (h1.to(gl.uint64) << 32)
                    ),
                )
                lows += (
                    expand_small_head_pv_probability(
                        l0.to(gl.uint64) | (l1.to(gl.uint64) << 32)
                    ),
                )
            token = 2 * pairs
            first = buffer_load(
                KV.to(tl.pointer_type(gl.uint32)) + page * (KV_STRIDES[0] // 4),
                token[:, None] * (KV_STRIDES[1] // 4) + features[None, :],
                page_idx * 64 + token[:, None] < length,
                0,
            )
            second = buffer_load(
                KV.to(tl.pointer_type(gl.uint32)) + page * (KV_STRIDES[0] // 4),
                (token[:, None] + 1) * (KV_STRIDES[1] // 4) + features[None, :],
                page_idx * 64 + token[:, None] + 1 < length,
                0,
            )
            low_pair = interleave_mla_pv_bytes(first, second, 0x05010400)
            high_pair = interleave_mla_pv_bytes(first, second, 0x07030602)
            packed = (
                gl.join(low_pair, high_pair).reshape((32, 2, 64)).permute((1, 0, 2))
            )
            paired.slice(page_offset * 2, 2, 0).store(packed)
        gl.thread_barrier()
        for item in gl.static_range(4):
            he, ho, le, lo = accumulate_small_head_pv_limbs(
                highs[item],
                lows[item],
                base + (item // 2) * 16384 + (item % 2) * 4096,
                he,
                ho,
                le,
                lo,
            )
    if PARTS == 1:
        factor = buffer_load(RowFactor + batch * mh, heads, heads < mh, 0.0)
        even = (he.to(gl.float32) * 128.0 + le.to(gl.float32)) * factor[:, None, None]
        odd = (ho.to(gl.float32) * 128.0 + lo.to(gl.float32)) * factor[:, None, None]
        OUT: gl.constexpr = BlockedLayout([1, 1, 2], [16, 1, 4], [2, 2, 1], [0, 2, 1])
        result = gl.convert_layout(gl.join(even, odd).reshape((32, 2, 128)), OUT)
        oh = ht * 32 + gl.arange(0, 32, layout=SliceLayout(1, SliceLayout(2, OUT)))
        of = gl.arange(0, 128, layout=SliceLayout(0, SliceLayout(1, OUT)))
        half = gl.arange(0, 2, layout=SliceLayout(0, SliceLayout(2, OUT)))
        gl.store(
            Output
            + (batch * mh + oh[:, None, None]) * 512
            + dt * 256
            + half[None, :, None] * 128
            + of[None, None, :],
            result,
            oh[:, None, None] < mh,
        )
    else:
        HT: gl.constexpr = (H * SQ + 31) // 32
        tile = (batch * HT + ht) * 2 + dt
        OUT: gl.constexpr = BlockedLayout([1, 1, 2], [16, 1, 4], [2, 2, 1], [0, 2, 1])
        oh = gl.arange(0, 32, layout=SliceLayout(1, SliceLayout(2, OUT)))
        hf = gl.arange(0, 2, layout=SliceLayout(0, SliceLayout(2, OUT)))
        item = gl.arange(0, 32, layout=SliceLayout(0, SliceLayout(1, OUT)))
        readcol = gl.arange(0, 16, layout=SliceLayout(0, SliceLayout(1, C)))
        for block in gl.static_range(4):
            indices = (block * 16 + readcol)[None, None, :] + gl.full(
                (32, 2, 16), 0, gl.int32, C
            )
            high = gl.gather(he, indices, 2)
            low = gl.gather(le, indices, 2)
            even = gl.convert_layout(gl.join(high, low).reshape((32, 2, 32)), OUT)
            offset = (
                (
                    ((tile * PARTS + part) * 2 + hf[None, :, None]) * 2 * 64
                    + block * 16
                    + item[None, None, :] // 2
                )
                * 32
                + oh[:, None, None]
            ) * 2 + item[None, None, :] % 2
            buffer_store(
                even, Output, offset.to(gl.int32), ht * 32 + oh[:, None, None] < mh
            )
            high = gl.gather(ho, indices, 2)
            low = gl.gather(lo, indices, 2)
            odd = gl.convert_layout(gl.join(high, low).reshape((32, 2, 32)), OUT)
            buffer_store(
                odd,
                Output,
                (offset + 64 * 32 * 2).to(gl.int32),
                ht * 32 + oh[:, None, None] < mh,
            )


@g.jit
def merge_small_head_pv_integer(
    Partial,
    RowFactor,
    Output,
    BATCH: gl.constexpr,
    MH: gl.constexpr,
    PARTS: gl.constexpr,
    COLS: gl.constexpr,
):
    HT: gl.constexpr = (MH + 31) // 32
    L: gl.constexpr = BlockedLayout([1, 1, 1], [16, 4, 1], [2, 2, 1], [0, 1, 2])
    blockhalf, tile = gl.program_id(0), gl.program_id(1)
    hf, block = blockhalf // 4, blockhalf % 4
    batch, ht, dt = tile // (HT * 2), (tile // 2) % HT, tile % 2
    local = gl.arange(0, 32, layout=SliceLayout(1, SliceLayout(2, L)))
    heads = ht * 32 + local
    col = block * COLS + gl.arange(0, COLS, layout=SliceLayout(0, SliceLayout(2, L)))
    eo = gl.arange(0, 2, layout=SliceLayout(0, SliceLayout(1, L)))
    high = gl.full((32, COLS, 2), 0, gl.int32, L)
    low = high
    for part in gl.static_range(PARTS):
        offset = (
            (((tile * PARTS + part) * 2 + hf) * 2 + eo[None, None, :]) * 64
            + col[None, :, None]
        ) * 32 + local[:, None, None]
        word = buffer_load(
            Partial.to(tl.pointer_type(gl.uint64)),
            offset.to(gl.int32),
            heads[:, None, None] < MH,
            0,
        )
        high += word.to(gl.uint32).to(gl.int32, bitcast=True)
        low += (word >> 32).to(gl.uint32).to(gl.int32, bitcast=True)
    factor = buffer_load(RowFactor + batch * MH, heads, heads < MH, 0.0)
    value = (high.to(gl.float32) * 128.0 + low.to(gl.float32)) * factor[:, None, None]
    OUT: gl.constexpr = BlockedLayout([1, 2], [16, 4], [2, 2], [0, 1])
    value = gl.convert_layout(value.reshape((32, COLS * 2)), OUT)
    oh = ht * 32 + gl.arange(0, 32, layout=SliceLayout(1, OUT))
    of = (
        dt * 256
        + hf * 128
        + block * COLS * 2
        + gl.arange(0, COLS * 2, layout=SliceLayout(0, OUT))
    )
    gl.store(
        Output + (batch * MH + oh[:, None]) * 512 + of[None, :], value, oh[:, None] < MH
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

        self.native_pv_kernel = (
            small_head_int8_mla_pv if merged_heads == 32 else native_int8_mla_pv
        )
        self.native_pv_merge_kernel = (
            merge_small_head_pv_integer if merged_heads == 32 else merge_mla_pv_integer
        )
        self.has_native_pv = False
        kv, table, lengths = tensors[2], tensors[6], tensors[7]
        if (
            self.staged
            and (merged_heads == 32 or merged_heads >= 64)
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
                binary = self.native_pv_kernel.warmup(
                    kv,
                    table,
                    lengths,
                    self.packed,
                    self.weight,
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

        # Private PV partitions preserve the public num_splits contract.
        self.pv_splits = 1
        partial_workspace_bytes = 0
        pv_heads = 32 if merged_heads == 32 else 64
        minimum_pages = 64 if merged_heads == 32 else 256
        if self.has_native_pv and self.capacity >= minimum_pages:
            candidate_splits = 3
            cta_count = self.batch * triton.cdiv(merged_heads, pv_heads) * 2
            # Require enough partitioning benefit to cover the merge overhead.
            group_size = (6 if merged_heads == 32 else 3) * _get_num_sms(query.device)
            groups = triton.cdiv(cta_count, group_size)
            split_groups = triton.cdiv(cta_count * candidate_splits, group_size)
            partial_bytes = (
                8
                * candidate_splits
                * self.batch
                * (pv_heads * triton.cdiv(merged_heads, pv_heads))
                * 512
            )
            if (
                groups * candidate_splits * 100 >= split_groups * 135
                and staged_bytes + partial_bytes <= 4 * 1024**3
            ):
                tiles = 2 * self.batch * triton.cdiv(merged_heads, pv_heads)
                partial_shape = (
                    (tiles, candidate_splits, 2, 2, 64, 32, 2)
                    if merged_heads == 32
                    else (tiles, candidate_splits, 2, 128, 64, 2)
                )
                self.pv_partials = torch.empty(
                    partial_shape,
                    dtype=torch.int32,
                    device=query.device,
                )
                partial_workspace_bytes = partial_bytes
                with torch_device_fn.device(query.device):
                    binary = self.native_pv_kernel.warmup(
                        kv,
                        table,
                        lengths,
                        self.packed,
                        self.weight,
                        self.rowfactor,
                        self.pv_partials,
                        kv.stride(),
                        table.stride(0),
                        self.capacity,
                        lengths.stride(0),
                        self.heads,
                        self.queries,
                        self.batch,
                        candidate_splits,
                        grid=(
                            2 * candidate_splits,
                            triton.cdiv(merged_heads, 64),
                            self.batch,
                        ),
                        num_warps=4,
                        num_stages=1,
                    )
                if binary.metadata.shared == 32768 and binary.metadata.warp_size == 64:
                    self.pv_splits = candidate_splits

        self.pv_order = None
        if (
            self.has_native_pv
            and 32 <= self.batch <= 1024
            and staged_bytes + partial_workspace_bytes + self.batch * 4 <= 4 * 1024**3
        ):
            self.pv_order = torch.empty(
                (self.batch,), dtype=torch.int32, device=query.device
            )
            native_output = self.pv_partials if self.pv_splits > 1 else self.output
            with torch_device_fn.device(query.device):
                binary = self.native_pv_kernel.warmup(
                    kv,
                    table,
                    lengths,
                    self.packed,
                    self.weight,
                    self.rowfactor,
                    native_output,
                    kv.stride(),
                    table.stride(0),
                    self.capacity,
                    lengths.stride(0),
                    self.heads,
                    self.queries,
                    self.batch,
                    self.pv_splits,
                    Order=self.pv_order,
                    grid=(
                        2 * self.pv_splits,
                        triton.cdiv(merged_heads, pv_heads),
                        self.batch,
                    ),
                    num_warps=4,
                    num_stages=1,
                )
            if binary.metadata.shared != 32768 or binary.metadata.warp_size != 64:
                self.pv_order = None

        self.pv_head_tile = pv_heads
        self.pv_warps = 4
        if (
            self.has_native_pv
            and self.pv_splits == 1
            and merged_heads >= 128
            and merged_heads % 128 == 0
            and 2 * self.batch * triton.cdiv(merged_heads, 128)
            >= 2 * _get_num_sms(query.device)
        ):
            with torch_device_fn.device(query.device):
                binary = native_int8_mla_pv_wide.warmup(
                    kv,
                    table,
                    lengths,
                    self.packed,
                    self.weight,
                    self.rowfactor,
                    self.output,
                    kv.stride(),
                    table.stride(0),
                    self.capacity,
                    lengths.stride(0),
                    self.heads,
                    self.queries,
                    self.batch,
                    1,
                    Order=self.pv_order,
                    NORMALIZED=True,
                    grid=(2, triton.cdiv(merged_heads, 128), self.batch),
                    num_warps=8,
                    num_stages=1,
                )
                binary._init_handles()
            if (
                binary.metadata.shared == 32768
                and binary.metadata.warp_size == 64
                and binary.n_spills == 0
                and binary.n_regs <= 192
            ):
                self.native_pv_kernel = native_int8_mla_pv_wide
                self.pv_head_tile = 128
                self.pv_warps = 8

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch == 0 or self.heads == 0:
            return self.output, self.lse
        query, rope, kv, kv_rope, qs, ks, table, lengths = self.tensors
        with torch_device_fn.device(query.device):
            if self.staged:
                merged_heads = self.queries * self.heads
                page_group = 4 if merged_heads > 64 else 1
                qk_heads = min(128, triton.next_power_of_2(merged_heads))
                qk_block = 128
                pv_heads = self.pv_head_tile
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
                normalize_heads = 8 if self.capacity <= 1024 else 1
                _staged_int8_mla_normalize[
                    (triton.cdiv(merged_heads, normalize_heads), self.batch)
                ](
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
                if self.has_native_pv:
                    if self.pv_order is not None:
                        order_mla_batches[(1,)](
                            lengths,
                            self.pv_order,
                            self.batch,
                            lengths.stride(0),
                            triton.next_power_of_2(self.batch),
                            num_warps=4,
                            num_stages=1,
                        )
                    if self.pv_splits > 1:
                        self.native_pv_kernel[
                            (
                                2 * self.pv_splits,
                                triton.cdiv(merged_heads, 64),
                                self.batch,
                            )
                        ](
                            kv,
                            table,
                            lengths,
                            self.packed,
                            self.weight,
                            self.rowfactor,
                            self.pv_partials,
                            kv.stride(),
                            table.stride(0),
                            self.capacity,
                            lengths.stride(0),
                            self.heads,
                            self.queries,
                            self.batch,
                            self.pv_splits,
                            Order=self.pv_order,
                            num_warps=4,
                            num_stages=1,
                        )
                        self.native_pv_merge_kernel[
                            (8, 2 * self.batch * triton.cdiv(merged_heads, 64))
                        ](
                            self.pv_partials,
                            self.rowfactor,
                            self.output,
                            self.batch,
                            merged_heads,
                            self.pv_splits,
                            16,
                            num_warps=4,
                        )
                    else:
                        self.native_pv_kernel[
                            (
                                2,
                                triton.cdiv(merged_heads, self.pv_head_tile),
                                self.batch,
                            )
                        ](
                            kv,
                            table,
                            lengths,
                            self.packed,
                            self.weight,
                            self.rowfactor,
                            self.output,
                            kv.stride(),
                            table.stride(0),
                            self.capacity,
                            lengths.stride(0),
                            self.heads,
                            self.queries,
                            self.batch,
                            self.pv_splits,
                            Order=self.pv_order,
                            num_warps=self.pv_warps,
                            num_stages=1,
                            **({"NORMALIZED": True} if self.pv_warps == 8 else {}),
                        )
                else:
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
        if queries * heads == 16:
            # Limit page work per partition for long requests with small heads.
            length_splits = min(
                16, triton.next_power_of_2(triton.cdiv(block_table.shape[1], 16))
            )
            # Preserve the supported range of the original workspace check.
            if length_splits * batch * queries * heads * 512 * 4 < 2**31:
                splits = max(splits, length_splits)
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
