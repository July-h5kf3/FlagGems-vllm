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
import triton.language as tl

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
):
    ht, page_idx, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    if page_idx * 64 >= length:
        return
    heads = ht * BH + tl.arange(0, BH)
    queries, qheads = heads // H, heads % H
    mh: tl.constexpr = H * SQ
    tokens = tl.arange(0, 64)
    dims = tl.arange(0, QK_BLOCK)
    rd = tl.arange(0, 64)
    page = tl.load(Table + batch * TABLE_STRIDE + page_idx).to(tl.int64)
    valid = page_idx * 64 + tokens < length
    qb = (
        Q
        + batch * Q_STRIDES[0]
        + queries[:, None] * Q_STRIDES[1]
        + qheads[:, None] * Q_STRIDES[2]
    )
    kb = KV + page * KV_STRIDES[0] + tokens[:, None] * KV_STRIDES[1]
    content = tl.zeros((BH, 64), tl.int32)
    for chunk in range(512 // QK_BLOCK):
        q = tl.load(qb + chunk * QK_BLOCK + dims[None, :], heads[:, None] < mh, 0)
        k = tl.load(kb + chunk * QK_BLOCK + dims[None, :], valid[:, None], 0)
        content = tl.dot(q, tl.trans(k), content, out_dtype=tl.int32)
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
    qs = tl.load(
        QS + batch * QS_STRIDES[0] + queries * QS_STRIDES[1] + qheads * QS_STRIDES[2],
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
        page_idx * 64 + tokens[None, :] < query_length[:, None], score, -float("inf")
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

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch == 0 or self.heads == 0:
            return self.output, self.lse
        query, rope, kv, kv_rope, qs, ks, table, lengths = self.tensors
        with torch_device_fn.device(query.device):
            if self.staged:
                merged_heads = self.queries * self.heads
                qk_heads = 32 if merged_heads == 32 else 128
                qk_block = 512 if merged_heads == 32 else 128
                pv_heads = 32 if merged_heads == 32 else 64
                _staged_int8_mla_qk[
                    (triton.cdiv(merged_heads, qk_heads), self.capacity, self.batch)
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
