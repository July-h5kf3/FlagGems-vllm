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

from flaggems_vllm.ops.flash_mla import _get_num_sms
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils.shape_utils import can_use_int32_index

# Python frontends may be newer than the loaded native plugin.
try:
    from triton._C.libtriton.gluon_ir import supports_maca_async_copy
except ImportError:
    supports_maca_async_copy = False

try:
    from triton._C.libtriton.gluon_ir import supports_maca_int8_col_major
except ImportError:
    supports_maca_int8_col_major = False


@triton.jit
def accumulate_int8_pv(
    KVBase, Valid, PHigh, PLow, Rescale, PScale, Accumulator, OFFSET: tl.constexpr
):
    # Reload V in small tiles to bound live registers and shared-memory usage.
    dims = OFFSET + tl.arange(0, 128)
    value = tl.load(KVBase + dims[None, :], Valid[:, None], 0)
    high = tl.dot(tl.trans(value), tl.trans(PHigh), out_dtype=tl.int32).to(tl.float32)
    low = tl.dot(tl.trans(value), tl.trans(PLow), out_dtype=tl.int32).to(tl.float32)
    return (
        Accumulator * Rescale[None, :] + (high + low * (1.0 / 127.0)) * PScale[None, :]
    )


@libentry()
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
):
    head_tile, batch, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    heads = head_tile * BH + tl.arange(0, BH)
    queries, qheads = heads // H, heads % H
    merged_heads = SQ * H
    content_dims = tl.arange(0, 64)
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
        # A full query tile kept across PV spills on C550.
        for chunk in tl.range(8, loop_unroll_factor=1, disable_licm=True):
            dims = chunk * 64 + content_dims
            q = tl.load(
                q_base + dims[None, :], heads[:, None] < merged_heads, 0, volatile=True
            )
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
        high = tl.floor(scaled + 0.5).to(tl.int8)
        low = tl.floor((scaled - high.to(tl.float32)) * 127.0 + 0.5).to(tl.int8)
        acc0 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc0, 0
        )
        acc1 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc1, 128
        )
        acc2 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc2, 256
        )
        acc3 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc3, 384
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
    NORMALIZED: tl.constexpr = False,
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
    if NORMALIZED:
        weights = sums * weights / tl.where(denominator > 0, denominator, 1.0)
        output = tl.sum(values * weights[:, None], 0)
    else:
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


class FlashMLAInt8PreparedHandle:
    """Reuse workspace for decode; calls must be serialized on one stream."""

    def __init__(
        self, tensors: tuple[torch.Tensor, ...], splits: int, scale: float, causal: bool
    ):
        self.tensors = tensors
        query, rope, _, _, _, _, _, _ = tensors
        self.batch, self.queries, self.heads, _ = query.shape
        self.causal = causal
        self.head_tile = 16
        self.splits = splits
        self.scale = scale
        self.num_warps = 4
        self.kernel = dense_int8_mla_kernel
        self.kernel_arguments = {}
        self.batch_order = None
        self.batch_order_kernel = None
        self.grid = (
            triton.cdiv(self.queries * self.heads, self.head_tile),
            self.batch,
            self.splits,
        )
        self.normalized_partials = False
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

    def _prepare_batch_order(self):
        if self.batch_order is not None:
            lengths = self.tensors[7]
            self.batch_order_kernel[(1,)](
                lengths,
                self.batch_order,
                lengths.stride(0),
                self.batch,
                triton.next_power_of_2(self.batch),
                num_warps=4,
                num_stages=1,
            )

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch == 0 or self.heads == 0:
            return self.output, self.lse
        query, rope, kv, kv_rope, qs, ks, table, lengths = self.tensors
        with torch_device_fn.device(query.device):
            self._prepare_batch_order()
            self.kernel[self.grid](
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
                BH=self.head_tile,
                num_warps=self.num_warps,
                num_stages=2,
                scenario="mla",
                **self.kernel_arguments,
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
                    NORMALIZED=self.normalized_partials,
                    num_warps=4,
                )
        return self.output, self.lse


class FlashMLAInt8SmallHeadHandle(FlashMLAInt8PreparedHandle):
    def __init__(self, tensors: tuple[torch.Tensor, ...], scale: float, causal: bool):
        capacity = tensors[6].shape[1] * 64
        splits = min(
            4 if capacity <= 4096 else 8, triton.next_power_of_2(tensors[6].shape[1])
        )
        super().__init__(tensors, splits, scale, causal)
        from flaggems_vllm.runtime.backend._metax.ops.flash_mla_int8_gluon import (
            dense_int8_batch_order,
            dense_int8_mla_smallheads,
        )

        self.head_tile = min(
            32, max(16, triton.next_power_of_2(self.queries * self.heads))
        )
        self.kernel = dense_int8_mla_smallheads
        if self.head_tile == 32 and 1 < self.batch <= 256:
            self.batch_order = torch.empty(
                (self.batch,), dtype=torch.int32, device=tensors[0].device
            )
            self.batch_order_kernel = dense_int8_batch_order
            self.kernel_arguments = {"BatchOrder": self.batch_order}
        self.grid = (1, 1, self.batch * self.splits)
        self.normalized_partials = True


class FlashMLAInt8GluonHandle(FlashMLAInt8PreparedHandle):
    def __init__(self, tensors: tuple[torch.Tensor, ...], scale: float, causal: bool):
        super().__init__(tensors, 1, scale, causal)
        from flaggems_vllm.runtime.backend._metax.ops import flash_mla_int8_gluon

        self.kernels = flash_mla_int8_gluon
        query, _, _, _, _, _, table, _ = tensors
        merged_heads = self.queries * self.heads
        rows = self.batch * merged_heads
        self.capacity = table.shape[1] * 64
        self.head_tile = min(64, max(16, triton.next_power_of_2(merged_heads)))
        self.prequantized = self.capacity <= 16384 and merged_heads < 64
        self.grouped = merged_heads <= 32 and (
            not self.prequantized or merged_heads > 16
        )
        self.quantized_cache = not self.prequantized and not self.grouped
        self.qk_native_statistics = (
            self.quantized_cache and supports_maca_int8_col_major
        )
        if self.quantized_cache and 1 < self.batch <= 256:
            self.batch_order = torch.empty(
                (self.batch,), dtype=torch.int32, device=query.device
            )
            self.batch_order_kernel = self.kernels.dense_int8_batch_order
        self.pv_head_tile = 64 if self.quantized_cache else min(32, self.head_tile)
        self.adaptive = not self.prequantized and merged_heads <= 64
        self.value_merge_heads = 4 if self.adaptive else 0
        self.sort_batches = (
            self.quantized_cache
            and self.batch <= 256
            and (merged_heads > 64 or self.capacity >= 16384)
        )
        self.value_splits = (
            2
            if self.adaptive
            and self.quantized_cache
            and 16384 <= self.capacity <= 32768
            else 4 if self.adaptive else 1
        )
        self.uniform_value_splits = 2 if self.value_splits == 2 else 1
        self.score_splits = (
            triton.next_power_of_2(triton.cdiv(self.capacity, 128))
            if self.grouped and not self.prequantized
            else (
                16
                if self.quantized_cache
                and merged_heads <= 128
                and self.capacity >= 8192
                else 4
            )
        )
        if self.quantized_cache:
            self.scores = torch.empty(
                (self.batch, self.capacity // 64, merged_heads, 64),
                dtype=torch.uint16,
                device=query.device,
            )
            self.cache_metadata = torch.empty(
                (self.batch, self.capacity // 64, 2, merged_heads, 4),
                dtype=torch.float32,
                device=query.device,
            )
        else:
            self.scores = torch.empty(
                (rows, self.capacity), dtype=torch.float32, device=query.device
            )
            self.cache_metadata = torch.empty(
                (self.batch, self.capacity), dtype=torch.float32, device=query.device
            )
        self.probability_scale = torch.empty(
            (rows,), dtype=torch.float32, device=query.device
        )
        flag_shape = (self.batch + 1,) if self.sort_batches else ()
        self.split_flag = torch.empty(
            flag_shape, dtype=torch.int32, device=query.device
        )
        if self.prequantized:
            self.probabilities_high = torch.empty(
                (rows, self.capacity), dtype=torch.int8, device=query.device
            )
            self.probabilities_low = torch.empty_like(self.probabilities_high)
        else:
            stats_shape = (
                (self.batch, self.score_splits, 3, 4, merged_heads)
                if self.quantized_cache
                else (rows, self.score_splits, 2)
            )
            self.stats = torch.empty(
                stats_shape, dtype=torch.float32, device=query.device
            )
            self.probability_metadata = torch.empty(
                (rows, 2), dtype=torch.float32, device=query.device
            )
            workspace_splits = self.value_splits if self.adaptive else 0
            self.partial = torch.empty(
                (workspace_splits, rows, 512), dtype=torch.float32, device=query.device
            )
            self.partial_denominator = torch.empty(
                (workspace_splits, rows), dtype=torch.float32, device=query.device
            )

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        query, rope, cache, cache_rope, qs, ks, table, lengths = self.tensors
        merged_heads = self.queries * self.heads
        rows = self.batch * merged_heads
        score_arguments = (
            query.stride(),
            rope.stride(),
            cache.stride(),
            cache_rope.stride(),
            qs.stride(),
            ks.stride(),
            table.stride(0),
            lengths.stride(0),
            self.heads,
            self.queries,
            self.capacity,
            self.scale,
            self.causal,
        )
        with torch_device_fn.device(query.device):
            self._prepare_batch_order()
            if self.grouped:
                groups = (
                    triton.cdiv(self.capacity, 128)
                    if self.prequantized
                    else self.score_splits
                )
                self.kernels.dense_int8_scores_grouped[
                    (groups, triton.cdiv(merged_heads, self.head_tile), self.batch)
                ](
                    *self.tensors,
                    self.scores,
                    self.cache_metadata,
                    self.stats,
                    *score_arguments,
                    BH=self.head_tile,
                    SPLITS=self.score_splits,
                    WITH_STATS=not self.prequantized,
                    num_warps=4,
                    num_stages=1,
                )
            elif self.prequantized:
                self.kernels.dense_int8_scores_paged[
                    (
                        self.capacity // 64,
                        triton.cdiv(merged_heads, self.head_tile),
                        self.batch,
                    )
                ](
                    *self.tensors,
                    self.scores,
                    self.cache_metadata,
                    *score_arguments,
                    BH=self.head_tile,
                    num_warps=4,
                    num_stages=1,
                )
            else:
                self.kernels.dense_int8_scores_resident[
                    (
                        self.score_splits,
                        triton.cdiv(merged_heads, self.head_tile),
                        self.batch,
                    )
                ](
                    *self.tensors,
                    self.scores,
                    self.cache_metadata,
                    self.stats,
                    *score_arguments,
                    BH=self.head_tile,
                    SPLITS=self.score_splits,
                    COL_STATS=self.qk_native_statistics,
                    BatchOrder=self.batch_order,
                    num_warps=4,
                    num_stages=1,
                    scenario="mla",
                )
            if self.prequantized:
                self.kernels.dense_int8_probabilities[(rows,)](
                    self.scores,
                    self.cache_metadata,
                    self.probabilities_high,
                    self.probabilities_low,
                    self.probability_scale,
                    self.lse,
                    self.heads,
                    self.queries,
                    self.capacity,
                    triton.next_power_of_2(self.capacity),
                    lengths,
                    lengths.stride(0),
                    num_warps=8,
                    num_stages=1,
                )
                self.kernels.dense_int8_values_prequantized[
                    (2, triton.cdiv(merged_heads, self.pv_head_tile), self.batch)
                ](
                    cache,
                    table,
                    lengths,
                    self.probabilities_high,
                    self.probabilities_low,
                    self.probability_scale,
                    self.output,
                    cache.stride(),
                    table.stride(0),
                    lengths.stride(0),
                    self.heads,
                    self.queries,
                    self.capacity,
                    BH=self.pv_head_tile,
                    num_warps=4,
                    num_stages=1,
                )
            else:
                self.kernels.dense_int8_probability_merge[(triton.cdiv(rows, 32),)](
                    self.stats,
                    self.probability_metadata,
                    self.probability_scale,
                    self.lse,
                    lengths,
                    self.split_flag,
                    lengths.stride(0),
                    self.batch,
                    triton.next_power_of_2(self.batch),
                    self.heads,
                    self.queries,
                    rows,
                    self.score_splits,
                    self.adaptive,
                    QUANTIZED_CACHE=self.quantized_cache,
                    SORT_BATCHES=self.sort_batches,
                    num_warps=4,
                    num_stages=1,
                )
                self.kernels.dense_int8_values[
                    (
                        2,
                        triton.cdiv(merged_heads, self.pv_head_tile),
                        self.batch * self.value_splits,
                    )
                ](
                    cache,
                    table,
                    lengths,
                    self.scores,
                    self.cache_metadata,
                    self.probability_metadata,
                    self.probability_scale,
                    self.output,
                    self.lse,
                    self.partial,
                    self.partial_denominator,
                    self.split_flag,
                    cache.stride(),
                    table.stride(0),
                    lengths.stride(0),
                    self.batch,
                    self.heads,
                    self.queries,
                    self.capacity,
                    BH=self.pv_head_tile,
                    SPLITS=self.value_splits,
                    ADAPTIVE=self.adaptive,
                    UNIFORM_SPLITS=self.uniform_value_splits,
                    QUANTIZED_CACHE=self.quantized_cache,
                    SORT_BATCHES=self.sort_batches,
                    num_warps=4,
                    num_stages=1,
                    scenario="mla" if self.quantized_cache else "",
                )
                if self.adaptive:
                    self.kernels.dense_int8_value_merge[
                        (self.batch, triton.cdiv(merged_heads, 4))
                    ](
                        self.partial,
                        self.partial_denominator,
                        self.probability_metadata,
                        self.probability_scale,
                        self.output,
                        self.lse,
                        self.split_flag,
                        self.batch,
                        self.heads,
                        self.queries,
                        self.value_splits,
                        QUANTIZED_CACHE=self.quantized_cache,
                        UNIFORM_SPLITS=self.uniform_value_splits,
                        num_warps=4,
                        num_stages=1,
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
        raise ValueError("requires MetaX device tensors")
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
    capacity = block_table.shape[1] * 64
    merged_heads = queries * heads
    rows = batch * merged_heads
    # Bound INT32 PV sums, four-split grid dimensions and 16-byte copy rows.
    fast_path = (
        supports_maca_async_copy
        and torch.cuda.get_device_capability(q_nope.device) == (8, 0)
        and num_splits is None
        and 0 < batch < 2**14
        and heads > 0
        and triton.cdiv(merged_heads, 32) < 2**16
        and capacity <= 65536
        and batch * capacity < 2**31
        and all(
            tensor.data_ptr() % 16 == 0 for tensor in (q_nope, q_rope, k_lora, k_rope)
        )
        and all(
            stride % 16 == 0
            for tensor in (q_nope, k_lora)
            for stride in tensor.stride()[:-1]
        )
        and all(
            stride % 8 == 0
            for tensor in (q_rope, k_rope)
            for stride in tensor.stride()[:-1]
        )
        and not torch.cuda.is_current_stream_capturing()
    )
    small_heads = supports_maca_int8_col_major and merged_heads <= 32 and batch < 2**13
    if fast_path:
        prequantized = capacity <= 16384 and merged_heads < 64
        score_splits = (
            triton.next_power_of_2(triton.cdiv(capacity, 128))
            if merged_heads <= 32 and not prequantized
            else 16 if 32 < merged_heads <= 128 and capacity >= 8192 else 4
        )
        workspace_bytes = (
            4 * rows * capacity + 4 * batch * capacity + output_bytes + 4 * (rows + 1)
        )
        if prequantized:
            workspace_bytes += 2 * rows * capacity + 4 * rows
        else:
            workspace_bytes += rows * (2 * score_splits + 3) * 4
            if merged_heads <= 64:
                workspace_bytes += rows * 4 * (512 + 1) * 4
        if not prequantized and merged_heads > 32:
            workspace_bytes = (
                output_bytes
                + 4 * rows
                + 4
                * (
                    1
                    + (
                        batch
                        if batch <= 256 and (merged_heads > 64 or capacity >= 16384)
                        else 0
                    )
                )
            )
            workspace_bytes += rows * capacity * 2 + rows * (capacity // 64) * 8 * 4
            workspace_bytes += rows * score_splits * 12 * 4 + rows * 3 * 4
            if merged_heads <= 64:
                workspace_bytes += rows * 4 * (512 + 1) * 4
        if small_heads:
            workspace_bytes = output_bytes + 4 * rows + 8 * rows * (512 + 2) * 4
        if 1 < batch <= 256 and (
            (small_heads and merged_heads > 16)
            or (not prequantized and merged_heads > 32)
        ):
            workspace_bytes += 4 * batch
        free_bytes, _ = torch.cuda.mem_get_info(q_nope.device)
        # Cached allocator blocks are reusable without more device memory.
        free_bytes += torch.cuda.memory_reserved(
            q_nope.device
        ) - torch.cuda.memory_allocated(q_nope.device)
        fast_path = workspace_bytes <= min(8 * 1024**3, free_bytes // 2)
    if fast_path and small_heads:
        handle = FlashMLAInt8SmallHeadHandle(tensors, scale, causal)
    elif fast_path:
        handle = FlashMLAInt8GluonHandle(tensors, scale, causal)
    else:
        handle = FlashMLAInt8PreparedHandle(tensors, splits, scale, causal)
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
