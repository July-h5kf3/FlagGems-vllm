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
from flaggems_vllm.runtime.backend._thead.fused.attention import (
    _flash_int8_merge_splits,
)
from flaggems_vllm.utils import libentry, libtuner


@triton.jit
def quantize_mla_probabilities(normalized_probabilities):
    # Direct conversion preserves round-half-up without a float floor round trip.
    coarse, residual = tl.inline_asm_elementwise(
        """{
        .reg .s32 coarse, residual;
        .reg .f32 rounded, coarse_fp, error;
        ppu.add.f32 rounded, $2, 0f3F000000;
        ppu.cvt.rmi.s32.f32 coarse, rounded;
        ppu.cvt.rn.f32.s32 coarse_fp, coarse;
        ppu.sub.f32 error, $2, coarse_fp;
        ppu.fma.rn.f32 rounded, error, 0f42FE0000, 0f3F000000;
        ppu.cvt.rmi.s32.f32 residual, rounded;
        ppu.mov.b32 $0, coarse;
        ppu.mov.b32 $1, residual;
        }""",
        "=r,=r,f",
        [normalized_probabilities],
        dtype=(tl.int32, tl.int32),
        is_pure=True,
        pack=1,
    )
    return coarse.to(tl.int8), residual.to(tl.int8)


@triton.jit
def _split_mla_features(values, rows: tl.constexpr, half_width: tl.constexpr):
    return tl.split(tl.permute(tl.reshape(values, (rows, 2, half_width)), (0, 2, 1)))


@triton.jit
def _sum_and_max(sum_left, max_left, sum_right, max_right):
    return sum_left + sum_right, tl.maximum(max_left, max_right)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dense_int8_mla_kernel"),
    key=["B", "H", "HEADS_PER_QUERY", "SPLITS", "TABLE_STRIDE", "CAUSAL"],
    use_cuda_graph=True,
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
    Offsets,
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
    HEADS_PER_QUERY: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    FUSED_PV: tl.constexpr,
    COLUMN_SOFTMAX: tl.constexpr,
    QK_SPLITS: tl.constexpr,
):
    head_tile, batch, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    heads = head_tile * BH + tl.arange(0, BH)
    query_index = heads // HEADS_PER_QUERY
    query_head = heads % HEADS_PER_QUERY
    dims = tl.arange(0, 512)
    rope_dims = tl.arange(0, 64)
    tokens = tl.arange(0, BK)
    reduction_axis: tl.constexpr = 0 if COLUMN_SOFTMAX else 1
    q = tl.load(
        Q
        + batch * Q_STRIDES[0]
        + query_index[:, None] * Q_STRIDES[1]
        + query_head[:, None] * Q_STRIDES[2]
        + dims[None, :],
        heads[:, None] < H,
        0,
    )
    if QK_SPLITS == 4:
        q_left, q_right = _split_mla_features(q, BH, 256)
        q0, q1 = _split_mla_features(q_left, BH, 128)
        q2, q3 = _split_mla_features(q_right, BH, 128)
    qr = tl.load(
        QR
        + batch * QR_STRIDES[0]
        + query_index[:, None] * QR_STRIDES[1]
        + query_head[:, None] * QR_STRIDES[2]
        + rope_dims[None, :],
        heads[:, None] < H,
        0,
    )
    qs = tl.load(
        QS
        + batch * QS_STRIDES[0]
        + query_index * QS_STRIDES[1]
        + query_head * QS_STRIDES[2],
        heads < H,
        0,
    )
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    query_limit = (
        tl.maximum(length - H // HEADS_PER_QUERY + query_index + 1, 0)
        if CAUSAL
        else tl.full((BH,), 0, tl.int32) + length
    )
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.zeros((BH,), tl.float32)
    accumulator = tl.zeros((BH, 512), tl.float32)
    valid_pages = tl.cdiv(length, 64)
    first_page = valid_pages * split // SPLITS
    end_page = valid_pages * (split + 1) // SPLITS
    for page_idx in range(first_page, end_page, BK // 64):
        page_offsets = page_idx + tokens // 64
        if BK == 64:
            page = tl.load(Table + batch * TABLE_STRIDE + page_idx).to(tl.int64)[None]
        else:
            page = tl.load(
                Table + batch * TABLE_STRIDE + page_offsets,
                page_offsets < end_page,
                0,
            ).to(tl.int64)
        valid_tokens = (page_idx * 64 + tokens < length) & (page_offsets < end_page)
        kv = tl.load(
            KV
            + page[:, None] * KV_STRIDES[0]
            + (tokens % 64)[:, None] * KV_STRIDES[1]
            + dims[None, :],
            valid_tokens[:, None],
            0,
        )
        kr = tl.load(
            KR
            + page[:, None] * KR_STRIDES[0]
            + (tokens % 64)[:, None] * KR_STRIDES[1]
            + rope_dims[None, :],
            valid_tokens[:, None],
            0,
        )
        ks = tl.load(
            KS + page * KS_STRIDES[0] + (tokens % 64) * KS_STRIDES[1], valid_tokens, 0
        )
        if QK_SPLITS == 4:
            kv_left, kv_right = _split_mla_features(kv, BK, 256)
            kv0, kv1 = _split_mla_features(kv_left, BK, 128)
            kv2, kv3 = _split_mla_features(kv_right, BK, 128)
            # INT32 partial sums are exact: 512 * 128 * 128 fits in INT32.
            content_scores = (
                tl.dot(q0, tl.trans(kv0), out_dtype=tl.int32)
                + tl.dot(q1, tl.trans(kv1), out_dtype=tl.int32)
                + tl.dot(q2, tl.trans(kv2), out_dtype=tl.int32)
                + tl.dot(q3, tl.trans(kv3), out_dtype=tl.int32)
            ).to(tl.float32)
        else:
            content_scores = tl.dot(q, tl.trans(kv), out_dtype=tl.int32).to(tl.float32)
        rope_scores = tl.dot(qr, tl.trans(kr), out_dtype=tl.float32)
        # Separating the dot chains avoids replicated QK warps on older compilers.
        qk = content_scores + rope_scores
        qk = tl.trans(qk) if COLUMN_SOFTMAX else qk
        scores = (
            qk
            * tl.expand_dims(qs, reduction_axis)
            * tl.expand_dims(ks, 1 - reduction_axis)
            * (SCALE * 1.4426950408889634)
        )
        scores = tl.where(
            (
                page_idx * 64 + tl.expand_dims(tokens, 1 - reduction_axis)
                < tl.expand_dims(query_limit, reduction_axis)
            )
            & tl.expand_dims(page_offsets < end_page, 1 - reduction_axis),
            scores,
            -float("inf"),
        )
        new_maximum = tl.maximum(maximum, tl.max(scores, reduction_axis))
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        probabilities = tl.exp2(scores - tl.expand_dims(safe_maximum, reduction_axis))
        rescale = tl.exp2(maximum - safe_maximum)
        # Fold each token's KV scale into P so PV remains an INT8 dot.
        scaled_probabilities = probabilities * tl.expand_dims(ks, 1 - reduction_axis)
        # Share cross-warp synchronization for the two row statistics.
        probability_sum, scaled_probability_max = tl.reduce(
            (probabilities, scaled_probabilities), reduction_axis, _sum_and_max
        )
        denominator = denominator * rescale + probability_sum
        probability_scale = tl.maximum(scaled_probability_max / 127.0, 1.0e-30)
        quantized_probabilities, quantized_residual = quantize_mla_probabilities(
            scaled_probabilities / tl.expand_dims(probability_scale, reduction_axis)
        )
        quantized_probabilities = (
            tl.trans(quantized_probabilities)
            if COLUMN_SOFTMAX
            else quantized_probabilities
        )
        quantized_residual = (
            tl.trans(quantized_residual) if COLUMN_SOFTMAX else quantized_residual
        )
        if FUSED_PV:
            # Up to 256 tokens keep 127 * high + correction within INT32.
            high = tl.dot(quantized_probabilities, kv, out_dtype=tl.int32)
            contribution = tl.dot(
                quantized_residual, kv, high * 127, out_dtype=tl.int32
            ).to(tl.float32)
            contribution_scale = probability_scale * (1.0 / 127.0)
        else:
            contribution = tl.dot(quantized_probabilities, kv, out_dtype=tl.int32).to(
                tl.float32
            )
            correction = tl.dot(quantized_residual, kv, out_dtype=tl.int32).to(
                tl.float32
            )
            contribution += correction * (1.0 / 127.0)
            contribution_scale = probability_scale
        accumulator = (
            accumulator * rescale[:, None] + contribution * contribution_scale[:, None]
        )
        maximum = new_maximum
    if SPLITS == 1:
        normalized = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(
            Output + (batch * H + heads[:, None]) * 512 + dims[None, :],
            normalized,
            heads[:, None] < H,
        )
        logsum = tl.where(
            denominator > 0,
            maximum * 0.6931471805599453 + tl.log(denominator),
            float("inf"),
        )
        tl.store(LSE + heads * B + batch, logsum, heads < H)
    else:
        tl.store(
            Partial + ((split * B + batch) * H + heads[:, None]) * 512 + dims[None, :],
            accumulator,
            heads[:, None] < H,
        )
        tl.store(Stats + (split * 2 * H + heads) * B + batch, maximum, heads < H)
        tl.store(
            Stats + ((split * 2 + 1) * H + heads) * B + batch, denominator, heads < H
        )
        if head_tile == 0 and split == 0:
            tl.store(Offsets + batch, batch)
            if batch == 0:
                tl.store(Offsets + B, B)


class FlashMLAInt8PreparedHandle:
    """Reuse workspace for one/two-token decode; calls must be serialized on one stream."""

    def __init__(
        self,
        tensors: tuple[torch.Tensor, ...],
        splits: int,
        scale: float,
        causal: bool = False,
    ):
        self.tensors = tensors
        query, rope = tensors[:2]
        self.batch, self.queries, self.heads, _ = query.shape
        self.total_heads = self.queries * self.heads
        self.causal = causal
        self.splits = splits
        self.scale = scale
        self.output = torch.empty(
            (self.batch, self.queries, self.heads, 512),
            dtype=rope.dtype,
            device=query.device,
        )
        self.lse = torch.empty_strided(
            (self.batch, self.heads, self.queries),
            (1, self.batch, self.batch * self.heads),
            dtype=torch.float32,
            device=query.device,
        )
        workspace_splits = splits if splits > 1 else 0
        self.partial = torch.empty(
            (workspace_splits, self.batch, self.total_heads, 512),
            dtype=torch.float32,
            device=query.device,
        )
        self.stats = torch.empty(
            (workspace_splits, 2, self.total_heads, self.batch),
            dtype=torch.float32,
            device=query.device,
        )
        self.offsets = torch.empty(
            (self.batch + 1 if splits > 1 else 0,),
            dtype=torch.int32,
            device=query.device,
        )

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch == 0 or self.heads == 0:
            return self.output, self.lse
        query, rope, kv, kv_rope, qs, ks, table, lengths = self.tensors
        dense_int8_mla_kernel[
            lambda meta: (
                triton.cdiv(self.total_heads, meta["BH"]),
                self.batch,
                self.splits,
            )
        ](
            *self.tensors,
            self.partial,
            self.stats,
            self.offsets,
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
            self.total_heads,
            self.heads,
            self.splits,
            self.scale,
            self.causal,
        )
        if self.splits > 1:
            _flash_int8_merge_splits[(self.batch, self.total_heads)](
                self.partial,
                self.stats,
                self.output,
                self.lse,
                self.offsets,
                self.batch,
                self.total_heads,
                512,
                self.batch,
                self.splits,
                self.total_heads * 512,
                512,
                True,
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
    """RoPE inputs are divided by the content scale, matching PR851.

    Active page IDs and lengths must be valid; active scales finite and positive.
    Unused cache slots may contain NaNs.
    Queries are the final one or two cache tokens; causal masks preserve their order.
    """
    if q_nope.ndim != 4 or q_nope.shape[1] not in (1, 2) or q_nope.shape[-1] != 512:
        raise NotImplementedError("requires Q [batch, 1 or 2, heads, 512]")
    batch, queries, heads, _ = q_nope.shape
    if k_lora.ndim != 3 or k_lora.shape[1:] != (64, 512) or head_dim_v != 512:
        raise NotImplementedError("requires KV [pages, 64, 512] and head_dim_v=512")
    expected_shapes = (
        (batch, queries, heads, 64),
        (k_lora.shape[0], 64, 64),
        (batch, queries, heads, 1),
        (k_lora.shape[0], 64, 1),
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
    scale = 576**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    splits = num_splits
    if splits is None:
        splits = min(
            16,
            triton.next_power_of_2(
                max(1, 64 // max(1, batch * triton.cdiv(queries * heads, 16)))
            ),
        )
        splits = min(splits, max(1, triton.next_power_of_2(block_table.shape[1])))
    if splits not in (1, 2, 4, 8, 16):
        raise ValueError("num_splits must be one of 1, 2, 4, 8, 16")
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
