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

from flaggems_vllm.runtime.backend._thead.fused.attention import (
    _flash_int8_merge_splits,
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
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    BH: tl.constexpr,
):
    head_tile, batch, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    heads = head_tile * BH + tl.arange(0, BH)
    dims = tl.arange(0, 512)
    rope_dims = tl.arange(0, 64)
    tokens = tl.arange(0, 64)
    q = tl.load(
        Q + batch * Q_STRIDES[0] + heads[:, None] * Q_STRIDES[2] + dims[None, :],
        heads[:, None] < H,
        0,
    )
    qr = tl.load(
        QR
        + batch * QR_STRIDES[0]
        + heads[:, None] * QR_STRIDES[2]
        + rope_dims[None, :],
        heads[:, None] < H,
        0,
    )
    qs = tl.load(QS + batch * QS_STRIDES[0] + heads * QS_STRIDES[2], heads < H, 0)
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.zeros((BH,), tl.float32)
    accumulator = tl.zeros((BH, 512), tl.float32)
    valid_pages = tl.cdiv(length, 64)
    first_page = valid_pages * split // SPLITS
    end_page = valid_pages * (split + 1) // SPLITS
    for page_idx in range(first_page, end_page):
        page = tl.load(Table + batch * TABLE_STRIDE + page_idx).to(tl.int64)
        kv = tl.load(
            KV + page * KV_STRIDES[0] + tokens[:, None] * KV_STRIDES[1] + dims[None, :]
        )
        kr = tl.load(
            KR
            + page * KR_STRIDES[0]
            + tokens[:, None] * KR_STRIDES[1]
            + rope_dims[None, :]
        )
        ks = tl.load(KS + page * KS_STRIDES[0] + tokens * KS_STRIDES[1])
        content_scores = tl.dot(q, tl.trans(kv), out_dtype=tl.int32).to(tl.float32)
        rope_scores = tl.dot(qr, tl.trans(kr), out_dtype=tl.float32)
        scores = (
            (content_scores + rope_scores)
            * qs[:, None]
            * ks[None, :]
            * (SCALE * 1.4426950408889634)
        )
        scores = tl.where(
            page_idx * 64 + tokens[None, :] < length, scores, -float("inf")
        )
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        probabilities = tl.exp2(scores - new_maximum[:, None])
        rescale = tl.exp2(maximum - new_maximum)
        denominator = denominator * rescale + tl.sum(probabilities, 1)
        # Fold each token's KV scale into P so PV remains an INT8 dot.
        scaled_probabilities = probabilities * ks[None, :]
        probability_scale = tl.maximum(tl.max(scaled_probabilities, 1) / 127.0, 1.0e-30)
        quantized_probabilities = tl.floor(
            scaled_probabilities / probability_scale[:, None] + 0.5
        ).to(tl.int8)
        # A second INT8 dot retains small probabilities without an FP16 PV path.
        residual = (
            scaled_probabilities / probability_scale[:, None]
            - quantized_probabilities.to(tl.float32)
        ) * 127.0
        quantized_residual = tl.floor(residual + 0.5).to(tl.int8)
        contribution = tl.dot(quantized_probabilities, kv, out_dtype=tl.int32).to(
            tl.float32
        )
        correction = tl.dot(quantized_residual, kv, out_dtype=tl.int32).to(tl.float32)
        contribution += correction * (1.0 / 127.0)
        accumulator = (
            accumulator * rescale[:, None] + contribution * probability_scale[:, None]
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
    """Reuse workspace for single-token decode; calls must be serialized on one stream."""

    def __init__(self, tensors: tuple[torch.Tensor, ...], splits: int, scale: float):
        self.tensors = tensors
        query, rope, _, _, _, _, table, _ = tensors
        self.batch, _, self.heads, _ = query.shape
        self.splits = splits
        self.scale = scale
        self.pages_per_split = triton.cdiv(table.shape[1], splits)
        self.num_warps = (
            8
            if self.pages_per_split <= 2
            and self.batch * triton.cdiv(self.heads, 16) < 64
            else 4
        )
        self.output = torch.empty(
            (self.batch, 1, self.heads, 512), dtype=rope.dtype, device=query.device
        )
        self.lse = torch.empty_strided(
            (self.batch, self.heads, 1),
            (1, self.batch, 1),
            dtype=torch.float32,
            device=query.device,
        )
        workspace_splits = splits if splits > 1 else 0
        self.partial = torch.empty(
            (workspace_splits, self.batch, self.heads, 512),
            dtype=torch.float32,
            device=query.device,
        )
        self.stats = torch.empty(
            (workspace_splits, 2, self.heads, self.batch),
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
        dense_int8_mla_kernel[(triton.cdiv(self.heads, 16), self.batch, self.splits)](
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
            self.heads,
            self.splits,
            self.scale,
            BH=16,
            num_warps=self.num_warps,
            num_stages=1,
        )
        if self.splits > 1:
            _flash_int8_merge_splits[(self.batch, self.heads)](
                self.partial,
                self.stats,
                self.output,
                self.lse,
                self.offsets,
                self.batch,
                self.heads,
                512,
                self.batch,
                self.splits,
                self.heads * 512,
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

    Active page IDs and lengths must be valid, scales finite and positive.
    Both causal settings attend the entire valid cache for single-token decode.
    """
    if q_nope.ndim != 4 or q_nope.shape[1] != 1 or q_nope.shape[-1] != 512:
        raise NotImplementedError("requires Q [batch, 1, heads, 512]")
    batch, _, heads, _ = q_nope.shape
    if k_lora.ndim != 3 or k_lora.shape[1:] != (64, 512) or head_dim_v != 512:
        raise NotImplementedError("requires KV [pages, 64, 512] and head_dim_v=512")
    expected_shapes = (
        (batch, 1, heads, 64),
        (k_lora.shape[0], 64, 64),
        (batch, 1, heads, 1),
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
                max(1, 64 // max(1, batch * triton.cdiv(heads, 16)))
            ),
        )
        splits = min(splits, max(1, triton.next_power_of_2(block_table.shape[1])))
    if splits not in (1, 2, 4, 8, 16):
        raise ValueError("num_splits must be one of 1, 2, 4, 8, 16")
    handle = FlashMLAInt8PreparedHandle(tensors, splits, scale)
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
