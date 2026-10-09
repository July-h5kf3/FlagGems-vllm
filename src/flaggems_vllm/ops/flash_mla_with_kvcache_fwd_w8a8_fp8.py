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

from __future__ import annotations

from typing import Optional, Sequence

import torch

import flaggems_vllm


class FlashMLAFp8SplitKSchedMeta:
    """Reusable Split-K scheduling metadata."""

    def __init__(self) -> None:
        self.have_initialized = False
        self.config = None
        self.tile_scheduler_metadata = None
        self.num_splits = None
        self.split_batch = None
        self.split_page_begin = None
        self.split_page_end = None
        self.split_num_pages = None
        self.max_splits = 1
        self.total_split_capacity = 0
        self.max_pages_per_split = 0
        self.lifetime_safe_one_pair = True
        self.cache_seqlens_data_ptr = 0
        self.cache_seqlens_version = -1
        self.num_splits_data_ptr = 0
        self.num_splits_version = -1
        self.adaptive_fixed_pages = None
        self.adaptive_fixed_pairs = None
        self.adaptive_selection = ()
        self.capacity_splits = ()
        self.padded_pages = 0


class FlashMLAFp8PreparedHandle:
    """Nominal base for the public prepared handle returned by backend registration."""


def get_mla_fp8_metadata(
    cache_seqlens: Optional[torch.Tensor] = None,
    num_q_heads_per_k_head: Optional[int] = None,
    num_k_heads: int = 1,
    *,
    pages_per_split: int = 2,
    max_splits: Optional[int] = None,
) -> tuple[FlashMLAFp8SplitKSchedMeta, Optional[torch.Tensor]]:
    implementation = flaggems_vllm.get_mla_fp8_metadata
    if implementation is get_mla_fp8_metadata:
        raise NotImplementedError("FP8 dense MLA is not registered for this backend")
    return implementation(
        cache_seqlens,
        num_q_heads_per_k_head,
        num_k_heads,
        pages_per_split=pages_per_split,
        max_splits=max_splits,
    )


def prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLAFp8SplitKSchedMeta | None = None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    pages_per_split: int = 2,
    max_splits: Optional[int] = None,
    *,
    initial_cache_seqlens: Sequence[int],
    max_cache_seqlens: Sequence[int],
) -> tuple[FlashMLAFp8PreparedHandle, tuple[torch.Tensor, torch.Tensor]]:
    implementation = flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_fp8
    if implementation is prepare_flash_mla_with_kvcache_fwd_w8a8_fp8:
        raise NotImplementedError("FP8 dense MLA is not registered for this backend")
    return implementation(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        causal,
        pages_per_split,
        max_splits,
        initial_cache_seqlens=initial_cache_seqlens,
        max_cache_seqlens=max_cache_seqlens,
    )


def flash_mla_with_kvcache_fwd_w8a8_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLAFp8SplitKSchedMeta | None = None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    pages_per_split: int = 2,
    max_splits: Optional[int] = None,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    implementation = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_fp8
    if implementation is flash_mla_with_kvcache_fwd_w8a8_fp8:
        raise NotImplementedError("FP8 dense MLA is not registered for this backend")
    return implementation(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        causal,
        pages_per_split,
        max_splits,
        out,
        lse,
    )
