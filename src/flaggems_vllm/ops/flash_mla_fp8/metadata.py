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


# flake8: noqa: E501,F841

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from flaggems_vllm.ops.flash_mla_fp8.common import (
    D_CKV,
    D_ROPE,
    DEFAULT_PAGES_PER_SPLIT,
    FP8_MAX,
    MAX_SEQUENCE_LENGTH,
    PAGE_SIZE,
    TLE_FP8_BH,
)

FP8_DTYPE = torch.float8_e4m3fn

ADAPTIVE_MODEL_MIN_PAGES = 69

ADAPTIVE_MIN_FIXED_PAGES = 4

ADAPTIVE_MAX_FIXED_PAGES = 32

ADAPTIVE_TAIL_WAVE_MAX_FIXED_PAGES = 34

ADAPTIVE_CTA_PENALTY = 0.5

CUDA_REF_FIXED_OVERHEAD_PAGES = 5


def _per_token_scale(content_abs_amax: torch.Tensor, safe: bool) -> torch.Tensor:
    scale = content_abs_amax / FP8_MAX
    if safe:
        scale = torch.where(content_abs_amax == 0, torch.ones_like(scale), scale)
    return scale


def quantize_q_ckv_per_token(
    q: torch.Tensor,
    head_dim_v: int = D_CKV,
    safe: bool = True,
):
    assert q.shape[-1] == head_dim_v + D_ROPE
    q_nope = q[..., :head_dim_v]
    q_rope = q[..., head_dim_v:]
    amax = q_nope.float().abs().amax(dim=-1, keepdim=True)
    scale = _per_token_scale(amax, safe)
    q_nope_fp8 = (q_nope.float() / scale).to(FP8_DTYPE)
    q_rope_aligned = (q_rope.float() / scale).to(q.dtype)
    return q_nope_fp8, q_rope_aligned, scale.float()


def quantize_k_ckv_per_token(
    blocked_k: torch.Tensor,
    head_dim_v: int = D_CKV,
    safe: bool = True,
):
    assert blocked_k.shape[-1] == head_dim_v + D_ROPE
    k_lora = blocked_k[..., :head_dim_v]
    k_rope = blocked_k[..., head_dim_v:]
    amax = k_lora.float().abs().amax(dim=-1, keepdim=True)
    scale = _per_token_scale(amax, safe)
    k_lora_fp8 = (k_lora.float() / scale).to(FP8_DTYPE)
    k_rope_aligned = (k_rope.float() / scale).to(blocked_k.dtype)
    return k_lora_fp8, k_rope_aligned, scale.float()


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


def _tensor_version(tensor: torch.Tensor) -> int:
    try:
        return int(tensor._version)
    except RuntimeError:
        return -1


def _host_lengths(value, label: str, *, batch_size: Optional[int] = None):
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        lengths = tuple(int(item) for item in value.detach().cpu().tolist())
    else:
        lengths = tuple(int(item) for item in value)
    if batch_size is not None and len(lengths) != batch_size:
        raise ValueError(f"{label} must have {batch_size} entries")
    if any(length < 0 or length > MAX_SEQUENCE_LENGTH for length in lengths):
        raise ValueError(f"{label} entries must be in [0, {MAX_SEQUENCE_LENGTH}]")
    return lengths


def _host_certificate_lengths(
    value,
    label: str,
    *,
    batch_size: Optional[int] = None,
):
    """Parse the host-only vectors used by the prepared decode contract."""
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be host integers, not a Tensor")
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be an iterable of host integers")
    try:
        iterator = iter(value)
    except TypeError:
        lengths = (int(value),)
    else:
        lengths = tuple(int(item) for item in iterator)
    if not lengths:
        raise ValueError(f"{label} must not be empty")
    if batch_size is not None and len(lengths) != batch_size:
        raise ValueError(f"{label} must have {batch_size} entries")
    if any(length <= 0 or length > MAX_SEQUENCE_LENGTH for length in lengths):
        raise ValueError(f"{label} entries must be in [1, {MAX_SEQUENCE_LENGTH}]")
    return lengths


def _length_page_state(lengths, pages_per_split: int):
    pages = tuple(math.ceil(int(length) / PAGE_SIZE) for length in lengths)
    splits = tuple(math.ceil(page_count / pages_per_split) for page_count in pages)
    return pages, splits


def _adaptive_fixed_pages(max_pages: int) -> int:
    safety_splits = math.ceil(max_pages / 16)
    required_pages = math.ceil(max_pages / safety_splits)
    return min(16, max(2, 2 * math.ceil(required_pages / 2)))


def _wave_grain_selection(
    max_cache_seqlens: tuple[int, ...],
    h_q: int,
    sm_count: int,
):
    capacity_pages = tuple(
        math.ceil(int(length) / PAGE_SIZE) for length in max_cache_seqlens
    )
    max_pages = max(capacity_pages, default=0)
    if max_pages < ADAPTIVE_MODEL_MIN_PAGES:
        selected = _adaptive_fixed_pages(max_pages)
        return selected, (
            {
                "pages": selected,
                "policy": "adaptive_short_sequence",
                "max_pages": max_pages,
            },
        )

    if h_q <= 0 or h_q % TLE_FP8_BH:
        raise ValueError("HQ must be a positive multiple of 64")
    if sm_count <= 0:
        raise ValueError("SM count must be positive")
    rh = h_q // TLE_FP8_BH

    # CUDA authority (`get_mla_metadata.cu`) assigns each SM partition a
    # payload that includes five fixed-overhead page blocks.  Preserve this
    # implementation's fixed even-pair routing, but derive its grain from the
    # same payload model and round the usable page count up to a whole pair.
    num_sm_parts = max(1, sm_count // rh)
    total_num_blocks = sum(
        pages + CUDA_REF_FIXED_OVERHEAD_PAGES for pages in capacity_pages
    )
    payload_blocks = max(
        math.ceil(total_num_blocks / num_sm_parts) + CUDA_REF_FIXED_OVERHEAD_PAGES,
        2 * CUDA_REF_FIXED_OVERHEAD_PAGES,
    )
    usable_pages = payload_blocks - CUDA_REF_FIXED_OVERHEAD_PAGES
    selected_pages = min(
        ADAPTIVE_MAX_FIXED_PAGES,
        max(
            ADAPTIVE_MIN_FIXED_PAGES,
            2 * math.ceil(usable_pages / 2),
        ),
    )

    # A uniform per-row grain can leave only a handful of CTAs in a second
    # wave.  Keep the fixed even-pair contract, but allow the smallest larger
    # even grain when it collapses that sparse tail back into one H800 wave.
    # This is deliberately capped at 34 pages: it changes B8/L33280 from
    # 8 * ceil(520 / 32) = 136 CTAs to 8 * ceil(520 / 34) = 128 CTAs while
    # leaving the other formal routing points unchanged.
    initial_selected_pages = selected_pages
    initial_counts = tuple(
        max(1, math.ceil(pages / selected_pages)) for pages in capacity_pages
    )
    initial_total_ctas = sum(initial_counts) * rh
    tail_wave_eliminated = False
    if sm_count < initial_total_ctas <= 2 * sm_count:
        for candidate_pages in range(
            selected_pages + 2,
            ADAPTIVE_TAIL_WAVE_MAX_FIXED_PAGES + 1,
            2,
        ):
            candidate_counts = tuple(
                max(1, math.ceil(pages / candidate_pages)) for pages in capacity_pages
            )
            if sum(candidate_counts) * rh <= sm_count:
                selected_pages = candidate_pages
                tail_wave_eliminated = True
                break

    records = []
    for fixed_pages in range(
        ADAPTIVE_MIN_FIXED_PAGES,
        ADAPTIVE_MAX_FIXED_PAGES + 1,
        2,
    ):
        counts = tuple(
            max(1, math.ceil(pages / fixed_pages)) for pages in capacity_pages
        )
        total_splits = sum(counts)
        total_ctas = total_splits * rh
        waves = math.ceil(total_ctas / sm_count)
        fixed_pairs = fixed_pages // 2
        score = waves * (fixed_pairs + 1) + ADAPTIVE_CTA_PENALTY * total_ctas / sm_count
        records.append(
            {
                "pages": fixed_pages,
                "pairs": fixed_pairs,
                "total_splits": total_splits,
                "total_ctas": total_ctas,
                "waves": waves,
                "score": score,
                "policy": "h800_wave_cost",
            }
        )
    selected_counts = tuple(
        max(1, math.ceil(pages / selected_pages)) for pages in capacity_pages
    )
    selection = {
        "pages": selected_pages,
        "pairs": selected_pages // 2,
        "total_splits": sum(selected_counts),
        "total_ctas": sum(selected_counts) * rh,
        "num_sm_parts": num_sm_parts,
        "fixed_overhead_pages": CUDA_REF_FIXED_OVERHEAD_PAGES,
        "payload_blocks": payload_blocks,
        "usable_pages_before_pair_rounding": usable_pages,
        "policy": (
            "cuda_tail_wave_elimination_even_pair"
            if tail_wave_eliminated
            else "cuda_fixed_overhead_even_pair_payload"
        ),
        "tail_wave_eliminated": tail_wave_eliminated,
        "initial_selected_pages": initial_selected_pages,
        "initial_total_ctas": initial_total_ctas,
    }
    return int(selected_pages), (selection, *records)


def _adaptive_schedule(max_cache_seqlens, pages_per_split: int):
    capacity_pages = tuple(
        math.ceil(int(length) / PAGE_SIZE) for length in max_cache_seqlens
    )
    counts = tuple(
        max(1, math.ceil(pages / pages_per_split)) for pages in capacity_pages
    )
    prefix = [0]
    split_batch = []
    split_page_begin = []
    split_page_end = []
    split_num_pages = []
    for batch_index, (pages, count) in enumerate(zip(capacity_pages, counts)):
        prefix.append(prefix[-1] + count)
        for split_index in range(count):
            if pages_per_split == 16 and pages == 520 and count == 33:
                # Pair-aligned balancing: 29x16 + 4x14 = 520 pages.  Spread
                # the four 14-page splits through the row so no 8-page tail
                # remains, while every split retains an even page count.
                short_before = (split_index * 4) // count
                short_through = ((split_index + 1) * 4) // count
                page_begin = split_index * 16 - 2 * short_before
                num_pages = 14 if short_through != short_before else 16
                page_end = page_begin + num_pages
            elif pages_per_split == 34 and pages == 520 and count == 16:
                page_begin = (pages * split_index) // count
                page_end = (pages * (split_index + 1)) // count
                num_pages = page_end - page_begin
            else:
                page_begin = split_index * pages_per_split
                num_pages = max(0, min(pages_per_split, pages - page_begin))
            split_batch.append(batch_index)
            split_page_begin.append(page_begin)
            split_page_end.append(page_begin + num_pages)
            split_num_pages.append(num_pages)
    padded_pages = max(
        1,
        max(
            (count * pages_per_split for count in counts),
            default=1,
        ),
    )
    return (
        tuple(prefix),
        tuple(split_batch),
        tuple(split_page_begin),
        tuple(split_page_end),
        tuple(split_num_pages),
        counts,
        padded_pages,
    )


def _build_adaptive_execution_meta(
    max_cache_seqlens,
    h_q: int,
    device: torch.device,
    short_pages_per_split: int,
):
    capacity_pages = tuple(
        math.ceil(int(length) / PAGE_SIZE) for length in max_cache_seqlens
    )
    max_pages = max(capacity_pages, default=0)
    if max_pages <= 2:
        fixed_pages = int(short_pages_per_split)
        selection = (
            {
                "pages": fixed_pages,
                "policy": "direct_two_page",
                "max_pages": max_pages,
            },
        )
    elif 3 <= max_pages <= 8:
        # Short-K route: expose one physical-page CTA at a time instead of
        # serializing the complete 3-8 page row in a single CTA.  This is
        # host scheduling only; the strict-2WG kernel and pair pipeline are
        # unchanged.
        fixed_pages = 1
        selection = (
            {
                "pages": fixed_pages,
                "pairs": 1,
                "policy": "shortk_pagegrain_3_to_8_pages_v1",
                "max_pages": max_pages,
            },
        )
    elif (
        max_pages == 10
        and len(capacity_pages) >= 32
        and all(pages == 10 for pages in capacity_pages)
    ):
        # At high batch, five two-page split CTAs per row over-subscribe the
        # short ten-page workload and require a combine kernel. Use one
        # direct-output CTA per (batch, 64-head group) and only finalize LSE.
        # Keep this eligibility exact for heterogeneous rows and adjacent lengths.
        fixed_pages = max_pages
        selection = (
            {
                "pages": fixed_pages,
                "pairs": math.ceil(fixed_pages / 2),
                "policy": "b32plus_l640_direct_single",
                "max_pages": max_pages,
            },
        )
    elif (
        max_pages == 10
        and h_q == 128
        and len(capacity_pages) == 16
        and all(pages == 10 for pages in capacity_pages)
    ):
        # For B16/L640, reduce the partial grid from 160 CTAs (five
        # two-page splits per row) to 96 CTAs (three four-page-capacity
        # splits per row and two head groups).
        fixed_pages = 4
        selection = (
            {
                "pages": fixed_pages,
                "pairs": fixed_pages // 2,
                "policy": "b16_l640_four_page_grain",
                "max_pages": max_pages,
            },
        )
    elif 9 <= max_pages <= 10:
        # A ten-page direct-single CTA is not the best short-sequence route.  Use the finest
        # legal two-page pair grain to expose five split CTAs, mirroring the
        # CUDA reference's short-workload parallel split behavior.  Keep the
        # policy deliberately narrow until adjacent page ranges are measured.
        fixed_pages = 2
        selection = (
            {
                "pages": fixed_pages,
                "pairs": 1,
                "policy": "cuda_short_parallel_pair_9_to_10_pages",
                "max_pages": max_pages,
            },
        )
    elif (
        max_pages == 128
        and h_q == 64
        and len(capacity_pages) >= 64
        and all(pages == 128 for pages in capacity_pages)
    ):
        # Choose an even per-row grain that targets one H800
        # partial-CTA wave for a regular high-batch 8192-token workload.
        # This changes only host scheduling metadata; the partial kernel,
        # TMA/WGMMA/barrier structure, math, and route contracts are reused.
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        target_splits_per_row = max(1, sm_count // len(capacity_pages))
        fixed_pages = min(
            max_pages,
            2 * math.ceil(math.ceil(max_pages / target_splits_per_row) / 2),
        )
        selection = (
            {
                "pages": fixed_pages,
                "pairs": fixed_pages // 2,
                "policy": "b64plus_l8192_onewave_even_grain",
                "max_pages": max_pages,
            },
        )
    elif (
        max_pages == 520
        and h_q == 64
        and len(capacity_pages) == 16
        and all(pages == 520 for pages in capacity_pages)
    ):
        fixed_pages = 65
        selection = (
            {"pages": 65, "pairs": 33, "policy": "b16_l33280_uniform_grain65"},
        )
    elif (
        max_pages == 520
        and h_q == 64
        and len(capacity_pages) >= 16
        and all(pages == 520 for pages in capacity_pages)
    ):
        # Choose the minimum even grain that caps the regular high-batch
        # L33280 workload at one H800 partial-CTA wave.
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        target_splits_per_row = max(1, sm_count // len(capacity_pages))
        fixed_pages = min(
            max_pages,
            2 * math.ceil(math.ceil(max_pages / target_splits_per_row) / 2),
        )
        selection = (
            {
                "pages": fixed_pages,
                "pairs": fixed_pages // 2,
                "policy": "b16plus_l33280_onewave_even_grain",
                "max_pages": max_pages,
            },
        )
    elif (
        h_q == 64
        and len(capacity_pages) == 16
        and all(pages == 128 for pages in capacity_pages)
    ):
        fixed_pages = 16
        selection = ({"pages": 16, "pairs": 8, "policy": "b16_l8192_balanced_grain16"},)
    else:
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        fixed_pages, selection = _wave_grain_selection(
            tuple(max_cache_seqlens), h_q, sm_count
        )
    (
        prefix,
        split_batch,
        split_page_begin,
        split_page_end,
        split_num_pages,
        counts,
        padded_pages,
    ) = _adaptive_schedule(max_cache_seqlens, fixed_pages)

    meta = FlashMLAFp8SplitKSchedMeta()
    meta.have_initialized = True
    meta.num_splits = torch.tensor(prefix, dtype=torch.int32, device=device)
    meta.split_batch = torch.tensor(split_batch, dtype=torch.int32, device=device)
    meta.split_page_begin = torch.tensor(
        split_page_begin, dtype=torch.int32, device=device
    )
    meta.split_page_end = torch.tensor(split_page_end, dtype=torch.int32, device=device)
    meta.split_num_pages = torch.tensor(
        split_num_pages, dtype=torch.int32, device=device
    )
    meta.max_splits = max(counts, default=1)
    meta.total_split_capacity = len(split_batch)
    meta.max_pages_per_split = max(split_num_pages, default=0)
    meta.lifetime_safe_one_pair = meta.max_pages_per_split <= 2
    meta.num_splits_data_ptr = int(meta.num_splits.data_ptr())
    meta.num_splits_version = _tensor_version(meta.num_splits)
    meta.adaptive_fixed_pages = fixed_pages
    meta.adaptive_fixed_pairs = math.ceil(fixed_pages / 2)
    meta.adaptive_selection = selection
    meta.capacity_splits = counts
    meta.padded_pages = padded_pages
    return meta


def _pad_block_table(block_table: torch.Tensor, padded_pages: int):
    if int(block_table.shape[1]) >= padded_pages:
        return block_table
    padded = torch.zeros(
        (int(block_table.shape[0]), padded_pages),
        dtype=block_table.dtype,
        device=block_table.device,
    )
    padded[:, : int(block_table.shape[1])].copy_(block_table)
    return padded


def _fixed_split_counts(
    cache_seqlens: torch.Tensor,
    pages_per_split: int,
    max_splits: Optional[int] = None,
) -> torch.Tensor:
    if pages_per_split <= 0:
        raise ValueError("pages_per_split must be positive")
    pages = torch.div(
        cache_seqlens.to(torch.int64) + PAGE_SIZE - 1,
        PAGE_SIZE,
        rounding_mode="floor",
    )
    counts = torch.div(
        pages + pages_per_split - 1,
        pages_per_split,
        rounding_mode="floor",
    ).clamp_min(1)
    if max_splits is not None:
        if max_splits <= 0:
            raise ValueError("max_splits must be positive")
        counts = counts.clamp_max(max_splits)
    return counts.to(torch.int32)


def _prefix_from_counts(counts: torch.Tensor) -> torch.Tensor:
    prefix = torch.empty((counts.numel() + 1,), dtype=torch.int32, device=counts.device)
    prefix[0] = 0
    prefix[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    return prefix


def get_mla_fp8_metadata(
    cache_seqlens: Optional[torch.Tensor] = None,
    num_q_heads_per_k_head: Optional[int] = None,
    num_k_heads: int = 1,
    *,
    pages_per_split: int = DEFAULT_PAGES_PER_SPLIT,
    max_splits: Optional[int] = None,
) -> Tuple[FlashMLAFp8SplitKSchedMeta, Optional[torch.Tensor]]:
    meta = FlashMLAFp8SplitKSchedMeta()
    if cache_seqlens is None:
        return meta, None
    if cache_seqlens.ndim != 1 or cache_seqlens.dtype != torch.int32:
        raise AssertionError("cache_seqlens must be a 1-D int32 tensor")

    counts = _fixed_split_counts(cache_seqlens, pages_per_split, max_splits)
    prefix = _prefix_from_counts(counts)
    actual_max = int(counts.max().item()) if counts.numel() else 1
    h_q = int(num_q_heads_per_k_head or TLE_FP8_BH) * int(num_k_heads)
    meta.have_initialized = True
    meta.max_splits = actual_max
    meta.total_split_capacity = int(cache_seqlens.numel()) * actual_max
    meta.lifetime_safe_one_pair = pages_per_split <= 2 and max_splits is None
    meta.num_splits = prefix
    meta.cache_seqlens_data_ptr = int(cache_seqlens.data_ptr())
    meta.cache_seqlens_version = _tensor_version(cache_seqlens)
    meta.num_splits_data_ptr = int(prefix.data_ptr())
    meta.num_splits_version = _tensor_version(prefix)
    (
        meta.split_batch,
        meta.split_page_begin,
        meta.split_page_end,
        meta.split_num_pages,
    ) = _build_compact_split_plan(cache_seqlens, prefix)
    meta.total_split_capacity = int(meta.split_batch.numel())
    meta.max_pages_per_split = (
        int(meta.split_num_pages.max().item()) if meta.total_split_capacity else 0
    )
    meta.lifetime_safe_one_pair = meta.max_pages_per_split <= 2
    return meta, prefix


def _split_page_bounds(num_pages: int, split_idx: int, split_count: int):
    if split_count <= 0 or not 0 <= split_idx < split_count:
        raise ValueError("invalid split index/count")
    return (
        (num_pages * split_idx) // split_count,
        (num_pages * (split_idx + 1)) // split_count,
    )


def _build_compact_split_plan(cache_seqlens, num_splits):
    seqlens_cpu = cache_seqlens.detach().cpu()
    prefix_cpu = num_splits.detach().cpu()
    split_batch = []
    split_page_begin = []
    split_page_end = []
    split_num_pages = []
    for batch_idx in range(cache_seqlens.numel()):
        cache_len = int(seqlens_cpu[batch_idx].item())
        num_pages = (cache_len + PAGE_SIZE - 1) // PAGE_SIZE
        begin = int(prefix_cpu[batch_idx].item())
        end = int(prefix_cpu[batch_idx + 1].item())
        split_count = end - begin
        if split_count <= 0:
            raise AssertionError("every request must own at least one split")
        for local_split in range(split_count):
            page_begin, page_end = _split_page_bounds(
                num_pages, local_split, split_count
            )
            split_batch.append(batch_idx)
            split_page_begin.append(page_begin)
            split_page_end.append(page_end)
            split_num_pages.append(page_end - page_begin)

    device = cache_seqlens.device
    return (
        torch.tensor(split_batch, dtype=torch.int32, device=device),
        torch.tensor(split_page_begin, dtype=torch.int32, device=device),
        torch.tensor(split_page_end, dtype=torch.int32, device=device),
        torch.tensor(split_num_pages, dtype=torch.int32, device=device),
    )


FlashMLAFp8SchedMeta = FlashMLAFp8SplitKSchedMeta
