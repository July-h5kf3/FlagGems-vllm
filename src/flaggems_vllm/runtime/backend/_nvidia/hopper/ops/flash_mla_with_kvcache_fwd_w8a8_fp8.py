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

import math
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.flash_mla import HAS_TLE_FLASH_MLA as HAS_TLE
from flaggems_vllm.ops.flash_mla import (
    _ensure_triton_descriptor_allocator,
    _get_num_sms,
    _get_tensor_descriptor_cls,
)
from flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8 import (
    FlashMLAFp8PreparedHandle as PreparedHandleBase,
)
from flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8 import (
    FlashMLAFp8SplitKSchedMeta,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.mla_fp8_constants import (
    ADAPTIVE_CTA_PENALTY,
    ADAPTIVE_MAX_FIXED_PAGES,
    ADAPTIVE_MIN_FIXED_PAGES,
    ADAPTIVE_MODEL_MIN_PAGES,
    ADAPTIVE_TAIL_WAVE_MAX_FIXED_PAGES,
    COMBINE_BLOCK_D,
    COMBINE_BLOCK_SPLITS,
    CUDA_COARSE_COMBINE_BLOCK_ROWS,
    CUDA_COARSE_COMBINE_BLOCK_SPLITS,
    CUDA_COARSE_COMBINE_MIN_BATCH,
    CUDA_REF_FIXED_OVERHEAD_PAGES,
    D_CKV,
    D_QK,
    D_ROPE,
    DEFAULT_PAGES_PER_SPLIT,
    K_CONTENT_TILE_HOST,
    LSE_FINALIZE_BLOCK,
    MAX_SEQUENCE_LENGTH,
    PAGE_SIZE,
    TLE_FP8_BH,
    TLE_FP8_BK,
    TLE_FP8_DPH,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.mla_fp8_kernels import (
    fp8_dense_mla_splitk_partial,
    triton_fp8_coarse_combine_kernel,
    triton_fp8_single_split_lse_finalize_kernel,
    triton_fp8_splitk_combine_kernel,
)
from flaggems_vllm.utils import libentry, libtuner


def tensor_version(tensor: torch.Tensor) -> int:
    try:
        return int(tensor._version)
    except RuntimeError:
        return -1


def host_certificate_lengths(
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


def length_page_state(lengths, pages_per_split: int):
    pages = tuple(math.ceil(int(length) / PAGE_SIZE) for length in lengths)
    splits = tuple(math.ceil(page_count / pages_per_split) for page_count in pages)
    return pages, splits


def adaptive_fixed_pages(max_pages: int) -> int:
    safety_splits = math.ceil(max_pages / 16)
    required_pages = math.ceil(max_pages / safety_splits)
    return min(16, max(2, 2 * math.ceil(required_pages / 2)))


def wave_grain_selection(
    max_cache_seqlens: tuple[int, ...],
    h_q: int,
    sm_count: int,
):
    capacity_pages = tuple(
        math.ceil(int(length) / PAGE_SIZE) for length in max_cache_seqlens
    )
    max_pages = max(capacity_pages, default=0)
    if max_pages < ADAPTIVE_MODEL_MIN_PAGES:
        selected = adaptive_fixed_pages(max_pages)
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


def adaptive_schedule(max_cache_seqlens, pages_per_split: int):
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


def build_adaptive_execution_meta(
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
    elif (
        max_pages in (4, 8, 16, 32, 64)
        and h_q == 128
        and len(capacity_pages) == 128
        and all(pages == max_pages for pages in capacity_pages)
    ):
        # The measured high-batch workload already fills two CTA waves without
        # splitting; finer grains add redundant query loads and merge traffic.
        fixed_pages = max_pages
        selection = (
            {
                "pages": fixed_pages,
                "policy": "high_batch_uniform_row_direct",
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
        sm_count = int(_get_num_sms(device))
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
        sm_count = int(_get_num_sms(device))
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
        sm_count = int(_get_num_sms(device))
        fixed_pages, selection = wave_grain_selection(
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
    ) = adaptive_schedule(max_cache_seqlens, fixed_pages)

    meta = FlashMLAFp8SplitKSchedMeta()
    meta.have_initialized = True
    meta.num_splits = torch.empty((len(prefix),), dtype=torch.int32, device=device)
    meta.split_batch = torch.empty(
        (len(split_batch),), dtype=torch.int32, device=device
    )
    meta.split_page_begin = torch.empty_like(meta.split_batch)
    meta.split_page_end = torch.empty_like(meta.split_batch)
    meta.split_num_pages = torch.empty_like(meta.split_batch)
    initialize_capacity_plan[(len(capacity_pages),)](
        meta.num_splits,
        meta.split_batch,
        meta.split_page_begin,
        meta.split_page_end,
        meta.split_num_pages,
        capacity_pages,
        prefix,
        BATCH=len(capacity_pages),
        GRAIN=fixed_pages,
        BLOCK=triton.next_power_of_2(max(counts, default=1)),
        num_warps=4,
    )
    meta.max_splits = max(counts, default=1)
    meta.total_split_capacity = len(split_batch)
    meta.max_pages_per_split = max(split_num_pages, default=0)
    meta.lifetime_safe_one_pair = meta.max_pages_per_split <= 2
    meta.num_splits_data_ptr = int(meta.num_splits.data_ptr())
    meta.num_splits_version = tensor_version(meta.num_splits)
    meta.adaptive_fixed_pages = fixed_pages
    meta.adaptive_fixed_pairs = math.ceil(fixed_pages / 2)
    meta.adaptive_selection = selection
    meta.capacity_splits = counts
    meta.padded_pages = padded_pages
    return meta


@triton.jit
def initialize_capacity_plan(
    Prefix,
    Batch,
    Begin,
    End,
    Pages,
    CapacityPages,
    PrefixValues,
    BATCH: tl.constexpr,
    GRAIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    capacity = 0
    start = 0
    stop = 0
    for row in tl.static_range(BATCH):
        capacity = tl.where(batch == row, CapacityPages[row], capacity)
        start = tl.where(batch == row, PrefixValues[row], start)
        stop = tl.where(batch == row, PrefixValues[row + 1], stop)
    count = stop - start
    local = tl.arange(0, BLOCK)
    page_begin = local * GRAIN
    num_pages = tl.maximum(0, tl.minimum(GRAIN, capacity - page_begin))
    if GRAIN == 16:
        short_before = local * 4 // count
        short_through = (local + 1) * 4 // count
        balanced = (capacity == 520) & (count == 33)
        page_begin = tl.where(balanced, local * 16 - 2 * short_before, page_begin)
        num_pages = tl.where(
            balanced, tl.where(short_through != short_before, 14, 16), num_pages
        )
    elif GRAIN == 34:
        balanced = (capacity == 520) & (count == 16)
        begin = capacity * local // count
        end = capacity * (local + 1) // count
        page_begin = tl.where(balanced, begin, page_begin)
        num_pages = tl.where(balanced, end - begin, num_pages)
    else:
        pass
    mask = local < count
    tl.store(Prefix + batch, start)
    if batch == BATCH - 1:
        tl.store(Prefix + BATCH, stop)
    tl.store(Batch + start + local, batch, mask)
    tl.store(Begin + start + local, page_begin, mask)
    tl.store(End + start + local, page_begin + num_pages, mask)
    tl.store(Pages + start + local, num_pages, mask)


@triton.jit
def initialize_length_plan(
    Lengths,
    Prefix,
    Batch,
    Begin,
    End,
    Pages,
    BATCH: tl.constexpr,
    GRAIN: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_B)
    lengths = tl.load(Lengths + rows, rows < BATCH, 0)
    pages = tl.cdiv(tl.maximum(lengths, 0), 64)
    counts = tl.where(
        rows < BATCH, tl.minimum(tl.maximum(tl.cdiv(pages, GRAIN), 1), LIMIT), 0
    )
    start = tl.sum(tl.where(rows < batch, counts, 0), 0)
    count = tl.sum(tl.where(rows == batch, counts, 0), 0)
    capacity = tl.sum(tl.where(rows == batch, pages, 0), 0)
    local = tl.arange(0, BLOCK_S)
    begin = capacity * local // count
    end = capacity * (local + 1) // count
    tl.store(Prefix + batch, start)
    if batch == BATCH - 1:
        tl.store(Prefix + BATCH, start + count)
    mask = local < count
    tl.store(Batch + start + local, batch, mask)
    tl.store(Begin + start + local, begin, mask)
    tl.store(End + start + local, end, mask)
    tl.store(Pages + start + local, end - begin, mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("flash_mla_fp8_copy_table"),
    key=["BATCH", "COLS", "PADDED"],
    use_cuda_graph=True,
)
@triton.jit
def copy_block_table(
    Source,
    Destination,
    BATCH: tl.constexpr,
    COLS: tl.constexpr,
    PADDED: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows, cols = offsets // PADDED, offsets % PADDED
    values = tl.load(
        Source + rows * ROW_STRIDE + cols * COL_STRIDE,
        (rows < BATCH) & (cols < COLS),
        0,
    )
    tl.store(Destination + offsets, values, rows < BATCH)


def pad_block_table(block_table: torch.Tensor, padded_pages: int):
    if int(block_table.shape[1]) >= padded_pages:
        return block_table
    padded = torch.empty(
        (block_table.shape[0], padded_pages),
        dtype=block_table.dtype,
        device=block_table.device,
    )
    copy_block_table[lambda meta: (triton.cdiv(padded.numel(), meta["BLOCK"]),)](
        block_table,
        padded,
        block_table.shape[0],
        block_table.shape[1],
        padded_pages,
        *block_table.stride(),
    )
    return padded


def get_mla_fp8_metadata(
    cache_seqlens: Optional[torch.Tensor] = None,
    num_q_heads_per_k_head: Optional[int] = None,
    num_k_heads: int = 1,
    *,
    pages_per_split: int = DEFAULT_PAGES_PER_SPLIT,
    max_splits: Optional[int] = None,
) -> Tuple[FlashMLAFp8SplitKSchedMeta, Optional[torch.Tensor]]:
    """Create GPU split metadata with host capacities instead of reading GPU counts."""
    meta = FlashMLAFp8SplitKSchedMeta()
    if cache_seqlens is None:
        return meta, None
    if cache_seqlens.ndim != 1 or cache_seqlens.dtype != torch.int32:
        raise ValueError("cache_seqlens must be a 1-D int32 tensor")
    if pages_per_split <= 0 or (max_splits is not None and max_splits <= 0):
        raise ValueError("split capacities must be positive")
    batch = cache_seqlens.numel()
    limit = triton.cdiv(triton.cdiv(MAX_SEQUENCE_LENGTH, PAGE_SIZE), pages_per_split)
    limit = min(limit, max_splits) if max_splits is not None else limit
    meta.num_splits = torch.empty(
        (batch + 1,), dtype=torch.int32, device=cache_seqlens.device
    )
    meta.total_split_capacity = batch * limit
    meta.split_batch = torch.empty(
        (batch * limit,), dtype=torch.int32, device=cache_seqlens.device
    )
    meta.split_page_begin = torch.empty_like(meta.split_batch)
    meta.split_page_end = torch.empty_like(meta.split_batch)
    meta.split_num_pages = torch.empty_like(meta.split_batch)
    if batch == 0:
        raise NotImplementedError("metadata requires a nonempty batch")
    if cache_seqlens.device.type != "cuda" or cache_seqlens.stride(0) != 1:
        raise NotImplementedError("metadata requires contiguous NVIDIA CUDA lengths")
    if batch:
        initialize_length_plan[(batch,)](
            cache_seqlens,
            meta.num_splits,
            meta.split_batch,
            meta.split_page_begin,
            meta.split_page_end,
            meta.split_num_pages,
            batch,
            pages_per_split,
            limit,
            triton.next_power_of_2(batch),
            triton.next_power_of_2(limit),
            num_warps=4,
        )
    meta.max_splits = limit
    meta.max_pages_per_split = triton.cdiv(MAX_SEQUENCE_LENGTH // PAGE_SIZE, limit)
    meta.have_initialized = True
    return meta, meta.num_splits


@triton.jit
def write_cache_lengths(Lengths, Values, BATCH: tl.constexpr):
    row = tl.program_id(0)
    value = 0
    for index in tl.static_range(BATCH):
        value = tl.where(row == index, Values[index], value)
    tl.store(Lengths + row, value)


def validate_dense_inputs(
    q_nope,
    q_rope,
    k_cache_lora,
    k_cache_rope,
    q_scale,
    k_scale,
    block_table,
    cache_seqlens,
    head_dim_v,
):
    if q_nope.device.type != "cuda":
        raise NotImplementedError("requires NVIDIA Hopper tensors")
    if q_nope.ndim != 4 or q_nope.shape[1] != 1 or q_nope.shape[-1] != D_CKV:
        raise NotImplementedError("requires Q [batch, 1, heads, 512]")
    batch, _, heads, _ = q_nope.shape
    if batch <= 0 or heads <= 0 or heads % TLE_FP8_BH:
        raise NotImplementedError(
            "requires positive batch and a head count divisible by 64"
        )
    if (
        k_cache_lora.ndim not in (3, 4)
        or k_cache_lora.shape[1] != PAGE_SIZE
        or k_cache_lora.shape[-1] != D_CKV
    ):
        raise NotImplementedError(
            "requires 64-token KV pages with 512 content dimensions"
        )
    if k_cache_lora.ndim == 4 and k_cache_lora.shape[2] != 1:
        raise NotImplementedError("requires one KV head")
    if head_dim_v != D_CKV:
        raise NotImplementedError("requires head_dim_v=512")
    expected = (
        (batch, 1, heads, D_ROPE),
        (*k_cache_lora.shape[:-1], D_ROPE),
        (batch, 1, heads, 1),
        (*k_cache_lora.shape[:-1], 1),
        (batch,),
    )
    for tensor, shape in zip(
        (q_rope, k_cache_rope, q_scale, k_scale, cache_seqlens), expected
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"expected tensor shape {shape}, got {tuple(tensor.shape)}"
            )
    if q_nope.dtype != torch.float8_e4m3fn or k_cache_lora.dtype != torch.float8_e4m3fn:
        raise TypeError("NoPE tensors must be float8_e4m3fn")
    if q_rope.dtype != torch.bfloat16 or k_cache_rope.dtype != torch.bfloat16:
        raise TypeError("RoPE tensors must be bfloat16")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("scales must be float32")
    if cache_seqlens.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise TypeError("lengths and block table must be int32")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] <= 0
    ):
        raise ValueError(
            "block table must have positive page capacity for every request"
        )
    descriptor_inputs = (q_nope, q_rope, q_scale, k_cache_lora, k_cache_rope, k_scale)
    for tensor in (*descriptor_inputs, block_table, cache_seqlens):
        if tensor.device != q_nope.device:
            raise ValueError("all inputs must share the query device")
    for tensor in descriptor_inputs:
        if not tensor.is_contiguous():
            raise NotImplementedError("dense TMA inputs require contiguous storage")
        if tensor.data_ptr() % 16:
            raise NotImplementedError("dense TMA inputs require 16-byte alignment")
    if cache_seqlens.stride(0) != 1:
        raise NotImplementedError("length storage must be contiguous")
    if min(q_nope.shape[0], k_cache_lora.shape[0]) <= 0:
        raise ValueError("query and KV storage must be nonempty")


def prepare_compiled_runner(
    jit_function,
    args,
    grid,
    *,
    num_warps: int = 4,
    launch_pdl: bool = False,
):
    """Bind one already-specialized Triton kernel to its direct launcher."""
    from triton.runtime._async_compile import FutureKernel
    from triton.runtime.driver import driver

    if jit_function.pre_run_hooks:
        raise RuntimeError(
            "prepared compiled launch does not support JIT pre-run hooks"
        )
    device = driver.active.get_current_device()
    _, _, _, _, binder = jit_function.device_caches[device]
    bound_args, _, _ = binder(
        *args,
        num_warps=num_warps,
        num_stages=1,
        launch_pdl=launch_pdl,
    )
    kernel = jit_function.warmup(
        *args,
        grid=grid,
        num_warps=num_warps,
        num_stages=1,
        launch_pdl=launch_pdl,
    )
    if isinstance(kernel, FutureKernel):
        kernel = kernel.result()
    grid3 = tuple(grid) + (1,) * (3 - len(grid))
    return kernel[grid3], tuple(bound_args.values())


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
    pages_per_split: int = DEFAULT_PAGES_PER_SPLIT,
    max_splits: Optional[int] = None,
    *,
    initial_cache_seqlens: Sequence[int],
    max_cache_seqlens: Sequence[int],
) -> tuple[FlashMLAFp8PreparedHandle, tuple[torch.Tensor, torch.Tensor]]:
    """Build an adaptive split plan and a CUDA Graph-compatible replay handle."""
    if not HAS_TLE:
        raise NotImplementedError("FP8 MLA requires Hopper and FlagTree GPU extensions")
    validate_dense_inputs(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
    )
    batch_size = int(q_nope.shape[0])
    if batch_size <= 0 or int(cache_seqlens.numel()) != batch_size:
        raise ValueError("batch dimensions do not match")
    if int(q_nope.shape[1]) != 1:
        raise ValueError("SQ=1 decode only")
    h_q = int(q_nope.shape[2])
    if h_q <= 0 or h_q % TLE_FP8_BH:
        raise ValueError("HQ must be a positive multiple of 64")
    initial = host_certificate_lengths(
        initial_cache_seqlens,
        "initial_cache_seqlens",
        batch_size=batch_size,
    )
    capacity = host_certificate_lengths(
        max_cache_seqlens,
        "max_cache_seqlens",
        batch_size=batch_size,
    )
    if any(current > maximum for current, maximum in zip(initial, capacity)):
        raise ValueError("initial_cache_seqlens cannot exceed max_cache_seqlens")
    required_pages = max(math.ceil(length / PAGE_SIZE) for length in capacity)
    if block_table.ndim != 2 or int(block_table.shape[0]) != batch_size:
        raise ValueError("block_table must be a two-dimensional batch table")
    if int(block_table.shape[1]) < required_pages:
        raise ValueError(
            "block_table does not cover the prepared max_cache_seqlens capacity"
        )
    meta = build_adaptive_execution_meta(
        capacity,
        h_q,
        q_nope.device,
        pages_per_split,
    )
    if max_splits is not None and int(max_splits) < max(meta.capacity_splits):
        raise ValueError("max_splits cannot be below an adaptive row capacity")
    execution_block_table = pad_block_table(block_table, meta.padded_pages)
    total_splits = int(meta.split_batch.numel())
    scale = float(softmax_scale if softmax_scale is not None else D_QK**-0.5)

    direct_single_output = bool(meta.capacity_splits) and all(
        count == 1 for count in meta.capacity_splits
    )
    if direct_single_output:
        partial_out = torch.empty((0,), dtype=torch.float32, device=q_nope.device)
    else:
        partial_out = torch.empty(
            (total_splits, h_q, D_CKV),
            dtype=torch.float32,
            device=q_nope.device,
        )
    partial_lse2 = torch.empty(
        (total_splits, h_q), dtype=torch.float32, device=q_nope.device
    )
    out = torch.empty(
        (batch_size, 1, h_q, head_dim_v), dtype=q_rope.dtype, device=q_nope.device
    )
    lse = torch.empty((batch_size, h_q, 1), dtype=torch.float32, device=q_nope.device)

    handle = FlashMLAFp8PreparedHandle(
        q_nope,
        q_rope,
        q_scale,
        k_cache_lora,
        k_cache_rope,
        k_scale,
        block_table,
        execution_block_table,
        cache_seqlens,
        meta,
        partial_out,
        partial_lse2,
        out,
        lse,
        h_q,
        scale,
        head_dim_v,
        initial,
        capacity,
    )
    first_result = handle()
    return handle, first_result


class FlashMLAFp8PreparedHandle(PreparedHandleBase):
    """Callable prepared decode handle with stable descriptors and workspace."""

    __slots__ = (
        "q_nope",
        "q_rope",
        "q_scale",
        "k_cache_lora",
        "k_cache_rope",
        "k_scale",
        "block_table",
        "execution_block_table",
        "cache_seqlens",
        "meta",
        "partial_out",
        "partial_lse2",
        "out",
        "lse",
        "h_q",
        "scale",
        "head_dim_v",
        "initial_cache_seqlens",
        "max_cache_seqlens",
        "cache_seqlens_host",
        "cache_version",
        "num_pages",
        "logical_active_splits",
        "direct_single_output",
        "in_use",
        "q_desc",
        "qr_desc",
        "qs_desc",
        "out_desc",
        "k_desc",
        "kr_desc",
        "ks_desc",
        "launch_pack_key",
        "partial_compiled_runner",
        "partial_compiled_args",
        "aux_compiled_runner",
        "aux_compiled_args",
        "launch_pack_reuses",
        "cuda_graph_key",
        "cuda_graph",
        "cuda_graph_capture_stream",
        "cuda_graph_eligible",
        "has_length_certificate",
    )

    def __init__(
        self,
        q_nope,
        q_rope,
        q_scale,
        k_cache_lora,
        k_cache_rope,
        k_scale,
        block_table,
        execution_block_table,
        cache_seqlens,
        meta,
        partial_out,
        partial_lse2,
        out,
        lse,
        h_q,
        scale,
        head_dim_v,
        initial_cache_seqlens,
        max_cache_seqlens,
        has_length_certificate=True,
    ) -> None:
        self.has_length_certificate = has_length_certificate
        self.q_nope = q_nope
        self.q_rope = q_rope
        self.q_scale = q_scale
        self.k_cache_lora = k_cache_lora
        self.k_cache_rope = k_cache_rope
        self.k_scale = k_scale
        self.block_table = block_table
        self.execution_block_table = execution_block_table
        self.cache_seqlens = cache_seqlens
        self.meta = meta
        self.partial_out = partial_out
        self.partial_lse2 = partial_lse2
        self.out = out
        self.lse = lse
        self.h_q = h_q
        self.scale = scale
        self.head_dim_v = head_dim_v
        self.direct_single_output = bool(meta.capacity_splits) and all(
            count == 1 for count in meta.capacity_splits
        )
        # Prepared replay keeps the bound Q/K tensors and their storage
        # addresses stable.  Match CUDA's launch-parameter lifetime by
        # materializing the six immutable TMA descriptors once instead of
        # rebuilding them on every decode step.
        _ensure_triton_descriptor_allocator(q_nope.device)
        self.q_desc = _get_tensor_descriptor_cls().from_tensor(
            q_nope.view(-1, D_CKV), block_shape=[TLE_FP8_BH, D_CKV]
        )
        self.qr_desc = _get_tensor_descriptor_cls().from_tensor(
            q_rope.view(-1, D_ROPE), block_shape=[TLE_FP8_BH, D_ROPE]
        )
        self.qs_desc = _get_tensor_descriptor_cls().from_tensor(
            q_scale.view(-1, h_q), block_shape=[1, TLE_FP8_BH]
        )
        # Rebound in _partial_launch_args when caller-provided output storage
        # changes; non-direct routes never consume this placeholder.
        self.out_desc = self.q_desc
        self.k_desc = _get_tensor_descriptor_cls().from_tensor(
            k_cache_lora.view(-1, D_CKV),
            block_shape=[TLE_FP8_BK, K_CONTENT_TILE_HOST],
        )
        self.kr_desc = _get_tensor_descriptor_cls().from_tensor(
            k_cache_rope.view(-1, D_ROPE), block_shape=[TLE_FP8_BK, D_ROPE]
        )
        self.ks_desc = _get_tensor_descriptor_cls().from_tensor(
            k_scale.view(-1, TLE_FP8_BK), block_shape=[1, TLE_FP8_BK]
        )
        self.initial_cache_seqlens = tuple(initial_cache_seqlens)
        self.max_cache_seqlens = tuple(max_cache_seqlens)
        self.cache_seqlens_host = tuple(initial_cache_seqlens)
        self.cache_version = tensor_version(cache_seqlens)
        self.num_pages, self.logical_active_splits = length_page_state(
            self.cache_seqlens_host,
            int(meta.adaptive_fixed_pages),
        )
        if self.direct_single_output:
            if int(meta.split_batch.numel()) != len(self.max_cache_seqlens):
                raise AssertionError(
                    "direct-single schedule must contain one split per batch row"
                )
            if self.partial_out.numel() != 0:
                raise AssertionError(
                    "direct-single schedule must not allocate an FP32 output workspace"
                )
        self.in_use = False
        self.launch_pack_key = None
        self.partial_compiled_runner = None
        self.partial_compiled_args = None
        self.aux_compiled_runner = None
        self.aux_compiled_args = None
        self.launch_pack_reuses = 0
        self.cuda_graph_key = None
        self.cuda_graph = None
        self.cuda_graph_capture_stream = None
        self.cuda_graph_eligible = False

    def programmatic_dependency_capacity(self):
        batch_size = int(self.out.shape[0])
        partial_ctas = int(self.meta.split_batch.numel()) * (self.h_q // TLE_FP8_BH)
        consumer_ctas = batch_size * math.ceil(
            self.h_q / CUDA_COARSE_COMBINE_BLOCK_ROWS
        )
        sm_count = int(_get_num_sms(self.out.device))
        return partial_ctas, consumer_ctas, sm_count

    def use_programmatic_dependent_launch(self) -> bool:
        partial_ctas, consumer_ctas, sm_count = self.programmatic_dependency_capacity()
        # Keep one full consumer grid of scheduling headroom beyond the
        # producer and consumer fit.
        return (
            not self.direct_single_output
            and int(self.out.shape[0]) >= CUDA_COARSE_COMBINE_MIN_BATCH
            and partial_ctas + 2 * consumer_ctas <= sm_count + 1
        )

    def use_full_tail_specialization(self) -> bool:
        # Every scheduled capacity page must be real and complete.  A handle
        # whose current lengths have not reached prepared capacity keeps the
        # masked kernel even when the current token count is 64-aligned.
        return (
            self.has_length_certificate
            and self.cache_seqlens_host == self.max_cache_seqlens
            and all(length % PAGE_SIZE == 0 for length in self.cache_seqlens_host)
            and tuple(self.logical_active_splits) == tuple(self.meta.capacity_splits)
        )

    def use_merged_state_v_completion(self) -> bool:
        return (
            self.h_q == 64
            # Admit B8 full-tail shapes to the split-structure-agnostic
            # merged-completion schedule.
            and int(self.out.shape[0]) >= 8
            # L8192 also satisfies the full-tail certificate required below.
            and all(length in (33280, 8192) for length in self.max_cache_seqlens)
            and self.use_full_tail_specialization()
        )

    def use_pretranspose_v1(self) -> bool:
        return (
            self.h_q == 128
            and int(self.out.shape[0]) in (16, 32)
            and all(length == 640 for length in self.max_cache_seqlens)
        )

    def use_fixed_ten_page_v1(self) -> bool:
        return (
            self.h_q == 128
            and int(self.out.shape[0]) in (32, 64, 128)
            and all(length == 640 for length in self.max_cache_seqlens)
            and self.use_full_tail_specialization()
        )

    def use_fixed_two_page_v1(self) -> bool:
        # The direct one-pair family has one CTA per batch row.  With every
        # certified length in (64, 128], each CTA has exactly two real pages,
        # although page one may be partial.  Constant-fold only num_pages;
        # retain the masked math and worker schedule unchanged.
        return (
            self.has_length_certificate
            and self.direct_single_output
            and int(self.meta.max_pages_per_split) == 2
            and self.cache_seqlens_host == self.max_cache_seqlens
            and all(
                PAGE_SIZE < length <= 2 * PAGE_SIZE
                for length in self.cache_seqlens_host
            )
        )

    def use_direct_lse_v2(self) -> bool:
        # A direct-single route has exactly one partial CTA for each output
        # row/head block, so its WG0 LSE store has no cross-split reduction.
        # Every direct-output route can write natural-log LSE here and omit
        # the separate conversion kernel.
        return self.direct_single_output

    def partial_launch_args(self, target_out, target_lse):
        h_q = self.h_q
        rh = h_q // TLE_FP8_BH
        if self.direct_single_output:
            self.out_desc = _get_tensor_descriptor_cls().from_tensor(
                target_out.view(-1, D_CKV), block_shape=[TLE_FP8_BH, D_ROPE]
            )
        return (
            self.q_nope,
            self.q_rope,
            self.q_scale,
            self.k_cache_lora,
            self.k_cache_rope,
            self.k_scale,
            self.execution_block_table,
            self.cache_seqlens,
            self.meta.split_batch,
            self.meta.split_page_begin,
            self.meta.split_num_pages,
            target_out,
            target_lse,
            self.q_desc,
            self.qr_desc,
            self.qs_desc,
            self.out_desc,
            self.k_desc,
            self.kr_desc,
            self.ks_desc,
            self.q_nope.stride(0),
            self.q_nope.stride(2),
            self.q_rope.stride(0),
            self.q_rope.stride(2),
            self.q_scale.stride(0),
            self.q_scale.stride(2),
            self.k_cache_lora.stride(0),
            self.k_cache_lora.stride(1),
            self.k_cache_rope.stride(0),
            self.k_cache_rope.stride(1),
            self.k_scale.stride(0),
            self.k_scale.stride(1),
            self.execution_block_table.stride(0),
            self.execution_block_table.stride(1),
            self.cache_seqlens.stride(0),
            self.meta.split_batch.stride(0),
            self.meta.split_page_begin.stride(0),
            self.meta.split_num_pages.stride(0),
            target_out.stride(0),
            target_out.stride(2 if self.direct_single_output else 1),
            target_lse.stride(0),
            target_lse.stride(1),
            self.scale,
            64 * 512,
            64 * 64 * 2,
            64 * 4,
            64 * K_CONTENT_TILE_HOST,
            64 * 64 * 2,
            64 * 4,
            D_CKV,
            D_ROPE,
            TLE_FP8_BK,
            TLE_FP8_BH,
            h_q,
            rh,
            PAGE_SIZE,
            TLE_FP8_DPH,
            int(self.meta.adaptive_fixed_pairs) >= 2,
            self.use_full_tail_specialization(),
            int(self.meta.max_pages_per_split) <= 2,
            self.use_merged_state_v_completion(),
            self.direct_single_output,
            (
                10
                if self.use_fixed_ten_page_v1()
                else (
                    2
                    if self.use_fixed_two_page_v1()
                    else (
                        int(self.meta.adaptive_fixed_pages)
                        if self.use_full_tail_specialization()
                        and all(
                            (length // PAGE_SIZE) % int(self.meta.adaptive_fixed_pages)
                            == 0
                            for length in self.max_cache_seqlens
                        )
                        else 0
                    )
                )
            ),
            self.use_direct_lse_v2(),
        )

    def aux_launch_spec(self, out, lse):
        if self.direct_single_output:
            total = int(lse.shape[0]) * self.h_q
            return (
                triton_fp8_single_split_lse_finalize_kernel,
                (
                    self.partial_lse2,
                    lse,
                    self.partial_lse2.stride(0),
                    self.partial_lse2.stride(1),
                    lse.stride(0),
                    lse.stride(1),
                    self.h_q,
                    total,
                    LSE_FINALIZE_BLOCK,
                ),
                (triton.cdiv(total, LSE_FINALIZE_BLOCK),),
            )

        common_args = (
            self.partial_out,
            self.partial_lse2,
            self.meta.num_splits,
            out,
            lse,
            self.partial_out.stride(0),
            self.partial_out.stride(1),
            self.partial_lse2.stride(0),
            self.partial_lse2.stride(1),
            self.meta.num_splits.stride(0),
            out.stride(0),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            self.h_q,
            D_CKV,
        )
        batch_size = int(out.shape[0])
        if batch_size >= CUDA_COARSE_COMBINE_MIN_BATCH:

            return (
                triton_fp8_coarse_combine_kernel,
                common_args
                + (
                    CUDA_COARSE_COMBINE_BLOCK_SPLITS,
                    CUDA_COARSE_COMBINE_BLOCK_ROWS,
                    self.use_programmatic_dependent_launch(),
                ),
                (
                    batch_size,
                    math.ceil(self.h_q / CUDA_COARSE_COMBINE_BLOCK_ROWS),
                ),
            )
        return (
            triton_fp8_splitk_combine_kernel,
            common_args + (COMBINE_BLOCK_SPLITS, COMBINE_BLOCK_D),
            (
                batch_size * self.h_q,
                math.ceil(D_CKV / COMBINE_BLOCK_D),
            ),
        )

    def ensure_compiled_launch_pack(self, out, lse) -> None:
        # Exact pointer identity preserves every alignment specialization that
        # the public caller-provided output contract previously admitted. Keep
        # only the most recent pack so workloads that rotate output buffers do
        # not grow an unbounded launcher cache.
        key = (
            int(out.data_ptr()),
            int(lse.data_ptr()),
            self.use_full_tail_specialization(),
            self.use_merged_state_v_completion(),
            self.use_pretranspose_v1(),
            self.use_fixed_ten_page_v1(),
            self.use_fixed_two_page_v1(),
        )
        if key == self.launch_pack_key:
            self.launch_pack_reuses += 1
            return

        target_out = out if self.direct_single_output else self.partial_out
        direct_lse = self.use_direct_lse_v2()
        target_lse = lse if direct_lse else self.partial_lse2
        use_pdl = self.use_programmatic_dependent_launch()
        partial_grid = (int(self.meta.split_batch.numel()) * (self.h_q // TLE_FP8_BH),)
        pretranspose_v1 = not use_pdl and self.use_pretranspose_v1()
        partial_runner, partial_args = prepare_compiled_runner(
            fp8_dense_mla_splitk_partial,
            self.partial_launch_args(target_out, target_lse)
            + (use_pdl, pretranspose_v1),
            partial_grid,
            launch_pdl=False,
        )
        if direct_lse:
            aux_runner, aux_bound_args = None, None
        else:
            aux_jit, aux_args, aux_grid = self.aux_launch_spec(out, lse)
            aux_runner, aux_bound_args = prepare_compiled_runner(
                aux_jit,
                aux_args,
                aux_grid,
                num_warps=(8 if aux_jit is triton_fp8_coarse_combine_kernel else 4),
                launch_pdl=use_pdl,
            )
        self.partial_compiled_runner = partial_runner
        self.partial_compiled_args = partial_args
        self.aux_compiled_runner = aux_runner
        self.aux_compiled_args = aux_bound_args
        self.launch_pack_key = key
        self.launch_pack_reuses = 0
        self.cuda_graph_key = None
        self.cuda_graph = None
        self.cuda_graph_capture_stream = None
        self.cuda_graph_eligible = not use_pdl

    def ensure_graph_replay(self):
        """Capture the stable two-kernel prepared replay after one pointer hit."""
        key = self.launch_pack_key
        if key is None or not self.cuda_graph_eligible or self.launch_pack_reuses < 1:
            return None
        if self.cuda_graph_key == key and self.cuda_graph is not None:
            return self.cuda_graph

        current_stream = torch.cuda.current_stream(self.out.device)
        capture_stream = torch.cuda.Stream(device=self.out.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            self.partial_compiled_runner(*self.partial_compiled_args)
            if self.aux_compiled_runner is not None:
                self.aux_compiled_runner(*self.aux_compiled_args)
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            self.partial_compiled_runner(*self.partial_compiled_args)
            if self.aux_compiled_runner is not None:
                self.aux_compiled_runner(*self.aux_compiled_args)
        current_stream.wait_stream(capture_stream)
        self.cuda_graph_key = key
        self.cuda_graph = graph
        self.cuda_graph_capture_stream = capture_stream
        return graph

    def claim(self) -> None:
        if self.in_use:
            raise RuntimeError("prepared handle is already in use")
        self.in_use = True

    def validate_length_bounds(self, values) -> None:
        for batch_index, (value, previous, capacity) in enumerate(
            zip(values, self.cache_seqlens_host, self.max_cache_seqlens)
        ):
            if value < previous:
                raise RuntimeError(f"cache_seqlens[{batch_index}] must be monotonic")
            if value > capacity:
                raise RuntimeError(
                    f"cache_seqlens[{batch_index}] {value} exceeds prepared "
                    f"capacity {capacity}"
                )

    def apply_length_certificate(
        self,
        cache_seqlens,
        *,
        require_version_change: bool,
    ) -> None:
        values = host_certificate_lengths(
            cache_seqlens,
            "cache_seqlens",
            batch_size=len(self.cache_seqlens_host),
        )
        self.validate_length_bounds(values)

        version = tensor_version(self.cache_seqlens)
        changed = values != self.cache_seqlens_host
        if changed and require_version_change and version == self.cache_version:
            raise RuntimeError(
                "cache_seqlens storage did not receive an observable in-place "
                "PyTorch update before the new host certificate"
            )
        if not changed and version != self.cache_version:
            raise RuntimeError(
                "cache_seqlens changed without a new host length certificate"
            )

        pages, logical_active_splits = length_page_state(
            values,
            int(self.meta.adaptive_fixed_pages),
        )
        for batch_index, (logical, capacity) in enumerate(
            zip(logical_active_splits, self.meta.capacity_splits)
        ):
            if logical > capacity:
                raise RuntimeError(
                    f"logical split count for batch {batch_index} exceeds capacity"
                )

        self.cache_seqlens_host = values
        self.num_pages = pages
        self.logical_active_splits = logical_active_splits
        self.cache_version = version

    def set_cache_seqlens_(self, cache_seqlens: Sequence[int]) -> None:
        """Own a monotonic in-place length update on the bound CUDA stream."""
        values = host_certificate_lengths(
            cache_seqlens,
            "cache_seqlens",
            batch_size=len(self.cache_seqlens_host),
        )
        self.validate_length_bounds(values)
        self.claim()
        try:
            if values != self.cache_seqlens_host:
                write_cache_lengths[(len(values),)](
                    self.cache_seqlens,
                    values,
                    BATCH=len(values),
                    num_warps=4,
                )
            self.apply_length_certificate(
                values,
                require_version_change=False,
            )
        finally:
            self.in_use = False

    def validate_output(self, tensor: torch.Tensor, *, lse: bool) -> None:
        template = self.lse if lse else self.out
        label = "lse" if lse else "out"
        if tuple(tensor.shape) != tuple(template.shape):
            raise RuntimeError(
                f"{label} shape must be {tuple(template.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != template.dtype or tensor.device != template.device:
            raise RuntimeError(f"{label} dtype/device mismatch")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{label} must be contiguous")

    def launch(
        self,
        *,
        cache_seqlens: Sequence[int] | None = None,
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Submit one prepared decode step using a host length certificate."""
        self.claim()
        try:
            if cache_seqlens is None:
                if tensor_version(self.cache_seqlens) != self.cache_version:
                    raise RuntimeError(
                        "cache_seqlens changed without a host length certificate"
                    )
            else:
                self.apply_length_certificate(
                    cache_seqlens,
                    require_version_change=True,
                )

            if (out is None) != (lse is None):
                raise RuntimeError(
                    "out and lse must either both be supplied or both omitted"
                )
            if out is None:
                out = torch.empty_like(self.out)
                lse = torch.empty_like(self.lse)
            else:
                self.validate_output(out, lse=False)
                self.validate_output(lse, lse=True)

            self.ensure_compiled_launch_pack(out, lse)
            graph = self.ensure_graph_replay()
            if graph is None:
                self.partial_compiled_runner(*self.partial_compiled_args)
                if self.aux_compiled_runner is not None:
                    self.aux_compiled_runner(*self.aux_compiled_args)
            else:
                graph.replay()
            return out, lse
        finally:
            self.in_use = False

    __call__ = launch


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
    pages_per_split: int = DEFAULT_PAGES_PER_SPLIT,
    max_splits: Optional[int] = None,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public one-shot op: metadata + prepare + run."""
    if not HAS_TLE:
        raise NotImplementedError("FP8 MLA requires Hopper and FlagTree GPU extensions")
    validate_dense_inputs(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
    )
    batch_size = int(q_nope.shape[0])
    h_q = int(q_nope.shape[2])
    if h_q <= 0 or h_q % TLE_FP8_BH:
        raise ValueError("HQ must be a positive multiple of 64")
    capacity = (
        min(MAX_SEQUENCE_LENGTH, block_table.shape[1] * PAGE_SIZE),
    ) * batch_size
    meta = build_adaptive_execution_meta(
        capacity,
        h_q,
        q_nope.device,
        pages_per_split,
    )
    if max_splits is not None and int(max_splits) < max(meta.capacity_splits):
        raise ValueError("max_splits cannot be below an adaptive row capacity")
    execution_block_table = pad_block_table(block_table, meta.padded_pages)
    if softmax_scale is None:
        softmax_scale = float(D_QK**-0.5)
    if out is None:
        out = torch.empty(
            (batch_size, 1, int(q_nope.shape[2]), head_dim_v),
            dtype=q_rope.dtype,
            device=q_nope.device,
        )
    if lse is None:
        lse = torch.empty(
            (batch_size, int(q_nope.shape[2]), 1),
            dtype=torch.float32,
            device=q_nope.device,
        )
    handle = FlashMLAFp8PreparedHandle(
        q_nope,
        q_rope,
        q_scale,
        k_cache_lora,
        k_cache_rope,
        k_scale,
        block_table,
        execution_block_table,
        cache_seqlens,
        meta,
        (
            torch.empty((0,), dtype=torch.float32, device=q_nope.device)
            if bool(meta.capacity_splits)
            and all(count == 1 for count in meta.capacity_splits)
            else torch.empty(
                (int(meta.split_batch.numel()), h_q, D_CKV),
                dtype=torch.float32,
                device=q_nope.device,
            )
        ),
        torch.empty(
            (int(meta.split_batch.numel()), h_q),
            dtype=torch.float32,
            device=q_nope.device,
        ),
        out,
        lse,
        h_q,
        float(softmax_scale),
        head_dim_v,
        capacity,
        capacity,
        has_length_certificate=False,
    )
    return handle(out=out, lse=lse)
