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
from typing import Optional

import torch
import triton

from flaggems_vllm.ops.flash_mla_fp8.common import (
    D_CKV,
    D_ROPE,
    DEFAULT_PAGES_PER_SPLIT,
    HAS_TLE,
    K_CONTENT_TILE_HOST,
    MAX_SEQUENCE_LENGTH,
    PAGE_SIZE,
    TLE_FP8_BH,
)
from flaggems_vllm.ops.flash_mla_fp8.dense_kernels import (
    _fp8_dense_mla_splitk_partial,
    _fp8_dense_mla_splitk_partial_pdl,
    _fp8_dense_mla_splitk_partial_pretranspose,
    _triton_fp8_cuda_coarse_combine_kernel,
    _triton_fp8_cuda_coarse_combine_pdl_kernel,
    _triton_fp8_single_split_lse_finalize_kernel,
    _triton_fp8_splitk_combine_kernel,
)
from flaggems_vllm.ops.flash_mla_fp8.metadata import (
    _build_adaptive_execution_meta,
    _host_certificate_lengths,
    _host_lengths,
    _length_page_state,
    _pad_block_table,
    _tensor_version,
    get_mla_fp8_metadata,
)

D_QK = 576  # Q/K head dim (content 512 + rope 64)

TLE_FP8_BK = 64  # KV tokens per iteration (= PAGE_SIZE)

TLE_FP8_DPH = 256  # output 512 dim split into left/right halves of 256

COMBINE_BLOCK_SPLITS = 8

COMBINE_BLOCK_D = 128

CUDA_COARSE_COMBINE_BLOCK_SPLITS = 32

CUDA_COARSE_COMBINE_BLOCK_ROWS = 8

CUDA_COARSE_COMBINE_MIN_BATCH = 4

LSE_FINALIZE_BLOCK = 256


def _set_triton_descriptor_allocator(device: torch.device) -> None:
    """Install the allocator required by FlagTree TLE shared descriptors."""
    assert triton is not None

    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=device)

    try:
        triton.set_allocator(alloc_fn)
    except AttributeError:
        pass


def _make_tma_descriptor(tensor: torch.Tensor, block_shape):
    from triton.tools.tensor_descriptor import TensorDescriptor

    return TensorDescriptor.from_tensor(tensor, block_shape=block_shape)


def _prepare_compiled_runner(
    jit_function,
    args,
    grid,
    *,
    num_warps: int = 4,
    launch_pdl: bool = False,
):
    """Bind one already-specialized Triton kernel to its direct launcher."""
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
    if hasattr(kernel, "result"):
        kernel = kernel.result()
    grid3 = tuple(grid) + (1,) * (3 - len(grid))
    return kernel[grid3], tuple(bound_args.values())


def _launch_partial(
    q_nope,
    q_rope,
    q_scale,
    k_cache_lora,
    k_cache_rope,
    k_scale,
    block_table,
    cache_seqlens,
    split_batch,
    split_page_begin,
    split_num_pages,
    target_out,
    partial_lse2,
    q_desc,
    qr_desc,
    qs_desc,
    k_desc,
    kr_desc,
    ks_desc,
    h_q: int,
    softmax_scale: float,
    direct_output: bool = False,
):
    if not HAS_TLE:
        raise RuntimeError("Split-K FP8 MLA launcher called without TLE support")

    rh = h_q // TLE_FP8_BH
    partial_grid = (int(split_batch.numel()) * rh,)

    _fp8_dense_mla_splitk_partial[partial_grid](
        q_nope,
        q_rope,
        q_scale,
        k_cache_lora,
        k_cache_rope,
        k_scale,
        block_table,
        cache_seqlens,
        split_batch,
        split_page_begin,
        split_num_pages,
        target_out,
        partial_lse2,
        q_desc,
        qr_desc,
        qs_desc,
        k_desc,
        kr_desc,
        ks_desc,
        q_nope.stride(0),
        q_nope.stride(2),
        q_rope.stride(0),
        q_rope.stride(2),
        q_scale.stride(0),
        q_scale.stride(2),
        k_cache_lora.stride(0),
        k_cache_lora.stride(1),
        k_cache_rope.stride(0),
        k_cache_rope.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        cache_seqlens.stride(0),
        split_batch.stride(0),
        split_page_begin.stride(0),
        split_num_pages.stride(0),
        target_out.stride(0),
        target_out.stride(2 if direct_output else 1),
        partial_lse2.stride(0),
        partial_lse2.stride(1),
        softmax_scale,
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
        True,
        False,
        num_warps=4,
        num_stages=1,
    )


def _launch_combine(
    partial_out,
    partial_lse2,
    num_splits,
    out,
    lse,
    h_q: int,
):
    batch_size = int(out.shape[0])
    common_args = (
        partial_out,
        partial_lse2,
        num_splits,
        out,
        lse,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_lse2.stride(0),
        partial_lse2.stride(1),
        num_splits.stride(0),
        out.stride(0),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        h_q,
        D_CKV,
    )
    if batch_size >= CUDA_COARSE_COMBINE_MIN_BATCH:
        combine_grid = (
            batch_size,
            math.ceil(h_q / CUDA_COARSE_COMBINE_BLOCK_ROWS),
        )
        _triton_fp8_cuda_coarse_combine_kernel[combine_grid](
            *common_args,
            CUDA_COARSE_COMBINE_BLOCK_SPLITS,
            CUDA_COARSE_COMBINE_BLOCK_ROWS,
            num_warps=8,
            num_stages=1,
        )
        return

    # Use the fine-grained combine path for B=1 and B=2.
    combine_grid = (
        batch_size * h_q,
        math.ceil(D_CKV / COMBINE_BLOCK_D),
    )
    _triton_fp8_splitk_combine_kernel[combine_grid](
        *common_args,
        COMBINE_BLOCK_SPLITS,
        COMBINE_BLOCK_D,
        num_warps=4,
        num_stages=1,
    )


def _launch_single_split_lse_finalize(
    partial_lse2,
    lse,
    h_q: int,
):
    total = int(lse.shape[0]) * h_q
    grid = (triton.cdiv(total, LSE_FINALIZE_BLOCK),)
    _triton_fp8_single_split_lse_finalize_kernel[grid](
        partial_lse2,
        lse,
        partial_lse2.stride(0),
        partial_lse2.stride(1),
        lse.stride(0),
        lse.stride(1),
        h_q,
        total,
        LSE_FINALIZE_BLOCK,
        num_warps=4,
        num_stages=1,
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
    tile_scheduler_metadata=None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    pages_per_split: int = DEFAULT_PAGES_PER_SPLIT,
    max_splits: Optional[int] = None,
    *,
    initial_cache_seqlens,
    max_cache_seqlens,
):
    """Build an adaptive split plan and a CUDA Graph-compatible replay handle."""
    if not HAS_TLE:
        raise NotImplementedError("FP8 MLA requires Hopper and FlagTree GPU extensions")
    batch_size = int(q_nope.shape[0])
    if batch_size <= 0 or int(cache_seqlens.numel()) != batch_size:
        raise ValueError("batch dimensions do not match")
    if int(q_nope.shape[1]) != 1:
        raise ValueError("SQ=1 decode only")
    h_q = int(q_nope.shape[2])
    if h_q <= 0 or h_q % TLE_FP8_BH:
        raise ValueError("HQ must be a positive multiple of 64")
    initial = _host_certificate_lengths(
        initial_cache_seqlens,
        "initial_cache_seqlens",
        batch_size=batch_size,
    )
    capacity = _host_certificate_lengths(
        max_cache_seqlens,
        "max_cache_seqlens",
        batch_size=batch_size,
    )
    if any(current > maximum for current, maximum in zip(initial, capacity)):
        raise ValueError("initial_cache_seqlens cannot exceed max_cache_seqlens")
    current = _host_lengths(cache_seqlens, "cache_seqlens", batch_size=batch_size)
    if current != initial:
        raise ValueError(
            "cache_seqlens storage must match initial_cache_seqlens at prepare"
        )
    required_pages = max(math.ceil(length / PAGE_SIZE) for length in capacity)
    if block_table.ndim != 2 or int(block_table.shape[0]) != batch_size:
        raise ValueError("block_table must be a two-dimensional batch table")
    if int(block_table.shape[1]) < required_pages:
        raise ValueError(
            "block_table does not cover the prepared max_cache_seqlens capacity"
        )
    if num_splits is None:
        if tile_scheduler_metadata is not None:
            num_splits = tile_scheduler_metadata.num_splits
        else:
            _, num_splits = get_mla_fp8_metadata(
                cache_seqlens,
                int(q_nope.shape[2]),
                1,
                pages_per_split=pages_per_split,
                max_splits=max_splits,
            )
    if num_splits is None:
        raise ValueError("no split plan")

    # Preserve caller-visible capacity metadata while selecting a compact
    # 4..32-page execution grain.
    meta = _build_adaptive_execution_meta(
        capacity,
        h_q,
        q_nope.device,
        pages_per_split,
    )
    if max_splits is not None and int(max_splits) < max(meta.capacity_splits):
        raise ValueError("max_splits cannot be below an adaptive row capacity")
    execution_block_table = _pad_block_table(block_table, meta.padded_pages)
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

    handle = _FlashMLAFp8PreparedHandle(
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


class _FlashMLAFp8PreparedHandle:
    """Callable prepared decode handle with stable descriptors and workspace."""

    __slots__ = (
        "q_nope",
        "q_rope",
        "q_scale",
        "k_cache_lora",
        "k_cache_rope",
        "k_scale",
        "block_table",
        "_execution_block_table",
        "cache_seqlens",
        "_meta",
        "_partial_out",
        "_partial_lse2",
        "_out",
        "_lse",
        "_h_q",
        "_scale",
        "_head_dim_v",
        "_initial_cache_seqlens",
        "_max_cache_seqlens",
        "_cache_seqlens_host",
        "_cache_version",
        "_num_pages",
        "_logical_active_splits",
        "_direct_single_output",
        "_in_use",
        "_q_desc",
        "_qr_desc",
        "_qs_desc",
        "_out_desc",
        "_k_desc",
        "_kr_desc",
        "_ks_desc",
        "_launch_pack_key",
        "_partial_compiled_runner",
        "_partial_compiled_args",
        "_aux_compiled_runner",
        "_aux_compiled_args",
        "_launch_pack_reuses",
        "_cuda_graph_key",
        "_cuda_graph",
        "_cuda_graph_capture_stream",
        "_cuda_graph_eligible",
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
    ) -> None:
        self.q_nope = q_nope
        self.q_rope = q_rope
        self.q_scale = q_scale
        self.k_cache_lora = k_cache_lora
        self.k_cache_rope = k_cache_rope
        self.k_scale = k_scale
        self.block_table = block_table
        self._execution_block_table = execution_block_table
        self.cache_seqlens = cache_seqlens
        self._meta = meta
        self._partial_out = partial_out
        self._partial_lse2 = partial_lse2
        self._out = out
        self._lse = lse
        self._h_q = h_q
        self._scale = scale
        self._head_dim_v = head_dim_v
        self._direct_single_output = bool(meta.capacity_splits) and all(
            count == 1 for count in meta.capacity_splits
        )
        # Prepared replay keeps the bound Q/K tensors and their storage
        # addresses stable.  Match CUDA's launch-parameter lifetime by
        # materializing the six immutable TMA descriptors once instead of
        # rebuilding them on every decode step.
        _set_triton_descriptor_allocator(q_nope.device)
        self._q_desc = _make_tma_descriptor(
            q_nope.reshape(-1, D_CKV), [TLE_FP8_BH, D_CKV]
        )
        self._qr_desc = _make_tma_descriptor(
            q_rope.reshape(-1, D_ROPE), [TLE_FP8_BH, D_ROPE]
        )
        self._qs_desc = _make_tma_descriptor(q_scale.reshape(-1, h_q), [1, TLE_FP8_BH])
        # Rebound in _partial_launch_args when caller-provided output storage
        # changes; non-direct routes never consume this placeholder.
        self._out_desc = self._q_desc
        self._k_desc = _make_tma_descriptor(
            k_cache_lora.reshape(-1, D_CKV),
            [TLE_FP8_BK, K_CONTENT_TILE_HOST],
        )
        self._kr_desc = _make_tma_descriptor(
            k_cache_rope.reshape(-1, D_ROPE), [TLE_FP8_BK, D_ROPE]
        )
        self._ks_desc = _make_tma_descriptor(
            k_scale.reshape(-1, TLE_FP8_BK), [1, TLE_FP8_BK]
        )
        self._initial_cache_seqlens = tuple(initial_cache_seqlens)
        self._max_cache_seqlens = tuple(max_cache_seqlens)
        self._cache_seqlens_host = tuple(initial_cache_seqlens)
        self._cache_version = _tensor_version(cache_seqlens)
        self._num_pages, self._logical_active_splits = _length_page_state(
            self._cache_seqlens_host,
            int(meta.adaptive_fixed_pages),
        )
        if self._direct_single_output:
            if int(meta.split_batch.numel()) != len(self._max_cache_seqlens):
                raise AssertionError(
                    "direct-single schedule must contain one split per batch row"
                )
            if self._partial_out.numel() != 0:
                raise AssertionError(
                    "direct-single schedule must not allocate an FP32 output workspace"
                )
        self._in_use = False
        self._launch_pack_key = None
        self._partial_compiled_runner = None
        self._partial_compiled_args = None
        self._aux_compiled_runner = None
        self._aux_compiled_args = None
        self._launch_pack_reuses = 0
        self._cuda_graph_key = None
        self._cuda_graph = None
        self._cuda_graph_capture_stream = None
        self._cuda_graph_eligible = False

    def _programmatic_dependency_capacity(self):
        batch_size = int(self._out.shape[0])
        partial_ctas = int(self._meta.split_batch.numel()) * (self._h_q // TLE_FP8_BH)
        consumer_ctas = batch_size * math.ceil(
            self._h_q / CUDA_COARSE_COMBINE_BLOCK_ROWS
        )
        sm_count = int(
            torch.cuda.get_device_properties(self._out.device).multi_processor_count
        )
        return partial_ctas, consumer_ctas, sm_count

    def _use_programmatic_dependent_launch(self) -> bool:
        partial_ctas, consumer_ctas, sm_count = self._programmatic_dependency_capacity()
        # Keep one full consumer grid of scheduling headroom beyond the
        # producer and consumer fit.
        return (
            not self._direct_single_output
            and int(self._out.shape[0]) >= CUDA_COARSE_COMBINE_MIN_BATCH
            and partial_ctas + 2 * consumer_ctas <= sm_count + 1
        )

    def _use_full_tail_specialization(self) -> bool:
        # Every scheduled capacity page must be real and complete.  A handle
        # whose current lengths have not reached prepared capacity keeps the
        # masked kernel even when the current token count is 64-aligned.
        return (
            self._cache_seqlens_host == self._max_cache_seqlens
            and all(length % PAGE_SIZE == 0 for length in self._cache_seqlens_host)
            and tuple(self._logical_active_splits) == tuple(self._meta.capacity_splits)
        )

    def _use_merged_state_v_completion(self) -> bool:
        return (
            self._h_q == 64
            # Admit B8 full-tail shapes to the split-structure-agnostic
            # merged-completion schedule.
            and int(self._out.shape[0]) >= 8
            # L8192 also satisfies the full-tail certificate required below.
            and all(length in (33280, 8192) for length in self._max_cache_seqlens)
            and self._use_full_tail_specialization()
        )

    def _use_pretranspose_v1(self) -> bool:
        return (
            self._h_q == 128
            and int(self._out.shape[0]) in (16, 32)
            and all(length == 640 for length in self._max_cache_seqlens)
        )

    def _use_fixed_ten_page_v1(self) -> bool:
        return (
            self._h_q == 128
            and int(self._out.shape[0]) in (32, 64, 128)
            and all(length == 640 for length in self._max_cache_seqlens)
            and self._use_full_tail_specialization()
        )

    def _use_fixed_two_page_v1(self) -> bool:
        # The direct one-pair family has one CTA per batch row.  With every
        # certified length in (64, 128], each CTA has exactly two real pages,
        # although page one may be partial.  Constant-fold only num_pages;
        # retain the masked math and worker schedule unchanged.
        return (
            self._direct_single_output
            and int(self._meta.max_pages_per_split) == 2
            and self._cache_seqlens_host == self._max_cache_seqlens
            and all(
                PAGE_SIZE < length <= 2 * PAGE_SIZE
                for length in self._cache_seqlens_host
            )
        )

    def _use_direct_lse_v2(self) -> bool:
        # A direct-single route has exactly one partial CTA for each output
        # row/head block, so its WG0 LSE store has no cross-split reduction.
        # Every direct-output route can write natural-log LSE here and omit
        # the separate conversion kernel.
        return self._direct_single_output

    def _partial_launch_args(self, target_out, target_lse):
        h_q = self._h_q
        rh = h_q // TLE_FP8_BH
        if self._direct_single_output:
            self._out_desc = _make_tma_descriptor(
                target_out.reshape(-1, D_CKV), [TLE_FP8_BH, D_ROPE]
            )
        return (
            self.q_nope,
            self.q_rope,
            self.q_scale,
            self.k_cache_lora,
            self.k_cache_rope,
            self.k_scale,
            self._execution_block_table,
            self.cache_seqlens,
            self._meta.split_batch,
            self._meta.split_page_begin,
            self._meta.split_num_pages,
            target_out,
            target_lse,
            self._q_desc,
            self._qr_desc,
            self._qs_desc,
            self._out_desc,
            self._k_desc,
            self._kr_desc,
            self._ks_desc,
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
            self._execution_block_table.stride(0),
            self._execution_block_table.stride(1),
            self.cache_seqlens.stride(0),
            self._meta.split_batch.stride(0),
            self._meta.split_page_begin.stride(0),
            self._meta.split_num_pages.stride(0),
            target_out.stride(0),
            target_out.stride(2 if self._direct_single_output else 1),
            target_lse.stride(0),
            target_lse.stride(1),
            self._scale,
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
            int(self._meta.adaptive_fixed_pairs) >= 2,
            self._use_full_tail_specialization(),
            int(self._meta.max_pages_per_split) <= 2,
            self._use_merged_state_v_completion(),
            self._direct_single_output,
            (
                10
                if self._use_fixed_ten_page_v1()
                else (
                    2
                    if self._use_fixed_two_page_v1()
                    else (
                        int(self._meta.adaptive_fixed_pages)
                        if self._use_full_tail_specialization()
                        and all(
                            (length // PAGE_SIZE) % int(self._meta.adaptive_fixed_pages)
                            == 0
                            for length in self._max_cache_seqlens
                        )
                        else 0
                    )
                )
            ),
            self._use_direct_lse_v2(),
        )

    def _aux_launch_spec(self, out, lse):
        if self._direct_single_output:
            total = int(lse.shape[0]) * self._h_q
            return (
                _triton_fp8_single_split_lse_finalize_kernel,
                (
                    self._partial_lse2,
                    lse,
                    self._partial_lse2.stride(0),
                    self._partial_lse2.stride(1),
                    lse.stride(0),
                    lse.stride(1),
                    self._h_q,
                    total,
                    LSE_FINALIZE_BLOCK,
                ),
                (triton.cdiv(total, LSE_FINALIZE_BLOCK),),
            )

        common_args = (
            self._partial_out,
            self._partial_lse2,
            self._meta.num_splits,
            out,
            lse,
            self._partial_out.stride(0),
            self._partial_out.stride(1),
            self._partial_lse2.stride(0),
            self._partial_lse2.stride(1),
            self._meta.num_splits.stride(0),
            out.stride(0),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            self._h_q,
            D_CKV,
        )
        batch_size = int(out.shape[0])
        if batch_size >= CUDA_COARSE_COMBINE_MIN_BATCH:
            coarse_jit = (
                _triton_fp8_cuda_coarse_combine_pdl_kernel
                if self._use_programmatic_dependent_launch()
                else _triton_fp8_cuda_coarse_combine_kernel
            )
            return (
                coarse_jit,
                common_args
                + (
                    CUDA_COARSE_COMBINE_BLOCK_SPLITS,
                    CUDA_COARSE_COMBINE_BLOCK_ROWS,
                ),
                (
                    batch_size,
                    math.ceil(self._h_q / CUDA_COARSE_COMBINE_BLOCK_ROWS),
                ),
            )
        return (
            _triton_fp8_splitk_combine_kernel,
            common_args + (COMBINE_BLOCK_SPLITS, COMBINE_BLOCK_D),
            (
                batch_size * self._h_q,
                math.ceil(D_CKV / COMBINE_BLOCK_D),
            ),
        )

    def _ensure_compiled_launch_pack(self, out, lse) -> None:
        # Exact pointer identity preserves every alignment specialization that
        # the public caller-provided output contract previously admitted. Keep
        # only the most recent pack so workloads that rotate output buffers do
        # not grow an unbounded launcher cache.
        key = (
            int(out.data_ptr()),
            int(lse.data_ptr()),
            self._use_full_tail_specialization(),
            self._use_merged_state_v_completion(),
            self._use_pretranspose_v1(),
            self._use_fixed_ten_page_v1(),
            self._use_fixed_two_page_v1(),
        )
        if key == self._launch_pack_key:
            self._launch_pack_reuses += 1
            return

        target_out = out if self._direct_single_output else self._partial_out
        direct_lse = self._use_direct_lse_v2()
        target_lse = lse if direct_lse else self._partial_lse2
        use_pdl = self._use_programmatic_dependent_launch()
        partial_grid = (
            int(self._meta.split_batch.numel()) * (self._h_q // TLE_FP8_BH),
        )
        partial_jit = (
            _fp8_dense_mla_splitk_partial_pdl
            if use_pdl
            else (
                _fp8_dense_mla_splitk_partial_pretranspose
                if self._use_pretranspose_v1()
                else _fp8_dense_mla_splitk_partial
            )
        )
        partial_runner, partial_args = _prepare_compiled_runner(
            partial_jit,
            self._partial_launch_args(target_out, target_lse),
            partial_grid,
            launch_pdl=False,
        )
        if direct_lse:
            aux_runner, aux_bound_args = None, None
        else:
            aux_jit, aux_args, aux_grid = self._aux_launch_spec(out, lse)
            aux_runner, aux_bound_args = _prepare_compiled_runner(
                aux_jit,
                aux_args,
                aux_grid,
                num_warps=(
                    8
                    if aux_jit
                    in (
                        _triton_fp8_cuda_coarse_combine_kernel,
                        _triton_fp8_cuda_coarse_combine_pdl_kernel,
                    )
                    else 4
                ),
                launch_pdl=use_pdl,
            )
        self._partial_compiled_runner = partial_runner
        self._partial_compiled_args = partial_args
        self._aux_compiled_runner = aux_runner
        self._aux_compiled_args = aux_bound_args
        self._launch_pack_key = key
        self._launch_pack_reuses = 0
        self._cuda_graph_key = None
        self._cuda_graph = None
        self._cuda_graph_capture_stream = None
        self._cuda_graph_eligible = not use_pdl

    def _ensure_cuda_graph_replay(self):
        """Capture the stable two-kernel prepared replay after one pointer hit."""
        key = self._launch_pack_key
        if key is None or not self._cuda_graph_eligible or self._launch_pack_reuses < 1:
            return None
        if self._cuda_graph_key == key and self._cuda_graph is not None:
            return self._cuda_graph

        current_stream = torch.cuda.current_stream(self._out.device)
        capture_stream = torch.cuda.Stream(device=self._out.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            self._partial_compiled_runner(*self._partial_compiled_args)
            if self._aux_compiled_runner is not None:
                self._aux_compiled_runner(*self._aux_compiled_args)
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            self._partial_compiled_runner(*self._partial_compiled_args)
            if self._aux_compiled_runner is not None:
                self._aux_compiled_runner(*self._aux_compiled_args)
        current_stream.wait_stream(capture_stream)
        self._cuda_graph_key = key
        self._cuda_graph = graph
        self._cuda_graph_capture_stream = capture_stream
        return graph

    def _claim(self) -> None:
        if self._in_use:
            raise RuntimeError("prepared handle is already in use")
        self._in_use = True

    def _validate_length_bounds(self, values) -> None:
        for batch_index, (value, previous, capacity) in enumerate(
            zip(values, self._cache_seqlens_host, self._max_cache_seqlens)
        ):
            if value < previous:
                raise RuntimeError(f"cache_seqlens[{batch_index}] must be monotonic")
            if value > capacity:
                raise RuntimeError(
                    f"cache_seqlens[{batch_index}] {value} exceeds prepared "
                    f"capacity {capacity}"
                )

    def _apply_length_certificate(
        self,
        cache_seqlens,
        *,
        require_version_change: bool,
    ) -> None:
        values = _host_certificate_lengths(
            cache_seqlens,
            "cache_seqlens",
            batch_size=len(self._cache_seqlens_host),
        )
        self._validate_length_bounds(values)

        version = _tensor_version(self.cache_seqlens)
        changed = values != self._cache_seqlens_host
        if changed and require_version_change and version == self._cache_version:
            raise RuntimeError(
                "cache_seqlens storage did not receive an observable in-place "
                "PyTorch update before the new host certificate"
            )
        if not changed and version != self._cache_version:
            raise RuntimeError(
                "cache_seqlens changed without a new host length certificate"
            )

        pages, logical_active_splits = _length_page_state(
            values,
            int(self._meta.adaptive_fixed_pages),
        )
        for batch_index, (logical, capacity) in enumerate(
            zip(logical_active_splits, self._meta.capacity_splits)
        ):
            if logical > capacity:
                raise RuntimeError(
                    f"logical split count for batch {batch_index} exceeds capacity"
                )

        self._cache_seqlens_host = values
        self._num_pages = pages
        self._logical_active_splits = logical_active_splits
        self._cache_version = version

    def set_cache_seqlens_(self, cache_seqlens) -> None:
        """Own a monotonic in-place length update on the bound CUDA stream."""
        values = _host_certificate_lengths(
            cache_seqlens,
            "cache_seqlens",
            batch_size=len(self._cache_seqlens_host),
        )
        self._validate_length_bounds(values)
        self._claim()
        try:
            if values != self._cache_seqlens_host:
                update = torch.tensor(
                    values,
                    dtype=self.cache_seqlens.dtype,
                    device=self.cache_seqlens.device,
                )
                self.cache_seqlens.copy_(update)
            self._apply_length_certificate(
                values,
                require_version_change=False,
            )
        finally:
            self._in_use = False

    def _validate_output(self, tensor: torch.Tensor, *, lse: bool) -> None:
        template = self._lse if lse else self._out
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

    def launch(self, *, cache_seqlens=None, out=None, lse=None):
        """Submit one prepared decode step using a host length certificate."""
        self._claim()
        try:
            if cache_seqlens is None:
                if _tensor_version(self.cache_seqlens) != self._cache_version:
                    raise RuntimeError(
                        "cache_seqlens changed without a host length certificate"
                    )
            else:
                self._apply_length_certificate(
                    cache_seqlens,
                    require_version_change=True,
                )

            if (out is None) != (lse is None):
                raise RuntimeError(
                    "out and lse must either both be supplied or both omitted"
                )
            if out is None:
                out = torch.empty_like(self._out)
                lse = torch.empty_like(self._lse)
            else:
                self._validate_output(out, lse=False)
                self._validate_output(lse, lse=True)

            self._ensure_compiled_launch_pack(out, lse)
            graph = self._ensure_cuda_graph_replay()
            if graph is None:
                self._partial_compiled_runner(*self._partial_compiled_args)
                if self._aux_compiled_runner is not None:
                    self._aux_compiled_runner(*self._aux_compiled_args)
            else:
                graph.replay()
            return out, lse
        finally:
            self._in_use = False

    __call__ = launch

    def debug_state(self) -> dict:
        return {
            "batch_parallel": True,
            "programmatic_dependent_launch": (
                self._use_programmatic_dependent_launch()
            ),
            "pdl_scope": (
                "consumer_headroom_coarse_combine"
                if self._use_programmatic_dependent_launch()
                else None
            ),
            "pdl_partial_ctas": (self._programmatic_dependency_capacity()[0]),
            "pdl_consumer_ctas": (self._programmatic_dependency_capacity()[1]),
            "pdl_sm_count": (self._programmatic_dependency_capacity()[2]),
            "pdl_capacity_safe": (
                sum(self._programmatic_dependency_capacity()[:2])
                <= self._programmatic_dependency_capacity()[2]
            ),
            "pdl_consumer_headroom_safe": (
                self._programmatic_dependency_capacity()[0]
                + 2 * self._programmatic_dependency_capacity()[1]
                <= self._programmatic_dependency_capacity()[2]
            ),
            "per_batch_loop": False,
            "batch_launch_count": 1,
            "partial": True,
            "combine": not self._direct_single_output,
            "cuda_coarse_combine": (
                not self._direct_single_output
                and int(self._out.shape[0]) >= CUDA_COARSE_COMBINE_MIN_BATCH
            ),
            "combine_policy": (
                None
                if self._direct_single_output
                else (
                    "cuda_coarse_8_rows"
                    if int(self._out.shape[0]) >= CUDA_COARSE_COMBINE_MIN_BATCH
                    else "fine_splitk"
                )
            ),
            "combine_block_rows": (
                CUDA_COARSE_COMBINE_BLOCK_ROWS
                if (
                    not self._direct_single_output
                    and int(self._out.shape[0]) >= CUDA_COARSE_COMBINE_MIN_BATCH
                )
                else None
            ),
            "combine_min_batch": CUDA_COARSE_COMBINE_MIN_BATCH,
            "direct_single_output": self._direct_single_output,
            "direct_output_dtype": (
                str(self._out.dtype) if self._direct_single_output else None
            ),
            "lse_only_finalize": self._direct_single_output,
            "full_dv_finalize": not self._direct_single_output,
            "compact_workspace_bytes": int(
                self._partial_out.numel() * self._partial_out.element_size()
                + self._partial_lse2.numel() * self._partial_lse2.element_size()
            ),
            "max_splits": int(self._meta.max_splits),
            "max_pages_per_split": int(self._meta.max_pages_per_split),
            "total_splits": int(self._meta.split_batch.numel()),
            "adaptive_fixed_pages": int(self._meta.adaptive_fixed_pages),
            "adaptive_fixed_pairs": int(self._meta.adaptive_fixed_pairs),
            "adaptive_selection": list(self._meta.adaptive_selection),
            "capacity_splits": list(self._meta.capacity_splits),
            "initial_cache_seqlens": list(self._initial_cache_seqlens),
            "cache_seqlens": list(self._cache_seqlens_host),
            "num_pages": list(self._num_pages),
            "logical_active_splits": list(self._logical_active_splits),
            "full_tail_specialization": self._use_full_tail_specialization(),
            "fixed_two_page_specialization": self._use_fixed_two_page_v1(),
            "merged_state_v_completion": self._use_merged_state_v_completion(),
            "masked_splits": int(
                sum(self._meta.capacity_splits) - sum(self._logical_active_splits)
            ),
            "max_cache_seqlens": list(self._max_cache_seqlens),
            "immutable_capacity_schedule": True,
            "fresh_output_storage": "two_empty_like",
            "padded_block_table": self._execution_block_table is not self.block_table,
            "schedule": "all_batch_csr_h800_wave_cost_fixed_pairs_tail34",
            "explicit_pipeline": True,
        }


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
    tile_scheduler_metadata=None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    pages_per_split: int = DEFAULT_PAGES_PER_SPLIT,
    max_splits: Optional[int] = None,
    out=None,
    lse=None,
):
    """Public one-shot op: metadata + prepare + run."""
    if not HAS_TLE:
        raise NotImplementedError("FP8 MLA requires Hopper and FlagTree GPU extensions")
    batch_size = int(q_nope.shape[0])
    h_q = int(q_nope.shape[2])
    if h_q <= 0 or h_q % TLE_FP8_BH:
        raise ValueError("HQ must be a positive multiple of 64")
    if num_splits is None:
        _, num_splits = get_mla_fp8_metadata(
            cache_seqlens,
            h_q,
            1,
            pages_per_split=pages_per_split,
            max_splits=max_splits,
        )
    capacity = _host_lengths(
        cache_seqlens,
        "cache_seqlens",
        batch_size=batch_size,
    )
    if any(length <= 0 or length > MAX_SEQUENCE_LENGTH for length in capacity):
        raise ValueError(f"cache_seqlens entries must be in [1, {MAX_SEQUENCE_LENGTH}]")
    required_pages = max(math.ceil(length / PAGE_SIZE) for length in capacity)
    if block_table.ndim != 2 or int(block_table.shape[0]) != batch_size:
        raise ValueError("block_table must be a two-dimensional batch table")
    if int(block_table.shape[1]) < required_pages:
        raise ValueError("block_table does not cover cache_seqlens")
    meta = _build_adaptive_execution_meta(
        capacity,
        h_q,
        q_nope.device,
        pages_per_split,
    )
    if max_splits is not None and int(max_splits) < max(meta.capacity_splits):
        raise ValueError("max_splits cannot be below an adaptive row capacity")
    execution_block_table = _pad_block_table(block_table, meta.padded_pages)
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
    handle = _FlashMLAFp8PreparedHandle(
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
    )
    return handle(out=out, lse=lse)
