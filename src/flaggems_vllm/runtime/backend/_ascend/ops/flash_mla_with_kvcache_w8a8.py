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

import functools
import importlib.util
import logging
import math
from typing import Optional

import torch
import triton

from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import (
    D_CKV,
    D_ROPE,
    PAGE_SIZE,
    validate_dense_query_layout,
)

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def ascend_mla_available() -> bool:
    if (
        importlib.util.find_spec("flagtree") is None
        or importlib.util.find_spec("triton.language.extra.cann.extension") is None
        or importlib.util.find_spec("triton.backends.ascend.driver") is None
    ):
        return False
    from triton.backends.ascend.driver import NPUUtils

    return NPUUtils().get_arch().startswith("Ascend910B")


def flash_mla_with_kvcache_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int = D_CKV,
    tile_scheduler_metadata=None,
    num_splits: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode paged INT8 MLA. An explicit split count selects the Triton split-KV path."""
    logger.debug("GEMS_ASCEND FLASH_MLA_WITH_KVCACHE_W8A8")
    if tile_scheduler_metadata is not None:
        raise NotImplementedError(
            "Ascend does not consume external tile scheduler metadata"
        )
    if num_splits is not None:
        if type(num_splits) is not int:
            raise TypeError("num_splits must be a Python integer or None")
        if num_splits not in (1, 2, 4, 8, 16):
            raise ValueError("num_splits must be one of 1, 2, 4, 8, 16")
    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    if softmax_scale is not None and not isinstance(softmax_scale, (int, float)):
        raise TypeError("softmax_scale must be a Python number")
    scale = (D_CKV + D_ROPE) ** -0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    validate_dense_query_layout(q_nope)
    if k_cache_lora.ndim not in (3, 4) or block_table.ndim != 2:
        raise ValueError("Expected paged KV cache and rank-2 block table")
    batch, queries, heads, dim = q_nope.shape
    cache_prefix = k_cache_lora.shape[:-1]
    if (
        dim != D_CKV
        or head_dim_v != D_CKV
        or k_cache_lora.shape[-1] != D_CKV
        or cache_prefix[1:] not in ((PAGE_SIZE,), (PAGE_SIZE, 1))
    ):
        raise ValueError(
            "Expected Sq=1/2, H=16/32/64/128, page=64, Dv=512 and one KV head"
        )
    tensors = (
        (q_nope, q_nope.shape, torch.int8),
        (q_rope, (*q_nope.shape[:-1], D_ROPE), torch.bfloat16),
        (q_scale, (*q_nope.shape[:-1], 1), torch.float32),
        (k_cache_lora, k_cache_lora.shape, torch.int8),
        (k_cache_rope, (*cache_prefix, D_ROPE), torch.bfloat16),
        (k_scale, (*cache_prefix, 1), torch.float32),
        (block_table, (batch, block_table.shape[1]), torch.int32),
        (cache_seqlens, (batch,), torch.int32),
    )
    device = q_nope.device
    if device.type != "npu":
        raise ValueError("Ascend dense MLA requires NPU tensors")
    if not ascend_mla_available():
        raise NotImplementedError(
            "Ascend dense W8A8 MLA requires FlagTree on Ascend910B"
        )
    for tensor, shape, dtype in tensors:
        if tensor.device != device or tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(f"Expected {shape}, {dtype} on {device}")
        if not tensor.is_contiguous():
            raise ValueError("Ascend dense MLA requires contiguous input tensors")
    if out is None:
        out = torch.empty(q_nope.shape, device=device, dtype=q_rope.dtype)
    if lse is None:
        lse = torch.empty((batch, heads, queries), device=device, dtype=torch.float32)
    for tensor, shape, dtype in (
        (out, q_nope.shape, q_rope.dtype),
        (lse, (batch, heads, queries), torch.float32),
    ):
        if (
            tensor.device != device
            or tensor.shape != shape
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"Expected a contiguous output {shape}, {dtype} on {device}"
            )
    if batch:
        from flaggems_vllm.runtime.backend._ascend.fused import mla_w8a8

        with torch.npu.device(device.index):
            tensors = (
                q_nope,
                q_rope,
                k_cache_lora,
                k_cache_rope,
                q_scale,
                k_scale,
                block_table,
                cache_seqlens,
            )
            width = block_table.shape[1]
            rows = batch * queries * heads
            if width == 0:
                mla_w8a8.empty_decode[(rows,)](out, lse)
            elif num_splits is not None:
                partial, stats = out, lse
                if num_splits > 1:
                    partial = torch.empty(
                        (rows, num_splits, D_CKV), device=device, dtype=torch.float32
                    )
                    stats = torch.empty(
                        (rows, num_splits, 2), device=device, dtype=torch.float32
                    )
                mla_w8a8.launch(
                    mla_w8a8.paged_decode,
                    rows // 16 * num_splits,
                    (*tensors, partial, stats),
                    (heads, queries, width, scale, causal, num_splits),
                )
                if num_splits > 1:
                    mla_w8a8.portable_merge[(rows,)](
                        partial,
                        stats,
                        out,
                        lse,
                        heads,
                        queries,
                        num_splits,
                        num_splits,
                        heads,
                    )
            elif width <= 6 and mla_w8a8.requires_portable_short():
                page_heads = max(32, heads)
                page_rows = batch * queries * page_heads
                partial = torch.empty(
                    (page_rows, width, D_CKV), device=device, dtype=torch.float32
                )
                stats = torch.empty(
                    (page_rows, width, 2), device=device, dtype=torch.float32
                )
                tasks = batch * queries * triton.cdiv(heads, 32) * width * 4
                mla_w8a8.portable_page[(tasks,)](
                    *tensors,
                    partial,
                    stats,
                    page_heads,
                    heads,
                    queries,
                    width,
                    scale,
                    causal,
                )
                mla_w8a8.portable_merge[(rows,)](
                    partial,
                    stats,
                    out,
                    lse,
                    heads,
                    queries,
                    width,
                    triton.next_power_of_2(width),
                    page_heads,
                )
            elif width <= 6:
                tasks = batch * queries * (heads // 16)
                mla_w8a8.launch(
                    mla_w8a8.paged_decode,
                    tasks,
                    (*tensors, out, lse),
                    (heads, queries, width, scale, causal, 0),
                )
            else:
                from flaggems_vllm.runtime.backend._ascend.fused import mla_w8a8_window

                if mla_w8a8_window.HAS_PAGED_INT8_OPS:
                    window_rows = min(128, heads * queries)
                    tasks = batch * (heads * queries // window_rows)
                    cores = min(20, tasks)
                    plan_stride = (
                        max(8, triton.next_power_of_2(triton.cdiv(tasks, cores)))
                        if batch <= 128
                        else 0
                    )
                    words = cores * 2 * window_rows * 384
                    c, r, p, pv = mla_w8a8.allocate_scratch(
                        device,
                        words,
                        cores * (4 * window_rows * 512 + 32 + 16896 + plan_stride),
                    )
                    mla_w8a8.launch(
                        mla_w8a8_window.window_pipeline,
                        cores,
                        (
                            q_nope,
                            q_rope,
                            k_cache_lora,
                            k_cache_rope,
                            q_scale,
                            k_scale,
                            block_table,
                            cache_seqlens,
                            c,
                            r,
                            p,
                            pv,
                            pv.view(torch.float16),
                            pv.view(torch.int8),
                            out,
                            lse,
                        ),
                        (
                            tasks,
                            width,
                            cores,
                            scale,
                            heads,
                            queries,
                            causal,
                            window_rows,
                            batch,
                            plan_stride,
                        ),
                    )
                else:
                    mla_w8a8.launch(
                        mla_w8a8.paged_decode,
                        rows // 16,
                        (*tensors, out, lse),
                        (heads, queries, width, scale, causal, 1),
                    )
    return out, lse
