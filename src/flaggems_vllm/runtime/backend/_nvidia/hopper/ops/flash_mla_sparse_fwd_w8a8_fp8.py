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

from typing import Callable, Optional, Tuple

import torch
import triton

from flaggems_vllm.ops.flash_mla import HAS_TLE_FLASH_MLA as HAS_TLE
from flaggems_vllm.ops.flash_mla import _get_num_sms
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.mla_sparse_fp8_pipeline import (
    sparse_fp8_checked_merge,
    sparse_fp8_empty,
    sparse_fp8_merge,
    sparse_fp8_repair,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.mla_sparse_fp8_tiles import (
    sparse_fp8_compact,
    sparse_fp8_tile,
    sparse_fp8_warp_specialized,
)
from flaggems_vllm.utils.libentry import LibEntry

SPARSE_LAUNCH_CACHE_LIMIT = 128
SPARSE_LAUNCH_CACHE: dict[tuple, tuple[Callable[..., None], tuple[int | bool, ...]]] = (
    {}
)


def launch_sparse_entry(
    entry: LibEntry,
    grid: tuple[int, ...] | Callable[[dict[str, object]], tuple[int, ...]],
    *arguments: torch.Tensor | int | float | bool | None,
    **launch_kwargs: int | bool,
) -> None:
    # Retain dynamic FlagTune hooks; ordinary entries tune once per exact ABI.
    if entry._has_flagtune_tuner:
        entry[grid](*arguments, **launch_kwargs)
        return
    signature = tuple(
        (
            (argument.dtype, argument.data_ptr() % entry.divisibility == 0)
            if isinstance(argument, torch.Tensor)
            else (type(argument), argument)
        )
        for argument in arguments
    )
    key = (entry, torch.cuda.current_device(), signature, tuple(launch_kwargs.items()))
    prepared = SPARSE_LAUNCH_CACHE.get(key)
    if prepared is None:
        kernel, constants = entry[grid](*arguments, **launch_kwargs)
        trailing = tuple(
            (
                launch_kwargs[param.name]
                if param.name in launch_kwargs
                else constants[param.name]
            )
            for param in entry.jit_function.params[len(arguments) :]
        )
        if callable(grid):
            resolved_grid = grid(
                {**dict(zip(entry.arg_names, arguments)), **launch_kwargs, **constants}
            )
        else:
            resolved_grid = grid
        while len(SPARSE_LAUNCH_CACHE) >= SPARSE_LAUNCH_CACHE_LIMIT:
            SPARSE_LAUNCH_CACHE.clear()
        # Retain launch metadata without holding input or output Tensor storage.
        SPARSE_LAUNCH_CACHE[key] = (kernel[resolved_grid], trailing)
        return
    runner, trailing = prepared
    runner(*arguments, *trailing)


def flash_mla_sparse_fwd_w8a8_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode using the separate per-token cache format of dense MLA.

    q_nope [B, 1, H, 512] and k_cache_lora [P, 64, 512] are FP8 e4m3fn;
    q_rope [B, 1, H, 64] and k_cache_rope [P, 64, 64] are BF16.
    Both NoPE and RoPE store values divided by the corresponding FP32 scale:
    q_scale [B, 1, H, 1] and k_scale [P, 64, 1]. This directly accepts
    quantize_q_ckv_per_token / quantize_k_ckv_per_token outputs from dense MLA.
    V is the dequantized 512-dimensional NoPE cache. H must be 64 or 128.

    indices [B, 1, topk] contains int32 physical token IDs (page * 64 + slot).
    Negative and out-of-range IDs are ignored. topk_length [B] optionally
    limits the number of entries read per request. attn_sink [H] affects
    output only. Inputs must be finite, with positive finite scales.

    Returns BF16 output [B, 1, H, 512] and natural-log FP32 LSE [B, H, 1].
    Empty attention produces zero output and +inf LSE. Forward only.
    Sensitive rows use FP32 recomputation inline, during merge, or in the
    existing separate repair kernel, depending on the selected path.
    Requires FlagTree cross-dtype WGMMA support (PR #1001).
    """
    if not HAS_TLE:
        raise NotImplementedError(
            "FP8 sparse MLA requires Hopper and FlagTree GPU extensions"
        )
    if q_nope.device.type != "cuda":
        raise NotImplementedError("FP8 sparse MLA requires NVIDIA Hopper CUDA")
    if q_nope.dtype != torch.float8_e4m3fn or k_cache_lora.dtype != torch.float8_e4m3fn:
        raise TypeError("NoPE tensors must have dtype float8_e4m3fn")
    if q_rope.dtype != torch.bfloat16 or k_cache_rope.dtype != torch.bfloat16:
        raise TypeError("RoPE tensors must have dtype bfloat16")
    if q_nope.ndim != 4 or k_cache_lora.ndim != 3 or indices.ndim != 3:
        raise ValueError("q_nope, cache and indices must have ranks 4, 3 and 3")
    batch, query_length, heads, dim = q_nope.shape
    pages = k_cache_lora.shape[0]
    topk = indices.shape[-1]
    if query_length != 1 or heads not in (64, 128) or dim != 512:
        raise NotImplementedError(
            "Requires one query, 64/128 heads and 512 NoPE dimensions"
        )
    if q_rope.shape != (batch, 1, heads, 64):
        raise ValueError("q_rope must have shape [batch, 1, heads, 64]")
    if k_cache_lora.shape != (pages, 64, 512) or k_cache_rope.shape != (pages, 64, 64):
        raise ValueError(
            "Caches must have page size 64 and NoPE/RoPE dimensions 512/64"
        )
    if indices.shape != (batch, 1, topk) or indices.dtype != torch.int32:
        raise ValueError("indices must be int32 [batch, 1, topk]")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("q_scale and k_scale must have dtype float32")
    if q_scale.shape != (batch, 1, heads, 1) or k_scale.shape != (pages, 64, 1):
        raise ValueError("Scale shapes must be [batch, 1, heads, 1] and [pages, 64, 1]")
    for tensor in (q_nope, q_rope, k_cache_lora, k_cache_rope):
        if tensor.stride(-1) != 1:
            raise ValueError("NoPE and RoPE must be contiguous in the last dimension")
    for tensor in (
        q_rope,
        k_cache_lora,
        k_cache_rope,
        indices,
        q_scale,
        k_scale,
        attn_sink,
        topk_length,
    ):
        if tensor is not None and tensor.device != q_nope.device:
            raise ValueError("All tensors must be on the same CUDA device")
    if attn_sink is not None and (
        attn_sink.shape != (heads,)
        or attn_sink.dtype != torch.float32
        or not attn_sink.is_contiguous()
    ):
        raise ValueError("attn_sink must be contiguous float32 [heads]")
    if topk_length is not None and (
        topk_length.shape != (batch,)
        or topk_length.dtype != torch.int32
        or not topk_length.is_contiguous()
    ):
        raise ValueError("topk_length must be contiguous int32 [batch]")
    output = torch.empty(
        (batch, 1, heads, 512), device=q_nope.device, dtype=torch.bfloat16
    )
    lse = torch.empty((batch, heads, 1), device=q_nope.device, dtype=torch.float32)
    if batch == 0:
        return output, lse
    if pages == 0 or topk == 0:
        rows = batch * heads
        sparse_fp8_empty[(triton.cdiv(rows * 512, 1024),)](
            output, lse, rows, num_warps=4
        )
        return output, lse
    softmax_scale = 576**-0.5 if softmax_scale is None else float(softmax_scale)
    use_partitioned = batch <= 16
    use_tile = False
    num_sms = _get_num_sms(q_nope.device)
    head_groups = batch * (heads // 64)
    desired_splits = max(1, num_sms // head_groups)
    keys_per_split = 64 if batch <= 16 else 256
    max_splits = min(desired_splits, 32, max(1, topk // keys_per_split))
    splits = 1 << (max_splits.bit_length() - 1)
    if use_partitioned:
        can_async_copy = all(
            tensor.data_ptr() % 16 == 0
            and tensor.stride(0) * tensor.element_size() % 16 == 0
            and tensor.stride(-2) * tensor.element_size() % 16 == 0
            for tensor in (q_nope, q_rope, k_cache_lora, k_cache_rope)
        )
        tile_splits = triton.next_power_of_2(max(1, triton.cdiv(topk, 64)))
        # Keep value tiling in its validated workset; short rows use the compact CTA.
        use_tile = topk >= 512 and head_groups <= 8 and tile_splits <= 32
        if use_tile:
            splits = tile_splits
        else:
            # Checked merge consumes partial statistics, including for empty/short rows.
            splits = max(splits, 2)
    repair_splits = splits
    flag_heads = heads // 64
    repair_flags = torch.empty(
        (batch, flag_heads, splits), device=q_nope.device, dtype=torch.int32
    )
    if splits > 1:
        partial = torch.empty(
            (batch, splits, heads, 512),
            device=q_nope.device,
            dtype=torch.float32,
        )
        stats = torch.empty(
            (batch, splits, heads, 2), device=q_nope.device, dtype=torch.float32
        )
    else:
        partial, stats = output, lse
    if use_tile:
        launch_sparse_entry(
            sparse_fp8_tile,
            lambda meta: (batch, heads // 64, splits * (512 // meta["BLOCK_D"])),
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            partial,
            stats,
            repair_flags,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            splits,
            topk_length is not None,
            attn_sink is not None,
            CAN_ASYNC=can_async_copy,
        )
    elif use_partitioned:
        launch_sparse_entry(
            sparse_fp8_compact,
            (batch, heads // 64, splits),
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            partial,
            stats,
            repair_flags,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            splits,
            topk_length is not None,
            attn_sink is not None,
            CAN_ASYNC=can_async_copy,
        )
    else:
        launch_sparse_entry(
            sparse_fp8_warp_specialized,
            (batch, heads // 64, splits),
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            partial,
            stats,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            topk_length is not None,
            splits,
            attn_sink is not None,
            repair_flags,
        )
    if not use_partitioned:
        launch_sparse_entry(
            sparse_fp8_repair,
            lambda meta: (batch, triton.cdiv(heads, meta["BLOCK_H"]), splits),
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            indices,
            q_scale,
            k_scale,
            attn_sink,
            topk_length,
            output,
            lse,
            partial,
            stats,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            indices.stride(0),
            indices.stride(2),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            splits,
            attn_sink is not None,
            topk_length is not None,
            repair_flags,
            REPAIR_SPLITS=repair_splits,
        )
    if use_partitioned and not use_tile:
        sparse_fp8_checked_merge[(batch * heads,)](
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            repair_flags,
            partial,
            stats,
            output,
            lse,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            heads,
            pages * 64,
            topk,
            softmax_scale,
            splits,
            topk_length is not None,
            attn_sink is not None,
            num_warps=4,
        )
    elif splits > 1:
        sparse_fp8_merge[(batch * heads,)](
            partial,
            stats,
            attn_sink,
            output,
            lse,
            heads,
            splits,
            attn_sink is not None,
            num_warps=4,
        )
    return output, lse
