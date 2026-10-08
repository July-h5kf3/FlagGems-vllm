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


import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl

from flaggems_vllm.ops.flash_kernel import (
    apply_alibi,
    apply_mask,
    apply_softcap,
    softmax_rescale,
    virtual_to_cache_offset,
)
from flaggems_vllm.runtime import torch_device_fn

# log2(e), so exp2(score * LOG2E) matches exp(score).
# Ascend Triton only allows constexpr globals inside kernels.
LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)
# Signed INT8 stores probability levels 0..255 with zero point 128.
PROB_QUANT_LEVELS = tl.constexpr(255)
DESCALE_BLOCK = tl.constexpr(128)
# Cube INT8 MMA wants multiples of 16. 64 stays inside one 128-wide descale block.
BLOCK_M = tl.constexpr(64)
BLOCK_N = tl.constexpr(64)
# One program per query tile. Stay far below Ascend's 65535 launch limit.
LAUNCH_CHUNK = 2048


@triton.jit
def _flash_int8_fwd(
    Q,
    K,
    V,
    O,
    LSE,
    PART_OUT,
    PART_MAX,
    PART_DENOM,
    CUQ,
    CUK,
    USED,
    TABLE,
    QS,
    KS,
    VS,
    ALIBI,
    task_begin,
    batch_size,
    heads,
    group,
    max_blocks,
    sq,
    hq,
    sk,
    hk,
    pk,
    sv,
    hv,
    pv,
    so,
    ho,
    qs0,
    qs1,
    qs2,
    ks0,
    ks1,
    ks2,
    vs0,
    vs1,
    vs2,
    table_stride,
    alibi_batch_stride,
    alibi_head_stride,
    total_q,
    scale,
    causal: tl.constexpr,
    left: tl.constexpr,
    right: tl.constexpr,
    softcap: tl.constexpr,
    has_alibi: tl.constexpr,
    write_lse: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    SPLITS: tl.constexpr,
):
    task = task_begin + tl.program_id(0)
    # heads is the KV head count. Query heads that share it occupy the M tile.
    tile_part = task // (batch_size * heads)
    part = tile_part % SPLITS
    tile = tile_part // SPLITS
    rem = task % (batch_size * heads)
    batch = rem // heads
    kv_head = rem % heads
    lane = tl.arange(0, BLOCK_M)
    rows_per_tile = BLOCK_M // group
    q_row = tile * rows_per_tile + lane // group
    q_head = kv_head * group + lane % group
    q_start_i32 = tl.load(CUQ + batch)
    q_end_i32 = tl.load(CUQ + batch + 1)
    q_start = q_start_i32.to(tl.int64)
    nq = q_end_i32 - q_start_i32
    if PAGED:
        k_start = tl.full((), 0, tl.int64)
        nk = tl.load(USED + batch)
    else:
        k_start_i32 = tl.load(CUK + batch)
        k_end_i32 = tl.load(CUK + batch + 1)
        k_start = k_start_i32.to(tl.int64)
        nk = k_end_i32 - k_start_i32
    # Unused lanes appear when group does not divide the M tile.
    row_ok = (lane < rows_per_tile * group) & (q_row < nq)
    rows_safe = tl.where(row_ok, q_row, 0).to(tl.int64)
    d = tl.arange(0, D)
    if tile * rows_per_tile < nq:
        q = tl.load(
            Q + (q_start + rows_safe[:, None]) * sq + q_head[:, None] * hq + d[None, :],
            row_ok[:, None],
            0,
        )
        q_scale = tl.load(
            QS + batch * qs0 + q_head * qs1 + (q_row // DESCALE_BLOCK) * qs2,
            row_ok,
            0,
        )
        maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        denom = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        slope = tl.zeros((BLOCK_M,), dtype=tl.float32)
        if has_alibi:
            slope = tl.load(
                ALIBI + batch * alibi_batch_stride + q_head * alibi_head_stride
            )
        first = 0
        end = nk
        if left >= 0:
            first = tl.maximum(0, tile * rows_per_tile + nk - nq - left) // BLOCK_N
        if causal:
            end = tl.minimum(end, (tile + 1) * rows_per_tile + nk - nq)
        if right >= 0:
            end = tl.minimum(end, (tile + 1) * rows_per_tile + nk - nq + right)
        part_blocks = (max_blocks + SPLITS - 1) // SPLITS
        for start in tl.range(part * part_blocks, (part + 1) * part_blocks):
            if (start >= first) & (start < max_blocks) & (start * BLOCK_N < end):
                n = start * BLOCK_N + tl.arange(0, BLOCK_N)
                n_ok = n < nk
                n_safe = tl.where(n_ok, n, 0).to(tl.int64)
                if PAGED:
                    k_row = virtual_to_cache_offset(
                        n_safe,
                        nk,
                        TABLE + batch * table_stride,
                        PAGE,
                        sk,
                        pk,
                        boundary_check=True,
                    )
                    v_row = virtual_to_cache_offset(
                        n_safe,
                        nk,
                        TABLE + batch * table_stride,
                        PAGE,
                        sv,
                        pv,
                        boundary_check=True,
                    )
                else:
                    k_row = (k_start + n_safe) * sk
                    v_row = (k_start + n_safe) * sv
                if PAGED and PAGE == BLOCK_N:
                    page_id = tl.load(
                        TABLE + batch * table_stride + start,
                        start * PAGE < nk,
                        0,
                    ).to(tl.int64)
                    k_page = tl.make_block_ptr(
                        base=K + page_id * pk + kv_head * hk,
                        shape=(PAGE, D),
                        strides=(sk, 1),
                        offsets=(0, 0),
                        block_shape=(BLOCK_N, D),
                        order=(1, 0),
                    )
                    k = tl.trans(tl.load(k_page))
                elif PAGED and PAGE == 16:
                    k_rows = tl.full((BLOCK_N, D), 0, tl.int8)
                    first_page = start * (BLOCK_N // PAGE)
                    for page_in_tile in tl.static_range(0, BLOCK_N // PAGE):
                        page_number = first_page + page_in_tile
                        physical_page = tl.load(
                            TABLE + batch * table_stride + page_number,
                            page_number < (nk + PAGE - 1) // PAGE,
                            0,
                        ).to(tl.int64)
                        k_page = tl.make_block_ptr(
                            base=K + physical_page * pk + kv_head * hk,
                            shape=(PAGE, D),
                            strides=(sk, 1),
                            offsets=(0, 0),
                            block_shape=(PAGE, D),
                            order=(1, 0),
                        )
                        k_rows = tle.dsa.insert_slice(
                            k_rows,
                            tl.load(k_page),
                            offsets=[page_in_tile * PAGE, 0],
                            sizes=[PAGE, D],
                            strides=[1, 1],
                        )
                    k = tl.trans(k_rows)
                elif not PAGED and (start + 1) * BLOCK_N <= nk:
                    k_block_ptr = tl.make_block_ptr(
                        base=K + k_start * sk + kv_head * hk,
                        shape=(nk, D),
                        strides=(sk, 1),
                        offsets=(start * BLOCK_N, 0),
                        block_shape=(BLOCK_N, D),
                        order=(1, 0),
                    )
                    k = tl.trans(tl.load(k_block_ptr))
                else:
                    k = tl.load(
                        K + k_row[None, :] + kv_head * hk + d[:, None],
                        n_ok[None, :],
                        0,
                    )
                descale_block = start * BLOCK_N // DESCALE_BLOCK
                ks = tl.load(KS + batch * ks0 + kv_head * ks1 + descale_block * ks2)
                vs = tl.load(VS + batch * vs0 + kv_head * vs1 + descale_block * vs2)
                scores = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
                scores = scores * (q_scale * ks * scale)[:, None]
                if softcap > 0:
                    scores = softcap * apply_softcap(
                        scores, 1.0 / softcap, is_softcap=True
                    )
                else:
                    scores = apply_softcap(scores, 1.0, is_softcap=False)
                scores = apply_alibi(
                    scores,
                    n,
                    q_row,
                    nq,
                    nk,
                    is_causal=False,
                    is_alibi=has_alibi,
                    alibi_slope=slope[:, None],
                )
                scores = apply_mask(
                    scores,
                    n,
                    q_row,
                    nq,
                    nk,
                    nk if left < 0 else left,
                    (
                        (0 if causal else nq)
                        if right < 0
                        else (tl.minimum(right, 0) if causal else right)
                    ),
                    is_even_mn=False,
                    is_causal=causal,
                    is_local=left >= 0 or right >= 0,
                )
                scores = tl.where(row_ok[:, None], scores * LOG2E, float("-inf"))
                tile_max = tl.max(scores, 1)
                acc, p, maximum, denom = softmax_rescale(
                    acc,
                    scores,
                    maximum,
                    denom,
                    softmax_scale_log2e=1.0,
                    is_border=True,
                    use_tile_max=True,
                )
                safe_max = tl.where(maximum == float("-inf"), 0, maximum)
                beta = tl.exp2(tile_max - safe_max)
                p_scale = beta * (1.0 / PROB_QUANT_LEVELS)
                p_int8 = (tl.floor(p * PROB_QUANT_LEVELS + 0.5) - 128).to(tl.int8)
                v = tl.load(
                    V + v_row[:, None] + kv_head * hv + d[None, :],
                    n_ok[:, None],
                    0,
                )
                # p_int8 = level - 128, and masked V lanes are zero, so
                # 128 * sum(V) restores the nonnegative levels.
                partial = tl.dot(p_int8, v, out_dtype=tl.int32).to(tl.float32)
                partial += 128.0 * tl.sum(v.to(tl.int32), 0).to(tl.float32)
                acc = acc + partial * (p_scale * vs)[:, None]
        if SPLITS > 1:
            part_row = task * BLOCK_M + lane
            tl.store(PART_OUT + part_row[:, None] * D + d[None, :], acc)
            tl.store(PART_MAX + part_row, maximum)
            tl.store(PART_DENOM + part_row, denom)
        else:
            result = acc / tl.where(denom > 0, denom, 1)[:, None]
            tl.store(
                O
                + (q_start + rows_safe[:, None]) * so
                + q_head[:, None] * ho
                + d[None, :],
                result.to(OUT_DTYPE),
                row_ok[:, None],
            )
            if write_lse:
                lse = tl.where(denom > 0, maximum * LN2 + tl.log(denom), float("inf"))
                tl.store(LSE + q_head * total_q + q_start + rows_safe, lse, row_ok)


@triton.jit
def _flash_int8_merge(
    O,
    LSE,
    PART_OUT,
    PART_MAX,
    PART_DENOM,
    CUQ,
    batch_size,
    heads,
    group,
    total_q,
    so,
    ho,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLITS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    task = tl.program_id(0)
    tile = task // (batch_size * heads)
    rem = task % (batch_size * heads)
    batch = rem // heads
    kv_head = rem % heads
    lane = tl.arange(0, BLOCK_M)
    d = tl.arange(0, D)
    q_start = tl.load(CUQ + batch).to(tl.int64)
    nq = tl.load(CUQ + batch + 1) - q_start
    q_row = tile * (BLOCK_M // group) + lane // group
    q_head = kv_head * group + lane % group
    row_ok = (lane < (BLOCK_M // group) * group) & (q_row < nq)
    rows_safe = tl.where(row_ok, q_row, 0).to(tl.int64)
    if tile * (BLOCK_M // group) < nq:
        maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        denom = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)
        for part in tl.static_range(0, SPLITS):
            part_row = (
                (tile * SPLITS + part) * batch_size * heads + rem
            ) * BLOCK_M + lane
            part_m = tl.load(PART_MAX + part_row)
            part_d = tl.load(PART_DENOM + part_row)
            part_a = tl.load(PART_OUT + part_row[:, None] * D + d[None, :])
            new_max = tl.maximum(maximum, part_m)
            safe_max = tl.where(new_max == float("-inf"), 0, new_max)
            alpha = tl.exp2(maximum - safe_max)
            beta = tl.exp2(part_m - safe_max)
            denom = denom * alpha + part_d * beta
            acc = acc * alpha[:, None] + part_a * beta[:, None]
            maximum = new_max
        result = acc / tl.where(denom > 0, denom, 1)[:, None]
        tl.store(
            O + (q_start + rows_safe[:, None]) * so + q_head[:, None] * ho + d[None, :],
            result.to(OUT_DTYPE),
            row_ok[:, None],
        )
        if WRITE_LSE:
            lse = tl.where(denom > 0, maximum * LN2 + tl.log(denom), float("inf"))
            tl.store(LSE + q_head * total_q + q_start + rows_safe, lse, row_ok)


def _check_descale(name, scale, batch, nheads, length, device):
    if scale is None or scale.dtype != torch.float32 or scale.ndim != 3:
        raise TypeError(f"{name} must be a rank-3 float32 tensor")
    if scale.device != device:
        raise ValueError(f"{name} must be on the same device as Q")
    if scale.shape[0] != batch or scale.shape[1] != nheads:
        raise ValueError(f"{name} must have shape [batch, heads, blocks]")
    if length > 0:
        needed = (length + DESCALE_BLOCK.value - 1) // DESCALE_BLOCK.value
        if scale.stride(-1) != 0 and scale.shape[-1] < needed:
            raise ValueError(f"{name} covers {scale.shape[-1]} blocks, need {needed}")


def flash_attn_varlen_func_w8a8_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    q_v: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[Tuple[int, int]] = None,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    scheduler_metadata: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
    fa_version: int = 2,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Ascend INT8 variable-length FlashAttention-2, inference forward only.

    Q is [total_q, heads, D] and K/V are packed [total_k, kv_heads, D] or paged
    [pages, page_size, kv_heads, D]. D is 64 or 128. Q/K/V are INT8. FP32
    descales are [batch, heads, ceil(length / 128)] and are indexed by logical
    position. QK is an INT8 dot with INT32 accumulation. Softmax stays in FP32.
    The general path uses 256 probability levels and an INT8 PV dot with a
    -128 zero point. The mixed Cube/Vector path adds a signed INT8 probability
    residual to improve accuracy. Output is BF16 unless out is FP16 or BF16.
    """
    if (
        dropout_p == 0
        and not return_attn_probs
        and q_v is None
        and scheduler_metadata is None
        and s_aux is None
        and num_splits == 0
        and cp_world_size == 1
        and cp_rank == 0
        and cp_tot_seqused_k is None
        and fa_version == 2
        and not return_softmax_lse
        and alibi_slopes is None
        and softcap == 0
        and (
            softmax_scale is None
            or (isinstance(softmax_scale, (int, float)) and softmax_scale == 128**-0.5)
        )
        and (
            window_size is None
            or (
                isinstance(window_size, (tuple, list))
                and len(window_size) == 2
                and all(
                    isinstance(bound, (int, float)) and bound < 0
                    for bound in window_size
                )
            )
        )
    ):
        inline_out = try_run(
            q,
            k,
            v,
            max_seqlen_q,
            cu_seqlens_q,
            max_seqlen_k,
            block_table,
            seqused_k,
            q_descale,
            k_descale,
            v_descale,
            out,
            causal,
        )
        if inline_out is not None:
            return inline_out
    if dropout_p != 0 or return_attn_probs or q_v is not None:
        raise NotImplementedError(
            "Only attention inference without dropout is supported"
        )
    if scheduler_metadata is not None or s_aux is not None or num_splits != 0:
        raise NotImplementedError(
            "Scheduler, auxiliary and split-KV modes are unsupported"
        )
    if cp_world_size != 1 or cp_rank != 0 or cp_tot_seqused_k is not None:
        raise NotImplementedError("Context parallel attention is unsupported")
    if fa_version != 2:
        raise NotImplementedError("Only FA2 is implemented")
    from flaggems_vllm.ops.flash_api import CHECK_DEVICE

    CHECK_DEVICE(q)
    CHECK_DEVICE(k)
    CHECK_DEVICE(v)
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
        raise NotImplementedError("Q/K/V must use INT8")
    if q.ndim != 3:
        raise ValueError("Q must have shape [total_q, heads, D]")
    if q.shape[-1] not in (64, 128):
        raise NotImplementedError("Only head dimensions 64 and 128 are supported")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise NotImplementedError("The last dimension of Q/K/V must be contiguous")
    paged = block_table is not None
    if paged and (k.ndim != 4 or v.ndim != 4):
        raise ValueError("Paged K/V must have shape [pages, page_size, kv_heads, D]")
    if not paged and (k.ndim != 3 or v.ndim != 3):
        raise ValueError("Packed K/V must have shape [total_k, kv_heads, D]")
    if seqused_k is not None and not paged:
        raise NotImplementedError("seqused_k requires paged KV")
    if not paged and cu_seqlens_k is None:
        raise ValueError("Packed KV requires cu_seqlens_k")
    if paged and (seqused_k is None or block_table.ndim != 2):
        raise ValueError("Paged KV requires seqused_k and a rank-2 block_table")

    total, heads, dim = q.shape
    kv_heads = k.shape[-2]
    if kv_heads == 0 or heads % kv_heads != 0:
        raise ValueError("Query heads must be a positive multiple of KV heads")
    batch = cu_seqlens_q.shape[0] - 1
    if batch < 1:
        raise ValueError("cu_seqlens_q must contain at least one sequence")
    group = heads // kv_heads
    page = k.shape[1] if paged else 1
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    elif out.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("out must be FP16 or BF16")
    elif out.shape != q.shape or out.device != q.device or out.stride(-1) != 1:
        raise ValueError(
            "out must match Q shape and device, with a contiguous last dim"
        )
    lse = (
        torch.empty((heads, total), dtype=torch.float32, device=q.device)
        if return_softmax_lse
        else None
    )
    if total == 0:
        return (out, lse) if return_softmax_lse else out
    _check_descale("q_descale", q_descale, batch, heads, max_seqlen_q, q.device)
    _check_descale("k_descale", k_descale, batch, kv_heads, max_seqlen_k, q.device)
    _check_descale("v_descale", v_descale, batch, kv_heads, max_seqlen_k, q.device)

    left, right = (-1, -1) if window_size is None else window_size
    if group > BLOCK_M.value:
        raise NotImplementedError("KV group larger than the M tile is unsupported")
    block_m = min(BLOCK_M.value, triton.next_power_of_2(max(16, group * max_seqlen_q)))
    # Shared attention helpers need additional UB scratch on the general path.
    block_m = min(block_m, 32) if group <= 32 else block_m
    query_tile = block_m // group
    num_q_tiles = triton.cdiv(max_seqlen_q, query_tile)
    num_tasks = num_q_tiles * batch * kv_heads
    # A split is useful for long decode, where the unsplit grid has too few
    # independent query tiles. Larger N only fits the small M decode tile.
    splits = 4 if max_seqlen_q == 1 and max_seqlen_k >= 512 else 1
    block_n = (
        128
        if splits > 1 and paged and page == 16 and block_m == 16 and dim == 128
        else BLOCK_N.value
    )
    block_n = 32 if block_m > 32 else block_n
    max_blocks = triton.cdiv(max_seqlen_k, block_n)
    alibi_batch_stride = 0
    alibi_head_stride = 0
    alibi_ptr = q_descale
    if alibi_slopes is not None:
        alibi_ptr = alibi_slopes
        if alibi_slopes.ndim == 1:
            alibi_head_stride = alibi_slopes.stride(0)
        else:
            alibi_batch_stride = alibi_slopes.stride(0)
            alibi_head_stride = alibi_slopes.stride(1)
    lse_ptr = (
        lse if lse is not None else torch.empty(1, dtype=torch.float32, device=q.device)
    )
    out_dtype = tl.bfloat16 if out.dtype == torch.bfloat16 else tl.float16
    scale = dim**-0.5 if softmax_scale is None else softmax_scale
    k_token_stride = k.stride(-3)
    v_token_stride = v.stride(-3)
    k_page_stride = k.stride(0) if paged else 0
    v_page_stride = v.stride(0) if paged else 0
    # Split long decode sequences across cores, then merge online-softmax states.
    part_out = (
        torch.empty(
            (num_tasks * splits * block_m, dim), dtype=torch.float32, device=q.device
        )
        if splits > 1
        else out
    )
    part_max = (
        torch.empty(num_tasks * splits * block_m, dtype=torch.float32, device=q.device)
        if splits > 1
        else out
    )
    part_denom = (
        torch.empty(num_tasks * splits * block_m, dtype=torch.float32, device=q.device)
        if splits > 1
        else out
    )
    with torch_device_fn.device(q.device):
        task_begin = 0
        while task_begin < num_tasks * splits:
            count = min(LAUNCH_CHUNK, num_tasks * splits - task_begin)
            _flash_int8_fwd[(count,)](
                q,
                k,
                v,
                out,
                lse_ptr,
                part_out,
                part_max,
                part_denom,
                cu_seqlens_q,
                cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q,
                seqused_k if seqused_k is not None else cu_seqlens_q,
                block_table if block_table is not None else cu_seqlens_q,
                q_descale,
                k_descale,
                v_descale,
                alibi_ptr,
                task_begin,
                batch,
                kv_heads,
                group,
                max_blocks,
                q.stride(0),
                q.stride(1),
                k_token_stride,
                k.stride(-2),
                k_page_stride,
                v_token_stride,
                v.stride(-2),
                v_page_stride,
                out.stride(0),
                out.stride(1),
                q_descale.stride(0),
                q_descale.stride(1),
                q_descale.stride(2),
                k_descale.stride(0),
                k_descale.stride(1),
                k_descale.stride(2),
                v_descale.stride(0),
                v_descale.stride(1),
                v_descale.stride(2),
                block_table.stride(0) if paged else 0,
                alibi_batch_stride,
                alibi_head_stride,
                total,
                scale,
                causal,
                left,
                right,
                softcap,
                alibi_slopes is not None,
                return_softmax_lse,
                D=dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                PAGE=page,
                PAGED=paged,
                OUT_DTYPE=out_dtype,
                SPLITS=splits,
            )
            task_begin += count
        if splits > 1:
            _flash_int8_merge[(num_tasks,)](
                out,
                lse_ptr,
                part_out,
                part_max,
                part_denom,
                cu_seqlens_q,
                batch,
                kv_heads,
                group,
                total,
                out.stride(0),
                out.stride(1),
                D=dim,
                BLOCK_M=block_m,
                SPLITS=splits,
                OUT_DTYPE=out_dtype,
                WRITE_LSE=return_softmax_lse,
            )
    return (out, lse) if return_softmax_lse else out


_HEAD_CAUSAL_MIN_CONTEXT = 3072


_HEAD_UNMASKED_MIN_CONTEXT = 2048


_PACKED_DECODE_MIN_GROUPS = 16


_PACKED_DECODE_MIN_KV = 1024


TILING_FIELDS = (
    "heads",
    "kvHeads",
    "tasks",
    "vectorBlocks",
    "tableStride",
    "pageStride",
    "rowStride",
    "ksBlocks",
    "vsBlocks",
    "cubeBlocks",
    "totalGroups",
    "groupSize",
    "headGroupSize",
    "groupSplits",
    "queryLen",
    "originalQueryLen",
    "queryGroups",
    "hasCuq",
    "causal",
    "qsBlocks",
    "qsStrideB",
    "qsStrideH",
    "qsStrideBlock",
    "ksStrideB",
    "ksStrideH",
    "ksStrideBlock",
    "vsStrideB",
    "vsStrideH",
    "vsStrideBlock",
    "nTiles",
    "scoreBytes",
    "probBytes",
    "pvBytes",
    "alphaBytes",
    "vNzBytes",
    "separateCorrection",
    "corrProbBytes",
    "corrPvBytes",
)


@lru_cache(maxsize=None)
def _bitcode_key(path):
    # File contents, not only the Python registration, affect generated code.
    # The bundle is immutable for the lifetime of the process.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def _supported_device(device):
    # The mixed core mapping have been validated for this physical SKU.
    with torch_device_fn.device(device):
        return "910B4" in torch_device_fn.get_device_name(device)


def _scale_supported(scale, batch, heads, length, device):
    if not isinstance(scale, torch.Tensor):
        return False
    if (
        scale.dtype != torch.float32
        or scale.ndim != 3
        or scale.device != device
        or scale.shape[:2] != (batch, heads)
    ):
        return False
    if length > 0 and (
        scale.shape[2] == 0
        or (scale.stride(2) != 0 and scale.shape[2] < triton.cdiv(length, 128))
    ):
        return False
    # The device tiling uses uint32 strides and offsets. Non-contiguous and
    # broadcast scale views are supported, but their reachable offset must fit.
    if any(stride < 0 or stride >= 2**32 for stride in scale.stride()):
        return False
    return (
        sum(
            max(size - 1, 0) * stride
            for size, stride in zip(scale.shape, scale.stride())
        )
        < 2**32
    )


def _eligible(q, k, v, cuq, table, used, qs, ks, vs, out, maxq, maxk):
    if not all(isinstance(x, torch.Tensor) for x in (q, k, v, cuq, table, used)):
        return False
    if (
        q.device.type != "npu"
        or q.ndim != 3
        or q.shape[2] != 128
        or k.ndim != 4
        or k.shape[1] != 16
        or k.shape[3] != 128
        or v.shape != k.shape
        or table.ndim != 2
        or q.shape[0] == 0
    ):
        return False
    if (
        not isinstance(maxq, int)
        or not isinstance(maxk, int)
        or not 0 < maxq < 2**32
        or not 0 <= maxk < 2**32
    ):
        return False
    batch = table.shape[0]
    if (
        batch < 1
        or cuq.shape != (batch + 1,)
        or used.shape != (batch,)
        or table.shape[1] < triton.cdiv(maxk, 16)
        or q.shape[0] > batch * maxq
    ):
        return False
    heads, kv_heads = q.shape[1], k.shape[2]
    if heads < 1 or kv_heads < 1 or heads % kv_heads:
        return False
    single_head = q.shape[0] == batch * maxq and maxq >= 128
    # Grouped kernels physically pack four Q heads per KV head. The long
    # uniform single-head kernels instead map queryHead // group on device.
    if heads // kv_heads != 4 and not single_head:
        return False
    # C220 Nd2NzParams.srcDValue is uint16. Grouped Q copies individual rows
    # with source stride D; the single-head path copies across token rows.
    if k.stride(1) >= 2**16 or (single_head and q.stride(0) >= 2**16):
        return False
    for tensor in (q, k, v):
        if (
            tensor.dtype != torch.int8
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            return False
    for tensor in (cuq, table, used):
        if (
            tensor.dtype != torch.int32
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            return False
    if max(x.numel() for x in (q, k, v, cuq, table, used)) >= 2**32:
        return False
    if out is not None and (
        not isinstance(out, torch.Tensor)
        or out.dtype != torch.bfloat16
        or out.shape != q.shape
        or out.device != q.device
        or not out.is_contiguous()
    ):
        return False
    for scale, heads, length in (
        (qs, heads, maxq),
        (ks, kv_heads, maxk),
        (vs, kv_heads, maxk),
    ):
        if not _scale_supported(scale, batch, heads, length, q.device):
            return False
    return True


def _metadata(
    fields,
    q,
    k,
    table,
    qs,
    ks,
    vs,
    maxq,
    causal,
    width,
    tile,
    head=False,
    small=False,
    packed=False,
):
    batch = table.shape[0]
    rows = tile if head else tile * 4
    pv_rows = 2 * rows + 1
    pv_m = triton.cdiv(pv_rows, 16) * 16
    query_groups = 2 if small else triton.cdiv(maxq, tile)
    heads, kv_heads = q.shape[1], k.shape[2]
    groups = batch * query_groups * (heads if head else kv_heads)
    blocks = min(20, groups)
    values = dict.fromkeys(fields, 0)
    values.update(
        heads=heads,
        kvHeads=kv_heads,
        tasks=batch * query_groups * tile * heads,
        hasCuq=1,
        vectorBlocks=2 * blocks,
        tableStride=table.shape[1],
        pageStride=k.stride(0),
        rowStride=k.stride(1),
        cubeBlocks=blocks,
        totalGroups=groups,
        groupSize=4,
        headGroupSize=4,
        groupSplits=1,
        queryLen=tile,
        originalQueryLen=maxq,
        queryGroups=query_groups,
        causal=int(causal),
        qsBlocks=qs.shape[2],
        ksBlocks=ks.shape[2],
        vsBlocks=vs.shape[2],
        qsStrideB=qs.stride(0),
        qsStrideH=qs.stride(1),
        qsStrideBlock=qs.stride(2),
        ksStrideB=ks.stride(0),
        ksStrideH=ks.stride(1),
        ksStrideBlock=ks.stride(2),
        vsStrideB=vs.stride(0),
        vsStrideH=vs.stride(1),
        vsStrideBlock=vs.stride(2),
        nTiles=triton.cdiv(table.shape[1], 8),
        scoreBytes=blocks * 2 * rows * width * 4,
        probBytes=blocks * 2 * pv_m * width,
        pvBytes=blocks * 2 * pv_rows * 128 * 4,
    )
    if head:
        # Q128 uses separate high/low PV operations; smaller grouped tiles
        # concatenate both planes and their correction row in one PV matrix.
        high_m = triton.cdiv(rows + 1, 16) * 16
        values["probBytes"] = blocks * 2 * (high_m + rows) * width
        values["pvBytes"] = blocks * 2 * (2 * rows + 1) * 128 * 4
    if width >= 512:
        extra_rows = triton.cdiv(rows // 2, 16) * 16
        values["corrProbBytes"] = blocks * 2 * 2 * extra_rows * width
        values["corrPvBytes"] = blocks * 2 * rows * 128 * 4
    if packed:
        groups = batch * (kv_heads // 4)
        blocks = min(20, groups)
        values.update(
            totalGroups=groups,
            cubeBlocks=blocks,
            vectorBlocks=2 * blocks,
            headGroupSize=16,
            scoreBytes=blocks * 2 * 16 * width * 4,
            probBytes=blocks * 2 * 64 * width,
            pvBytes=blocks * 2 * 36 * 128 * 4,
            corrProbBytes=blocks * 2 * 4 * 16 * width,
            corrPvBytes=blocks * 2 * 4 * 4 * 128 * 4,
        )
    if any(not 0 <= value < 2**32 for value in values.values()):
        return None
    workspace_bytes = (
        values["scoreBytes"]
        + values["probBytes"]
        + values["pvBytes"]
        + values["corrProbBytes"]
        + values["corrPvBytes"]
    )
    workspace_bytes += blocks * 2 * 64 if width >= 512 else 0
    return tuple(values[name] for name in fields), blocks, workspace_bytes


def try_run(q, k, v, maxq, cuq, maxk, table, used, qs, ks, vs, out, causal):
    """Return None for unsupported scope, otherwise return the public output.

    CUQ/used/table *contents* follow the same validity contract as the public
    operator; this path never copies sequence metadata to the CPU to inspect it.
    """
    if not _eligible(q, k, v, cuq, table, used, qs, ks, vs, out, maxq, maxk):
        return None
    if not _supported_device(q.device):
        return None
    width = 256 if vs.stride(2) == 0 or vs.shape[2] == 1 else 128
    uniform = q.shape[0] == table.shape[0] * maxq
    minimum_wide_k = 256 if maxq <= 4 else 512
    width = (
        (1024 if maxk >= 2048 else 512)
        if uniform and maxq <= 8 and maxk > minimum_wide_k and width == 256
        else width
    )
    estimated_visible_keys = maxk - (maxq - 1) / 2 if causal else maxk
    if (
        uniform
        and maxq >= 128
        and width == 256
        and estimated_visible_keys
        >= (_HEAD_CAUSAL_MIN_CONTEXT if causal else _HEAD_UNMASKED_MIN_CONTEXT)
    ):
        width = 512
    # Q128 tails can regress small mixed batches (for example Q=[129, 5]).
    # Restrict the larger tile to long-prefill mixtures validated in benchmarks.
    mixed_head = not uniform and maxq >= 1024 and width == 256
    # Unlike grouped copies, single-head Nd2Nz uses the inter-token Q stride.
    if mixed_head and q.stride(0) >= 2**16:
        return None
    head = (uniform and maxq >= 128) or mixed_head
    partitioned_head = head and uniform and width == 512 and table.shape[0] > 1
    packed_decode = (
        uniform
        and maxq == 1
        and maxk >= _PACKED_DECODE_MIN_KV
        and q.shape[1] == 4 * k.shape[2]
        and k.shape[2] % 4 == 0
        and table.shape[0] * (k.shape[2] // 4) >= _PACKED_DECODE_MIN_GROUPS
        and (vs.stride(2) == 0 or vs.shape[2] == 1)
    )
    if packed_decode:
        width = 512
    if head:
        tile = 128
    elif width >= 512 and not packed_decode:
        tile = 4
    else:
        tile = min(16, triton.next_power_of_2(maxq))
    prepared = _metadata(
        TILING_FIELDS,
        q,
        k,
        table,
        qs,
        ks,
        vs,
        maxq,
        causal,
        width,
        tile,
        head=head,
        packed=packed_decode,
    )
    if prepared is None:
        return None
    metadata, blocks, workspace_bytes = prepared
    mixed = mixed_head or (not uniform and tile == 16 and width == 256)
    small = None
    if mixed or partitioned_head:
        small = _metadata(
            TILING_FIELDS,
            q,
            k,
            table,
            qs,
            ks,
            vs,
            maxq,
            causal,
            256 if partitioned_head else width,
            128 if partitioned_head else 2,
            head=partitioned_head,
            small=not partitioned_head,
        )
        if small is None:
            return None
        workspace_bytes = max(workspace_bytes, small[2])
    if head and width == 512:
        flag_offset = triton.cdiv(workspace_bytes, 64) * 16
        metadata = (*metadata, flag_offset)
        workspace_bytes = flag_offset * 4 + metadata[10] * 2 * 4
    from triton.experimental.tle.language.dsa.ascend.custom_ops.registry import (
        CUSTOM_OPS_BITCODE,
    )

    from ._flash_attn_varlen_int8_cube import (
        launch_cube_tle,
        launch_grouped_tle,
        launch_hybrid_large_tle,
        launch_hybrid_small_tle,
        launch_packed_tle,
    )

    ordinary_head = head and not (partitioned_head or mixed)
    cube_bundle_key = (_bitcode_key(CUSTOM_OPS_BITCODE),)
    small_bundle_key = cube_bundle_key
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    args = (q, k, v, table, used, qs, ks, vs, out, workspace.view(torch.int32), cuq)

    def launch_cube(launch_blocks, launch_width, mode, launch_metadata):
        if launch_width == 128:
            # Uniform queries are required by the N128 Vector path.
            launch_args = (*args[:9], workspace.view(torch.int32), None)
        else:
            launch_args = args
        workspace_i32 = workspace.view(torch.int32)
        if mode == 0:
            launch_kernel = launch_grouped_tle
            launch_args = (
                *args[:9],
                workspace.view(torch.int32),
                cuq,
                cube_bundle_key,
                launch_width,
                mode,
                launch_metadata,
            )
        elif mode == 6:
            launch_kernel = launch_packed_tle
            launch_args = (
                *args[:9],
                workspace.view(torch.int32),
                cuq,
                cube_bundle_key,
                launch_width,
                mode,
                launch_metadata,
            )
        elif mode == 2:
            launch_kernel = launch_hybrid_large_tle
            launch_args = (*args, cube_bundle_key, launch_width, mode, launch_metadata)
        elif mode == 1:
            launch_kernel = launch_hybrid_small_tle
            launch_args = (
                *args[:9],
                workspace.view(torch.int32),
                cuq,
                small_bundle_key,
                launch_width,
                mode,
                launch_metadata,
            )
        else:
            launch_kernel = launch_cube_tle
            launch_args = (
                *launch_args,
                workspace_i32,
                cube_bundle_key,
                launch_width,
                mode,
                launch_metadata,
            )
        launch_kernel[(launch_blocks,)](
            *launch_args,
            disable_auto_inject_block_sync=True,
            multibuffer=False,
            enable_legacy_insert_load_store_for_mix_cv=mode == 1,
            enable_auto_bind_sub_block=True,
            disable_fma=launch_width == 128 or mode == 6,
            enable_ubuf_saving=(
                (
                    mode in (0, 2, 3, 4, 5, 6)
                    and launch_width != 128
                    and not (mode == 0 and launch_width == 1024)
                )
                or (
                    launch_width == 128
                    and not (
                        launch_metadata[15] % 128 == 0
                        and launch_metadata[10] <= launch_blocks
                    )
                )
            ),
        )

        if mode in (3, 4, 5) and launch_width == 512:
            from ._flash_attn_varlen_int8_cube import launch_replay_tle as replay_kernel

            replay_metadata = list(launch_metadata[:38])
            replay_metadata[14] = 32
            replay_metadata[16] = triton.cdiv(replay_metadata[15], 32)
            replay_metadata[10] = (
                replay_metadata[10] // launch_metadata[16] * replay_metadata[16]
            )
            replay_metadata[9] = min(20, replay_metadata[10])
            replay_metadata[3] = 2 * replay_metadata[9]
            replay_metadata[2] = replay_metadata[10] * 32
            rings = 2 * replay_metadata[9]
            replay_metadata[30] = rings * 32 * 512 * 4
            replay_metadata[31] = rings * 80 * 512
            replay_metadata[32] = rings * 65 * 128 * 4
            replay_metadata[36] = rings * 32 * 512
            replay_metadata[37] = rings * 32 * 128 * 4
            replay_metadata = (
                *replay_metadata,
                launch_metadata[38],
                launch_metadata[16],
            )
            replay_kernel[(replay_metadata[9],)](
                *launch_args[:-1],
                replay_metadata,
                disable_auto_inject_block_sync=True,
                multibuffer=False,
                enable_legacy_insert_load_store_for_mix_cv=False,
                enable_auto_bind_sub_block=True,
                disable_fma=False,
                enable_ubuf_saving=True,
            )

    with torch_device_fn.device(q.device):
        if partitioned_head:
            small_metadata, small_blocks, _ = small
            launch_cube(small_blocks, 256, 5, small_metadata)
            launch_cube(blocks, 512, 5, metadata)
        elif mixed:
            small_metadata, small_blocks, _ = small
            launch_cube(small_blocks, width, 1, small_metadata)
            if mixed_head:
                launch_cube(blocks, width, 4, metadata)
            else:
                launch_cube(blocks, width, 2, metadata)
        elif ordinary_head:
            launch_cube(blocks, width, 3, metadata)
        else:
            if packed_decode:
                launch_cube(blocks, width, 6, metadata)
            else:
                launch_cube(blocks, width, 0, metadata)
    return out
