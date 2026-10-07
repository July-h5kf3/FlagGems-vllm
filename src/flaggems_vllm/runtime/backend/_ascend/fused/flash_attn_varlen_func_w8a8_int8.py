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


from typing import Optional, Tuple, Union

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl

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
    causal,
    left,
    right,
    softcap,
    has_alibi,
    write_lse,
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
                    page = tl.load(
                        TABLE + batch * table_stride + n_safe // PAGE, n_ok, 0
                    ).to(tl.int64)
                    k_row = page * pk + (n_safe % PAGE) * sk
                    v_row = page * pv + (n_safe % PAGE) * sv
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
                # Keep the divisor nonzero when softcap is disabled.
                cap = tl.where(softcap > 0, softcap, 1.0)
                capped = cap * (2 / (1 + tl.exp(2 * (-scores / cap))) - 1)
                scores = tl.where(softcap > 0, capped, scores)
                position = q_row + nk - nq
                if has_alibi:
                    scores -= slope[:, None] * tl.abs(position[:, None] - n[None, :])
                valid = row_ok[:, None] & n_ok[None, :]
                if causal:
                    valid &= n[None, :] <= position[:, None]
                if left >= 0:
                    valid &= n[None, :] >= position[:, None] - left
                if right >= 0:
                    valid &= n[None, :] <= position[:, None] + right
                scores = tl.where(valid, scores * LOG2E, float("-inf"))
                tile_max = tl.max(scores, 1)
                new_max = tl.maximum(maximum, tile_max)
                safe_max = tl.where(new_max == float("-inf"), 0, new_max)
                alpha = tl.exp2(maximum - safe_max)
                # Quantize exp2 values at the tile max. beta returns them to
                # the running max, so the divisor stays a compile-time constant.
                safe_tile = tl.where(tile_max == float("-inf"), 0, tile_max)
                p = tl.exp2(scores - safe_tile[:, None])
                beta = tl.exp2(tile_max - safe_max)
                denom = denom * alpha + tl.sum(p, 1) * beta
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
                acc = acc * alpha[:, None] + partial * (p_scale * vs)[:, None]
                maximum = new_max
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
        from ._flash_attn_varlen_int8_inline import try_run

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
    # More query tiles improve parallelism on large paged prefill, while
    # keeping the working set below the Ascend UB limit.
    if (
        paged
        and max_seqlen_q >= 128
        and max_seqlen_k >= 256
        and dim == 128
        and group <= 32
    ):
        block_m = min(block_m, 32)
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
