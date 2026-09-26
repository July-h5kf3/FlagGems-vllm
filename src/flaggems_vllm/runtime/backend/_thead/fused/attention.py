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

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import has_triton_tle_attrs, libentry

if has_triton_tle_attrs(("load",), 3, 6, 0):
    try:
        import triton.experimental.tle.language as tle
        from triton._C.libtriton import ppu

        HAS_AIU_K = ppu.passes.ttppugpuir.add_tle_promote_async_load_to_aiu is not None
    except (ImportError, AttributeError):
        HAS_AIU_K = False
        tle = None
else:
    HAS_AIU_K = False
    tle = None


LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)
# Signed INT8 stores 0..255 probability levels with a -128 zero point.
PROB_QUANT_LEVELS = tl.constexpr(255)
# Two signed bytes represent hi * 256 + lo without overflowing the high byte.
PRECISE_PROB_LEVELS = tl.constexpr(127 * 256)
# Descales are indexed by logical sequence position in blocks of DESCALE_BLOCK.
DESCALE_BLOCK = tl.constexpr(128)
PACK_TILE = tl.constexpr(32)
SHORT_QUERY_LIMIT = tl.constexpr(16)
WORKLIST_TILE = tl.constexpr(128)


@triton.jit
def _flash_int8_prepare_worklist(
    CUQ,
    WORK,
    BATCH: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    CAPACITY: tl.constexpr,
    TILE_PITCH: tl.constexpr,
):
    batch = tl.program_id(0)
    batches = tl.arange(0, triton.next_power_of_2(BATCH))
    begins = tl.load(CUQ + batches, batches < BATCH, 0)
    ends = tl.load(CUQ + batches + 1, batches < BATCH, 0)
    lengths = ends - begins
    counts = tl.where(
        (batches < BATCH) & (lengths > SHORT_QUERY_LIMIT),
        tl.cdiv(lengths, QUERY_TILE),
        0,
    )
    offset = tl.sum(tl.where(batches < batch, counts, 0))
    count = tl.sum(tl.where(batches == batch, counts, 0))
    if batch == 0:
        tl.store(WORK + CAPACITY, tl.sum(counts))
    for start in range(tl.cdiv(count, WORKLIST_TILE)):
        tile = start * WORKLIST_TILE + tl.arange(0, WORKLIST_TILE)
        tl.store(WORK + offset + tile, batch * TILE_PITCH + tile, tile < count)


@triton.jit
def _flash_int8_pack_kv(
    K,
    V,
    KP,
    VP,
    TABLE,
    USED,
    CUQ,
    N: tl.constexpr,
    D: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    TS: tl.constexpr,
    PK: tl.constexpr,
    SK: tl.constexpr,
    HK: tl.constexpr,
    PV: tl.constexpr,
    SV: tl.constexpr,
    HV: tl.constexpr,
    LOGICAL: tl.constexpr,
    SKIP_SHORT: tl.constexpr,
):
    n = tl.program_id(0) * PACK_TILE + tl.arange(0, PACK_TILE)
    d = tl.arange(0, D)
    if LOGICAL:
        batch, h = tl.program_id(1), tl.program_id(2)
    else:
        batch, h = 0, tl.program_id(1)
    should_pack = tl.full((), True, tl.int1)
    if SKIP_SHORT:
        nq = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
        should_pack = nq > SHORT_QUERY_LIMIT
    if should_pack:
        if LOGICAL:
            nk = tl.load(USED + batch)
            page = tl.load(TABLE + batch * TS + n // PAGE, n < nk, 0)
        else:
            nk = N
            page = n // PAGE
        k = tl.load(
            K + page[:, None] * PK + (n[:, None] % PAGE) * SK + h * HK + d[None, :],
            n[:, None] < nk,
            0,
        )
        v = tl.load(
            V + page[:, None] * PV + (n[:, None] % PAGE) * SV + h * HV + d[None, :],
            n[:, None] < nk,
            0,
        )
        off = ((batch * HEADS + h) * N + n[:, None]) * D + d[None, :]
        tl.store(KP + off, k, n[:, None] < N)
        tl.store(VP + off, v.to(tl.float16), n[:, None] < N)


@libentry()
@triton.jit
def _flash_int8_fwd(
    Q,
    K,
    V,
    O,
    LSE,
    CUQ,
    CUK,
    USED,
    TABLE,
    WORK,
    QS,
    KS,
    VS,
    ALIBI,
    sq: tl.constexpr,
    hq: tl.constexpr,
    sk: tl.constexpr,
    hk: tl.constexpr,
    pk: tl.constexpr,
    sv: tl.constexpr,
    hv: tl.constexpr,
    pv: tl.constexpr,
    so: tl.constexpr,
    ho: tl.constexpr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    table_stride: tl.constexpr,
    alibi_stride: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    CAUSAL: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    CAP: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    HALF_PV: tl.constexpr,
    PRECISE_PV: tl.constexpr,
    ASYNC_K: tl.constexpr,
    SEPARATE_MASK: tl.constexpr,
    LOGICAL_KV: tl.constexpr,
    BATCH: tl.constexpr,
    COMPACT: tl.constexpr,
    USE_WORKLIST: tl.constexpr,
    WORK_CAPACITY: tl.constexpr,
    TILE_PITCH: tl.constexpr,
    SHORT_ONLY: tl.constexpr,
    FOLD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kv_head = head if FOLD else head // GROUP
    query_tile: tl.constexpr = BM // GROUP if FOLD else BM
    active = tl.full((), True, tl.int1)
    if USE_WORKLIST:
        count = tl.load(WORK + WORK_CAPACITY)
        active = tile < count
        entry = tl.load(WORK + tile, active, 0)
        batch = entry // TILE_PITCH
        tile = entry % TILE_PITCH
    elif COMPACT:
        batches = tl.arange(0, triton.next_power_of_2(BATCH))
        begins = tl.load(CUQ + batches, batches < BATCH, 0)
        ends = tl.load(CUQ + batches + 1, batches < BATCH, 0)
        counts = tl.cdiv(ends - begins, query_tile)
        cumulative = tl.cumsum(counts)
        # The host grid bounds sum(ceil(q_len / query_tile)); excess tiles
        # map past the final request and are skipped by the query-length guard.
        batch = tl.minimum(
            tl.sum(((tile >= cumulative) & (batches < BATCH)).to(tl.int32)), BATCH - 1
        )
        tile -= tl.sum(tl.where(batches < batch, counts, 0))
    q_start = tl.load(CUQ + batch)
    nq = tl.load(CUQ + batch + 1) - q_start
    if SHORT_ONLY:
        active &= nq <= SHORT_QUERY_LIMIT
    if PAGED:
        k_start = 0
        nk = tl.load(USED + batch)
    else:
        k_start = tl.load(CUK + batch)
        nk = tl.load(CUK + batch + 1) - k_start
    if SHORT_ONLY:
        query_iterations = tl.where(active, tl.cdiv(nq, query_tile), 0)
    else:
        query_iterations = 1
    for query_index in range(query_iterations):
        current_tile = query_index if SHORT_ONLY else tile
        rows = current_tile * BM + tl.arange(0, BM)
        m = rows // GROUP if FOLD else rows
        h = kv_head * GROUP + rows % GROUP if FOLD else tl.full((BM,), head, tl.int32)
        d = tl.arange(0, D)
        if active & (current_tile * query_tile < nq):
            q = tl.load(
                Q + (q_start + m[:, None]) * sq + h[:, None] * hq + d[None, :],
                m[:, None] < nq,
                0,
            )
            q_scale = tl.load(
                QS + batch * qs0 + h * qs1 + (m // DESCALE_BLOCK) * qs2, m < nq, 0
            )
            if HALF_PV and not COMPACT and CAP <= 0 and not HAS_ALIBI:
                q_scale = q_scale * (SCALE * LOG2E)
            maximum = tl.full((BM,), float("-inf"), tl.float32)
            denom = tl.full((BM,), 0, tl.float32)
            acc = tl.full((BM, D), 0, tl.float32)
            if HAS_ALIBI:
                slope = tl.load(ALIBI + batch * alibi_stride + h)
            # Skip KV blocks that are masked for every query in this query tile.
            first = 0
            end = nk
            if LEFT >= 0:
                first = tl.maximum(0, current_tile * query_tile + nk - nq - LEFT) // BN
            if CAUSAL:
                end = tl.minimum(end, (current_tile + 1) * query_tile + nk - nq)
            if RIGHT >= 0:
                end = tl.minimum(end, (current_tile + 1) * query_tile + nk - nq + RIGHT)
            # Only the boundary interval needs per-row causal and length masks.
            use_dense: tl.constexpr = SEPARATE_MASK and LEFT < 0 and RIGHT < 0
            end_block = tl.cdiv(tl.maximum(end, 0), BN)
            if use_dense:
                full_keys = nk
                if CAUSAL:
                    full_keys = tl.minimum(nk, current_tile * query_tile + nk - nq + 1)
                full_hi = tl.maximum(full_keys, 0) // BN
            for mask_phase in tl.static_range(2 if use_dense else 1):
                if use_dense:
                    begin_block = first if mask_phase == 0 else full_hi
                    stop_block = full_hi if mask_phase == 0 else end_block
                else:
                    begin_block, stop_block = first, end_block
                for start in range(begin_block, stop_block):
                    n = start * BN + tl.arange(0, BN)
                    if LOGICAL_KV:
                        k_row = batch * pk + n * sk
                        v_row = batch * pv + n * sv
                    elif PAGED:
                        if not use_dense or mask_phase == 1:
                            page = tl.load(
                                TABLE + batch * table_stride + n // PAGE, n < nk, 0
                            )
                        else:
                            page = tl.load(TABLE + batch * table_stride + n // PAGE)
                        k_row = page * pk + (n % PAGE) * sk
                        v_row = page * pv + (n % PAGE) * sv
                    else:
                        k_row = (k_start + n) * sk
                        v_row = (k_start + n) * sv
                    if ASYNC_K and use_dense and mask_phase == 0:
                        # AIU copies contiguous packed K without materializing a
                        # per-element global-load address in every thread.
                        k_ptr = tl.make_block_ptr(
                            K + batch * pk + kv_head * hk,
                            shape=(D, nk),
                            strides=(1, sk),
                            offsets=(0, start * BN),
                            block_shape=(D, BN),
                            order=(0, 1),
                        )
                        k = tle.load(k_ptr, is_async=True)
                    elif not use_dense or mask_phase == 1:
                        k = tl.load(
                            K + k_row[None, :] + kv_head * hk + d[:, None],
                            n[None, :] < nk,
                            0,
                        )
                    else:
                        k = tl.load(K + k_row[None, :] + kv_head * hk + d[:, None])
                    descale_block = start * BN // DESCALE_BLOCK
                    ks = tl.load(KS + batch * ks0 + kv_head * ks1 + descale_block * ks2)
                    vs = tl.load(VS + batch * vs0 + kv_head * vs1 + descale_block * vs2)
                    scores = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
                    if HALF_PV and not COMPACT and CAP <= 0 and not HAS_ALIBI:
                        scores = scores * (q_scale * ks)[:, None]
                    else:
                        scores = scores * (q_scale * ks * SCALE)[:, None]
                    if CAP > 0:
                        scores = CAP * (2 / (1 + tl.exp(2 * (-scores / CAP))) - 1)
                    position = m + nk - nq
                    if HAS_ALIBI:
                        scores -= slope[:, None] * tl.abs(
                            position[:, None] - n[None, :]
                        )
                    if not use_dense or mask_phase == 1:
                        if SEPARATE_MASK:
                            # Invalid query rows are excluded by Q loads and output stores.
                            valid = n[None, :] < nk
                        else:
                            valid = (m[:, None] < nq) & (n[None, :] < nk)
                        if CAUSAL:
                            valid &= n[None, :] <= position[:, None]
                        if LEFT >= 0:
                            valid &= n[None, :] >= position[:, None] - LEFT
                        if RIGHT >= 0:
                            valid &= n[None, :] <= position[:, None] + RIGHT
                    if not (HALF_PV and not COMPACT and CAP <= 0 and not HAS_ALIBI):
                        scores = scores * LOG2E
                    if not use_dense or mask_phase == 1:
                        scores = tl.where(valid, scores, float("-inf"))
                    tile_max = tl.max(scores, 1)
                    new_max = tl.maximum(maximum, tile_max)
                    safe_max = tl.where(new_max == float("-inf"), 0, new_max)
                    alpha = tl.exp2(maximum - safe_max)
                    if HALF_PV:
                        p = tl.exp2(scores - safe_max[:, None])
                        denom = denom * alpha + tl.sum(p, 1)
                    else:
                        # Normalize within this KV tile so probability quantization needs
                        # only a constant multiply. beta brings its sum/PV into the running scale.
                        safe_tile = tl.where(tile_max == float("-inf"), 0, tile_max)
                        p = tl.exp2(scores - safe_tile[:, None])
                        beta = tl.exp2(tile_max - safe_max)
                        denom = denom * alpha + tl.sum(p, 1) * beta
                        if PRECISE_PV:
                            tl.static_assert(BN <= 128)
                            p_scale = beta * (1.0 / PRECISE_PROB_LEVELS)
                            fixed = (p * PRECISE_PROB_LEVELS + 0.5).to(tl.int32)
                            p_hi = ((fixed + 128) >> 8).to(tl.int8)
                            p_lo = fixed.to(tl.int8)
                        else:
                            p_scale = beta * (1.0 / PROB_QUANT_LEVELS)
                            # Nonnegative inputs make truncation round-half-up.
                            p_int8 = (
                                (p * PROB_QUANT_LEVELS + 0.5).to(tl.int32) - 128
                            ).to(tl.int8)
                    if not use_dense or mask_phase == 1:
                        v = tl.load(
                            V + v_row[:, None] + kv_head * hv + d[None, :],
                            n[:, None] < nk,
                            0,
                        )
                    else:
                        v = tl.load(V + v_row[:, None] + kv_head * hv + d[None, :])
                    if HALF_PV:
                        if vs2 == 0:
                            if BM >= 128:
                                # Rescale before materializing P to shorten large-tile live ranges.
                                acc = acc * alpha[:, None]
                                acc = tl.dot(p.to(tl.float16), v.to(tl.float16), acc)
                            else:
                                acc = tl.dot(
                                    p.to(tl.float16),
                                    v.to(tl.float16),
                                    acc * alpha[:, None],
                                )
                        else:
                            partial = tl.dot(
                                p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32
                            )
                            acc = acc * alpha[:, None] + partial * vs
                    else:
                        if PRECISE_PV:
                            # BN <= 128 keeps the combined INT32 result in range.
                            high = tl.dot(p_hi, v, out_dtype=tl.int32)
                            partial = tl.dot(p_lo, v, high << 8, out_dtype=tl.int32).to(
                                tl.float32
                            )
                        else:
                            correction = tl.dot(
                                tl.full((BM, BN), -128, tl.int8), v, out_dtype=tl.int32
                            )
                            partial = tl.dot(
                                p_int8, v, -correction, out_dtype=tl.int32
                            ).to(tl.float32)
                        acc = acc * alpha[:, None] + partial * (p_scale * vs)[:, None]
                    maximum = new_max
            result = acc / tl.where(denom > 0, denom, 1)[:, None]
            if HALF_PV and vs2 == 0:
                v_scale = tl.load(VS + batch * vs0 + kv_head * vs1, nk > 0, 0)
                result *= v_scale
            tl.store(
                O + (q_start + m[:, None]) * so + h[:, None] * ho + d[None, :],
                result,
                m[:, None] < nq,
            )
            if WRITE_LSE:
                lse = tl.where(denom > 0, maximum * LN2 + tl.log(denom), float("inf"))
                tl.store(LSE + h * TOTAL_Q + q_start + m, lse, m < nq)


def flash_attn_varlen_func_w8a8_int8(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """PPU INT8 variable-length attention, inference forward only.

    Q is [total_q, heads, D]; K/V are [total_k, kv_heads, D] or paged
    [pages, page_size, kv_heads, D]. D is 64 or 128, heads must be a multiple
    of kv_heads, and the last dimension must be contiguous. Q/K/V are INT8.
    Required FP32 descales have shape [batch, heads, ceil(max_length / 128)]
    (kv_heads for K/V), indexed by logical sequence position even for paged KV.
    Each real value is the INT8 value multiplied by its descale.

    QK uses INT8 dot with INT32 accumulation. Paged long-query paths pack
    K/V by KV head and convert V to FP16. The D=128, GQA=4 fast path gathers
    logical KV positions while staying within the physical-cache workspace
    budget; other long-query paths preserve physical page indices. When the
    compiler provides PPU AIU loading, dense tiles use asynchronous INT8 K
    copies; masked boundary tiles retain regular loads. Selected
    short-query paths also use FP16 PV, converting V inside the kernel.
    Remaining paths quantize probabilities to 256 levels and use INT8 PV
    with zero-point correction. Single-token D=128, GQA=4 decode with at most
    256 KV positions uses two signed INT8 probability components to reduce
    quantization error. Softmax and online accumulation use FP32.
    Packing requires at most three bytes per element of the physical K cache.
    Output is BF16 by default, or uses the supplied FP16/BF16 out buffer.
    LSE is [heads, total_q], FP32. Fully masked rows return zero and LSE +inf,
    following FlashAttention's convention. Inputs and out must not overlap.
    """
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
    if q.shape[-1] not in (64, 128):
        raise NotImplementedError("Only head dimensions 64 and 128 are supported")
    if seqused_k is not None and block_table is None:
        raise NotImplementedError("seqused_k requires paged KV")

    batch = cu_seqlens_q.numel() - 1
    total, heads, dim = q.shape
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    lse = (
        torch.empty((heads, total), dtype=torch.float32, device=q.device)
        if return_softmax_lse
        else None
    )
    if total == 0:
        return (out, lse) if return_softmax_lse else out
    left, right = (-1, -1) if window_size is None else window_size
    paged = block_table is not None
    group = heads // k.shape[-2]
    kv_heads = k.shape[-2]
    long_query = paged and max_seqlen_q >= 128
    pack_kv = long_query and k.shape[0] > 0
    common_gqa = paged and dim == 128 and group == 4
    # Do not exceed the physical-cache workspace budget for shared KV pages.
    logical_kv = (
        pack_kv
        and common_gqa
        and max_seqlen_k > 0
        and batch * max_seqlen_k <= k.shape[0] * k.shape[1]
    )
    use_aiu_k = logical_kv and HAS_AIU_K
    block_m = 64 if not paged and not causal and max_seqlen_q >= 512 else 16
    fold = paged and group > 1 and group <= 16 and group & (group - 1) == 0
    if fold:
        block_m = 32 if max_seqlen_q > 16 else 16
    block_n = 128 if long_query else 64
    num_warps = 8 if block_m == 64 and dim == 128 else 4
    num_stages = 3 if long_query else 1
    half_pv = long_query
    if logical_kv:
        # A 64-column tile reduces pressure on the long-query accumulator.
        block_m, block_n, num_warps = 64, 64, 4
        fold = max_seqlen_q > 512
        if not use_aiu_k and max_seqlen_q >= 4096 and left < 0 and right < 0:
            block_m = 128
            num_stages = 2 if max_seqlen_q < 8192 else 3
    elif common_gqa and max_seqlen_q <= 16:
        parallel_heads = batch * kv_heads
        if max_seqlen_q > 4:
            # Reuse each KV tile across more short-query rows.
            block_m, block_n, num_stages, half_pv = 32, 128, 3, True
            num_stages = 2 if left < 0 and right < 0 else 3
        elif max_seqlen_q == 1 and parallel_heads <= 64:
            block_n, num_warps, half_pv = 128, 8, True
        elif left < 0 and right < 0:
            # One-token GQA uses only four rows; a smaller tile reduces padding.
            block_m = 8 if max_seqlen_q == 1 else 16
            block_n, num_warps, num_stages = 128, 2, 2
        elif parallel_heads > 128:
            block_n = 128
    query_tile = block_m // group if fold else block_m
    grid_heads = kv_heads if fold else heads
    compact = paged and max_seqlen_q > 16 and total * 2 < batch * max_seqlen_q
    split_queries = logical_kv and compact
    tile_pitch = triton.next_power_of_2(triton.cdiv(max_seqlen_q, query_tile))
    split_queries = split_queries and batch * tile_pitch < 2**31
    if split_queries:
        compact = False
    elif logical_kv and compact:
        num_stages = 2
    work_capacity = triton.cdiv(total, query_tile) + batch - 1
    work = None
    raw_k, raw_v = k, v
    grid = (
        (triton.cdiv(total, query_tile) + batch - 1, 1, grid_heads)
        if compact
        else (triton.cdiv(max_seqlen_q, query_tile), batch, grid_heads)
    )
    with torch_device_fn.device(q.device):
        if split_queries:
            work = torch.empty((work_capacity + 1,), dtype=torch.int32, device=q.device)
            _flash_int8_prepare_worklist[(batch,)](
                cu_seqlens_q,
                work,
                batch,
                query_tile,
                work_capacity,
                tile_pitch,
            )
            grid = (work_capacity, 1, grid_heads)
        if pack_kv:
            if logical_kv:
                n = max_seqlen_k
                packed_shape = (batch, n, kv_heads, dim)
                packed_stride = (kv_heads * n * dim, dim, n * dim, 1)
                pack_grid = (triton.cdiv(n, PACK_TILE.value), batch, kv_heads)
            else:
                n = k.shape[0] * k.shape[1]
                packed_shape = k.shape
                packed_stride = (k.shape[1] * dim, dim, n * dim, 1)
                pack_grid = (triton.cdiv(n, PACK_TILE.value), kv_heads)
            kp = torch.empty_strided(
                packed_shape, packed_stride, dtype=torch.int8, device=k.device
            )
            vp = torch.empty_strided(
                packed_shape, packed_stride, dtype=torch.float16, device=v.device
            )
            _flash_int8_pack_kv[pack_grid](
                k,
                v,
                kp,
                vp,
                block_table,
                seqused_k,
                cu_seqlens_q,
                n,
                dim,
                kv_heads,
                k.shape[1],
                block_table.stride(0),
                *k.stride()[:3],
                *v.stride()[:3],
                LOGICAL=logical_kv,
                SKIP_SHORT=split_queries,
            )
            k, v = kp, vp
        for phase in range(2 if split_queries else 1):
            short_phase = phase == 1
            if short_phase:
                k, v = raw_k, raw_v
                fold, logical_kv, compact = True, False, False
                block_m, num_warps, num_stages, half_pv = 16, 4, 1, False
                # An occupancy hint only; GPU CUQ values still decide which
                # requests execute, including empty and padded requests.
                min_long = triton.cdiv(
                    max(total - batch * SHORT_QUERY_LIMIT.value, 0),
                    max_seqlen_q - SHORT_QUERY_LIMIT.value,
                )
                short_head_bound = max(batch - min_long, 0) * kv_heads
                block_n = 128 if short_head_bound > 128 else 64
                if short_head_bound <= 64:
                    block_n, num_warps, half_pv = 128, 8, True
                elif left < 0 and right < 0:
                    block_n, num_warps, num_stages = 128, 2, 2
                grid = (1, batch, kv_heads)
            _flash_int8_fwd[grid](
                q,
                k,
                v,
                out,
                lse,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                block_table,
                work,
                q_descale,
                k_descale,
                v_descale,
                alibi_slopes,
                q.stride(0),
                q.stride(1),
                k.stride(-3),
                k.stride(-2),
                k.stride(0) if paged else 0,
                v.stride(-3),
                v.stride(-2),
                v.stride(0) if paged else 0,
                out.stride(0),
                out.stride(1),
                *q_descale.stride(),
                *k_descale.stride(),
                *v_descale.stride(),
                block_table.stride(0) if paged else 0,
                (
                    alibi_slopes.stride(0)
                    if alibi_slopes is not None and alibi_slopes.ndim == 2
                    else 0
                ),
                total,
                heads // k.shape[-2],
                dim,
                k.shape[1] if paged else 0,
                paged,
                causal,
                left,
                right,
                softcap,
                dim**-0.5 if softmax_scale is None else softmax_scale,
                alibi_slopes is not None,
                return_softmax_lse,
                HALF_PV=half_pv,
                ASYNC_K=use_aiu_k and not short_phase,
                PRECISE_PV=(
                    common_gqa
                    and max_seqlen_q == 1
                    and max_seqlen_k <= 2 * DESCALE_BLOCK.value
                ),
                SEPARATE_MASK=common_gqa,
                LOGICAL_KV=logical_kv,
                BATCH=batch,
                COMPACT=compact,
                USE_WORKLIST=split_queries and not short_phase,
                WORK_CAPACITY=work_capacity if split_queries and not short_phase else 0,
                TILE_PITCH=tile_pitch if split_queries and not short_phase else 1,
                SHORT_ONLY=short_phase,
                FOLD=fold,
                BM=block_m,
                BN=block_n,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return (out, lse) if return_softmax_lse else out
