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

import triton
import triton.language as tl

from flaggems_vllm.ops.flash_mla_fp8.common import (
    _TLE_LN2,
    _TLE_LOG2E,
    _TLE_NEG_INF,
    HAS_TLE,
    K_CONTENT_TILE,
    tle,
)
from flaggems_vllm.ops.flash_mla_fp8.dense_query import _fp8_mla_wg0
from flaggems_vllm.ops.flash_mla_fp8.dense_value import (
    _fp8_mla_wg1,
    _fp8_mla_wg1_pretranspose,
)

_TLE_POS_INF = tl.constexpr(float("inf"))

if HAS_TLE:

    @triton.jit
    def _fp8_dense_mla_splitk_partial(
        qc_ptr,
        qr_ptr,
        qs_ptr,
        kc_ptr,
        kr_ptr,
        ks_ptr,
        block_table,
        cache_seqlens,
        split_batch_ptr,
        split_page_begin_ptr,
        split_num_pages_ptr,
        partial_out_ptr,
        partial_lse2_ptr,
        q_desc,
        qr_desc,
        qs_desc,
        out_desc,
        k_desc,
        kr_desc,
        ks_desc,
        stride_qc_b: tl.constexpr,
        stride_qc_h: tl.constexpr,
        stride_qr_b: tl.constexpr,
        stride_qr_h: tl.constexpr,
        stride_qs_b: tl.constexpr,
        stride_qs_h: tl.constexpr,
        stride_kc_blk: tl.constexpr,
        stride_kc_pg: tl.constexpr,
        stride_kr_blk: tl.constexpr,
        stride_kr_pg: tl.constexpr,
        stride_ks_blk: tl.constexpr,
        stride_ks_pg: tl.constexpr,
        stride_bt_b: tl.constexpr,
        stride_bt_pg: tl.constexpr,
        stride_seqlen: tl.constexpr,
        stride_split_batch: tl.constexpr,
        stride_split_begin: tl.constexpr,
        stride_split_num_pages: tl.constexpr,
        stride_po_split: tl.constexpr,
        stride_po_h: tl.constexpr,
        stride_pl_split: tl.constexpr,
        stride_pl_h: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_CKV_BYTES: tl.constexpr,
        Q_ROPE_BYTES: tl.constexpr,
        Q_SCALE_BYTES: tl.constexpr,
        K_CONTENT_TILE_BYTES: tl.constexpr,
        K_ROPE_BYTES: tl.constexpr,
        K_SCALE_BYTES: tl.constexpr,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        RH: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        DP: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        PAGE_GRAIN_TAIL_ZERO: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        FIXED_NUM_PAGES: tl.constexpr,
        DIRECT_LSE: tl.constexpr,
    ):
        """One strict-2WG CTA per (split, head block)."""
        pid = tl.program_id(0)
        global_split = pid // RH
        h_base = (pid % RH) * BH
        global_split64 = global_split.to(tl.int64)
        batch_idx = tl.load(split_batch_ptr + global_split64 * stride_split_batch)
        batch_idx64 = batch_idx.to(tl.int64)
        page_begin = tl.load(split_page_begin_ptr + global_split64 * stride_split_begin)
        split_num_pages_runtime = tl.load(
            split_num_pages_ptr + global_split64 * stride_split_num_pages
        )
        split_num_pages = (
            FIXED_NUM_PAGES
            if USE_TMA_OUTPUT and FIXED_NUM_PAGES > 0 and FIXED_NUM_PAGES <= 10
            else split_num_pages_runtime
        )
        page_end = page_begin + split_num_pages
        full_cache_seqlen = tl.load(cache_seqlens + batch_idx64 * stride_seqlen)
        token_begin = page_begin * PAGE_SIZE
        token_end = tl.minimum(page_end * PAGE_SIZE, full_cache_seqlen)
        split_cache_seqlen = tl.maximum(token_end - token_begin, 0)

        block_table_ptr = (
            block_table
            + batch_idx64 * stride_bt_b
            + page_begin.to(tl.int64) * stride_bt_pg
        )
        out_split_ptr = partial_out_ptr + global_split64 * stride_po_split
        lse2_split_ptr = partial_lse2_ptr + global_split64 * stride_pl_split

        s_q = tle.gpu.alloc([BH, CKV], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_qr = tle.gpu.alloc([BH, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)

        s_kc_a0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_a = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_kc_b0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_b = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_p_a = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_p_b = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_beta_a = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_beta_b = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)

        s_state0_m = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)
        s_state1_m = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)

        # One TMA copy == one completion-barrier generation.
        q_ckv_full = tle.gpu.alloc_barrier(expect_bytes=Q_CKV_BYTES)
        q_rope_full = tle.gpu.alloc_barrier(expect_bytes=Q_ROPE_BYTES)
        q_scale_full = tle.gpu.alloc_barrier(expect_bytes=Q_SCALE_BYTES)
        k_content_full = tle.gpu.alloc_barriers(8, expect_bytes=K_CONTENT_TILE_BYTES)
        k_rope_full = tle.gpu.alloc_barriers(2, expect_bytes=K_ROPE_BYTES)
        k_scale_full = tle.gpu.alloc_barriers(2, expect_bytes=K_SCALE_BYTES)

        # Cross-WG handoffs use named barriers: 128 producer arrivals plus
        # 128 consumer waiters complete each handshake. Per-WG tail-zero
        # synchronization retains its separate phaseful mbarriers.
        initialization_done = tle.gpu.alloc_barrier(arrive_count=1)
        control_barriers = tle.gpu.alloc_barriers(num_barriers=8, arrive_count=1)
        handoff_barriers = tle.gpu.alloc_barriers(num_barriers=8, arrive_count=256)
        state0_ready = handoff_barriers[0]
        state1_ready = handoff_barriers[1]
        # P is stored before V repack.  Publishing v*_ready after the repack
        # therefore certifies both P and V visibility to the remote PV owner.
        p0_ready = handoff_barriers[2]
        p1_ready = handoff_barriers[3]
        v0_ready = handoff_barriers[2]
        v1_ready = handoff_barriers[3]
        slot0_empty = handoff_barriers[4]
        slot1_empty = handoff_barriers[5]

        # CUDA fill_oob_V publishes shared zeros before the LDSM transpose.
        # Reuse one previously idle control mbarrier per compute warp-group;
        # each has one fixed elected writer and a private pair generation.
        tail0_zero_ready = control_barriers[6]
        tail1_zero_ready = control_barriers[7]

        row0 = (batch_idx * HQ + h_base).to(tl.int32)

        # Named objects still initialize temporary shared storage. Join the
        # default WG before warp-specialize captures can reuse those bytes.
        tle.gpu.barrier_arrive(initialization_done, phaseIdx=0)
        tle.gpu.barrier_wait(initialization_done, phaseIdx=0)
        tle.gpu.warp_specialize(
            [
                (
                    _fp8_mla_wg0,
                    (
                        q_desc,
                        qr_desc,
                        qs_desc,
                        out_desc,
                        k_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        tail0_zero_ready,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_a,
                        s_vt0_a,
                        s_vt1_a,
                        s_vt0_b,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        out_split_ptr,
                        lse2_split_ptr,
                        stride_po_h,
                        stride_pl_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        PAGE_GRAIN_TAIL_ZERO,
                        MERGE_STATE_V,
                        False,
                        USE_TMA_OUTPUT,
                        DIRECT_LSE,
                        FIXED_NUM_PAGES,
                    ),
                ),
                (
                    _fp8_mla_wg1,
                    (
                        k_desc,
                        kr_desc,
                        ks_desc,
                        out_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        tail1_zero_ready,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kr_a,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_b,
                        s_vt0_b,
                        s_vt1_b,
                        s_vt1_a,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        out_split_ptr,
                        stride_po_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        PAGE_GRAIN_TAIL_ZERO,
                        MERGE_STATE_V,
                        USE_TMA_OUTPUT,
                        FIXED_NUM_PAGES,
                    ),
                ),
            ],
            [4],
            [255],
        )

    @triton.jit
    def _fp8_dense_mla_splitk_partial_pdl(
        qc_ptr,
        qr_ptr,
        qs_ptr,
        kc_ptr,
        kr_ptr,
        ks_ptr,
        block_table,
        cache_seqlens,
        split_batch_ptr,
        split_page_begin_ptr,
        split_num_pages_ptr,
        partial_out_ptr,
        partial_lse2_ptr,
        q_desc,
        qr_desc,
        qs_desc,
        out_desc,
        k_desc,
        kr_desc,
        ks_desc,
        stride_qc_b: tl.constexpr,
        stride_qc_h: tl.constexpr,
        stride_qr_b: tl.constexpr,
        stride_qr_h: tl.constexpr,
        stride_qs_b: tl.constexpr,
        stride_qs_h: tl.constexpr,
        stride_kc_blk: tl.constexpr,
        stride_kc_pg: tl.constexpr,
        stride_kr_blk: tl.constexpr,
        stride_kr_pg: tl.constexpr,
        stride_ks_blk: tl.constexpr,
        stride_ks_pg: tl.constexpr,
        stride_bt_b: tl.constexpr,
        stride_bt_pg: tl.constexpr,
        stride_seqlen: tl.constexpr,
        stride_split_batch: tl.constexpr,
        stride_split_begin: tl.constexpr,
        stride_split_num_pages: tl.constexpr,
        stride_po_split: tl.constexpr,
        stride_po_h: tl.constexpr,
        stride_pl_split: tl.constexpr,
        stride_pl_h: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_CKV_BYTES: tl.constexpr,
        Q_ROPE_BYTES: tl.constexpr,
        Q_SCALE_BYTES: tl.constexpr,
        K_CONTENT_TILE_BYTES: tl.constexpr,
        K_ROPE_BYTES: tl.constexpr,
        K_SCALE_BYTES: tl.constexpr,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        RH: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        DP: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        PAGE_GRAIN_TAIL_ZERO: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        FIXED_NUM_PAGES: tl.constexpr,
        DIRECT_LSE: tl.constexpr,
    ):
        """One strict-2WG CTA per (split, head block)."""
        pid = tl.program_id(0)
        global_split = pid // RH
        h_base = (pid % RH) * BH
        global_split64 = global_split.to(tl.int64)
        batch_idx = tl.load(split_batch_ptr + global_split64 * stride_split_batch)
        batch_idx64 = batch_idx.to(tl.int64)
        page_begin = tl.load(split_page_begin_ptr + global_split64 * stride_split_begin)
        split_num_pages_runtime = tl.load(
            split_num_pages_ptr + global_split64 * stride_split_num_pages
        )
        split_num_pages = (
            FIXED_NUM_PAGES
            if USE_TMA_OUTPUT and FIXED_NUM_PAGES > 0 and FIXED_NUM_PAGES <= 10
            else split_num_pages_runtime
        )
        page_end = page_begin + split_num_pages
        full_cache_seqlen = tl.load(cache_seqlens + batch_idx64 * stride_seqlen)
        token_begin = page_begin * PAGE_SIZE
        token_end = tl.minimum(page_end * PAGE_SIZE, full_cache_seqlen)
        split_cache_seqlen = tl.maximum(token_end - token_begin, 0)

        block_table_ptr = (
            block_table
            + batch_idx64 * stride_bt_b
            + page_begin.to(tl.int64) * stride_bt_pg
        )
        out_split_ptr = partial_out_ptr + global_split64 * stride_po_split
        lse2_split_ptr = partial_lse2_ptr + global_split64 * stride_pl_split

        s_q = tle.gpu.alloc([BH, CKV], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_qr = tle.gpu.alloc([BH, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)

        s_kc_a0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_a = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_kc_b0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_b = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_p_a = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_p_b = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_beta_a = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_beta_b = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)

        s_state0_m = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)
        s_state1_m = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)

        # One TMA copy == one completion-barrier generation.
        q_ckv_full = tle.gpu.alloc_barrier(expect_bytes=Q_CKV_BYTES)
        q_rope_full = tle.gpu.alloc_barrier(expect_bytes=Q_ROPE_BYTES)
        q_scale_full = tle.gpu.alloc_barrier(expect_bytes=Q_SCALE_BYTES)
        k_content_full = tle.gpu.alloc_barriers(8, expect_bytes=K_CONTENT_TILE_BYTES)
        k_rope_full = tle.gpu.alloc_barriers(2, expect_bytes=K_ROPE_BYTES)
        k_scale_full = tle.gpu.alloc_barriers(2, expect_bytes=K_SCALE_BYTES)

        # Cross-WG handoffs use named barriers; per-WG tail-zero operations
        # retain phaseful mbarriers. Both compute partitions have 128 threads.
        initialization_done = tle.gpu.alloc_barrier(arrive_count=1)
        control_barriers = tle.gpu.alloc_barriers(num_barriers=8, arrive_count=1)
        handoff_barriers = tle.gpu.alloc_barriers(num_barriers=8, arrive_count=256)
        state0_ready = handoff_barriers[0]
        state1_ready = handoff_barriers[1]
        p0_ready = handoff_barriers[2]
        p1_ready = handoff_barriers[3]
        v0_ready = handoff_barriers[2]
        v1_ready = handoff_barriers[3]
        slot0_empty = handoff_barriers[4]
        slot1_empty = handoff_barriers[5]

        tail0_zero_ready = control_barriers[6]
        tail1_zero_ready = control_barriers[7]

        row0 = (batch_idx * HQ + h_base).to(tl.int32)

        # Retire temporary initialization before capture-mailbox reuse.
        tle.gpu.barrier_arrive(initialization_done, phaseIdx=0)
        tle.gpu.barrier_wait(initialization_done, phaseIdx=0)
        tle.gpu.warp_specialize(
            [
                (
                    _fp8_mla_wg0,
                    (
                        q_desc,
                        qr_desc,
                        qs_desc,
                        out_desc,
                        k_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        tail0_zero_ready,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_a,
                        s_vt0_a,
                        s_vt1_a,
                        s_vt0_b,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        out_split_ptr,
                        lse2_split_ptr,
                        stride_po_h,
                        stride_pl_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        PAGE_GRAIN_TAIL_ZERO,
                        MERGE_STATE_V,
                        True,
                        USE_TMA_OUTPUT,
                        DIRECT_LSE,
                        FIXED_NUM_PAGES,
                    ),
                ),
                (
                    _fp8_mla_wg1,
                    (
                        k_desc,
                        kr_desc,
                        ks_desc,
                        out_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        tail1_zero_ready,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kr_a,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_b,
                        s_vt0_b,
                        s_vt1_b,
                        s_vt1_a,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        out_split_ptr,
                        stride_po_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        PAGE_GRAIN_TAIL_ZERO,
                        MERGE_STATE_V,
                        USE_TMA_OUTPUT,
                        FIXED_NUM_PAGES,
                    ),
                ),
            ],
            [4],
            [255],
        )

    @triton.jit
    def _fp8_dense_mla_splitk_partial_pretranspose(
        qc_ptr,
        qr_ptr,
        qs_ptr,
        kc_ptr,
        kr_ptr,
        ks_ptr,
        block_table,
        cache_seqlens,
        split_batch_ptr,
        split_page_begin_ptr,
        split_num_pages_ptr,
        partial_out_ptr,
        partial_lse2_ptr,
        q_desc,
        qr_desc,
        qs_desc,
        out_desc,
        k_desc,
        kr_desc,
        ks_desc,
        stride_qc_b: tl.constexpr,
        stride_qc_h: tl.constexpr,
        stride_qr_b: tl.constexpr,
        stride_qr_h: tl.constexpr,
        stride_qs_b: tl.constexpr,
        stride_qs_h: tl.constexpr,
        stride_kc_blk: tl.constexpr,
        stride_kc_pg: tl.constexpr,
        stride_kr_blk: tl.constexpr,
        stride_kr_pg: tl.constexpr,
        stride_ks_blk: tl.constexpr,
        stride_ks_pg: tl.constexpr,
        stride_bt_b: tl.constexpr,
        stride_bt_pg: tl.constexpr,
        stride_seqlen: tl.constexpr,
        stride_split_batch: tl.constexpr,
        stride_split_begin: tl.constexpr,
        stride_split_num_pages: tl.constexpr,
        stride_po_split: tl.constexpr,
        stride_po_h: tl.constexpr,
        stride_pl_split: tl.constexpr,
        stride_pl_h: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_CKV_BYTES: tl.constexpr,
        Q_ROPE_BYTES: tl.constexpr,
        Q_SCALE_BYTES: tl.constexpr,
        K_CONTENT_TILE_BYTES: tl.constexpr,
        K_ROPE_BYTES: tl.constexpr,
        K_SCALE_BYTES: tl.constexpr,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        RH: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        DP: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        PAGE_GRAIN_TAIL_ZERO: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        FIXED_NUM_PAGES: tl.constexpr,
        DIRECT_LSE: tl.constexpr,
    ):
        """One strict-2WG CTA per (split, head block)."""
        pid = tl.program_id(0)
        global_split = pid // RH
        h_base = (pid % RH) * BH
        global_split64 = global_split.to(tl.int64)
        batch_idx = tl.load(split_batch_ptr + global_split64 * stride_split_batch)
        batch_idx64 = batch_idx.to(tl.int64)
        page_begin = tl.load(split_page_begin_ptr + global_split64 * stride_split_begin)
        split_num_pages_runtime = tl.load(
            split_num_pages_ptr + global_split64 * stride_split_num_pages
        )
        split_num_pages = (
            FIXED_NUM_PAGES
            if USE_TMA_OUTPUT and FIXED_NUM_PAGES > 0 and FIXED_NUM_PAGES <= 10
            else split_num_pages_runtime
        )
        page_end = page_begin + split_num_pages
        full_cache_seqlen = tl.load(cache_seqlens + batch_idx64 * stride_seqlen)
        token_begin = page_begin * PAGE_SIZE
        token_end = tl.minimum(page_end * PAGE_SIZE, full_cache_seqlen)
        split_cache_seqlen = tl.maximum(token_end - token_begin, 0)

        block_table_ptr = (
            block_table
            + batch_idx64 * stride_bt_b
            + page_begin.to(tl.int64) * stride_bt_pg
        )
        out_split_ptr = partial_out_ptr + global_split64 * stride_po_split
        lse2_split_ptr = partial_lse2_ptr + global_split64 * stride_pl_split

        s_q = tle.gpu.alloc([BH, CKV], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_qr = tle.gpu.alloc([BH, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)

        s_kc_a0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_a = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_kc_b0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_b = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_p_a = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_p_b = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_beta_a = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_beta_b = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)

        s_state0_m = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)
        s_state1_m = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)

        # One TMA copy == one completion-barrier generation.
        q_ckv_full = tle.gpu.alloc_barrier(expect_bytes=Q_CKV_BYTES)
        q_rope_full = tle.gpu.alloc_barrier(expect_bytes=Q_ROPE_BYTES)
        q_scale_full = tle.gpu.alloc_barrier(expect_bytes=Q_SCALE_BYTES)
        k_content_full = tle.gpu.alloc_barriers(8, expect_bytes=K_CONTENT_TILE_BYTES)
        k_rope_full = tle.gpu.alloc_barriers(2, expect_bytes=K_ROPE_BYTES)
        k_scale_full = tle.gpu.alloc_barriers(2, expect_bytes=K_SCALE_BYTES)

        # Cross-WG handoffs use named barriers; per-WG tail-zero operations
        # retain phaseful mbarriers. Both compute partitions have 128 threads.
        initialization_done = tle.gpu.alloc_barrier(arrive_count=1)
        control_barriers = tle.gpu.alloc_barriers(num_barriers=8, arrive_count=1)
        handoff_barriers = tle.gpu.alloc_barriers(num_barriers=8, arrive_count=256)
        state0_ready = handoff_barriers[0]
        state1_ready = handoff_barriers[1]
        # P is stored before V repack.  Publishing v*_ready after the repack
        # therefore certifies both P and V visibility to the remote PV owner.
        p0_ready = handoff_barriers[2]
        p1_ready = handoff_barriers[3]
        v0_ready = handoff_barriers[2]
        v1_ready = handoff_barriers[3]
        slot0_empty = handoff_barriers[4]
        slot1_empty = handoff_barriers[5]

        tail0_zero_ready = control_barriers[6]
        tail1_zero_ready = control_barriers[7]

        row0 = (batch_idx * HQ + h_base).to(tl.int32)

        # Retire temporary initialization before capture-mailbox reuse.
        tle.gpu.barrier_arrive(initialization_done, phaseIdx=0)
        tle.gpu.barrier_wait(initialization_done, phaseIdx=0)
        tle.gpu.warp_specialize(
            [
                (
                    _fp8_mla_wg0,
                    (
                        q_desc,
                        qr_desc,
                        qs_desc,
                        out_desc,
                        k_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        tail0_zero_ready,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_a,
                        s_vt0_a,
                        s_vt1_a,
                        s_vt0_b,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        out_split_ptr,
                        lse2_split_ptr,
                        stride_po_h,
                        stride_pl_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        PAGE_GRAIN_TAIL_ZERO,
                        MERGE_STATE_V,
                        False,
                        USE_TMA_OUTPUT,
                        DIRECT_LSE,
                        FIXED_NUM_PAGES,
                    ),
                ),
                (
                    _fp8_mla_wg1_pretranspose,
                    (
                        k_desc,
                        kr_desc,
                        ks_desc,
                        out_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kr_a,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_b,
                        s_vt0_b,
                        s_vt1_b,
                        s_vt1_a,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        out_split_ptr,
                        stride_po_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        MERGE_STATE_V,
                        USE_TMA_OUTPUT,
                        FIXED_NUM_PAGES,
                    ),
                ),
            ],
            [4],
            [255],
        )

    @triton.jit
    def _triton_fp8_splitk_combine_kernel(
        partial_out_ptr,
        partial_lse2_ptr,
        num_splits_ptr,
        out_ptr,
        lse_ptr,
        stride_po_split,
        stride_po_h,
        stride_pl_split,
        stride_pl_h,
        stride_ns,
        stride_out_b,
        stride_out_h,
        stride_lse_b,
        stride_lse_h,
        HQ: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        d_block = tl.program_id(1)
        batch_idx = row // HQ
        head_idx = row % HQ
        split_begin = tl.load(num_splits_ptr + batch_idx * stride_ns)
        split_end = tl.load(num_splits_ptr + (batch_idx + 1) * stride_ns)
        split_count = split_end - split_begin

        split_lanes = tl.arange(0, BLOCK_SPLITS)
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < DV

        max_lse2 = _TLE_NEG_INF
        for split_base in tl.range(0, split_count, BLOCK_SPLITS):
            local_split = split_base + split_lanes
            mask_split = local_split < split_count
            global_split = split_begin + local_split
            local_lse2 = tl.load(
                partial_lse2_ptr
                + global_split * stride_pl_split
                + head_idx * stride_pl_h,
                mask=mask_split,
                other=_TLE_NEG_INF,
            )
            finite_lse = (
                mask_split & (local_lse2 > _TLE_NEG_INF) & (local_lse2 < _TLE_POS_INF)
            )
            local_lse2 = tl.where(finite_lse, local_lse2, _TLE_NEG_INF)
            max_lse2 = tl.maximum(max_lse2, tl.max(local_lse2, axis=0))

        finite_max = max_lse2 != _TLE_NEG_INF
        safe_max = tl.where(finite_max, max_lse2, 0.0)
        denom = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for split_base in tl.range(0, split_count, BLOCK_SPLITS):
            local_split = split_base + split_lanes
            mask_split = local_split < split_count
            global_split = split_begin + local_split
            local_lse2 = tl.load(
                partial_lse2_ptr
                + global_split * stride_pl_split
                + head_idx * stride_pl_h,
                mask=mask_split,
                other=_TLE_NEG_INF,
            )
            finite_lse = (
                mask_split & (local_lse2 > _TLE_NEG_INF) & (local_lse2 < _TLE_POS_INF)
            )
            local_lse2 = tl.where(finite_lse, local_lse2, _TLE_NEG_INF)
            weights = tl.where(
                finite_lse,
                tl.exp2(local_lse2 - safe_max),
                0.0,
            )
            partial = tl.load(
                partial_out_ptr
                + global_split[:, None] * stride_po_split
                + head_idx * stride_po_h
                + offs_d[None, :],
                mask=mask_split[:, None] & mask_d[None, :],
                other=0.0,
            )
            partial = tl.where(finite_lse[:, None], partial, 0.0)
            denom += tl.sum(weights, axis=0)
            acc += tl.sum(partial * weights[:, None], axis=0)

        valid = finite_max & (denom > 0.0)
        safe_denom = tl.where(valid, denom, 1.0)
        result = tl.where(valid, acc / safe_denom, 0.0)
        tl.store(
            out_ptr + batch_idx * stride_out_b + head_idx * stride_out_h + offs_d,
            result,
            mask=mask_d,
        )

        global_lse = tl.where(
            valid,
            (safe_max + tl.log(safe_denom) * _TLE_LOG2E) * _TLE_LN2,
            _TLE_NEG_INF,
        )
        tl.store(
            lse_ptr + batch_idx * stride_lse_b + head_idx * stride_lse_h,
            global_lse,
            mask=d_block == 0,
        )

    @triton.jit
    def _triton_fp8_cuda_coarse_combine_kernel_impl(
        partial_out_ptr,
        partial_lse2_ptr,
        num_splits_ptr,
        out_ptr,
        lse_ptr,
        stride_po_split,
        stride_po_h,
        stride_pl_split,
        stride_pl_h,
        stride_ns,
        stride_out_b,
        stride_out_h,
        stride_lse_b,
        stride_lse_h,
        HQ: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        ENABLE_PDL: tl.constexpr,
    ):
        """CUDA-aligned combine: one warp owns one row, eight rows per CTA."""
        # A PDL consumer may become resident before the partial grid retires.
        # No workspace or split metadata read is legal before this wait.
        if ENABLE_PDL:
            tl.extra.cuda.gdc_wait()

        batch_idx = tl.program_id(0)
        row_block = tl.program_id(1)
        heads = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        mask_h = heads < HQ

        split_begin = tl.load(num_splits_ptr + batch_idx * stride_ns)
        split_end = tl.load(num_splits_ptr + (batch_idx + 1) * stride_ns)
        split_count = split_end - split_begin

        split_lanes = tl.arange(0, BLOCK_SPLITS)
        max_lse2 = tl.full((BLOCK_ROWS,), _TLE_NEG_INF, tl.float32)
        for split_base in tl.range(0, split_count, BLOCK_SPLITS):
            local_splits = split_base + split_lanes
            mask_split = local_splits < split_count
            global_splits = split_begin + local_splits
            local_lse2 = tl.load(
                partial_lse2_ptr
                + global_splits[None, :] * stride_pl_split
                + heads[:, None] * stride_pl_h,
                mask=mask_h[:, None] & mask_split[None, :],
                other=_TLE_NEG_INF,
            )
            finite_lse = (
                mask_h[:, None]
                & mask_split[None, :]
                & (local_lse2 > _TLE_NEG_INF)
                & (local_lse2 < _TLE_POS_INF)
            )
            local_lse2 = tl.where(finite_lse, local_lse2, _TLE_NEG_INF)
            max_lse2 = tl.maximum(max_lse2, tl.max(local_lse2, axis=1))

        finite_max = mask_h & (max_lse2 != _TLE_NEG_INF)
        safe_max = tl.where(finite_max, max_lse2, 0.0)
        denom = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
        offs_d = tl.arange(0, DV)
        acc = tl.zeros((BLOCK_ROWS, DV), dtype=tl.float32)

        # Match CUDA split-order accumulation. One warp owns one logical row
        # and each lane owns DV/32 output columns.
        for local_split in tl.range(0, split_count):
            global_split = split_begin + local_split
            local_lse2 = tl.load(
                partial_lse2_ptr + global_split * stride_pl_split + heads * stride_pl_h,
                mask=mask_h,
                other=_TLE_NEG_INF,
            )
            finite_lse = (
                mask_h & (local_lse2 > _TLE_NEG_INF) & (local_lse2 < _TLE_POS_INF)
            )
            weights = tl.where(
                finite_lse,
                tl.exp2(local_lse2 - safe_max),
                0.0,
            )
            partial = tl.load(
                partial_out_ptr
                + global_split * stride_po_split
                + heads[:, None] * stride_po_h
                + offs_d[None, :],
                mask=mask_h[:, None],
                other=0.0,
            )
            denom += weights
            acc += partial * weights[:, None]

        valid = finite_max & (denom > 0.0)
        safe_denom = tl.where(valid, denom, 1.0)
        result = tl.where(valid[:, None], acc / safe_denom[:, None], 0.0)
        tl.store(
            out_ptr
            + batch_idx * stride_out_b
            + heads[:, None] * stride_out_h
            + offs_d[None, :],
            result,
            mask=mask_h[:, None],
        )

        global_lse = tl.where(
            valid,
            (safe_max + tl.log(safe_denom) * _TLE_LOG2E) * _TLE_LN2,
            _TLE_NEG_INF,
        )
        tl.store(
            lse_ptr + batch_idx * stride_lse_b + heads * stride_lse_h,
            global_lse,
            mask=mask_h,
        )

    @triton.jit
    def _triton_fp8_cuda_coarse_combine_kernel(
        partial_out_ptr,
        partial_lse2_ptr,
        num_splits_ptr,
        out_ptr,
        lse_ptr,
        stride_po_split,
        stride_po_h,
        stride_pl_split,
        stride_pl_h,
        stride_ns,
        stride_out_b,
        stride_out_h,
        stride_lse_b,
        stride_lse_h,
        HQ: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Run the non-PDL coarse combine path."""
        _triton_fp8_cuda_coarse_combine_kernel_impl(
            partial_out_ptr,
            partial_lse2_ptr,
            num_splits_ptr,
            out_ptr,
            lse_ptr,
            stride_po_split,
            stride_po_h,
            stride_pl_split,
            stride_pl_h,
            stride_ns,
            stride_out_b,
            stride_out_h,
            stride_lse_b,
            stride_lse_h,
            HQ,
            DV,
            BLOCK_SPLITS,
            BLOCK_ROWS,
            False,
        )

    @triton.jit
    def _triton_fp8_cuda_coarse_combine_pdl_kernel(
        partial_out_ptr,
        partial_lse2_ptr,
        num_splits_ptr,
        out_ptr,
        lse_ptr,
        stride_po_split,
        stride_po_h,
        stride_pl_split,
        stride_pl_h,
        stride_ns,
        stride_out_b,
        stride_out_h,
        stride_lse_b,
        stride_lse_h,
        HQ: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """PDL consumer entrypoint; the wait is inlined before all reads."""
        _triton_fp8_cuda_coarse_combine_kernel_impl(
            partial_out_ptr,
            partial_lse2_ptr,
            num_splits_ptr,
            out_ptr,
            lse_ptr,
            stride_po_split,
            stride_po_h,
            stride_pl_split,
            stride_pl_h,
            stride_ns,
            stride_out_b,
            stride_out_h,
            stride_lse_b,
            stride_lse_h,
            HQ,
            DV,
            BLOCK_SPLITS,
            BLOCK_ROWS,
            True,
        )

    @triton.jit
    def _triton_fp8_single_split_lse_finalize_kernel(
        partial_lse2_ptr,
        lse_ptr,
        stride_pl_split: tl.constexpr,
        stride_pl_h: tl.constexpr,
        stride_lse_b: tl.constexpr,
        stride_lse_h: tl.constexpr,
        HQ: tl.constexpr,
        TOTAL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Convert one-split log2 LSE to the public natural-log convention."""
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL
        batch_idx = offsets // HQ
        head_idx = offsets - batch_idx * HQ
        lse2 = tl.load(
            partial_lse2_ptr
            + batch_idx.to(tl.int64) * stride_pl_split
            + head_idx.to(tl.int64) * stride_pl_h,
            mask=mask,
            other=0.0,
        )
        tl.store(
            lse_ptr
            + batch_idx.to(tl.int64) * stride_lse_b
            + head_idx.to(tl.int64) * stride_lse_h,
            lse2 * _TLE_LN2,
            mask=mask,
        )

else:
    _fp8_dense_mla_splitk_partial = None
    _fp8_dense_mla_splitk_partial_pdl = None
    _fp8_dense_mla_splitk_partial_pretranspose = None
    _triton_fp8_splitk_combine_kernel = None
    _triton_fp8_cuda_coarse_combine_kernel_impl = None
    _triton_fp8_cuda_coarse_combine_kernel = None
    _triton_fp8_cuda_coarse_combine_pdl_kernel = None
    _triton_fp8_single_split_lse_finalize_kernel = None
