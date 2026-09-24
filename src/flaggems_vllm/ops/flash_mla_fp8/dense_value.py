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
    _TLE_FP8_MAX,
    _TLE_LOG2E,
    _TLE_NEG_INF,
    _TLE_P_AMAX_FLOOR,
    HAS_TLE,
    K_CONTENT_TILE,
    tle,
)
from flaggems_vllm.ops.flash_mla_fp8.layout import (
    _cuda_vtranspose_fp8_64x128,
    _publish_p_fp8_sw64_cuda_native_coupled_stmatrix,
    _zero_invalid_fp8_rows_sw128_x4,
)

if HAS_TLE:

    @triton.jit
    def _fp8_mla_wg1(
        k_desc,
        kr_desc,
        ks_desc,
        out_desc,
        block_table,
        stride_bt_pg,
        row0,
        num_pages,
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
        out_ptr,
        stride_po_h,
        h_base,
        softmax_scale,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        DP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        PAGE_GRAIN_TAIL_ZERO: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        KNOWN_NUM_PAGES: tl.constexpr,
    ):
        """WG1: odd-page math and the right output half."""
        if KNOWN_NUM_PAGES > 0:
            # This is a host-certified logical page count, not a token mask.
            tl.assume(num_pages == KNOWN_NUM_PAGES)
        s_state1_m_row = s_state1_m.slot(0)
        s_beta_a_row = s_beta_a.slot(0)
        s_beta_b_row = s_beta_b.slot(0)
        tle.gpu.barrier_wait(q_ckv_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_rope_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_scale_full, phaseIdx=0)

        offs_t = tl.arange(0, BK)
        offs_h = h_base + tl.arange(0, BH)
        mask_h = offs_h < HQ
        state_idx = tl.arange(0, BH)
        qs = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), volatile=True)

        acc_right = tl.zeros((BH, DP), dtype=tl.float32)
        state_m = tl.full((BH,), float("-inf"), tl.float32)
        state_s = tl.full((BH,), 1.0, tl.float32)
        state_l = tl.zeros((BH,), dtype=tl.float32)
        state_valid = tl.zeros((BH,), dtype=tl.int32) != 0

        q_rows_d128 = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, K_CONTENT_TILE))
        q_c0_cols = tl.broadcast_to(
            tl.arange(0, K_CONTENT_TILE)[None, :], (BH, K_CONTENT_TILE)
        )
        q_c1_cols = tl.broadcast_to(
            (K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c2_cols = tl.broadcast_to(
            (2 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c3_cols = tl.broadcast_to(
            (3 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c0 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c0_cols)))
        q_c1 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c1_cols)))
        q_c2 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c2_cols)))
        q_c3 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c3_cols)))
        k_a_c2 = s_kc_a2
        k_a_c3 = s_kc_a3
        k_b_c0 = s_kc_b0
        k_b_c1 = s_kc_b1
        k_b_c2 = s_kc_b2
        k_b_c3 = s_kc_b3
        prow = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, BK))
        pcol = tl.broadcast_to(tl.arange(0, BK)[None, :], (BH, BK))
        kv_rows_d128 = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DP // 2))
        kv_c0_cols = tl.broadcast_to(tl.arange(0, DP // 2)[None, :], (BK, DP // 2))
        kv_c1_cols = tl.broadcast_to(
            (DP // 2 + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        kv_c2_cols = tl.broadcast_to(
            (DP + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        kv_c3_cols = tl.broadcast_to(
            (DP + DP // 2 + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        vt_c0_rows = tl.broadcast_to(tl.arange(0, DP // 2)[:, None], (DP // 2, BK))
        vt_c1_rows = tl.broadcast_to(
            (DP // 2 + tl.arange(0, DP // 2))[:, None], (DP // 2, BK)
        )
        vt_cols_d128 = tl.broadcast_to(tl.arange(0, BK)[None, :], (DP // 2, BK))

        num_pairs = (num_pages + 1) // 2
        # WG1 completes generation zero for both slots. The writer groups use
        # disjoint slices and independent completion barriers.
        if num_pages > 0:
            first_phys = tl.load(block_table)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_a_c2,
                [BK, K_CONTENT_TILE],
                [first_base, 2 * K_CONTENT_TILE],
                barrier=k_content_full[2],
            )
            tle.gpu.copy(
                k_desc,
                k_a_c3,
                [BK, K_CONTENT_TILE],
                [first_base, 3 * K_CONTENT_TILE],
                barrier=k_content_full[3],
            )
            tle.gpu.copy(
                kr_desc,
                s_kr_a,
                [BK, ROPE],
                [first_base, 0],
                barrier=k_rope_full[0],
            )
            tle.gpu.copy(
                ks_desc,
                s_beta_a,
                [1, BK],
                [first_phys, 0],
                barrier=k_scale_full[0],
            )
        if num_pages > 1:
            first_phys = tl.load(block_table + stride_bt_pg)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_b_c2,
                [BK, K_CONTENT_TILE],
                [first_base, 2 * K_CONTENT_TILE],
                barrier=k_content_full[6],
            )
            tle.gpu.copy(
                k_desc,
                k_b_c3,
                [BK, K_CONTENT_TILE],
                [first_base, 3 * K_CONTENT_TILE],
                barrier=k_content_full[7],
            )
            tle.gpu.copy(
                kr_desc,
                s_kr_b,
                [BK, ROPE],
                [first_base, 0],
                barrier=k_rope_full[1],
            )
            tle.gpu.copy(
                ks_desc,
                s_beta_b,
                [1, BK],
                [first_phys, 0],
                barrier=k_scale_full[1],
            )

        # Cold prime: page 1 QK, scale, and V become loop live-ins. No page-1
        # QK is repeated in pair zero.
        qk = tl.zeros((BH, BK), dtype=tl.float32)
        ks = tl.zeros((BK,), dtype=tl.float32)
        if num_pages > 1:
            tle.gpu.barrier_wait(k_content_full[4], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c0, k_b_c0, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[5], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c1, k_b_c1, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[6], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c2, k_b_c2, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[7], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c3, k_b_c3, qk, trans_b=True)
            tle.gpu.barrier_wait(k_rope_full[1], phaseIdx=0)
            qk = tle.gpu.wgmma(s_qr, s_kr_b, qk, trans_b=True)
            qk = tle.gpu.wgmma_wait(0, qk)

            tle.gpu.barrier_wait(k_scale_full[1], phaseIdx=0)
            prime_valid = PAGE_SIZE + offs_t < split_cache_seqlen
            ks_raw = tl.load(tle.gpu.local_ptr(s_beta_b_row, (offs_t,)))
            ks = ks_raw if FULL_TAIL else tl.where(prime_valid, ks_raw, 0.0)

        full_pairs = tl.maximum(num_pages // 2 - 1, 0)
        for pair in tl.range(full_pairs, disable_licm=True):
            even_page = pair * 2
            odd_page = even_page + 1

            if MERGE_STATE_V:
                # The odd-page V repack reads only
                # this WG's already-waited K content (k_content_full[4..7]
                # retired by the QK chain that produced the resident qk) and
                # its prior-generation readers retired through slot1_empty
                # (WG0 PV, waited last iteration) and this WG's own
                # wgmma_wait.  It does not depend on WG0's incoming state, P,
                # or V, so issue it before the merged completion wait and
                # remove it from the wait->v1_ready critical path.  The
                # v1_ready arrive below still follows every one of these
                # shared writes in program order.
                _cuda_vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, FULL_TAIL)

                # The merged completion is intentionally later than the old
                # state-only publication.  Hide part of that wait with the
                # page-local score work, which depends only on the resident
                # QK accumulator and scales, not on WG0's incoming state.
                if FULL_TAIL:
                    valid = tl.full((BK,), True, tl.int1)
                else:
                    valid = odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                valid_row = valid[None, :]
                score = qk * qs[:, None] * ks[None, :] * softmax_scale
                score_safe = score if FULL_TAIL else tl.where(valid_row, score, 0.0)
                x = score_safe * _TLE_LOG2E
                page_m = tl.max(
                    x if FULL_TAIL else tl.where(valid_row, x, _TLE_NEG_INF),
                    axis=1,
                )
                tle.gpu.barrier_wait(v0_ready)
            else:
                tle.gpu.barrier_wait(state0_ready)
            state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
            state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
            state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
            state_valid = tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0

            beta1 = tl.full((BH,), 1.0, tl.float32)
            if True:
                # Preserve the same schedule for every non-merged specialization.  MERGE_STATE_V is constexpr, so only one
                # copy of this page-local chain survives lowering.
                if not MERGE_STATE_V:
                    if FULL_TAIL:
                        valid = tl.full((BK,), True, tl.int1)
                    else:
                        valid = odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                    valid_row = valid[None, :]
                    score = qk * qs[:, None] * ks[None, :] * softmax_scale
                    score_safe = score if FULL_TAIL else tl.where(valid_row, score, 0.0)
                    x = score_safe * _TLE_LOG2E
                    page_m = tl.max(
                        x if FULL_TAIL else tl.where(valid_row, x, _TLE_NEG_INF),
                        axis=1,
                    )
                old_m = tl.where(state_valid, state_m, _TLE_NEG_INF)
                old_s = tl.where(state_valid, state_s, 1.0)
                old_l = tl.where(state_valid, state_l, 0.0)
                m_new = tl.maximum(old_m, page_m)
                m_safe = tl.where(m_new == _TLE_NEG_INF, 0.0, m_new)
                e = (
                    tl.exp2(x - m_safe[:, None])
                    if FULL_TAIL
                    else tl.where(valid_row, tl.exp2(x - m_safe[:, None]), 0.0)
                )
                f = e * ks[None, :]
                amax = tl.max(tl.abs(f), axis=1)
                s_new = tl.where(
                    amax == 0.0,
                    1.0,
                    tl.maximum(amax, _TLE_P_AMAX_FLOOR) / _TLE_FP8_MAX,
                )
                page_valid = (
                    True if FULL_TAIL else odd_page * PAGE_SIZE < split_cache_seqlen
                )
                if USE_HOTLOOP_RECIP:
                    inv_s_new = 1.0 / s_new
                    p_scaled = f * inv_s_new[:, None]
                else:
                    p_scaled = f / s_new[:, None]
                p_new = tl.clamp(p_scaled, -_TLE_FP8_MAX, _TLE_FP8_MAX)
                p1 = (
                    p_new
                    if FULL_TAIL
                    else tl.where(page_valid, p_new, tl.zeros_like(p_new))
                )
                if FULL_TAIL:
                    _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p_b, p1)
                else:
                    p1_store = p_new.to(tl.float8e4nv)
                    p1_store = tl.where(page_valid, p1_store, tl.zeros_like(p1_store))
                    tl.store(tle.gpu.local_ptr(s_p_b, (prow, pcol)), p1_store)
                old_m_finite = tl.where(state_valid, old_m, 0.0)
                alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
                if USE_HOTLOOP_RECIP:
                    beta1 = alpha * old_s * inv_s_new
                    l_new = old_l * beta1 + tl.sum(e, axis=1) * inv_s_new
                else:
                    beta1 = alpha * old_s / s_new
                    l_new = old_l * beta1 + tl.sum(e, axis=1) / s_new
                state_m = tl.where(page_valid, m_new, old_m)
                state_s = tl.where(page_valid, s_new, old_s)
                state_l = tl.where(page_valid, l_new, old_l)
                beta1 = tl.where(page_valid, beta1, 1.0)
                state_valid = state_valid | page_valid

                tl.store(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)), beta1)
                tl.store(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), state_m)
                tl.store(tle.gpu.local_ptr(s_state1_s, (state_idx,)), state_s)
                tl.store(tle.gpu.local_ptr(s_state1_l, (state_idx,)), state_l)
                tl.store(
                    tle.gpu.local_ptr(s_state1_valid, (state_idx,)),
                    state_valid.to(tl.int32),
                )

                # Publish WG1 state before V repack/PV/next-QK, matching the
                # CUDA scale/state hand-off rather than delaying the consumer
                # behind unrelated work.
                if not MERGE_STATE_V:
                    tle.gpu.barrier_arrive(state1_ready)

                # full_pairs excludes the residual/tail pair.  The steady odd
                # page is therefore complete and can use CUDA's single
                # LDSM/PRMT/STSM path without a runtime fallback branch.
                # The merged-state specialization moves this repack before its
                # completion wait; all other specializations keep it here.
                if not MERGE_STATE_V:
                    _cuda_vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, FULL_TAIL)

                tle.gpu.barrier_arrive(v1_ready)

            # CUDA remote-P wait point for the current even page.
            if not MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready)
            beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
            acc_right *= beta0[:, None]
            acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)

            # These K/RoPE/scale reads have retired; PV uses distinct P/V
            # buffers. Issue p+2 transfers now, then drain PV before release.
            next_even_page = even_page + 2
            if True:
                next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
                next_even_base = (next_even_phys * BK).to(tl.int32)
                tle.gpu.copy(
                    k_desc,
                    k_a_c2,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 2 * K_CONTENT_TILE],
                    barrier=k_content_full[2],
                )
                tle.gpu.copy(
                    k_desc,
                    k_a_c3,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 3 * K_CONTENT_TILE],
                    barrier=k_content_full[3],
                )
                tle.gpu.copy(
                    kr_desc,
                    s_kr_a,
                    [BK, ROPE],
                    [next_even_base, 0],
                    barrier=k_rope_full[0],
                )
                tle.gpu.copy(
                    ks_desc,
                    s_beta_a,
                    [1, BK],
                    [next_even_phys, 0],
                    barrier=k_scale_full[0],
                )
            acc_right = tle.gpu.wgmma_wait(0, acc_right)
            tle.gpu.barrier_arrive(slot0_empty)

            next_qk = tl.zeros((BH, BK), dtype=tl.float32)
            next_ks = tl.zeros((BK,), dtype=tl.float32)
            if True:
                # CUDA local-P PV and wait0 precede p+3 upper transactions.
                acc_right *= beta1[:, None]
                acc_right = tle.gpu.wgmma(s_p_b, s_vt1_b, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)

                next_odd_page = odd_page + 2
                next_generation = pair + 1
                if True:
                    next_odd_phys = tl.load(block_table + next_odd_page * stride_bt_pg)
                    next_odd_base = (next_odd_phys * BK).to(tl.int32)
                    tle.gpu.copy(
                        k_desc,
                        k_b_c2,
                        [BK, K_CONTENT_TILE],
                        [next_odd_base, 2 * K_CONTENT_TILE],
                        barrier=k_content_full[6],
                    )
                    tle.gpu.copy(
                        k_desc,
                        k_b_c3,
                        [BK, K_CONTENT_TILE],
                        [next_odd_base, 3 * K_CONTENT_TILE],
                        barrier=k_content_full[7],
                    )
                    tle.gpu.copy(
                        kr_desc,
                        s_kr_b,
                        [BK, ROPE],
                        [next_odd_base, 0],
                        barrier=k_rope_full[1],
                    )

                    # CUDA QK phase-1 completes p+3 in this pair.
                    tle.gpu.barrier_wait(k_content_full[4], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c0, k_b_c0, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[5], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c1, k_b_c1, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[6], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c2, k_b_c2, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[7], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c3, k_b_c3, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_rope_full[1], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(s_qr, s_kr_b, next_qk, trans_b=True)
                    next_qk = tle.gpu.wgmma_wait(0, next_qk)

                tle.gpu.barrier_wait(slot1_empty)

                if True:
                    # Keep the scale copy after slot release to preserve its storage lifetime.
                    next_scale_phys = tl.load(
                        block_table + next_odd_page * stride_bt_pg
                    )
                    tle.gpu.copy(
                        ks_desc,
                        s_beta_b,
                        [1, BK],
                        [next_scale_phys, 0],
                        barrier=k_scale_full[1],
                    )
                    tle.gpu.barrier_wait(k_scale_full[1], phaseIdx=next_generation)
                    next_valid = next_odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                    next_ks_raw = tl.load(tle.gpu.local_ptr(s_beta_b_row, (offs_t,)))
                    next_ks = (
                        next_ks_raw
                        if FULL_TAIL
                        else tl.where(next_valid, next_ks_raw, 0.0)
                    )

            qk = next_qk
            ks = next_ks

        # CUDA-style WG1 epilogue. The first residual pair is either the last
        # full pair or the 3-page transition; an odd transition has one final
        # even-only pair after it.
        if num_pages > 0:
            pair = full_pairs
            even_page = pair * 2
            odd_page = even_page + 1

            if MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready)
            else:
                tle.gpu.barrier_wait(state0_ready)
            state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
            state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
            state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
            state_valid = tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0

            beta1 = tl.full((BH,), 1.0, tl.float32)
            if odd_page < num_pages:
                valid = odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                valid_row = valid[None, :]
                tail_ks_raw = tl.load(tle.gpu.local_ptr(s_beta_b_row, (offs_t,)))
                tail_ks = (
                    tail_ks_raw if FULL_TAIL else tl.where(valid, tail_ks_raw, 0.0)
                )
                score = qk * qs[:, None] * tail_ks[None, :] * softmax_scale
                score_safe = score if FULL_TAIL else tl.where(valid_row, score, 0.0)
                x = score_safe * _TLE_LOG2E
                page_m = tl.max(
                    x if FULL_TAIL else tl.where(valid_row, x, _TLE_NEG_INF), axis=1
                )
                old_m = tl.where(state_valid, state_m, _TLE_NEG_INF)
                old_s = tl.where(state_valid, state_s, 1.0)
                old_l = tl.where(state_valid, state_l, 0.0)
                m_new = tl.maximum(old_m, page_m)
                m_safe = tl.where(m_new == _TLE_NEG_INF, 0.0, m_new)
                e = (
                    tl.exp2(x - m_safe[:, None])
                    if FULL_TAIL
                    else tl.where(valid_row, tl.exp2(x - m_safe[:, None]), 0.0)
                )
                f = e * tail_ks[None, :]
                amax = tl.max(tl.abs(f), axis=1)
                s_new = tl.where(
                    amax == 0.0,
                    1.0,
                    tl.maximum(amax, _TLE_P_AMAX_FLOOR) / _TLE_FP8_MAX,
                )
                page_valid = (
                    True if FULL_TAIL else odd_page * PAGE_SIZE < split_cache_seqlen
                )
                inv_s_new = 1.0 / s_new
                p_new = tl.clamp(f * inv_s_new[:, None], -_TLE_FP8_MAX, _TLE_FP8_MAX)
                p1 = (
                    p_new
                    if FULL_TAIL
                    else tl.where(page_valid, p_new, tl.zeros_like(p_new))
                )
                if FULL_TAIL:
                    _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p_b, p1)
                else:
                    p1_store = p_new.to(tl.float8e4nv)
                    p1_store = tl.where(page_valid, p1_store, tl.zeros_like(p1_store))
                    tl.store(tle.gpu.local_ptr(s_p_b, (prow, pcol)), p1_store)
                old_m_finite = tl.where(state_valid, old_m, 0.0)
                alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
                beta1 = alpha * old_s * inv_s_new
                l_new = old_l * beta1 + tl.sum(e, axis=1) * inv_s_new
                state_m = tl.where(page_valid, m_new, old_m)
                state_s = tl.where(page_valid, s_new, old_s)
                state_l = tl.where(page_valid, l_new, old_l)
                beta1 = tl.where(page_valid, beta1, 1.0)
                state_valid = state_valid | page_valid

                tl.store(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)), beta1)
                tl.store(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), state_m)
                tl.store(tle.gpu.local_ptr(s_state1_s, (state_idx,)), state_s)
                tl.store(tle.gpu.local_ptr(s_state1_l, (state_idx,)), state_l)
                tl.store(
                    tle.gpu.local_ptr(s_state1_valid, (state_idx,)),
                    state_valid.to(tl.int32),
                )

                # Tail generation follows the same last-write publication rule.
                if not MERGE_STATE_V:
                    tle.gpu.barrier_arrive(state1_ready)

                # As on the even-page owner, invalid P columns are exact zero,
                # so a masked V transpose is unnecessary for PV correctness.
                if PAGE_GRAIN_TAIL_ZERO:
                    if not FULL_TAIL:
                        valid_tokens = tl.minimum(
                            split_cache_seqlen - odd_page * PAGE_SIZE,
                            BK,
                        )
                        valid_tokens = tl.maximum(valid_tokens, 0)
                        if valid_tokens < BK:
                            _zero_invalid_fp8_rows_sw128_x4(
                                s_kc_b0,
                                s_kc_b1,
                                s_kc_b2,
                                s_kc_b3,
                                valid_tokens,
                            )
                            tle.gpu.barrier_arrive(tail1_zero_ready, phaseIdx=pair)
                            tle.gpu.barrier_wait(tail1_zero_ready, phaseIdx=pair)
                    _cuda_vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, FULL_TAIL)
                elif FULL_TAIL or (odd_page + 1) * PAGE_SIZE <= split_cache_seqlen:
                    _cuda_vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, FULL_TAIL)
                else:
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b0, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt0_b, (vt_c0_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b1, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt0_b, (vt_c1_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b2, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt1_b, (vt_c0_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b3, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt1_b, (vt_c1_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                tle.gpu.barrier_arrive(v1_ready)

            if not MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready)
            beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
            acc_right *= beta0[:, None]
            acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
            acc_right = tle.gpu.wgmma_wait(0, acc_right)

            next_even_page = even_page + 2
            if next_even_page < num_pages:
                next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
                next_even_base = (next_even_phys * BK).to(tl.int32)
                tle.gpu.copy(
                    k_desc,
                    k_a_c2,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 2 * K_CONTENT_TILE],
                    barrier=k_content_full[2],
                )
                tle.gpu.copy(
                    k_desc,
                    k_a_c3,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 3 * K_CONTENT_TILE],
                    barrier=k_content_full[3],
                )
                tle.gpu.copy(
                    kr_desc,
                    s_kr_a,
                    [BK, ROPE],
                    [next_even_base, 0],
                    barrier=k_rope_full[0],
                )
                tle.gpu.copy(
                    ks_desc,
                    s_beta_a,
                    [1, BK],
                    [next_even_phys, 0],
                    barrier=k_scale_full[0],
                )
            tle.gpu.barrier_arrive(slot0_empty)

            if odd_page < num_pages:
                acc_right *= beta1[:, None]
                acc_right = tle.gpu.wgmma(s_p_b, s_vt1_b, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)
                tle.gpu.barrier_wait(slot1_empty)

            if next_even_page < num_pages:
                final_pair = pair + 1
                if MERGE_STATE_V:
                    tle.gpu.barrier_wait(v0_ready)
                else:
                    tle.gpu.barrier_wait(state0_ready)
                state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
                state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
                state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
                state_valid = (
                    tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0
                )
                if not MERGE_STATE_V:
                    tle.gpu.barrier_wait(v0_ready)
                beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
                acc_right *= beta0[:, None]
                acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)
                tle.gpu.barrier_arrive(slot0_empty)

        offs_d = tl.arange(0, DP)
        l_div = tl.where(state_l > 0.0, state_l, 1.0)
        inv_l_div = 1.0 / l_div
        out_right = tl.where(state_valid[:, None], acc_right * inv_l_div[:, None], 0.0)
        if USE_TMA_OUTPUT:
            out_right_lo, out_right_hi = tl.split(
                tl.permute(tl.reshape(out_right, (BH, 2, DP // 2)), (0, 2, 1))
            )
            out_right_0, out_right_1 = tl.split(
                tl.permute(tl.reshape(out_right_lo, (BH, 2, ROPE)), (0, 2, 1))
            )
            out_right_2, out_right_3 = tl.split(
                tl.permute(tl.reshape(out_right_hi, (BH, 2, ROPE)), (0, 2, 1))
            )
            tile_rows = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, ROPE))
            tile_cols = tl.broadcast_to(tl.arange(0, ROPE)[None, :], (BH, ROPE))
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_0.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP])
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_1.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + ROPE])
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_2.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + 2 * ROPE])
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_3.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + 3 * ROPE])
        else:
            tl.store(
                out_ptr + offs_h[:, None] * stride_po_h + DP + offs_d[None, :],
                out_right,
                mask=mask_h[:, None],
            )

    @triton.jit
    def _fp8_mla_wg1_pretranspose(
        k_desc,
        kr_desc,
        ks_desc,
        out_desc,
        block_table,
        stride_bt_pg,
        row0,
        num_pages,
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
        out_ptr,
        stride_po_h,
        h_base,
        softmax_scale,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        DP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        KNOWN_NUM_PAGES: tl.constexpr,
    ):
        """WG1: odd-page math and the right output half."""
        if KNOWN_NUM_PAGES > 0:
            # This is a host-certified logical page count, not a token mask.
            tl.assume(num_pages == KNOWN_NUM_PAGES)
        s_state1_m_row = s_state1_m.slot(0)
        s_beta_a_row = s_beta_a.slot(0)
        s_beta_b_row = s_beta_b.slot(0)
        tle.gpu.barrier_wait(q_ckv_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_rope_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_scale_full, phaseIdx=0)

        offs_t = tl.arange(0, BK)
        offs_h = h_base + tl.arange(0, BH)
        mask_h = offs_h < HQ
        state_idx = tl.arange(0, BH)
        qs = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), volatile=True)

        acc_right = tl.zeros((BH, DP), dtype=tl.float32)
        state_m = tl.full((BH,), float("-inf"), tl.float32)
        state_s = tl.full((BH,), 1.0, tl.float32)
        state_l = tl.zeros((BH,), dtype=tl.float32)
        state_valid = tl.zeros((BH,), dtype=tl.int32) != 0

        q_rows_d128 = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, K_CONTENT_TILE))
        q_c0_cols = tl.broadcast_to(
            tl.arange(0, K_CONTENT_TILE)[None, :], (BH, K_CONTENT_TILE)
        )
        q_c1_cols = tl.broadcast_to(
            (K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c2_cols = tl.broadcast_to(
            (2 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c3_cols = tl.broadcast_to(
            (3 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c0 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c0_cols)))
        q_c1 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c1_cols)))
        q_c2 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c2_cols)))
        q_c3 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c3_cols)))
        k_a_c2 = s_kc_a2
        k_a_c3 = s_kc_a3
        k_b_c0 = s_kc_b0
        k_b_c1 = s_kc_b1
        k_b_c2 = s_kc_b2
        k_b_c3 = s_kc_b3
        prow = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, BK))
        pcol = tl.broadcast_to(tl.arange(0, BK)[None, :], (BH, BK))
        kv_rows_d128 = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DP // 2))
        kv_c0_cols = tl.broadcast_to(tl.arange(0, DP // 2)[None, :], (BK, DP // 2))
        kv_c1_cols = tl.broadcast_to(
            (DP // 2 + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        kv_c2_cols = tl.broadcast_to(
            (DP + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        kv_c3_cols = tl.broadcast_to(
            (DP + DP // 2 + tl.arange(0, DP // 2))[None, :], (BK, DP // 2)
        )
        vt_c0_rows = tl.broadcast_to(tl.arange(0, DP // 2)[:, None], (DP // 2, BK))
        vt_c1_rows = tl.broadcast_to(
            (DP // 2 + tl.arange(0, DP // 2))[:, None], (DP // 2, BK)
        )
        vt_cols_d128 = tl.broadcast_to(tl.arange(0, BK)[None, :], (DP // 2, BK))

        num_pairs = (num_pages + 1) // 2
        # WG1 completes generation zero for both slots. The writer groups use
        # disjoint slices and independent completion barriers.
        if num_pages > 0:
            first_phys = tl.load(block_table)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_a_c2,
                [BK, K_CONTENT_TILE],
                [first_base, 2 * K_CONTENT_TILE],
                barrier=k_content_full[2],
            )
            tle.gpu.copy(
                k_desc,
                k_a_c3,
                [BK, K_CONTENT_TILE],
                [first_base, 3 * K_CONTENT_TILE],
                barrier=k_content_full[3],
            )
            tle.gpu.copy(
                kr_desc,
                s_kr_a,
                [BK, ROPE],
                [first_base, 0],
                barrier=k_rope_full[0],
            )
            tle.gpu.copy(
                ks_desc,
                s_beta_a,
                [1, BK],
                [first_phys, 0],
                barrier=k_scale_full[0],
            )
        if num_pages > 1:
            first_phys = tl.load(block_table + stride_bt_pg)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_b_c2,
                [BK, K_CONTENT_TILE],
                [first_base, 2 * K_CONTENT_TILE],
                barrier=k_content_full[6],
            )
            tle.gpu.copy(
                k_desc,
                k_b_c3,
                [BK, K_CONTENT_TILE],
                [first_base, 3 * K_CONTENT_TILE],
                barrier=k_content_full[7],
            )
            tle.gpu.copy(
                kr_desc,
                s_kr_b,
                [BK, ROPE],
                [first_base, 0],
                barrier=k_rope_full[1],
            )
            tle.gpu.copy(
                ks_desc,
                s_beta_b,
                [1, BK],
                [first_phys, 0],
                barrier=k_scale_full[1],
            )

        # Cold prime: page 1 QK, scale, and V become loop live-ins. No page-1
        # QK is repeated in pair zero.
        qk = tl.zeros((BH, BK), dtype=tl.float32)
        ks = tl.zeros((BK,), dtype=tl.float32)
        if num_pages > 1:
            tle.gpu.barrier_wait(k_content_full[4], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c0, k_b_c0, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[5], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c1, k_b_c1, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[6], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c2, k_b_c2, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[7], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c3, k_b_c3, qk, trans_b=True)
            tle.gpu.barrier_wait(k_rope_full[1], phaseIdx=0)
            qk = tle.gpu.wgmma(s_qr, s_kr_b, qk, trans_b=True)
            qk = tle.gpu.wgmma_wait(0, qk)

            tle.gpu.barrier_wait(k_scale_full[1], phaseIdx=0)
            prime_valid = PAGE_SIZE + offs_t < split_cache_seqlen
            ks_raw = tl.load(tle.gpu.local_ptr(s_beta_b_row, (offs_t,)))
            ks = ks_raw if FULL_TAIL else tl.where(prime_valid, ks_raw, 0.0)

        full_pairs = tl.maximum(num_pages // 2 - 1, 0)
        for pair in tl.range(full_pairs, disable_licm=True):
            even_page = pair * 2
            odd_page = even_page + 1

            # V1 is independent of WG0's state payload.  Execute useful
            # transpose work while WG0 completes state0; keep publication after
            # P1 so the v1_ready payload/happens-before edge is unchanged.
            _cuda_vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, FULL_TAIL)
            _cuda_vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, FULL_TAIL)
            _cuda_vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, FULL_TAIL)
            _cuda_vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, FULL_TAIL)

            if MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready)
            else:
                tle.gpu.barrier_wait(state0_ready)
            state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
            state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
            state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
            state_valid = tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0

            beta1 = tl.full((BH,), 1.0, tl.float32)
            if True:
                if FULL_TAIL:
                    valid = tl.full((BK,), True, tl.int1)
                else:
                    valid = odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                valid_row = valid[None, :]
                score = qk * qs[:, None] * ks[None, :] * softmax_scale
                score_safe = score if FULL_TAIL else tl.where(valid_row, score, 0.0)
                x = score_safe * _TLE_LOG2E
                page_m = tl.max(
                    x if FULL_TAIL else tl.where(valid_row, x, _TLE_NEG_INF), axis=1
                )
                old_m = tl.where(state_valid, state_m, _TLE_NEG_INF)
                old_s = tl.where(state_valid, state_s, 1.0)
                old_l = tl.where(state_valid, state_l, 0.0)
                m_new = tl.maximum(old_m, page_m)
                m_safe = tl.where(m_new == _TLE_NEG_INF, 0.0, m_new)
                e = (
                    tl.exp2(x - m_safe[:, None])
                    if FULL_TAIL
                    else tl.where(valid_row, tl.exp2(x - m_safe[:, None]), 0.0)
                )
                f = e * ks[None, :]
                amax = tl.max(tl.abs(f), axis=1)
                s_new = tl.where(
                    amax == 0.0,
                    1.0,
                    tl.maximum(amax, _TLE_P_AMAX_FLOOR) / _TLE_FP8_MAX,
                )
                page_valid = (
                    True if FULL_TAIL else odd_page * PAGE_SIZE < split_cache_seqlen
                )
                if USE_HOTLOOP_RECIP:
                    inv_s_new = 1.0 / s_new
                    p_scaled = f * inv_s_new[:, None]
                else:
                    p_scaled = f / s_new[:, None]
                p_new = tl.clamp(p_scaled, -_TLE_FP8_MAX, _TLE_FP8_MAX)
                p1 = (
                    p_new
                    if FULL_TAIL
                    else tl.where(page_valid, p_new, tl.zeros_like(p_new))
                )
                if FULL_TAIL:
                    _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p_b, p1)
                else:
                    p1_store = p_new.to(tl.float8e4nv)
                    p1_store = tl.where(page_valid, p1_store, tl.zeros_like(p1_store))
                    tl.store(tle.gpu.local_ptr(s_p_b, (prow, pcol)), p1_store)
                old_m_finite = tl.where(state_valid, old_m, 0.0)
                alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
                if USE_HOTLOOP_RECIP:
                    beta1 = alpha * old_s * inv_s_new
                    l_new = old_l * beta1 + tl.sum(e, axis=1) * inv_s_new
                else:
                    beta1 = alpha * old_s / s_new
                    l_new = old_l * beta1 + tl.sum(e, axis=1) / s_new
                state_m = tl.where(page_valid, m_new, old_m)
                state_s = tl.where(page_valid, s_new, old_s)
                state_l = tl.where(page_valid, l_new, old_l)
                beta1 = tl.where(page_valid, beta1, 1.0)
                state_valid = state_valid | page_valid

                tl.store(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)), beta1)
                tl.store(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), state_m)
                tl.store(tle.gpu.local_ptr(s_state1_s, (state_idx,)), state_s)
                tl.store(tle.gpu.local_ptr(s_state1_l, (state_idx,)), state_l)
                tl.store(
                    tle.gpu.local_ptr(s_state1_valid, (state_idx,)),
                    state_valid.to(tl.int32),
                )

                # Publish WG1 state before V repack/PV/next-QK, matching the
                # CUDA scale/state hand-off rather than delaying the consumer
                # behind unrelated work.
                if not MERGE_STATE_V:
                    tle.gpu.barrier_arrive(state1_ready)

                tle.gpu.barrier_arrive(v1_ready)

            # CUDA remote-P wait point for the current even page.
            if not MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready)
            beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
            acc_right *= beta0[:, None]
            acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
            acc_right = tle.gpu.wgmma_wait(0, acc_right)

            # After remote-P wait0, issue p+2 content2/3/rope/scale.
            next_even_page = even_page + 2
            if True:
                next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
                next_even_base = (next_even_phys * BK).to(tl.int32)
                tle.gpu.copy(
                    k_desc,
                    k_a_c2,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 2 * K_CONTENT_TILE],
                    barrier=k_content_full[2],
                )
                tle.gpu.copy(
                    k_desc,
                    k_a_c3,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 3 * K_CONTENT_TILE],
                    barrier=k_content_full[3],
                )
                tle.gpu.copy(
                    kr_desc,
                    s_kr_a,
                    [BK, ROPE],
                    [next_even_base, 0],
                    barrier=k_rope_full[0],
                )
                tle.gpu.copy(
                    ks_desc,
                    s_beta_a,
                    [1, BK],
                    [next_even_phys, 0],
                    barrier=k_scale_full[0],
                )
            tle.gpu.barrier_arrive(slot0_empty)

            next_qk = tl.zeros((BH, BK), dtype=tl.float32)
            next_ks = tl.zeros((BK,), dtype=tl.float32)
            if True:
                # CUDA local-P PV and wait0 precede p+3 upper transactions.
                acc_right *= beta1[:, None]
                acc_right = tle.gpu.wgmma(s_p_b, s_vt1_b, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)

                next_odd_page = odd_page + 2
                next_generation = pair + 1
                if True:
                    next_odd_phys = tl.load(block_table + next_odd_page * stride_bt_pg)
                    next_odd_base = (next_odd_phys * BK).to(tl.int32)
                    tle.gpu.copy(
                        k_desc,
                        k_b_c2,
                        [BK, K_CONTENT_TILE],
                        [next_odd_base, 2 * K_CONTENT_TILE],
                        barrier=k_content_full[6],
                    )
                    tle.gpu.copy(
                        k_desc,
                        k_b_c3,
                        [BK, K_CONTENT_TILE],
                        [next_odd_base, 3 * K_CONTENT_TILE],
                        barrier=k_content_full[7],
                    )
                    tle.gpu.copy(
                        kr_desc,
                        s_kr_b,
                        [BK, ROPE],
                        [next_odd_base, 0],
                        barrier=k_rope_full[1],
                    )

                    # CUDA QK phase-1 completes p+3 in this pair.
                    tle.gpu.barrier_wait(k_content_full[4], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c0, k_b_c0, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[5], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c1, k_b_c1, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[6], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c2, k_b_c2, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[7], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c3, k_b_c3, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_rope_full[1], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(s_qr, s_kr_b, next_qk, trans_b=True)
                    next_qk = tle.gpu.wgmma_wait(0, next_qk)

                tle.gpu.barrier_wait(slot1_empty)

                if True:
                    # Keep the scale copy after slot release to preserve its storage lifetime.
                    next_scale_phys = tl.load(
                        block_table + next_odd_page * stride_bt_pg
                    )
                    tle.gpu.copy(
                        ks_desc,
                        s_beta_b,
                        [1, BK],
                        [next_scale_phys, 0],
                        barrier=k_scale_full[1],
                    )
                    tle.gpu.barrier_wait(k_scale_full[1], phaseIdx=next_generation)
                    next_valid = next_odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                    next_ks_raw = tl.load(tle.gpu.local_ptr(s_beta_b_row, (offs_t,)))
                    next_ks = (
                        next_ks_raw
                        if FULL_TAIL
                        else tl.where(next_valid, next_ks_raw, 0.0)
                    )

            qk = next_qk
            ks = next_ks

        # CUDA-style WG1 epilogue. The first residual pair is either the last
        # full pair or the 3-page transition; an odd transition has one final
        # even-only pair after it.
        if num_pages > 0:
            pair = full_pairs
            even_page = pair * 2
            odd_page = even_page + 1

            if MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready)
            else:
                tle.gpu.barrier_wait(state0_ready)
            state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
            state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
            state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
            state_valid = tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0

            beta1 = tl.full((BH,), 1.0, tl.float32)
            if odd_page < num_pages:
                valid = odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                valid_row = valid[None, :]
                tail_ks_raw = tl.load(tle.gpu.local_ptr(s_beta_b_row, (offs_t,)))
                tail_ks = (
                    tail_ks_raw if FULL_TAIL else tl.where(valid, tail_ks_raw, 0.0)
                )
                score = qk * qs[:, None] * tail_ks[None, :] * softmax_scale
                score_safe = score if FULL_TAIL else tl.where(valid_row, score, 0.0)
                x = score_safe * _TLE_LOG2E
                page_m = tl.max(
                    x if FULL_TAIL else tl.where(valid_row, x, _TLE_NEG_INF), axis=1
                )
                old_m = tl.where(state_valid, state_m, _TLE_NEG_INF)
                old_s = tl.where(state_valid, state_s, 1.0)
                old_l = tl.where(state_valid, state_l, 0.0)
                m_new = tl.maximum(old_m, page_m)
                m_safe = tl.where(m_new == _TLE_NEG_INF, 0.0, m_new)
                e = (
                    tl.exp2(x - m_safe[:, None])
                    if FULL_TAIL
                    else tl.where(valid_row, tl.exp2(x - m_safe[:, None]), 0.0)
                )
                f = e * tail_ks[None, :]
                amax = tl.max(tl.abs(f), axis=1)
                s_new = tl.where(
                    amax == 0.0,
                    1.0,
                    tl.maximum(amax, _TLE_P_AMAX_FLOOR) / _TLE_FP8_MAX,
                )
                page_valid = (
                    True if FULL_TAIL else odd_page * PAGE_SIZE < split_cache_seqlen
                )
                inv_s_new = 1.0 / s_new
                p_new = tl.clamp(f * inv_s_new[:, None], -_TLE_FP8_MAX, _TLE_FP8_MAX)
                p1 = (
                    p_new
                    if FULL_TAIL
                    else tl.where(page_valid, p_new, tl.zeros_like(p_new))
                )
                if FULL_TAIL:
                    _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p_b, p1)
                else:
                    p1_store = p_new.to(tl.float8e4nv)
                    p1_store = tl.where(page_valid, p1_store, tl.zeros_like(p1_store))
                    tl.store(tle.gpu.local_ptr(s_p_b, (prow, pcol)), p1_store)
                old_m_finite = tl.where(state_valid, old_m, 0.0)
                alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
                beta1 = alpha * old_s * inv_s_new
                l_new = old_l * beta1 + tl.sum(e, axis=1) * inv_s_new
                state_m = tl.where(page_valid, m_new, old_m)
                state_s = tl.where(page_valid, s_new, old_s)
                state_l = tl.where(page_valid, l_new, old_l)
                beta1 = tl.where(page_valid, beta1, 1.0)
                state_valid = state_valid | page_valid

                tl.store(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)), beta1)
                tl.store(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), state_m)
                tl.store(tle.gpu.local_ptr(s_state1_s, (state_idx,)), state_s)
                tl.store(tle.gpu.local_ptr(s_state1_l, (state_idx,)), state_l)
                tl.store(
                    tle.gpu.local_ptr(s_state1_valid, (state_idx,)),
                    state_valid.to(tl.int32),
                )

                # Tail generation follows the same last-write publication rule.
                if not MERGE_STATE_V:
                    tle.gpu.barrier_arrive(state1_ready)

                if FULL_TAIL or (odd_page + 1) * PAGE_SIZE <= split_cache_seqlen:
                    _cuda_vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, FULL_TAIL)
                    _cuda_vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, FULL_TAIL)
                else:
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b0, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt0_b, (vt_c0_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b1, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt0_b, (vt_c1_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b2, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt1_b, (vt_c0_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                    kc_tile = tl.load(
                        tle.gpu.local_ptr(s_kc_b3, (kv_rows_d128, kv_c0_cols))
                    )
                    kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                    tl.store(
                        tle.gpu.local_ptr(s_vt1_b, (vt_c1_rows, vt_cols_d128)),
                        tl.trans(kc_tile),
                    )
                tle.gpu.barrier_arrive(v1_ready)

            if not MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready)
            beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
            acc_right *= beta0[:, None]
            acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
            acc_right = tle.gpu.wgmma_wait(0, acc_right)

            next_even_page = even_page + 2
            if next_even_page < num_pages:
                next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
                next_even_base = (next_even_phys * BK).to(tl.int32)
                tle.gpu.copy(
                    k_desc,
                    k_a_c2,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 2 * K_CONTENT_TILE],
                    barrier=k_content_full[2],
                )
                tle.gpu.copy(
                    k_desc,
                    k_a_c3,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 3 * K_CONTENT_TILE],
                    barrier=k_content_full[3],
                )
                tle.gpu.copy(
                    kr_desc,
                    s_kr_a,
                    [BK, ROPE],
                    [next_even_base, 0],
                    barrier=k_rope_full[0],
                )
                tle.gpu.copy(
                    ks_desc,
                    s_beta_a,
                    [1, BK],
                    [next_even_phys, 0],
                    barrier=k_scale_full[0],
                )
            tle.gpu.barrier_arrive(slot0_empty)

            if odd_page < num_pages:
                acc_right *= beta1[:, None]
                acc_right = tle.gpu.wgmma(s_p_b, s_vt1_b, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)
                tle.gpu.barrier_wait(slot1_empty)

            if next_even_page < num_pages:
                final_pair = pair + 1
                if MERGE_STATE_V:
                    tle.gpu.barrier_wait(v0_ready)
                else:
                    tle.gpu.barrier_wait(state0_ready)
                state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
                state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
                state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
                state_valid = (
                    tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0
                )
                if not MERGE_STATE_V:
                    tle.gpu.barrier_wait(v0_ready)
                beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
                acc_right *= beta0[:, None]
                acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)
                tle.gpu.barrier_arrive(slot0_empty)

        offs_d = tl.arange(0, DP)
        l_div = tl.where(state_l > 0.0, state_l, 1.0)
        inv_l_div = 1.0 / l_div
        out_right = tl.where(state_valid[:, None], acc_right * inv_l_div[:, None], 0.0)
        if USE_TMA_OUTPUT:
            out_right_lo, out_right_hi = tl.split(
                tl.permute(tl.reshape(out_right, (BH, 2, DP // 2)), (0, 2, 1))
            )
            out_right_0, out_right_1 = tl.split(
                tl.permute(tl.reshape(out_right_lo, (BH, 2, ROPE)), (0, 2, 1))
            )
            out_right_2, out_right_3 = tl.split(
                tl.permute(tl.reshape(out_right_hi, (BH, 2, ROPE)), (0, 2, 1))
            )
            tile_rows = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, ROPE))
            tile_cols = tl.broadcast_to(tl.arange(0, ROPE)[None, :], (BH, ROPE))
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_0.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP])
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_1.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + ROPE])
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_2.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + 2 * ROPE])
            tl.store(
                tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                out_right_3.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + 3 * ROPE])
        else:
            tl.store(
                out_ptr + offs_h[:, None] * stride_po_h + DP + offs_d[None, :],
                out_right,
                mask=mask_h[:, None],
            )

else:
    _fp8_mla_wg1 = None
    _fp8_mla_wg1_pretranspose = None
