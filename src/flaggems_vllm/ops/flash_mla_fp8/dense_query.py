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
    _TLE_LN2,
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
    def _fp8_mla_wg0(
        q_desc,
        qr_desc,
        qs_desc,
        out_desc,
        k_desc,
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
        out_ptr,
        lse2_ptr,
        stride_po_h,
        stride_pl_h,
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
        ENABLE_PDL: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        DIRECT_LSE: tl.constexpr,
        KNOWN_NUM_PAGES: tl.constexpr,
    ):
        """WG0: Q owner, even-page math, and the left output half."""
        if KNOWN_NUM_PAGES > 0:
            # The immutable host plan proves this logical page count.
            # Keep it opaque to layout propagation and specialize in LLVM.
            tl.assume(num_pages == KNOWN_NUM_PAGES)
        # The three CUDA-aligned Q payloads are one-shot TMA transactions.  The
        # scale temporarily occupies state1_m; WG1 cannot overwrite that field
        # until state0_ready, after both workers have consumed Q scale.
        s_state1_m_row = s_state1_m.slot(0)
        s_beta_a_row = s_beta_a.slot(0)
        s_beta_b_row = s_beta_b.slot(0)
        state_idx = tl.arange(0, BH)
        tle.gpu.copy(q_desc, s_q, [BH, CKV], [row0, 0], barrier=q_ckv_full)
        tle.gpu.copy(qr_desc, s_qr, [BH, ROPE], [row0, 0], barrier=q_rope_full)
        tle.gpu.copy(
            qs_desc,
            s_state1_m,
            [1, BH],
            [row0 // HQ, h_base],
            barrier=q_scale_full,
        )
        tle.gpu.barrier_wait(q_ckv_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_rope_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_scale_full, phaseIdx=0)

        offs_t = tl.arange(0, BK)
        offs_h = h_base + tl.arange(0, BH)
        mask_h = offs_h < HQ
        qs = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), volatile=True)

        acc_left = tl.zeros((BH, DP), dtype=tl.float32)
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
        k_a_c0 = s_kc_a0
        k_a_c1 = s_kc_a1
        k_a_c2 = s_kc_a2
        k_a_c3 = s_kc_a3
        k_b_c0 = s_kc_b0
        k_b_c1 = s_kc_b1
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
        # Fixed writer ownership applies to cold prime and steady state: WG0
        # issues content tiles 0/1 for both physical slots.
        if num_pages > 0:
            first_phys = tl.load(block_table)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_a_c0,
                [BK, K_CONTENT_TILE],
                [first_base, 0],
                barrier=k_content_full[0],
            )
            tle.gpu.copy(
                k_desc,
                k_a_c1,
                [BK, K_CONTENT_TILE],
                [first_base, K_CONTENT_TILE],
                barrier=k_content_full[1],
            )
        if num_pages > 1:
            first_phys = tl.load(block_table + stride_bt_pg)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_b_c0,
                [BK, K_CONTENT_TILE],
                [first_base, 0],
                barrier=k_content_full[4],
            )
            tle.gpu.copy(
                k_desc,
                k_b_c1,
                [BK, K_CONTENT_TILE],
                [first_base, K_CONTENT_TILE],
                barrier=k_content_full[5],
            )

        # Cold prime: page 0 QK, scale, and V are steady-loop live-ins. Rope
        # accumulates after content tile 3, matching the CUDA rP0 sequence.
        qk = tl.zeros((BH, BK), dtype=tl.float32)
        ks = tl.zeros((BK,), dtype=tl.float32)
        if num_pages > 0:
            tle.gpu.barrier_wait(k_content_full[0], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c0, k_a_c0, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[1], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c1, k_a_c1, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[2], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c2, k_a_c2, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[3], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c3, k_a_c3, qk, trans_b=True)
            tle.gpu.barrier_wait(k_rope_full[0], phaseIdx=0)
            qk = tle.gpu.wgmma(s_qr, s_kr_a, qk, trans_b=True)
            qk = tle.gpu.wgmma_wait(0, qk)

            tle.gpu.barrier_wait(k_scale_full[0], phaseIdx=0)
            prime_valid = offs_t < split_cache_seqlen
            ks_raw = tl.load(tle.gpu.local_ptr(s_beta_a_row, (offs_t,)))
            ks = ks_raw if FULL_TAIL else tl.where(prime_valid, ks_raw, 0.0)

        steady_pairs = tl.maximum(num_pairs - 1, 0)
        for pair in tl.range(steady_pairs, disable_licm=True):
            page = pair * 2
            if FULL_TAIL:
                valid = tl.full((BK,), True, tl.int1)
            else:
                valid = page * PAGE_SIZE + offs_t < split_cache_seqlen
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
            page_valid = True if FULL_TAIL else page * PAGE_SIZE < split_cache_seqlen
            if USE_HOTLOOP_RECIP:
                inv_s_new = 1.0 / s_new
                p_scaled = f * inv_s_new[:, None]
            else:
                p_scaled = f / s_new[:, None]
            p_new = tl.clamp(p_scaled, -_TLE_FP8_MAX, _TLE_FP8_MAX)
            p0 = (
                p_new
                if FULL_TAIL
                else tl.where(page_valid, p_new, tl.zeros_like(p_new))
            )
            if FULL_TAIL:
                _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p_a, p0)
            else:
                p0_store = p_new.to(tl.float8e4nv)
                p0_store = tl.where(page_valid, p0_store, tl.zeros_like(p0_store))
                tl.store(tle.gpu.local_ptr(s_p_a, (prow, pcol)), p0_store)
            old_m_finite = tl.where(state_valid, old_m, 0.0)
            alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
            if USE_HOTLOOP_RECIP:
                beta = alpha * old_s * inv_s_new
                l_new = old_l * beta + tl.sum(e, axis=1) * inv_s_new
            else:
                beta = alpha * old_s / s_new
                l_new = old_l * beta + tl.sum(e, axis=1) / s_new
            state_m = tl.where(page_valid, m_new, old_m)
            state_s = tl.where(page_valid, s_new, old_s)
            state_l = tl.where(page_valid, l_new, old_l)
            beta = tl.where(page_valid, beta, 1.0)
            state_valid = state_valid | page_valid

            tl.store(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)), beta)
            tl.store(tle.gpu.local_ptr(s_state0_m, (state_idx,)), state_m)
            tl.store(tle.gpu.local_ptr(s_state0_s, (state_idx,)), state_s)
            tl.store(tle.gpu.local_ptr(s_state0_l, (state_idx,)), state_l)
            tl.store(
                tle.gpu.local_ptr(s_state0_valid, (state_idx,)),
                state_valid.to(tl.int32),
            )

            # CUDA publishes the completed online-softmax state at its last
            # shared write. Do not serialize WG1 softmax behind the unrelated
            # V repack that follows in WG0.
            if not MERGE_STATE_V:
                tle.gpu.barrier_arrive(state0_ready)

            # This loop excludes the final pair, so its even page is always a
            # complete logical page.  Match CUDA's compile-time steady-state
            # specialization and keep the masked tensor fallback in the
            # epilogue only.
            _cuda_vtranspose_fp8_64x128(s_kc_a0, s_vt0_a, 0, FULL_TAIL)
            _cuda_vtranspose_fp8_64x128(s_kc_a1, s_vt0_a, DP // 2, FULL_TAIL)
            _cuda_vtranspose_fp8_64x128(s_kc_a2, s_vt1_a, 0, FULL_TAIL)
            _cuda_vtranspose_fp8_64x128(s_kc_a3, s_vt1_a, DP // 2, FULL_TAIL)

            tle.gpu.barrier_arrive(v0_ready)

            # CUDA local-P wait point: finish current local PV, then launch
            # slot-A generation pair+1 content0/1 for p+2.
            acc_left *= beta[:, None]
            acc_left = tle.gpu.wgmma(s_p_a, s_vt0_a, acc_left, trans_b=True)
            acc_left = tle.gpu.wgmma_wait(0, acc_left)

            next_even_page = page + 2
            next_generation = pair + 1
            next_qk = tl.zeros((BH, BK), dtype=tl.float32)
            next_ks = tl.zeros((BK,), dtype=tl.float32)
            if True:
                next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
                next_even_base = (next_even_phys * BK).to(tl.int32)
                tle.gpu.copy(
                    k_desc,
                    k_a_c0,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 0],
                    barrier=k_content_full[0],
                )
                tle.gpu.copy(
                    k_desc,
                    k_a_c1,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, K_CONTENT_TILE],
                    barrier=k_content_full[1],
                )

            odd_page = page + 1
            if True:
                tle.gpu.barrier_wait(v1_ready)
                beta1 = tl.load(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)))
                acc_left *= beta1[:, None]
                acc_left = tle.gpu.wgmma(s_p_b, s_vt0_b, acc_left, trans_b=True)

                # Keep the async rP0 chain inside one real-p+2 branch:
                # TLE permits loop-carried accumulators but not an async value
                # yielded through an intermediate scf.if.
                if True:
                    # CUDA QK phase-0. Two younger QK groups allow wait2 to
                    # retire only the oldest remote-P group.
                    tle.gpu.barrier_wait(k_content_full[0], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c0, k_a_c0, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[1], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c1, k_a_c1, next_qk, trans_b=True)
                    phase0_waited_qk = tle.gpu.wgmma_wait(2, next_qk)
                    tle.gpu.barrier_arrive(slot1_empty)

                    # CUDA wait2 point starts p+3 content0/1 before p+2
                    # phase-2.
                    next_odd_page = odd_page + 2
                    if next_odd_page < num_pages:
                        next_odd_phys = tl.load(
                            block_table + next_odd_page * stride_bt_pg
                        )
                        next_odd_base = (next_odd_phys * BK).to(tl.int32)
                        tle.gpu.copy(
                            k_desc,
                            k_b_c0,
                            [BK, K_CONTENT_TILE],
                            [next_odd_base, 0],
                            barrier=k_content_full[4],
                        )
                        tle.gpu.copy(
                            k_desc,
                            k_b_c1,
                            [BK, K_CONTENT_TILE],
                            [next_odd_base, K_CONTENT_TILE],
                            barrier=k_content_full[5],
                        )

                    # CUDA QK phase-2 completes p+2 in the current pair.
                    tle.gpu.barrier_wait(k_content_full[2], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c2, k_a_c2, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_content_full[3], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(q_c3, k_a_c3, next_qk, trans_b=True)
                    tle.gpu.barrier_wait(k_rope_full[0], phaseIdx=next_generation)
                    next_qk = tle.gpu.wgmma(s_qr, s_kr_a, next_qk, trans_b=True)
                    next_qk = tle.gpu.wgmma_wait(0, next_qk)
                    # The wait is global in hardware, but TLE also requires
                    # the remote-P SSA value itself to pass through a wait.
                    acc_left = tle.gpu.wgmma_wait(0, acc_left)

                    tle.gpu.barrier_wait(k_scale_full[0], phaseIdx=next_generation)
                    next_valid = (
                        next_even_page * PAGE_SIZE + offs_t < split_cache_seqlen
                    )
                    next_ks_raw = tl.load(tle.gpu.local_ptr(s_beta_a_row, (offs_t,)))
                    next_ks = (
                        next_ks_raw
                        if FULL_TAIL
                        else tl.where(next_valid, next_ks_raw, 0.0)
                    )

                else:
                    # Tail pair: no younger QK groups exist to retain.
                    acc_left = tle.gpu.wgmma_wait(0, acc_left)
                    tle.gpu.barrier_arrive(slot1_empty)

                if not MERGE_STATE_V:
                    tle.gpu.barrier_wait(state1_ready)
                state_m = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)))
                state_s = tl.load(tle.gpu.local_ptr(s_state1_s, (state_idx,)))
                state_l = tl.load(tle.gpu.local_ptr(s_state1_l, (state_idx,)))
                state_valid = (
                    tl.load(tle.gpu.local_ptr(s_state1_valid, (state_idx,))) != 0
                )

            # WG1 publishes this only after its remote P0/V0 wait0.
            tle.gpu.barrier_wait(slot0_empty)
            qk = next_qk
            ks = next_ks

        # CUDA-style epilogue: the final pair never creates a younger QK
        # accumulator, so every PV dependency is retired inside this tail.
        if num_pairs > 0:
            pair = steady_pairs
            page = pair * 2
            valid = page * PAGE_SIZE + offs_t < split_cache_seqlen
            valid_row = valid[None, :]
            tail_ks_raw = tl.load(tle.gpu.local_ptr(s_beta_a_row, (offs_t,)))
            tail_ks = tail_ks_raw if FULL_TAIL else tl.where(valid, tail_ks_raw, 0.0)
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
            page_valid = True if FULL_TAIL else page * PAGE_SIZE < split_cache_seqlen
            inv_s_new = 1.0 / s_new
            p_new = tl.clamp(f * inv_s_new[:, None], -_TLE_FP8_MAX, _TLE_FP8_MAX)
            p0 = (
                p_new
                if FULL_TAIL
                else tl.where(page_valid, p_new, tl.zeros_like(p_new))
            )
            if FULL_TAIL:
                _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p_a, p0)
            else:
                p0_store = p_new.to(tl.float8e4nv)
                p0_store = tl.where(page_valid, p0_store, tl.zeros_like(p0_store))
                tl.store(tle.gpu.local_ptr(s_p_a, (prow, pcol)), p0_store)
            old_m_finite = tl.where(state_valid, old_m, 0.0)
            alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
            beta = alpha * old_s * inv_s_new
            l_new = old_l * beta + tl.sum(e, axis=1) * inv_s_new
            state_m = tl.where(page_valid, m_new, old_m)
            state_s = tl.where(page_valid, s_new, old_s)
            state_l = tl.where(page_valid, l_new, old_l)
            beta = tl.where(page_valid, beta, 1.0)
            state_valid = state_valid | page_valid

            tl.store(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)), beta)
            tl.store(tle.gpu.local_ptr(s_state0_m, (state_idx,)), state_m)
            tl.store(tle.gpu.local_ptr(s_state0_s, (state_idx,)), state_s)
            tl.store(tle.gpu.local_ptr(s_state0_l, (state_idx,)), state_l)
            tl.store(
                tle.gpu.local_ptr(s_state0_valid, (state_idx,)),
                state_valid.to(tl.int32),
            )

            # Tail generation follows the same last-write publication rule.
            if not MERGE_STATE_V:
                tle.gpu.barrier_arrive(state0_ready)

            # Invalid probability columns are already exact FP8 zero after
            # the masked softmax above, so their V values cannot contribute
            # to PV.  Reuse the CUDA-aligned vectorized transpose for a
            # partial physical page instead of materializing a masked tensor
            # transpose in registers.
            if PAGE_GRAIN_TAIL_ZERO:
                if not FULL_TAIL:
                    valid_tokens = tl.minimum(split_cache_seqlen, BK)
                    if valid_tokens < BK:
                        _zero_invalid_fp8_rows_sw128_x4(
                            s_kc_a0,
                            s_kc_a1,
                            s_kc_a2,
                            s_kc_a3,
                            valid_tokens,
                        )
                        tle.gpu.barrier_arrive(tail0_zero_ready, phaseIdx=pair)
                        tle.gpu.barrier_wait(tail0_zero_ready, phaseIdx=pair)
                _cuda_vtranspose_fp8_64x128(s_kc_a0, s_vt0_a, 0, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_a1, s_vt0_a, DP // 2, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_a2, s_vt1_a, 0, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_a3, s_vt1_a, DP // 2, FULL_TAIL)
            elif FULL_TAIL or (page + 1) * PAGE_SIZE <= split_cache_seqlen:
                _cuda_vtranspose_fp8_64x128(s_kc_a0, s_vt0_a, 0, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_a1, s_vt0_a, DP // 2, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_a2, s_vt1_a, 0, FULL_TAIL)
                _cuda_vtranspose_fp8_64x128(s_kc_a3, s_vt1_a, DP // 2, FULL_TAIL)
            else:
                kc_tile = tl.load(
                    tle.gpu.local_ptr(s_kc_a0, (kv_rows_d128, kv_c0_cols))
                )
                kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                tl.store(
                    tle.gpu.local_ptr(s_vt0_a, (vt_c0_rows, vt_cols_d128)),
                    tl.trans(kc_tile),
                )
                kc_tile = tl.load(
                    tle.gpu.local_ptr(s_kc_a1, (kv_rows_d128, kv_c0_cols))
                )
                kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                tl.store(
                    tle.gpu.local_ptr(s_vt0_a, (vt_c1_rows, vt_cols_d128)),
                    tl.trans(kc_tile),
                )
                kc_tile = tl.load(
                    tle.gpu.local_ptr(s_kc_a2, (kv_rows_d128, kv_c0_cols))
                )
                kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                tl.store(
                    tle.gpu.local_ptr(s_vt1_a, (vt_c0_rows, vt_cols_d128)),
                    tl.trans(kc_tile),
                )
                kc_tile = tl.load(
                    tle.gpu.local_ptr(s_kc_a3, (kv_rows_d128, kv_c0_cols))
                )
                kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                tl.store(
                    tle.gpu.local_ptr(s_vt1_a, (vt_c1_rows, vt_cols_d128)),
                    tl.trans(kc_tile),
                )

            tle.gpu.barrier_arrive(v0_ready)

            acc_left *= beta[:, None]
            acc_left = tle.gpu.wgmma(s_p_a, s_vt0_a, acc_left, trans_b=True)
            acc_left = tle.gpu.wgmma_wait(0, acc_left)

            odd_page = page + 1
            if odd_page < num_pages:
                tle.gpu.barrier_wait(v1_ready)
                beta1 = tl.load(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)))
                acc_left *= beta1[:, None]
                acc_left = tle.gpu.wgmma(s_p_b, s_vt0_b, acc_left, trans_b=True)
                acc_left = tle.gpu.wgmma_wait(0, acc_left)
                tle.gpu.barrier_arrive(slot1_empty)

                if not MERGE_STATE_V:
                    tle.gpu.barrier_wait(state1_ready)
                state_m = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)))
                state_s = tl.load(tle.gpu.local_ptr(s_state1_s, (state_idx,)))
                state_l = tl.load(tle.gpu.local_ptr(s_state1_l, (state_idx,)))
                state_valid = (
                    tl.load(tle.gpu.local_ptr(s_state1_valid, (state_idx,))) != 0
                )

            tle.gpu.barrier_wait(slot0_empty)

        # CUDA-aligned programmatic dependency trigger.  Only the B>=4
        # coarse-combine specialization receives ENABLE_PDL=True.
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()

        offs_d = tl.arange(0, DP)
        l_div = tl.where(state_l > 0.0, state_l, 1.0)
        inv_l_div = 1.0 / l_div
        out_left = tl.where(state_valid[:, None], acc_left * inv_l_div[:, None], 0.0)
        if USE_TMA_OUTPUT:
            # The final K-rope read has retired before the loop exits, so its
            # existing 64x64 BF16 buffer can stage four output chunks without
            # adding shared memory. Each copy is a complete TMA S2G group; the
            # TLE store scheduler inserts the reuse-safe commit/wait sequence.
            out_left_lo, out_left_hi = tl.split(
                tl.permute(tl.reshape(out_left, (BH, 2, DP // 2)), (0, 2, 1))
            )
            out_left_0, out_left_1 = tl.split(
                tl.permute(tl.reshape(out_left_lo, (BH, 2, ROPE)), (0, 2, 1))
            )
            out_left_2, out_left_3 = tl.split(
                tl.permute(tl.reshape(out_left_hi, (BH, 2, ROPE)), (0, 2, 1))
            )
            tile_rows = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, ROPE))
            tile_cols = tl.broadcast_to(tl.arange(0, ROPE)[None, :], (BH, ROPE))
            tl.store(
                tle.gpu.local_ptr(s_kr_a, (tile_rows, tile_cols)),
                out_left_0.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_a, out_desc, [BH, ROPE], [row0, 0])
            tl.store(
                tle.gpu.local_ptr(s_kr_a, (tile_rows, tile_cols)),
                out_left_1.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_a, out_desc, [BH, ROPE], [row0, ROPE])
            tl.store(
                tle.gpu.local_ptr(s_kr_a, (tile_rows, tile_cols)),
                out_left_2.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_a, out_desc, [BH, ROPE], [row0, 2 * ROPE])
            tl.store(
                tle.gpu.local_ptr(s_kr_a, (tile_rows, tile_cols)),
                out_left_3.to(tl.bfloat16),
            )
            tle.gpu.copy(s_kr_a, out_desc, [BH, ROPE], [row0, 3 * ROPE])
        else:
            tl.store(
                out_ptr + offs_h[:, None] * stride_po_h + offs_d[None, :],
                out_left,
                mask=mask_h[:, None],
            )
        lse_arg = state_l * state_s
        lse_ok = state_valid & (lse_arg > 0.0)
        lse2_value = tl.where(
            lse_ok,
            state_m + tl.log(tl.where(lse_arg > 0.0, lse_arg, 1.0)) * _TLE_LOG2E,
            _TLE_NEG_INF,
        )
        tl.store(
            lse2_ptr + offs_h * stride_pl_h,
            lse2_value * _TLE_LN2 if DIRECT_LSE else lse2_value,
            mask=mask_h,
        )

else:
    _fp8_mla_wg0 = None
