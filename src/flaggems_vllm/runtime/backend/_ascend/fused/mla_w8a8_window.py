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

import importlib.util

import triton
import triton.experimental.tle as tle
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.runtime.backend._ascend.fused.attention import PIPE, stage_sync

REQUIRED_WINDOW_CUSTOM_OPS = (
    "nd2nz_gm_to_l1",
    "gather_blocks_gm_to_l1",
    "gather_blocks_matmul_pair_l1_to_gm",
    "matmul_l1_to_gm",
    "matmul_l1_to_gm_deqf16",
    "nd2nz_matmul_l1_to_gm_deqf16",
    "online_softmax_quant_s8",
    "online_rescaled_accum",
)
CUSTOM_OPS_SPEC = importlib.util.find_spec(
    "triton.experimental.tle.language.dsa.ascend.custom_ops"
)
if CUSTOM_OPS_SPEC is not None:
    from triton.experimental.tle.language.dsa.ascend import (
        custom_ops as ascend_custom_ops,
    )

    HAS_PAGED_INT8_OPS = all(
        op_name in ascend_custom_ops.__all__ for op_name in REQUIRED_WINDOW_CUSTOM_OPS
    )
else:
    HAS_PAGED_INT8_OPS = False

# Fixed FP32 dequantization scale (2^-14) for Fixpipe DEQF16 PV outputs on
# M >= 64 so INT32 PV accumulations stay within FP16 range.
DEQ_SCALE_FP16 = tl.constexpr(1.0 / 16384.0)
DEQ_INV_SCALE_FP16 = tl.constexpr(16384.0)


@triton.jit(do_not_specialize=["TASKS", "WIDTH", "BATCH"])
def window_pipeline(
    Q,
    QR,
    K,
    KR,
    QS,
    KS,
    Table,
    Lens,
    C,
    R,
    P,
    PV,
    PV_h,
    Out,
    Lse,
    TASKS,
    WIDTH,
    CORES: tl.constexpr,
    SCALE: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    CAUSAL: tl.constexpr,
    M: tl.constexpr,
    BATCH,
    STRIDE: tl.constexpr,
):
    USE_FP16_PV: tl.constexpr = M >= 64
    G_PAGES: tl.constexpr = 6
    G_TOKENS: tl.constexpr = 384
    core = tl.program_id(0)

    # Per-core scratch layout after PV ping-pong buffers (4 * M * 512 int32s/core):
    # - Flags:   CORES * 32 int32s (16 int32s per (core, slot))
    # - PairIds: CORES * 8 int32s (query pair indices when S == 2 and band < H)
    # - Plan:    CORES * STRIDE int32s (LPT task schedule)
    Flags = PV + CORES * (4 * M * 512)
    PairIds = Flags + CORES * 32
    Plan = PairIds + CORES * 8

    GROUPS_PER_REQ: tl.constexpr = (H * S) // M
    SUB_M: tl.constexpr = M // 2
    band: tl.constexpr = M // S
    OSQ_TILE_R: tl.constexpr = 16 if SUB_M >= 16 else 8
    OSQ_NBANKS: tl.constexpr = 2 if (SUB_M // OSQ_TILE_R) >= 2 else 1
    OSQ_WORK_F: tl.constexpr = (
        OSQ_NBANKS * 2 * OSQ_TILE_R * G_TOKENS
        + ((SUB_M + 63) & ~63)
        + 2 * G_TOKENS
        + (G_TOKENS // 2)
        + OSQ_TILE_R * 64
        + 512
    )
    UPD_TILE_R_F16: tl.constexpr = 32 if SUB_M >= 32 else SUB_M
    UPD_NBANKS_F16: tl.constexpr = 2 if (SUB_M // UPD_TILE_R_F16) >= 2 else 1
    UPD_LO_R_F16: tl.constexpr = 8 if UPD_TILE_R_F16 >= 8 else UPD_TILE_R_F16
    UPD_WORK_F16: tl.constexpr = (
        (UPD_NBANKS_F16 * UPD_TILE_R_F16 + UPD_LO_R_F16) * 256
        + 256
        + UPD_TILE_R_F16 * 16
        + 128
    )
    UPD_NBANKS_F32: tl.constexpr = 2 if (SUB_M // 8) >= 2 else 1
    UPD_WORK_F32: tl.constexpr = (UPD_NBANKS_F32 + 1) * 8 * 512 + 512 + 8 * 16 + 128
    UPD_WORK_F: tl.constexpr = UPD_WORK_F16 if USE_FP16_PV else UPD_WORK_F32
    SHARED_WORK_F: tl.constexpr = OSQ_WORK_F if OSQ_WORK_F >= UPD_WORK_F else UPD_WORK_F

    with al.scope(core_mode="cube"):
        al.sync_block_wait("vector", "cube", 7, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
        al.sync_block_set("cube", "vector", 7, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)

        k0_buf = tle.dsa.alloc([G_TOKENS, 512], tl.int8, tle.dsa.ascend.L1)
        k1_buf = tle.dsa.alloc([G_TOKENS, 512], tl.int8, tle.dsa.ascend.L1)
        q_buf = tle.dsa.alloc([M, 512], tl.int8, tle.dsa.ascend.L1)
        qr_buf = tle.dsa.alloc([M, 64], tl.bfloat16, tle.dsa.ascend.L1)

        for index in tl.range(core, TASKS, CORES, disable_licm=True):
            task = index
            if STRIDE:
                task = tl.load(Plan + core * STRIDE + index // CORES)
            b = task // (H // band)
            head = task % (H // band) * band
            n = tl.load(Lens + b)
            pages = tl.cdiv(n, 64)
            groups = tl.cdiv(n, G_TOKENS)
            if groups > 0:
                if S == 2 and band < H:
                    q_row0 = b.to(tl.int64) * 2 * H + head
                    q_l1 = tle_raw.call(
                        "ascend_gather_blocks_gm_to_l1",
                        (Q + q_row0 * 512).to(tl.uint64),
                        (PairIds + core * 8).to(tl.uint64),
                        2,
                        2,
                        band,
                        512,
                        H * 512,
                        out=tle.dsa.to_tensor(q_buf, writable=True),
                    )
                    qr_l1 = tle_raw.call(
                        "ascend_gather_blocks_gm_to_l1",
                        (QR + q_row0 * 64).to(tl.uint64),
                        (PairIds + core * 8).to(tl.uint64),
                        2,
                        2,
                        band,
                        64,
                        H * 64,
                        out=tle.dsa.to_tensor(qr_buf, writable=True),
                    )
                else:
                    q_row = b.to(tl.int64) * S * H + head
                    q_l1 = tle_raw.call(
                        "ascend_nd2nz_gm_to_l1",
                        (Q + q_row * 512).to(tl.uint64),
                        M,
                        512,
                        512,
                        out=tle.dsa.to_tensor(q_buf, writable=True),
                    )
                    qr_l1 = tle_raw.call(
                        "ascend_nd2nz_gm_to_l1",
                        (QR + q_row * 64).to(tl.uint64),
                        M,
                        64,
                        64,
                        out=tle.dsa.to_tensor(qr_buf, writable=True),
                    )

                c_base0 = (core * 2).to(tl.int64) * M * G_TOKENS
                tab_off0 = b.to(tl.int64) * WIDTH
                rem0 = tl.minimum(pages, G_PAGES)
                kr_buf0 = tle.dsa.alloc([G_TOKENS, 64], tl.bfloat16, tle.dsa.ascend.L1)
                k0_init, _kr0 = tle_raw.call(
                    "ascend_gather_blocks_matmul_pair_l1_to_gm",
                    q_l1,
                    K.to(tl.uint64),
                    qr_l1,
                    KR.to(tl.uint64),
                    (Table + tab_off0).to(tl.uint64),
                    (C + c_base0).to(tl.uint64),
                    (R + c_base0).to(tl.uint64),
                    G_PAGES,
                    rem0,
                    64,
                    M,
                    512,
                    64,
                    64 * 512,
                    64 * 64,
                    G_TOKENS,
                    tl.cast(0, tl.uint32),
                    out=(
                        tle.dsa.to_tensor(k0_buf, writable=True),
                        tle.dsa.to_tensor(kr_buf0, writable=True),
                    ),
                )
                tle.dsa.to_buffer(k0_init, bind_buffer=k0_buf)
                stage_sync(0, 0, False)

                for group in tl.range(0, groups, disable_licm=True):
                    slot = group % 2
                    if group + 1 < groups:
                        next_slot = 1 - slot
                        c_base_n = (core * 2 + next_slot).to(tl.int64) * M * G_TOKENS
                        tab_off_n = b.to(tl.int64) * WIDTH + (group + 1) * G_PAGES
                        rem_n = tl.minimum(pages - (group + 1) * G_PAGES, G_PAGES)
                        preloaded_n = (
                            tl.where(group >= 1, 1, 0).to(tl.uint32)
                            if USE_FP16_PV
                            else tl.cast(0, tl.uint32)
                        )
                        kr_buf_n = tle.dsa.alloc(
                            [G_TOKENS, 64], tl.bfloat16, tle.dsa.ascend.L1
                        )
                        if slot == 0:
                            k_n, _kr_n = tle_raw.call(
                                "ascend_gather_blocks_matmul_pair_l1_to_gm",
                                q_l1,
                                K.to(tl.uint64),
                                qr_l1,
                                KR.to(tl.uint64),
                                (Table + tab_off_n).to(tl.uint64),
                                (C + c_base_n).to(tl.uint64),
                                (R + c_base_n).to(tl.uint64),
                                G_PAGES,
                                rem_n,
                                64,
                                M,
                                512,
                                64,
                                64 * 512,
                                64 * 64,
                                G_TOKENS,
                                preloaded_n,
                                out=(
                                    tle.dsa.to_tensor(k1_buf, writable=True),
                                    tle.dsa.to_tensor(kr_buf_n, writable=True),
                                ),
                            )
                            tle.dsa.to_buffer(k_n, bind_buffer=k1_buf)
                        else:
                            k_n, _kr_n = tle_raw.call(
                                "ascend_gather_blocks_matmul_pair_l1_to_gm",
                                q_l1,
                                K.to(tl.uint64),
                                qr_l1,
                                KR.to(tl.uint64),
                                (Table + tab_off_n).to(tl.uint64),
                                (C + c_base_n).to(tl.uint64),
                                (R + c_base_n).to(tl.uint64),
                                G_PAGES,
                                rem_n,
                                64,
                                M,
                                512,
                                64,
                                64 * 512,
                                64 * 64,
                                G_TOKENS,
                                preloaded_n,
                                out=(
                                    tle.dsa.to_tensor(k0_buf, writable=True),
                                    tle.dsa.to_tensor(kr_buf_n, writable=True),
                                ),
                            )
                            tle.dsa.to_buffer(k_n, bind_buffer=k0_buf)
                        stage_sync(next_slot, 0, False)

                    stage_sync(slot, 1, True)
                    use_fast_1d = (M >= 64) & (n > 1024) & ((n - group * G_TOKENS) > 64)
                    slot_idx_c = (core * 2 + slot).to(tl.int64)
                    p_hi_base_c = slot_idx_c * (2 * M * G_TOKENS)
                    p_lo_base_c = p_hi_base_c + M * G_TOKENS
                    pv_hi_base_c = slot_idx_c * (2 * M * 512)
                    pv_lo_base_c = pv_hi_base_c + M * 512

                    need_lo = 1
                    if use_fast_1d:
                        f0 = tl.load(Flags + slot_idx_c * 16)
                        f1 = tl.load(Flags + slot_idx_c * 16 + 8)
                        need_lo = tl.where((f0 | f1) != 0, 1, 0)

                    p_buf = tle.dsa.alloc([M, G_TOKENS], tl.int8, tle.dsa.ascend.L1)
                    if USE_FP16_PV:
                        has_pf = group + 2 < groups
                        pf_tab_off = b.to(tl.int64) * WIDTH + tl.minimum(
                            (group + 2) * G_PAGES, pages - 1
                        )
                        pf_rem = tl.maximum(
                            tl.minimum(pages - (group + 2) * G_PAGES, G_PAGES), 0
                        ).to(tl.uint32)
                        pf_hi_gm = tl.where(
                            has_pf & (need_lo == 0),
                            K.to(tl.uint64),
                            tl.cast(0, tl.uint64),
                        )
                        pf_lo_gm = tl.where(
                            has_pf, K.to(tl.uint64), tl.cast(0, tl.uint64)
                        )
                        pf_ids_u64 = (Table + pf_tab_off).to(tl.uint64)
                        if slot == 0:
                            _p0, k0_pf = tle_raw.call(
                                "ascend_nd2nz_matmul_l1_to_gm_deqf16",
                                (P + p_hi_base_c).to(tl.uint64),
                                (PV_h + pv_hi_base_c).to(tl.uint64),
                                M,
                                512,
                                G_TOKENS,
                                512,
                                DEQ_SCALE_FP16,
                                pf_hi_gm,
                                pf_ids_u64,
                                tl.cast(2, tl.uint32),
                                pf_rem,
                                tl.cast(64, tl.uint32),
                                tl.cast(64 * 512, tl.uint32),
                                out=(
                                    tle.dsa.to_tensor(p_buf, writable=True),
                                    tle.dsa.to_tensor(k0_buf, writable=True),
                                ),
                            )
                            tle.dsa.to_buffer(k0_pf, bind_buffer=k0_buf)
                            if need_lo != 0:
                                _p0l, k0_pfl = tle_raw.call(
                                    "ascend_nd2nz_matmul_l1_to_gm_deqf16",
                                    (P + p_lo_base_c).to(tl.uint64),
                                    (PV_h + pv_lo_base_c).to(tl.uint64),
                                    M,
                                    512,
                                    G_TOKENS,
                                    512,
                                    DEQ_SCALE_FP16,
                                    pf_lo_gm,
                                    pf_ids_u64,
                                    tl.cast(2, tl.uint32),
                                    pf_rem,
                                    tl.cast(64, tl.uint32),
                                    tl.cast(64 * 512, tl.uint32),
                                    out=(
                                        tle.dsa.to_tensor(p_buf, writable=True),
                                        tle.dsa.to_tensor(k0_buf, writable=True),
                                    ),
                                )
                                tle.dsa.to_buffer(k0_pfl, bind_buffer=k0_buf)
                        else:
                            _p1, k1_pf = tle_raw.call(
                                "ascend_nd2nz_matmul_l1_to_gm_deqf16",
                                (P + p_hi_base_c).to(tl.uint64),
                                (PV_h + pv_hi_base_c).to(tl.uint64),
                                M,
                                512,
                                G_TOKENS,
                                512,
                                DEQ_SCALE_FP16,
                                pf_hi_gm,
                                pf_ids_u64,
                                tl.cast(2, tl.uint32),
                                pf_rem,
                                tl.cast(64, tl.uint32),
                                tl.cast(64 * 512, tl.uint32),
                                out=(
                                    tle.dsa.to_tensor(p_buf, writable=True),
                                    tle.dsa.to_tensor(k1_buf, writable=True),
                                ),
                            )
                            tle.dsa.to_buffer(k1_pf, bind_buffer=k1_buf)
                            if need_lo != 0:
                                _p1l, k1_pfl = tle_raw.call(
                                    "ascend_nd2nz_matmul_l1_to_gm_deqf16",
                                    (P + p_lo_base_c).to(tl.uint64),
                                    (PV_h + pv_lo_base_c).to(tl.uint64),
                                    M,
                                    512,
                                    G_TOKENS,
                                    512,
                                    DEQ_SCALE_FP16,
                                    pf_lo_gm,
                                    pf_ids_u64,
                                    tl.cast(2, tl.uint32),
                                    pf_rem,
                                    tl.cast(64, tl.uint32),
                                    tl.cast(64 * 512, tl.uint32),
                                    out=(
                                        tle.dsa.to_tensor(p_buf, writable=True),
                                        tle.dsa.to_tensor(k1_buf, writable=True),
                                    ),
                                )
                                tle.dsa.to_buffer(k1_pfl, bind_buffer=k1_buf)
                    else:
                        p_hi_l1 = tle_raw.call(
                            "ascend_nd2nz_gm_to_l1",
                            (P + p_hi_base_c).to(tl.uint64),
                            M,
                            G_TOKENS,
                            G_TOKENS,
                            out=tle.dsa.to_tensor(p_buf, writable=True),
                        )
                        if slot == 0:
                            tle_raw.call(
                                "ascend_matmul_l1_to_gm",
                                p_hi_l1,
                                tle.dsa.to_tensor(k0_buf, writable=False),
                                (PV + pv_hi_base_c).to(tl.uint64),
                                M,
                                512,
                                G_TOKENS,
                                512,
                                0,
                            )
                        else:
                            tle_raw.call(
                                "ascend_matmul_l1_to_gm",
                                p_hi_l1,
                                tle.dsa.to_tensor(k1_buf, writable=False),
                                (PV + pv_hi_base_c).to(tl.uint64),
                                M,
                                512,
                                G_TOKENS,
                                512,
                                0,
                            )
                        if need_lo != 0:
                            p_lo_l1 = tle_raw.call(
                                "ascend_nd2nz_gm_to_l1",
                                (P + p_lo_base_c).to(tl.uint64),
                                M,
                                G_TOKENS,
                                G_TOKENS,
                                out=tle.dsa.to_tensor(p_buf, writable=True),
                            )
                            if slot == 0:
                                tle_raw.call(
                                    "ascend_matmul_l1_to_gm",
                                    p_lo_l1,
                                    tle.dsa.to_tensor(k0_buf, writable=False),
                                    (PV + pv_lo_base_c).to(tl.uint64),
                                    M,
                                    512,
                                    G_TOKENS,
                                    512,
                                    0,
                                )
                            else:
                                tle_raw.call(
                                    "ascend_matmul_l1_to_gm",
                                    p_lo_l1,
                                    tle.dsa.to_tensor(k1_buf, writable=False),
                                    (PV + pv_lo_base_c).to(tl.uint64),
                                    M,
                                    512,
                                    G_TOKENS,
                                    512,
                                    0,
                                )

                    stage_sync(slot, 2, False)
            al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)

    with al.scope(core_mode="vector"):
        sub_id = al.sub_vec_id()
        if sub_id == 0:
            if S == 2 and band < H:
                tl.store(PairIds + core * 8, 0)
                tl.store(PairIds + core * 8 + 1, 1)
            if STRIDE:
                requests = tl.arange(0, 128)
                lengths = tl.load(Lens + requests, mask=requests < BATCH, other=0)
                wave = tl.arange(0, STRIDE)
                start = wave * CORES
                tasks = start + core
                if tl.min(tl.where(requests < BATCH, lengths, 2147483647), 0) != tl.max(
                    lengths, 0
                ):
                    cost = lengths // G_TOKENS + (lengths % G_TOKENS != 0).to(tl.int32)
                    keys = tl.where(
                        requests < BATCH,
                        tl.minimum(cost, 131071) * 128 + 127 - requests,
                        -1,
                    )
                    ranked = 127 - (
                        al.sort(keys.to(tl.float32), descending=True).to(tl.int32) & 127
                    )
                    width = tl.minimum(CORES, TASKS - start)
                    slot_s = tl.where((wave & 1) == 0, tasks, start + width - 1 - core)
                    safe_slot = tl.minimum(tl.maximum(slot_s, 0), TASKS - 1)
                    req_rank = safe_slot // GROUPS_PER_REQ
                    b_r = tl.gather(ranked.to(tl.float32), req_rank, 0).to(tl.int32)
                    tasks = b_r * GROUPS_PER_REQ + (safe_slot % GROUPS_PER_REQ)
                tl.store(Plan + core * STRIDE + wave, tasks)
        al.sync_block_set("vector", "cube", 7, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
        al.sync_block_wait("cube", "vector", 7, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)

        sub = sub_id * SUB_M
        offs_256 = tl.arange(0, 256)
        offs_sub = tl.arange(0, SUB_M)
        EMPTY_TILE_R: tl.constexpr = 16 if SUB_M >= 16 else 8
        offs_er = tl.arange(0, EMPTY_TILE_R)

        ks_dummy = tle.dsa.alloc([1, G_TOKENS], tl.float32, tle.dsa.ascend.UB)
        tle.dsa.to_buffer(
            tl.zeros((1, G_TOKENS), dtype=tl.float32), bind_buffer=ks_dummy
        )
        if USE_FP16_PV:
            acc_buf = tle.dsa.alloc([SUB_M, 512], tl.float16, tle.dsa.ascend.UB)
        else:
            acc_buf = tle.dsa.alloc([SUB_M, 512], tl.float32, tle.dsa.ascend.UB)
        state0_buf = tle.dsa.alloc([5, SUB_M], tl.float32, tle.dsa.ascend.UB)
        state1_buf = tle.dsa.alloc([5, SUB_M], tl.float32, tle.dsa.ascend.UB)
        shared_work_buf = tle.dsa.alloc([SHARED_WORK_F], tl.float32, tle.dsa.ascend.UB)
        tle.dsa.to_buffer(
            tl.zeros((5, SUB_M), dtype=tl.float32), bind_buffer=state0_buf
        )
        tle.dsa.to_buffer(
            tl.zeros((5, SUB_M), dtype=tl.float32), bind_buffer=state1_buf
        )

        for index in tl.range(core, TASKS, CORES, disable_licm=True):
            task = index
            if STRIDE:
                task = tl.load(Plan + core * STRIDE + index // CORES)
            b = task // (H // band)
            head = task % (H // band) * band
            if S == 2:
                q_sub = sub_id
                h_sub = head
                row_base = (b.to(tl.int64) * 2 + q_sub) * H + h_sub
            else:
                q_sub = 0
                h_sub = head + sub
                row_base = b.to(tl.int64) * H + h_sub

            n = tl.load(Lens + b)
            groups = tl.cdiv(n, G_TOKENS)
            tab_base = b.to(tl.int64) * WIDTH

            if groups == 0:
                empty_lse_off = (b.to(tl.int64) * H + h_sub + offs_sub) * S + q_sub
                tl.store(
                    Lse + empty_lse_off, tl.full((SUB_M,), float("inf"), tl.float32)
                )
                for t in tl.static_range(SUB_M // EMPTY_TILE_R):
                    r_idx = row_base + t * EMPTY_TILE_R + offs_er
                    tl.store(
                        Out + r_idx[:, None] * 512 + offs_256[None, :],
                        tl.zeros((EMPTY_TILE_R, 256), dtype=tl.bfloat16),
                    )
                    tl.store(
                        Out + r_idx[:, None] * 512 + (256 + offs_256)[None, :],
                        tl.zeros((EMPTY_TILE_R, 256), dtype=tl.bfloat16),
                    )
            else:
                causal_sub = 1 if (S == 2 and CAUSAL and sub_id == 0) else 0

                for group in tl.range(0, groups, disable_licm=True):
                    slot = group % 2
                    prev_slot = 1 - slot
                    slot_idx = (core * 2 + slot).to(tl.int64)
                    visible = n - group * G_TOKENS
                    valid = tl.maximum(visible - causal_sub, 0)
                    use_fast_1d = (M >= 64) & (n > 1024) & (visible > 64)

                    stage_sync(slot, 0, True)

                    c_sub_base = slot_idx * (M * G_TOKENS) + sub * G_TOKENS
                    p_hi_base = slot_idx * (2 * M * G_TOKENS) + sub * G_TOKENS
                    p_lo_base = p_hi_base + M * G_TOKENS
                    tab_grp_u64 = (Table + tab_base + group * G_PAGES).to(tl.uint64)
                    flag_u64 = tl.where(
                        use_fast_1d,
                        (Flags + slot_idx * 16 + sub_id * 8).to(tl.uint64),
                        tl.cast(0, tl.uint64),
                    )

                    f_first = tl.where(group == 0, 1, 0)
                    f_force2d = tl.where(use_fast_1d, 0, 4)
                    f_rnorm = 8 if USE_FP16_PV else 0
                    f_grpfast = tl.where(use_fast_1d & (n >= 3072), 16, 0)
                    osq_flags = (f_first | f_force2d | f_rnorm | f_grpfast).to(
                        tl.uint32
                    )
                    out_scale_f = DEQ_INV_SCALE_FP16 if USE_FP16_PV else 1.0

                    if slot == 0:
                        st_res, _osq_w = tle_raw.call(
                            "ascend_online_softmax_quant_s8",
                            (C + c_sub_base).to(tl.uint64),
                            (R + c_sub_base).to(tl.uint64),
                            (QS + row_base).to(tl.uint64),
                            KS.to(tl.uint64),
                            tab_grp_u64,
                            flag_u64,
                            tle.dsa.to_tensor(ks_dummy, writable=False),
                            tle.dsa.to_tensor(state1_buf, writable=False),
                            (P + p_hi_base).to(tl.uint64),
                            (P + p_lo_base).to(tl.uint64),
                            SUB_M,
                            G_TOKENS,
                            valid.to(tl.uint32),
                            SCALE,
                            out_scale_f,
                            osq_flags,
                            out=(
                                tle.dsa.to_tensor(state0_buf, writable=True),
                                tle.dsa.to_tensor(shared_work_buf, writable=True),
                            ),
                        )
                        tle.dsa.to_buffer(st_res, bind_buffer=state0_buf)
                    else:
                        st_res, _osq_w = tle_raw.call(
                            "ascend_online_softmax_quant_s8",
                            (C + c_sub_base).to(tl.uint64),
                            (R + c_sub_base).to(tl.uint64),
                            (QS + row_base).to(tl.uint64),
                            KS.to(tl.uint64),
                            tab_grp_u64,
                            flag_u64,
                            tle.dsa.to_tensor(ks_dummy, writable=False),
                            tle.dsa.to_tensor(state0_buf, writable=False),
                            (P + p_hi_base).to(tl.uint64),
                            (P + p_lo_base).to(tl.uint64),
                            SUB_M,
                            G_TOKENS,
                            valid.to(tl.uint32),
                            SCALE,
                            out_scale_f,
                            osq_flags,
                            out=(
                                tle.dsa.to_tensor(state1_buf, writable=True),
                                tle.dsa.to_tensor(shared_work_buf, writable=True),
                            ),
                        )
                        tle.dsa.to_buffer(st_res, bind_buffer=state1_buf)

                    stage_sync(slot, 1, False)

                    if group > 0:
                        stage_sync(prev_slot, 2, True)
                        prior_idx = (core * 2 + prev_slot).to(tl.int64)
                        pv_hi_base = prior_idx * (2 * M * 512) + sub * 512
                        pv_lo_base = pv_hi_base + M * 512
                        pv_hi_u64 = (
                            (PV_h + pv_hi_base).to(tl.uint64)
                            if USE_FP16_PV
                            else (PV + pv_hi_base).to(tl.uint64)
                        )
                        pv_lo_u64 = (
                            (PV_h + pv_lo_base).to(tl.uint64)
                            if USE_FP16_PV
                            else (PV + pv_lo_base).to(tl.uint64)
                        )
                        upd_flags = (tl.where(group == 1, 1, 0) | 16).to(tl.uint32)
                        if prev_slot == 0:
                            acc_res, _upd_w = tle_raw.call(
                                "ascend_online_rescaled_accum",
                                pv_hi_u64,
                                pv_lo_u64,
                                tl.cast(0, tl.uint64),
                                tle.dsa.to_tensor(state0_buf, writable=False),
                                tl.cast(0, tl.uint64),
                                SUB_M,
                                512,
                                0.0,
                                1.0 / 127.0,
                                upd_flags,
                                out=(
                                    tle.dsa.to_tensor(acc_buf, writable=True),
                                    tle.dsa.to_tensor(shared_work_buf, writable=True),
                                ),
                            )
                        else:
                            acc_res, _upd_w = tle_raw.call(
                                "ascend_online_rescaled_accum",
                                pv_hi_u64,
                                pv_lo_u64,
                                tl.cast(0, tl.uint64),
                                tle.dsa.to_tensor(state1_buf, writable=False),
                                tl.cast(0, tl.uint64),
                                SUB_M,
                                512,
                                0.0,
                                1.0 / 127.0,
                                upd_flags,
                                out=(
                                    tle.dsa.to_tensor(acc_buf, writable=True),
                                    tle.dsa.to_tensor(shared_work_buf, writable=True),
                                ),
                            )
                        tle.dsa.to_buffer(acc_res, bind_buffer=acc_buf)

                last = (groups - 1) % 2
                last_idx = (core * 2 + last).to(tl.int64)
                stage_sync(last, 2, True)
                pv_hi_base = last_idx * (2 * M * 512) + sub * 512
                pv_lo_base = pv_hi_base + M * 512
                first_empty = (n == 1) & (S == 2) & CAUSAL & (sub_id == 0)

                pv_hi_u64 = (
                    (PV_h + pv_hi_base).to(tl.uint64)
                    if USE_FP16_PV
                    else (PV + pv_hi_base).to(tl.uint64)
                )
                pv_lo_u64 = (
                    (PV_h + pv_lo_base).to(tl.uint64)
                    if USE_FP16_PV
                    else (PV + pv_lo_base).to(tl.uint64)
                )
                last_flags = (
                    tl.where(groups == 1, 1, 0)
                    | 2
                    | tl.where(first_empty, 4, 0)
                    | (0 if USE_FP16_PV else 8)
                    | 16
                ).to(tl.uint32)
                if last == 0:
                    acc_last, _upd_last_w = tle_raw.call(
                        "ascend_online_rescaled_accum",
                        pv_hi_u64,
                        pv_lo_u64,
                        tl.cast(0, tl.uint64),
                        tle.dsa.to_tensor(state0_buf, writable=False),
                        (Out + row_base * 512).to(tl.uint64),
                        SUB_M,
                        512,
                        0.0,
                        1.0 / 127.0,
                        last_flags,
                        out=(
                            tle.dsa.to_tensor(acc_buf, writable=True),
                            tle.dsa.to_tensor(shared_work_buf, writable=True),
                        ),
                    )
                else:
                    acc_last, _upd_last_w = tle_raw.call(
                        "ascend_online_rescaled_accum",
                        pv_hi_u64,
                        pv_lo_u64,
                        tl.cast(0, tl.uint64),
                        tle.dsa.to_tensor(state1_buf, writable=False),
                        (Out + row_base * 512).to(tl.uint64),
                        SUB_M,
                        512,
                        0.0,
                        1.0 / 127.0,
                        last_flags,
                        out=(
                            tle.dsa.to_tensor(acc_buf, writable=True),
                            tle.dsa.to_tensor(shared_work_buf, writable=True),
                        ),
                    )
                tle.dsa.to_buffer(acc_last, bind_buffer=acc_buf)

                m0 = tle.dsa.to_tensor(
                    tle.dsa.subview(state0_buf, [0, 0], [1, SUB_M], [1, 1]),
                    writable=False,
                )
                m1 = tle.dsa.to_tensor(
                    tle.dsa.subview(state1_buf, [0, 0], [1, SUB_M], [1, 1]),
                    writable=False,
                )
                l0 = tle.dsa.to_tensor(
                    tle.dsa.subview(state0_buf, [1, 0], [1, SUB_M], [1, 1]),
                    writable=False,
                )
                l1 = tle.dsa.to_tensor(
                    tle.dsa.subview(state1_buf, [1, 0], [1, SUB_M], [1, 1]),
                    writable=False,
                )
                m_v = tl.where(last == 0, m0, m1).reshape(SUB_M)
                l_v = tl.where(last == 0, l0, l1).reshape(SUB_M)
                lse_v = tl.where(
                    first_empty,
                    tl.full((SUB_M,), float("inf"), dtype=tl.float32),
                    m_v + tl.log(tl.maximum(l_v, 1.0e-30)),
                )
                lse_off = (b.to(tl.int64) * H + h_sub + offs_sub) * S + q_sub
                tl.store(Lse + lse_off, lse_v)
            al.sync_block_set("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
