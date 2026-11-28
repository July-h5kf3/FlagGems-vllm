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
    "matmul_l1_to_gm",
    "matmul_l1_to_gm_deqf16",
    "online_softmax_quant_s8",
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

# Single fixed FP32 dequantization scale (2^-14) shared across vsum, PV_hi,
# and PV_lo on M >= 64 so Fixpipe DEQF16 outputs stay within FP16 range.
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
    PV_i8,
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
    core = tl.program_id(0)

    # Per-core scratch layout after PV ping-pong buffers (4 * M * 512 int32s/core):
    # - Flags:   CORES * 32 int32s (16 int32s per (core, slot))
    # - Vsum_h:  CORES * 16384 fp16s (8192 fp16s = 16x512 per (core, slot))
    # - Ones_i8: CORES * 6144 int8s (16x384 ones matrix for shifted vsum)
    # - PairIds: CORES * 8 int32s (query pair indices when S == 2)
    # - Plan:    CORES * STRIDE int32s (LPT task schedule)
    Flags = PV + CORES * (4 * M * 512)
    Vsum_h = PV_h + CORES * (8 * M * 512 + 64)
    Ones_i8 = PV_i8 + CORES * (16 * M * 512 + 32896)
    PairIds = Flags + CORES * (32 + 10000)
    Plan = Flags + CORES * (32 + 16896)

    GROUPS_PER_REQ: tl.constexpr = (H * S) // M
    SUB_M: tl.constexpr = M // 2
    band: tl.constexpr = M // S
    ROW_TILE_UPD: tl.constexpr = 16 if SUB_M >= 16 else 8
    NUM_TILES_UPD: tl.constexpr = SUB_M // ROW_TILE_UPD
    OSQ_TILE_R: tl.constexpr = 16 if SUB_M >= 16 else 8
    OSQ_NBANKS: tl.constexpr = 2 if (SUB_M // OSQ_TILE_R) >= 2 else 1
    OSQ_WORK_F: tl.constexpr = (
        OSQ_NBANKS * 2 * OSQ_TILE_R * 384
        + ((SUB_M + 63) & ~63)
        + 384
        + 192
        + OSQ_TILE_R * 64
        + 512
    )

    with al.scope(core_mode="cube"):
        al.sync_block_wait("vector", "cube", 7, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
        al.sync_block_set("cube", "vector", 7, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)

        k0_buf = tle.dsa.alloc([384, 512], tl.int8, tle.dsa.ascend.L1)
        k1_buf = tle.dsa.alloc([384, 512], tl.int8, tle.dsa.ascend.L1)
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
            groups = tl.cdiv(n, 384)
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

                c_base0 = (core * 2).to(tl.int64) * M * 384
                tab_off0 = b.to(tl.int64) * WIDTH
                rem0 = tl.minimum(pages, 6)
                kr_buf0 = tle.dsa.alloc([384, 64], tl.bfloat16, tle.dsa.ascend.L1)
                k0_init = tle_raw.call(
                    "ascend_gather_blocks_gm_to_l1",
                    K.to(tl.uint64),
                    (Table + tab_off0).to(tl.uint64),
                    6,
                    rem0,
                    64,
                    512,
                    64 * 512,
                    out=tle.dsa.to_tensor(k0_buf, writable=True),
                )
                tle.dsa.to_buffer(k0_init, bind_buffer=k0_buf)
                kr0_init = tle_raw.call(
                    "ascend_gather_blocks_gm_to_l1",
                    KR.to(tl.uint64),
                    (Table + tab_off0).to(tl.uint64),
                    6,
                    rem0,
                    64,
                    64,
                    64 * 64,
                    out=tle.dsa.to_tensor(kr_buf0, writable=True),
                )
                tle_raw.call(
                    "ascend_matmul_l1_to_gm",
                    q_l1,
                    tle.dsa.to_tensor(k0_buf, writable=False),
                    (C + c_base0).to(tl.uint64),
                    M,
                    384,
                    512,
                    384,
                    1,
                )
                tle_raw.call(
                    "ascend_matmul_l1_to_gm",
                    qr_l1,
                    kr0_init,
                    (R + c_base0).to(tl.uint64),
                    M,
                    384,
                    64,
                    384,
                    1,
                )
                stage_sync(0, 0, False)

                for group in tl.range(0, groups, disable_licm=True):
                    slot = group % 2
                    if group + 1 < groups:
                        next_slot = 1 - slot
                        c_base_n = (core * 2 + next_slot).to(tl.int64) * M * 384
                        tab_off_n = b.to(tl.int64) * WIDTH + (group + 1) * 6
                        rem_n = tl.minimum(pages - (group + 1) * 6, 6)
                        kr_buf_n = tle.dsa.alloc(
                            [384, 64], tl.bfloat16, tle.dsa.ascend.L1
                        )
                        if slot == 0:
                            k_n = tle_raw.call(
                                "ascend_gather_blocks_gm_to_l1",
                                K.to(tl.uint64),
                                (Table + tab_off_n).to(tl.uint64),
                                6,
                                rem_n,
                                64,
                                512,
                                64 * 512,
                                out=tle.dsa.to_tensor(k1_buf, writable=True),
                            )
                            tle.dsa.to_buffer(k_n, bind_buffer=k1_buf)
                            kr_n = tle_raw.call(
                                "ascend_gather_blocks_gm_to_l1",
                                KR.to(tl.uint64),
                                (Table + tab_off_n).to(tl.uint64),
                                6,
                                rem_n,
                                64,
                                64,
                                64 * 64,
                                out=tle.dsa.to_tensor(kr_buf_n, writable=True),
                            )
                            tle_raw.call(
                                "ascend_matmul_l1_to_gm",
                                q_l1,
                                tle.dsa.to_tensor(k1_buf, writable=False),
                                (C + c_base_n).to(tl.uint64),
                                M,
                                384,
                                512,
                                384,
                                1,
                            )
                        else:
                            k_n = tle_raw.call(
                                "ascend_gather_blocks_gm_to_l1",
                                K.to(tl.uint64),
                                (Table + tab_off_n).to(tl.uint64),
                                6,
                                rem_n,
                                64,
                                512,
                                64 * 512,
                                out=tle.dsa.to_tensor(k0_buf, writable=True),
                            )
                            tle.dsa.to_buffer(k_n, bind_buffer=k0_buf)
                            kr_n = tle_raw.call(
                                "ascend_gather_blocks_gm_to_l1",
                                KR.to(tl.uint64),
                                (Table + tab_off_n).to(tl.uint64),
                                6,
                                rem_n,
                                64,
                                64,
                                64 * 64,
                                out=tle.dsa.to_tensor(kr_buf_n, writable=True),
                            )
                            tle_raw.call(
                                "ascend_matmul_l1_to_gm",
                                q_l1,
                                tle.dsa.to_tensor(k0_buf, writable=False),
                                (C + c_base_n).to(tl.uint64),
                                M,
                                384,
                                512,
                                384,
                                1,
                            )
                        tle_raw.call(
                            "ascend_matmul_l1_to_gm",
                            qr_l1,
                            kr_n,
                            (R + c_base_n).to(tl.uint64),
                            M,
                            384,
                            64,
                            384,
                            1,
                        )
                        stage_sync(next_slot, 0, False)

                    stage_sync(slot, 1, True)
                    # Shifted 8-bit P_hi with vsum compensation covers [0, 255] on
                    # long windows; short contexts and tails use exact 2-digit INT8.
                    use_shifted = (M >= 64) & (n > 1024) & ((n - group * 384) > 64)
                    slot_idx_c = (core * 2 + slot).to(tl.int64)
                    p_hi_base_c = slot_idx_c * (2 * M * 384)
                    p_lo_base_c = p_hi_base_c + M * 384
                    pv_hi_base_c = slot_idx_c * (2 * M * 512)
                    pv_lo_base_c = pv_hi_base_c + M * 512

                    need_lo = 1
                    if use_shifted:
                        f0 = tl.load(Flags + slot_idx_c * 16)
                        f1 = tl.load(Flags + slot_idx_c * 16 + 8)
                        need_lo = tl.where((f0 | f1) != 0, 1, 0)

                    if USE_FP16_PV:
                        if use_shifted:
                            # Allocate ones_buf locally so it aliases the 48 KiB
                            # kr_buf/p_buf slot and M == 128 stays within 512 KiB L1.
                            ones_buf = tle.dsa.alloc(
                                [16, 384], tl.int8, tle.dsa.ascend.L1
                            )
                            ones_l1 = tle_raw.call(
                                "ascend_nd2nz_gm_to_l1",
                                (Ones_i8 + core.to(tl.int64) * 6144).to(tl.uint64),
                                16,
                                384,
                                384,
                                out=tle.dsa.to_tensor(ones_buf, writable=True),
                            )
                            if slot == 0:
                                tle_raw.call(
                                    "ascend_matmul_l1_to_gm_deqf16",
                                    ones_l1,
                                    tle.dsa.to_tensor(k0_buf, writable=False),
                                    (Vsum_h + slot_idx_c * 8192).to(tl.uint64),
                                    16,
                                    512,
                                    384,
                                    512,
                                    0,
                                    DEQ_SCALE_FP16,
                                )
                            else:
                                tle_raw.call(
                                    "ascend_matmul_l1_to_gm_deqf16",
                                    ones_l1,
                                    tle.dsa.to_tensor(k1_buf, writable=False),
                                    (Vsum_h + slot_idx_c * 8192).to(tl.uint64),
                                    16,
                                    512,
                                    384,
                                    512,
                                    0,
                                    DEQ_SCALE_FP16,
                                )
                        p_buf = tle.dsa.alloc([M, 384], tl.int8, tle.dsa.ascend.L1)
                        p_hi_l1 = tle_raw.call(
                            "ascend_nd2nz_gm_to_l1",
                            (P + p_hi_base_c).to(tl.uint64),
                            M,
                            384,
                            384,
                            out=tle.dsa.to_tensor(p_buf, writable=True),
                        )
                        if slot == 0:
                            tle_raw.call(
                                "ascend_matmul_l1_to_gm_deqf16",
                                p_hi_l1,
                                tle.dsa.to_tensor(k0_buf, writable=False),
                                (PV_h + pv_hi_base_c).to(tl.uint64),
                                M,
                                512,
                                384,
                                512,
                                0,
                                DEQ_SCALE_FP16,
                            )
                        else:
                            tle_raw.call(
                                "ascend_matmul_l1_to_gm_deqf16",
                                p_hi_l1,
                                tle.dsa.to_tensor(k1_buf, writable=False),
                                (PV_h + pv_hi_base_c).to(tl.uint64),
                                M,
                                512,
                                384,
                                512,
                                0,
                                DEQ_SCALE_FP16,
                            )
                        if need_lo != 0:
                            p_lo_l1 = tle_raw.call(
                                "ascend_nd2nz_gm_to_l1",
                                (P + p_lo_base_c).to(tl.uint64),
                                M,
                                384,
                                384,
                                out=tle.dsa.to_tensor(p_buf, writable=True),
                            )
                            if slot == 0:
                                tle_raw.call(
                                    "ascend_matmul_l1_to_gm_deqf16",
                                    p_lo_l1,
                                    tle.dsa.to_tensor(k0_buf, writable=False),
                                    (PV_h + pv_lo_base_c).to(tl.uint64),
                                    M,
                                    512,
                                    384,
                                    512,
                                    0,
                                    DEQ_SCALE_FP16,
                                )
                            else:
                                tle_raw.call(
                                    "ascend_matmul_l1_to_gm_deqf16",
                                    p_lo_l1,
                                    tle.dsa.to_tensor(k1_buf, writable=False),
                                    (PV_h + pv_lo_base_c).to(tl.uint64),
                                    M,
                                    512,
                                    384,
                                    512,
                                    0,
                                    DEQ_SCALE_FP16,
                                )
                    else:
                        p_buf = tle.dsa.alloc([M, 384], tl.int8, tle.dsa.ascend.L1)
                        p_hi_l1 = tle_raw.call(
                            "ascend_nd2nz_gm_to_l1",
                            (P + p_hi_base_c).to(tl.uint64),
                            M,
                            384,
                            384,
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
                                384,
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
                                384,
                                512,
                                0,
                            )
                        if need_lo != 0:
                            p_lo_l1 = tle_raw.call(
                                "ascend_nd2nz_gm_to_l1",
                                (P + p_lo_base_c).to(tl.uint64),
                                M,
                                384,
                                384,
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
                                    384,
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
                                    384,
                                    512,
                                    0,
                                )

                    stage_sync(slot, 2, False)
            al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)

    with al.scope(core_mode="vector"):
        if al.sub_vec_id() == 0:
            if USE_FP16_PV:
                ones_offs = tl.arange(0, 6144)
                tl.store(
                    Ones_i8 + core.to(tl.int64) * 6144 + ones_offs,
                    tl.full((6144,), 1, tl.int8),
                )
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
                    cost = lengths // 384 + (lengths % 384 != 0).to(tl.int32)
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

        sub_id = al.sub_vec_id()
        sub = sub_id * SUB_M
        offs_r = tl.arange(0, ROW_TILE_UPD)
        offs_64 = tl.arange(0, 64)
        offs_256 = tl.arange(0, 256)
        offs_sub = tl.arange(0, SUB_M)

        ks_buf = tle.dsa.alloc([1, 384], tl.float32, tle.dsa.ascend.UB)
        if USE_FP16_PV:
            acc_buf = tle.dsa.alloc([SUB_M, 512], tl.float16, tle.dsa.ascend.UB)
        else:
            acc_buf = tle.dsa.alloc([SUB_M, 512], tl.float32, tle.dsa.ascend.UB)
        state0_buf = tle.dsa.alloc([5, SUB_M], tl.float32, tle.dsa.ascend.UB)
        state1_buf = tle.dsa.alloc([5, SUB_M], tl.float32, tle.dsa.ascend.UB)
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
            pages = tl.cdiv(n, 64)
            groups = tl.cdiv(n, 384)
            tab_base = b.to(tl.int64) * WIDTH

            if groups == 0:
                empty_lse_off = (b.to(tl.int64) * H + h_sub + offs_sub) * S + q_sub
                tl.store(
                    Lse + empty_lse_off, tl.full((SUB_M,), float("inf"), tl.float32)
                )
                for t in tl.static_range(NUM_TILES_UPD):
                    r_idx = row_base + t * ROW_TILE_UPD + offs_r
                    tl.store(
                        Out + r_idx[:, None] * 512 + offs_256[None, :],
                        tl.zeros((ROW_TILE_UPD, 256), dtype=tl.bfloat16),
                    )
                    tl.store(
                        Out + r_idx[:, None] * 512 + (256 + offs_256)[None, :],
                        tl.zeros((ROW_TILE_UPD, 256), dtype=tl.bfloat16),
                    )
            else:
                causal_sub = 1 if (S == 2 and CAUSAL and sub_id == 0) else 0

                for group in tl.range(0, groups, disable_licm=True):
                    slot = group % 2
                    prev_slot = 1 - slot
                    slot_idx = (core * 2 + slot).to(tl.int64)
                    visible = n - group * 384
                    valid = tl.maximum(visible - causal_sub, 0)
                    use_shifted = (M >= 64) & (n > 1024) & (visible > 64)

                    stage_sync(slot, 0, True)

                    offs_4 = tl.arange(0, 4)
                    pg4 = tl.load(
                        Table + tab_base + tl.minimum(group * 6 + offs_4, pages - 1)
                    ).to(tl.int64)
                    m4 = (offs_4[:, None] * 64 + offs_64[None, :]) < visible
                    ks4 = tl.load(
                        KS + pg4[:, None] * 64 + offs_64[None, :], mask=m4, other=0.0
                    ).reshape(1, 256)
                    tle.dsa.to_buffer(
                        ks4,
                        bind_buffer=tle.dsa.subview(ks_buf, [0, 0], [1, 256], [1, 1]),
                    )
                    offs_2 = tl.arange(0, 2)
                    pg2 = tl.load(
                        Table + tab_base + tl.minimum(group * 6 + 4 + offs_2, pages - 1)
                    ).to(tl.int64)
                    m2 = ((4 + offs_2)[:, None] * 64 + offs_64[None, :]) < visible
                    ks2 = tl.load(
                        KS + pg2[:, None] * 64 + offs_64[None, :], mask=m2, other=0.0
                    ).reshape(1, 128)
                    tle.dsa.to_buffer(
                        ks2,
                        bind_buffer=tle.dsa.subview(ks_buf, [0, 256], [1, 128], [1, 1]),
                    )

                    c_sub_base = slot_idx * (M * 384) + sub * 384
                    p_hi_base = slot_idx * (2 * M * 384) + sub * 384
                    p_lo_base = p_hi_base + M * 384

                    f_first = tl.where(group == 0, 1, 0)
                    f_shifted = tl.where(use_shifted, 2, 0)
                    f_force2d = tl.where(use_shifted, 0, 4)
                    f_rnorm = 8 if USE_FP16_PV else 0
                    osq_flags = (f_first | f_shifted | f_force2d | f_rnorm).to(
                        tl.uint32
                    )
                    out_scale_f = DEQ_INV_SCALE_FP16 if USE_FP16_PV else 1.0

                    work_buf = tle.dsa.alloc(
                        [OSQ_WORK_F], tl.float32, tle.dsa.ascend.UB
                    )
                    if slot == 0:
                        st_res, _ = tle_raw.call(
                            "ascend_online_softmax_quant_s8",
                            (C + c_sub_base).to(tl.uint64),
                            (R + c_sub_base).to(tl.uint64),
                            (QS + row_base).to(tl.uint64),
                            tle.dsa.to_tensor(ks_buf, writable=False),
                            tle.dsa.to_tensor(state1_buf, writable=False),
                            (P + p_hi_base).to(tl.uint64),
                            (P + p_lo_base).to(tl.uint64),
                            SUB_M,
                            384,
                            valid.to(tl.uint32),
                            SCALE,
                            out_scale_f,
                            osq_flags,
                            out=(
                                tle.dsa.to_tensor(state0_buf, writable=True),
                                tle.dsa.to_tensor(work_buf, writable=True),
                            ),
                        )
                        tle.dsa.to_buffer(st_res, bind_buffer=state0_buf)
                    else:
                        st_res, _ = tle_raw.call(
                            "ascend_online_softmax_quant_s8",
                            (C + c_sub_base).to(tl.uint64),
                            (R + c_sub_base).to(tl.uint64),
                            (QS + row_base).to(tl.uint64),
                            tle.dsa.to_tensor(ks_buf, writable=False),
                            tle.dsa.to_tensor(state0_buf, writable=False),
                            (P + p_hi_base).to(tl.uint64),
                            (P + p_lo_base).to(tl.uint64),
                            SUB_M,
                            384,
                            valid.to(tl.uint32),
                            SCALE,
                            out_scale_f,
                            osq_flags,
                            out=(
                                tle.dsa.to_tensor(state1_buf, writable=True),
                                tle.dsa.to_tensor(work_buf, writable=True),
                            ),
                        )
                        tle.dsa.to_buffer(st_res, bind_buffer=state1_buf)

                    if use_shifted:
                        # Mark read-only state subviews writable=False so scalar
                        # tl.where lowering to tensor.insert_slice cannot mutate
                        # state1_buf in place during one-shot bufferization.
                        n0 = tle.dsa.to_tensor(
                            tle.dsa.subview(state0_buf, [4, 0], [1, 8], [1, 1]),
                            writable=False,
                        )
                        n1 = tle.dsa.to_tensor(
                            tle.dsa.subview(state1_buf, [4, 0], [1, 8], [1, 1]),
                            writable=False,
                        )
                        cur_need_f = tl.max(tl.where(slot == 0, n0, n1))
                        sub_need_lo = tl.where(cur_need_f > 0.5, 1, 0).to(tl.int32)
                        tl.store(Flags + slot_idx * 16 + sub_id * 8, sub_need_lo)

                    stage_sync(slot, 1, False)

                    if group > 0:
                        stage_sync(prev_slot, 2, True)
                        prior_idx = (core * 2 + prev_slot).to(tl.int64)
                        pv_hi_base = prior_idx * (2 * M * 512) + sub * 512
                        pv_lo_base = pv_hi_base + M * 512
                        prior_shifted = (M >= 64) & (n > 1024)
                        pn0 = tle.dsa.to_tensor(
                            tle.dsa.subview(state0_buf, [4, 0], [1, 8], [1, 1]),
                            writable=False,
                        )
                        pn1 = tle.dsa.to_tensor(
                            tle.dsa.subview(state1_buf, [4, 0], [1, 8], [1, 1]),
                            writable=False,
                        )
                        prior_lo = tl.where(
                            tl.max(tl.where(prev_slot == 0, pn0, pn1)) > 0.5, 1, 0
                        )

                        for col in range(2):
                            if USE_FP16_PV:
                                vsum_c = (
                                    tl.load(
                                        Vsum_h + prior_idx * 8192 + col * 256 + offs_256
                                    )
                                    * tl.full((256,), 128.0, tl.float16)
                                    if prior_shifted
                                    else tl.zeros((256,), dtype=tl.float16)
                                )
                            for t in range(NUM_TILES_UPD):
                                t_off = t * ROW_TILE_UPD
                                acc_sv = tle.dsa.subview(
                                    acc_buf,
                                    [t_off, col * 256],
                                    [ROW_TILE_UPD, 256],
                                    [1, 1],
                                )
                                pv_off = (t_off + offs_r)[:, None] * 512 + (
                                    col * 256 + offs_256
                                )[None, :]
                                a0 = tle.dsa.to_tensor(
                                    tle.dsa.subview(
                                        state0_buf,
                                        [2, t_off],
                                        [1, ROW_TILE_UPD],
                                        [1, 1],
                                    ),
                                    writable=False,
                                )
                                a1 = tle.dsa.to_tensor(
                                    tle.dsa.subview(
                                        state1_buf,
                                        [2, t_off],
                                        [1, ROW_TILE_UPD],
                                        [1, 1],
                                    ),
                                    writable=False,
                                )
                                s0 = tle.dsa.to_tensor(
                                    tle.dsa.subview(
                                        state0_buf,
                                        [3, t_off],
                                        [1, ROW_TILE_UPD],
                                        [1, 1],
                                    ),
                                    writable=False,
                                )
                                s1 = tle.dsa.to_tensor(
                                    tle.dsa.subview(
                                        state1_buf,
                                        [3, t_off],
                                        [1, ROW_TILE_UPD],
                                        [1, 1],
                                    ),
                                    writable=False,
                                )
                                alpha_t = tl.where(prev_slot == 0, a0, a1).reshape(
                                    ROW_TILE_UPD, 1
                                )
                                step_t = tl.where(prev_slot == 0, s0, s1).reshape(
                                    ROW_TILE_UPD, 1
                                )

                                if USE_FP16_PV:
                                    upd_h = tl.load(PV_h + pv_hi_base + pv_off)
                                    if prior_shifted:
                                        upd_h = upd_h + vsum_c[None, :]
                                    if prior_lo != 0:
                                        upd_h = upd_h + tl.load(
                                            PV_h + pv_lo_base + pv_off
                                        ) * tl.full((1, 1), 1.0 / 127.0, tl.float16)
                                    w_cur_h = step_t.to(tl.float16)
                                    if group == 1:
                                        acc_new = upd_h * w_cur_h
                                    else:
                                        acc_new = (
                                            tle.dsa.to_tensor(acc_sv, writable=False)
                                            * alpha_t.to(tl.float16)
                                            + upd_h * w_cur_h
                                        )
                                else:
                                    upd = tl.load(PV + pv_hi_base + pv_off).to(
                                        tl.float32
                                    )
                                    if prior_lo != 0:
                                        upd = upd + tl.load(
                                            PV + pv_lo_base + pv_off
                                        ).to(tl.float32) * (1.0 / 127.0)
                                    if group == 1:
                                        acc_new = upd * step_t
                                    else:
                                        acc_new = (
                                            tle.dsa.to_tensor(acc_sv, writable=False)
                                            * alpha_t
                                            + upd * step_t
                                        )
                                tle.dsa.to_buffer(acc_new, bind_buffer=acc_sv)

                last = (groups - 1) % 2
                last_idx = (core * 2 + last).to(tl.int64)
                stage_sync(last, 2, True)
                pv_hi_base = last_idx * (2 * M * 512) + sub * 512
                pv_lo_base = pv_hi_base + M * 512
                last_shifted = (M >= 64) & (n > 1024) & ((n - (groups - 1) * 384) > 64)
                ln0 = tle.dsa.to_tensor(
                    tle.dsa.subview(state0_buf, [4, 0], [1, 8], [1, 1]),
                    writable=False,
                )
                ln1 = tle.dsa.to_tensor(
                    tle.dsa.subview(state1_buf, [4, 0], [1, 8], [1, 1]),
                    writable=False,
                )
                last_lo = tl.where(tl.max(tl.where(last == 0, ln0, ln1)) > 0.5, 1, 0)
                first_empty = (n == 1) & (S == 2) & CAUSAL & (sub_id == 0)

                for t in range(NUM_TILES_UPD):
                    t_off = t * ROW_TILE_UPD
                    m0 = tle.dsa.to_tensor(
                        tle.dsa.subview(
                            state0_buf, [0, t_off], [1, ROW_TILE_UPD], [1, 1]
                        ),
                        writable=False,
                    )
                    m1 = tle.dsa.to_tensor(
                        tle.dsa.subview(
                            state1_buf, [0, t_off], [1, ROW_TILE_UPD], [1, 1]
                        ),
                        writable=False,
                    )
                    l0 = tle.dsa.to_tensor(
                        tle.dsa.subview(
                            state0_buf, [1, t_off], [1, ROW_TILE_UPD], [1, 1]
                        ),
                        writable=False,
                    )
                    l1 = tle.dsa.to_tensor(
                        tle.dsa.subview(
                            state1_buf, [1, t_off], [1, ROW_TILE_UPD], [1, 1]
                        ),
                        writable=False,
                    )
                    a0 = tle.dsa.to_tensor(
                        tle.dsa.subview(
                            state0_buf, [2, t_off], [1, ROW_TILE_UPD], [1, 1]
                        ),
                        writable=False,
                    )
                    a1 = tle.dsa.to_tensor(
                        tle.dsa.subview(
                            state1_buf, [2, t_off], [1, ROW_TILE_UPD], [1, 1]
                        ),
                        writable=False,
                    )
                    s0 = tle.dsa.to_tensor(
                        tle.dsa.subview(
                            state0_buf, [3, t_off], [1, ROW_TILE_UPD], [1, 1]
                        ),
                        writable=False,
                    )
                    s1 = tle.dsa.to_tensor(
                        tle.dsa.subview(
                            state1_buf, [3, t_off], [1, ROW_TILE_UPD], [1, 1]
                        ),
                        writable=False,
                    )
                    m_t = tl.where(last == 0, m0, m1).reshape(ROW_TILE_UPD, 1)
                    l_t = tl.where(last == 0, l0, l1).reshape(ROW_TILE_UPD, 1)
                    alpha_t = tl.where(last == 0, a0, a1).reshape(ROW_TILE_UPD, 1)
                    step_t = tl.where(last == 0, s0, s1).reshape(ROW_TILE_UPD, 1)

                    if first_empty:
                        lse_t = tl.full(
                            (ROW_TILE_UPD, 1), float("inf"), dtype=tl.float32
                        )
                    else:
                        lse_t = m_t + tl.log(tl.maximum(l_t, 1.0e-30))
                    lse_off = (b.to(tl.int64) * H + h_sub + t_off + offs_r) * S + q_sub
                    tl.store(Lse + lse_off, lse_t.reshape(ROW_TILE_UPD))

                    inv_l = 1.0 / tl.maximum(l_t, 1.0e-30)
                    for col in range(2):
                        acc_sv = tle.dsa.subview(
                            acc_buf,
                            [t_off, col * 256],
                            [ROW_TILE_UPD, 256],
                            [1, 1],
                        )
                        pv_off = (t_off + offs_r)[:, None] * 512 + (
                            col * 256 + offs_256
                        )[None, :]
                        if USE_FP16_PV:
                            vsum_c = (
                                tl.load(Vsum_h + last_idx * 8192 + col * 256 + offs_256)
                                * tl.full((256,), 128.0, tl.float16)
                                if last_shifted
                                else tl.zeros((256,), dtype=tl.float16)
                            )
                            upd_h = tl.load(PV_h + pv_hi_base + pv_off)
                            if last_shifted:
                                upd_h = upd_h + vsum_c[None, :]
                            if last_lo != 0:
                                upd_h = upd_h + tl.load(
                                    PV_h + pv_lo_base + pv_off
                                ) * tl.full((1, 1), 1.0 / 127.0, tl.float16)
                            w_cur_h = step_t.to(tl.float16)
                            if groups == 1:
                                acc_new_h = upd_h * w_cur_h
                            else:
                                acc_new_h = (
                                    tle.dsa.to_tensor(acc_sv, writable=False)
                                    * alpha_t.to(tl.float16)
                                    + upd_h * w_cur_h
                                )
                            if first_empty:
                                out_val = tl.zeros(
                                    (ROW_TILE_UPD, 256), dtype=tl.bfloat16
                                )
                            else:
                                out_val = acc_new_h.to(tl.float32).to(tl.bfloat16)
                        else:
                            upd = tl.load(PV + pv_hi_base + pv_off).to(tl.float32)
                            if last_lo != 0:
                                upd = upd + tl.load(PV + pv_lo_base + pv_off).to(
                                    tl.float32
                                ) * (1.0 / 127.0)
                            if groups == 1:
                                acc_new = upd * step_t
                            else:
                                acc_new = (
                                    tle.dsa.to_tensor(acc_sv, writable=False) * alpha_t
                                    + upd * step_t
                                )
                            if first_empty:
                                out_val = tl.zeros(
                                    (ROW_TILE_UPD, 256), dtype=tl.bfloat16
                                )
                            else:
                                out_val = (acc_new * inv_l).to(tl.bfloat16)
                        r_idx = row_base + t_off + offs_r
                        tl.store(
                            Out
                            + r_idx[:, None] * 512
                            + (col * 256 + offs_256)[None, :],
                            out_val,
                        )
            al.sync_block_set("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
