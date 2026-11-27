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
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.runtime.backend._ascend.fused.attention import PIPE, stage_sync

CUSTOM_OPS_SPEC = importlib.util.find_spec(
    "triton.experimental.tle.language.dsa.ascend.custom_ops"
)
if CUSTOM_OPS_SPEC is not None:
    from triton.experimental.tle.language.dsa.ascend import (
        custom_ops as ascend_custom_ops,
    )

    HAS_CAST_FP16_TO_INT8_RINT = tl.constexpr(
        "cast_fp16_to_int8_rint" in ascend_custom_ops.__all__
    )
else:
    HAS_CAST_FP16_TO_INT8_RINT = tl.constexpr(False)

HAS_PAGED_INT8_OPS = True


@triton.jit
def quantize_to_int8(x_f32, ELEMS: tl.constexpr, USE_CAST_OP: tl.constexpr):
    if USE_CAST_OP:
        x_f16 = x_f32.to(tl.float16).reshape(ELEMS)
        return tle.dsa.ascend.raw(
            "cast_fp16_to_int8_rint",
            x_f16,
            ELEMS,
            out=tl.full((ELEMS,), 0, tl.int8),
        )
    return (
        tl.minimum(tl.maximum(tl.floor(x_f32 + 0.5), -128.0), 127.0)
        .to(tl.int8)
        .reshape(ELEMS)
    )


@triton.jit
def cube_window_qk(
    q,
    qr,
    K,
    KR,
    Table,
    C,
    R,
    b,
    WIDTH,
    core,
    group,
    slot,
    pages,
    M: tl.constexpr,
):
    rows_m = tl.arange(0, M)
    lanes = tl.arange(0, 64)
    cols_512 = tl.arange(0, 512)
    cols_64 = tl.arange(0, 64)
    c_base = (core * 2 + slot).to(tl.int64) * M * 384
    tab_base = b.to(tl.int64) * WIDTH
    for p in range(6):
        p_idx = tl.minimum(group * 6 + p, pages - 1)
        pg = tl.load(Table + tab_base + p_idx).to(tl.int64)
        tok = pg * 64 + lanes
        k_p = tl.load(K + tok[:, None] * 512 + cols_512[None, :])
        kr_p = tl.load(KR + tok[:, None] * 64 + cols_64[None, :])
        c_p = tl.dot(q, tl.trans(k_p), out_dtype=tl.int32)
        r_p = tl.dot(qr, tl.trans(kr_p), out_dtype=tl.float32)
        off = c_base + rows_m[:, None] * 384 + (p * 64 + lanes)[None, :]
        tl.store(C + off, c_p)
        tl.store(R + off, r_p)


@triton.jit
def cube_window_qk_s2(
    q0,
    q1,
    qr0,
    qr1,
    K,
    KR,
    Table,
    C,
    R,
    b,
    WIDTH,
    core,
    group,
    slot,
    pages,
    M: tl.constexpr,
):
    BAND: tl.constexpr = M // 2
    rows_b = tl.arange(0, BAND)
    lanes = tl.arange(0, 64)
    cols_512 = tl.arange(0, 512)
    cols_64 = tl.arange(0, 64)
    c_base0 = (core * 2 + slot).to(tl.int64) * M * 384
    c_base1 = c_base0 + BAND * 384
    tab_base = b.to(tl.int64) * WIDTH
    for p in range(6):
        p_idx = tl.minimum(group * 6 + p, pages - 1)
        pg = tl.load(Table + tab_base + p_idx).to(tl.int64)
        tok = pg * 64 + lanes
        k_p = tl.load(K + tok[:, None] * 512 + cols_512[None, :])
        kr_p = tl.load(KR + tok[:, None] * 64 + cols_64[None, :])
        c0 = tl.dot(q0, tl.trans(k_p), out_dtype=tl.int32)
        c1 = tl.dot(q1, tl.trans(k_p), out_dtype=tl.int32)
        r0 = tl.dot(qr0, tl.trans(kr_p), out_dtype=tl.float32)
        r1 = tl.dot(qr1, tl.trans(kr_p), out_dtype=tl.float32)
        off0 = c_base0 + rows_b[:, None] * 384 + (p * 64 + lanes)[None, :]
        off1 = c_base1 + rows_b[:, None] * 384 + (p * 64 + lanes)[None, :]
        tl.store(C + off0, c0)
        tl.store(C + off1, c1)
        tl.store(R + off0, r0)
        tl.store(R + off1, r1)


@triton.jit
def cube_window_pv(
    K,
    Table,
    P,
    PV,
    Flags,
    Vsum,
    b,
    WIDTH,
    core,
    group,
    slot,
    pages,
    use_shifted,
    M: tl.constexpr,
):
    rows_m = tl.arange(0, M)
    lanes = tl.arange(0, 64)
    PV_COLS: tl.constexpr = 4 if M == 128 else 2
    COL_W: tl.constexpr = 512 // PV_COLS
    cols = tl.arange(0, COL_W)
    slot_idx = (core * 2 + slot).to(tl.int64)
    p_hi_base = slot_idx * (2 * M * 384)
    p_lo_base = p_hi_base + M * 384
    pv_hi_base = slot_idx * (2 * M * 512)
    pv_lo_base = pv_hi_base + M * 512
    vsum_base = slot_idx * 512
    tab_base = b.to(tl.int64) * WIDTH

    need_lo = 1
    if use_shifted:
        f0 = tl.load(Flags + slot_idx * 16)
        f1 = tl.load(Flags + slot_idx * 16 + 8)
        need_lo = tl.where((f0 | f1) != 0, 1, 0)

    ones = tl.full((16, 64), 1, dtype=tl.int8)
    for col in tl.static_range(PV_COLS):
        pv_hi_acc = tl.zeros((M, COL_W), dtype=tl.int32)
        vsum_acc = tl.zeros((16, COL_W), dtype=tl.int32)
        for p in range(6):
            p_idx = tl.minimum(group * 6 + p, pages - 1)
            pg = tl.load(Table + tab_base + p_idx).to(tl.int64)
            tok = pg * 64 + lanes
            v_p = tl.load(K + tok[:, None] * 512 + (col * COL_W + cols)[None, :])
            p_off = rows_m[:, None] * 384 + (p * 64 + lanes)[None, :]
            p_hi_p = tl.load(P + p_hi_base + p_off)
            pv_hi_acc += tl.dot(p_hi_p, v_p, out_dtype=tl.int32)
            if use_shifted:
                vsum_acc += tl.dot(ones, v_p, out_dtype=tl.int32)
        out_off = rows_m[:, None] * 512 + (col * COL_W + cols)[None, :]
        tl.store(PV + pv_hi_base + out_off, pv_hi_acc)
        if use_shifted:
            vsum_row = tle.dsa.extract_slice(vsum_acc, [0, 0], [1, COL_W], [1, 1])
            tl.store(
                Vsum + vsum_base + (col * COL_W + cols)[None, :],
                vsum_row,
            )

    if need_lo != 0:
        for col in tl.static_range(PV_COLS):
            pv_lo_acc = tl.zeros((M, COL_W), dtype=tl.int32)
            for p in range(6):
                p_idx = tl.minimum(group * 6 + p, pages - 1)
                pg = tl.load(Table + tab_base + p_idx).to(tl.int64)
                tok = pg * 64 + lanes
                v_p = tl.load(K + tok[:, None] * 512 + (col * COL_W + cols)[None, :])
                p_off = rows_m[:, None] * 384 + (p * 64 + lanes)[None, :]
                p_lo_p = tl.load(P + p_lo_base + p_off)
                pv_lo_acc += tl.dot(p_lo_p, v_p, out_dtype=tl.int32)
            out_off = rows_m[:, None] * 512 + (col * COL_W + cols)[None, :]
            tl.store(PV + pv_lo_base + out_off, pv_lo_acc)


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
    USE_CAST_OP: tl.constexpr = HAS_CAST_FP16_TO_INT8_RINT
    # Per-core scratch layout after PV ping-pong buffers (in int32 words):
    # - PV ping-pong: CORES * 4 * M * 512 int32s
    # - Flags: CORES * 32 int32s (16 int32s per (core, slot))
    # - Vsum: CORES * 2 * 512 float32s (reinterpreted from int32 region)
    # - Plan: CORES * STRIDE int32s
    core = tl.program_id(0)
    Flags = PV + CORES * (4 * M * 512)
    Vsum = Flags + CORES * 32
    Plan = Flags + CORES * (32 + 1024)
    GROUPS_PER_REQ: tl.constexpr = (H * S) // M
    SUB_M: tl.constexpr = M // 2
    ROW_TILE: tl.constexpr = 8
    NUM_TILES: tl.constexpr = SUB_M // ROW_TILE
    band: tl.constexpr = M // S

    with al.scope(core_mode="cube"):
        al.sync_block_wait("vector", "cube", 7, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
        al.sync_block_set("cube", "vector", 7, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
        offs_m = tl.arange(0, M)
        cols_512 = tl.arange(0, 512)
        cols_64 = tl.arange(0, 64)
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
                if S == 2:
                    offs_b = tl.arange(0, band)
                    q_rows0 = b.to(tl.int64) * 2 * H + head + offs_b
                    q_rows1 = q_rows0 + H
                    q0 = tl.load(Q + q_rows0[:, None] * 512 + cols_512[None, :])
                    q1 = tl.load(Q + q_rows1[:, None] * 512 + cols_512[None, :])
                    qr0 = tl.load(QR + q_rows0[:, None] * 64 + cols_64[None, :])
                    qr1 = tl.load(QR + q_rows1[:, None] * 64 + cols_64[None, :])
                    cube_window_qk_s2(
                        q0,
                        q1,
                        qr0,
                        qr1,
                        K,
                        KR,
                        Table,
                        C,
                        R,
                        b,
                        WIDTH,
                        core,
                        0,
                        0,
                        pages,
                        M,
                    )
                else:
                    q_rows = b.to(tl.int64) * H + head + offs_m
                    q = tl.load(Q + q_rows[:, None] * 512 + cols_512[None, :])
                    qr = tl.load(QR + q_rows[:, None] * 64 + cols_64[None, :])
                    cube_window_qk(
                        q, qr, K, KR, Table, C, R, b, WIDTH, core, 0, 0, pages, M
                    )
                stage_sync(0, 0, False)
                for group in tl.range(0, groups, disable_licm=True):
                    slot = group % 2
                    if group + 1 < groups:
                        if S == 2:
                            cube_window_qk_s2(
                                q0,
                                q1,
                                qr0,
                                qr1,
                                K,
                                KR,
                                Table,
                                C,
                                R,
                                b,
                                WIDTH,
                                core,
                                group + 1,
                                1 - slot,
                                pages,
                                M,
                            )
                        else:
                            cube_window_qk(
                                q,
                                qr,
                                K,
                                KR,
                                Table,
                                C,
                                R,
                                b,
                                WIDTH,
                                core,
                                group + 1,
                                1 - slot,
                                pages,
                                M,
                            )
                        stage_sync(1 - slot, 0, False)
                    stage_sync(slot, 1, True)
                    use_shifted = (M >= 64) & (n > 1024) & ((n - group * 384) > 64)
                    cube_window_pv(
                        K,
                        Table,
                        P,
                        PV,
                        Flags,
                        Vsum,
                        b,
                        WIDTH,
                        core,
                        group,
                        slot,
                        pages,
                        use_shifted,
                        M,
                    )
                    stage_sync(slot, 2, False)
            al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)

    with al.scope(core_mode="vector"):
        if al.sub_vec_id() == 0:
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
        offs_r = tl.arange(0, ROW_TILE)
        offs_64 = tl.arange(0, 64)
        offs_128 = tl.arange(0, 128)
        offs_256 = tl.arange(0, 256)
        offs_sub = tl.arange(0, SUB_M)
        ELEMS: tl.constexpr = ROW_TILE * 128

        s_buf = tle.dsa.alloc([ROW_TILE, 384], tl.float32, tle.dsa.ascend.UB)
        ks_buf = tle.dsa.alloc([1, 384], tl.float32, tle.dsa.ascend.UB)
        acc_buf = tle.dsa.alloc([SUB_M, 512], tl.float32, tle.dsa.ascend.UB)
        m_buf = tle.dsa.alloc([2 * SUB_M, 1], tl.float32, tle.dsa.ascend.UB)
        l_buf = tle.dsa.alloc([2 * SUB_M, 1], tl.float32, tle.dsa.ascend.UB)
        alpha_buf = tle.dsa.alloc([2 * SUB_M, 1], tl.float32, tle.dsa.ascend.UB)
        step_buf = tle.dsa.alloc([2 * SUB_M, 1], tl.float32, tle.dsa.ascend.UB)
        need_lo_buf = tle.dsa.alloc([2, 8], tl.int32, tle.dsa.ascend.UB)

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
                for t in tl.static_range(NUM_TILES):
                    r_idx = row_base + t * ROW_TILE + offs_r
                    tl.store(
                        Out + r_idx[:, None] * 512 + offs_256[None, :],
                        tl.zeros((ROW_TILE, 256), dtype=tl.bfloat16),
                    )
                    tl.store(
                        Out + r_idx[:, None] * 512 + (256 + offs_256)[None, :],
                        tl.zeros((ROW_TILE, 256), dtype=tl.bfloat16),
                    )
            else:
                causal_drop = 1 if (S == 2 and CAUSAL) else 0
                causal_sub = tl.where(sub_id == 0, causal_drop, 0)
                tle.dsa.to_buffer(
                    tl.full((SUB_M, 1), -float("inf"), tl.float32),
                    bind_buffer=tle.dsa.subview(m_buf, [SUB_M, 0], [SUB_M, 1], [1, 1]),
                )
                tle.dsa.to_buffer(
                    tl.zeros((SUB_M, 1), tl.float32),
                    bind_buffer=tle.dsa.subview(l_buf, [SUB_M, 0], [SUB_M, 1], [1, 1]),
                )

                for group in tl.range(0, groups, disable_licm=True):
                    slot = group % 2
                    prev_slot = 1 - slot
                    slot_idx = (core * 2 + slot).to(tl.int64)
                    visible = n - group * 384
                    valid = tl.maximum(visible - causal_sub, 0)
                    use_shifted = (M >= 64) & (n > 1024) & (visible > 64)

                    stage_sync(slot, 0, True)

                    # Load 6 contiguous 64-token KS pages and detect outlier scale.
                    ks_max_v = tl.zeros((1, 64), dtype=tl.float32)
                    ks_sum_v = tl.zeros((1, 64), dtype=tl.float32)
                    for p in range(6):
                        p_idx = tl.minimum(group * 6 + p, pages - 1)
                        pg = tl.load(Table + tab_base + p_idx).to(tl.int64)
                        tok_mask = (p * 64 + offs_64) < visible
                        ks_p = tl.load(
                            KS + pg * 64 + offs_64, mask=tok_mask, other=0.0
                        )[None, :]
                        tle.dsa.to_buffer(
                            ks_p,
                            bind_buffer=tle.dsa.subview(
                                ks_buf, [0, p * 64], [1, 64], [1, 1]
                            ),
                        )
                        ks_max_v = tl.maximum(ks_max_v, ks_p)
                        ks_sum_v += ks_p
                    ks_max_s = tl.max(ks_max_v)
                    ks_sum_s = tl.sum(ks_sum_v)
                    eff_vis = tl.minimum(visible, 384).to(tl.float32)
                    ks_outlier = (ks_max_s * eff_vis) > (4.0 * ks_sum_s)

                    c_sub_base = (slot_idx * M + sub) * 384
                    p_hi_base = slot_idx * (2 * M * 384) + sub * 384
                    p_lo_base = p_hi_base + M * 384
                    sub_need_lo = 0

                    for t in range(NUM_TILES):
                        t_off = t * ROW_TILE
                        qs_t = tl.load(QS + row_base + t_off + offs_r) * SCALE
                        m_prev_sv = tle.dsa.subview(
                            m_buf, [prev_slot * SUB_M + t_off, 0], [ROW_TILE, 1], [1, 1]
                        )
                        l_prev_sv = tle.dsa.subview(
                            l_buf, [prev_slot * SUB_M + t_off, 0], [ROW_TILE, 1], [1, 1]
                        )
                        m_prev_t = tle.dsa.to_tensor(m_prev_sv)
                        l_prev_t = tle.dsa.to_tensor(l_prev_sv)

                        t_max = tl.full(
                            (ROW_TILE, 128), -float("inf"), dtype=tl.float32
                        )
                        for p in range(3):
                            ks_sv = tle.dsa.subview(
                                ks_buf, [0, p * 128], [1, 128], [1, 1]
                            )
                            ks_p = tle.dsa.to_tensor(ks_sv)
                            c_off = (
                                c_sub_base
                                + (t_off + offs_r)[:, None] * 384
                                + (p * 128 + offs_128)[None, :]
                            )
                            s_p = (
                                (tl.load(C + c_off).to(tl.float32) + tl.load(R + c_off))
                                * qs_t[:, None]
                                * ks_p
                            )
                            if valid < 384:
                                col_pos = p * 128 + offs_128
                                s_p = tl.where(
                                    col_pos[None, :] < valid, s_p, -float("inf")
                                )
                            tle.dsa.to_buffer(
                                s_p,
                                bind_buffer=tle.dsa.subview(
                                    s_buf, [0, p * 128], [ROW_TILE, 128], [1, 1]
                                ),
                            )
                            t_max = tl.maximum(t_max, s_p)

                        m_local = tl.max(t_max, axis=1, keep_dims=True)
                        safe_local = tl.maximum(m_local, -1.0e30)
                        if group == 0:
                            m_now = m_local
                            alpha_t = tl.zeros((ROW_TILE, 1), dtype=tl.float32)
                            beta_t = tl.full((ROW_TILE, 1), 1.0, dtype=tl.float32)
                        else:
                            m_now = tl.maximum(m_prev_t, m_local)
                            safe_now = tl.maximum(m_now, -1.0e30)
                            alpha_t = tl.exp(m_prev_t - safe_now)
                            beta_t = tl.exp(m_local - safe_now)

                        t_sum = tl.zeros((ROW_TILE, 128), dtype=tl.float32)
                        t_hi = tl.zeros((ROW_TILE, 128), dtype=tl.float32)
                        for p in range(3):
                            s_sv = tle.dsa.subview(
                                s_buf, [0, p * 128], [ROW_TILE, 128], [1, 1]
                            )
                            ks_sv = tle.dsa.subview(
                                ks_buf, [0, p * 128], [1, 128], [1, 1]
                            )
                            w_p = tl.exp(tle.dsa.to_tensor(s_sv) - safe_local)
                            if valid < 384:
                                col_pos = p * 128 + offs_128
                                w_p = tl.where(col_pos[None, :] < valid, w_p, 0.0)
                            t_sum += w_p
                            rk_p = w_p * tle.dsa.to_tensor(ks_sv)
                            tle.dsa.to_buffer(rk_p, bind_buffer=s_sv)
                            t_hi = tl.maximum(t_hi, rk_p)

                        local_sum = tl.sum(t_sum, axis=1, keep_dims=True)
                        if group == 0:
                            l_now = local_sum
                        else:
                            l_now = l_prev_t * alpha_t + local_sum * beta_t

                        tile_need_lo = (
                            (not use_shifted)
                            | ks_outlier
                            | ((valid > 4) & (tl.min(l_now) < 2.0))
                        )
                        if tile_need_lo:
                            sub_need_lo = 1

                        max_q = tl.where(use_shifted, 255.0, 127.0)
                        high_t = tl.max(t_hi, axis=1, keep_dims=True)
                        step_t = (high_t / max_q) * beta_t
                        inv_step = max_q / tl.maximum(high_t, 1.0e-30)
                        shift_v = tl.where(use_shifted, 128.0, 0.0)

                        tle.dsa.to_buffer(
                            m_now,
                            bind_buffer=tle.dsa.subview(
                                m_buf, [slot * SUB_M + t_off, 0], [ROW_TILE, 1], [1, 1]
                            ),
                        )
                        tle.dsa.to_buffer(
                            l_now,
                            bind_buffer=tle.dsa.subview(
                                l_buf, [slot * SUB_M + t_off, 0], [ROW_TILE, 1], [1, 1]
                            ),
                        )
                        tle.dsa.to_buffer(
                            alpha_t,
                            bind_buffer=tle.dsa.subview(
                                alpha_buf,
                                [slot * SUB_M + t_off, 0],
                                [ROW_TILE, 1],
                                [1, 1],
                            ),
                        )
                        tle.dsa.to_buffer(
                            step_t,
                            bind_buffer=tle.dsa.subview(
                                step_buf,
                                [slot * SUB_M + t_off, 0],
                                [ROW_TILE, 1],
                                [1, 1],
                            ),
                        )

                        for p in range(3):
                            s_sv = tle.dsa.subview(
                                s_buf, [0, p * 128], [ROW_TILE, 128], [1, 1]
                            )
                            q_f = tle.dsa.to_tensor(s_sv) * inv_step - shift_v
                            q_hi = quantize_to_int8(q_f, ELEMS, USE_CAST_OP).reshape(
                                ROW_TILE, 128
                            )
                            p_off = (t_off + offs_r)[:, None] * 384 + (
                                p * 128 + offs_128
                            )[None, :]
                            tl.store(P + p_hi_base + p_off, q_hi)
                            if tile_need_lo:
                                q_lo_f = (q_f - q_hi.to(tl.float32)) * 127.0
                                q_lo = quantize_to_int8(
                                    q_lo_f, ELEMS, USE_CAST_OP
                                ).reshape(ROW_TILE, 128)
                                tl.store(P + p_lo_base + p_off, q_lo)

                    tle.dsa.to_buffer(
                        tl.full((1, 8), sub_need_lo, tl.int32),
                        bind_buffer=tle.dsa.subview(
                            need_lo_buf, [slot, 0], [1, 8], [1, 1]
                        ),
                    )
                    if use_shifted:
                        tl.store(Flags + slot_idx * 16 + sub_id * 8, sub_need_lo)

                    stage_sync(slot, 1, False)

                    # Accumulate prior window (group - 1, on prev_slot) while Cube computes PV(group).
                    if group > 0:
                        stage_sync(prev_slot, 2, True)
                        prior_idx = (core * 2 + prev_slot).to(tl.int64)
                        pv_hi_base = prior_idx * (2 * M * 512) + sub * 512
                        pv_lo_base = pv_hi_base + M * 512
                        prior_shifted = (M >= 64) & (n > 1024)
                        prior_lo = tl.max(
                            tle.dsa.to_tensor(
                                tle.dsa.subview(
                                    need_lo_buf, [prev_slot, 0], [1, 8], [1, 1]
                                )
                            )
                        )
                        for col in range(2):
                            vsum_c = (
                                tl.load(
                                    Vsum + prior_idx * 512 + col * 256 + offs_256
                                ).to(tl.float32)
                                * 128.0
                                if prior_shifted
                                else tl.zeros((256,), dtype=tl.float32)
                            )
                            for t in range(NUM_TILES):
                                t_off = t * ROW_TILE
                                acc_sv = tle.dsa.subview(
                                    acc_buf,
                                    [t_off, col * 256],
                                    [ROW_TILE, 256],
                                    [1, 1],
                                )
                                pv_off = (t_off + offs_r)[:, None] * 512 + (
                                    col * 256 + offs_256
                                )[None, :]
                                upd = tl.load(PV + pv_hi_base + pv_off).to(tl.float32)
                                if prior_shifted:
                                    upd = upd + vsum_c[None, :]
                                if prior_lo != 0:
                                    upd = upd + tl.load(PV + pv_lo_base + pv_off).to(
                                        tl.float32
                                    ) * (1.0 / 127.0)
                                step_t = tle.dsa.to_tensor(
                                    tle.dsa.subview(
                                        step_buf,
                                        [prev_slot * SUB_M + t_off, 0],
                                        [ROW_TILE, 1],
                                        [1, 1],
                                    )
                                )
                                if group == 1:
                                    acc_new = upd * step_t
                                else:
                                    alpha_t = tle.dsa.to_tensor(
                                        tle.dsa.subview(
                                            alpha_buf,
                                            [prev_slot * SUB_M + t_off, 0],
                                            [ROW_TILE, 1],
                                            [1, 1],
                                        )
                                    )
                                    acc_new = (
                                        tle.dsa.to_tensor(acc_sv) * alpha_t
                                        + upd * step_t
                                    )
                                tle.dsa.to_buffer(acc_new, bind_buffer=acc_sv)

                # Final window accumulation & writeback.
                last = (groups - 1) % 2
                last_idx = (core * 2 + last).to(tl.int64)
                stage_sync(last, 2, True)
                pv_hi_base = last_idx * (2 * M * 512) + sub * 512
                pv_lo_base = pv_hi_base + M * 512
                last_shifted = (M >= 64) & (n > 1024) & ((n - (groups - 1) * 384) > 64)
                last_lo = tl.max(
                    tle.dsa.to_tensor(
                        tle.dsa.subview(need_lo_buf, [last, 0], [1, 8], [1, 1])
                    )
                )
                first_empty = (n == 1) & (S == 2) & CAUSAL & (sub_id == 0)

                for t in range(NUM_TILES):
                    t_off = t * ROW_TILE
                    m_t = tle.dsa.to_tensor(
                        tle.dsa.subview(
                            m_buf, [last * SUB_M + t_off, 0], [ROW_TILE, 1], [1, 1]
                        )
                    )
                    l_t = tle.dsa.to_tensor(
                        tle.dsa.subview(
                            l_buf, [last * SUB_M + t_off, 0], [ROW_TILE, 1], [1, 1]
                        )
                    )
                    alpha_t = tle.dsa.to_tensor(
                        tle.dsa.subview(
                            alpha_buf,
                            [last * SUB_M + t_off, 0],
                            [ROW_TILE, 1],
                            [1, 1],
                        )
                    )
                    step_t = tle.dsa.to_tensor(
                        tle.dsa.subview(
                            step_buf,
                            [last * SUB_M + t_off, 0],
                            [ROW_TILE, 1],
                            [1, 1],
                        )
                    )
                    inv_l = 1.0 / tl.maximum(l_t, 1.0e-30)
                    if first_empty:
                        lse_t = tl.full((ROW_TILE, 1), float("inf"), dtype=tl.float32)
                    else:
                        lse_t = m_t + tl.log(tl.maximum(l_t, 1.0e-30))
                    lse_off = (b.to(tl.int64) * H + h_sub + t_off + offs_r) * S + q_sub
                    tl.store(Lse + lse_off, lse_t.reshape(ROW_TILE))

                    for col in range(2):
                        vsum_c = (
                            tl.load(Vsum + last_idx * 512 + col * 256 + offs_256).to(
                                tl.float32
                            )
                            * 128.0
                            if last_shifted
                            else tl.zeros((256,), dtype=tl.float32)
                        )
                        acc_sv = tle.dsa.subview(
                            acc_buf, [t_off, col * 256], [ROW_TILE, 256], [1, 1]
                        )
                        pv_off = (t_off + offs_r)[:, None] * 512 + (
                            col * 256 + offs_256
                        )[None, :]
                        upd = tl.load(PV + pv_hi_base + pv_off).to(tl.float32)
                        if last_shifted:
                            upd = upd + vsum_c[None, :]
                        if last_lo != 0:
                            upd = upd + tl.load(PV + pv_lo_base + pv_off).to(
                                tl.float32
                            ) * (1.0 / 127.0)
                        if groups == 1:
                            acc_new = upd * step_t
                        else:
                            acc_new = tle.dsa.to_tensor(acc_sv) * alpha_t + upd * step_t
                        if first_empty:
                            out_val = tl.zeros((ROW_TILE, 256), dtype=tl.bfloat16)
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
