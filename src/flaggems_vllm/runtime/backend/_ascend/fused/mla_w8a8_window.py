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

    HAS_PAGED_INT8_OPS = (
        "paged_int8_qk" in ascend_custom_ops.__all__
        and "paged_int8_pv" in ascend_custom_ops.__all__
        and "paged_int8_prob" in ascend_custom_ops.__all__
        and "paged_int8_update" in ascend_custom_ops.__all__
    )
else:
    HAS_PAGED_INT8_OPS = False


@triton.jit
def window_scores(
    Q,
    QR,
    K,
    KR,
    Table,
    C,
    R,
    row,
    core,
    group,
    slot,
    pages,
    STRIDE: tl.constexpr,
    M: tl.constexpr,
):
    byte = ((core * 2 + slot) * M * 384 * 4).to(tl.uint64)
    tle.dsa.ascend.raw(
        "paged_int8_qk",
        M,
        Q.to(tl.uint64) + row * 512,
        QR.to(tl.uint64) + row * 128,
        K.to(tl.uint64),
        KR.to(tl.uint64),
        Table + group * 24,
        C.to(tl.uint64) + byte,
        R.to(tl.uint64) + byte,
        pages - group * 6,
        tl.cast(group == 0, tl.uint32),
        slot,
        STRIDE,
    )


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
    # Per-core scratch layout after PV ping-pong buffers:
    # - Flags: CORES * 32 int32s (16 int32s per (core, slot))
    # - Ones + Vsum: CORES * 16896 int32s (512 int32s ones + 2 * 16 * 512 vsum)
    # - Plan: CORES * STRIDE int32s (LPT schedule table)
    core = tl.program_id(0)
    flag_base = PV.to(tl.uint64) + CORES * (4 * M * 2048)
    ones_base = flag_base + CORES * 128
    ones_gm = ones_base + core.to(tl.uint64) * (16896 * 4)
    Plan = PV + CORES * (4 * M * 512 + 32 + 16896)
    GROUPS_PER_REQ: tl.constexpr = (H * S) // M
    band: tl.constexpr = M // S
    stride: tl.constexpr = (8 if M == 16 else M // 4) if S == 1 else H
    with al.scope(core_mode="cube"):
        al.sync_block_wait("vector", "cube", 7, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
        al.sync_block_set("cube", "vector", 7, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
        for index in tl.range(core, TASKS, CORES, disable_licm=True):
            task = index
            if STRIDE:
                task = tl.load(Plan + core * STRIDE + index // CORES)
            b = task // (H // band)
            head = task % (H // band) * band
            row = b.to(tl.uint64) * S * H + head
            n = tl.load(Lens + b)
            pages = tl.cdiv(n, 64)
            groups = tl.cdiv(n, 384)
            table = Table.to(tl.uint64) + b.to(tl.uint64) * WIDTH * 4
            if groups > 0:
                window_scores(
                    Q, QR, K, KR, table, C, R, row, core, 0, 0, pages, stride, M
                )
                stage_sync(0, 0, False)
                for group in tl.range(0, groups, disable_licm=True):
                    slot = group % 2
                    if group + 1 < groups:
                        window_scores(
                            Q,
                            QR,
                            K,
                            KR,
                            table,
                            C,
                            R,
                            row,
                            core,
                            group + 1,
                            1 - slot,
                            pages,
                            stride,
                            M,
                        )
                        stage_sync(1 - slot, 0, False)
                    stage_sync(slot, 1, True)
                    slot_idx = (core * 2 + slot).to(tl.uint64)
                    # Long-context windows use shifted 8-bit P_hi with vsum
                    # compensation and adaptive 2-digit fallback; short
                    # sequences and short tails use exact unshifted 2-digit.
                    use_shifted = (M >= 64) & (n > 1024) & ((n - group * 384) > 64)
                    flag_gm = tl.where(
                        use_shifted, flag_base + slot_idx * 64, tl.cast(0, tl.uint64)
                    )
                    ones_arg = tl.where(use_shifted, ones_gm, tl.cast(0, tl.uint64))
                    tle.dsa.ascend.raw(
                        "paged_int8_pv",
                        M,
                        P.to(tl.uint64) + slot_idx * (2 * M * 384),
                        PV.to(tl.uint64) + slot_idx * (2 * M * 2048),
                        K.to(tl.uint64),
                        KR.to(tl.uint64),
                        table + (group + 2) * 24,
                        flag_gm,
                        ones_arg,
                        slot,
                        tl.maximum(0, pages - (group + 2) * 6),
                        tl.cast(group + 1 == groups, tl.uint32),
                    )
                    stage_sync(slot, 2, False)
            al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
    with al.scope(core_mode="vector"):
        if al.sub_vec_id() == 0:
            if M >= 64:
                ones_ptr = ones_gm.to(tl.pointer_type(tl.int32))
                tl.store(
                    ones_ptr + tl.arange(0, 512),
                    tl.full([512], 0x01010101, tl.int32),
                )
            if STRIDE:
                requests = tl.arange(0, 128)
                lengths = tl.load(Lens + requests, mask=requests < BATCH, other=0)
                wave = tl.arange(0, STRIDE)
                start = wave * CORES
                tasks = start + core
                if tl.min(tl.where(requests < BATCH, lengths, 2147483647), 0) != tl.max(
                    lengths, 0
                ):
                    # One integer key carries the page-group cost and its request id.
                    cost = lengths // 384 + (lengths % 384 != 0).to(tl.int32)
                    # Float sort represents the request id exactly up to this cost cap.
                    keys = tl.where(
                        requests < BATCH,
                        tl.minimum(cost, 131071) * 128 + 127 - requests,
                        -1,
                    )
                    ranked = 127 - (
                        al.sort(keys.to(tl.float32), descending=True).to(tl.int32) & 127
                    )
                    width = tl.minimum(CORES, TASKS - start)
                    slot = tl.where((wave & 1) == 0, tasks, start + width - 1 - core)
                    safe_slot = tl.minimum(tl.maximum(slot, 0), TASKS - 1)
                    req_rank = safe_slot // GROUPS_PER_REQ
                    b = tl.gather(ranked.to(tl.float32), req_rank, 0).to(tl.int32)
                    tasks = b * GROUPS_PER_REQ + (safe_slot % GROUPS_PER_REQ)
                tl.store(Plan + core * STRIDE + wave, tasks)
        al.sync_block_set("vector", "cube", 7, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
        al.sync_block_wait("cube", "vector", 7, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
        sub_id = al.sub_vec_id().to(tl.uint64)
        sub = sub_id * (M // 2)
        for index in tl.range(core, TASKS, CORES, disable_licm=True):
            task = index
            if STRIDE:
                task = tl.load(Plan + core * STRIDE + index // CORES)
            b = task // (H // band)
            head = task % (H // band) * band + sub // S
            row = b.to(tl.uint64) * S * H + head
            n = tl.load(Lens + b)
            groups = tl.cdiv(n, 384)
            dst = Out.to(tl.uint64) + row * 1024
            lse = Lse.to(tl.uint64) + (b.to(tl.uint64) * H + head) * S * 4
            lse_stage = lse
            if S == 2:
                lse_stage = (
                    PV.to(tl.uint64) + ((core * 2).to(tl.uint64) * (2 * M) + sub) * 2048
                )
            for group in tl.range(0, groups, disable_licm=True):
                slot = group % 2
                slot_idx = (core * 2 + slot).to(tl.uint64)
                buf = slot_idx * M + sub
                p_sub = slot_idx * (2 * M * 384) + sub * 384
                use_shifted = (M >= 64) & (n > 1024) & ((n - group * 384) > 64)
                flag_gm = tl.where(
                    use_shifted,
                    flag_base + slot_idx * 64 + sub_id * 32,
                    tl.cast(0, tl.uint64),
                )
                stage_sync(slot, 0, True)
                prior_idx = (core * 2 + 1 - slot).to(tl.uint64)
                pv_row_bytes: tl.constexpr = 1024 if M >= 64 else 2048
                prior_pv = (
                    PV.to(tl.uint64) + prior_idx * (2 * M * 2048) + sub * pv_row_bytes
                )
                prior_vsum = tl.where(
                    (M >= 64) & (n > 1024),
                    ones_gm + (1 + (1 - slot) * 16) * 2048,
                    tl.cast(0, tl.uint64),
                )
                for phase in tl.static_range(2):
                    if phase == 1 and group > 0:
                        stage_sync(1 - slot, 2, True)
                    tle.dsa.ascend.raw(
                        "paged_int8_prob",
                        M,
                        phase,
                        C.to(tl.uint64) + buf * 1536,
                        R.to(tl.uint64) + buf * 1536,
                        P.to(tl.uint64) + p_sub,
                        QS.to(tl.uint64) + row * 4,
                        KS.to(tl.uint64),
                        Table.to(tl.uint64) + (b.to(tl.uint64) * WIDTH + group * 6) * 4,
                        prior_pv,
                        flag_gm,
                        n - group * 384,
                        tl.cast(group == 0, tl.uint32),
                        slot,
                        SCALE,
                        stride,
                        1 if S == 2 and CAUSAL else 0,
                    )
                stage_sync(slot, 1, False)
                if group > 0:
                    tle.dsa.ascend.raw(
                        "paged_int8_update",
                        M,
                        prior_pv,
                        prior_vsum,
                        dst,
                        lse,
                        1 - slot,
                        tl.cast(group == 1, tl.uint32),
                        0,
                        0,
                        stride,
                        0,
                    )
            if groups > 0:
                last = (groups - 1) % 2
                last_idx = (core * 2 + last).to(tl.uint64)
                last_shifted = (M >= 64) & (n > 1024) & ((n - (groups - 1) * 384) > 64)
                last_vsum = tl.where(
                    last_shifted,
                    ones_gm + (1 + last * 16) * 2048,
                    tl.cast(0, tl.uint64),
                )
                stage_sync(last, 2, True)
                tle.dsa.ascend.raw(
                    "paged_int8_update",
                    M,
                    PV.to(tl.uint64)
                    + last_idx * (2 * M * 2048)
                    + sub * (1024 if M >= 64 else 2048),
                    last_vsum,
                    dst,
                    lse_stage,
                    last,
                    tl.cast(groups == 1, tl.uint32),
                    1,
                    0,
                    stride,
                    tl.cast((n == 1) & (S == 2) & CAUSAL, tl.uint32),
                )
            else:
                tle.dsa.ascend.raw(
                    "paged_int8_update",
                    M,
                    PV.to(tl.uint64),
                    0,
                    dst,
                    lse_stage,
                    0,
                    1,
                    1,
                    1,
                    stride,
                    0,
                )
            if S == 2:
                source = lse_stage.to(tl.pointer_type(tl.float32))
                block = tl.load(source + tl.arange(0, M // 2)).reshape(2, M // 4)
                tl.store(
                    lse.to(tl.pointer_type(tl.float32)) + tl.arange(0, M // 2),
                    tl.trans(block).reshape(M // 2),
                )
            al.sync_block_set("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
