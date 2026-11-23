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

from pathlib import Path

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.runtime.backend._ascend.fused.attention import PIPE, stage_sync

NATIVE_DIR = Path(__file__).with_name("mla_w8a8_native")


class QueryKey:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_FIX
    mode = al.MODE.SIMD

    def __init__(self, q, qr, k, kr, table, c, r, remaining, first, slot, head_stride):
        self.arg_type.update(
            q=tl.uint64,
            qr=tl.uint64,
            k=tl.uint64,
            kr=tl.uint64,
            table=tl.uint64,
            c=tl.uint64,
            r=tl.uint64,
            remaining=tl.uint32,
            first=tl.uint32,
            slot=tl.uint32,
            head_stride=tl.uint32,
        )


class ProbabilityValue:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_FIX
    mode = al.MODE.SIMD

    def __init__(self, p, dst, k, kr, table, slot, remaining, last):
        self.arg_type.update(
            p=tl.uint64,
            dst=tl.uint64,
            k=tl.uint64,
            kr=tl.uint64,
            table=tl.uint64,
            slot=tl.uint32,
            remaining=tl.uint32,
            last=tl.uint32,
        )


class ProbabilityTile:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(
        self,
        c,
        r,
        prob,
        qs,
        ks,
        table,
        pv,
        visible,
        first,
        slot,
        atten_scale,
        head_stride,
        causal_pair,
    ):
        self.arg_type.update(
            c=tl.uint64,
            r=tl.uint64,
            prob=tl.uint64,
            qs=tl.uint64,
            ks=tl.uint64,
            table=tl.uint64,
            pv=tl.uint64,
            visible=tl.uint32,
            first=tl.uint32,
            slot=tl.uint32,
            atten_scale=tl.float32,
            head_stride=tl.uint32,
            causal_pair=tl.uint32,
        )


class OutputUpdate:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(
        self, pv, out, lse, slot, first, last, empty, head_stride, first_empty
    ):
        self.arg_type.update(
            pv=tl.uint64,
            out=tl.uint64,
            lse=tl.uint64,
            slot=tl.uint32,
            first=tl.uint32,
            last=tl.uint32,
            empty=tl.uint32,
            head_stride=tl.uint32,
            first_empty=tl.uint32,
        )


# FlagTree reads the Python signature to marshal each stage. The three row
# geometries share their stage ABI and link different compiled symbols.
for prefix, native_stem in (("m16_", "m16"), ("m32_", "m32"), ("", "window")):
    for stage, descriptor, engine in (
        ("fenced_window_qk", QueryKey, "cube"),
        ("fenced_window_pv", ProbabilityValue, "cube"),
        ("vecfences_first", ProbabilityTile, "vec"),
        ("vecfences_second", ProbabilityTile, "vec"),
        ("vecfences_update", OutputUpdate, "vec"),
    ):
        symbol = prefix + stage
        al.register_custom_op(
            type(
                symbol,
                (descriptor,),
                {
                    "symbol": symbol,
                    "bitcode": str(NATIVE_DIR / f"{native_stem}_{engine}.bc"),
                },
            )
        )


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
    QK: tl.constexpr,
):
    byte = ((core * 2 + slot) * M * 384 * 4).to(tl.uint64)
    al.custom(
        QK,
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


@triton.jit
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
    BATCH: tl.constexpr,
    STRIDE: tl.constexpr,
):
    Plan = PV + CORES * 2 * M * 512
    core = tl.program_id(0)
    QK: tl.constexpr = (
        "m16_fenced_window_qk"
        if M == 16
        else ("m32_fenced_window_qk" if M == 32 else "fenced_window_qk")
    )
    PV_OP: tl.constexpr = (
        "m16_fenced_window_pv"
        if M == 16
        else ("m32_fenced_window_pv" if M == 32 else "fenced_window_pv")
    )
    FIRST: tl.constexpr = (
        "m16_vecfences_first"
        if M == 16
        else ("m32_vecfences_first" if M == 32 else "vecfences_first")
    )
    SECOND: tl.constexpr = (
        "m16_vecfences_second"
        if M == 16
        else ("m32_vecfences_second" if M == 32 else "vecfences_second")
    )
    UPDATE: tl.constexpr = (
        "m16_vecfences_update"
        if M == 16
        else ("m32_vecfences_update" if M == 32 else "vecfences_update")
    )
    band: tl.constexpr = M // S
    stride: tl.constexpr = (8 if M == 16 else M // 4) if S == 1 else H
    with al.scope(core_mode="cube"):
        if STRIDE:
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
                    Q, QR, K, KR, table, C, R, row, core, 0, 0, pages, stride, M, QK
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
                            QK,
                        )
                        stage_sync(1 - slot, 0, False)
                    stage_sync(slot, 1, True)
                    buf = (core * 2 + slot).to(tl.uint64) * M
                    al.custom(
                        PV_OP,
                        P.to(tl.uint64) + buf * 384,
                        PV.to(tl.uint64) + buf * 2048,
                        K.to(tl.uint64),
                        KR.to(tl.uint64),
                        table + (group + 2) * 24,
                        slot,
                        tl.maximum(0, pages - (group + 2) * 6),
                        tl.cast(group + 1 == groups, tl.uint32),
                    )
                    stage_sync(slot, 2, False)
            al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
    with al.scope(core_mode="vector"):
        if STRIDE:
            if al.sub_vec_id() == 0:
                core = tl.program_id(0)
                requests = tl.arange(0, 128)
                lengths = tl.load(Lens + requests, mask=requests < BATCH, other=0)
                wave = tl.arange(0, STRIDE)
                start = wave * CORES
                tasks = start + core
                if tl.min(tl.where(requests < BATCH, lengths, 2147483647), 0) != tl.max(
                    lengths, 0
                ):
                    # One integer key carries the page-group cost and its request id.
                    cost = lengths // (384) + (lengths % (384) != 0).to(tl.int32)
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
                    b = tl.gather(ranked.to(tl.float32), safe_slot, 0).to(tl.int32)
                    tasks = b
                tl.store(Plan + core * STRIDE + wave, tasks)
            al.sync_block_set("vector", "cube", 7, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
            al.sync_block_wait("cube", "vector", 7, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
        sub = al.sub_vec_id().to(tl.uint64) * (M // 2)
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
                    PV.to(tl.uint64) + ((core * 2).to(tl.uint64) * M + sub) * 2048
                )
            for group in tl.range(0, groups, disable_licm=True):
                slot = group % 2
                buf = (core * 2 + slot).to(tl.uint64) * M + sub
                stage_sync(slot, 0, True)
                prior = (core * 2 + 1 - slot).to(tl.uint64) * M + sub
                prior_pv = PV.to(tl.uint64) + prior * 2048
                al.custom(
                    FIRST,
                    C.to(tl.uint64) + buf * 1536,
                    R.to(tl.uint64) + buf * 1536,
                    P.to(tl.uint64) + buf * 384,
                    QS.to(tl.uint64) + row * 4,
                    KS.to(tl.uint64),
                    Table.to(tl.uint64) + (b.to(tl.uint64) * WIDTH + group * 6) * 4,
                    prior_pv,
                    n - group * 384,
                    tl.cast(group == 0, tl.uint32),
                    slot,
                    SCALE,
                    stride,
                    1 if S == 2 and CAUSAL else 0,
                )
                if group > 0:
                    stage_sync(1 - slot, 2, True)
                al.custom(
                    SECOND,
                    C.to(tl.uint64) + buf * 1536,
                    R.to(tl.uint64) + buf * 1536,
                    P.to(tl.uint64) + buf * 384,
                    QS.to(tl.uint64) + row * 4,
                    KS.to(tl.uint64),
                    Table.to(tl.uint64) + (b.to(tl.uint64) * WIDTH + group * 6) * 4,
                    prior_pv,
                    n - group * 384,
                    tl.cast(group == 0, tl.uint32),
                    slot,
                    SCALE,
                    stride,
                    1 if S == 2 and CAUSAL else 0,
                )
                stage_sync(slot, 1, False)
                if group > 0:
                    al.custom(
                        UPDATE,
                        PV.to(tl.uint64) + prior * 2048,
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
                buf = (core * 2 + last).to(tl.uint64) * M + sub
                stage_sync(last, 2, True)
                al.custom(
                    UPDATE,
                    PV.to(tl.uint64) + buf * 2048,
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
                al.custom(
                    UPDATE, PV.to(tl.uint64), dst, lse_stage, 0, 1, 1, 1, stride, 0
                )
            if S == 2:
                source = lse_stage.to(tl.pointer_type(tl.float32))
                block = tl.load(source + tl.arange(0, M // 2)).reshape(2, M // 4)
                tl.store(
                    lse.to(tl.pointer_type(tl.float32)) + tl.arange(0, M // 2),
                    tl.trans(block).reshape(M // 2),
                )
            al.sync_block_set("vector", "cube", 6, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
