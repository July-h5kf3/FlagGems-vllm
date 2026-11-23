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

import functools
import subprocess
from pathlib import Path
from threading import RLock

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from flagtree import _flagprism
from triton import knobs
from triton.backends.ascend.compiler import _get_npucompiler_path
from triton.runtime.driver import driver

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


@triton.jit
def short_decode(
    Q,
    QR,
    K,
    KR,
    QS,
    KS,
    Table,
    Length,
    Out,
    Lse,
    H: tl.constexpr,
    S: tl.constexpr,
    WIDTH,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    task = tl.program_id(0)
    batch = task // ((H // 16) * S)
    query = task // (H // 16) % S
    heads = task % (H // 16) * 16 + tl.arange(0, 16)
    rows = task * 16 + tl.arange(0, 16)
    channels = tl.arange(0, 512)
    lanes = tl.arange(0, 64)
    q = tl.load(Q + rows[:, None] * 512 + channels[None, :])
    qr = tl.load(QR + rows[:, None] * 64 + lanes[None, :])
    qs = tl.load(QS + rows)
    length = tl.load(Length + batch)
    if CAUSAL:
        length = tl.maximum(length - S + query + 1, 0)
    anchor = tl.full((16,), -float("inf"), tl.float32)
    norm = tl.full((16,), 0.0, tl.float32)
    acc = tl.full((16, 512), 0.0, tl.float32)
    # Each remaining coefficient contributes <= 128*tolerance to each channel.
    tolerance = (0.0001 / 128.0) / tl.maximum(length, 1)
    for page in range(tl.cdiv(length, 64)):
        physical = tl.load(Table + batch * WIDTH + page).to(tl.int64)
        token = physical * 64 + lanes
        valid = page * 64 + lanes < length
        k = tl.load(K + token[:, None] * 512 + channels[None, :])
        kr = tl.load(KR + token[:, None] * 64 + lanes[None, :])
        ks = tl.load(KS + token, mask=valid, other=0.0)
        content = tl.dot(q, tl.trans(k), out_dtype=tl.int32)
        rope = tl.dot(qr, tl.trans(kr), out_dtype=tl.float32)
        score = (content.to(tl.float32) + rope) * qs[:, None] * ks[None, :] * SCALE
        score = tl.where(valid[None, :], score, -float("inf"))
        next_anchor = tl.maximum(anchor, tl.max(score, 1))
        correction = tl.exp(anchor - next_anchor)
        weights = tl.exp(score - next_anchor[:, None])
        norm = norm * correction + tl.sum(weights, 1)
        acc = acc * correction[:, None]
        residual = weights * ks[None, :]
        peak = tl.max(tl.abs(residual), 1)
        while tl.max(peak, 0) > tolerance:
            exponent = (peak.to(tl.int32, bitcast=True) >> 23) - 6
            unit = (tl.maximum(exponent, 1) << 23).to(tl.float32, bitcast=True)
            scaled = residual / tl.where(peak > tolerance, unit, 1.0)[:, None]
            digit = tl.where(
                peak[:, None] > tolerance,
                tl.maximum(-127.0, tl.minimum(127.0, tl.floor(scaled + 0.5))),
                0.0,
            ).to(tl.int8)
            product = tl.dot(digit, k, out_dtype=tl.int32)
            acc += product.to(tl.float32) * unit[:, None]
            residual -= digit.to(tl.float32) * unit[:, None]
            peak = tl.max(tl.abs(residual), 1)
        anchor = next_anchor
    denominator = tl.where(norm > 0.0, norm, 1.0)
    tl.store(Out + rows[:, None] * 512 + channels[None, :], acc / denominator[:, None])
    tl.store(
        Lse + (batch * H + heads) * S + query,
        tl.where(norm > 0.0, anchor + tl.log(denominator), float("inf")),
    )


@triton.jit
def portable_page(
    Q,
    QR,
    K,
    KR,
    QS,
    KS,
    Table,
    Length,
    Partial,
    Stats,
    H: tl.constexpr,
    S: tl.constexpr,
    W: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    job = tl.program_id(0)
    tile = job % 4
    page = job // 4 % W
    group = job // (4 * W)
    groups: tl.constexpr = (H + 31) // 32
    b = group // (groups * S)
    query = group // groups % S
    heads = (group % groups) * 32 + tl.arange(0, 32)
    rows = (b * S + query) * H + heads
    lanes = tl.arange(0, 64)
    channels = tl.arange(0, 512)
    output_channels = tile * 128 + tl.arange(0, 128)
    length = tl.load(Length + b)
    if CAUSAL:
        length = tl.maximum(length - S + query + 1, 0)
    maximum = tl.full((32,), -float("inf"), tl.float32)
    denom = tl.full((32,), 0.0, tl.float32)
    acc = tl.full((32, 128), 0.0, tl.float32)
    if page * 64 < length:
        physical = tl.load(Table + b * W + page).to(tl.int64)
        token = physical * 64 + lanes
        valid = page * 64 + lanes < length
        q = tl.load(Q + rows[:, None] * 512 + channels[None, :])
        qr = tl.load(QR + rows[:, None] * 64 + lanes[None, :])
        k = tl.load(K + token[:, None] * 512 + channels[None, :])
        kr = tl.load(KR + token[:, None] * 64 + lanes[None, :])
        qs = tl.load(QS + rows)
        ks = tl.load(KS + token, mask=valid, other=0.0)
        content = tl.dot(q, tl.trans(k), out_dtype=tl.int32)
        rope = tl.dot(qr, tl.trans(kr), out_dtype=tl.float32)
        scores = (content.to(tl.float32) + rope) * qs[:, None] * ks[None, :] * SCALE
        scores = tl.where(valid[None, :], scores, -float("inf"))
        maximum = tl.max(scores, 1)
        probs = tl.exp(scores - maximum[:, None])
        denom = tl.sum(probs, 1)
        residual = probs * ks[None, :]
        peak = tl.max(tl.abs(residual), 1)
        tolerance = (0.0001 / 128.0) / length
        values = tl.load(K + token[:, None] * 512 + output_channels[None, :])
        while tl.max(peak, 0) > tolerance:
            exponent = (peak.to(tl.int32, bitcast=True) >> 23) - 6
            unit = (tl.maximum(exponent, 1) << 23).to(tl.float32, bitcast=True)
            scaled = residual / tl.where(peak > tolerance, unit, 1.0)[:, None]
            digit = tl.where(
                peak[:, None] > tolerance,
                tl.maximum(-127.0, tl.minimum(127.0, tl.floor(scaled + 0.5))),
                0.0,
            ).to(tl.int8)
            product = tl.dot(digit, values, out_dtype=tl.int32)
            acc += product.to(tl.float32) * unit[:, None]
            residual -= digit.to(tl.float32) * unit[:, None]
            peak = tl.max(tl.abs(residual), 1)
    page_rows = rows * W + page
    tl.store(Partial + page_rows[:, None] * 512 + output_channels[None, :], acc)
    if tile == 0:
        tl.store(Stats + page_rows * 2, maximum)
        tl.store(Stats + page_rows * 2 + 1, denom)


@triton.jit
def portable_merge(
    Partial,
    Stats,
    Out,
    Lse,
    H: tl.constexpr,
    S: tl.constexpr,
    W: tl.constexpr,
    PAD: tl.constexpr,
    PAGE_H: tl.constexpr,
):
    row = tl.program_id(0)
    pages = tl.arange(0, PAD)
    channels = tl.arange(0, 512)
    idx = (row // H * PAGE_H + row % H) * W + pages
    maxima = tl.load(Stats + idx * 2, mask=pages < W, other=-float("inf"))
    norms = tl.load(Stats + idx * 2 + 1, mask=pages < W, other=0.0)
    maximum = tl.max(maxima, 0)
    factors = tl.exp(maxima - tl.where(maximum == -float("inf"), 0.0, maximum))
    denom = tl.sum(norms * factors, 0)
    out = tl.load(
        Partial + idx[:, None] * 512 + channels[None, :],
        mask=pages[:, None] < W,
        other=0.0,
    )
    value = tl.sum(out * factors[:, None], 0) / tl.where(denom > 0.0, denom, 1.0)
    tl.store(Out + row * 512 + channels, value)
    b = row // (H * S)
    query = row // H % S
    head = row % H
    tl.store(
        Lse + (b * H + head) * S + query,
        tl.where(denom > 0.0, maximum + tl.log(denom), float("inf")),
    )


@functools.lru_cache(maxsize=1)
def requires_portable_short():
    compiler, env = _get_npucompiler_path()
    version = subprocess.run(
        [compiler, "--version"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return version.returncode == 0 and version.stdout.startswith(
        b"bishengir-compile 1.2."
    )


# These options and the validated tensor dtypes are fixed for both entry points.
LAUNCH_OPTIONS = dict(
    enable_auto_bind_sub_block=False,
    disable_auto_inject_block_sync=True,
    multibuffer=False,
)
compiled_runners = {}
compile_lock = RLock()
MISSING = object()


def launch(fn, cores, tensors, scalars):
    options = {} if fn is short_decode else LAUNCH_OPTIONS
    grid = (cores,)
    device = tensors[0].device.index
    # Components can inject options or observe a run. Keep their normal JIT
    # behavior, including any changes between successive calls.
    if (
        fn.pre_run_hooks
        or fn.debug
        or knobs.runtime.debug
        or _flagprism._registered_callbacks("apply_compile_options")
    ):
        return fn[grid](*tensors, *scalars, **options)

    # Ascend specializes pointers on divisibility by 16. Include every actual
    # scalar (both runtime and constexpr), even when the JIT uses a coarser key.
    alignment = tuple(t.data_ptr() % 16 == 0 for t in tensors)
    key = (fn, device, cores, alignment, scalars, tuple(options.items()))
    runner = compiled_runners.get(key)
    if runner is None:
        with compile_lock:
            runner = compiled_runners.get(key)
            if runner is None:
                compiled = fn[grid](*tensors, *scalars, **options)
                if compiled is not None:
                    compiled_runners[key] = compiled[(cores, 1, 1)]
                return
    for (name, _), (value, namespace) in fn.used_global_vals.items():
        if namespace.get(name, MISSING) != value:
            raise RuntimeError(
                f"Global variable {name} changed since kernel compilation"
            )
    # The public entry point selected this device. Read the actual caller stream.
    runner(*tensors, *scalars, stream=driver.active.get_current_stream(device))


def allocate_scratch(device, words, pv_words):
    # Dtype views of disjoint ranges on one fresh allocation. Pass actual
    # interior tensor pointers to the native launcher for each typed region.
    backing = torch.empty(
        2 * words + words // 4 + pv_words, dtype=torch.int32, device=device
    )
    c, r, p, pv = backing.split((words, words, words // 4, pv_words))
    return c, r.view(torch.float32), p.view(torch.int8), pv
