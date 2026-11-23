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
from threading import RLock

import torch
import triton
import triton.language as tl
from flagtree import _flagprism
from triton import knobs
from triton.backends.ascend.compiler import _get_npucompiler_path
from triton.runtime.driver import driver


@triton.jit
def scaled_page_scores(q, qr, k, kr, qs, ks, valid, SCALE: tl.constexpr):
    content = tl.dot(q, tl.trans(k), out_dtype=tl.int32)
    rope = tl.dot(qr, tl.trans(kr), out_dtype=tl.float32)
    scores = (content.to(tl.float32) + rope) * qs[:, None] * ks[None, :] * SCALE
    return tl.where(valid[None, :], scores, -float("inf"))


@triton.jit
def paged_decode(
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
    SPLITS: tl.constexpr,
):
    task = tl.program_id(0)
    if SPLITS > 1:
        split = task % SPLITS
        task = task // SPLITS
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
    pages = tl.cdiv(length, 64)
    first, end = 0, pages
    if SPLITS > 1:
        first, end = pages * split // SPLITS, pages * (split + 1) // SPLITS
    for page in range(first, end):
        physical = tl.load(Table + batch * WIDTH + page).to(tl.int64)
        token = physical * 64 + lanes
        valid = page * 64 + lanes < length
        k = tl.load(K + token[:, None] * 512 + channels[None, :])
        kr = tl.load(KR + token[:, None] * 64 + lanes[None, :])
        ks = tl.load(KS + token, mask=valid, other=0.0)
        score = scaled_page_scores(q, qr, k, kr, qs, ks, valid, SCALE)
        next_anchor = tl.maximum(anchor, tl.max(score, 1))
        correction = tl.exp(anchor - next_anchor)
        weights = tl.exp(score - next_anchor[:, None])
        norm = norm * correction + tl.sum(weights, 1)
        acc = acc * correction[:, None]
        residual = weights * ks[None, :]
        peak = tl.max(tl.abs(residual), 1)
        if SPLITS:
            # The split-KV INT8 path represents P * token_scale with a high
            # digit and a residual digit, then accumulates both INT32 PV dots.
            step = peak / 127.0
            scaled = residual / tl.where(step > 0, step, 1.0)[:, None]
            high = tl.maximum(-127.0, tl.minimum(127.0, tl.floor(scaled + 0.5))).to(
                tl.int8
            )
            low = tl.floor((scaled - high.to(tl.float32)) * 127.0 + 0.5).to(tl.int8)
            pv_hi = tl.dot(high, k, out_dtype=tl.int32).to(tl.float32)
            pv_lo = tl.dot(low, k, out_dtype=tl.int32).to(tl.float32)
            acc += (pv_hi + pv_lo * (1.0 / 127.0)) * step[:, None]
        else:
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
    if SPLITS > 1:
        slot = rows.to(tl.int64) * SPLITS + split
        tl.store(Out + slot[:, None] * 512 + channels[None, :], acc)
        tl.store(Lse + slot * 2, anchor)
        tl.store(Lse + slot * 2 + 1, norm)
    else:
        denominator = tl.where(norm > 0.0, norm, 1.0)
        tl.store(
            Out + rows[:, None] * 512 + channels[None, :], acc / denominator[:, None]
        )
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
        scores = scaled_page_scores(q, qr, k, kr, qs, ks, valid, SCALE)
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
    idx = (row // H * PAGE_H + row % H).to(tl.int64) * W + pages
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
    options = {} if fn is paged_decode else LAUNCH_OPTIONS
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
