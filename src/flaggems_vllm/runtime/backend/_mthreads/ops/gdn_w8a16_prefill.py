# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Chunked GDN prefill for normalized 128-wide MP31 activations.

Each 32-token output block uses the preceding 32 tokens as history. A bounded
input, contractive forget gates, and a small triangular residual are required;
otherwise the entire sequence/head is recomputed in FP32. Only the final state
is quantized. See the state and residual bounds by the respective checks.
"""
import torch
import triton
import triton.language as tl
from triton.experimental.tle import language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

from flaggems_vllm.runtime.backend._mthreads.ops.gdn_w8a16_packed import (
    GDN_VECTOR_LIBRARY,
    load_fp8_values4,
    store_fp8_values4,
)


@triton.jit
def _chunk_map(
    Cu, Map, T: tl.constexpr, NS: tl.constexpr, BLOCK: tl.constexpr, COUNT: tl.constexpr
):
    c = tl.program_id(0) * 32 + tl.arange(0, 32)
    si = tl.arange(0, triton.next_power_of_2(NS))
    starts = tl.load(Cu + si, si < NS, 0).to(tl.int32)
    n = tl.max(
        tl.where(
            (si < NS)[None, :] & ((starts // BLOCK + si)[None, :] <= c[:, None]),
            si[None, :],
            0,
        ),
        1,
    )
    bos = tl.load(Cu + n).to(tl.int32)
    end = tl.load(Cu + n + 1).to(tl.int32)
    beg = bos + (c - (bos // BLOCK + n)) * BLOCK
    tl.store(Map + c * 4, n, c < COUNT)
    tl.store(Map + c * 4 + 1, bos, c < COUNT)
    tl.store(Map + c * 4 + 2, end, c < COUNT)
    tl.store(Map + c * 4 + 3, beg, c < COUNT)


@triton.jit
def _window_output(
    Q,
    K,
    V,
    G,
    B,
    S,
    Sc,
    Cu,
    Index,
    O,
    Map,
    Dirty,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    NS: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK: tl.constexpr,
    SL0: tl.constexpr,
    SL1: tl.constexpr,
    SL2: tl.constexpr,
    SL3: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    SCALE: tl.constexpr,
    VAR: tl.constexpr,
    INDEXED: tl.constexpr,
    WITH_INITIAL: tl.constexpr,
    QDesc,
    Bounds,
    RAW: tl.constexpr,
):
    if RAW and not WITH_INITIAL:
        ready = tle.gpu.alloc_barrier(expect_bytes=BLOCK * 128 * 2)
    (c, h, tv) = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    if WITH_INITIAL:
        n = c
        if VAR:
            bos = tl.load(Cu + n).to(tl.int32)
            end = tl.load(Cu + n + 1).to(tl.int32)
        else:
            bos = n * T
            end = bos + T
        beg = bos
    elif VAR and NS != 1:
        n = tl.load(Map + c * 4)
        bos = tl.load(Map + c * 4 + 1)
        end = tl.load(Map + c * 4 + 2)
        beg = tl.load(Map + c * 4 + 3)
    elif VAR:
        bos = tl.load(Cu).to(tl.int32)
        end = tl.load(Cu + 1).to(tl.int32)
        n = 0
        beg = bos + c * BLOCK
    else:
        n = c // tl.cdiv(T, BLOCK)
        bos = n * T
        end = bos + T
        beg = bos + c % tl.cdiv(T, BLOCK) * BLOCK
    if beg >= end:
        return
    if not WITH_INITIAL and beg < bos + 64:
        return
    slot = tl.load(Index + n).to(tl.int64) if INDEXED else n.to(tl.int64)
    seqstart = beg - (BT - BLOCK)
    if seqstart < bos:
        seqstart = bos
    i = tl.arange(0, BT)
    j = tl.arange(0, BT if WITH_INITIAL else BLOCK)
    k = tl.arange(0, 128)
    v = tv * BV + tl.arange(0, BV)
    t = seqstart + i
    to = beg + j
    qh = h // (HV // H)
    if slot < 0:
        tl.store(O + (to[:, None] * HV + h) * 128 + v[None, :], 0.0, to[:, None] < end)
        return
    kk = tl.load(K + (t[:, None] * H + qh) * 128 + k[None, :], t[:, None] < end, 0.0)
    if not RAW or WITH_INITIAL:
        qq = tl.load(
            Q + (to[:, None] * H + qh) * 128 + k[None, :], to[:, None] < end, 0.0
        )
        vv = tl.load(
            V + (t[:, None] * HV + h) * 128 + v[None, :], t[:, None] < end, 0.0
        )
    else:
        value_max = tl.load(Bounds + t * HV + h, t < end, 0.0)
    gv = tl.load(G + t * HV + h, t < end, 0.0).to(tl.float32)
    bv = tl.load(B + t * HV + h, t < end, 0.0).to(tl.float32)
    pref = tl.cumsum(gv, 0)
    # Center exponents to form exp(prefix[i]-prefix[j]) as an outer
    # product without exponentiating a whole matrix.
    bottom = tl.min(pref, 0)
    center = 0.5 * bottom
    left = tl.exp(tl.minimum(80.0, pref - center))
    right = tl.exp(tl.minimum(80.0, center - pref))
    tri = i[:, None] > i[None, :]
    if RAW and not WITH_INITIAL:
        # Reuse operand A after its K products complete. The descriptor copy
        # preserves the SQMMA swizzle; raw shared-memory pointers do not.
        key_slots = tle.gpu.alloc(
            (2, BLOCK, 128),
            K.dtype.element_ty,
            init_value=tl.reshape(kk, (2, BLOCK, 128)),
            nv_mma_shared_layout=True,
        )
        key_a = key_slots.slot(0)
        key_b = tle.gpu.alloc(
            (BT, 128), K.dtype.element_ty, init_value=kk, nv_mma_shared_layout=True
        )
        gram_first = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(key_a, key_b, trans_b=True))
        gram_last = tle.gpu.wgmma_wait(
            0, tle.gpu.wgmma(key_slots.slot(1), key_b, trans_b=True)
        )
        tle.gpu.copy(QDesc, key_a, (BLOCK, 128), (beg, qh * 128), barrier=ready)
        tle.gpu.barrier_wait(ready, phaseIdx=0)
        qgram = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(key_a, key_b, trans_b=True))
        gram = tl.reshape(
            tl.permute(tl.join(gram_first, gram_last), (2, 0, 1)), (BT, BT)
        )
    else:
        kk0 = tle.gpu.alloc(
            (BT, 128), K.dtype.element_ty, init_value=kk, nv_mma_shared_layout=True
        )
        kk1 = tle.gpu.alloc(
            (BT, 128), K.dtype.element_ty, init_value=kk, nv_mma_shared_layout=True
        )
        qq0 = tle.gpu.alloc(
            (BT if WITH_INITIAL else BLOCK, 128),
            Q.dtype.element_ty,
            init_value=qq,
            nv_mma_shared_layout=True,
        )
        a = tle.gpu.wgmma(kk0, kk1, trans_b=True)
        gram = tle.gpu.wgmma_wait(0, a)
        cmat = tle.gpu.wgmma(qq0, kk1, trans_b=True)
        qgram = tle.gpu.wgmma_wait(0, cmat)
    L = tl.where(tri, -gram * (left[:, None] * right[None, :]) * bv[:, None], 0.0).to(
        tl.float16
    )
    causal = to[:, None] >= t[None, :]
    po = tl.gather(pref, tl.minimum(j + (beg - seqstart), BT - 1), 0)
    qk = tl.where(
        causal,
        qgram
        * (
            tl.gather(left, tl.minimum(j + (beg - seqstart), BT - 1), 0)[:, None]
            * right[None, :]
        ),
        0.0,
    ).to(tl.float16)
    if RAW:
        ggood = 1
    else:
        ggood = tl.min(
            ((gv >= -80.0) & (gv <= 0.0) & (bv >= 0.0) & (bv <= 1.0)).to(tl.int32), 0
        )
    if RAW and not WITH_INITIAL:
        value_ok = tl.max(value_max, 0) <= 1.0
    else:
        value_ok = tl.min((tl.abs(vv.to(tl.float32)) <= 1.0).to(tl.int32))
    # With normalized keys and beta in [0,1], the state transition is
    # non-expansive. Each accepted full block attenuates prior state by
    # exp(-14) or less, before adding at most 32 bounded updates.
    block_log = tl.sum(tl.where((t >= beg) & (t < beg + BLOCK), gv, 0.0), 0)
    bad = (
        (ggood == 0)
        | (value_ok == 0)
        | (beg + BLOCK <= end) & (block_log > -14.0)
        | (tl.abs(SCALE) > 0.1)
    )
    # Use the measured row norm in the Neumann remainder bound;
    # the accepted output residual budget is 1.2e-4.
    lnorm = tl.max(tl.sum(tl.abs(L).to(tl.float32), 1), 0)
    bad = bad | ~(lnorm <= 0.5) | ~(bottom >= -140.0)
    if WITH_INITIAL:
        bad = bad | (bos + 64 <= end) & (tl.sum(tl.where(i >= 32, gv, 0.0), 0) > -14.0)
    if not RAW or WITH_INITIAL:
        rhs = vv.to(tl.float32)
    base = tl.full((BT if WITH_INITIAL else BLOCK, BV), 0.0, tl.float32)
    if WITH_INITIAL:
        packed: tl.constexpr = (
            SL3 == 1 and SL0 % 4 == 0 and SL1 % 4 == 0 and SL2 % 4 == 0
        )
        if packed:
            s = (
                load_fp8_values4(
                    S, slot, h, k, tv * (BV // 4) + tl.arange(0, BV // 4), SL0, SL1, SL2
                )
                * tl.load(Sc + slot * C0 + h * C1 + v * C2)[None, :]
            )
        else:
            s = (
                tl.load(
                    S + slot * SL0 + h * SL1 + k[:, None] * SL2 + v[None, :] * SL3
                ).to(tl.float32)
                * tl.load(Sc + slot * C0 + h * C1 + v * C2)[None, :]
            )
        bad = bad | ~(tl.max(tl.abs(s)) <= 4.0)
        sf = s.to(tl.float16)
        rhs -= tl.dot(kk.to(tl.float16), sf) * tl.exp(pref)[:, None]
        base = tl.dot(qq.to(tl.float16), sf) * tl.exp(po)[:, None]
    if not RAW or WITH_INITIAL:
        val = (rhs * bv[:, None]).to(tl.float16)
    q0 = qk.to(tl.float16)
    e1 = tl.dot(q0, L)
    row_norm = tl.sum(tl.abs(L).to(tl.float32), 1)
    if RAW and not WITH_INITIAL:
        magnitude = tl.max(value_max * tl.abs(bv), 0) * tl.abs(SCALE)
    else:
        magnitude = tl.max(tl.abs(val).to(tl.float32)) * tl.abs(SCALE)
    residual_budget = 1.2e-4 * (1.0 - lnorm)
    if (
        tl.max(tl.sum(tl.abs(e1) * row_norm[None, :], 1), 0) * magnitude
        <= residual_budget
    ):
        qk = (q0 + e1).to(tl.float16)
    else:
        e2 = tl.dot(e1.to(tl.float16), L)
        residual = tl.max(tl.sum(tl.abs(e2) * row_norm[None, :], 1), 0)
        bad = bad | ~(residual * magnitude <= residual_budget)
        qk = (q0 + e1 + e2).to(tl.float16)
    if RAW and not WITH_INITIAL:
        vv = tl.load(
            V + (t[:, None] * HV + h) * 128 + v[None, :], t[:, None] < end, 0.0
        )
        val = (vv.to(tl.float32) * bv[:, None]).to(tl.float16)
    x = val.to(tl.float32)
    # Each CTA only marks a head when its window is unsafe. The next
    # kernel observes all marks after the predictor launches complete.
    if bad:
        tl.atomic_or(Dirty + n * HV + h, 1, sem="relaxed")
    out = base + tl.dot(qk, x.to(tl.float16))
    tl.store(
        O + (to[:, None] * HV + h) * 128 + v[None, :], out * SCALE, to[:, None] < end
    )


@triton.jit
def _clear_dirty(Dirty, COUNT: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(Dirty + i, 0, i < COUNT)


@triton.jit
def _try_final(
    K,
    V,
    G,
    B,
    S,
    Sc,
    Cu,
    Index,
    Bad,
    Retry,
    Final,
    FinalScale,
    FM: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    BV: tl.constexpr,
    SL: tl.constexpr,
    CS: tl.constexpr,
    VAR: tl.constexpr,
    IDX: tl.constexpr,
):
    n, h, tv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(Cu + n).to(tl.int32) if VAR else n * T
    end = tl.load(Cu + n + 1).to(tl.int32) if VAR else bos + T
    slot = tl.load(Index + n).to(tl.int64) if IDX else n.to(tl.int64)
    flag = Retry + (n * HV + h) * (128 // BV) + tv
    if slot < 0 or bos >= end or tl.load(Bad + n * HV + h) != 0:
        tl.store(flag, 1)
        return
    z = tl.arange(0, 64)
    k = tl.arange(0, 128)
    v = tv * BV + tl.arange(0, BV)
    t = end - 64 + z
    g = tl.load(G + t * HV + h, t >= bos, 0.0).to(tl.float32)
    b = tl.load(B + t * HV + h, t >= bos, 0.0).to(tl.float32)
    kv = tl.load(
        K + (t[:, None] * H + h // (HV // H)) * 128 + k[None, :], t[:, None] >= bos, 0.0
    ).to(tl.float16)
    values = tl.load(
        V + (t[:, None] * HV + h) * 128 + v[None, :], t[:, None] >= bos, 0.0
    ).to(tl.float32)
    pref = tl.cumsum(g, 0)
    total = tl.sum(g, 0)
    rhs = values * b[:, None]
    initial = tl.full((128, BV), 0.0, tl.float32)
    if end - bos <= 64:
        initial = (
            tl.load(
                S + slot * SL[0] + h * SL[1] + k[:, None] * SL[2] + v[None, :] * SL[3]
            ).to(tl.float32)
            * tl.load(Sc + slot * CS[0] + h * CS[1] + v * CS[2])[None, :]
        )
        sh = initial.to(tl.float16)
        rhs -= (
            b[:, None]
            * tl.exp(pref)[:, None]
            * (
                tl.dot(kv, sh)
                + tl.dot(kv, (initial - sh.to(tl.float32)).to(tl.float16))
            )
        )
    tri = z[:, None] > z[None, :]
    L = tl.where(
        tri,
        -tl.dot(kv, tl.trans(kv))
        * b[:, None]
        * tl.exp(tl.where(tri, pref[:, None] - pref[None, :], 0.0)),
        0.0,
    )
    lh = L.to(tl.float16)
    y = rhs
    for j in tl.static_range(2):
        y = rhs + tl.dot(lh, y.to(tl.float16))
    yh = y.to(tl.float16)
    refined = (
        rhs
        + tl.dot(lh, yh)
        + tl.dot((L - lh.to(tl.float32)).to(tl.float16), yh)
        + tl.dot(lh, (y - yh.to(tl.float32)).to(tl.float16))
    )
    weights = tl.exp(total - pref)
    w = refined * (64.0 * weights)[:, None]
    wh = w.to(tl.float16)
    st = initial * tl.exp(total) + (
        tl.dot(tl.trans(kv), wh)
        + tl.dot(tl.trans(kv), (w - wh.to(tl.float32)).to(tl.float16))
    ) * (1.0 / 64.0)
    mag = tl.max(tl.abs(st), 0)
    # Residuals act as additive rank-one updates. The normalized-key recurrent
    # transition is nonexpansive; weight every residual by its remaining decay
    # instead of applying the largest early residual to the entire tail. This
    # bound does not require the triangular matrix to contract in infinity norm.
    residual_weights = tl.sum(weights[:, None] * tl.abs(L), 0)
    err = 1.01 * tl.sum(residual_weights[:, None] * tl.abs(refined - y), 0)
    discarded = tl.where(end - bos > 64, 130.0 * tl.exp(total), 0.0)
    error = err + discarded
    accepted = tl.min((error * 4096.0 <= mag).to(tl.int32), 0) != 0
    if not accepted:
        safe_bound = tl.where(mag > 0.0, mag, 448.0)
        rounded = (st * (448.0 / safe_bound)[None, :]).to(S.dtype.element_ty).to(
            tl.float32
        ) * (safe_bound / 448.0)[None, :]
        quant_error = tl.max(tl.abs(rounded - st), 0)
        # Use actual FP8 rounding distance. This is stricter than the existing
        # channel-normalized bound, including uncertainty of reference amax.
        tau = 16.0 / 448.0 + 4.0e-4
        accepted = (
            tl.min((quant_error + (1.0 + tau) * error <= tau * mag).to(tl.int32), 0)
            != 0
        )
    if not accepted:
        tl.store(flag, 1)
        return
    tl.store(flag, 0)
    safe = tl.where(mag > 0.0, mag, 448.0)
    packed: tl.constexpr = (
        FM[3] == 1 and FM[0] % 4 == 0 and FM[1] % 4 == 0 and FM[2] % 4 == 0
    )
    if packed:
        store_fp8_values4(
            Final,
            st * (448.0 / safe)[None, :],
            slot,
            h,
            k,
            tv * (BV // 4) + tl.arange(0, BV // 4),
            FM[0],
            FM[1],
            FM[2],
            BV,
        )
    else:
        tl.store(
            Final + slot * FM[0] + h * FM[1] + k[:, None] * FM[2] + v[None, :] * FM[3],
            (st * (448.0 / safe)[None, :]).to(S.dtype.element_ty),
        )
    tl.store(FinalScale + slot * FM[4] + h * FM[5] + v * FM[6], safe / 448.0)


@triton.jit
def _finish(
    K,
    V,
    G,
    B,
    S,
    Sc,
    Cu,
    Index,
    Q,
    O,
    Bad,
    Retry,
    Final,
    FinalScale,
    FMETA: tl.constexpr,
    COPY_EMPTY: tl.constexpr,
    Scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    NS: tl.constexpr,
    BV: tl.constexpr,
    SL0: tl.constexpr,
    SL1: tl.constexpr,
    SL2: tl.constexpr,
    SL3: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    VAR: tl.constexpr,
    INDEXED: tl.constexpr,
    WINDOW: tl.constexpr,
    FINAL_BV: tl.constexpr = 32,
    PARTIAL: tl.constexpr = False,
):
    n, h, tv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(Cu + n).to(tl.int32) if VAR else n * T
    end = tl.load(Cu + n + 1).to(tl.int32) if VAR else bos + T
    slot = tl.load(Index + n).to(tl.int64) if INDEXED else n.to(tl.int64)
    if slot < 0 or (bos >= end and not COPY_EMPTY):
        return
    if tl.load(Retry + (n * HV + h) * (128 // FINAL_BV) + tv // (FINAL_BV // BV)) == 0:
        return
    layout: tl.constexpr = tle.gpu.BlockEncoding([1, 4], [1, 32], [4, 1], [1, 0])
    k = tle.gpu.set_layout(tl.arange(0, 128), tle.gpu.SlicedEncoding(0, layout))
    v = tv * BV + tle.gpu.set_layout(
        tl.arange(0, BV), tle.gpu.SlicedEncoding(1, layout)
    )
    hq = h // (HV // H)
    so = slot * SL0 + h * SL1 + k[None, :] * SL2 + v[:, None] * SL3
    co = slot * C0 + h * C1 + v * C2
    if COPY_EMPTY and bos == end:
        dst = (
            slot * FMETA[0]
            + h * FMETA[1]
            + k[None, :] * FMETA[2]
            + v[:, None] * FMETA[3]
        )
        tl.store(Final + dst, tl.load(S + so))
        tl.store(
            FinalScale + slot * FMETA[4] + h * FMETA[5] + v * FMETA[6], tl.load(Sc + co)
        )
        return
    if PARTIAL:
        fallback = (
            tl.load(Bad + (n * HV + h) * (128 // FINAL_BV) + tv // (FINAL_BV // BV))
            != 0
        )
    else:
        fallback = tl.load(Bad + n * HV + h) != 0
    start = tl.where(fallback, bos, tl.maximum(bos, end - WINDOW))
    s = tl.full((BV, 128), 0.0, tl.float32)
    if start == bos:
        s = tl.load(S + so).to(tl.float32) * tl.load(Sc + co)[:, None]
    for t in range(start, end):
        kk = tl.load(K + (t * H + hq) * 128 + k).to(tl.float32)
        vv = tl.load(V + (t * HV + h) * 128 + v).to(tl.float32)
        decay = tl.exp(tl.load(G + t * HV + h).to(tl.float32))
        beta = tl.load(B + t * HV + h).to(tl.float32)
        s *= decay
        dif = (vv - tl.sum(s * kk[None, :], 1)) * beta
        s = tl.fma(dif[:, None], kk[None, :], s)
        if fallback:
            qq = tl.load(Q + (t * H + hq) * 128 + k).to(tl.float32)
            tl.store(O + (t * HV + h) * 128 + v, tl.sum(s * (qq * Scale)[None, :], 1))
    m = tl.max(tl.abs(s), 1)
    if not fallback and start > bos:
        z = tl.arange(0, WINDOW)
        ll = tl.sum(tl.load(G + (start + z) * HV + h).to(tl.float32), 0)
        # Re-evaluate rows whose admissible truncation error is not negligible
        # against the final channel scale. A 64-token tail spans two guarded blocks.
        if (
            tl.min(tl.broadcast_to(m[:, None], (BV, 128)), axis=None)
            < 130.0 * tl.exp(ll) * 4096.0
        ):
            s = tl.load(S + so).to(tl.float32) * tl.load(Sc + co)[:, None]
            for ti in range(bos, end):
                kv = tl.load(K + (ti * H + hq) * 128 + k).to(tl.float32)
                vv = tl.load(V + (ti * HV + h) * 128 + v).to(tl.float32)
                s *= tl.exp(tl.load(G + ti * HV + h).to(tl.float32))
                dif = (vv - tl.sum(s * kv[None, :], 1)) * tl.load(B + ti * HV + h).to(
                    tl.float32
                )
                s = tl.fma(dif[:, None], kv[None, :], s)
            m = tl.max(tl.abs(s), 1)
    safe = tl.where(m > 0.0, m, 448.0)
    tl.store(
        Final
        + slot * FMETA[0]
        + h * FMETA[1]
        + k[None, :] * FMETA[2]
        + v[:, None] * FMETA[3],
        (s * (448.0 / safe)[:, None]).to(S.dtype.element_ty),
    )
    tl.store(
        FinalScale + slot * FMETA[4] + h * FMETA[5] + v * FMETA[6], safe * (1.0 / 448.0)
    )


@triton.jit
def _split_dot(a, b):
    ah = a.to(tl.float16)
    bh = b.to(tl.float16)
    z = tl.dot(ah, bh)
    if a.dtype == tl.float32:
        z += tl.dot((a - ah.to(tl.float32)).to(tl.float16), bh)
    if b.dtype == tl.float32:
        z += tl.dot(ah, (b - bh.to(tl.float32)).to(tl.float16))
    return z


@triton.jit
def _short_factors(
    Q,
    K,
    G,
    B,
    F,
    Cu,
    VAR: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    bid = tl.program_id(0)
    h = tl.program_id(1)
    beg = (tl.load(Cu).to(tl.int32) if VAR else 0) + bid * BT
    end = tl.load(Cu + 1).to(tl.int32) if VAR else T
    i = tl.arange(0, BT)
    t = beg + i
    k = tl.arange(0, 128)
    kk = tl.load(
        K + (t[:, None] * H + h // (HV // H)) * 128 + k[None, :], t[:, None] < end, 0.0
    ).to(tl.float16)
    qq = tl.load(
        Q + (t[:, None] * H + h // (HV // H)) * 128 + k[None, :], t[:, None] < end, 0.0
    ).to(tl.float16)
    g = tl.load(G + t * HV + h, t < end, 0.0)
    b = tl.load(B + t * HV + h, t < end, 0.0)
    pref = tl.cumsum(g, 0)
    tri = i[:, None] >= i[None, :]
    d = tl.exp(tl.where(tri, pref[:, None] - pref[None, :], 0.0))
    ka = tle.gpu.alloc((BT, 128), tl.float16, init_value=kk, nv_mma_shared_layout=True)
    kb = tle.gpu.alloc((BT, 128), tl.float16, init_value=kk, nv_mma_shared_layout=True)
    qa = tle.gpu.alloc((BT, 128), tl.float16, init_value=qq, nv_mma_shared_layout=True)
    gram = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(ka, kb, trans_b=True))
    qgram = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(qa, kb, trans_b=True))
    L = tl.where(i[:, None] > i[None, :], -gram * d * b[:, None], 0.0)
    qk = tl.where(tri, qgram * d, 0.0)
    eye = (i[:, None] == i[None, :]).to(tl.float32)
    # Use nilpotence: (I-L)^-1 is the product of I+L^(2^j).
    P = L.to(tl.float16)
    pow2 = tl.dot(P, P).to(tl.float16)
    A = eye + L + tl.dot((eye + L).to(tl.float16), pow2)
    pow4 = tl.dot(pow2, pow2).to(tl.float16)
    A += tl.dot(A.to(tl.float16), pow4)
    rn = tl.max(tl.sum(tl.abs(pow4).to(tl.float32), 1), 0)
    # Large triangular powers can cancel catastrophically even when the
    # finite inverse exists. A NaN sentinel preserves state for FP32 replay.
    if ~(rn <= 0.0009):
        A = tl.full((BT, BT), float("nan"), tl.float32)
    xf = A * b[None, :]
    off = F + (bid * HV + h) * (2 * BT * BT + 2 * BT) + i[:, None] * BT + i[None, :]
    tl.store(off, xf)
    tl.store(off + BT * BT, qk)
    tl.store(
        F + (bid * HV + h) * (2 * BT * BT + 2 * BT) + 2 * BT * BT + i, tl.exp(pref)
    )
    tl.store(
        F + (bid * HV + h) * (2 * BT * BT + 2 * BT) + 2 * BT * BT + BT + i,
        tl.exp(tl.sum(g, 0) - pref),
    )


@triton.jit
def _short_apply(
    Q,
    K,
    V,
    F,
    G,
    B,
    FUSED: tl.constexpr,
    S,
    Sc,
    I,
    Cu,
    VAR: tl.constexpr,
    O,
    Final,
    FinalScale,
    FM: tl.constexpr,
    IDX: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    SL: tl.constexpr,
    CS: tl.constexpr,
    Scale: tl.constexpr,
    COPY_EMPTY: tl.constexpr,
    Retry,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    vv = tl.program_id(2) * BV + tl.arange(0, BV)
    sidx = tl.load(I + n).to(tl.int64) if IDX else n.to(tl.int64)
    beg = tl.load(Cu).to(tl.int32) if VAR else 0
    end = tl.load(Cu + 1).to(tl.int32) if VAR else T
    i = tl.arange(0, BT)
    t = beg + i
    k = tl.arange(0, 128)
    if sidx < 0:
        for z in tl.static_range(triton.cdiv(T, BT)):
            tok = beg + z * BT + i
            tl.store(
                O + (tok[:, None] * HV + h) * 128 + vv[None, :], 0.0, tok[:, None] < end
            )
        return

    so = S + sidx * SL[0] + h * SL[1] + k[:, None] * SL[2] + vv[None, :] * SL[3]
    co = Sc + sidx * CS[0] + h * CS[1] + vv * CS[2]
    if end <= beg:
        tl.store(Retry + h * (128 // BV) + tl.program_id(2), 0)
        if COPY_EMPTY:
            tl.store(
                Final
                + sidx * FM[0]
                + h * FM[1]
                + k[:, None] * FM[2]
                + vv[None, :] * FM[3],
                tl.load(so),
            )
            tl.store(FinalScale + sidx * FM[4] + h * FM[5] + vv * FM[6], tl.load(co))
        return
    packed_input: tl.constexpr = (
        (T > 64 or FUSED)
        and SL[3] == 1
        and SL[0] % 4 == 0
        and SL[1] % 4 == 0
        and SL[2] % 4 == 0
    )
    v4 = tl.program_id(2) * (BV // 4) + tl.arange(0, BV // 4)
    if packed_input:
        st = (
            load_fp8_values4(S, sidx, h, k, v4, SL[0], SL[1], SL[2])
            * tl.load(co)[None, :]
        )
    else:
        st = tl.load(so).to(tl.float32) * tl.load(co)[None, :]
    for z in tl.static_range(triton.cdiv(T, BT)):
        t = beg + z * BT + i
        kk = tl.load(
            K + (t[:, None] * H + h // (HV // H)) * 128 + k[None, :],
            t[:, None] < end,
            0.0,
        ).to(tl.float16)
        qq = tl.load(
            Q + (t[:, None] * H + h // (HV // H)) * 128 + k[None, :],
            t[:, None] < end,
            0.0,
        ).to(tl.float16)
        value = tl.load(
            V + (t[:, None] * HV + h) * 128 + vv[None, :], t[:, None] < end, 0.0
        ).to(tl.float32)
        if FUSED:
            g = tl.load(G + t * HV + h, t < end, 0.0)
            b = tl.load(B + t * HV + h, t < end, 0.0)
            pref = tl.cumsum(g, 0)
            tri = i[:, None] >= i[None, :]
            d = tl.exp(tl.where(tri, pref[:, None] - pref[None, :], 0.0))
            ka = tle.gpu.alloc(
                (BT, 128), tl.float16, init_value=kk, nv_mma_shared_layout=True
            )
            kb = tle.gpu.alloc(
                (BT, 128), tl.float16, init_value=kk, nv_mma_shared_layout=True
            )
            qa = tle.gpu.alloc(
                (BT, 128), tl.float16, init_value=qq, nv_mma_shared_layout=True
            )
            gram = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(ka, kb, trans_b=True))
            qgram = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(qa, kb, trans_b=True))
            L = tl.where(i[:, None] > i[None, :], -gram * d * b[:, None], 0.0)
            qk = tl.where(tri, qgram * d, 0.0)
            eye = (i[:, None] == i[None, :]).to(tl.float32)
            # Use nilpotence: (I-L)^-1 is the product of I+L^(2^j).
            P = L.to(tl.float16)
            pow2 = tl.dot(P, P).to(tl.float16)
            A = eye + L + tl.dot((eye + L).to(tl.float16), pow2)
            pow4 = tl.dot(pow2, pow2).to(tl.float16)
            A += tl.dot(A.to(tl.float16), pow4)
            rn = tl.max(tl.sum(tl.abs(pow4).to(tl.float32), 1), 0)
            # Large triangular powers can cancel catastrophically even when the
            # finite inverse exists. A NaN sentinel preserves state for FP32 replay.
            if ~(rn <= 0.0009):
                A = tl.full((BT, BT), float("nan"), tl.float32)
            xf = A * b[None, :]
            c = xf
            qmat = qk
            pre = tl.exp(pref)
            rev = tl.exp(tl.sum(g, 0) - pref)
        else:
            f = F + ((n * triton.cdiv(T, BT) + z) * HV + h) * (2 * BT * BT + 2 * BT)
            pre = tl.load(f + 2 * BT * BT + i)
            rev = tl.load(f + 2 * BT * BT + BT + i)
        rhs = value - pre[:, None] * tl.dot(kk, st.to(tl.float16))
        if not FUSED:
            c = tl.load(f + i[:, None] * BT + i[None, :])
        x = tl.dot(c.to(tl.float16), rhs.to(tl.float16))
        if not FUSED:
            qmat = tl.load(f + BT * BT + i[:, None] * BT + i[None, :])
        o = pre[:, None] * tl.dot(qq, st.to(tl.float16)) + tl.dot(
            qmat.to(tl.float16), x.to(tl.float16)
        )
        tl.store(
            O + (t[:, None] * HV + h) * 128 + vv[None, :], o * Scale, t[:, None] < end
        )
        w = x * (64.0 * rev)[:, None]
        st = st * tl.sum(tl.where(i == BT - 1, pre, 0.0), 0) + _split_dot(
            tl.trans(kk), w
        ) * (1.0 / 64.0)
    m = tl.max(tl.abs(st), 0)
    check_t = beg + tl.arange(0, triton.next_power_of_2(T))
    check_g = tl.load(G + check_t * HV + h, check_t < end, 0.0)
    check_b = tl.load(B + check_t * HV + h, check_t < end, 0.0)
    valid_g = (
        tl.min(
            (
                (check_g >= -80.0)
                & (check_g <= 0.0)
                & (check_b >= 0.0)
                & (check_b <= 1.0)
            ).to(tl.int32),
            0,
        )
        != 0
    )
    valid_s = tl.max(tl.abs(tl.load(co)), 0) <= 4.0 / 448.0
    finite = tl.min(((m == m) & (m < float("inf"))).to(tl.int32), 0) != 0
    # Commit a complete V tile only after its range checks pass. Rejected
    # tiles keep their original input state for the subsequent FP32 replay.
    bad = ~valid_g | ~valid_s | ~finite
    tl.store(Retry + h * (128 // BV) + tl.program_id(2), bad)
    if bad:
        return
    safe = tl.where(m > 0.0, m, 448.0)
    packed_output: tl.constexpr = (
        (T > 64 or FUSED)
        and FM[3] == 1
        and FM[0] % 4 == 0
        and FM[1] % 4 == 0
        and FM[2] % 4 == 0
    )
    if packed_output:
        store_fp8_values4(
            Final, st * (448.0 / safe)[None, :], sidx, h, k, v4, FM[0], FM[1], FM[2], BV
        )
    else:
        tl.store(
            Final + sidx * FM[0] + h * FM[1] + k[:, None] * FM[2] + vv[None, :] * FM[3],
            (st * (448.0 / safe)[None, :]).to(S.dtype.element_ty),
        )
    tl.store(
        FinalScale + sidx * FM[4] + h * FM[5] + vv * FM[6],
        tl.where((m == m) & (m < float("inf")), safe / 448.0, float("nan")),
    )


def chunk_prefill(
    q,
    k,
    v,
    g,
    beta,
    scale,
    state,
    sc,
    cu,
    idx,
    out,
    final_state=None,
    final_scale=None,
    copy_empty=False,
    value_bounds=None,
):
    batch, tokens, heads, _ = q.shape
    hv = v.shape[2]
    ns = batch if cu is None else cu.numel() - 1
    if ns == 1 and 32 < tokens <= 128:
        real_final = state if final_state is None else final_state
        real_scale = sc if final_scale is None else final_scale
        final_state, final_scale = real_final, real_scale
        bt = 64
        fused = tokens <= 64
        bv = 128 if fused else 64
        nc = triton.cdiv(tokens, bt)
        retry = torch.empty((hv, 128 // bv), device=q.device, dtype=torch.int32)
        coeff = (
            None
            if fused
            else torch.empty(
                (nc, hv, 2 * bt * bt + 2 * bt), device=q.device, dtype=torch.float32
            )
        )
        if final_state is None:
            final_state, final_scale = state, sc
        fm = (*final_state.stride(), *final_scale.stride())
        if not fused:
            _short_factors[(nc, hv)](
                q,
                k,
                g,
                beta,
                coeff,
                cu,
                cu is not None,
                heads,
                hv,
                tokens,
                bt,
                num_warps=16,
                num_stages=1,
                enable_backend_opt=True,
            )
        _short_apply[(1, hv, 128 // bv)](
            q,
            k,
            v,
            coeff,
            g,
            beta,
            fused,
            state,
            sc,
            idx,
            cu,
            cu is not None,
            out,
            final_state,
            final_scale,
            fm,
            idx is not None,
            heads,
            hv,
            tokens,
            bt,
            bv,
            state.stride(),
            sc.stride(),
            scale,
            copy_empty,
            retry,
            num_warps=16 if fused else 8,
            num_stages=1,
            enable_backend_opt=True,
            extern_libs={"": GDN_VECTOR_LIBRARY},
        )
        rfm = (*real_final.stride(), *real_scale.stride())
        _finish[(1, hv, 16)](
            k,
            v,
            g,
            beta,
            state,
            sc,
            cu,
            idx,
            q,
            out,
            retry,
            retry,
            real_final,
            real_scale,
            rfm,
            copy_empty,
            scale,
            tokens,
            heads,
            hv,
            1,
            8,
            *state.stride(),
            *sc.stride(),
            cu is not None,
            idx is not None,
            64,
            bv,
            True,
            num_warps=4,
            num_stages=1,
        )
        return out, real_final, real_scale
    capacity = batch * triton.cdiv(tokens, 32) if cu is None else tokens // 32 + ns
    table = None
    if cu is not None and ns > 1:
        table = torch.empty((capacity, 4), device=q.device, dtype=torch.int32)
        _chunk_map[(triton.cdiv(capacity, 32),)](
            cu, table, tokens, ns, 32, capacity, num_warps=4
        )
    dirty = torch.empty((ns, hv), device=q.device, dtype=torch.int32)
    _clear_dirty[(triton.cdiv(ns * hv, 256),)](dirty, ns * hv, num_warps=2)
    query_desc = None
    if value_bounds is not None:
        query_desc = TensorDescriptor(
            q, [batch * tokens, heads * 128], [heads * 128, 1], [32, 128]
        )
    tensors = (q, k, v, g, beta, state, sc, cu, idx, out, table, dirty)
    strides = (*state.stride(), *sc.stride(), scale, cu is not None, idx is not None)
    # Most CTAs need only 32 output rows. The first 64 rows use the
    # actual initial state, so split V to expose enough parallelism.
    _window_output[(capacity, hv, 1)](
        *tensors,
        tokens,
        heads,
        hv,
        ns,
        64,
        128,
        32,
        *strides,
        False,
        query_desc,
        value_bounds,
        value_bounds is not None,
        num_warps=4,
        num_stages=1,
        enable_backend_opt=True,
    )
    prefix_v = 64
    _window_output[(ns, hv, 128 // prefix_v)](
        *tensors,
        tokens,
        heads,
        hv,
        ns,
        64,
        prefix_v,
        32,
        *strides,
        True,
        query_desc,
        value_bounds,
        value_bounds is not None,
        num_warps=8,
        num_stages=1,
        enable_backend_opt=True,
        extern_libs={"": GDN_VECTOR_LIBRARY},
    )
    if final_state is None:
        final_state, final_scale = state, sc
    fm = (*final_state.stride(), *final_scale.stride())
    final_v = 32
    retry = torch.empty((ns, hv, 128 // final_v), device=q.device, dtype=torch.int32)
    _try_final[(ns, hv, 128 // final_v)](
        k,
        v,
        g,
        beta,
        state,
        sc,
        cu,
        idx,
        dirty,
        retry,
        final_state,
        final_scale,
        fm,
        tokens,
        heads,
        hv,
        final_v,
        state.stride(),
        sc.stride(),
        cu is not None,
        idx is not None,
        num_warps=8,
        num_stages=1,
        enable_backend_opt=True,
        extern_libs={"": GDN_VECTOR_LIBRARY},
    )
    _finish[(ns, hv, 16)](
        k,
        v,
        g,
        beta,
        state,
        sc,
        cu,
        idx,
        q,
        out,
        dirty,
        retry,
        final_state,
        final_scale,
        fm,
        copy_empty,
        scale,
        tokens,
        heads,
        hv,
        ns,
        8,
        *state.stride(),
        *sc.stride(),
        cu is not None,
        idx is not None,
        64,
        final_v,
        num_warps=4,
        num_stages=1,
    )
    return out, final_state, final_scale
