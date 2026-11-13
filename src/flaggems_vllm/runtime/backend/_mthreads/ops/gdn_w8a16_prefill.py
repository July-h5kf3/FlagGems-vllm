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
):
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
    qq = tl.load(Q + (to[:, None] * H + qh) * 128 + k[None, :], to[:, None] < end, 0.0)
    vv = tl.load(V + (t[:, None] * HV + h) * 128 + v[None, :], t[:, None] < end, 0.0)
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
    ).to(tl.bfloat16)
    ggood = tl.min(
        ((gv >= -80.0) & (gv <= 0.0) & (bv >= 0.0) & (bv <= 1.0)).to(tl.int32), 0
    )
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
    # ||L||_inf <= 1/2 bounds the remaining Neumann series by twice
    # the first omitted term. The per-row sums below bound Q L^3.
    lnorm = tl.max(tl.sum(tl.abs(L).to(tl.float32), 1), 0)
    bad = bad | ~(lnorm <= 0.5) | ~(bottom >= -140.0)
    if WITH_INITIAL:
        bad = bad | (bos + 64 <= end) & (tl.sum(tl.where(i >= 32, gv, 0.0), 0) > -14.0)
    rhs = vv.to(tl.float32)
    base = tl.full((BT if WITH_INITIAL else BLOCK, BV), 0.0, tl.float32)
    if WITH_INITIAL:
        s = (
            tl.load(S + slot * SL0 + h * SL1 + k[:, None] * SL2 + v[None, :] * SL3).to(
                tl.float32
            )
            * tl.load(Sc + slot * C0 + h * C1 + v * C2)[None, :]
        )
        bad = bad | ~(tl.max(tl.abs(s)) <= 4.0)
        sf = s.to(tl.float16)
        rhs -= tl.dot(kk.to(tl.float16), sf) * tl.exp(pref)[:, None]
        base = tl.dot(qq.to(tl.float16), sf) * tl.exp(po)[:, None]
    val = (rhs * bv[:, None]).to(tl.float16)
    q0 = qk.to(tl.float16)
    e1 = tl.dot(q0, L)
    e2 = tl.dot(e1.to(tl.float16), L)
    bound = tl.max(
        tl.sum(tl.abs(e2) * tl.sum(tl.abs(L).to(tl.float32), 1)[None, :], 1), 0
    )
    bad = bad | ~(bound * tl.max(tl.abs(val).to(tl.float32)) * tl.abs(SCALE) <= 6e-05)
    qk = (q0 + e1 + e2).to(tl.float16)
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
):
    n, h, tv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(Cu + n).to(tl.int32) if VAR else n * T
    end = tl.load(Cu + n + 1).to(tl.int32) if VAR else bos + T
    slot = tl.load(Index + n).to(tl.int64) if INDEXED else n.to(tl.int64)
    if slot < 0 or (bos >= end and not COPY_EMPTY):
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
):
    batch, tokens, heads, _ = q.shape
    hv = v.shape[2]
    ns = batch if cu is None else cu.numel() - 1
    capacity = batch * triton.cdiv(tokens, 32) if cu is None else tokens // 32 + ns
    table = None
    if cu is not None and ns > 1:
        table = torch.empty((capacity, 4), device=q.device, dtype=torch.int32)
        _chunk_map[(triton.cdiv(capacity, 32),)](
            cu, table, tokens, ns, 32, capacity, num_warps=4
        )
    dirty = torch.empty((ns, hv), device=q.device, dtype=torch.int32)
    _clear_dirty[(triton.cdiv(ns * hv, 256),)](dirty, ns * hv, num_warps=2)
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
        num_warps=4,
        num_stages=1,
        enable_backend_opt=True,
    )
    _window_output[(ns, hv, 2)](
        *tensors,
        tokens,
        heads,
        hv,
        ns,
        64,
        64,
        32,
        *strides,
        True,
        num_warps=8,
        num_stages=1,
        enable_backend_opt=True,
    )
    if final_state is None:
        final_state, final_scale = state, sc
    fm = (*final_state.stride(), *final_scale.stride())
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
        num_warps=4,
        num_stages=1,
    )
    return out, final_state, final_scale
