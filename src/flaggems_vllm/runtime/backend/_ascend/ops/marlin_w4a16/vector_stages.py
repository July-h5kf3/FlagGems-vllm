# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Triton row packing, SwiGLU and weighted unpermutation."""
import triton
import triton.experimental.tle as tle
import triton.language as tl


@triton.jit
def silu_kernel(
    H,
    Offsets,
    O,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    E: tl.constexpr,
    RB: tl.constexpr,
    NC: tl.constexpr,
):
    active = TOTAL // N
    if E >= 0:
        active = tl.load(Offsets + E)
    for block in range(
        tl.program_id(0), tl.cdiv(active, RB) * tl.cdiv(N, NC), tl.num_programs(0)
    ):
        row = (block // tl.cdiv(N, NC)) * RB
        col = (block % tl.cdiv(N, NC)) * NC
        pa = tl.make_block_ptr(
            H,
            shape=(active, 2 * N),
            strides=(2 * N, 1),
            offsets=(row, col),
            block_shape=(RB, NC),
            order=(1, 0),
        )
        pb = tl.make_block_ptr(
            H,
            shape=(active, 2 * N),
            strides=(2 * N, 1),
            offsets=(row, col + N),
            block_shape=(RB, NC),
            order=(1, 0),
        )
        po = tl.make_block_ptr(
            O,
            shape=(active, N),
            strides=(N, 1),
            offsets=(row, col),
            block_shape=(RB, NC),
            order=(1, 0),
        )
        a = tl.load(pa, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b = tl.load(pb, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        out = a / (1 + tl.exp(-a)) * b
        tl.store(po, out.to(O.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def combine_kernel(
    A,
    P,
    Inv,
    O,
    K: tl.constexpr,
    T: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    HAS_INV: tl.constexpr,
):
    for block in range(tl.program_id(0), M * tl.cdiv(K, B), tl.num_programs(0)):
        row = block // tl.cdiv(K, B)
        cols = block % tl.cdiv(K, B) * B + tl.arange(0, B)
        acc = tl.full((B,), 0, tl.float32)
        for t in range(T):
            route = row * T + t
            pos = route
            if HAS_INV:
                pos = tl.load(Inv + route)
            prob = tl.load(P + route)
            a = tl.load(A + pos * K + cols, cols < K, other=0).to(tl.float32)
            acc += a * prob
        tl.store(O + row * K + cols, acc, cols < K)


def silu(h, out, off=None, b=32):
    silu_kernel[(cores(h),)](
        h,
        off if off is not None else out,
        out,
        out.shape[1],
        out.numel(),
        off.numel() - 1 if off is not None else -1,
        b,
        min(out.shape[1], 256),
        num_warps=1,
        multibuffer=False,
        enable_fp_fusion=False,
    )


def combine(a, p, out, inv=None, b=4096):
    combine_kernel[(cores(a),)](
        a,
        p,
        inv if inv is not None else p,
        out,
        out.shape[1],
        p.shape[1],
        out.shape[0],
        min(b, triton.next_power_of_2(out.shape[1])),
        inv is not None,
        num_warps=1,
        multibuffer=True,
        enable_fp_fusion=False,
    )


@triton.jit
def pack_kernel(
    X,
    R,
    C,
    Offsets,
    Experts,
    O,
    K: tl.constexpr,
    T: tl.constexpr,
    RR: tl.constexpr,
    BM: tl.constexpr,
    TILES: tl.constexpr,
    BR: tl.constexpr,
    PK: tl.constexpr,
):
    for task in range(tl.program_id(0), TILES * (BM // BR), tl.num_programs(0)):
        tile = task // (BM // BR)
        ex = tl.load(Experts + tile)
        if ex >= 0:
            begin = tl.load(Offsets + ex)
            count = tl.load(C + ex)
            row = task * BR + tl.arange(0, BR)
            local = row - begin
            route = tl.load(R + ex * RR + local, local < count, other=0)
            kk = tl.arange(0, PK)
            x = tl.load(
                X + (route // T)[:, None] * K + kk[None, :],
                (local < count)[:, None] & (kk < K)[None, :],
                other=0,
            )
            if PK == K:
                # The loaded tensor already zeroes invalid routes.
                # All destinations fit a complete padded M tile.
                buf = tle.dsa.to_buffer(x, space=tle.dsa.ascend.UB)
                with tle.dsa.hint(inter_no_alias=True):
                    tle.dsa.copy(buf, O + row[:, None] * K + kk[None, :], [BR, K])
            else:
                tl.store(O + row[:, None] * K + kk[None, :], x, (kk < K)[None, :])


def cores(a):
    return triton.runtime.driver.active.utils.get_device_properties(a.device.index)[
        "num_vectorcore"
    ]


def pack(x, r, c, off, e, out, bm, t, br=4):
    if r.shape[1] <= c.numel() * bm // 2:
        # Sparse routing produces mostly padding: skip input loads for zero rows.
        pack_rows_kernel[(min(e.numel(), cores(x)),)](
            x,
            r,
            c,
            off,
            e,
            out,
            x.shape[1],
            t,
            r.shape[1],
            bm,
            e.numel(),
            triton.next_power_of_2(x.shape[1]),
            num_warps=1,
            multibuffer=False,
        )
        return
    pack_kernel[(cores(x),)](
        x,
        r,
        c,
        off,
        e,
        out,
        x.shape[1],
        t,
        r.shape[1],
        bm,
        e.numel(),
        br,
        triton.next_power_of_2(x.shape[1]),
        num_warps=1,
        multibuffer=False,
    )


@triton.jit
def pack_rows_kernel(
    X,
    R,
    C,
    Offsets,
    Experts,
    O,
    K: tl.constexpr,
    T: tl.constexpr,
    RR: tl.constexpr,
    BM: tl.constexpr,
    TILES: tl.constexpr,
    PK: tl.constexpr,
):
    kk = tl.arange(0, PK)
    for tile in range(tl.program_id(0), TILES, tl.num_programs(0)):
        ex = tl.load(Experts + tile)
        if ex >= 0:
            begin = tl.load(Offsets + ex)
            count = tl.load(C + ex)
            for row in range(BM):
                local = tile * BM + row - begin
                value = tl.full((PK,), 0, tl.bfloat16)
                if local < count:
                    route = tl.load(R + ex * RR + local)
                    value = tl.load(X + (route // T) * K + kk, kk < K, other=0)
                tl.store(O + (tile * BM + row) * K + kk, value, kk < K)
