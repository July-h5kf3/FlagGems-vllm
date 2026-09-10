# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Small MoE with TLE Cube GEMMs and FlagTree INT4 Cast."""

from functools import lru_cache

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from flaggems_vllm.runtime.backend._ascend.ops.marlin_w4a16.prepare_packed import (
    prepare,
)

from . import primitives as boundary
from .cube import cube
from .dequant import dequant


@triton.jit
def _small_fused_kernel(
    X,
    Q1,
    S1,
    F1,
    Q2,
    S2,
    F2,
    IDs,
    P,
    EP,
    AX,
    H,
    Act,
    Z,
    W,
    O,
    K: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    PK: tl.constexpr,
    BN2: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
):
    ep = EP
    ax = AX
    h = H
    a = Act
    z = Z
    work = W
    pid = tl.program_id(0)
    # Every physical pair enters each barrier, including pairs with no routes.
    # Global barrier 10 is separate from the GEMM ring flags 2 and 3.
    with al.scope(core_mode="cube"):
        al.sync_block_all("all", 10)
        cube(ax, work, ep, h, pid, 2 * N, K, 16, 128, M * T * (2 * N // 128), G, False)
        al.sync_block_all("all", 10)
        al.sync_block_all("all", 10)
        cube(a, work, ep, z, pid, K, N, 16, BN2, M * T * (K // BN2), G, False)
        al.sync_block_all("all", 10)
        al.sync_block_all("all", 10)
    with al.scope(core_mode="vector"):
        sub = al.sub_vec_id()
        vp = pid * 2 + sub
        kk = tl.arange(0, PK)
        for route in range(vp, M * T, G * 2):
            expert = tl.load(IDs + route)
            tl.store(ep + route, expert)
            x = tl.load(X + (route // T) * K + kk, kk < K, other=0)
            tl.store(ax + route * 16 * K + kk, x, kk < K)
            for row in range(1, 16):
                tl.store(ax + (route * 16 + row) * K + kk, 0, kk < K)
        al.sync_block_all("all", 10)
        dequant(
            Q1,
            S1,
            F1,
            ep,
            work,
            pid,
            sub,
            2 * N,
            K,
            128,
            M * T * (2 * N // 128),
            G,
            False,
            "cast_int4_to_fp16",
        )
        al.sync_block_all("all", 10)
        for act_block in range(vp, M * T * 16 * tl.cdiv(N, 256), G * 2):
            act_row = act_block // tl.cdiv(N, 256)
            act_col = act_block % tl.cdiv(N, 256) * 256 + tl.arange(0, 256)
            av = tl.load(h + act_row * N * 2 + act_col, act_col < N, other=0).to(
                tl.float32
            )
            bv = tl.load(h + act_row * N * 2 + act_col + N, act_col < N, other=0).to(
                tl.float32
            )
            value = av / (1 + tl.exp(-av)) * bv
            tl.store(a + act_row * N + act_col, value, act_col < N)
        al.sync_block_all("all", 10)
        dequant(
            Q2,
            S2,
            F2,
            ep,
            work,
            pid,
            sub,
            K,
            N,
            BN2,
            M * T * (K // BN2),
            G,
            False,
            "cast_int4_to_fp16",
        )
        al.sync_block_all("all", 10)
        for combine_block in range(vp, M * tl.cdiv(K, 256), G * 2):
            combine_row = combine_block // tl.cdiv(K, 256)
            combine_col = combine_block % tl.cdiv(K, 256) * 256 + tl.arange(0, 256)
            total = tl.full((256,), 0, tl.float32)
            for j in range(T):
                restore_route = combine_row * T + j
                zv = tl.load(
                    z + restore_route * 16 * K + combine_col, combine_col < K, other=0
                ).to(tl.float32)
                prob = tl.load(P + restore_route)
                total = total + zv * prob
            tl.store(O + combine_row * K + combine_col, total, combine_col < K)
        al.sync_block_all("all", 10)


@lru_cache(maxsize=128)
def config(m, k, n, t, g):
    r = m * t
    sizes = (
        r * 16 * k,
        r * 16 * 2 * n,
        r * 16 * n,
        r * 16 * k,
        g * 2 * max(128, min(k & -k, 256)) * 128,
    )
    c1 = 0
    c2 = 0
    return sizes, (c1, c2)


def run(x, w1, w2, s1, s2, p, ids):
    boundary.register()
    m, k = x.shape
    n = w1.shape[1] // 2
    t = ids.shape[1]
    g = triton.runtime.driver.active.utils.get_device_properties(x.device.index)[
        "num_aicore"
    ]
    sizes, ops = config(m, k, n, t, g)
    q1, s1, f1 = prepare(w1, s1)
    q2, s2, f2 = prepare(w2, s2)
    meta = torch.empty(m * t, device=x.device, dtype=torch.int32)
    # Direct typed allocations avoid unsupported pointer casts and view-dispatch overhead.
    buffers = [torch.empty(size, device=x.device, dtype=x.dtype) for size in sizes]
    out = torch.empty_like(x)
    _small_fused_kernel[(g,)](
        x,
        q1,
        s1,
        f1,
        q2,
        s2,
        f2,
        ids,
        p,
        meta,
        *buffers,
        out,
        k,
        n,
        m,
        t,
        g,
        triton.next_power_of_2(k),
        min(k & -k, 256),
        *ops,
        disable_auto_inject_block_sync=True,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    return out
