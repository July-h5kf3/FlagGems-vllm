# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""One mixed launch with Triton vector stages and native INT4 GEMMs."""
from functools import lru_cache

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.runtime.backend._ascend.ops.marlin_w4a16.custom_mixed import register
from flaggems_vllm.runtime.backend._ascend.ops.marlin_w4a16.prepare_packed import (
    prepare,
)


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
    SW2: tl.constexpr,
    C1: tl.constexpr,
    V1: tl.constexpr,
    C2: tl.constexpr,
    V2: tl.constexpr,
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
        dummy = tl.full((16,), 0, tl.int32)
        al.custom(C1, ax, work, ep, h, pid, out=dummy)
        al.sync_block_all("all", 10)
        al.sync_block_all("all", 10)
        al.custom(C2, a, work, ep, z, pid, out=dummy)
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
        scratch1 = tl.full((7 * 64 * 128 // 4,), 0, tl.int32)
        al.custom(V1, Q1, S1, F1, ep, work, pid, sub, out=scratch1)
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
        scratch2 = tl.full((SW2,), 0, tl.int32)
        al.custom(V2, Q2, S2, F2, ep, work, pid, sub, out=scratch2)
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
    c1, v1 = register(2 * n, k, 16, min(2 * n, 128), r * (2 * n // min(2 * n, 128)), g)
    c2, v2 = register(k, n, 16, min(k & -k, 256), r * (k // min(k & -k, 256)), g)
    return sizes, (c1, v1, c2, v2)


def run(x, w1, w2, s1, s2, p, ids):
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
        7 * (min(k & -k, 256) // 2) * 128 // 4,
        *ops,
        disable_auto_inject_block_sync=True,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    return out
