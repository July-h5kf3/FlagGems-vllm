# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import weakref

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(
    W, S, Q, T, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, TASKS: tl.constexpr
):
    for pid in range(tl.program_id(0), TASKS, tl.num_programs(0)):
        pn = pid % tl.cdiv(N, BN)
        g = (pid // tl.cdiv(N, BN)) % (K // 128)
        e = pid // (tl.cdiv(N, BN) * (K // 128))
        ns = pn * BN + tl.arange(0, BN)
        kh = tl.arange(0, 64)
        v = tl.load(
            W + e * N * (K // 2) + ns[:, None] * (K // 2) + g * 64 + kh[None, :],
            ns[:, None] < N,
            other=0,
        )
        tl.store(
            Q + ((e * (K // 128) + g) * N + ns[:, None]) * 64 + kh[None, :],
            v ^ 0x88,
            ns[:, None] < N,
        )
        s = tl.load(S + e * N * (K // 128) + ns * (K // 128) + g, ns < N, other=0)
        tl.store(T + (e * (K // 128) + g) * N + ns, s, ns < N)


_cache = {}


def prepare(w, s):
    key = (id(w), id(s))
    try:
        version = (w._version, s._version, w.data_ptr(), s.data_ptr())
    except RuntimeError:
        version = None
    item = _cache.get(key)
    if (
        version is not None
        and item is not None
        and item[0]() is w
        and item[1]() is s
        and item[2] == version
    ):
        return item[3:]
    e, n, k2 = w.shape
    k = k2 * 2
    q = torch.empty((e, k // 128, n, 64), device=w.device, dtype=torch.uint8)
    scale = torch.empty((e, k // 128, n), device=s.device, dtype=s.dtype)
    tasks = e * (k // 128) * triton.cdiv(n, 32)
    _pack[(min(tasks, 1024),)](w, s, q, scale, n, k, 32, tasks)

    def remove(_):
        _cache.pop(key, None)

    if version is not None:
        _cache[key] = (
            weakref.ref(w, remove),
            weakref.ref(s, remove),
            version,
            q,
            scale,
        )
    return q, scale
