# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
from flaggems_vllm.runtime.backend._ascend.ops.marlin_w4a16.prepare_packed import (
    prepare,
)

from .cube import cube
from .dequant import dequant


@triton.jit
def kernel(
    A,
    Q,
    S,
    Safe,
    Experts,
    Work,
    Output,
    BM: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    TASKS: tl.constexpr,
    GRID: tl.constexpr,
    MERGE: tl.constexpr,
    OP: tl.constexpr,
):
    with tle.scope(core_mode="cube"):
        cube(
            A, Work, Experts, Output, tl.program_id(0), N, K, BM, BN, TASKS, GRID, MERGE
        )
    with tle.scope(core_mode="vector"):
        dequant(
            Q,
            S,
            Safe,
            Experts,
            Work,
            tl.program_id(0),
            tle.dsa.ascend.sub_vec_id(),
            N,
            K,
            BN,
            TASKS,
            GRID,
            MERGE,
            OP,
        )


def gemm(a, w, s, experts, out, bm, bn=128):
    n, k = out.shape[1], a.shape[1]
    bn = min(n & -n, 256, 32768 // bm)
    merge = bm == 128 and ((k == 256 and n == 4096) or (k == 4096 and n == 512))
    input_bm = bm
    if merge:
        bm, bn = 256, 128
    tasks = a.shape[0] // input_bm * (n // bn)
    cores = triton.runtime.driver.active.utils.get_device_properties(a.device.index)[
        "num_aicore"
    ]
    grid = min(tasks, cores)
    if merge and k == 4096:
        grid = min(grid, 19)
    q, scale, safe = prepare(w, s)
    work = torch.empty((grid * 2 * bn * 128,), device=a.device, dtype=a.dtype)
    kernel[(grid,)](
        a,
        q,
        scale,
        safe,
        experts,
        work,
        out,
        bm,
        n,
        k,
        bn,
        tasks,
        grid,
        merge,
        "cast_int4_to_fp16",
        disable_auto_inject_block_sync=True,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )
