# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import triton
import triton.experimental.tle as tle
import triton.language as tl


@triton.jit
def tile(
    A,
    Work,
    Output,
    row,
    col,
    pid,
    iteration,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    mi = tl.arange(0, BM)
    ni = tl.arange(0, BN)
    ki = tl.arange(0, 128)
    acc = tl.full((BM, BN), 0, tl.float32)
    for kb in range(K // 128):
        tle.dsa.ascend.sync_block_wait(
            "vector",
            "cube",
            2,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
        )
        a = tl.load(A + (row + mi[:, None]) * K + kb * 128 + ki[None, :])
        b = tl.load(
            Work
            + (pid * 2 + iteration % 2) * BN * 128
            + ni[None, :] * 128
            + ki[:, None]
        )
        tle.dsa.ascend.sync_block_set(
            "cube",
            "vector",
            3,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
        )
        acc = tl.dot(a, b, acc)
        iteration += 1
    tl.store(Output + (row + mi[:, None]) * N + col + ni[None, :], acc.to(tl.bfloat16))
    return iteration


@triton.jit
def cube(
    A,
    Work,
    Experts,
    Output,
    PID,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TASKS: tl.constexpr,
    GRID: tl.constexpr,
    MERGE: tl.constexpr,
):
    tle.dsa.ascend.raw("cube_begin", PID)
    iteration = 0
    for task in range(PID, TASKS, GRID):
        tile_id = task // (N // BN)
        col = task % (N // BN) * BN
        expert = tl.load(Experts + tile_id)
        active = expert >= 0
        if MERGE:
            previous = tile_id - 1
            local_tile = 0
            same = active
            while (previous >= 0) & same:
                before = tl.load(Experts + previous)
                same = before == expert
                local_tile += same.to(tl.int32)
                previous -= 1
            active = active & (local_tile % 2 == 0)
        if active:
            if MERGE:
                next_expert = -2
                if tile_id + 1 < TASKS // (N // BN):
                    next_expert = tl.load(Experts + tile_id + 1)
                if next_expert == expert:
                    iteration = tile(
                        A,
                        Work,
                        Output,
                        tile_id * 128,
                        col,
                        PID,
                        iteration,
                        K,
                        N,
                        256,
                        BN,
                    )
                else:
                    iteration = tile(
                        A,
                        Work,
                        Output,
                        tile_id * 128,
                        col,
                        PID,
                        iteration,
                        K,
                        N,
                        128,
                        BN,
                    )
            else:
                iteration = tile(
                    A, Work, Output, tile_id * BM, col, PID, iteration, K, N, BM, BN
                )

    tle.dsa.ascend.raw("cube_end", PID)
