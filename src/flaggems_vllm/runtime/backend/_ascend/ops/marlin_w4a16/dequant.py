# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""TLE Vector dequantization with only native INT4-to-FP16 Cast."""

import triton
import triton.experimental.tle as tle
import triton.language as tl


@triton.jit
def compute(q, scales, fast, RB: tl.constexpr, OP: tl.constexpr):
    h = tl.full((RB * 128,), 0, tl.float16)
    h = tle.dsa.ascend.raw("cast_int4_to_fp16", q, out=h)
    values_h = tl.reshape(h, (RB, 128))
    result = (values_h.to(tl.float32) * scales.to(tl.float32)[:, None]).to(tl.bfloat16)
    return result


@triton.jit
def dequant(
    Q,
    S,
    Safe,
    Experts,
    Work,
    PID,
    SUB,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    TASKS: tl.constexpr,
    GRID: tl.constexpr,
    MERGE: tl.constexpr,
    OP: tl.constexpr,
):
    VBN: tl.constexpr = BN // 2
    # Limit temporary UB usage for FP32 scaling on CANN 9.0.
    CB: tl.constexpr = 64 if VBN > 64 else VBN
    ns = tl.arange(0, CB)
    ks = tl.arange(0, 128)
    iteration = 0
    for task in range(PID, TASKS, GRID):
        tile = task // (N // BN)
        pn = task % (N // BN) * 2 + SUB
        expert = tl.load(Experts + tile)
        active = expert >= 0
        if MERGE:
            previous = tile - 1
            local_tile = 0
            same = active
            while (previous >= 0) & same:
                before = tl.load(Experts + previous)
                same = before == expert
                local_tile += same.to(tl.int32)
                previous -= 1
            active = active & (local_tile % 2 == 0)
        if active:
            for kb in range(K // 128):
                base = (expert * (K // 128) + kb) * N + pn * VBN
                fast = False
                for chunk in range(VBN // CB):
                    packed = tl.load(
                        Q + (base + chunk * CB) * 64 + tl.arange(0, CB * 64)
                    )
                    scale = tl.load(S + base + chunk * CB + ns)
                    result = compute(packed, scale, fast, CB, OP)
                    # Wait once before overwriting either half of the GM stage.
                    if iteration >= 2 and chunk == 0:
                        tle.dsa.ascend.sync_block_wait(
                            "cube",
                            "vector",
                            3,
                            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
                            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
                        )
                    offset = (
                        (PID * 2 + iteration % 2) * BN * 128
                        + SUB * VBN * 128
                        + chunk * CB * 128
                    )
                    tl.store(Work + offset + ns[:, None] * 128 + ks[None, :], result)
                # Notify Cube only after all subtiles have been stored.
                tle.dsa.ascend.sync_block_set(
                    "vector",
                    "cube",
                    2,
                    sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
                    receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
                )
                iteration += 1
    for _ in range(tl.minimum(iteration, 2)):
        tle.dsa.ascend.sync_block_wait(
            "cube",
            "vector",
            3,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
        )
