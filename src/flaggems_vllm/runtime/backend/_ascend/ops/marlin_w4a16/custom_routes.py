# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import triton
import triton.experimental.tle as tle
import triton.language as tl


@triton.jit
def kernel(
    IDs,
    Routes,
    Counts,
    R: tl.constexpr,
    E: tl.constexpr,
    OP: tl.constexpr,
    CMP: tl.constexpr,
    B: tl.constexpr,
):
    lane = tl.arange(0, B)
    source = lane.to(tl.float32)
    for expert in range(tl.program_id(0), E, tl.num_programs(0)):
        count = 0
        for start in range(0, R, B):
            ids = tl.load(IDs + start + lane, start + lane < R, other=-1)
            mask = tl.full((B // 16,), 0, tl.uint16)
            mask = tle.dsa.ascend.raw(
                "compare_scalar", ids.to(tl.float32), expert.to(tl.float32), out=mask
            )
            output = tl.full((B,), 0, tl.float32)
            number = tl.full((8,), 0, tl.int32)
            output, number = tle.dsa.ascend.raw(
                "gather_mask", source, mask, out=[output, number]
            )
            found = tl.sum(tl.where(tl.arange(0, 8) == 0, number, 0), 0)
            if found > 0:
                indices = output.to(tl.int32) + start
                tl.store(Routes + expert * R + count + lane, indices, lane < found)
                count += found
        tl.store(Counts + expert, count)


def routes(ids, output, counts):
    cores = triton.runtime.driver.active.utils.get_device_properties(ids.device.index)[
        "num_vectorcore"
    ]
    grid = min(counts.numel(), cores)
    block = 4096
    kernel[(grid,)](
        ids,
        output,
        counts,
        ids.numel(),
        counts.numel(),
        "gather_mask",
        "compare_scalar",
        block,
        disable_auto_inject_block_sync=True,
        num_warps=1,
    )
