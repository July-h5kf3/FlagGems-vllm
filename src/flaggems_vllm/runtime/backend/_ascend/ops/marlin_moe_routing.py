# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stable small-route sorting for Ascend INT8 Marlin MoE."""

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.experimental.tle.language.dsa.ascend import custom_ops  # noqa: F401


@triton.jit
def sort_small_route_keys(
    IDs,
    SortedRoutes,
    SortedValues,
    E: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    # Unique integer keys below 2**21 preserve expert and original route order.
    tl.static_assert(E <= 512 and B <= 4096)
    lane = tl.arange(0, B)
    ids = tl.load(IDs + lane, lane < R, other=0).to(tl.int32)
    keys = (E - ids.to(tl.float32) - 1) * B + B - 1 - lane.to(tl.float32)
    keys = tl.where(lane < R, keys, float("-inf"))
    scratch = tl.full((B * 4,), 0, tl.float32)
    proposals = tl.full((B * 2,), 0, tl.float32)
    proposals = al.custom("sort_1d_pack", keys, scratch, True, B, 0, 0, out=proposals)
    values = tl.full((B,), 0, tl.float32)
    route_ids = tl.full((B,), 0, tl.int32)
    values, route_ids = al.custom("unpack_sort", proposals, B, out=[values, route_ids])
    tl.store(SortedRoutes + lane, route_ids, lane < R)
    tl.store(SortedValues + lane, values)


@triton.jit
def write_sorted_metadata(
    SortedValues,
    Counts,
    Offsets,
    Starts,
    E: tl.constexpr,
    EP: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    B: tl.constexpr,
    STEPS: tl.constexpr,
):
    values = tl.load(SortedValues + tl.arange(0, B))
    expert_lane = tl.arange(0, EP)
    threshold = (E - expert_lane.to(tl.float32)) * B
    low = tl.full((EP,), 0, tl.float32)
    high = tl.full((EP,), R, tl.float32)
    for step in tl.static_range(STEPS):
        active = low < high
        middle = ((low + high) * 0.5).to(tl.int32).to(tl.float32)
        probe = tl.gather(values, tl.minimum(middle, B - 1).to(tl.int32), 0)
        move_right = probe >= threshold
        low = tl.where(active & move_right, middle + 1, low)
        high = tl.where(active & ~move_right, middle, high)
    following = tl.gather(low, tl.minimum(expert_lane + 1, EP - 1), 0)
    following = tl.where(expert_lane + 1 < E, following, R)
    counts = tl.where(expert_lane < E, following - low, 0.0)
    padded = ((counts + BM - 1) * (1.0 / BM)).to(tl.int32).to(tl.float32) * BM
    ends = padded
    for stage in tl.static_range(9):
        shift = 1 << stage
        if shift < EP:
            prior = tl.gather(ends, tl.maximum(expert_lane - shift, 0), 0)
            ends += tl.where(expert_lane >= shift, prior, 0.0)
    begins = ends - padded
    total = tl.sum(padded, 0).to(tl.int32)
    tl.store(Counts + expert_lane, counts, expert_lane < E)
    tl.store(Offsets + expert_lane, begins, expert_lane < E)
    tl.store(Offsets + E, total)
    tl.store(Starts + expert_lane, low, expert_lane < E)


@triton.jit
def map_sorted_small_routes(
    SortedRoutes,
    Starts,
    Counts,
    Offsets,
    Routes,
    Experts,
    Inv,
    E: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    TILES: tl.constexpr,
):
    lane = tl.arange(0, BM)
    for expert in range(tl.program_id(0), E, tl.num_programs(0)):
        count = tl.load(Counts + expert)
        begin = tl.load(Offsets + expert)
        first = tl.load(Starts + expert)
        for tile in range(tl.cdiv(count, BM)):
            local = tile * BM + lane
            route = tl.load(SortedRoutes + first + local, local < count, other=0)
            tl.store(Routes + expert * R + local, route, local < count)
            tl.store(Inv + route, begin + local, local < count)
            tl.store(Experts + begin // BM + tile, expert)
    if tl.program_id(0) == 0:
        first_unused = tl.load(Offsets + E) // BM
        tail_lane = tl.arange(0, 256)
        for start in range(first_unused, TILES, 256):
            tile = start + tail_lane
            tl.store(Experts + tile, -1, tile < TILES)
