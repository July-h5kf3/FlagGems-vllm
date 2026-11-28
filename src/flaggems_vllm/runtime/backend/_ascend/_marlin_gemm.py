# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Cube/Vector pipelines for group128 Marlin expert projections on Ascend."""

import ctypes
import inspect
from functools import lru_cache
from operator import attrgetter

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.runtime.backend._ascend.device import get_device_core_counts
from flaggems_vllm.runtime.backend._ascend.ops.fused_moe import silu_mul

try:
    import triton.experimental.tle as tle_api
except (AttributeError, ImportError):
    tle_api = None

try:
    import triton.experimental.tle.language.raw as tle_raw
except (AttributeError, ImportError):
    tle_raw = None

tle = tle_api
USE_TLE_RAW_CALL = tl.constexpr(
    hasattr(tle_raw, "call") and "out" in inspect.signature(tle_raw.call).parameters
)

# Stagger wide FP16 workspaces by one 4 KiB page per Cube/Vector pair.
CORE_WORKSPACE_PADDING = 2048  # FP16 elements

TLE_COMMON_FEATURES = (
    "scope",
    "dsa.ascend.sync_block_wait",
    "dsa.ascend.sync_block_set",
    "dsa.ascend.sync_block_all",
    "dsa.ascend.sub_vec_id",
    "dsa.alloc",
    "dsa.copy",
    "dsa.extract_slice",
    "dsa.hint",
    "dsa.subview",
    "dsa.to_buffer",
    "dsa.to_tensor",
)


@lru_cache(maxsize=2)
def check_tle_support(bits: int) -> None:
    if tle_api is None:
        raise NotImplementedError("Ascend fused_marlin_moe requires Triton TLE support")
    features = TLE_COMMON_FEATURES
    if bits == 4:
        features += ("raw.call" if USE_TLE_RAW_CALL.value else "dsa.ascend.raw",)
    if hasattr(tle_api, "is_tle_supported") and not tle_api.is_tle_supported(features):
        raise NotImplementedError(
            f"Ascend Marlin MoE requires TLE primitives {features}"
        )
    for name in (
        *features,
        "dsa.ascend.L1",
        "dsa.ascend.UB",
        "dsa.ascend.PIPE.PIPE_MTE2",
        "dsa.ascend.PIPE.PIPE_MTE3",
    ):
        namespace = tle_raw if name == "raw.call" else tle_api
        try:
            attrgetter("call" if name == "raw.call" else name)(namespace)
        except AttributeError as error:
            raise NotImplementedError(f"Missing TLE primitive {name}") from error


@lru_cache(maxsize=32)
def weight_cache_offset(device_index: int) -> int:
    """Return the device's non-allocating GM alias for streaming packed weights."""
    if not torch.npu.get_device_name(device_index).startswith("Ascend910B"):
        return 0
    try:
        query = ctypes.CDLL("libruntime.so").rtGetL2CacheOffset
    except (OSError, AttributeError):
        return 0
    query.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
    query.restype = ctypes.c_int
    offset = ctypes.c_uint64()
    # CANN maps the user-visible index to the driver device internally.
    with torch.npu.device(device_index):
        status = query(device_index, ctypes.byref(offset))
    return offset.value if status == 0 else 0


@triton.jit
def streaming_weight_pointer(Q, OFFSET: tl.constexpr):
    # The alias reads the same packed bytes without displacing the decoded tiles in L2.
    if OFFSET:
        return (Q.to(tl.uint64) + OFFSET).to(Q.dtype)
    return Q


@triton.jit
def sync_ring_wait(SRC: tl.constexpr, DST: tl.constexpr, FLAG: tl.constexpr):
    if SRC == "vector":
        tle.dsa.ascend.sync_block_wait(
            SRC,
            DST,
            FLAG,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
        )
    else:
        tle.dsa.ascend.sync_block_wait(
            SRC,
            DST,
            FLAG,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
        )


@triton.jit
def sync_ring_set(SRC: tl.constexpr, DST: tl.constexpr, FLAG: tl.constexpr):
    if SRC == "vector":
        tle.dsa.ascend.sync_block_set(
            SRC,
            DST,
            FLAG,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
        )
    else:
        tle.dsa.ascend.sync_block_set(
            SRC,
            DST,
            FLAG,
            sender_pipe=tle.dsa.ascend.PIPE.PIPE_MTE2,
            receiver_pipe=tle.dsa.ascend.PIPE.PIPE_MTE3,
        )


@triton.jit
def load_a_tile(
    A,
    row,
    k_offset,
    K: tl.constexpr,
    BM: tl.constexpr,
    RowEnd=None,
    USE_BLOCK_PTR: tl.constexpr = False,
    RESIDENT: tl.constexpr = False,
):
    if RESIDENT:
        a = tle.dsa.extract_slice(A, [row, k_offset], [BM, 128], [1, 1])
    elif USE_BLOCK_PTR and RowEnd is not None:
        ptr = tl.make_block_ptr(
            A + row * K, (RowEnd - row, K), (K, 1), (0, k_offset), (BM, 128), (1, 0)
        )
        a = tl.load(ptr, boundary_check=(0,), padding_option="zero")
    elif USE_BLOCK_PTR:
        a = tl.load(
            tl.make_block_ptr(
                A + row * K, (BM, K), (K, 1), (0, k_offset), (BM, 128), (1, 0)
            )
        )
    else:
        mi = tl.arange(0, BM)
        ki = tl.arange(0, 128)
        ptr = A + (row + mi[:, None]) * K + k_offset + ki[None, :]
        if RowEnd is None:
            a = tl.load(ptr)
        else:
            a = tl.load(ptr, (row + mi[:, None]) < RowEnd, other=0)
    return a


@triton.jit
def load_cube_weight_stage(
    Work,
    pid,
    iteration,
    BN: tl.constexpr,
    PANEL_N: tl.constexpr = 0,
    STREAM_FIRST: tl.constexpr = False,
    STAGE_N: tl.constexpr = 0,
):
    ni = tl.arange(0, triton.next_power_of_2(BN))
    ki = tl.arange(0, 128)
    if PANEL_N:
        if STREAM_FIRST:
            sync_ring_wait("vector", "cube", 2)
        if BN & (BN - 1):
            ptr = tl.make_block_ptr(
                Work + iteration * PANEL_N * 128,
                (BN, 128),
                (128, 1),
                (0, 0),
                (BN, 128),
                (1, 0),
            )
            b = tl.trans(tl.load(ptr))
        else:
            b = tl.load(
                Work + iteration * PANEL_N * 128 + ni[None, :] * 128 + ki[:, None]
            )
        if STREAM_FIRST:
            sync_ring_set("cube", "vector", 4)
    else:
        EFF_STAGE_N: tl.constexpr = STAGE_N if STAGE_N > 0 else BN
        STAGES: tl.constexpr = (
            2
            if STAGE_N > 0
            else (4 if Work.dtype.element_ty == tl.float16 and BN == 512 else 2)
        )
        tl.static_assert((STAGES & (STAGES - 1)) == 0)
        sync_ring_wait("vector", "cube", 2)
        b = tl.load(
            Work
            + (pid * STAGES + (iteration & (STAGES - 1))) * EFF_STAGE_N * 128
            + ni[None, :] * 128
            + ki[:, None]
        )
        sync_ring_set("cube", "vector", 3)
    return b


@triton.jit
def dot_weight_pairs(
    a_all,
    Work,
    pid,
    iteration,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PANEL_N: tl.constexpr = 0,
    STREAM_FIRST: tl.constexpr = False,
    acc=None,
    row=None,
    RowEnd=None,
    USE_BLOCK_PTR: tl.constexpr = False,
    STAGE_N: tl.constexpr = 0,
):
    RESIDENT: tl.constexpr = row is None
    if RESIDENT:
        row = pid * 0
    if acc is None:
        acc = tl.full((BM, BN), 0, tl.float32)
    for pair in range(K // 256):
        # Fetch both K128 operands before the dependent pair of matrix multiplies.
        left = ()
        right = ()
        for half in tl.static_range(2):
            left += (
                load_a_tile(
                    a_all,
                    row,
                    pair * 256 + half * 128,
                    K,
                    BM,
                    RowEnd,
                    USE_BLOCK_PTR,
                    RESIDENT,
                ),
            )
            right += (
                load_cube_weight_stage(
                    Work, pid, iteration + half, BN, PANEL_N, STREAM_FIRST, STAGE_N
                ),
            )
        for half in tl.static_range(2):
            if PANEL_N and BM >= 96 and left[half].dtype == tl.float16:
                al.compile_hint(left[half], "dot_pad_only_k")
                al.compile_hint(right[half], "dot_pad_only_k")
            acc = tl.dot(left[half], right[half], acc)
        iteration += 2
    if K % 256:
        a0 = load_a_tile(a_all, row, K - 128, K, BM, RowEnd, USE_BLOCK_PTR, RESIDENT)
        b0 = load_cube_weight_stage(
            Work, pid, iteration, BN, PANEL_N, STREAM_FIRST, STAGE_N
        )
        if PANEL_N and BM >= 96 and a0.dtype == tl.float16:
            al.compile_hint(a0, "dot_pad_only_k")
            al.compile_hint(b0, "dot_pad_only_k")
        acc = tl.dot(a0, b0, acc)
        iteration += 1
    return acc, iteration


@triton.jit
def cube_tile(
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
    RowEnd=None,
):
    mi = tl.arange(0, BM)
    ni = tl.arange(0, BN)
    acc = tl.full((BM, BN), 0, tl.float32)
    for kb in range(K // 128):
        a = load_a_tile(A, row, kb * 128, K, BM, RowEnd)
        b = load_cube_weight_stage(Work, pid, iteration, BN)
        acc = tl.dot(a, b, acc)
        iteration += 1
    tl.store(
        Output + (row + mi[:, None]) * N + col + ni[None, :],
        acc.to(Output.dtype.element_ty),
    )
    return iteration


@triton.jit
def cube_tile_pair(
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
    RowEnd=None,
    PANEL_N: tl.constexpr = 0,
    STREAM_FIRST: tl.constexpr = False,
    STAGE_N: tl.constexpr = 0,
    USE_BLOCK_PTR: tl.constexpr = False,
):
    mi = tl.arange(0, triton.next_power_of_2(BM))
    ni = tl.arange(0, triton.next_power_of_2(BN))
    RESIDENT: tl.constexpr = (
        not USE_BLOCK_PTR
        and K == 4096
        and (
            (BN == 512 and (BM == 32 or (BM == 16 and RowEnd is not None)))
            or (PANEL_N > 0 and BN == 256 and (BM == 16 or BM == 32))
        )
        and (A.dtype.element_ty == tl.float16)
    )
    if RESIDENT:
        all_k = tl.arange(0, K)
        if RowEnd is None:
            a_all = tl.load(A + (row + mi[:, None]) * K + all_k[None, :])
        else:
            a_all = tl.load(
                A + (row + mi[:, None]) * K + all_k[None, :],
                row + mi[:, None] < RowEnd,
                other=0,
            )
        acc, iteration = dot_weight_pairs(
            a_all, Work, pid, iteration, K, BM, BN, PANEL_N, STREAM_FIRST
        )
    elif (
        A.dtype.element_ty == tl.float16
        and BM == 64
        and BN == 512
        and not PANEL_N
        and not STAGE_N
        and K % 1024 == 0
        and N % 1024 == 0
    ):
        acc = tl.full((BM, BN), 0, tl.float32)
        for slab in range(K // 1024):
            kk = slab * 1024 + tl.arange(0, 1024)
            ptr = A + (row + mi[:, None]) * K + kk[None, :]
            if RowEnd is None:
                a_slab = tl.load(ptr)
            else:
                a_slab = tl.load(ptr, row + mi[:, None] < RowEnd, other=0)
            acc, iteration = dot_weight_pairs(
                a_slab, Work, pid, iteration, 1024, BM, BN, acc=acc
            )
    else:
        acc, iteration = dot_weight_pairs(
            A,
            Work,
            pid,
            iteration,
            K,
            BM,
            BN,
            PANEL_N,
            STREAM_FIRST,
            row=row,
            RowEnd=RowEnd,
            USE_BLOCK_PTR=USE_BLOCK_PTR,
            STAGE_N=STAGE_N,
        )
    if USE_BLOCK_PTR and RowEnd is not None:
        cp = tl.make_block_ptr(
            Output + row * N + col, (RowEnd - row, BN), (N, 1), (0, 0), (BM, BN), (1, 0)
        )
        tl.store(cp, acc.to(Output.dtype.element_ty), boundary_check=(0,))
    elif USE_BLOCK_PTR:
        cp = tl.make_block_ptr(
            Output + row * N + col, (BM, BN), (N, 1), (0, 0), (BM, BN), (1, 0)
        )
        tl.store(cp, acc.to(Output.dtype.element_ty))
    else:
        tl.store(
            Output + (row + mi[:, None]) * N + col + ni[None, :],
            acc.to(Output.dtype.element_ty),
        )
    return iteration


@triton.jit
def direct_cube_tile_pair(
    A,
    Routes,
    Counts,
    Offsets,
    Work,
    Output,
    expert,
    row,
    col,
    pid,
    iteration,
    K: tl.constexpr,
    N: tl.constexpr,
    TopK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    mi = tl.arange(0, BM)
    ni = tl.arange(0, BN)
    begin = tl.load(Offsets + expert)
    count = tl.load(Counts + expert)
    a_buffer = tle.dsa.alloc((BM, K), A.dtype.element_ty, tle.dsa.ascend.L1)
    for local_row in tl.static_range(BM):
        route_row = row + local_row - begin
        if route_row < count:
            token = tl.load(Routes + row + local_row) // TopK
            source = tl.make_block_ptr(
                A + token * K,
                (1, K),
                (K, 1),
                (0, 0),
                (1, K),
                (1, 0),
            )
            destination = tle.dsa.subview(a_buffer, [local_row, 0], [1, K], [1, 1])
            tle.dsa.copy(source, destination, [1, K])
    a_all = tle.dsa.to_tensor(a_buffer, writable=False)
    acc, iteration = dot_weight_pairs(a_all, Work, pid, iteration, K, BM, BN)
    tl.store(
        Output + (row + mi[:, None]) * N + col + ni[None, :],
        acc.to(Output.dtype.element_ty),
    )
    return iteration


@triton.jit
def direct_cube_gemm(
    A,
    Routes,
    Counts,
    Offsets,
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
    TopK: tl.constexpr,
):
    iteration = 0
    for task in range(PID, TASKS, GRID):
        tile_id = task // (N // BN)
        col = task % (N // BN) * BN
        expert = tl.load(Experts + tile_id)
        if expert >= 0:
            iteration = direct_cube_tile_pair(
                A,
                Routes,
                Counts,
                Offsets,
                Work,
                Output,
                expert,
                tile_id * BM,
                col,
                PID,
                iteration,
                K,
                N,
                TopK,
                BM,
                BN,
            )


@triton.jit
def tail_task_info(Experts, Counts, Offsets, task, N: tl.constexpr):
    tile = task // (N // 128)
    panel = task % (N // 128)
    expert = tl.load(Experts + tile)
    active = expert >= 0
    paired = False
    remaining = 0
    if active:
        begin = tl.load(Offsets + expert)
        local_tile = tile - begin // 128
        active = local_tile % 2 == 0
        if active:
            remaining = tl.load(Counts + expert) - local_tile * 128
            paired = remaining > 128
    return active, paired, expert, tile, panel, remaining


@triton.jit
def cube_adaptive_gemm(
    A,
    Work,
    Experts,
    Output,
    Counts,
    Offsets,
    PID,
    N: tl.constexpr,
    K: tl.constexpr,
    TASKS: tl.constexpr,
    GRID: tl.constexpr,
):
    iteration = 0
    for task in range(PID, TASKS, GRID):
        active, paired, expert, tile, panel, remaining = tail_task_info(
            Experts, Counts, Offsets, task, N
        )
        if active:
            if paired:
                iteration = cube_tile_pair(
                    A,
                    Work,
                    Output,
                    tile * 128,
                    panel * 128,
                    PID,
                    iteration,
                    K,
                    N,
                    256,
                    128,
                    STAGE_N=512,
                    USE_BLOCK_PTR=True,
                )
            elif remaining <= 64:
                if panel < N // 512:
                    iteration = cube_tile_pair(
                        A,
                        Work,
                        Output,
                        tile * 128,
                        panel * 512,
                        PID,
                        iteration,
                        K,
                        N,
                        64,
                        512,
                        STAGE_N=512,
                        USE_BLOCK_PTR=True,
                    )
            else:
                if panel < N // 256:
                    iteration = cube_tile_pair(
                        A,
                        Work,
                        Output,
                        tile * 128,
                        panel * 256,
                        PID,
                        iteration,
                        K,
                        N,
                        128,
                        256,
                        STAGE_N=512,
                        USE_BLOCK_PTR=True,
                    )


@triton.jit
def copy_decode_input(Source, Dest, row, COLS: tl.constexpr):
    ptr = tl.make_block_ptr(
        Source + row * COLS, (128, COLS), (COLS, 1), (0, 0), (128, COLS), (1, 0)
    )
    tle.dsa.copy(ptr, Dest, [128, COLS])


@triton.jit
def decode_ub_tile(q, s):
    signed = tle.dsa.to_tensor(q, writable=False).to(tl.int8, bitcast=True)
    scale = tle.dsa.to_tensor(s, writable=False)
    return signed.to(tl.float16) * scale


@triton.jit
def prefetch_weight_groups(
    Q,
    S,
    Work,
    expert,
    col,
    pid,
    sub,
    iteration,
    N: tl.constexpr,
    K: tl.constexpr,
    groups=None,
):
    if groups is None:
        groups = K // 128
    q0 = tle.dsa.alloc((128, 128), tl.uint8, tle.dsa.ascend.UB)
    q1 = tle.dsa.alloc((128, 128), tl.uint8, tle.dsa.ascend.UB)
    s0 = tle.dsa.alloc((128, 1), tl.float16, tle.dsa.ascend.UB)
    s1 = tle.dsa.alloc((128, 1), tl.float16, tle.dsa.ascend.UB)
    # Prefetch the next half tile while converting the current signed INT8 rows.
    first = expert.to(tl.int64) * (K // 128) * N + col + sub * 256
    ix = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    copy_decode_input(Q, q0, first, 128)
    copy_decode_input(S, s0, first, 1)
    for kb in range(groups):
        row = first + kb * N
        copy_decode_input(Q, q1, row + 128, 128)
        copy_decode_input(S, s1, row + 128, 1)
        h0 = decode_ub_tile(q0, s0)
        if iteration >= 4:
            sync_ring_wait("cube", "vector", 3)
        dest = Work + ((pid * 4 + (iteration & 3)) * 512 + sub * 256) * 128
        tl.store(dest + ix, h0)
        if kb + 1 < groups:
            copy_decode_input(Q, q0, row + N, 128)
            copy_decode_input(S, s0, row + N, 1)
        h1 = decode_ub_tile(q1, s1)
        tl.store(dest + 128 * 128 + ix, h1)
        sync_ring_set("vector", "cube", 2)
        iteration += 1
    return iteration


@triton.jit
def dequantize_k_stages(
    Q,
    S,
    Work,
    expert,
    col,
    PID,
    SUB,
    iteration,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BITS: tl.constexpr,
    STAGE_N: tl.constexpr = 0,
    groups=None,
):
    if (
        BITS == 8
        and Work.dtype.element_ty == tl.float16
        and BN == 512
        and STAGE_N == 0
        and (K % 1024 or N % 1024)
    ):
        return prefetch_weight_groups(
            Q, S, Work, expert, col, PID, SUB, iteration, N, K, groups
        )
    EFF_STAGE_N: tl.constexpr = STAGE_N if STAGE_N > 0 else BN
    STAGES: tl.constexpr = (
        2
        if STAGE_N > 0
        else (4 if Work.dtype.element_ty == tl.float16 and BN == 512 else 2)
    )
    tl.static_assert((STAGES & (STAGES - 1)) == 0)
    VBN: tl.constexpr = BN // 2
    MAX_ROWS: tl.constexpr = (
        256 if BITS == 8 and Work.dtype.element_ty == tl.float16 else 64
    )
    CB: tl.constexpr = VBN if VBN <= MAX_ROWS else MAX_ROWS
    ns = tl.arange(0, CB)
    ks = tl.arange(0, 128)
    if groups is None:
        groups = K // 128
    for kb in range(groups):
        for chunk in range(VBN // CB):
            result = load_weight_tile(
                Q,
                S,
                expert,
                col + SUB * VBN + chunk * CB,
                kb,
                N,
                K,
                CB,
                BITS,
            )
            # Wait once before overwriting either half of the GM stage.
            if iteration >= STAGES and chunk == 0:
                sync_ring_wait("cube", "vector", 3)
            offset = (
                (PID * STAGES + (iteration & (STAGES - 1))) * EFF_STAGE_N * 128
                + SUB * VBN * 128
                + chunk * CB * 128
            )
            tl.store(Work + offset + ns[:, None] * 128 + ks[None, :], result)
        # Notify Cube only after all subtiles have been stored.
        sync_ring_set("vector", "cube", 2)
        iteration += 1
    return iteration


@triton.jit
def drain_weight_ring(iteration, STAGES: tl.constexpr):
    for _ in range(tl.minimum(iteration, STAGES)):
        sync_ring_wait("cube", "vector", 3)


@triton.jit
def dequantize_adaptive_weights(
    Q,
    S,
    Experts,
    Counts,
    Offsets,
    Work,
    PID,
    SUB,
    N: tl.constexpr,
    K: tl.constexpr,
    TASKS: tl.constexpr,
    GRID: tl.constexpr,
):
    iteration = 0
    for task in range(PID, TASKS, GRID):
        active, paired, expert, tile, panel, remaining = tail_task_info(
            Experts, Counts, Offsets, task, N
        )
        if active:
            if paired:
                iteration = dequantize_k_stages(
                    Q,
                    S,
                    Work,
                    expert,
                    panel * 128,
                    PID,
                    SUB,
                    iteration,
                    N,
                    K,
                    128,
                    8,
                    512,
                )
            elif remaining <= 64:
                if panel < N // 512:
                    iteration = dequantize_k_stages(
                        Q,
                        S,
                        Work,
                        expert,
                        panel * 512,
                        PID,
                        SUB,
                        iteration,
                        N,
                        K,
                        512,
                        8,
                        512,
                    )
            else:
                if panel < N // 256:
                    iteration = dequantize_k_stages(
                        Q,
                        S,
                        Work,
                        expert,
                        panel * 256,
                        PID,
                        SUB,
                        iteration,
                        N,
                        K,
                        256,
                        8,
                        512,
                    )
    drain_weight_ring(iteration, 2)


@triton.jit
def cube_gemm(
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
    TILE: tl.constexpr = cube_tile,
    Counts=None,
    Offsets=None,
):
    iteration = 0
    for task in range(PID, TASKS, GRID):
        tile_id = task // (N // BN)
        col = task % (N // BN) * BN
        expert = tl.load(Experts + tile_id)
        active = expert >= 0
        if MERGE:
            begin = tl.load(Offsets + tl.maximum(expert, 0))
            active = active & ((tile_id - begin // 128) % 2 == 0)
        if active:
            if Counts is None:
                row_end = None
            else:
                row_end = tl.load(Offsets + expert) + tl.load(Counts + expert)
            if MERGE:
                next_expert = -2
                if tile_id + 1 < TASKS // (N // BN):
                    next_expert = tl.load(Experts + tile_id + 1)
                if next_expert == expert:
                    iteration = TILE(
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
                        row_end,
                    )
                else:
                    iteration = TILE(
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
                        row_end,
                    )
            else:
                iteration = TILE(
                    A,
                    Work,
                    Output,
                    tile_id * BM,
                    col,
                    PID,
                    iteration,
                    K,
                    N,
                    BM,
                    BN,
                    row_end,
                )


@triton.jit
def load_weight_tile(
    Q,
    S,
    expert,
    col,
    kb,
    N: tl.constexpr,
    K: tl.constexpr,
    RB: tl.constexpr,
    BITS: tl.constexpr,
):
    group_bytes: tl.constexpr = 128 // (8 // BITS)
    groups: tl.constexpr = K // 128
    base = (expert.to(tl.int64) * groups + kb) * N + col
    if BITS == 8 and S.dtype.element_ty == tl.float16 and RB == 256:
        # Keep the expert-bank base separate from the 2-D DMA tile on CANN 9.1.
        rows = tl.arange(0, RB)
        columns = tl.arange(0, 128)
        packed = tl.load(Q + base * 128 + rows[:, None] * 128 + columns[None, :])
        scale = tl.load(S + base + rows)
        return packed.to(tl.int8, bitcast=True).to(tl.float16) * scale[:, None]
    elif RB & (RB - 1):
        tl.static_assert(BITS == 8)
        p = tl.make_block_ptr(
            Q + base * 128, (RB, 128), (128, 1), (0, 0), (RB, 128), (1, 0)
        )
        sp = tl.make_block_ptr(S + base, (RB, 1), (1, 1), (0, 0), (RB, 1), (1, 0))
        return tl.load(p).to(tl.int8, bitcast=True).to(tl.float16) * tl.load(sp)
    q = tl.load(Q + base * group_bytes + tl.arange(0, RB * group_bytes))
    scales = tl.load(S + base + tl.arange(0, RB))
    if BITS == 4:
        h = tl.full((RB * 128,), 0, tl.float16)
        if USE_TLE_RAW_CALL:
            h = tle_raw.call("ascend_cast_int4_to_fp16", q, 0, RB * 128, out=h)
        else:
            # FlagTree 0.7 exposes registered Ascend bitcode only through this ABI.
            h = tle.dsa.ascend.raw("cast_int4_to_fp16", q, 0, RB * 128, out=h)
    else:
        h = q.to(tl.int8, bitcast=True).to(tl.float16)
    values_h = tl.reshape(h, (RB, 128))
    if scales.dtype == tl.float16:
        result = values_h * tl.reshape(scales, (RB, 1))
    else:
        result = (values_h.to(tl.float32) * scales.to(tl.float32)[:, None]).to(
            tl.bfloat16
        )
    return result


@triton.jit
def dequantize_weights(
    Q,
    S,
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
    BITS: tl.constexpr,
    Offsets=None,
):
    STAGES: tl.constexpr = 4 if Work.dtype.element_ty == tl.float16 and BN == 512 else 2
    iteration = 0
    for task in range(PID, TASKS, GRID):
        tile = task // (N // BN)
        col = task % (N // BN) * BN
        expert = tl.load(Experts + tile)
        active = expert >= 0
        if MERGE:
            begin = tl.load(Offsets + tl.maximum(expert, 0))
            active = active & ((tile - begin // 128) % 2 == 0)
        if active:
            iteration = dequantize_k_stages(
                Q, S, Work, expert, col, PID, SUB, iteration, N, K, BN, BITS
            )
    drain_weight_ring(iteration, STAGES)


@triton.jit
def gemm_kernel(
    A,
    Q,
    S,
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
    BITS: tl.constexpr,
    Counts=None,
    Offsets=None,
    TAIL_TILES: tl.constexpr = False,
    Routes=None,
    TopK: tl.constexpr = 1,
    WEIGHT_OFFSET: tl.constexpr = 0,
    CORE_PADDING: tl.constexpr = 0,
):
    Work = Work + tl.program_id(0) * CORE_PADDING
    with tle.scope(core_mode="cube"):
        if TAIL_TILES:
            cube_adaptive_gemm(
                A,
                Work,
                Experts,
                Output,
                Counts,
                Offsets,
                tl.program_id(0),
                N,
                K,
                TASKS,
                GRID,
            )
        elif Routes is not None:
            direct_cube_gemm(
                A,
                Routes,
                Counts,
                Offsets,
                Work,
                Experts,
                Output,
                tl.program_id(0),
                N,
                K,
                BM,
                BN,
                TASKS,
                GRID,
                TopK,
            )
        elif BITS == 8:
            cube_gemm(
                A,
                Work,
                Experts,
                Output,
                tl.program_id(0),
                N,
                K,
                BM,
                BN,
                TASKS,
                GRID,
                MERGE,
                cube_tile_pair,
                Counts,
                Offsets,
            )
        else:
            cube_gemm(
                A,
                Work,
                Experts,
                Output,
                tl.program_id(0),
                N,
                K,
                BM,
                BN,
                TASKS,
                GRID,
                MERGE,
                Offsets=Offsets,
            )
    with tle.scope(core_mode="vector"):
        Q = streaming_weight_pointer(Q, WEIGHT_OFFSET)
        S = streaming_weight_pointer(S, WEIGHT_OFFSET)
        if TAIL_TILES:
            dequantize_adaptive_weights(
                Q,
                S,
                Experts,
                Counts,
                Offsets,
                Work,
                tl.program_id(0),
                tle.dsa.ascend.sub_vec_id(),
                N,
                K,
                TASKS,
                GRID,
            )
        else:
            dequantize_weights(
                Q,
                S,
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
                BITS,
                Offsets,
            )


@triton.jit
def compact_panel_tile(
    A,
    panel,
    output,
    row,
    col,
    pid,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    FIRST: tl.constexpr = False,
    RowEnd=None,
):
    cube_tile_pair(
        A,
        panel,
        output,
        row,
        col,
        pid,
        pid * 0,
        K,
        N,
        BM,
        BN,
        RowEnd,
        BN,
        FIRST,
        USE_BLOCK_PTR=RowEnd is not None,
    )


@triton.jit
def decode_expert_panel(
    Q,
    S,
    Work,
    vector_expert,
    vector_col,
    pid,
    sub,
    vector_iteration,
    WEIGHT_N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    STAGES: tl.constexpr,
    FULL_PANEL: tl.constexpr,
):
    VBN: tl.constexpr = BN // 2
    ns = tl.arange(0, triton.next_power_of_2(VBN))
    ks = tl.arange(0, 128)
    if vector_iteration >= STAGES:
        if FULL_PANEL and vector_iteration % 2 != 0:
            sync_ring_wait("cube", "vector", 7)
        else:
            sync_ring_wait("cube", "vector", 3)
    vector_stage = (pid * STAGES + vector_iteration % STAGES) * BN * K
    for kb in range(K // 128):
        decoded = load_weight_tile(
            Q,
            S,
            vector_expert,
            vector_col,
            kb,
            WEIGHT_N,
            K,
            VBN,
            8,
        )
        if not FULL_PANEL and kb >= 2:
            sync_ring_wait("cube", "vector", 4)
        if VBN & (VBN - 1):
            dst = tl.make_block_ptr(
                Work + vector_stage + (kb * BN + sub * VBN) * 128,
                (VBN, 128),
                (128, 1),
                (0, 0),
                (VBN, 128),
                (1, 0),
            )
            tl.store(dst, decoded)
        else:
            tl.store(
                Work
                + vector_stage
                + kb * BN * 128
                + (sub * VBN + ns[:, None]) * 128
                + ks[None, :],
                decoded,
            )
        if not FULL_PANEL:
            sync_ring_set("vector", "cube", 2)
    if FULL_PANEL:
        if vector_iteration % 2 == 0:
            sync_ring_set("vector", "cube", 2)
        else:
            sync_ring_set("vector", "cube", 4)
    else:
        for _ in range(tl.minimum(K // 128, 2)):
            sync_ring_wait("cube", "vector", 4)


@triton.jit
def project_panel_rows(
    A,
    panel,
    output,
    row,
    col,
    pid,
    count,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    FIRST: tl.constexpr,
    TAIL16: tl.constexpr,
):
    compact_panel_tile(A, panel, output, row, col, pid, K, N, BM, BN, FIRST)
    for start in range(BM, count, BM):
        remaining = count - start
        if TAIL16 and remaining <= 16:
            compact_panel_tile(A, panel, output, row + start, col, pid, K, N, 16, BN)
        elif remaining <= 32:
            compact_panel_tile(A, panel, output, row + start, col, pid, K, N, 32, BN)
        elif remaining <= 64:
            compact_panel_tile(A, panel, output, row + start, col, pid, K, N, 64, BN)
        else:
            compact_panel_tile(A, panel, output, row + start, col, pid, K, N, BM, BN)


@triton.jit
def expert_panel_kernel(
    A,
    Q,
    S,
    Counts,
    Offsets,
    Work,
    Output,
    K: tl.constexpr,
    N: tl.constexpr,
    WEIGHT_N: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GRID: tl.constexpr,
    STAGES: tl.constexpr = 1,
    EXPERT_OUTER: tl.constexpr = False,
    GateCache=None,
    PANEL_ROWS: tl.constexpr = 0,
    PANEL_STRIDE: tl.constexpr = 0,
    WEIGHT_OFFSET: tl.constexpr = 0,
    FULL_PANEL: tl.constexpr = False,
    CORE_PADDING: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    COLS: tl.constexpr = N // BN
    Work = Work + tl.program_id(0) * CORE_PADDING
    with tle.scope(core_mode="cube"):
        cube_iteration = 0
        if GateCache is not None:
            # A panel contains matching gate/up columns. Vector consumes one
            # projection tile while Cube writes the next into the other slot.
            projection_tile = 0
            for task in range(pid, E * COLS, GRID):
                expert = task // COLS
                count = tl.load(Counts + expert)
                if count > 0:
                    begin = tl.load(Offsets + expert)
                    panel = Work + pid * BN * K
                    for start in range(0, count, 128):
                        if projection_tile >= 2:
                            tle.dsa.ascend.sync_block_wait(
                                "vector",
                                "cube",
                                6,
                                sender_pipe=al.PIPE.PIPE_MTE2,
                                receiver_pipe=al.PIPE.PIPE_FIX,
                            )
                        cache_tile = (
                            GateCache + (pid * 2 + (projection_tile & 1)) * 128 * BN
                        )
                        inputs = A + (begin + start) * K
                        if start == 0:
                            cube_tile_pair(
                                inputs,
                                panel,
                                cache_tile,
                                pid * 0,
                                0,
                                pid,
                                pid * 0,
                                K,
                                BN,
                                128,
                                BN,
                                count - start,
                                BN,
                                True,
                            )
                        else:
                            cube_tile_pair(
                                inputs,
                                panel,
                                cache_tile,
                                pid * 0,
                                0,
                                pid,
                                pid * 0,
                                K,
                                BN,
                                128,
                                BN,
                                count - start,
                                BN,
                                False,
                            )
                        tle.dsa.ascend.sync_block_set(
                            "cube",
                            "vector",
                            5,
                            sender_pipe=al.PIPE.PIPE_FIX,
                            receiver_pipe=al.PIPE.PIPE_MTE2,
                        )
                        projection_tile += 1
                    sync_ring_set("cube", "vector", 3)
            for _ in range(tl.minimum(projection_tile, 2)):
                tle.dsa.ascend.sync_block_wait(
                    "vector",
                    "cube",
                    6,
                    sender_pipe=al.PIPE.PIPE_MTE2,
                    receiver_pipe=al.PIPE.PIPE_FIX,
                )
        elif PANEL_ROWS:
            for task in range(pid, E * COLS, GRID):
                expert = task // COLS
                count = tl.load(Counts + expert)
                if count > 0:
                    row = tl.load(Offsets + expert)
                    col = task % COLS * BN
                    panel = Work + pid * K * BN
                    end = row + count
                    compact_panel_tile(
                        A,
                        panel,
                        Output,
                        row,
                        col,
                        pid,
                        K,
                        PANEL_STRIDE,
                        PANEL_ROWS,
                        BN,
                        True,
                        RowEnd=end,
                    )
                    for start in range(PANEL_ROWS, count, 64):
                        compact_panel_tile(
                            A,
                            panel,
                            Output,
                            row + start,
                            col,
                            pid,
                            K,
                            PANEL_STRIDE,
                            64,
                            BN,
                            RowEnd=end,
                        )
                    sync_ring_set("cube", "vector", 3)
        elif BM == 32:
            # Consume only this expert's 32-row-aligned range.
            for task in range(pid, E * COLS, GRID):
                expert = task // COLS
                col = task % COLS * BN
                count = tl.load(Counts + expert)
                if count > 0:
                    row = tl.load(Offsets + expert)
                    remaining = tl.cdiv(count, 32) * 32
                    panel = Work + (pid * STAGES + cube_iteration % STAGES) * BN * K
                    if remaining >= 128:
                        first_rows = 128
                        compact_panel_tile(
                            A, panel, Output, row, col, pid, K, N, 128, BN, True
                        )
                    elif remaining >= 64:
                        first_rows = 64
                        compact_panel_tile(
                            A, panel, Output, row, col, pid, K, N, 64, BN, True
                        )
                    else:
                        first_rows = 32
                        compact_panel_tile(
                            A, panel, Output, row, col, pid, K, N, 32, BN, True
                        )
                    row += first_rows
                    remaining -= first_rows
                    while remaining >= 128:
                        compact_panel_tile(
                            A, panel, Output, row, col, pid, K, N, 128, BN
                        )
                        row += 128
                        remaining -= 128
                    if remaining >= 64:
                        compact_panel_tile(
                            A, panel, Output, row, col, pid, K, N, 64, BN
                        )
                        row += 64
                        remaining -= 64
                    if remaining:
                        compact_panel_tile(
                            A, panel, Output, row, col, pid, K, N, 32, BN
                        )
                    sync_ring_set("cube", "vector", 3)
                    cube_iteration += 1
        elif EXPERT_OUTER:
            for cube_expert in range(pid, E, GRID):
                cube_count = tl.load(Counts + cube_expert)
                if cube_count > 0:
                    cube_begin = tl.load(Offsets + cube_expert)
                    for col_idx in range(COLS):
                        cube_col = col_idx * BN
                        cube_stage = (pid * STAGES + cube_iteration % STAGES) * BN * K
                        project_panel_rows(
                            A,
                            Work + cube_stage,
                            Output,
                            cube_begin,
                            cube_col,
                            pid,
                            cube_count,
                            K,
                            N,
                            BM,
                            BN,
                            True,
                            False,
                        )
                        sync_ring_set("cube", "vector", 3)
                        cube_iteration += 1
        else:
            for cube_task in range(pid, E * COLS, GRID):
                cube_expert = cube_task // COLS
                cube_col = cube_task % COLS * BN
                cube_count = tl.load(Counts + cube_expert)
                if cube_count > 0:
                    cube_begin = tl.load(Offsets + cube_expert)
                    if FULL_PANEL:
                        if cube_iteration % 2 == 0:
                            sync_ring_wait("vector", "cube", 2)
                        else:
                            sync_ring_wait("vector", "cube", 4)
                    cube_stage = (pid * STAGES + cube_iteration % STAGES) * BN * K
                    project_panel_rows(
                        A,
                        Work + cube_stage,
                        Output,
                        cube_begin,
                        cube_col,
                        pid,
                        cube_count,
                        K,
                        N,
                        BM,
                        BN,
                        not FULL_PANEL,
                        True,
                    )
                    if FULL_PANEL and cube_iteration % 2 != 0:
                        sync_ring_set("cube", "vector", 7)
                    else:
                        sync_ring_set("cube", "vector", 3)
                    cube_iteration += 1
    with tle.scope(core_mode="vector"):
        Q = streaming_weight_pointer(Q, WEIGHT_OFFSET)
        S = streaming_weight_pointer(S, WEIGHT_OFFSET)
        sub = tle.dsa.ascend.sub_vec_id()
        VBN: tl.constexpr = BN // 2
        ns = tl.arange(0, triton.next_power_of_2(VBN))
        vector_iteration = 0
        if EXPERT_OUTER:
            for vector_expert in range(pid, E, GRID):
                vector_count = tl.load(Counts + vector_expert)
                if vector_count > 0:
                    for col_idx in range(COLS):
                        vector_col = col_idx * BN + sub * VBN
                        decode_expert_panel(
                            Q,
                            S,
                            Work,
                            vector_expert,
                            vector_col,
                            pid,
                            sub,
                            vector_iteration,
                            WEIGHT_N,
                            K,
                            BN,
                            STAGES,
                            False,
                        )
                        vector_iteration += 1
        else:
            projection_tile = 0
            for vector_task in range(pid, E * COLS, GRID):
                vector_expert = vector_task // COLS
                if GateCache is not None:
                    vector_col = vector_task % COLS * VBN + sub * (N // 2)
                else:
                    vector_col = vector_task % COLS * BN + sub * VBN
                vector_count = tl.load(Counts + vector_expert)
                if vector_count > 0:
                    decode_expert_panel(
                        Q,
                        S,
                        Work,
                        vector_expert,
                        vector_col,
                        pid,
                        sub,
                        vector_iteration,
                        WEIGHT_N,
                        K,
                        BN,
                        STAGES,
                        FULL_PANEL,
                    )
                    vector_iteration += 1
                    if GateCache is not None:
                        begin = tl.load(Offsets + vector_expert)
                        col = vector_task % COLS * VBN
                        rows = sub * 64 + tl.arange(0, 64)
                        tile_offsets = rows[:, None] * BN + ns[None, :]
                        for start in range(0, vector_count, 128):
                            tle.dsa.ascend.sync_block_wait(
                                "cube",
                                "vector",
                                5,
                                sender_pipe=al.PIPE.PIPE_FIX,
                                receiver_pipe=al.PIPE.PIPE_MTE2,
                            )
                            cache_tile = (
                                GateCache + (pid * 2 + (projection_tile & 1)) * 128 * BN
                            )
                            gate = tl.load(cache_tile + tile_offsets).to(tl.float32)
                            up = tl.load(cache_tile + tile_offsets + VBN).to(tl.float32)
                            activated = silu_mul(gate, up)
                            tle.dsa.ascend.sync_block_set(
                                "vector",
                                "cube",
                                6,
                                sender_pipe=al.PIPE.PIPE_MTE2,
                                receiver_pipe=al.PIPE.PIPE_FIX,
                            )
                            tl.store(
                                Output
                                + (begin + start + rows[:, None]) * (N // 2)
                                + col
                                + ns[None, :],
                                activated.to(tl.float16),
                                start + rows[:, None] < vector_count,
                            )
                            projection_tile += 1
        if FULL_PANEL:
            for turn in range(
                tl.maximum(vector_iteration - STAGES, 0), vector_iteration
            ):
                if turn % 2 == 0:
                    sync_ring_wait("cube", "vector", 3)
                else:
                    sync_ring_wait("cube", "vector", 7)
        else:
            drain_weight_ring(vector_iteration, STAGES)


@triton.jit
def decode_expert_matrix(
    Q,
    S,
    B,
    Counts,
    expert,
    K: tl.constexpr,
    N: tl.constexpr,
    WEIGHT_OFFSET: tl.constexpr = 0,
):
    Q = streaming_weight_pointer(Q, WEIGHT_OFFSET)
    S = streaming_weight_pointer(S, WEIGHT_OFFSET)
    if tl.load(Counts + expert) > 0:
        index = tl.arange(0, 128)
        for job in range(tl.program_id(0), (K // 128) * (N // 128), tl.num_programs(0)):
            group = job // (N // 128)
            col = job % (N // 128) * 128
            weight = load_weight_tile(
                Q, S, tl.cast(expert, tl.int64), col, group, N, K, 128, 8
            )
            tl.store(
                B + (group * 128 + index[:, None]) * N + col + index[None, :],
                tl.trans(weight),
            )


@triton.jit
def expert_matrix_tile(
    A, B, Output, row, col, valid, K: tl.constexpr, N: tl.constexpr, FULL: tl.constexpr
):
    rows = tl.arange(0, 128)
    columns = tl.arange(0, 256)
    reduction = tl.arange(0, 256)
    acc = tl.full((128, 256), 0, tl.float32)
    for block in tl.range(K // 256):
        start = block * 256
        ap = A + (row + rows[:, None]) * K + start + reduction[None, :]
        if FULL:
            a = tl.load(ap)
        else:
            tail = tl.make_block_ptr(
                A + row * K, (valid, K), (K, 1), (0, start), (128, 256), (1, 0)
            )
            a = tl.load(tail, boundary_check=(0,), padding_option="zero")
        b = tl.load(B + (start + reduction[:, None]) * N + col + columns[None, :])
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc)
    cp = Output + (row + rows[:, None]) * N + col + columns[None, :]
    if FULL:
        tl.store(cp, acc.to(tl.float16))
    else:
        tail = tl.make_block_ptr(
            Output + row * N, (valid, N), (N, 1), (0, col), (128, 256), (1, 0)
        )
        tl.store(tail, acc.to(tl.float16), boundary_check=(0,))


@triton.jit
def project_expert_matrix(
    A, B, Output, Counts, Offsets, expert, K: tl.constexpr, N: tl.constexpr
):
    with tle.scope(core_mode="cube"):
        count = tl.load(Counts + expert)
        start = tl.load(Offsets + expert)
        tiles_m = tl.cdiv(count, 128)
        tiles_n: tl.constexpr = N // 256
        # Adjacent Cube programs reuse the same columns of this decoded expert.
        group_m: tl.constexpr = 16
        for job in range(tl.program_id(0), tiles_m * tiles_n, tl.num_programs(0)):
            first = job // (group_m * tiles_n) * group_m
            height = tl.minimum(group_m, tiles_m - first)
            local = job % (group_m * tiles_n)
            m = first + local % height
            col = local // height * 256
            remaining = count - m * 128
            if remaining >= 128:
                expert_matrix_tile(A, B, Output, start + m * 128, col, 128, K, N, True)
            else:
                expert_matrix_tile(
                    A, B, Output, start + m * 128, col, remaining, K, N, False
                )


def grouped_quantized_gemm(
    a,
    w,
    s,
    experts,
    out,
    bm,
    bits=4,
    counts=None,
    offsets=None,
    tail_tiles=False,
    expert_panel=False,
    routes=None,
    topk=1,
    fuse_swiglu=False,
    wide_input=False,
):
    n, k = out.shape[1], a.shape[1]
    cores = get_device_core_counts(a.device)[0]
    # Each group128 weight tile is read once per projected row block.
    weight_offset = (
        weight_cache_offset(a.device.index)
        if bits == 8
        and a.dtype == torch.float16
        and k % 1024 == 0
        and s.stride(1) % 1024 == 0
        else 0
    )
    core_padding = CORE_WORKSPACE_PADDING if weight_offset else 0
    if expert_panel:
        if fuse_swiglu:
            n *= 2
        panel_bn = 224 if wide_input else 256
        expert_outer = k <= 512 and bm != 32
        # With 128-row tiles, double-buffer the complete decoded panels.
        # Distinct ready/release flags let Vector prepare the next expert tile.
        full_panel = (
            bm == 128
            and k % 1024 == 0
            and n % 1024 == 0
            and w.shape[0] >= cores
            and not wide_input
            and not fuse_swiglu
        )
        panel_stages = 2 if expert_outer or full_panel else 1
        panel_grid = min(
            w.shape[0] if expert_outer else w.shape[0] * (n // panel_bn), cores
        )
        # For a long reduction with many rows per expert, the existing panel
        # budget can hold one complete decoded matrix. Reuse it across row tiles.
        if (
            bm == 32
            and not fuse_swiglu
            and k >= 8192
            and k % 256 == 0
            and n % 256 == 0
            and n == s.stride(1)
            and w.shape[0] < cores
            and a.shape[0] >= 512 * w.shape[0]
            and n <= panel_grid * panel_bn
            and bits == 8
            and a.dtype == torch.float16
        ):
            matrix = torch.empty((k, n), device=a.device, dtype=a.dtype)
            for expert in range(w.shape[0]):
                decode_expert_matrix[(2 * cores,)](
                    w,
                    s,
                    matrix,
                    counts,
                    expert,
                    k,
                    n,
                    num_warps=1,
                    multibuffer=False,
                    WEIGHT_OFFSET=weight_offset,
                )
                project_expert_matrix[(cores,)](
                    a,
                    matrix,
                    out,
                    counts,
                    offsets,
                    expert,
                    k,
                    n,
                    unit_flag=True,
                    sync_solver=False,
                    num_warps=1,
                    multibuffer=True,
                    enable_preload=True,
                    enable_fp_fusion=False,
                )
            return
        panel_work = torch.empty(
            panel_grid * (panel_stages * panel_bn * k + core_padding),
            device=a.device,
            dtype=a.dtype,
        )
        gate_cache = (
            torch.empty((panel_grid, 2, 128, panel_bn), device=a.device, dtype=a.dtype)
            if fuse_swiglu
            else None
        )
        sections = ((0, n, panel_bn),)
        if wide_input and n % panel_bn:
            # The final narrower panel reads from the original pitched weight
            # matrix and writes into the matching columns of the output.
            tail = n % panel_bn
            sections = ((0, n - tail, panel_bn), (n - tail, tail, tail))
        for start, width, tile_n in sections:
            if width == n:
                weights, scales, destination = w, s, out
            else:
                weights = w.narrow(2, start, width)
                scales = s.narrow(2, start, width)
                destination = out.narrow(1, start, width)
            expert_panel_kernel[(panel_grid,)](
                a,
                weights,
                scales,
                counts,
                offsets,
                panel_work,
                destination,
                k,
                width,
                s.stride(1),
                w.shape[0],
                bm,
                tile_n,
                panel_grid,
                panel_stages,
                expert_outer,
                GateCache=gate_cache,
                PANEL_ROWS=144 if wide_input else 0,
                PANEL_STRIDE=out.stride(0),
                FULL_PANEL=full_panel,
                WEIGHT_OFFSET=weight_offset,
                disable_auto_inject_block_sync=True,
                unit_flag=True,
                sync_solver=False,
                num_warps=1,
                enable_fp_fusion=False,
                multibuffer=False,
            )
        return
    # Limit BN to fit two decoded BF16 weight tiles and both A tiles in L1.
    bn = min(n & -n, 512 if bits == 8 and bm <= 64 else 256, 32768 // bm)
    large_fp16 = bits == 8 and a.dtype == torch.float16 and bm == 128
    merge = (
        bm == 128
        and ((k == 256 and n == 4096) or (k == 4096 and n == 512))
        and not large_fp16
    )
    tail_tiles = tail_tiles and not large_fp16
    counts = None if large_fp16 else counts
    offsets = None if large_fp16 else offsets
    assert not tail_tiles or (
        bits == 8 and merge and counts is not None and offsets is not None
    )
    input_bm = bm
    if merge:
        bm, bn = 256, 128
    tasks = out.shape[0] // input_bm * (n // bn)
    grid = min(tasks, cores)
    if merge and k == 4096:
        grid = min(grid, 19)
    stages = 4 if a.dtype == torch.float16 and bn == 512 else 2
    work = torch.empty(
        (grid * (stages * (512 if tail_tiles else bn) * 128 + core_padding),),
        device=a.device,
        dtype=a.dtype,
    )
    # Wide FP16 tiles reuse L0C across projections. Use the Cube unit flags
    # for the handoff from matrix accumulation to the FIX writeback pipeline.
    wide_fp16 = (
        bits == 8 and a.dtype == torch.float16 and k % 1024 == 0 and n % 1024 == 0
    )
    gemm_kernel[(grid,)](
        a,
        w,
        s,
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
        bits,
        counts,
        offsets,
        tail_tiles,
        routes,
        topk,
        WEIGHT_OFFSET=weight_offset,
        disable_auto_inject_block_sync=True,
        unit_flag=True if wide_fp16 else None,
        sync_solver=False if wide_fp16 else None,
        CORE_PADDING=core_padding,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
        enable_preload=(
            True if bits == 8 and a.dtype == torch.float16 and bn == 512 else None
        ),
    )
