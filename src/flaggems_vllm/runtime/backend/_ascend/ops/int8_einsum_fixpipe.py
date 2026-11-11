# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fine-grained Ascend Fixpipe epilogue for block-scaled INT8 einsum.

The INT8 Cube dot remains in Triton.  Each K=128 partial is drained from L0C
through the 910B Fixpipe fragment and accumulated by a small BF16 reduction.
This is only valid for the contiguous 128x128 scale layout; callers outside
that contract stay on the regular TLE epilogue.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._ascend.ops.ascendc.compile_fixpipe import (
    ensure_bitcode,
    makefile_compile,
    source_path,
)
from flaggems_vllm.runtime.backend._ascend.ops.ascendc.compile_fixpipe import (
    symbol as _fixpipe_symbol,
)
from flaggems_vllm.utils import libentry

_FIXPIPE_READY = False


def _register_fixpipe() -> None:
    global _FIXPIPE_READY
    if _FIXPIPE_READY:
        return
    _bitcode = ensure_bitcode()

    @al.register_custom_op
    class fixpipe_vdeqf16:  # noqa: N801
        name = "fixpipe_vdeqf16"
        core = al.CORE.CUBE
        pipe = al.PIPE.PIPE_FIX
        mode = al.MODE.SIMD
        symbol = _fixpipe_symbol()
        bitcode = str(_bitcode)
        source = str(source_path())
        compile = makefile_compile()

        def __init__(
            self,
            acc,
            deq,
            row_diag,
            c,
            off_m,
            off_n,
            tile_m,
            tile_n,
            acc_stride,
            ldc,
            load_deq,
            load_diag,
            out_bf16,
            out=None,
        ):
            assert out is not None, "fixpipe_vdeqf16 requires a dummy output tensor"
            for name in (
                "off_m",
                "off_n",
                "tile_m",
                "tile_n",
                "acc_stride",
                "ldc",
                "load_deq",
                "load_diag",
                "out_bf16",
            ):
                self.arg_type[name] = tl.int32
            del acc, deq, row_diag, c

    _FIXPIPE_READY = True


@triton.jit
def _pack_fixpipe_deq_kernel(
    YS,
    DEQ,
    N: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    N_PAD: tl.constexpr,
    SYS0: tl.constexpr,
    SYS1: tl.constexpr,
    SYS2: tl.constexpr,
):
    pid = tl.program_id(0)
    index = pid * 256 + tl.arange(0, 256)
    kb = index // N_PAD
    n = index % N_PAD
    valid = (kb < K_BLOCKS) & (n < N)
    group = n // 128
    scale = tl.load(YS + group * SYS1 + kb * SYS2, valid, other=0.0).to(tl.float32)
    bits = scale.to(tl.uint32, bitcast=True).to(tl.uint64)
    tl.store(DEQ + index, bits, index < K_BLOCKS * N_PAD)


@triton.jit
def _pack_fixpipe_diag_kernel(
    XS,
    DIAG,
    M: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUPS: tl.constexpr,
    M1: tl.constexpr,
    SXS0: tl.constexpr,
    SXS2: tl.constexpr,
):
    group_stride: tl.constexpr = M1 * M1 * 256
    diag_stride: tl.constexpr = GROUPS * group_stride
    pid = tl.program_id(0)
    index = pid * 256 + tl.arange(0, 256)
    kb = index // diag_stride
    rem = index % diag_stride
    group = rem // group_stride
    rem = rem % group_stride
    row_block = rem // (M1 * 256)
    rem = rem % (M1 * 256)
    col_block = rem // 256
    rem = rem % 256
    row_in = rem // 16
    col_in = rem % 16
    row = group * BLOCK_M + row_block * 16 + row_in
    diagonal = row_block == col_block and row_in == col_in and row < M
    scale = tl.load(XS + row * SXS0 + kb * SXS2, diagonal, other=0.0).to(tl.float16)
    tl.store(
        DIAG + index, tl.where(diagonal, scale, 0.0), index < K_BLOCKS * diag_stride
    )


@libentry()
@triton.jit
def _fixpipe_partial_kernel(
    X,
    Y,
    DEQ,
    DIAG,
    PARTIAL,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SX0: tl.constexpr,
    SX2: tl.constexpr,
    SY1: tl.constexpr,
    SY2: tl.constexpr,
    N_PAD: tl.constexpr,
    DIAG_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n: tl.constexpr = N // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    with al.scope(core_mode="cube"):
        dummy = tl.full((16,), 0, tl.float16)
        a_ptr = tl.make_block_ptr(
            X,
            (M, K),
            (SX0, SX2),
            (off_m, 0),
            (BLOCK_M, BLOCK_K),
            (1, 0),
        )
        b_ptr = tl.make_block_ptr(
            Y + off_n * SY1,
            (K, N),
            (SY2, SY1),
            (0, 0),
            (BLOCK_K, BLOCK_N),
            (1, 0),
        )
        for kb in tl.range(
            0, K, BLOCK_K, num_stages=2, disallow_acc_multi_buffer=False
        ):
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
            a = tl.load(a_ptr)
            b = tl.load(b_ptr)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            partial = PARTIAL + kb // BLOCK_K * M * N
            al.custom(
                "fixpipe_vdeqf16",
                acc,
                DEQ + (kb // BLOCK_K) * N_PAD,
                DIAG + (kb // BLOCK_K) * DIAG_STRIDE,
                partial,
                off_m,
                off_n,
                BLOCK_M,
                BLOCK_N,
                BLOCK_M,
                N,
                2,
                1,
                1,
                out=dummy,
            )
            a_ptr = tl.advance(a_ptr, (0, BLOCK_K))
            b_ptr = tl.advance(b_ptr, (BLOCK_K, 0))


@triton.jit
def _reduce_fixpipe_partials(
    PARTIAL,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n: tl.constexpr = N // BLOCK_N
    row = (pid // grid_n) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = (pid % grid_n) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for kb in range(K_BLOCKS):
        value = tl.load(PARTIAL + kb * M * N + row[:, None] * N + col[None, :]).to(
            tl.float32
        )
        acc += value
    tl.store(OUT + row[:, None] * N + col[None, :], acc.to(OUT.dtype.element_ty))


def try_fixpipe_int8_einsum(
    x: torch.Tensor,
    xs: torch.Tensor,
    y: torch.Tensor,
    ys: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Run the Fixpipe partial path when its strict layout contract holds."""
    # The CANN 9.1 runtime currently needs the matching FlagTree custom-op
    # lowering for this raw Fixpipe ABI; keep the fragment opt-in until that
    # compiler build is deployed with the operator.
    if os.environ.get("FLAGGEMS_ASCEND_EINSUM_FIXPIPE", "0") != "1":
        return False
    if output.dtype != torch.bfloat16 or x.dtype != torch.int8 or y.dtype != torch.int8:
        return False
    if x.ndim != 3 or y.ndim != 3 or x.shape[1] != 1 or y.shape[0] != 1:
        return False
    b, _, r = x.shape
    d = y.shape[1]
    if b > 256 or b % 16 != 0 or d % 128 != 0 or r % 128 != 0:
        return False
    if xs.dtype != torch.float32 or ys.dtype != torch.float32:
        return False
    if x.stride(2) != 1 or x.stride(1) != r or y.stride(2) != 1:
        return False
    _register_fixpipe()
    block_m, block_n, block_k = 16, 128, 128
    k_blocks = r // block_k
    n_pad = d
    groups = b // block_m
    m1 = block_m // 16
    diag_stride = groups * m1 * m1 * 256
    with torch_device_fn.device(x.device):
        deq = torch.empty((k_blocks * n_pad,), device=x.device, dtype=torch.uint64)
        diag = torch.empty(
            (k_blocks * diag_stride,), device=x.device, dtype=torch.float16
        )
        partial = torch.empty((k_blocks, b, d), device=x.device, dtype=torch.bfloat16)
        _pack_fixpipe_deq_kernel[(triton.cdiv(k_blocks * n_pad, 256),)](
            ys[0],
            deq,
            d,
            k_blocks,
            n_pad,
            ys.stride(0),
            ys.stride(1),
            ys.stride(2),
            num_warps=1,
            multibuffer=False,
        )
        _pack_fixpipe_diag_kernel[(triton.cdiv(k_blocks * diag_stride, 256),)](
            xs[:, 0, :],
            diag,
            b,
            k_blocks,
            block_m,
            groups,
            m1,
            xs.stride(0),
            xs.stride(2),
            num_warps=1,
            multibuffer=False,
        )
        _fixpipe_partial_kernel[(groups * (d // block_n),)](
            x[:, 0, :],
            y[0],
            deq,
            diag,
            partial,
            b,
            d,
            r,
            block_m,
            block_n,
            block_k,
            x.stride(0),
            x.stride(2),
            y.stride(1),
            y.stride(2),
            n_pad,
            diag_stride,
            mix_mode="aic",
            num_warps=1,
            num_stages=2,
            optimize_dynamic_offset=True,
            unit_flag=False,
            limit_auto_multi_buffer_of_local_buffer="no-l0c",
        )
        _reduce_fixpipe_partials[(groups * (d // block_n),)](
            partial,
            output,
            b,
            d,
            k_blocks,
            block_m,
            block_n,
            num_warps=1,
            multibuffer=False,
            enable_fp_fusion=False,
        )
    return True


__all__ = ["try_fixpipe_int8_einsum"]
