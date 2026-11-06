# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton._C.libtriton import mthreads as musa_compiler
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError
from triton.experimental.tle import language as tle
from triton.experimental.tle.language.gpu.mthreads.types import MusaSqmmaEncoding
from triton.language import core
from triton.runtime import driver
from triton.runtime.cache import get_cache_manager
from triton.runtime.errors import OutOfResources

# The mixed path reuses FP16 operands and crosses SQMMA accumulator layouts.
# Older FlagTree builds compile it but produce incorrect values.
_MIXED_PV_SUPPORTED = tl.constexpr(
    all(
        getattr(musa_compiler, feature, lambda: False)()
        for feature in (
            "supports_tle_sqmma_m16_m_split",
            "supports_tle_barrier_memory_ordering",
        )
    )
)

PUB = tl.constexpr(MusaSqmmaEncoding([3, 1], [4, 4], [64, 32, 128]))
PROW = tl.constexpr(
    tle.gpu.SlicedEncoding(1, MusaSqmmaEncoding([3, 1], [4, 4], [64, 32, 128]))
)
PCOL = tl.constexpr(
    tle.gpu.SlicedEncoding(0, MusaSqmmaEncoding([3, 1], [4, 4], [64, 32, 128]))
)
PV = tl.constexpr(MusaSqmmaEncoding([3, 1], [4, 4], [64, 128, 128]))
OROW = tl.constexpr(
    tle.gpu.SlicedEncoding(1, MusaSqmmaEncoding([3, 1], [4, 4], [64, 128, 128]))
)
OCOL = tl.constexpr(
    tle.gpu.SlicedEncoding(0, MusaSqmmaEncoding([3, 1], [4, 4], [64, 128, 128]))
)

NATIVE_IR = r"""
declare void @llvm.musa.lma.wait()
define i32 @sparse_lma_wait(i32 %token) alwaysinline nounwind {
 call void @llvm.musa.lma.wait()
 ret i32 %token
}
declare void @llvm.musa.memcpy.g2s.robust.v4(ptr addrspace(3),ptr addrspace(1),i32,<4 x i32>,i32)
declare void @llvm.musa.memcpy.g2s.wait()
define i32 @bounded_copy16(i32 %dst,i64 %src,i64 %base,i64 %size) alwaysinline nounwind {
  %d = inttoptr i32 %dst to ptr addrspace(3)
  %s = inttoptr i64 %src to ptr addrspace(1)
  %b = bitcast i64 %base to <2 x i32>
  %z = bitcast i64 %size to <2 x i32>
  %desc = shufflevector <2 x i32> %b,<2 x i32> %z,<4 x i32> <i32 0,i32 1,i32 2,i32 3>
  call void @llvm.musa.memcpy.g2s.robust.v4(ptr addrspace(3) %d,ptr addrspace(1) %s,i32 16,<4 x i32> %desc,i32 0)
  ret i32 %dst
}
define i32 @bounded_copy_wait(i32 %token) alwaysinline nounwind {
  call void @llvm.musa.memcpy.g2s.wait()
  ret i32 %token
}

declare <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8>)
define i64 @sparse_promote_fp8_four(i32 %bytes) alwaysinline nounwind readnone {
  %v = bitcast i32 %bytes to <4 x i8>
  %h = call <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8> %v)
  %result = bitcast <4 x half> %h to i64
  ret i64 %result
}
"""


@lru_cache(maxsize=1)
def native_library():
    cache = get_cache_manager(hashlib.sha256(NATIVE_IR.encode()).hexdigest())
    filename = "sparse-fp8-pipeline.ll"
    return cache.get_file(filename) or cache.put(NATIVE_IR, filename, binary=False)


@core.extern
def lma_wait(token, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [token],
        {(tl.int32,): ("sparse_lma_wait", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def bounded_copy16(dst, src, base, size, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [dst, src, base, size],
        {(tl.uint32, tl.uint64, tl.uint64, tl.uint64): ("bounded_copy16", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def bounded_copy_wait(token, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [token],
        {(tl.int32,): ("bounded_copy_wait", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def promote_fp8_four(word, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [word],
        {(tl.uint32,): ("sparse_promote_fp8_four", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def consume(
    Output,
    PScale,
    ph,
    pl,
    sk,
    FinalInv,
    Full,
    PEmpty,
    Active,
    K: tl.constexpr,
    BN: tl.constexpr,
):
    if Active:
        row = tl.program_id(0).to(tl.int64)
        accum = tle.gpu.set_layout(tl.full((64, 512), 0.0, tl.float32), PV)
        for page in range(tl.cdiv(K, BN)):
            tle.gpu.barrier_wait(Full, phaseIdx=page & 1)
            values = sk.slot(page & 1)
            accum = tle.gpu.wgmma(ph, values, acc=accum)
            accum = tle.gpu.wgmma(pl, values, acc=accum)
            accum = tle.gpu.wgmma_wait(0, accum)
            tle.gpu.barrier_arrive(PEmpty, phaseIdx=page & 1)
        oh = tl.inline_asm_elementwise(
            "",
            constraints="=R,0",
            args=[tl.full((), 256, tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        od = tl.inline_asm_elementwise(
            "",
            constraints="=R,0",
            args=[tl.full((), 1024, tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        h = tle.gpu.set_layout(tl.arange(256, 320) - oh, OROW)
        d = tle.gpu.set_layout(tl.arange(1024, 1536) - od, OCOL)
        inverse = tl.load(tle.gpu.local_ptr(FinalInv, (h,)), volatile=True)
        scale = tl.load(PScale + row * 64 + h, volatile=True)
        result = accum * scale[:, None] * inverse[:, None]
        tl.store(Output + (row * 64 + h[:, None]) * 512 + d[None, :], result)


@triton.jit
def produce(
    Q,
    KV,
    QS,
    Indices,
    PScale,
    LSE,
    ph,
    pl,
    sk,
    FinalInv,
    Full,
    PEmpty,
    Sync,
    SumParts,
    SumSync,
    CopySync,
    Selected,
    Active,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
):
    if Active:
        row = tl.program_id(0).to(tl.int64)
        h = tle.gpu.set_layout(tl.arange(0, 64), PROW)
        d = tle.gpu.set_layout(tl.arange(0, 512), PCOL)
        col = tle.gpu.set_layout(tl.arange(0, BN), PCOL)
        q = tle.gpu.alloc((64, 512), tl.float8e4nv, nv_mma_shared_layout=True)
        qh = tl.broadcast_to(h[:, None], (64, 512))
        qd = tl.broadcast_to(d[None, :], (64, 512))
        tl.store(
            tle.gpu.local_ptr(q, (qh, qd)),
            tl.load(Q + (row * 64 + h[:, None]) * 512 + d[None, :]),
        )
        qs = tl.load(QS + row * 64 + h)
        ds = tl.load(PScale + row * 64 + h)
        reciprocal = tl.div_rn(1.0, ds)
        copy_origin = tl.inline_asm_elementwise(
            "",
            constraints="=R,0",
            args=[tl.full((), 128, tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        cr = tl.arange(128, 128 + BN) - copy_origin
        cd = tl.arange(0, 32) * 16
        copy_layout: tl.constexpr = tle.gpu.BlockEncoding(
            [1, 1], [2, 16], [16, 1], [1, 0]
        )
        crr = tle.gpu.set_layout(tl.broadcast_to(cr[:, None], (BN, 32)), copy_layout)
        cdd = tle.gpu.set_layout(tl.broadcast_to(cd[None, :], (BN, 32)), copy_layout)
        init_ids = tl.load(Indices + row * K + crr, crr < K, TOKENS, volatile=True)
        init_src = KV + init_ids.to(tl.int64) * 512 + cdd
        init_dst0 = (
            tle.gpu.local_ptr(sk.slot(0), (crr, cdd)).to(tl.uint64).to(tl.uint32)
        )
        bounded_copy16(
            init_dst0,
            init_src.to(tl.uint64),
            KV.to(tl.uint64),
            tl.full((), TOKENS * 512, tl.uint64),
        )
        bounded_copy_wait(tl.full((), 0, tl.int32))
        lma_wait(tl.full((), 0, tl.int32))
        tle.gpu.barrier_arrive(Sync, phaseIdx=0)
        tle.gpu.barrier_wait(Sync, phaseIdx=0)
        total_matrix = tle.gpu.set_layout(tl.full((64, BN), 0.0, tl.float32), PUB)
        for page in range(tl.cdiv(K, BN)):
            slot = page & 1
            offsets = page * BN + col
            ks = tl.load(Selected + row * K + offsets, offsets < K, 0.0)
            valid = (offsets < K) & (ks > 0.0)
            dot = tle.gpu.wgmma(q, sk.slot(slot), trans_b=True)
            dot = tle.gpu.wgmma_wait(0, dot).to(tl.float32)
            tle.gpu.barrier_wait(PEmpty, phaseIdx=page & 1)
            shifted = dot * qs[:, None] * ks[None, :] * SCALE
            probability = tl.where(valid[None, :], tl.exp(shifted), 0.0)
            total_matrix += probability
            normalized = probability * ks[None, :] * reciprocal[:, None]
            high = normalized.to(tl.float8e4nv)
            low = (normalized - high.to(tl.float32)).to(tl.float8e4nv)
            phh = tl.broadcast_to(h[:, None], (64, BN))
            phc = tl.broadcast_to(col[None, :], (64, BN))
            tl.store(tle.gpu.local_ptr(ph, (phh, phc)), high)
            tl.store(tle.gpu.local_ptr(pl, (phh, phc)), low)
            if page + 1 == tl.cdiv(K, BN):
                partial = tl.sum(tl.reshape(total_matrix, (64, 4, 32)), 2)
                hi = tl.broadcast_to(h[:, None], (64, BN))
                ci = tl.broadcast_to((col // 32)[None, :], (64, BN))
                rh = tl.max(tl.reshape(hi, (64, 4, 32)), 2)
                rc = tl.max(tl.reshape(ci, (64, 4, 32)), 2)
                tl.store(tle.gpu.local_ptr(SumParts, (rh, rc)), partial)
                lma_wait(tl.full((), 0, tl.int32))
                # This barrier is used once, regardless of the number of tiles.
                tle.gpu.barrier_arrive(SumSync, phaseIdx=0)
                tle.gpu.barrier_wait(SumSync, phaseIdx=0)
                s0 = tl.load(
                    tle.gpu.local_ptr(SumParts, (h, tl.full((64,), 0, tl.int32))),
                    volatile=True,
                )
                s1 = tl.load(
                    tle.gpu.local_ptr(SumParts, (h, tl.full((64,), 1, tl.int32))),
                    volatile=True,
                )
                s2 = tl.load(
                    tle.gpu.local_ptr(SumParts, (h, tl.full((64,), 2, tl.int32))),
                    volatile=True,
                )
                s3 = tl.load(
                    tle.gpu.local_ptr(SumParts, (h, tl.full((64,), 3, tl.int32))),
                    volatile=True,
                )
                total = (s0 + s1) + (s2 + s3)
                inverse = tl.where(
                    total > 0, 1.0 / tl.where(total > 0, total, 1.0), 0.0
                )
                tl.store(tle.gpu.local_ptr(FinalInv, (h,)), inverse)
                lse = tl.where(total > 0, tl.log(total), float("inf"))
                tl.store(LSE + row * 64 + h, lse)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Sync, phaseIdx=(page + 1) & 1)
            tle.gpu.barrier_wait(Sync, phaseIdx=(page + 1) & 1)
            tle.gpu.barrier_arrive(Full, phaseIdx=page & 1)
            if page + 1 < tl.cdiv(K, BN):
                ns = (page + 1) & 1
                copy_offsets = (page + 1) * BN + crr
                copy_ids = tl.load(
                    Indices + row * K + copy_offsets,
                    copy_offsets < K,
                    TOKENS,
                    volatile=True,
                )
                copy_src = KV + copy_ids.to(tl.int64) * 512 + cdd
                copy_dst0 = (
                    tle.gpu.local_ptr(sk.slot(ns), (crr, cdd))
                    .to(tl.uint64)
                    .to(tl.uint32)
                )
                bounded_copy16(
                    copy_dst0,
                    copy_src.to(tl.uint64),
                    KV.to(tl.uint64),
                    tl.full((), TOKENS * 512, tl.uint64),
                )
                bounded_copy_wait(tl.full((), 0, tl.int32))
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(CopySync, phaseIdx=page & 1)
                tle.gpu.barrier_wait(CopySync, phaseIdx=page & 1)


@triton.jit
def fused_fp8_guarded(
    Q,
    KV,
    QS,
    Indices,
    PScale,
    Fast,
    Output,
    LSE,
    Selected,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
):
    BN: tl.constexpr = 128
    sk = tle.gpu.alloc((2, BN, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
    inverse = tle.gpu.alloc((64,), tl.float32, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barrier(arrive_count=16)
    pe = tle.gpu.alloc_barrier(arrive_count=16, init=tle.gpu.READY)
    sync = tle.gpu.alloc_barrier(arrive_count=16)
    sp = tle.gpu.alloc((64, 4), tl.float32, nv_mma_shared_layout=False)
    sb = tle.gpu.alloc_barrier(arrive_count=16)
    cs = tle.gpu.alloc_barrier(arrive_count=16)
    active = tl.load(Fast + tl.program_id(0), volatile=True) != 0
    tl.debug_barrier()
    tle.gpu.warp_specialize(
        [
            (
                produce,
                (
                    Q,
                    KV,
                    QS,
                    Indices,
                    PScale,
                    LSE,
                    ph,
                    pl,
                    sk,
                    inverse,
                    full,
                    pe,
                    sync,
                    sp,
                    sb,
                    cs,
                    Selected,
                    active,
                    K,
                    TOKENS,
                    BN,
                    SCALE,
                ),
            ),
            (consume, (Output, PScale, ph, pl, sk, inverse, full, pe, active, K, BN)),
        ],
        worker_num_warps=[16],
        worker_num_regs=[128],
    )


@triton.jit
def gather_scales(
    Indices,
    KS,
    Selected,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(Indices + x, x < ROWS * K, -1)
    valid = (x < ROWS * K) & (ids >= 0) & (ids < TOKENS)
    scale = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
    tl.store(Selected + x, scale, x < ROWS * K)


@triton.jit
def cache_bounds(
    KV,
    KS,
    Indices,
    Selected,
    NormParts,
    ScaleParts,
    BadCount,
    TOKENS: tl.constexpr,
    SELECTED: tl.constexpr,
    BT: tl.constexpr,
):
    if tl.program_id(0) == 0:
        tl.store(BadCount, 0)
    offset = tl.program_id(0) * BT + tl.arange(0, BT)
    if SELECTED:
        token = tl.load(Indices + offset, offset < SELECTED, -1)
        valid = (offset < SELECTED) & (token >= 0) & (token < TOKENS)
    else:
        token = offset
        valid = token < TOKENS
    d = tl.arange(0, 512)
    raw = tl.load(
        KV + token[:, None].to(tl.int64) * 512 + d[None, :],
        valid[:, None],
        0.0,
    ).to(tl.float32)
    scale = tl.load(KS + token, valid, 0.0)
    if SELECTED:
        tl.store(Selected + offset, scale, offset < SELECTED)
    norm2 = tl.sum(raw * raw, 1) * scale * scale
    tl.store(NormParts + tl.program_id(0), tl.max(norm2, 0))
    tl.store(ScaleParts + tl.program_id(0), tl.max(scale, 0))


@triton.jit
def reduce_bounds(NormParts, ScaleParts, Global, PARTS: tl.constexpr, BP: tl.constexpr):
    part = tl.arange(0, BP)
    knorm2 = tl.max(tl.load(NormParts + part, part < PARTS, 0.0), 0)
    tl.store(Global, knorm2)
    kscale = tl.max(tl.load(ScaleParts + part, part < PARTS, 0.0), 0)
    tl.store(Global + 1, kscale)


@triton.jit
def query_bounds_global(
    Q, QS, Global, PScale, Fast, BadRows, BadCount, SCALE: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    knorm2 = tl.load(Global)
    kscale = tl.load(Global + 1)
    h = tl.arange(0, 64)
    d = tl.arange(0, 512)
    q = tl.load(Q + (row * 64 + h[:, None]) * 512 + d[None, :]).to(tl.float32)
    qs = tl.load(QS + row * 64 + h)
    qnorm2 = tl.sum(q * q, 1) * qs * qs
    bound = tl.sqrt(qnorm2 * knorm2) * SCALE * 1.0005 + 1.0e-5
    safe_bound = tl.minimum(bound, 20.0)
    scale = kscale * tl.exp(safe_bound) * (1.0001 / 448.0)
    error = 0.5 * kscale * tl.exp(2.0 * safe_bound) * 1.0001
    eligible = (
        tl.min(
            (
                (error < 0.0019)
                & (bound < 20.0)
                & (scale > 0.0)
                & (bound * (2.0 * 512.0 * 2.0**-24 / (1.0 - 512.0 * 2.0**-24)) < 1.0e-4)
            ).to(tl.int32),
            0,
        )
        != 0
    )
    tl.store(PScale + row * 64 + h, scale)
    tl.store(Fast + row, eligible.to(tl.int32))
    if not eligible:
        index = tl.atomic_add(BadCount, 1)
        tl.store(BadRows + index, row)


@lru_cache(maxsize=64)
def compiled_pipeline(topk, tokens, scale, output_dtype, device_index, target):
    """Cache compiled code by metadata; never retain caller Tensor storage."""
    fn = fused_fp8_guarded
    signature = dict(
        zip(
            fn.arg_names[:9],
            [
                "*fp8e4nv",
                "*fp8e4nv",
                "*fp32",
                "*i32",
                "*fp32",
                "*i32",
                output_dtype,
                "*fp32",
                "*fp32",
            ],
        )
    )
    constants = dict(K=topk, TOKENS=tokens, SCALE=scale)
    library = native_library()
    try:
        return triton.compile(
            ASTSource(
                fn,
                signature,
                constants,
                attrs={(i,): [["tt.divisibility", 16]] for i in range(9)},
            ),
            target=target,
            options=dict(
                num_warps=16,
                num_stages=1,
                maxnreg=128,
                enable_backend_opt=True,
                enable_llc_opt=True,
                extern_libs={"native": library},
            ),
        )
    except (CompilationError, OutOfResources, RuntimeError) as error:
        logging.getLogger(__name__).debug("Sparse FP8 pipeline unavailable: %s", error)
        return None


def prepare_pipeline(q, kv, qs, ks, indices, output, lse, scale):
    rows = q.shape[0]
    topk = indices.shape[-1]
    tokens = kv.shape[0] * 64
    # Only selected tokens contribute to QK/PV; their maxima give the same
    # error guarantees without scanning a larger, mostly unused cache.
    selected_count = rows * topk if rows * topk < tokens else 0
    parts = triton.cdiv(selected_count or tokens, 32)
    workspace_bytes = 4 * rows * topk + rows * (64 * 4 + 8) + 8 * parts + 12
    if workspace_bytes > 8 * 1024**3:
        return None
    if any(t.data_ptr() % 16 for t in (q, kv, qs, ks, indices, output, lse)):
        return None
    target = driver.active.get_current_target()
    dtype = "*bf16" if output.dtype == torch.bfloat16 else "*fp16"
    kernel = compiled_pipeline(topk, tokens, scale, dtype, q.device.index, target)
    if kernel is None:
        return None
    norm = torch.empty((parts,), dtype=torch.float32, device=q.device)
    scales = torch.empty_like(norm)
    global_bounds = torch.empty((2,), dtype=torch.float32, device=q.device)
    probability_scale = torch.empty((rows, 64), dtype=torch.float32, device=q.device)
    flags = torch.empty((rows,), dtype=torch.int32, device=q.device)
    bad_rows = torch.empty_like(flags)
    bad_count = torch.empty((1,), dtype=torch.int32, device=q.device)
    selected = torch.empty((rows, topk), dtype=torch.float32, device=q.device)
    a0 = (
        kv,
        ks,
        indices,
        selected,
        norm,
        scales,
        bad_count,
        tokens,
        selected_count,
        32,
    )
    a1 = (norm, scales, global_bounds, parts, triton.next_power_of_2(parts))
    a2 = (q, qs, global_bounds, probability_scale, flags, bad_rows, bad_count, scale)
    a3 = (indices, ks, selected, rows, topk, tokens, 1024)
    k0 = cache_bounds.warmup(*a0, num_warps=8, grid=(parts,))
    k1 = reduce_bounds.warmup(*a1, num_warps=16, grid=(1,))
    k2 = query_bounds_global.warmup(*a2, num_warps=16, grid=(rows,))
    g3 = triton.cdiv(rows * topk, 1024)
    if selected_count:
        r3 = None
    else:
        k3 = gather_scales.warmup(*a3, num_warps=4, grid=(g3,))
        r3 = k3[(g3, 1, 1)]
    r0 = k0[(parts, 1, 1)]
    r1 = k1[(1, 1, 1)]
    r2 = k2[(rows, 1, 1)]
    r4 = kernel[(rows, 1, 1)]
    a4 = (q, kv, qs, indices, probability_scale, flags, output, lse, selected)

    def run_pipeline():
        r0(*a0)
        r1(*a1)
        r2(*a2)
        if not selected_count:
            r3(*a3)
        r4(*a4)

    return run_pipeline, bad_rows, bad_count


def prepare_short_pipeline(q, kv, qs, ks, indices, output, lse, scale, combine):
    rows, topk = q.shape[0], indices.shape[-1]
    if any(t.data_ptr() % 16 for t in (q, kv, qs, ks, indices, output, lse)):
        return None
    if _MIXED_PV_SUPPORTED.value and 256 <= topk <= 1024:
        heads, warps = 32, 8
        splits = min(8, triton.cdiv(topk, 64), triton.cdiv(64, rows * 2))
    else:
        heads = 16 if topk <= 1024 else 64
        warps = 4 if heads == 16 else 16
        splits = (
            1
            if topk <= 1024
            else min(32, triton.cdiv(topk, 64), triton.cdiv(256, rows))
        )
    flags = torch.empty(
        (rows, splits * (64 // heads)), dtype=torch.int32, device=q.device
    )
    if splits == 1:
        partial, stats = output, lse
    else:
        partial = torch.empty(
            (rows, splits, 64, 512), dtype=torch.float32, device=q.device
        )
        stats = torch.empty((rows, splits, 64, 2), dtype=torch.float32, device=q.device)
    arguments = (
        q,
        kv,
        qs,
        ks,
        indices,
        partial,
        stats,
        flags,
        topk,
        kv.shape[0] * 64,
        scale,
        heads,
        splits,
    )
    library = native_library()
    try:
        kernel = short_sparse_fp8.warmup(
            *arguments,
            num_warps=warps,
            num_stages=1,
            extern_libs={"native": library},
            grid=(rows, 64 // heads, splits),
        )
    except (CompilationError, OutOfResources, RuntimeError) as error:
        logging.getLogger(__name__).debug(
            "Sparse FP8 short path unavailable: %s", error
        )
        return None
    runner = kernel[(rows, 64 // heads, splits)]
    if splits > 1:
        combine_args = (
            partial,
            stats,
            None,
            output,
            lse,
            rows,
            64,
            1,
            splits,
            False,
            triton.next_power_of_2(splits),
            512,
        )
        combined = combine.warmup(
            *combine_args, num_warps=4, num_stages=1, grid=(rows, 64, 1)
        )
        combine_runner = combined[(rows, 64, 1)]

    def run_short():
        runner(*arguments)
        if splits > 1:
            combine_runner(*combine_args)

    return run_short, flags


@triton.jit
def short_sparse_fp8(
    Q,
    KV,
    QS,
    KS,
    Indices,
    Output,
    LSE,
    Flags,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    BH: tl.constexpr,
    SPLITS: tl.constexpr,
):
    MIXED: tl.constexpr = _MIXED_PV_SUPPORTED and (K >= 256) and (K <= 1024)
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(2)
    begin = tl.cdiv(K, 64) * split // SPLITS
    end = tl.cdiv(K, 64) * (split + 1) // SPLITS
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    d = tl.arange(0, 512)
    col = tl.arange(0, 64)
    qv = tl.load(Q + (row * 64 + h[:, None]) * 512 + d[None, :])
    qs = tl.load(QS + row * 64 + h)
    q_l1 = tl.sum(tl.abs(qv.to(tl.float32)), 1) * qs
    q = tle.gpu.alloc(
        (BH, 512), tl.float8e4nv, init_value=qv, nv_mma_shared_layout=True
    )
    sk = tle.gpu.alloc((64, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    if MIXED:
        ph = tle.gpu.alloc((BH, 64), tl.float16, nv_mma_shared_layout=True)
        value = tle.gpu.alloc((64, 256), tl.float16, nv_mma_shared_layout=True)
    else:
        ph = tle.gpu.alloc((BH, 64), tl.float8e4nv, nv_mma_shared_layout=True)
        pl = tle.gpu.alloc((BH, 64), tl.float8e4nv, nv_mma_shared_layout=True)
    lma_wait(tl.full((), 0, tl.int32))
    tl.debug_barrier()
    cr = tl.arange(128, 192) - 128
    cd = tl.arange(1024, 1056) * 16 - 16384
    crr = tl.broadcast_to(cr[:, None], (64, 32))
    cdd = tl.broadcast_to(cd[None, :], (64, 32))
    dst = tle.gpu.local_ptr(sk, (crr, cdd)).to(tl.uint64).to(tl.uint32)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.full((BH,), 0.0, tl.float32)
    running = tl.full((BH,), 0.0, tl.float32)
    if MIXED:
        accum0 = tl.full((BH, 256), 0.0, tl.float32)
        accum1 = tl.full((BH, 256), 0.0, tl.float32)
    else:
        accum = tl.full((BH, 512), 0.0, tl.float32)
    needs_repair = tl.full((), 0, tl.int32)
    for page in range(begin, end):
        ids_copy = tl.load(
            Indices + row * K + page * 64 + crr, page * 64 + crr < K, TOKENS
        )
        src = KV + ids_copy.to(tl.int64) * 512 + cdd
        bounded_copy16(
            dst,
            src.to(tl.uint64),
            KV.to(tl.uint64),
            tl.full((), TOKENS * 512, tl.uint64),
        )
        bounded_copy_wait(tl.full((), 0, tl.int32))
        lma_wait(tl.full((), 0, tl.int32))
        tl.debug_barrier()
        dot = tle.gpu.wgmma(q, sk, trans_b=True)
        dot = tle.gpu.wgmma_wait(0, dot)
        ids = tl.load(Indices + row * K + page * 64 + col, page * 64 + col < K, -1)
        valid = (page * 64 + col < K) & (ids >= 0) & (ids < TOKENS)
        ks = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        error = (
            q_l1
            * (448.0 * tl.max(ks, 0))
            * SCALE
            * ((512.0 * 2.0**-24) / (1.0 - 512.0 * 2.0**-24))
        )
        needs_repair = tl.maximum(
            needs_repair, (tl.max(error, 0) > 1.0e-4).to(tl.int32)
        )
        scores = tl.where(
            valid[None, :],
            dot * qs[:, None] * ks[None, :] * (SCALE * 1.4426950408889634),
            -float("inf"),
        )
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        safe = tl.where(new_max == -float("inf"), 0.0, new_max)
        p = tl.exp2(scores - safe[:, None])
        correction = tl.exp2(maximum - safe)
        denominator = denominator * correction + tl.sum(p, 1)
        weights = p * ks[None, :]
        if MIXED:
            ds = tl.maximum(tl.max(weights, 1), 1.0e-30)
            inv = 1.0 / ds
            tl.store(tle.gpu.local_ptr(ph), (weights * inv[:, None]).to(tl.float16))
            vr = tl.arange(256, 320) - 256
            vc = tl.arange(4096, 4160) * 4 - 16384
            vrr = tl.broadcast_to(vr[:, None], (64, 64))
            vcc = tl.broadcast_to(vc[None, :], (64, 64))
            # E4M3 -> FP16 is exact. Reuse the 256-column V buffer so two
            # head groups can reside on an MPC, with both accumulators in FP32.
            for half in tl.static_range(2):
                raw = tl.load(
                    tle.gpu.local_ptr(sk, (vrr, vcc + half * 256)).to(
                        tl.pointer_type(tl.uint32, 3)
                    ),
                    volatile=True,
                )
                tl.store(
                    tle.gpu.local_ptr(value, (vrr, vcc)).to(
                        tl.pointer_type(tl.uint64, 3)
                    ),
                    promote_fp8_four(raw),
                )
                lma_wait(tl.full((), 0, tl.int32))
                tl.debug_barrier()
                if half == 0:
                    accum0 = tle.gpu.wgmma(
                        ph, value, acc=accum0 * (running * correction * inv)[:, None]
                    )
                    accum0 = tle.gpu.wgmma_wait(0, accum0)
                else:
                    accum1 = tle.gpu.wgmma(
                        ph, value, acc=accum1 * (running * correction * inv)[:, None]
                    )
                    accum1 = tle.gpu.wgmma_wait(0, accum1)
                tl.debug_barrier()
        else:
            ds = tl.maximum(tl.max(weights, 1) * (1.0 / 448.0), 1.0e-30)
            inv = 1.0 / ds
            norm = weights * inv[:, None]
            high = norm.to(tl.float8e4nv)
            low = ((norm - high.to(tl.float32)) * 16.0).to(tl.float8e4nv)
            tl.store(tle.gpu.local_ptr(ph), high)
            tl.store(tle.gpu.local_ptr(pl), low)
            lma_wait(tl.full((), 0, tl.int32))
            tl.debug_barrier()
            accum = tle.gpu.wgmma(
                ph, sk, acc=accum * (running * correction * inv)[:, None]
            )
            accum = tle.gpu.wgmma_wait(0, accum)
            accum = tle.gpu.wgmma(pl, sk, acc=accum * 16.0)
            accum = tle.gpu.wgmma_wait(0, accum) * (1.0 / 16.0)
            tl.debug_barrier()
        running = ds
        maximum = new_max
    tl.store(
        Flags + (row * SPLITS + split) * (64 // BH) + tl.program_id(1), needs_repair
    )
    factor = running
    if SPLITS == 1:
        factor *= tl.div_rn(1.0, tl.where(denominator > 0, denominator, 1.0))
    base = (row * SPLITS + split) * 64 + h
    if MIXED:
        od = tl.arange(0, 256)
        tl.store(Output + base[:, None] * 512 + od[None, :], accum0 * factor[:, None])
        tl.store(
            Output + base[:, None] * 512 + 256 + od[None, :], accum1 * factor[:, None]
        )
    else:
        tl.store(Output + base[:, None] * 512 + d[None, :], accum * factor[:, None])
    logsum = maximum * 0.6931471805599453 + tl.log(
        tl.where(denominator > 0, denominator, 1.0)
    )
    if SPLITS == 1:
        tl.store(LSE + row * 64 + h, tl.where(denominator > 0, logsum, float("inf")))
    else:
        tl.store(LSE + base * 2, maximum * 0.6931471805599453)
        tl.store(LSE + base * 2 + 1, denominator)
