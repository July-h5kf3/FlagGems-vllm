# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import os
import tempfile
from functools import lru_cache

import triton
import triton.language as tl
from triton.language import core as tl_core

from flaggems_vllm.utils.code_cache import code_cache_dir


@tl_core.extern
def row_max8(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("mla_fp8_max8", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def row_sum8(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("mla_fp8_sum8", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def exp2_two(a, b, _semantic=None):
    """exp2 of two values in one instruction; the pair returns packed in a uint64."""
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b],
        {(tl.float32, tl.float32): ("mla_fp8_exp2_two", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.builtin
def inline_jit(fn, args, _semantic=None, _generator=None):
    """Use the same frontend inlining as WS for nested, role-local helpers.

    Normal nested JIT calls survive the stock inliner in isolated WS regions
    and lose the caller's explicit layout. This stays within the operator.
    """
    fn = tl_core._unwrap_if_constexpr(fn)
    args = tl_core._unwrap_if_constexpr(args)
    if isinstance(args, tl_core.tuple):
        args = tuple(args.values)
    return _generator.inline_JitFunction(fn, args, kwargs={})


@triton.jit
def exp2_pairs(x):
    """exp2 over the score tile, two elements per hardware instruction.

    Pairs must be thread-local. SQMMA puts the low three token bits across
    lanes, so neighbours along the last axis live in different threads; the
    same permute the row reduction uses brings the register-resident bits to
    the end, and the pairs are taken there.
    """
    x = tl.permute(tl.reshape(x, (x.shape[0], 2, 2, 2, 8)), (0, 4, 1, 2, 3))
    x0, x1 = tl.split(x)
    x00, x01 = tl.split(x0)
    x10, x11 = tl.split(x1)
    a0, a4 = tl.split(x00)
    a2, a6 = tl.split(x01)
    a1, a5 = tl.split(x10)
    a3, a7 = tl.split(x11)

    p04 = exp2_two(a0, a4)
    p26 = exp2_two(a2, a6)
    p15 = exp2_two(a1, a5)
    p37 = exp2_two(a3, a7)
    b0 = (p04 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b4 = (p04 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b2 = (p26 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b6 = (p26 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b1 = (p15 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b5 = (p15 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b3 = (p37 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b7 = (p37 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    y = tl.join(
        tl.join(tl.join(b0, b4), tl.join(b2, b6)),
        tl.join(tl.join(b1, b5), tl.join(b3, b7)),
    )
    return tl.reshape(tl.permute(y, (0, 2, 3, 4, 1)), (x.shape[0], 64))


@triton.jit
def row_partials(x, MAXIMUM: tl.constexpr):
    # SQMMA distributes low three K bits across lanes; high bits are registers.
    x = tl.permute(tl.reshape(x, (x.shape[0], 2, 2, 2, 8)), (0, 4, 1, 2, 3))
    x0, x1 = tl.split(x)
    x00, x01 = tl.split(x0)
    x10, x11 = tl.split(x1)
    a0, a4 = tl.split(x00)
    a2, a6 = tl.split(x01)
    a1, a5 = tl.split(x10)
    a3, a7 = tl.split(x11)
    if MAXIMUM:
        partial = row_max8(a0, a1, a2, a3, a4, a5, a6, a7)
    else:
        partial = row_sum8(a0, a1, a2, a3, a4, a5, a6, a7)
    return partial


@triton.jit
def row_reduce(x, MAXIMUM: tl.constexpr):
    if MAXIMUM:
        partial = inline_jit(row_partials, (x, True))
        result = tl.max(partial, 1)
    else:
        partial = inline_jit(row_partials, (x, False))
        result = tl.sum(partial, 1)
    return result


@triton.jit
def sum_max_combine(sum0, max0, sum1, max1):
    return sum0 + sum1, tl.maximum(max0, max1)


@triton.jit
def row_sum_and_max(probabilities, weights):
    # Both reductions are independent after exponentiation; interleave shuffles.
    partial_sum = inline_jit(row_partials, (probabilities, False))
    partial_max = inline_jit(row_partials, (weights, True))
    return tl.reduce((partial_sum, partial_max), 1, sum_max_combine)


@tl_core.extern
def native_recip(value, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [value],
        {(tl.float32,): ("mla_fp8_recip", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


NATIVE_PRIMITIVES = r"""
target triple = "musa"

declare <4 x float> @llvm.musa.max.f.bst4.src02(<4 x float>, <4 x float>)
declare <2 x float> @llvm.musa.max.f.bst2.src02(<2 x float>, <2 x float>)
declare <2 x float> @llvm.musa.add.f.bst2.src01(<2 x float>, <2 x float>)
declare float @llvm.maxnum.f32(float, float)
define float @mla_fp8_max8(float %a0, float %a1, float %a2, float %a3,
                          float %a4, float %a5, float %a6, float %a7)
    alwaysinline nounwind readnone {
  %l0 = insertelement <4 x float> poison, float %a0, i32 0
  %l1 = insertelement <4 x float> %l0, float %a1, i32 1
  %l2 = insertelement <4 x float> %l1, float %a2, i32 2
  %l3 = insertelement <4 x float> %l2, float %a3, i32 3
  %r0 = insertelement <4 x float> poison, float %a4, i32 0
  %r1 = insertelement <4 x float> %r0, float %a5, i32 1
  %r2 = insertelement <4 x float> %r1, float %a6, i32 2
  %r3 = insertelement <4 x float> %r2, float %a7, i32 3
  %v = call <4 x float> @llvm.musa.max.f.bst4.src02(<4 x float> %l3, <4 x float> %r3)
  %lo = shufflevector <4 x float> %v, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %hi = shufflevector <4 x float> %v, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %p = call <2 x float> @llvm.musa.max.f.bst2.src02(<2 x float> %lo, <2 x float> %hi)
  %x = extractelement <2 x float> %p, i32 0
  %y = extractelement <2 x float> %p, i32 1
  %z = call float @llvm.maxnum.f32(float %x, float %y)
  ret float %z
}
define float @mla_fp8_sum8(float %a0, float %a1, float %a2, float %a3,
                          float %a4, float %a5, float %a6, float %a7)
    alwaysinline nounwind readnone {
  %l0 = insertelement <4 x float> poison, float %a0, i32 0
  %l1 = insertelement <4 x float> %l0, float %a1, i32 1
  %l2 = insertelement <4 x float> %l1, float %a2, i32 2
  %l3 = insertelement <4 x float> %l2, float %a3, i32 3
  %r0 = insertelement <4 x float> poison, float %a4, i32 0
  %r1 = insertelement <4 x float> %r0, float %a5, i32 1
  %r2 = insertelement <4 x float> %r1, float %a6, i32 2
  %r3 = insertelement <4 x float> %r2, float %a7, i32 3
  %v = fadd <4 x float> %l3, %r3
  %lo = shufflevector <4 x float> %v, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %hi = shufflevector <4 x float> %v, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %p = call <2 x float> @llvm.musa.add.f.bst2.src01(<2 x float> %lo, <2 x float> %hi)
  %x = extractelement <2 x float> %p, i32 0
  %y = extractelement <2 x float> %p, i32 1
  %z = fadd float %x, %y
  ret float %z
}


declare <2 x float> @llvm.musa.exp2.f.bst2(<2 x float>)
define i64 @mla_fp8_exp2_two(float %a, float %b) alwaysinline nounwind readnone {
  %v0 = insertelement <2 x float> poison, float %a, i32 0
  %v1 = insertelement <2 x float> %v0, float %b, i32 1
  %r = call <2 x float> @llvm.musa.exp2.f.bst2(<2 x float> %v1)
  %i = bitcast <2 x float> %r to i64
  ret i64 %i
}

declare float @llvm.musa.rcp(float)
define float @mla_fp8_recip(float %x) alwaysinline nounwind readnone {
  %r = call float @llvm.musa.rcp(float %x)
  ret float %r
}


declare <4 x float> @llvm.musa.mul.f.bst4.src01(<4 x float>,<4 x float>)
define i64 @mla_fp8_mul4_lo(float %a,float %b,float %c,float %d,
    float %e,float %f,float %g,float %h) alwaysinline nounwind readnone {
 %l0 = insertelement <4 x float> poison,float %a,i32 0
 %l1 = insertelement <4 x float> %l0,float %b,i32 1
 %l2 = insertelement <4 x float> %l1,float %c,i32 2
 %l3 = insertelement <4 x float> %l2,float %d,i32 3
 %r0 = insertelement <4 x float> poison,float %e,i32 0
 %r1 = insertelement <4 x float> %r0,float %f,i32 1
 %r2 = insertelement <4 x float> %r1,float %g,i32 2
 %r3 = insertelement <4 x float> %r2,float %h,i32 3
 %r = call <4 x float> @llvm.musa.mul.f.bst4.src01(<4 x float> %l3,<4 x float> %r3)
 %p = shufflevector <4 x float> %r,<4 x float> poison,<2 x i32> <i32 0,i32 1>
 %bits = bitcast <2 x float> %p to i64
 ret i64 %bits
}
define i64 @mla_fp8_mul4_hi(float %a,float %b,float %c,float %d,
    float %e,float %f,float %g,float %h) alwaysinline nounwind readnone {
 %l0 = insertelement <4 x float> poison,float %a,i32 0
 %l1 = insertelement <4 x float> %l0,float %b,i32 1
 %l2 = insertelement <4 x float> %l1,float %c,i32 2
 %l3 = insertelement <4 x float> %l2,float %d,i32 3
 %r0 = insertelement <4 x float> poison,float %e,i32 0
 %r1 = insertelement <4 x float> %r0,float %f,i32 1
 %r2 = insertelement <4 x float> %r1,float %g,i32 2
 %r3 = insertelement <4 x float> %r2,float %h,i32 3
 %r = call <4 x float> @llvm.musa.mul.f.bst4.src01(<4 x float> %l3,<4 x float> %r3)
 %p = shufflevector <4 x float> %r,<4 x float> poison,<2 x i32> <i32 2,i32 3>
 %bits = bitcast <2 x float> %p to i64
 ret i64 %bits
}
declare <4 x float> @llvm.musa.add.f.bst4.src01(<4 x float>,<4 x float>)
define i64 @mla_fp8_add4_lo(float %a,float %b,float %c,float %d,
    float %e,float %f,float %g,float %h) alwaysinline nounwind readnone {
 %l0 = insertelement <4 x float> poison,float %a,i32 0
 %l1 = insertelement <4 x float> %l0,float %b,i32 1
 %l2 = insertelement <4 x float> %l1,float %c,i32 2
 %l3 = insertelement <4 x float> %l2,float %d,i32 3
 %r0 = insertelement <4 x float> poison,float %e,i32 0
 %r1 = insertelement <4 x float> %r0,float %f,i32 1
 %r2 = insertelement <4 x float> %r1,float %g,i32 2
 %r3 = insertelement <4 x float> %r2,float %h,i32 3
 %r = call <4 x float> @llvm.musa.add.f.bst4.src01(<4 x float> %l3,<4 x float> %r3)
 %p = shufflevector <4 x float> %r,<4 x float> poison,<2 x i32> <i32 0,i32 1>
 %bits = bitcast <2 x float> %p to i64
 ret i64 %bits
}
define i64 @mla_fp8_add4_hi(float %a,float %b,float %c,float %d,
    float %e,float %f,float %g,float %h) alwaysinline nounwind readnone {
 %l0 = insertelement <4 x float> poison,float %a,i32 0
 %l1 = insertelement <4 x float> %l0,float %b,i32 1
 %l2 = insertelement <4 x float> %l1,float %c,i32 2
 %l3 = insertelement <4 x float> %l2,float %d,i32 3
 %r0 = insertelement <4 x float> poison,float %e,i32 0
 %r1 = insertelement <4 x float> %r0,float %f,i32 1
 %r2 = insertelement <4 x float> %r1,float %g,i32 2
 %r3 = insertelement <4 x float> %r2,float %h,i32 3
 %r = call <4 x float> @llvm.musa.add.f.bst4.src01(<4 x float> %l3,<4 x float> %r3)
 %p = shufflevector <4 x float> %r,<4 x float> poison,<2 x i32> <i32 2,i32 3>
 %bits = bitcast <2 x float> %p to i64
 ret i64 %bits
}
"""


@lru_cache(maxsize=1)
def primitives_lib():
    digest = hashlib.sha256(NATIVE_PRIMITIVES.encode()).hexdigest()[:16]
    path = code_cache_dir() / ("musa-mla-fp8-" + digest + ".ll")
    libraries = (("musa_mla_fp8", str(path)),)
    if path.exists():
        return libraries
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as source:
        source.write(NATIVE_PRIMITIVES)
    os.replace(source.name, path)
    return libraries


@lru_cache(maxsize=1)
def require_tme_groups():
    from triton._C.libtriton import mthreads

    try:
        supported = (
            mthreads.supports_tle_tme_completion_groups()
            and mthreads.supports_tle_sqmma_m16_m_split()
            and mthreads.supports_tle_barrier_release()
            and mthreads.supports_tle_barrier_memory_ordering()
        )
    except AttributeError as error:
        raise NotImplementedError(
            "FP8 MLA requires FlagTree MUSA TME groups, M16 SQMMA and shared-memory ordering"
        ) from error
    if not supported:
        raise NotImplementedError(
            "FP8 MLA requires FlagTree MUSA TME groups, M16 SQMMA and shared-memory ordering"
        )


@tl_core.extern
def mul4_lo(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("mla_fp8_mul4_lo", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def mul4_hi(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("mla_fp8_mul4_hi", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def add4_lo(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("mla_fp8_add4_lo", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def add4_hi(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("mla_fp8_add4_hi", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def pointwise_vec4(x, y, MULTIPLY: tl.constexpr):
    y = tl.broadcast_to(y, x.shape)
    BH: tl.constexpr = x.shape[0]
    x = tl.permute(tl.reshape(x, (BH, 2, 2, 2, 8)), (0, 4, 1, 2, 3))
    y = tl.permute(tl.reshape(y, (BH, 2, 2, 2, 8)), (0, 4, 1, 2, 3))
    x0, x1 = tl.split(x)
    x00, x01 = tl.split(x0)
    x10, x11 = tl.split(x1)
    a0, a4 = tl.split(x00)
    a2, a6 = tl.split(x01)
    a1, a5 = tl.split(x10)
    a3, a7 = tl.split(x11)
    y0, y1 = tl.split(y)
    y00, y01 = tl.split(y0)
    y10, y11 = tl.split(y1)
    c0, c4 = tl.split(y00)
    c2, c6 = tl.split(y01)
    c1, c5 = tl.split(y10)
    c3, c7 = tl.split(y11)
    if MULTIPLY:
        p04 = mul4_lo(a0, a4, a2, a6, c0, c4, c2, c6)
        p26 = mul4_hi(a0, a4, a2, a6, c0, c4, c2, c6)
        p15 = mul4_lo(a1, a5, a3, a7, c1, c5, c3, c7)
        p37 = mul4_hi(a1, a5, a3, a7, c1, c5, c3, c7)
    else:
        p04 = add4_lo(a0, a4, a2, a6, c0, c4, c2, c6)
        p26 = add4_hi(a0, a4, a2, a6, c0, c4, c2, c6)
        p15 = add4_lo(a1, a5, a3, a7, c1, c5, c3, c7)
        p37 = add4_hi(a1, a5, a3, a7, c1, c5, c3, c7)
    b0 = (p04 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b4 = (p04 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b2 = (p26 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b6 = (p26 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b1 = (p15 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b5 = (p15 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b3 = (p37 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b7 = (p37 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    z = tl.join(
        tl.join(tl.join(b0, b4), tl.join(b2, b6)),
        tl.join(tl.join(b1, b5), tl.join(b3, b7)),
    )
    return tl.reshape(tl.permute(z, (0, 2, 3, 4, 1)), (BH, 64))
