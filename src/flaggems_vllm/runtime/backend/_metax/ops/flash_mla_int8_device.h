// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <maca.h>
#include <math.h>

typedef unsigned uint2_ __attribute__((ext_vector_type(2)));
typedef unsigned uint4_ __attribute__((ext_vector_type(4)));
typedef int int4_ __attribute__((ext_vector_type(4)));
typedef float float2_ __attribute__((ext_vector_type(2)));
typedef float float4_ __attribute__((ext_vector_type(4)));
#define GLOBAL __attribute__((address_space(1)))
#define SHARED __attribute__((address_space(3)))
namespace {
template <class T>
__device__ __forceinline__ const GLOBAL T *input(long long p) {
  return (const GLOBAL T *)(__UINTPTR_TYPE__)p;
}
template <class T> __device__ __forceinline__ GLOBAL T *output(long long p) {
  return (GLOBAL T *)(__UINTPTR_TYPE__)p;
}
__device__ __forceinline__ int uniform(int x) {
  return __builtin_mxc_readfirstlane(x);
}
__device__ __forceinline__ long long uniform(long long x) {
  return (
      long long)((unsigned long long)(unsigned)uniform((int)x) |
                 ((unsigned long long)(unsigned)uniform((int)(x >> 32)) << 32));
}
__device__ __forceinline__ void sync() {
  __builtin_mxc_arrive_bsmcnt(0);
  __builtin_mxc_barrier_ex(2);
}
__device__ __forceinline__ unsigned perm(unsigned hi, unsigned lo,
                                         unsigned selector) {
  return __builtin_mxc_byte_perm(hi, lo, selector);
}
__device__ __forceinline__ int4_ mma(unsigned a, unsigned b, int4_ c) {
  return __builtin_mxc_mma_16x16x16i8((int)a, (int)b, c);
}
__device__ __forceinline__ float4_ rope_mma(uint2_ a, uint2_ b, float4_ c,
                                            int bf16) {
  if (bf16)
    return __builtin_mxc_mma_16x16x16bf16(a, b, c);
  return __builtin_mxc_mma_16x16x16f16(a, b, c);
}
__device__ __forceinline__ unsigned pack_bf16(float a, float b) {
  auto pair = __builtin_mxc_cvt_pk_f32tobf16((float2_){a, b});
  return *(unsigned *)&pair;
}
} // namespace
