// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "flash_mla_int8_device.h"
extern "C" __device__ __attribute__((const)) float
reciprocal(float) __asm("llvm.mxc.rcp.f32");
namespace {
__device__ __forceinline__ float max4(float4_ x) {
  return fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));
}
__device__ __forceinline__ float sum4(float4_ x) {
  return (x[0] + x[1]) + (x[2] + x[3]);
}

__device__ __forceinline__ int koff(int token, int dim) {
  return token * 256 + (dim ^ (((token >> 2) & 7) * 16));
}
__device__ __forceinline__ int roff(int token, int dim) {
  return token * 64 + (dim ^ (((token >> 2) & 3) * 16));
}

} // namespace
extern "C" __device__ __attribute__((used, always_inline)) float
mla_int8_scores(long long qp, long long qrp, long long kp, long long krp,
                long long qsp, long long ksp, long long tp, long long lp,
                long long probability_ptr, long long page_stats_ptr,
                long long split_stats_ptr, long long orderp, long long ts,
                long long ls, int hq, int sq, int capacity, int splits,
                int causal, int bf16, float scale_log2, int split,
                int head_tile, int batch) {
  qp = uniform(qp);
  qrp = uniform(qrp);
  kp = uniform(kp);
  krp = uniform(krp);
  qsp = uniform(qsp);
  ksp = uniform(ksp);
  tp = uniform(tp);
  lp = uniform(lp);
  probability_ptr = uniform(probability_ptr);
  page_stats_ptr = uniform(page_stats_ptr);
  split_stats_ptr = uniform(split_stats_ptr);
  orderp = uniform(orderp);
  ts = uniform(ts);
  ls = uniform(ls);
  hq = uniform(hq);
  sq = uniform(sq);
  capacity = uniform(capacity);
  splits = uniform(splits);
  causal = uniform(causal);
  bf16 = uniform(bf16);
  split = uniform(split);
  head_tile = uniform(head_tile);
  batch = uniform(batch);
  hq = FMLA_HQ;
  sq = FMLA_SQ;
  capacity = FMLA_CAP;
  splits = FMLA_SPLITS;
  causal = FMLA_CAUSAL;
  bf16 = FMLA_BF16;
  ts = FMLA_CAP / 64;
  ls = 1;
  scale_log2 = FMLA_SCALE;
  if (orderp)
    batch = uniform(input<int>(orderp)[batch]);
  const int t = (threadIdx.x & 255), lane = t & 63, wave = t >> 6,
            row = lane & 15, group = lane >> 4;
  const int head = head_tile * 64 + wave * 16 + row, heads = hq * sq;
  const int length = uniform(input<int>(lp)[batch * ls]),
            pages = (length + 63) / 64;
  const int begin = pages * split / splits, end = pages * (split + 1) / splits;
  const int bound = causal ? max(length - sq + head / hq + 1, 0) : length;
  const long long qrow = (long long)batch * heads + head;
  const float qs = input<float>(qsp)[qrow];
  // A lane owns one complete 16-token quantization group.
  unsigned query[32];
  uint2_ query_rope[4];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const uint4_ words = *input<uint4_>(qp + qrow * 512 + k * 64 + group * 16);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      query[4 * k + j] = words[j];
  }
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const uint4_ words =
        *input<uint4_>(qrp + (qrow * 64 + group * 16 + k * 8) * 2);
    query_rope[2 * k] = (uint2_){words[0], words[1]};
    query_rope[2 * k + 1] = (uint2_){words[2], words[3]};
  }
  __shared__ unsigned char key[16384];
  __shared__ unsigned short rope[4096];
  float maximum = -INFINITY, denominator = 0.f, pscale = 0.f;
  for (int page_index = begin; page_index < end; ++page_index) {
    const long long page = uniform(input<int>(tp)[batch * ts + page_index]);
    int4_ content[4];
#pragma unroll
    for (int n = 0; n < 4; ++n)
      content[n] = (int4_){0, 0, 0, 0};
#pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int i = t + c * 256, token = i >> 4,
                  dim = ((i & 15) ^ ((token >> 2) & 7)) * 16;
        void *dest = key + ((t & ~63) + c * 256) * 16;
        void *src = (void *)input<unsigned char>(
            kp + (page * 64 + token) * 512 + chunk * 256 + dim);
        __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false,
                                              true, 1, 2, 40);
      }
      if (chunk == 1) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int i = t + c * 256, token = i >> 3,
                    dim = ((i & 7) ^ (((token >> 2) & 3) * 2)) * 8;
          void *dest = (unsigned char *)rope + ((t & ~63) + c * 256) * 16;
          void *src = (void *)input<unsigned char>(
              krp + ((page * 64 + token) * 64 + dim) * 2);
          __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false,
                                                true, page_index * 64 + token,
                                                length, 40);
        }
      }
      __builtin_mxc_arrive_gvmcnt(0);
      __builtin_mxc_barrier_ex(2);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const uint4_ words =
              *(const uint4_ *)(key + koff(row * 4 + n, k * 64 + group * 16));
#pragma unroll
          for (int j = 0; j < 4; ++j)
            content[n] = __builtin_mxc_mma_16x16x16i8(
                (int)words[j], (int)query[chunk * 16 + k * 4 + j], content[n]);
        }
      }
      sync();
    }
    float4_ scores[4], weighted[4], scales[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      scales[j] = *input<float4_>(ksp + (page * 64 + group * 16 + j * 4) * 4);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      float4_ r = {0, 0, 0, 0};
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const uint4_ words =
            *(const uint4_ *)(rope + roff(row * 4 + n, group * 16 + k * 8));
        r = rope_mma((uint2_){words[0], words[1]}, query_rope[2 * k], r, bf16);
        r = rope_mma((uint2_){words[2], words[3]}, query_rope[2 * k + 1], r,
                     bf16);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int token = page_index * 64 + group * 16 + j * 4 + n;
        if (token >= length)
          scales[j][n] = 0.f;
        scores[n][j] = token < bound ? ((float)content[n][j] + r[j]) * qs *
                                           scales[j][n] * scale_log2
                                     : -INFINITY;
      }
    }
    const float local_max = fmaxf(fmaxf(max4(scores[0]), max4(scores[1])),
                                  fmaxf(max4(scores[2]), max4(scores[3])));
    const float safe = local_max == -INFINITY ? 0.f : local_max;
    float4_ sums, peaks;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      float4_ p;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        p[j] = exp2f(scores[n][j] - safe);
        weighted[n][j] = p[j] * scales[j][n];
      }
      sums[n] = sum4(p);
      peaks[n] = max4(weighted[n]);
    }
    const float local_sum = sum4(sums), peak = max4(peaks);
    const float local_scale = fmaxf(peak * (1.f / 32512.f), 1.e-30f),
                inv = reciprocal(local_scale);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      uint4_ packed;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const int i = half * 8 + w * 2;
        const unsigned lo =
            __float_as_uint(weighted[i % 4][i / 4] * inv + 8388608.f);
        const unsigned hi = __float_as_uint(
            weighted[(i + 1) % 4][(i + 1) / 4] * inv + 8388608.f);
        packed[w] = __builtin_mxc_byte_perm(hi, lo, 0x05040100);
      }
      const long long dest =
          ((long long)batch * capacity + page_index * 64) * heads + head * 64 +
          group * 16 + half * 8;
      *output<uint4_>(probability_ptr + dest * 2) = packed;
    }
    const long long offset =
        (((long long)batch * (capacity / 64) * 2 + page_index * 2) * heads +
         head) *
            4 +
        group;
    output<float>(page_stats_ptr)[offset] = local_max;
    output<float>(page_stats_ptr)[offset + heads * 4] = local_scale;
    const float raised = fmaxf(maximum, local_max),
                safe_max = raised == -INFINITY ? 0.f : raised;
    const float alpha = exp2f(maximum - safe_max),
                beta = exp2f(local_max - safe_max);
    denominator = denominator * alpha + local_sum * beta;
    pscale = fmaxf(pscale * alpha, local_scale * beta);
    maximum = raised;
    sync();
  }
  const long long offset =
      (((long long)batch * splits * 3 + split * 3) * 4 + group) * heads + head;
  output<float>(split_stats_ptr)[offset] = maximum;
  output<float>(split_stats_ptr)[offset + heads * 4] = pscale;
  output<float>(split_stats_ptr)[offset + heads * 8] = denominator;
  return 0.f;
}
