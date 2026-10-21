// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "flash_mla_int8_device.h"
// Coarse device-function organization follows FlagGems-vllm PR #972.
#ifndef VDIM
#define VDIM 512
#endif
constexpr int THREADS = VDIM;
constexpr int PCOPIES = 512 / THREADS;

namespace {

__device__ __forceinline__ int voffset(int token, int dim) {
  return token * VDIM + (dim ^ (((token >> 4) & 1) * 64));
}
__device__ __forceinline__ int poffset(int head, int token) {
  return head * 64 + (token ^ (((head >> 1) & 3) * 16));
}
__device__ __forceinline__ unsigned quantize(unsigned value, float factor) {
  const float biased = __uint_as_float(value | 0x4b000000u);
  const float normalized = fmaf(biased, factor, -8388608.f * factor);
  return __float_as_uint(normalized + 8388736.f);
}

} // namespace

extern "C" __device__ __attribute__((used, always_inline)) float
mla_int8_pv(long long kv_ptr, long long table_ptr, long long lengths_ptr,
            long long scores_ptr, long long cache_metadata_ptr,
            long long metadata_ptr, long long scale_ptr, long long output_ptr,
            long long partial_ptr, long long flag_ptr, long long kv_page_stride,
            long long kv_token_stride, long long table_stride,
            long long length_stride, int batches, int heads, int capacity,
            int splits, int adaptive, int sort_batches, int uniform_splits,
            int output_bf16, int value_tile, int head_tile, int job) {
  kv_ptr = uniform(kv_ptr);
  table_ptr = uniform(table_ptr);
  lengths_ptr = uniform(lengths_ptr);
  scores_ptr = uniform(scores_ptr);
  cache_metadata_ptr = uniform(cache_metadata_ptr);
  metadata_ptr = uniform(metadata_ptr);
  scale_ptr = uniform(scale_ptr);
  output_ptr = uniform(output_ptr);
  partial_ptr = uniform(partial_ptr);
  flag_ptr = uniform(flag_ptr);
  kv_page_stride = uniform(kv_page_stride);
  kv_token_stride = uniform(kv_token_stride);
  table_stride = uniform(table_stride);
  length_stride = uniform(length_stride);
  batches = uniform(batches);
  heads = uniform(heads);
  capacity = uniform(capacity);
  splits = uniform(splits);
  adaptive = uniform(adaptive);
  sort_batches = uniform(sort_batches);
  uniform_splits = uniform(uniform_splits);
  output_bf16 = uniform(output_bf16);
  value_tile = uniform(value_tile);
  head_tile = uniform(head_tile);
  job = uniform(job);
  batches = FMLA_B;
  heads = FMLA_HEADS;
  capacity = FMLA_CAP;
  splits = FMLA_SPLITS;
  adaptive = FMLA_ADAPTIVE;
  sort_batches = FMLA_SORT;
  uniform_splits = FMLA_UNIFORM;
  output_bf16 = FMLA_BF16;
  kv_page_stride = 32768;
  kv_token_stride = 512;
  table_stride = FMLA_CAP / 64;
  length_stride = 1;
  int batch = adaptive ? job % batches : job;
  const int split = adaptive ? job / batches : 0;
  const auto flags = input<int>(flag_ptr);
  const int split_count =
      adaptive ? (flags[0] != 0 ? splits : uniform_splits) : 1;
  if (split >= split_count)
    return 0.f;
  if (sort_batches)
    batch = uniform(flags[1 + batch]);
  const int length = uniform(input<int>(lengths_ptr)[batch * length_stride]);
  const int pages = (length + 63) / 64, begin = pages * split / split_count,
            end = pages * (split + 1) / split_count;
  const int t = threadIdx.x & (THREADS - 1), lane = t & 63, warp = t >> 6,
            row = lane & 15, group = lane >> 4;
  __shared__ unsigned char values[64 * VDIM];
  __shared__ unsigned char probability[2][2][64 * 64];
  int4_ high[4][4], low[4][4];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      high[m][n] = (int4_){0, 0, 0, 0};
      low[m][n] = (int4_){0, 0, 0, 0};
    }
  }
  float maximum[PCOPIES], inverse[PCOPIES];
#pragma unroll
  for (int c = 0; c < PCOPIES; ++c) {
    const int worker = t + c * THREADS, h = worker >> 3,
              head = head_tile * 64 + h;
    const long long offset = ((long long)batch * heads + head) * 2;
    maximum[c] = head < heads ? input<float>(metadata_ptr)[offset] : 0.f;
    inverse[c] = head < heads ? input<float>(metadata_ptr)[offset + 1] : 0.f;
  }
  struct ProbabilityData {
    uint4_ score[PCOPIES];
    float factor[PCOPIES];
  };
  auto load_probability = [&](int page_index) {
    ProbabilityData data;
#pragma unroll
    for (int c = 0; c < PCOPIES; ++c) {
      const int worker = t + c * THREADS, h = worker >> 3,
                tokens = (worker & 7) * 8, head = head_tile * 64 + h;
      if (head < heads) {
        const long long offset =
            ((long long)batch * capacity + page_index * 64) * heads +
            head * 64 + tokens;
        data.score[c] = *input<uint4_>(scores_ptr + offset * 2);
        const long long meta =
            (((long long)batch * (capacity / 64) * 2 + page_index * 2) * heads +
             head) *
                4 +
            tokens / 16;
        const float local_max = input<float>(cache_metadata_ptr)[meta];
        const float local_scale =
            input<float>(cache_metadata_ptr)[meta + heads * 4];
        data.factor[c] =
            local_scale * exp2f(local_max - maximum[c]) * inverse[c];
      } else {
        data.score[c] = (uint4_){0, 0, 0, 0};
        data.factor[c] = 0.f;
      }
    }
    return data;
  };
  auto store_probability = [&](const ProbabilityData &data, int slot) {
#pragma unroll
    for (int c = 0; c < PCOPIES; ++c) {
      const int worker = t + c * THREADS, h = worker >> 3,
                tokens = (worker & 7) * 8;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const unsigned a = data.score[c][2 * i], b = data.score[c][2 * i + 1];
        const unsigned x0 = quantize(a & 65535u, data.factor[c]),
                       x1 = quantize(a >> 16, data.factor[c]);
        const unsigned x2 = quantize(b & 65535u, data.factor[c]),
                       x3 = quantize(b >> 16, data.factor[c]);
        const unsigned p01 = perm(x1, x0, 0x05040100),
                       p23 = perm(x3, x2, 0x05040100);
        const int dest = poffset(h, tokens + i * 4);
        *(unsigned *)(probability[slot][0] + dest) = perm(p23, p01, 0x07050301);
        *(unsigned *)(probability[slot][1] + dest) =
            perm(p23, p01, 0x06040200) ^ 0x80808080u;
      }
    }
  };
  auto issue_values = [&](int page_index) {
    const long long page =
        uniform(input<int>(table_ptr)[batch * table_stride + page_index]);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int i = t + c * THREADS, token = i / (VDIM / 16),
                dim = ((i % (VDIM / 16)) ^ (((token >> 4) & 1) * 4)) * 16;
      void *dest = values + ((t & ~63) + c * THREADS) * 16;
      void *src = (void *)input<unsigned char>(kv_ptr + page * kv_page_stride +
                                               token * kv_token_stride +
                                               value_tile * VDIM + dim);
      __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false,
                                            true, 1, 2, 40);
    }
  };
  int slot = 0;
  if (begin < end) {
    issue_values(begin);
    const auto first = load_probability(begin);
    store_probability(first, 0);
    __builtin_mxc_arrive_gvmcnt(0);
    sync();
  }
  for (int page_index = begin; page_index < end; ++page_index) {
    const bool has_next = page_index + 1 < end;
    ProbabilityData next;
    if (has_next)
      next = load_probability(page_index + 1);
    unsigned fragment[4][4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int token = group * 16 + k * 4, col = warp * 64 + row * 4;
      const unsigned a = *(const unsigned *)(values + voffset(token, col));
      const unsigned b = *(const unsigned *)(values + voffset(token + 1, col));
      const unsigned c = *(const unsigned *)(values + voffset(token + 2, col));
      const unsigned d = *(const unsigned *)(values + voffset(token + 3, col));
      const unsigned p01 = perm(b, a, 0x05040100), p23 = perm(d, c, 0x05040100);
      const unsigned q01 = perm(b, a, 0x07060302), q23 = perm(d, c, 0x07060302);
      fragment[k][0] = perm(p23, p01, 0x06040200);
      fragment[k][1] = perm(p23, p01, 0x07050301);
      fragment[k][2] = perm(q23, q01, 0x06040200);
      fragment[k][3] = perm(q23, q01, 0x07050301);
    }
    sync();
    if (has_next) {
      store_probability(next, slot ^ 1);
      issue_values(page_index + 1);
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int p = poffset(n * 16 + row, group * 16);
      const uint4_ ph = *(const uint4_ *)(probability[slot][0] + p),
                   pl = *(const uint4_ *)(probability[slot][1] + p);
#pragma unroll
      for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          high[m][n] = mma(fragment[k][m], ph[k], high[m][n]);
          low[m][n] = mma(fragment[k][m], pl[k], low[m][n]);
        }
      }
    }
    if (has_next) {
      __builtin_mxc_arrive_gvmcnt(0);
      sync();
      slot ^= 1;
    }
  }
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const int head = head_tile * 64 + n * 16 + row;
    if (head < heads) {
      const float scale =
          split_count == 1
              ? input<float>(scale_ptr)[(long long)batch * heads + head]
              : 1.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float4_ raw;
#pragma unroll
        for (int m = 0; m < 4; ++m)
          raw[m] = (float)high[m][n][j] * 256.f + (float)low[m][n][j];
        const int dim = value_tile * VDIM + warp * 64 + group * 16 + j * 4;
        const long long index = ((long long)batch * heads + head) * 512 + dim;
        if (split_count == 1) {
          const float4_ result = raw * scale;
          if (output_bf16) {
            const uint2_ packed = {pack_bf16(result[0], result[1]),
                                   pack_bf16(result[2], result[3])};
            *output<uint2_>(output_ptr + index * 2) = packed;
          } else {
#pragma unroll
            for (int m = 0; m < 4; ++m)
              output<_Float16>(output_ptr)[index + m] = (_Float16)result[m];
          }
        } else {
          const long long partial_index =
              (long long)split * batches * heads * 512 + index;
          *output<float4_>(partial_ptr + partial_index * 4) = raw;
        }
      }
    }
  }
  return 0.f;
}
