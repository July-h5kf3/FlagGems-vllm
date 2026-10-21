// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "flash_mla_int8_device.h"
extern "C" __device__ __attribute__((const)) float
reciprocal(float) __asm("llvm.mxc.rcp.f32");
#ifndef HEADS
#define HEADS 16
#endif
namespace {
constexpr int NW = 64 / HEADS, NR = 4 / NW, QV = 8 / NW;

__device__ __forceinline__ int koff(int token, int dim) {
  return token * 128 + (dim ^ (((token >> 1) & 7) * 16));
}
__device__ __forceinline__ int roff(int token, int dim) {
  return token * 64 + (dim ^ (((token >> 1) & 3) * 16));
}

} // namespace

extern "C" __device__ __attribute__((used, always_inline)) float
mla_int8_small(long long qp, long long qrp, long long kp, long long krp,
               long long qsp, long long ksp, long long tp, long long lp,
               long long pp, long long sp, long long op, long long lsep,
               long long ts, long long ls, long long orderp, int batches,
               int hq, int sq, int splits, int causal, int bf16,
               float scale_log2, int head_tile, int batch, int split) {
  qp = uniform(qp);
  qrp = uniform(qrp);
  kp = uniform(kp);
  krp = uniform(krp);
  qsp = uniform(qsp);
  ksp = uniform(ksp);
  tp = uniform(tp);
  lp = uniform(lp);
  pp = uniform(pp);
  sp = uniform(sp);
  op = uniform(op);
  lsep = uniform(lsep);
  ts = uniform(ts);
  ls = uniform(ls);
  orderp = uniform(orderp);
  batches = uniform(batches);
  hq = uniform(hq);
  sq = uniform(sq);
  splits = uniform(splits);
  causal = uniform(causal);
  bf16 = uniform(bf16);
  head_tile = uniform(head_tile);
  batch = uniform(batch);
  split = uniform(split);
  batches = FMLA_B;
  hq = FMLA_HQ;
  sq = FMLA_SQ;
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
  const int wn = wave % NW, wm = wave / NW,
            head = head_tile * HEADS + wm * 16 + row;
  const int allheads = hq * sq, length = uniform(input<int>(lp)[batch * ls]);
  const int pages = (length + 63) / 64, begin = pages * split / splits,
            end = pages * (split + 1) / splits;
  const int bound = causal ? max(length - sq + head / hq + 1, 0) : length;
  const long long qrow = (long long)batch * allheads + head;
  const float qs = input<float>(qsp)[qrow];
#if HEADS == 32
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

#define QNOPE(chunk, k)                                                        \
  ((uint4_){query[(chunk) * 8 + (k) * 4], query[(chunk) * 8 + (k) * 4 + 1],    \
            query[(chunk) * 8 + (k) * 4 + 2],                                  \
            query[(chunk) * 8 + (k) * 4 + 3]})
#define QROPE(k)                                                               \
  ((uint4_){query_rope[2 * (k)][0], query_rope[2 * (k)][1],                    \
            query_rope[2 * (k) + 1][0], query_rope[2 * (k) + 1][1]})
#else
  __shared__ unsigned char qbuffer[8192];
  __shared__ unsigned char qrbuffer[2048];
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const int i = t + c * 256, r = i >> 5, dim = ((i & 31) ^ (r & 7)) * 16;
    void *dest = qbuffer + ((t & ~63) + c * 256) * 16;
    void *src = (void *)input<unsigned char>(
        qp + ((long long)batch * allheads + head_tile * 16 + r) * 512 + dim);
    __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false, true,
                                          1, 2, 40);
  }
  if (t < 128) {
    const int r = t >> 3, dim = ((t & 7) ^ ((r & 3) * 2)) * 8;
    void *dest = qrbuffer + (t & ~63) * 16;
    void *src = (void *)input<unsigned char>(
        qrp +
        (((long long)batch * allheads + head_tile * 16 + r) * 64 + dim) * 2);
    __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false, true,
                                          1, 2, 40);
  }
  __builtin_mxc_arrive_gvmcnt(0);
  sync();
#define QNOPE(chunk, k)                                                        \
  (*(const uint4_ *)(qbuffer + row * 512 +                                     \
                     (((chunk) * 128 + (k) * 64 + group * 16) ^                \
                      ((row & 7) * 16))))
#define QROPE(k)                                                               \
  (*(const uint4_ *)(qrbuffer + (row * 64 + ((group * 16 + (k) * 8) ^          \
                                             ((row & 3) * 16))) *              \
                                    2))
#endif
  // Two K planes overlap H16 copies with QK; H32 retains four planes for PV.
  __shared__ unsigned char planes[HEADS == 32 ? 4 : 2][8192];
  auto copy_wait = [&]() {
    __builtin_mxc_arrive_gvmcnt(0);
    __builtin_mxc_barrier_ex(2);
  };
  auto load_plane = [&](long long page, int page_index, int chunk) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int i = t + c * 256, token = i >> 3;
      const int dim = ((i & 7) ^ ((token >> 1) & 7)) * 16;
      void *dest = planes[HEADS == 32 ? chunk : (chunk & 1)] +
                   ((t & ~63) + c * 256) * 16;
      void *src = (void *)input<unsigned char>(kp + (page * 64 + token) * 512 +
                                               chunk * 128 + dim);
      __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false,
                                            true, 1, 2, 40);
    }
  };
  auto fragments = [&](const unsigned char *base, int token, int col,
                       unsigned (&words)[QV]) {
    const int aligned = col & ~3;
    const unsigned a = *(const unsigned *)(base + koff(token, aligned));
    const unsigned b = *(const unsigned *)(base + koff(token + 1, aligned));
    const unsigned c = *(const unsigned *)(base + koff(token + 2, aligned));
    const unsigned d = *(const unsigned *)(base + koff(token + 3, aligned));
    if (QV == 2) {
      const unsigned selector = 0x05040100 + (col & 2) * 0x01010101;
      const unsigned u = perm(b, a, selector), v = perm(d, c, selector);
      words[0] = perm(v, u, 0x06040200);
      words[1] = perm(v, u, 0x07050301);
    } else {
      const unsigned u = perm(b, a, 0x05040100), v = perm(d, c, 0x05040100);
      const unsigned w = perm(b, a, 0x07060302), z = perm(d, c, 0x07060302);
      words[0] = perm(v, u, 0x06040200);
      words[1] = perm(v, u, 0x07050301);
      words[2] = perm(z, w, 0x06040200);
      words[3] = perm(z, w, 0x07050301);
    }
  };
  float4_ acc[QV * 4];
#pragma unroll
  for (int a = 0; a < QV * 4; ++a)
    acc[a] = (float4_){0, 0, 0, 0};
  float maximum = -INFINITY, denominator = 0.f;
  for (int page_index = begin; page_index < end; ++page_index) {
    const long long page = uniform(input<int>(tp)[batch * ts + page_index]);
    int4_ content[NR];
#pragma unroll
    for (int n = 0; n < NR; ++n)
      content[n] = (int4_){0, 0, 0, 0};
    unsigned retained[HEADS == 16 ? 4 : 1][4][QV];
    if (HEADS == 32) {
#pragma unroll
      for (int chunk = 0; chunk < 4; ++chunk)
        load_plane(page, page_index, chunk);
      copy_wait();
    } else {
      load_plane(page, page_index, 0);
      load_plane(page, page_index, 1);
      copy_wait();
    }
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      if (HEADS == 16 && chunk == 2)
        copy_wait();
      const auto tile = planes[HEADS == 32 ? chunk : (chunk & 1)];
#pragma unroll
      for (int n = 0; n < NR; ++n) {
        const int token = wn * (64 / NW) + row * NR + n;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const uint4_ words =
              *(const uint4_ *)(tile + koff(token, k * 64 + group * 16));
          const uint4_ qwords = QNOPE(chunk, k);
#pragma unroll
          for (int j = 0; j < 4; ++j)
            content[n] = mma(words[j], qwords[j], content[n]);
        }
      }
      if (HEADS == 16 || chunk == 0) {
#pragma unroll
        for (int k = 0; k < 4; ++k)
          fragments(tile, group * 16 + k * 4, wn * (128 / NW) + row * QV,
                    retained[HEADS == 16 ? chunk : 0][k]);
      }
      if (HEADS == 16) {
        sync();
        if (chunk < 2)
          load_plane(page, page_index, chunk + 2);
      }
    }
    if (HEADS == 32)
      sync();
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int i = t + c * 256, token = i >> 3,
                dim = ((i & 7) ^ (((token >> 1) & 3) * 2)) * 8;
      void *dest = planes[0] + ((t & ~63) + c * 256) * 16;
      void *src = (void *)input<unsigned char>(
          krp + ((page * 64 + token) * 64 + dim) * 2);
      __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false,
                                            true, page_index * 64 + token,
                                            length, 40);
    }
    copy_wait();
    float4_ rope[NR], scales[NR], probability[NR];
    float local_max = -INFINITY;
#pragma unroll
    for (int n = 0; n < NR; ++n) {
      rope[n] = (float4_){0, 0, 0, 0};
      const int token = wn * (64 / NW) + row * NR + n;
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const uint4_ words =
            *(const uint4_ *)(planes[0] + roff(token, group * 16 + k * 8) * 2);
        const uint4_ qwords = QROPE(k);
        rope[n] = rope_mma((uint2_){words[0], words[1]},
                           (uint2_){qwords[0], qwords[1]}, rope[n], bf16);
        rope[n] = rope_mma((uint2_){words[2], words[3]},
                           (uint2_){qwords[2], qwords[3]}, rope[n], bf16);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
        scales[n][j] = input<float>(
            ksp)[page * 64 + wn * (64 / NW) + group * (NR * 4) + j * NR + n];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int token_index =
            page_index * 64 + wn * (64 / NW) + group * (NR * 4) + j * NR + n;
        if (token_index >= length)
          scales[n][j] = 0.f;
        probability[n][j] = token_index < bound
                                ? ((float)content[n][j] + rope[n][j]) * qs *
                                      scales[n][j] * scale_log2
                                : -INFINITY;
        local_max = fmaxf(local_max, probability[n][j]);
      }
    }
    sync();
    local_max = fmaxf(local_max, __shfl_xor(local_max, 16, 64));
    local_max = fmaxf(local_max, __shfl_xor(local_max, 32, 64));
    float *stats = (float *)(planes[0] + (HEADS == 16 ? 4096 : 0));
    if (group == 0)
      stats[wn * HEADS + wm * 16 + row] = local_max;
    sync();
    float page_max = -INFINITY;
#pragma unroll
    for (int n = 0; n < NW; ++n)
      page_max = fmaxf(page_max, stats[n * HEADS + wm * 16 + row]);
    const float next_max = fmaxf(maximum, page_max),
                safe = next_max == -INFINITY ? 0.f : next_max;
    float local_sum = 0.f, local_peak = 0.f;
#pragma unroll
    for (int n = 0; n < NR; ++n) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        probability[n][j] = exp2f(probability[n][j] - safe);
        local_sum += probability[n][j];
        probability[n][j] *= scales[n][j];
        local_peak = fmaxf(local_peak, probability[n][j]);
      }
    }
    local_sum += __shfl_xor(local_sum, 16, 64);
    local_sum += __shfl_xor(local_sum, 32, 64);
    local_peak = fmaxf(local_peak, __shfl_xor(local_peak, 16, 64));
    local_peak = fmaxf(local_peak, __shfl_xor(local_peak, 32, 64));
    if (HEADS == 32)
      sync();
    if (group == 0) {
      stats[((HEADS == 16 ? 2 * NW : 0) + wn) * HEADS + wm * 16 + row] =
          local_sum;
      stats[((HEADS == 16 ? 3 * NW : NW) + wn) * HEADS + wm * 16 + row] =
          local_peak;
    }
    sync();
    float page_sum = 0.f, peak = 0.f;
#pragma unroll
    for (int n = 0; n < NW; ++n) {
      page_sum +=
          stats[((HEADS == 16 ? 2 * NW : 0) + n) * HEADS + wm * 16 + row];
      peak = fmaxf(
          peak,
          stats[((HEADS == 16 ? 3 * NW : NW) + n) * HEADS + wm * 16 + row]);
    }
    const float pscale =
        fmaxf(peak / (HEADS == 32 ? 32512.f : 16129.f), 1.e-30f);
    const float next_den = denominator * exp2f(maximum - safe) + page_sum;
    const float inverse_den = reciprocal(next_den > 0.f ? next_den : 1.f);
    const float old_weight = denominator * exp2f(maximum - safe) * inverse_den,
                outscale = pscale * inverse_den;
    const float inv = reciprocal(pscale);
    if (HEADS == 32)
      sync();
#pragma unroll
    for (int word = 0; word < NR; ++word) {
      unsigned integer[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int index = word * 4 + j;
        const float p = probability[index % NR][index / NR];
        integer[j] = HEADS == 32 ? __float_as_uint(p * inv + 8388736.f)
                                 : (unsigned)(int)(p * inv + 0.5f);
      }
      const unsigned p01 = perm(integer[1], integer[0], 0x05040100);
      const unsigned p23 = perm(integer[3], integer[2], 0x05040100);
      unsigned high, low;
      if (HEADS == 32) {
        high = perm(p23, p01, 0x07050301);
        low = perm(p23, p01, 0x06040200) ^ 0x80808080u;
      } else {
        high = perm((p23 + 0x00400040u) >> 7, (p01 + 0x00400040u) >> 7,
                    0x06040200);
        low = perm(p23, p01, 0x06040200) ^ ((high & 0x01010101u) << 7);
      }
      const int token = wn * (64 / NW) + group * (NR * 4) + word * 4;
      const int offset =
          (wm * 16 + row) * 64 + (token ^ ((((wm * 16 + row) >> 1) & 3) * 16));
      *(unsigned *)(planes[0] + offset) = high;
      *(unsigned *)(planes[0] + HEADS * 64 + offset) = low;
    }
    sync();
    const int pbase = (wm * 16 + row) * 64 +
                      ((group * 16) ^ ((((wm * 16 + row) >> 1) & 3) * 16));
    const uint4_ ph = *(const uint4_ *)(planes[0] + pbase),
                 pl = *(const uint4_ *)(planes[0] + HEADS * 64 + pbase);
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      unsigned values[4][QV];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        if (HEADS == 16 || chunk == 0) {
#pragma unroll
          for (int m = 0; m < QV; ++m)
            values[k][m] = retained[HEADS == 16 ? chunk : 0][k][m];
        } else
          fragments(planes[chunk], group * 16 + k * 4,
                    wn * (128 / NW) + row * QV, values[k]);
      }
#pragma unroll
      for (int m = 0; m < QV; ++m) {
        int4_ high = {0, 0, 0, 0};
#pragma unroll
        for (int k = 0; k < 4; ++k)
          high = mma(values[k][m], ph[k], high);
        int4_ combined = high * (HEADS == 32 ? 256 : 128);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          combined = mma(values[k][m], pl[k], combined);
        float4_ converted;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          converted[j] = (float)combined[j];
        acc[chunk * QV + m] =
            acc[chunk * QV + m] * old_weight + converted * outscale;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          unsigned bits = __float_as_uint(acc[chunk * QV + m][j]);
          asm volatile("" : "+r"(bits));
          acc[chunk * QV + m][j] = __uint_as_float(bits);
        }
      }
    }
    maximum = next_max;
    denominator = next_den;
    sync();
  }
  const long long outrow = ((long long)batch * allheads + head) * 512;
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    const int dim = chunk * 128 + wn * (128 / NW) + group * (4 * QV);
    if (splits == 1 && bf16) {
#pragma unroll
      for (int v = 0; v < QV / 2; ++v) {
        uint4_ packed;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const int i = v * 8 + w * 2;
          packed[w] = pack_bf16(acc[chunk * QV + i % QV][i / QV],
                                acc[chunk * QV + (i + 1) % QV][(i + 1) / QV]);
        }
        *output<uint4_>(op + (outrow + dim + v * 8) * 2) = packed;
      }
    } else if (splits == 1) {
#pragma unroll
      for (int i = 0; i < 4 * QV; ++i)
        output<_Float16>(op)[outrow + dim + i] =
            (_Float16)acc[chunk * QV + i % QV][i / QV];
    } else {
#pragma unroll
      for (int v = 0; v < QV; ++v) {
        float4_ packed;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = v * 4 + j;
          packed[j] = acc[chunk * QV + i % QV][i / QV];
        }
        const long long index =
            (long long)split * batches * allheads * 512 + outrow + dim + v * 4;
        *output<float4_>(pp + index * 4) = packed;
      }
    }
  }
  if (group == 0 && wn == 0) {
    const long long r = (long long)batch * allheads + head;
    if (splits == 1)
      output<float>(
          lsep)[((long long)batch * hq + head % hq) * sq + head / hq] =
          denominator > 0.f ? maximum * 0.6931471805599453f + logf(denominator)
                            : INFINITY;
    else {
      output<float>(sp)[(long long)(split * 2) * batches * allheads + r] =
          maximum;
      output<float>(sp)[(long long)(split * 2 + 1) * batches * allheads + r] =
          denominator;
    }
  }
  return 0.f;
}
