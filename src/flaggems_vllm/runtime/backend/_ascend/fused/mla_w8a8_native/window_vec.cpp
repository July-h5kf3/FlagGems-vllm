/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Derived from arch22/incre_flash_attention_preload_mla.h at 1e878a0c.
// The special entry owns the Vector UB. The enclosing TLE scope uses scalar operands.
#include "kernel_operator.h"
using namespace AscendC;
namespace {
template<typename T> __aicore__ inline __ubuf__ T *ub(int offset) {
    return reinterpret_cast<__ubuf__ T *>(uint64_t(offset));
}
template<typename T> __aicore__ inline LocalTensor<T> tensor(int offset, int n) {
    return LocalTensor<T>(TPosition::VECCALC, offset, n);
}
__aicore__ inline void full_mask() {
    set_mask_norm();
    set_vector_mask(0, ~uint64_t(0));
}

template<int rows, int width>
__aicore__ inline void online_softmax_vexp(
    int score_addr, int work_addr, int sum_now, int sum_prev,
    int max_now, int max_prev, int alpha_now, int tmp_addr,
    uint32_t valid, uint32_t first)
{
    auto score = ub<float>(score_addr);
    auto t0 = ub<float>(work_addr);
    auto t1 = ub<float>(work_addr + 4096);
    auto t2 = ub<float>(work_addr + 8192);
    full_mask();
    vmax(t0, score, score + 64, rows, 1, 1, 1, 8, width / 8, width / 8);
    vmax(t1, score + 128, score + 192, rows, 1, 1, 1, 8, width / 8, width / 8);
    vmax(t2, score + 256, score + 320, rows, 1, 1, 1, 8, width / 8, width / 8);
    pipe_barrier(PIPE_V);
    vmax(t0, t0, t1, rows, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmax(t0, t0, t2, rows, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vcmax(ub<float>(tmp_addr), t0, rows, 1, 1, 8, Order_t::ONLY_VALUE);
    pipe_barrier(PIPE_V);

    auto row_max_t = tensor<float>(tmp_addr, rows);
    auto prev_max_t = tensor<float>(max_prev, rows);
    auto now_max_t = tensor<float>(max_now, rows);
    auto safe_max_t = tensor<float>(tmp_addr + 64, rows);
    auto now_alpha_t = tensor<float>(alpha_now, rows);
    if (first) {
        Adds(now_max_t, row_max_t, 0.f, rows);
    } else {
        Max(now_max_t, prev_max_t, row_max_t, rows);
    }
    pipe_barrier(PIPE_V);
    Maxs(safe_max_t, now_max_t, -1.e30f, rows);
    pipe_barrier(PIPE_V);
    if (!first) {
        Sub(now_alpha_t, prev_max_t, safe_max_t, rows);
        pipe_barrier(PIPE_V);
        Exp(now_alpha_t, now_alpha_t, rows);
    }
    full_mask();
    vbrcb(reinterpret_cast<__ubuf__ uint32_t*>(t0), reinterpret_cast<__ubuf__ uint32_t*>(ub<float>(tmp_addr + 64)), 1, 8, rows / 8);
    pipe_barrier(PIPE_V);
    for (int col = 0; col < width; col += 64)
        vsub(score + col, score + col, t0, rows, 1, 1, 0, width / 8, width / 8, 1);
    pipe_barrier(PIPE_V);
    if (valid < width) {
        Maxs(tensor<float>(score_addr, rows * width), tensor<float>(score_addr, rows * width), -90.f, rows * width);
        pipe_barrier(PIPE_V);
        full_mask();
    }
    vexp(score, score, rows * width / 64, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    if (valid < width) {
        for (int c = 0; c < width; c += 64) {
            if (int(valid) >= c + 64) continue;
            uint64_t mask = ~uint64_t(0);
            if (int(valid) > c) mask <<= int(valid) - c;
            set_vector_mask(0, mask);
            vector_dup(score + c, 0.f, rows, 1, 1, width / 8, 0);
        }
        full_mask();
        pipe_barrier(PIPE_V);
    }
    vadd(t0, score, score + 64, rows, 1, 1, 1, 8, width / 8, width / 8);
    vadd(t1, score + 128, score + 192, rows, 1, 1, 1, 8, width / 8, width / 8);
    vadd(t2, score + 256, score + 320, rows, 1, 1, 1, 8, width / 8, width / 8);
    pipe_barrier(PIPE_V);
    vadd(t0, t0, t1, rows, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(t0, t0, t2, rows, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vcadd(ub<float>(tmp_addr), t0, rows, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    auto now_sum_t = tensor<float>(sum_now, rows);
    if (first) {
        Adds(now_sum_t, tensor<float>(tmp_addr, rows), 0.f, rows);
    } else {
        Mul(now_sum_t, tensor<float>(sum_prev, rows), now_alpha_t, rows);
        pipe_barrier(PIPE_V);
        Add(now_sum_t, now_sum_t, tensor<float>(tmp_addr, rows), rows);
    }
    pipe_barrier(PIPE_V);
    full_mask();
}

template<int M>
struct VectorWindow {
    static constexpr int rows = M == 16 ? 8 : M / 4, heads = M / 2, width = 384, dim = 512;
    static constexpr int tiles = heads / rows;
    static constexpr int pv_stage = 32768; // M16 uses its own 16 KiB region for one prior PV.
    static_assert(M != 16 || (heads * dim * 4 <= pv_stage && pv_stage + heads * dim * 4 <= 65536));
    static constexpr int accum = 0, content = 65536, rotary = 114688, scratch = 163840;
    static constexpr int scales = 180224, query = scales + width * 4, bcast = query + heads * 4;
    static constexpr int sum_base = bcast + heads * 32, max_base = sum_base + heads * 2 * 4;
    static constexpr int alpha_base = max_base + heads * 2 * 4, step_base = alpha_base + heads * 2 * 4;
    static constexpr int step_lo_base = step_base + heads * 2 * 4;
    static constexpr int tmp_rows = step_lo_base + heads * 2 * 4;
    // A score/RoPE slot holds two 384-column planes or one 512-column PV result.
    static constexpr int tile_bytes = 2 * rows * width * 4;
    // Separate high and residual INT8 staging buffers overlap P_hi MTE3 DMA with P_lo conversion.
    static constexpr int p_hi_stage = 184320, p_lo_stage = 190464;
    static_assert(tmp_rows + 128 <= p_hi_stage);
    static_assert(p_lo_stage + rows * width <= 192 * 1024);

    template<int cols>
    static __aicore__ inline void row_mul(__ubuf__ float* dst, __ubuf__ float* src, __ubuf__ float* scalar) {
        for (int c = 0; c < cols; c += 64)
            vmul(dst + c, src + c, scalar, rows, 1, 1, 0, cols / 8, cols / 8, 1);
    }
    static __aicore__ inline void broadcast(__ubuf__ float* dst, __ubuf__ float* src) {
        vbrcb(reinterpret_cast<__ubuf__ uint32_t*>(dst), reinterpret_cast<__ubuf__ uint32_t*>(src), 1, 8, rows / 8);
    }
    static __aicore__ inline void valid_mask(__ubuf__ float* dst, int valid, float fill, int num_rows) {
        for (int c = 0; c < width; c += 64) {
            if (valid >= c + 64) continue;
            uint64_t mask = ~uint64_t(0);
            if (valid > c) mask <<= valid - c;
            set_vector_mask(0, mask);
            vector_dup(dst + c, fill, num_rows, 1, 1, width / 8, 0);
        }
        full_mask();
    }
    static __aicore__ inline void prepare_scores(uint64_t c, uint64_t r, uint64_t qs, uint64_t ks, uint64_t table,
                                                 uint32_t visible, uint32_t first, uint32_t head_stride) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
        if (first) {
            copy_gm_to_ubuf(ub<float>(query), reinterpret_cast<__gm__ float*>(qs), 0, tiles, rows / 8, (head_stride - rows) / 8, 0);
        }
        auto ids = reinterpret_cast<__gm__ const int32_t*>(table);
        const uint32_t pages = (visible + 63) / 64;
        for (int page = 0; page < 6; ++page) {
            uint64_t id = ids[page < pages ? page : pages - 1];
            copy_gm_to_ubuf(ub<float>(scales) + page * 64, reinterpret_cast<__gm__ float*>(ks) + id * 64, 0, 1, 8, 0, 0);
        }
        for (int part = 0; part < tiles; ++part) {
            int off = part * rows * width;
            copy_gm_to_ubuf(ub<int32_t>(content + part * tile_bytes), reinterpret_cast<__gm__ int32_t*>(c) + off, 0, 1, rows * width / 8, 0, 0);
            copy_gm_to_ubuf(ub<float>(content + part * tile_bytes + tile_bytes / 2), reinterpret_cast<__gm__ float*>(r) + off, 0, 1, rows * width / 8, 0, 0);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID3 + part);
        }
    }
    template<int tile>
    static __aicore__ inline void score_tile(uint64_t c, uint64_t r, uint64_t prob, uint64_t qs, uint64_t ks, uint64_t table, uint64_t pv,
                                             uint32_t visible, uint32_t first, uint32_t slot, float atten_scale, uint32_t head_stride, uint32_t causal_pair) {
        full_mask();
        if constexpr (tile == 0) {
            if constexpr (tiles == 2) prepare_scores(c, r, qs, ks, table, visible, first, head_stride);
        } else if (!first) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
            copy_gm_to_ubuf(ub<int32_t>(content), reinterpret_cast<__gm__ int32_t*>(pv), 0, 1, rows * dim / 8, 0, 0);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
        }
        int lane = tile * rows, off = lane * width;
        int now = (slot * heads + lane) * 4, prev = ((1 - slot) * heads + lane) * 4;
        int score_addr = content + tile * tile_bytes;
        int rope_addr = score_addr + tile_bytes / 2;
        auto score = ub<float>(score_addr);
        auto rope = ub<float>(rope_addr);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3 + tile);
        if (tile == 0) {
            if (visible < width) valid_mask(ub<float>(scales), visible, 0.f, 1);
            if (first) {
                Muls(tensor<float>(query, heads), tensor<float>(query, heads), atten_scale, heads);
                Duplicate(tensor<float>(sum_base + heads * 4, heads), 0.f, heads);
                Duplicate(tensor<float>(max_base + heads * 4, heads), -__builtin_inff(), heads);
                pipe_barrier(PIPE_V);
                broadcast(ub<float>(bcast), ub<float>(query));
                if constexpr (tiles == 2) broadcast(ub<float>(bcast) + rows * 8, ub<float>(query) + rows);
            }
        }
        vconv_s322f32(score, reinterpret_cast<__ubuf__ int32_t*>(score), rows * width / 64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(score, score, rope, rows * width / 64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        row_mul<width>(score, score, ub<float>(bcast) + lane * 8);
        pipe_barrier(PIPE_V);
        for (int col = 0; col < width; col += 64)
            vmul(score + col, score + col, ub<float>(scales) + col, rows, 1, 1, 1, width / 8, width / 8, 0);
        pipe_barrier(PIPE_V);
        uint32_t valid = visible - (tile == 0 ? causal_pair : 0);
        if (valid < width) valid_mask(score, valid, -__builtin_inff(), rows);
        pipe_barrier(PIPE_V);

        online_softmax_vexp<rows, width>(
            score_addr, scratch, sum_base + now, sum_base + prev,
            max_base + now, max_base + prev, alpha_base + now, tmp_rows, valid, first);

        for (int col = 0; col < width; col += 64)
            vmul(score + col, score + col, ub<float>(scales) + col, rows, 1, 1, 1, width / 8, width / 8, 0);
        // Use the dead RoPE tile for |P * kv_scale| so signed scales also quantize symmetrically.
        pipe_barrier(PIPE_V);
        vabs(rope, score, rows * width / 64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        auto t0 = ub<float>(scratch);
        auto t1 = ub<float>(scratch + 4096);
        auto t2 = ub<float>(scratch + 8192);
        vmax(t0, rope, rope + 64, rows, 1, 1, 1, 8, width / 8, width / 8);
        vmax(t1, rope + 128, rope + 192, rows, 1, 1, 1, 8, width / 8, width / 8);
        vmax(t2, rope + 256, rope + 320, rows, 1, 1, 1, 8, width / 8, width / 8);
        pipe_barrier(PIPE_V);
        vmax(t0, t0, t1, rows, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmax(t0, t0, t2, rows, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vcmax(ub<float>(tmp_rows), t0, rows, 1, 1, 8, Order_t::ONLY_VALUE);
        pipe_barrier(PIPE_V);
        auto high = tensor<float>(tmp_rows, rows);
        Muls(tensor<float>(step_base + now, rows), high, 1.f / 127.f, rows);
        Muls(tensor<float>(step_lo_base + now, rows), high, 1.f / 16129.f, rows);
        Maxs(high, high, 1.e-30f, rows);
        auto numerator = tensor<float>(tmp_rows + 64, rows);
        Duplicate(numerator, 127.f, rows);
        pipe_barrier(PIPE_V);
        Div(high, numerator, high, rows);
        pipe_barrier(PIPE_V);
        full_mask();
        broadcast(t0, ub<float>(tmp_rows));
        pipe_barrier(PIPE_V);
        row_mul<width>(score, score, t0);
        pipe_barrier(PIPE_V);
        auto halfs = ub<half>(scratch);
        vconv_f322f16(halfs, score, rows * width / 64, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);

        auto p_hi = ub<int8_t>(p_hi_stage);
        auto p_lo = ub<int8_t>(p_lo_stage);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
        set_vector_mask(~uint64_t(0), ~uint64_t(0));
        vconv_f162s8r(p_hi, halfs, rows * width / 128, 1, 1, 4, 8);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t*>(prob) + off, p_hi, 0, 1, rows * width / 32, 0, 0);
        full_mask();

        auto halfs_hi = reinterpret_cast<__ubuf__ half*>(rope);
        Cast(tensor<half>(rope_addr, rows * width), tensor<int8_t>(p_hi_stage, rows * width), RoundMode::CAST_NONE, rows * width);
        pipe_barrier(PIPE_V);
        set_vector_mask(~uint64_t(0), ~uint64_t(0));
        vsub(halfs, halfs, halfs_hi, rows * width / 128, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(halfs, halfs, static_cast<half>(127.f), rows * width / 128, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        if constexpr (tiles == 2 && tile == 0) {
            if (!first) SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
        vconv_f162s8r(p_lo, halfs, rows * width / 128, 1, 1, 4, 8);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
        copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t*>(prob) + 2 * heads * width + off, p_lo, 0, 1, rows * width / 32, 0, 0);
        full_mask();
        pipe_barrier(PIPE_V);
    }
    static __aicore__ inline void encode(uint32_t phase, uint64_t c, uint64_t r, uint64_t prob, uint64_t qs, uint64_t ks, uint64_t table, uint64_t pv,
                                         uint32_t visible, uint32_t first, uint32_t slot, float atten_scale, uint32_t head_stride, uint32_t causal_pair) {
        if (phase == 0) {
            if constexpr (tiles == 1) prepare_scores(c, r, qs, ks, table, visible, first, head_stride);
            else score_tile<0>(c, r, prob, qs, ks, table, pv, visible, first, slot, atten_scale, head_stride, causal_pair);
            return;
        }
        if constexpr (tiles == 1) {
            if (!first) {
                copy_gm_to_ubuf(ub<int32_t>(pv_stage), reinterpret_cast<__gm__ int32_t*>(pv), 0, 1, rows * dim / 8, 0, 0);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
            }
            score_tile<0>(c, r, prob, qs, ks, table, pv, visible, first, slot, atten_scale, head_stride, causal_pair);
        } else {
            score_tile<1>(c, r, prob, qs, ks, table, pv, visible, first, slot, atten_scale, head_stride, causal_pair);
        }
    }
    static __aicore__ inline void update(uint64_t pv, uint64_t out, uint64_t lse, uint32_t slot, uint32_t first, uint32_t last, uint32_t empty, uint32_t head_stride, uint32_t first_empty) {
        full_mask();
        if (empty) {
            Duplicate(tensor<float>(content, heads), __builtin_inff(), heads);
            Duplicate(tensor<int16_t>(rotary, heads * dim), int16_t(0), heads * dim);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
            copy_ubuf_to_gm(reinterpret_cast<__gm__ float*>(lse), ub<float>(content), 0, 1, heads / 8, 0, 0);
            copy_ubuf_to_gm(reinterpret_cast<__gm__ int16_t*>(out), ub<int16_t>(rotary), 0, tiles, rows * dim / 16, 0, (head_stride - rows) * dim / 16);
            pipe_barrier(PIPE_ALL);
            return;
        }
        constexpr int pv_bytes = rows * dim * 4;
        constexpr int pv_lo_stage = 131072;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
        copy_gm_to_ubuf(ub<int32_t>(pv_lo_stage), reinterpret_cast<__gm__ int32_t*>(pv) + 2 * heads * dim, 0, 1, rows * dim / 8, 0, 0);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID7);
        for (int tile = 0; tile < tiles; ++tile) {
            int off = tile * rows * dim;
            if (!last && tile == 0) continue;
            copy_gm_to_ubuf(ub<int32_t>(tiles == 1 ? pv_stage : content + tile * pv_bytes), reinterpret_cast<__gm__ int32_t*>(pv) + off, 0, 1, rows * dim / 8, 0, 0);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID3 + tile);
        }
        for (int tile = 0; tile < tiles; ++tile) {
            const int lane = tile * rows, off = lane * dim, now = (slot * heads + lane) * 4;
            auto update = ub<float>(tiles == 1 ? pv_stage : content + tile * pv_bytes);
            auto acc = ub<float>(accum) + off;
            auto factor = ub<float>(scratch);
            if (tile == 0 && !last) WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
            else WaitFlag<HardEvent::MTE2_V>(EVENT_ID3 + tile);
            if (tile > 0) {
                copy_gm_to_ubuf(ub<int32_t>(pv_lo_stage), reinterpret_cast<__gm__ int32_t*>(pv) + 2 * heads * dim + off, 0, 1, rows * dim / 8, 0, 0);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID7);
            }
            vconv_s322f32(update, reinterpret_cast<__ubuf__ int32_t*>(update), rows * dim / 64, 1, 1, 8, 8);
            broadcast(factor, ub<float>(step_base + now));
            broadcast(factor + 2 * rows * 8, ub<float>(step_lo_base + now));
            if (!first) broadcast(factor + rows * 8, ub<float>(alpha_base + now));
            pipe_barrier(PIPE_V);
            if (first) {
                row_mul<dim>(acc, update, factor);
            } else {
                row_mul<dim>(update, update, factor);
                row_mul<dim>(acc, acc, factor + rows * 8);
                pipe_barrier(PIPE_V);
                vadd(acc, acc, update, rows * dim / 64, 1, 1, 1, 8, 8, 8);
            }
            auto update_lo = ub<float>(pv_lo_stage);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID7);
            vconv_s322f32(update_lo, ub<int32_t>(pv_lo_stage), rows * dim / 64, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            row_mul<dim>(update_lo, update_lo, factor + 2 * rows * 8);
            pipe_barrier(PIPE_V);
            vadd(acc, acc, update_lo, rows * dim / 64, 1, 1, 1, 8, 8, 8);
            if (tile + 1 < tiles) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
            }
            if (last) {
                auto sums = tensor<float>(sum_base + now, rows), logs = tensor<float>(tmp_rows, rows);
                Ln(logs, sums, rows);
                pipe_barrier(PIPE_V);
                Add(logs, logs, tensor<float>(max_base + now, rows), rows);
                auto inverse = tensor<float>(tmp_rows + 64, rows);
                Duplicate(inverse, 1.f, rows);
                pipe_barrier(PIPE_V);
                Div(inverse, inverse, sums, rows);
                pipe_barrier(PIPE_V);
                full_mask();
                broadcast(factor, ub<float>(tmp_rows + 64));
                pipe_barrier(PIPE_V);
                row_mul<dim>(acc, acc, factor);
                pipe_barrier(PIPE_V);
                Cast(tensor<bfloat16_t>(content + tile * pv_bytes, rows * dim), tensor<float>(accum + off * 4, rows * dim), RoundMode::CAST_RINT, rows * dim);
                if (tile == 0 && first_empty) {
                    Duplicate(tensor<float>(tmp_rows, rows), __builtin_inff(), rows);
                    Duplicate(tensor<int16_t>(content + tile * pv_bytes, rows * dim), int16_t(0), rows * dim);
                    pipe_barrier(PIPE_V);
                }
                SetFlag<HardEvent::V_MTE3>(EVENT_ID3 + tile);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID3 + tile);
                copy_ubuf_to_gm(reinterpret_cast<__gm__ float*>(lse) + lane, ub<float>(tmp_rows), 0, 1, rows / 8, 0, 0);
                copy_ubuf_to_gm(reinterpret_cast<__gm__ int16_t*>(out) + tile * head_stride * dim, ub<int16_t>(content + tile * pv_bytes), 0, 1, rows * dim / 16, 0, 0);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID3 + tile);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID3 + tile);
            }
            pipe_barrier(PIPE_V);
        }
        if (last) pipe_barrier(PIPE_ALL); // Return UB and publish O/LSE to the enclosing TLE.
    }
};
} // namespace

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_mla_window_probability(uint32_t m, uint32_t phase, uint64_t c, uint64_t r, uint64_t prob, uint64_t qs, uint64_t ks, uint64_t table, uint64_t pv,
                                    uint32_t visible, uint32_t first, uint32_t slot, float atten_scale, uint32_t head_stride, uint32_t causal_pair) {
    if (m == 16) VectorWindow<16>::encode(phase, c, r, prob, qs, ks, table, pv, visible, first, slot, atten_scale, head_stride, causal_pair);
    else if (m == 32) VectorWindow<32>::encode(phase, c, r, prob, qs, ks, table, pv, visible, first, slot, atten_scale, head_stride, causal_pair);
    else VectorWindow<64>::encode(phase, c, r, prob, qs, ks, table, pv, visible, first, slot, atten_scale, head_stride, causal_pair);
}

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_mla_window_update(uint32_t m, uint64_t pv, uint64_t out, uint64_t lse, uint32_t slot, uint32_t first, uint32_t last, uint32_t empty, uint32_t head_stride, uint32_t first_empty) {
    if (m == 16) VectorWindow<16>::update(pv, out, lse, slot, first, last, empty, head_stride, first_empty);
    else if (m == 32) VectorWindow<32>::update(pv, out, lse, slot, first, last, empty, head_stride, first_empty);
    else VectorWindow<64>::update(pv, out, lse, slot, first, last, empty, head_stride, first_empty);
}
