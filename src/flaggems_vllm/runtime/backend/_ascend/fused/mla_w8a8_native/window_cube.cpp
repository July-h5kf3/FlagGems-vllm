/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Adapted from IfaMatmulFullQuant at 1e878a0cdf15eba39382def8cdd599e8233c11b8.
// Fused pairs of public 64-row ND cache pages feed one 128-column QK MMAD tile.
// The corresponding two L1 tiles live through the group's PV stage.
#include "kernel_operator.h"
using namespace AscendC;

template<typename T>
__aicore__ inline LocalTensor<T> local(TPosition pos, uint32_t offset) {
    return LocalTensor<T>(pos, offset, 65536 / sizeof(T));
}
template<typename T>
__aicore__ inline GlobalTensor<T> global(uint64_t addr) {
    GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(addr));
    return tensor;
}

// ND subset of CopyGmToL1 and CopyInMm1{B,BRope}ToL1ForPA. The L1 C0
// width depends on the operand type (32 for INT8 and 16 for BF16).
template<typename T>
__aicore__ inline void load_nd(LocalTensor<T> dst, uint64_t src,
                                uint32_t rows, uint32_t width, uint32_t tile_rows) {
    Nd2NzParams params{};
    params.ndNum = 1;
    params.nValue = rows;
    params.dValue = width;
    params.srcDValue = width;
    params.dstNzC0Stride = tile_rows;
    params.dstNzNStride = 1;
    params.srcNdMatrixStride = 0;
    params.dstNzMatrixStride = 0;
    DataCopy(dst, global<T>(src), params);
}
template<typename DT>
__aicore__ inline void LoadDataL0A(LocalTensor<DT>& aL0Tensor,
    const LocalTensor<DT>& aL1Tensor, uint32_t kIdx, uint32_t kSplitSize, uint32_t mSize, uint32_t subkSize)
{
    LocalTensor<DT> srcTensor = aL1Tensor[mSize * kSplitSize * kIdx];

    LoadData3DParamsV2<DT> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = mSize / 16; // Hin=M1=8
    loadData3DParams.l1W = 16; // Win=M0
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果
    // SetLoadToA0Params
    loadData3DParams.mExtension = mSize; // M
    loadData3DParams.kExtension = subkSize; // K
    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;
    loadData3DParams.channelSize = subkSize; // Cin=K
    LoadData(aL0Tensor, srcTensor, loadData3DParams);
}

template<typename DT>
__aicore__ inline void LoadDataMm1B(LocalTensor<DT> &l0Tensor,
    LocalTensor<DT> &l1Tensor, uint32_t idx, uint32_t kSplitSize, uint32_t kSize, uint32_t nSize)
{
    // N 方向全载
    LocalTensor<DT> srcTensor = l1Tensor[nSize * kSplitSize * idx];

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = (nSize + 15) / 16 * kSize / (32 / sizeof(DT));

    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0Tensor, srcTensor, loadData2DParams);
}

__aicore__ inline void LoadDataMm2B(LocalTensor<int8_t> &bL0Tensor, LocalTensor<int8_t> &bL1Tensor,
                                                           uint32_t idx, uint32_t nSize, uint32_t subkSize,
                                                           uint32_t kSplitSize, uint32_t kSize)
{
    // L1 128 * 256; L0 64 * 128
    uint32_t kloops = subkSize / 32;

    LocalTensor<int8_t> srcTensor = bL1Tensor[idx * kSplitSize * 32 / sizeof(int8_t)];
    for (uint32_t i = 0; i < kloops; i++) {
        LoadData2dTransposeParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = nSize / (32 / sizeof(int8_t));

        loadData2DParams.srcStride = kSize / 32;
        loadData2DParams.dstGap = 1;
        loadData2DParams.dstFracGap = 0;

        LocalTensor<int8_t> tmpSrcTensor;
        tmpSrcTensor = srcTensor[i * 32 * (32 / sizeof(int8_t))];

        LoadDataWithTranspose(bL0Tensor[i * 32 * nSize], tmpSrcTensor, loadData2DParams);
    }
}

// Each MMAD consumes one bank. Keep the previous bank available to M until
// it has completed, as in the vendor M_MTE1 / MTE1_M ping-pong protocol.
template<typename A, typename C>
__aicore__ inline void multiply(LocalTensor<C> dst, LocalTensor<A> a, LocalTensor<A> b,
                                 uint32_t m, uint32_t n, uint32_t k, bool init) {
    SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
    MmadParams params{};
    params.m = m;
    params.n = n;
    params.k = k;
    params.cmatrixInitVal = init;
    params.cmatrixSource = false;
    if ((m / 16) * (n / 16) < 10) PipeBarrier<PIPE_M>();
    Mmad(dst, a, b, params);
}
template<typename T>
__aicore__ inline void publish(uint64_t dst, LocalTensor<T> acc, uint32_t rows, uint32_t cols, uint32_t stride) {
    SetFlag<HardEvent::M_FIX>(EVENT_ID3);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID3);
    FixpipeParamsV220 fix{};
    fix.nSize = cols;
    fix.mSize = rows;
    fix.srcStride = rows;
    fix.dstStride = stride;
    fix.ndNum = 1;
    Fixpipe(global<T>(dst), acc, fix);
}
constexpr uint32_t query_base = 0, qr_base = 32768, key_base = 40960;
constexpr uint32_t kr_base = key_base + 6 * 65536, prob_base = kr_base + 3 * 16384;

// One token per L1 page pair. Refill can start as soon as its last L1 reader
// has copied both pages to L0; the MMAD and the subsequent Fixpipe may continue.
__aicore__ inline void prefetch(uint64_t k,uint64_t kr,uint64_t table,uint32_t remaining,uint32_t slot,uint32_t pair) {
    auto ids=reinterpret_cast<__gm__ const int32_t*>(table);
    auto kv1=local<int8_t>(TPosition::A1,key_base+(slot*3+pair)*65536);
    auto rope1=local<bfloat16_t>(TPosition::A1,kr_base+pair*16384);
    for (uint32_t page=0; page<2; ++page) {
        uint32_t p=pair*2+page;
        uint64_t id=ids[p<remaining ? p : remaining-1];
        load_nd(kv1[page*64*32],k+id*32768,64,512,128);
        load_nd(rope1[page*64*16],kr+id*8192,64,64,128);
    }
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3+pair);
}

__aicore__ inline void refill(uint64_t k,uint64_t kr,uint64_t table,uint32_t remaining,uint32_t slot,uint32_t pair) {
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3+pair);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3+pair);
    prefetch(k,kr,table,remaining,slot,pair);
}

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_mla_window_qk(uint32_t m, uint64_t q, uint64_t qr, uint64_t k, uint64_t kr, uint64_t table,
                            uint64_t c, uint64_t r, uint32_t remaining, uint32_t first, uint32_t slot, uint32_t head_stride) {
    constexpr uint32_t width = 384;
    auto q1 = local<int8_t>(TPosition::A1, query_base);
    auto qr1 = local<bfloat16_t>(TPosition::A1, qr_base);
    if (first) {
        if (m == 16 || head_stride == m / 4) {
            load_nd(q1, q, m, 512, m);
            load_nd(qr1, qr, m, 64, m);
        } else {
            for (uint32_t tile = 0; tile < 4; ++tile) {
                uint32_t source = (tile % 2) * head_stride + (tile / 2) * (m / 4);
                load_nd(q1[tile*(m/4)*32], q + source*512, m/4, 512, m);
                load_nd(qr1[tile*(m/4)*16], qr + source*128, m/4, 64, m);
            }
        }
    }
    if (first) {
        for (uint32_t bank=0; bank<2; ++bank) {
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + bank);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0 + bank);
        }
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0); // P can be written before the first PV.
        for (uint32_t pair=0; pair<3; ++pair) prefetch(k,kr,table,remaining,slot,pair);
    }
    for (uint32_t pair = 0; pair < 3; ++pair) {
        auto kv1 = local<int8_t>(TPosition::A1, key_base + (slot*3 + pair)*65536);
        auto rope1 = local<bfloat16_t>(TPosition::A1, kr_base + pair*16384);
        auto ic = local<int32_t>(TPosition::CO1, (pair%2)*65536);
        auto rc = local<float>(TPosition::CO1, (pair%2)*65536+32768);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3 + pair);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0 + pair%2);
        for (uint32_t part = 0; part < 2; ++part) {
            uint32_t instruction = 3*pair + part, bank = instruction%2;
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + bank);
            auto a = local<int8_t>(TPosition::A2, bank*32768);
            auto b = local<int8_t>(TPosition::B2, bank*32768);
            LoadDataL0A(a, q1, part, 256, m, 256);
            LoadDataMm1B(b, kv1, part, 256, 256, 128);
            multiply(ic, a, b, m, 128, 256, part == 0);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + bank);
        }
        publish(c + pair*512, ic, m, 128, width);
        uint32_t bank = pair % 2;
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + bank);
        auto a = local<bfloat16_t>(TPosition::A2, bank*32768);
        auto b = local<bfloat16_t>(TPosition::B2, bank*32768);
        LoadDataL0A(a, qr1, 0, 0, m, 64);
        LoadDataMm1B(b, rope1, 0, 0, 64, 128);
        if (first && remaining>6) refill(k,kr,table+24,remaining-6,1,pair);
        multiply(rc, a, b, m, 128, 64, true);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + bank);
        publish(r + pair*512, rc, m, 128, width);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0 + pair%2);
    }

}
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_mla_window_pv(uint32_t m, uint64_t p, uint64_t dst, uint64_t k, uint64_t kr, uint64_t table,
                            uint32_t slot, uint32_t remaining, uint32_t last) {
    auto p1 = local<int8_t>(TPosition::A1, prob_base);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    load_nd(p1, p, m, 384, m);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID7);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID7);
    for (uint32_t col = 0; col < 2; ++col) {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0 + col);
        auto c = local<int32_t>(TPosition::CO1, col*65536);
        for (uint32_t pair = 0; pair < 3; ++pair) {
            uint32_t instruction = col*3+pair, bank = instruction%2;
            auto v = local<int8_t>(TPosition::A1, key_base + (slot*3 + pair)*65536 + col*256*128);
            auto a = local<int8_t>(TPosition::A2, bank*32768);
            auto b = local<int8_t>(TPosition::B2, bank*32768);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + bank);
            LoadDataL0A(a, p1, pair, 128, m, 128);
            LoadDataMm2B(b, v, 0, 256, 128, 128, 128);
            if (col==1 && remaining) refill(k,kr,table,remaining,slot,pair);
            if (instruction==5) SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            multiply(c, a, b, m, 256, 128, pair == 0);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + bank);
        }
        publish(dst + col*256*4, c, m, 256, 512);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0 + col);
    }
    if (last) {
        for (uint32_t bank=0; bank<2; ++bank) {
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + bank);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0 + bank);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }
}
