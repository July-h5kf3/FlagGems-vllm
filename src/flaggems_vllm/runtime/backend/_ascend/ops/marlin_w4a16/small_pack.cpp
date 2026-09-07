// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;
template<class T> __aicore__ inline LocalTensor<T> Local(uint32_t off,uint32_t count) {
    TBuffAddr a{};a.dataLen=count*sizeof(T);a.bufferAddr=off;a.logicPos=static_cast<uint8_t>(TPosition::VECCALC);
    LocalTensor<T> t;t.SetAddr(a);return t;
}
extern "C" [aicore] __attribute__((always_inline)) void MRL_ENTRY(
    int64_t xp,int64_t ip,int64_t ep,int64_t vp,int64_t cp,int64_t op,int32_t pid,int64_t scratch) {
    PipeBarrier<PIPE_ALL>();
    auto buffer=Local<bfloat16_t>(scratch,MRL_BM*MRL_K);
    auto meta=Local<int32_t>(scratch+MRL_BM*MRL_K*2,8);
    for(int route=pid;route<MRL_R;route+=MRL_GRID) {
        int expert=reinterpret_cast<__gm__ int32_t*>(ip)[route];
        for(int offset=0;offset<MRL_BM*MRL_K;offset+=8192) {
            int count=MRL_BM*MRL_K-offset;if(count>8192)count=8192;
            Duplicate(buffer[offset],bfloat16_t(0.0f),count);
        }
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<bfloat16_t> x;x.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(xp)+(route/MRL_TOPK)*MRL_K);
        DataCopyExtParams in{1,MRL_K*2,0,0,0};
        DataCopyPad(buffer,x,in,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<bfloat16_t> out;out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(op)+route*MRL_BM*MRL_K);
        DataCopyExtParams out_params{1,MRL_BM*MRL_K*2,0,0,0};DataCopyPad(out,buffer,out_params);
        Duplicate(meta,expert,8);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<int32_t> e;e.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ep)+route);
        DataCopyExtParams one{1,4,0,0,0};DataCopyPad(e,meta,one);PipeBarrier<PIPE_ALL>();
        Duplicate(meta,route*MRL_BM,8);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<int32_t> inv;inv.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(vp)+route);
        DataCopyPad(inv,meta,one);PipeBarrier<PIPE_ALL>();
        if(route==0) {
            Duplicate(meta,MRL_R*MRL_BM,8);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            GlobalTensor<int32_t> active;active.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cp));
            DataCopyPad(active,meta,one);PipeBarrier<PIPE_ALL>();
        }
    }
}
