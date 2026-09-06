// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;
template<class T> __aicore__ inline LocalTensor<T> Local(uint32_t off,uint32_t count) {
    TBuffAddr a{};a.dataLen=count*sizeof(T);a.bufferAddr=off;a.logicPos=static_cast<uint8_t>(TPosition::VECCALC);
    LocalTensor<T> t;t.SetAddr(a);return t;
}
extern "C" [aicore] __attribute__((always_inline)) void MRL_ENTRY(
    int64_t ap,int64_t pp,int64_t ip,int64_t op,int32_t pid,int64_t scratch) {
    PipeBarrier<PIPE_ALL>();
    constexpr int B=MRL_B;
    auto ah=Local<bfloat16_t>(scratch,B);
    auto bh=Local<bfloat16_t>(scratch+B*2,B);
    auto a=Local<float>(scratch+B*4,B);
    auto b=Local<float>(scratch+B*8,B);
    auto c=Local<float>(scratch+B*12,B);
    auto d=Local<float>(scratch+B*16,B);
    for(int tile=pid;tile<MRL_TOTAL/B;tile+=MRL_GRID) {
        int row=tile/(MRL_K/B),col=tile%(MRL_K/B)*B;
        if constexpr (MRL_KIND==0) {
            GlobalTensor<bfloat16_t> ag,bg;
            ag.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ap)+row*MRL_K*2+col);
            bg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ap)+row*MRL_K*2+col+MRL_K);
            DataCopyExtParams cp{1,B*2,0,0,0};
            DataCopyPad(ah,ag,cp,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
            DataCopyPad(bh,bg,cp,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(a,ah,RoundMode::CAST_NONE,B);Cast(b,bh,RoundMode::CAST_NONE,B);PipeBarrier<PIPE_V>();
            Muls(c,a,-1.0f,B);PipeBarrier<PIPE_V>();Exp(c,c,B);PipeBarrier<PIPE_V>();
            Adds(c,c,1.0f,B);PipeBarrier<PIPE_V>();Div(d,a,c,B);PipeBarrier<PIPE_V>();
            Mul(d,d,b,B);PipeBarrier<PIPE_V>();
        } else {
            Duplicate(d,0.0f,B);PipeBarrier<PIPE_ALL>();
            for(int t=0;t<MRL_TOPK;++t) {
                int ri=row*MRL_TOPK+t;
                int pos=MRL_INV?reinterpret_cast<__gm__ int32_t*>(ip)[ri]:ri;
                float prob=reinterpret_cast<__gm__ float*>(pp)[ri];
                GlobalTensor<bfloat16_t> ag;ag.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ap)+pos*MRL_K+col);
                DataCopyExtParams cp{1,B*2,0,0,0};DataCopyPad(ah,ag,cp,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Cast(a,ah,RoundMode::CAST_NONE,B);PipeBarrier<PIPE_V>();
                Muls(a,a,prob,B);PipeBarrier<PIPE_V>();Add(d,d,a,B);PipeBarrier<PIPE_ALL>();
            }
        }
        Cast(ah,d,RoundMode::CAST_RINT,B);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<bfloat16_t> og;og.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(op)+row*MRL_K+col);
        DataCopyExtParams cp{1,B*2,0,0,0};DataCopyPad(og,ah,cp);PipeBarrier<PIPE_ALL>();
    }
}
