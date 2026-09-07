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
    if constexpr(MRL_KIND==1)SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
    int live_rows=MRL_TOTAL/MRL_K;
    if constexpr(MRL_KIND==0 && MRL_ACTIVE_E>=0)live_rows=reinterpret_cast<__gm__ int32_t*>(ip)[MRL_ACTIVE_E];
    int total=(live_rows*MRL_K+B-1)/B;
    for(int tile=pid;tile<total;tile+=MRL_GRID) {
        int row,col;
        if constexpr(MRL_KIND==0 && MRL_RP>1) {row=tile*MRL_RP;col=0;}
        else {row=tile/(MRL_K/B);col=tile%(MRL_K/B)*B;}
        int row_count=MRL_RP;
        if constexpr(MRL_KIND==0 && MRL_RP>1) {
            if(live_rows-row<row_count)row_count=live_rows-row;
        }
        if constexpr (MRL_KIND==0) {
            GlobalTensor<bfloat16_t> ag,bg;
            ag.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ap)+row*MRL_K*2+col);
            bg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ap)+row*MRL_K*2+col+MRL_K);
            DataCopyExtParams cp{static_cast<uint16_t>(row_count),(B/MRL_RP)*2,MRL_RP>1?MRL_K*2:0,0,0};
            DataCopyPad(ah,ag,cp,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
            DataCopyPad(bh,bg,cp,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(a,ah,RoundMode::CAST_NONE,B);Cast(b,bh,RoundMode::CAST_NONE,B);PipeBarrier<PIPE_V>();
            Muls(c,a,-1.0f,B);PipeBarrier<PIPE_V>();Exp(c,c,B);PipeBarrier<PIPE_V>();
            Adds(c,c,1.0f,B);PipeBarrier<PIPE_V>();Div(d,a,c,B);PipeBarrier<PIPE_V>();
            Mul(d,d,b,B);PipeBarrier<PIPE_V>();
        } else {
            Duplicate(d,0.0f,B);PipeBarrier<PIPE_V>();

            int first=row*MRL_TOPK;
            int first_pos=MRL_INV?reinterpret_cast<__gm__ int32_t*>(ip)[first]:first;
            GlobalTensor<bfloat16_t> initial;initial.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ap)+first_pos*MRL_K+col);
            DataCopyExtParams input_cp{1,B*2,0,0,0};
            DataCopyPad(ah,initial,input_cp,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            for(int t=0;t<MRL_TOPK;++t) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if(t>0)WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                if(t+1<MRL_TOPK) {
                    int next=row*MRL_TOPK+t+1;
                    int pos=MRL_INV?reinterpret_cast<__gm__ int32_t*>(ip)[next]:next;
                    GlobalTensor<bfloat16_t> ag;ag.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ap)+pos*MRL_K+col);
                    if(t%2==0)DataCopyPad(bh,ag,input_cp,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
                    else DataCopyPad(ah,ag,input_cp,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                }
                float prob=reinterpret_cast<__gm__ float*>(pp)[row*MRL_TOPK+t];
                if(t%2==0)Cast(a,ah,RoundMode::CAST_NONE,B);
                else Cast(a,bh,RoundMode::CAST_NONE,B);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                PipeBarrier<PIPE_V>();Axpy(d,a,prob,B);PipeBarrier<PIPE_V>();
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        }
        if constexpr(MRL_KIND==1)WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
        auto output_buffer=MRL_KIND==1?c.ReinterpretCast<bfloat16_t>():ah;
        Cast(output_buffer,d,RoundMode::CAST_RINT,B);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<bfloat16_t> og;og.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(op)+row*MRL_K+col);
        DataCopyExtParams cp{1,static_cast<uint32_t>(B/MRL_RP*row_count*2),0,0,0};DataCopyPad(og,output_buffer,cp);
        if constexpr(MRL_KIND==1) {SetFlag<HardEvent::MTE3_V>(EVENT_ID3);PipeBarrier<PIPE_V>();}
        else PipeBarrier<PIPE_ALL>();
    }
    if constexpr(MRL_KIND==1)WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
    PipeBarrier<PIPE_ALL>();
}
