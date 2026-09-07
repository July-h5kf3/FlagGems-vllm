// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;
template<class T> __aicore__ inline LocalTensor<T> Local(uint32_t off,uint32_t count) {
    TBuffAddr a{};a.dataLen=count*sizeof(T);a.bufferAddr=off;a.logicPos=static_cast<uint8_t>(TPosition::VECCALC);
    LocalTensor<T> t;t.SetAddr(a);return t;
}
extern "C" [aicore] __attribute__((always_inline)) void MRL_ENTRY(
    int64_t xp,int64_t rp,int64_t cp,int64_t offp,int64_t ep,int64_t op,int32_t pid,int64_t scratch) {
    PipeBarrier<PIPE_ALL>();
    for(int tile=pid;tile<MRL_TILES;tile+=MRL_GRID) {
    int expert=reinterpret_cast<__gm__ int32_t*>(ep)[tile];
    if(expert<0)continue;
    int begin=reinterpret_cast<__gm__ int32_t*>(offp)[expert];
    int count=reinterpret_cast<__gm__ int32_t*>(cp)[expert];
    auto buffer=Local<bfloat16_t>(scratch,16*MRL_K);
    for(int row=0;row<MRL_BM;row+=16) {
        if(tile*MRL_BM+row+16-begin>count) {
            for(int offset=0;offset<16*MRL_K;offset+=8192) {
                int len=16*MRL_K-offset;if(len>8192)len=8192;
                Duplicate(buffer[offset],bfloat16_t(0.0f),len);
            }
            PipeBarrier<PIPE_ALL>();
        }
        for(int ri=0;ri<16;++ri) {
            int local=tile*MRL_BM+row+ri-begin;
            if(local<count) {
                int route=reinterpret_cast<__gm__ int32_t*>(rp)[expert*MRL_R+local];
                GlobalTensor<bfloat16_t> x;x.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(xp)+(route/MRL_TOPK)*MRL_K);
                DataCopyExtParams cp{1,MRL_K*2,0,0,0};
                DataCopyPad(buffer[ri*MRL_K],x,cp,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
            }
        }
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<bfloat16_t> out;out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(op)+(tile*MRL_BM+row)*MRL_K);
        DataCopyExtParams cp{1,16*MRL_K*2,0,0,0};DataCopyPad(out,buffer,cp);PipeBarrier<PIPE_ALL>();
    }
    }
}
