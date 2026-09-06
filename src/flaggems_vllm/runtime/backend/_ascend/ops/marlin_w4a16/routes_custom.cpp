// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;
template<class T> __aicore__ inline LocalTensor<T> Local(uint32_t off,uint32_t count) {
    TBuffAddr a{};a.dataLen=count*sizeof(T);a.bufferAddr=off;a.logicPos=static_cast<uint8_t>(TPosition::VECCALC);
    LocalTensor<T> t;t.SetAddr(a);return t;
}
extern "C" [aicore] __attribute__((always_inline)) void MRL_ENTRY(
    int64_t ip,int64_t pattern,int64_t rp,int64_t cp,int32_t pid,int64_t scratch) {
    PipeBarrier<PIPE_ALL>();
    for(int expert=pid;expert<MRL_E;expert+=MRL_GRID) {
    constexpr int B=4096;
    auto input=Local<int32_t>(scratch,B);
    auto values=Local<float>(scratch+4*B,B);
    auto indices=Local<float>(scratch+8*B,B);
    auto selected=Local<float>(scratch+12*B,B);
    auto mask=Local<uint8_t>(scratch+16*B,B/8);
    GlobalTensor<float> pg;pg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pattern));
    DataCopyExtParams pp{1,B*4,0,0,0};DataCopyPad(indices,pg,pp,DataCopyPadExtParams<float>{false,0,0,0});
    PipeBarrier<PIPE_ALL>();
    int count=0;
    for(int start=0;start<MRL_R;start+=B) {
        int valid=MRL_R-start;if(valid>B)valid=B;
        Duplicate(input,int32_t(-1),B);PipeBarrier<PIPE_ALL>();
        GlobalTensor<int32_t> ig;ig.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ip)+start);
        DataCopyExtParams qp{1,static_cast<uint32_t>(valid*4),0,0,0};
        DataCopyPad(input,ig,qp,DataCopyPadExtParams<int32_t>{true,0,static_cast<uint8_t>((8-valid%8)%8),-1});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(values,input,RoundMode::CAST_RINT,B);PipeBarrier<PIPE_V>();
        CompareScalar(mask,values,static_cast<float>(expert),CMPMODE::EQ,B);PipeBarrier<PIPE_V>();
        uint64_t found=0;GatherMaskParams gp{1,1,8,1};
        GatherMask(selected,indices,mask.ReinterpretCast<uint32_t>(),true,B,gp,found);PipeBarrier<PIPE_ALL>();
        found=get_rsvd_cnt();
        if(found) {
            int rounded=(static_cast<int>(found)+63)/64*64;
            Adds(selected,selected,static_cast<float>(start),rounded);PipeBarrier<PIPE_V>();
            Cast(selected.ReinterpretCast<int32_t>(),selected,RoundMode::CAST_RINT,rounded);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            GlobalTensor<int32_t> rg;rg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rp)+expert*MRL_R+count);
            DataCopyExtParams outcp{1,static_cast<uint32_t>(found*4),0,0,0};DataCopyPad(rg,selected.ReinterpretCast<int32_t>(),outcp);
            count+=found;PipeBarrier<PIPE_ALL>();
        }
    }
    auto count_out=Local<int32_t>(scratch+16*B+B/8,8);
    Duplicate(count_out,count,8);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<int32_t> cg;cg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cp)+expert);
    DataCopyExtParams last{1,4,0,0,0};DataCopyPad(cg,count_out,last);PipeBarrier<PIPE_ALL>();
    }
}
