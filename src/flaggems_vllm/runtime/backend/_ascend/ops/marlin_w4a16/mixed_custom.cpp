// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;
template<class T> __aicore__ inline LocalTensor<T> Local(uint32_t off,uint32_t count,TPosition pos=TPosition::VECCALC) {
    TBuffAddr a{};a.dataLen=count*sizeof(T);a.bufferAddr=off;a.logicPos=static_cast<uint8_t>(pos);
    LocalTensor<T> t;t.SetAddr(a);return t;
}

#if defined(__DAV_C220_CUBE__)
extern "C" [aicore] __attribute__((always_inline)) void MRL_CUBE_ENTRY(
    int64_t ap,int64_t wp,int64_t ep,int64_t op,int32_t pid) {
    PipeBarrier<PIPE_ALL>();
    constexpr int BK=128;
    constexpr int AK_CAP=((512*1024-MRL_BN*BK*2)/(MRL_BM*2)/BK)*BK;
    constexpr int AK=MRL_BM<=32?(MRL_K<AK_CAP?MRL_K:AK_CAP):BK;
    constexpr int BANKS=(2*(MRL_BN>MRL_BM?MRL_BN:MRL_BM)*BK*2<=65536)?2:1;
    auto a1=Local<bfloat16_t>(0,MRL_BM*AK,TPosition::A1);
    auto b1=Local<bfloat16_t>(MRL_BM*AK*2,MRL_BN*BK,TPosition::B1);
    auto a2=Local<bfloat16_t>(0,BANKS*MRL_BM*BK,TPosition::A2);
    auto b2=Local<bfloat16_t>(0,BANKS*MRL_BN*BK,TPosition::B2);
    auto c0=Local<float>(0,MRL_BM*MRL_BN,TPosition::CO1);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    int iteration=0;
    for(int task=pid;task<MRL_TASKS;task+=MRL_GRID) {
        int tile=task/(MRL_N/MRL_BN),col=task%(MRL_N/MRL_BN)*MRL_BN;
        int expert=reinterpret_cast<__gm__ int32_t*>(ep)[tile];
        if(expert<0)continue;
        int row=tile*MRL_BM;
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        for(int ki=0;ki<MRL_K/BK;++ki) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            CrossCoreWaitFlag(2);
            GlobalTensor<bfloat16_t> ag,bg;
            ag.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ap)+row*MRL_K+ki*BK);
            bg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wp)+(pid*MRL_STAGES+iteration%MRL_STAGES)*MRL_BN*BK);
            Nd2NzParams pa;pa.ndNum=1;pa.nValue=MRL_BM;pa.dValue=BK;pa.srcNdMatrixStride=0;pa.srcDValue=MRL_K;
            pa.dstNzC0Stride=MRL_BM;pa.dstNzNStride=1;pa.dstNzMatrixStride=0;
            Nd2NzParams pb=pa;pb.nValue=MRL_BN;pb.dstNzC0Stride=MRL_BN;pb.srcDValue=BK;
            if(ki%(AK/BK)==0) {
                pa.dValue=(MRL_K-ki*BK)<AK?(MRL_K-ki*BK):AK;DataCopy(a1,ag,pa);
            }
            DataCopy(b1,bg,pb);
            CrossCoreSetFlag<2,PIPE_MTE2>(3);++iteration;
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            int bank=ki%BANKS;auto ev=bank?EVENT_ID1:EVENT_ID0;
            WaitFlag<HardEvent::M_MTE1>(ev);
            for(int mi=0;mi<MRL_BM/16;++mi) {
                LoadData2DParams ld;ld.repeatTimes=BK/16;ld.srcStride=MRL_BM/16;
                LoadData(a2[bank*MRL_BM*BK+mi*BK*16],a1[(ki%(AK/BK))*BK*MRL_BM+mi*256],ld);
            }
            LoadData2DParams ld;ld.repeatTimes=(BK/16)*(MRL_BN/16);ld.srcStride=1;
            LoadData(b2[bank*MRL_BN*BK],b1,ld);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE1_M>(ev);WaitFlag<HardEvent::MTE1_M>(ev);
            MmadParams mm;mm.m=MRL_BM;mm.n=MRL_BN;mm.k=BK;mm.cmatrixInitVal=ki==0;
            Mmad(c0,a2[bank*MRL_BM*BK],b2[bank*MRL_BN*BK],mm);
            SetFlag<HardEvent::M_MTE1>(ev);
        }
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        GlobalTensor<bfloat16_t> og;og.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(op)+row*MRL_N+col);
        FixpipeParamsV220 fp;fp.nSize=MRL_BN;fp.mSize=MRL_BM;fp.srcStride=MRL_BM;fp.dstStride=MRL_N;fp.quantPre=QuantMode_t::F322BF16;
        Fixpipe(og,c0,fp);SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);PipeBarrier<PIPE_ALL>();
}

#endif
#if defined(__DAV_C220_VEC__)
extern "C" [aicore] __attribute__((always_inline)) void MRL_VEC_ENTRY(
    int64_t wp,int64_t sp,int64_t fp,int64_t ep,int64_t op,int32_t pid,int32_t sub,int64_t scratch) {
    PipeBarrier<PIPE_ALL>();
    constexpr int L=MRL_VBN*128;
    auto q=Local<uint16_t>(scratch,L/4);
    auto scales=Local<bfloat16_t>(scratch+L/2,MRL_VBN);
    auto sf=Local<float>(scratch+L/2+256,MRL_VBN);
    auto scale_rows=Local<float>(scratch+L/2+768,MRL_VBN*8);
    auto h=Local<half>(scratch+L,L);
    auto v=Local<float>(scratch+3*L,L);
    auto result=h.ReinterpretCast<bfloat16_t>();
    int iteration=0;
    for(int task=pid;task<MRL_TASKS;task+=MRL_GRID) {
        int tile=task/(MRL_N/MRL_BN);
        int pn=(task%(MRL_N/MRL_BN))*2+sub;
        int expert=reinterpret_cast<__gm__ int32_t*>(ep)[tile];
        if(expert<0)continue;
        for(int kb=0;kb<MRL_K/128;++kb) {
        GlobalTensor<uint16_t> wg;wg.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(wp)+(expert*(MRL_K/128)+kb)*MRL_N*32+pn*MRL_VBN*32);
        GlobalTensor<bfloat16_t> sg;sg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sp)+(expert*(MRL_K/128)+kb)*MRL_N+pn*MRL_VBN);
        DataCopyExtParams qp{1,L/2,0,0,0};DataCopyExtParams spc{1,MRL_VBN*2,0,0,0};
        DataCopyPad(q,wg,qp,DataCopyPadExtParams<uint16_t>{false,0,0,0});
        DataCopyPad(scales,sg,spc,DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(h,q.ReinterpretCast<int4b_t>(),RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();

        Cast(sf,scales,RoundMode::CAST_NONE,MRL_VBN);PipeBarrier<PIPE_V>();
        int fast=1;
        auto flag=reinterpret_cast<__gm__ int32_t*>(fp)+(expert*(MRL_K/128)+kb)*(MRL_N/32)+pn*(MRL_VBN/32);
        for(int i=0;i<MRL_VBN/32;++i)fast&=flag[i];
        BrcbRepeatParams bc;bc.dstBlkStride=1;bc.dstRepStride=8;
        if(fast) {
            auto hs=scales.ReinterpretCast<half>();
            auto hr=scale_rows.ReinterpretCast<half>();
            Cast(hs,sf,RoundMode::CAST_RINT,MRL_VBN);PipeBarrier<PIPE_V>();
            Brcb(hr,hs,MRL_VBN/8,bc);PipeBarrier<PIPE_V>();
            BinaryRepeatParams sr;sr.dstBlkStride=8;sr.src0BlkStride=8;sr.src1BlkStride=1;
            sr.dstRepStride=1;sr.src0RepStride=1;sr.src1RepStride=0;
            for(int ri=0;ri<MRL_VBN;ri+=8)Mul(h[ri*128],h[ri*128],hr[ri*16],uint64_t(128),8,sr);
            PipeBarrier<PIPE_V>();Cast(v,h,RoundMode::CAST_NONE,L);
        } else {
            Cast(v,h,RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();
            Brcb(scale_rows,sf,MRL_VBN/8,bc);PipeBarrier<PIPE_V>();
            BinaryRepeatParams sr;sr.dstBlkStride=16;sr.src0BlkStride=16;sr.src1BlkStride=1;
            sr.dstRepStride=1;sr.src0RepStride=1;sr.src1RepStride=0;
            for(int ri=0;ri<MRL_VBN;ri+=8)Mul(v[ri*128],v[ri*128],scale_rows[ri*8],uint64_t(64),16,sr);
        }
        PipeBarrier<PIPE_V>();Cast(result,v,RoundMode::CAST_RINT,L);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if(iteration>=MRL_STAGES)CrossCoreWaitFlag(3);
            GlobalTensor<bfloat16_t> og;
            og.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(op)+(pid*MRL_STAGES+iteration%MRL_STAGES)*MRL_BN*128+sub*MRL_VBN*128);
            DataCopyExtParams outcp{1,L*2,0,0,0};DataCopyPad(og,result,outcp);
            CrossCoreSetFlag<2,PIPE_MTE3>(2);
            ++iteration;PipeBarrier<PIPE_ALL>();
        }
    }
    for(int i=0;i<(iteration<MRL_STAGES?iteration:MRL_STAGES);++i)CrossCoreWaitFlag(3);
    PipeBarrier<PIPE_ALL>();
}
#endif
