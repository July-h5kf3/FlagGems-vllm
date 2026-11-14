; Copyright 2026 FlagOS Contributors
; SPDX-License-Identifier: Apache-2.0
; MP31 four-lane FP32 primitives. Each i64 holds two FP32 register values.

declare <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8>)

declare <4 x i8> @llvm.musa.f2e4m3.rn.bst4(<4 x float>)

declare <4 x float> @llvm.musa.fma.f.bst4.src12(float, <4 x float>, <4 x float>)

declare <4 x float> @llvm.musa.mul.f.bst4.src01(<4 x float>, <4 x float>)

declare <4 x float> @llvm.musa.mul.f.bst4.src1(float, <4 x float>)

declare <4 x float> @llvm.musa.mul.fhf.bst4.src01(<4 x half>, <4 x float>)

define protected float @gdn_norm4(i64 %0, i32 %1) alwaysinline nounwind readnone {
  %3 = icmp eq i32 %1, 0
  %4 = bitcast i64 %0 to <4 x bfloat>
  %5 = fpext <4 x bfloat> %4 to <4 x float>
  %6 = bitcast i64 %0 to <4 x half>
  %7 = fpext <4 x half> %6 to <4 x float>
  %8 = select i1 %3, <4 x float> %7, <4 x float> %5
  %9 = tail call contract <4 x float> @llvm.musa.mul.f.bst4.src01(<4 x float> %8, <4 x float> %8)
  %10 = shufflevector <4 x float> %9, <4 x float> undef, <2 x i32> <i32 1, i32 2>
  %11 = shufflevector <4 x float> %9, <4 x float> undef, <2 x i32> <i32 0, i32 3>
  %12 = fadd contract <2 x float> %10, %11
  %13 = shufflevector <2 x float> %12, <2 x float> poison, <2 x i32> <i32 1, i32 undef>
  %14 = fadd contract <2 x float> %12, %13
  %15 = extractelement <2 x float> %14, i64 0
  ret float %15
}

define i64 @gdn_normalized_pair(i64 %a0, float %a1, i32 %a2, i32 %a3) alwaysinline nounwind readnone {
%h = bitcast i64 %a0 to <4 x half>
  %b = bitcast i64 %a0 to <4 x bfloat>
  %isbf = icmp ne i32 %a2, 0
  %hv = fpext <4 x half> %h to <4 x float>
  %bv = fpext <4 x bfloat> %b to <4 x float>
  %q = select i1 %isbf,<4 x float> %bv,<4 x float> %hv
  %v=call <4 x float> @llvm.musa.mul.f.bst4.src1(float %a1, <4 x float> %q)
  %vv = bitcast <4 x float> %v to <2 x i64>
  %pair = extractelement <2 x i64> %vv, i32 %a3
  ret i64 %pair
}

define i64 @gdn_state_pair(i32 %a0, float %a1, float %a2, i32 %a3) alwaysinline nounwind readnone {
%raw=bitcast i32 %a0 to <4 x i8>
  %h=call <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8> %raw)
  %v= fpext <4 x half> %h to <4 x float>
  %scaled=call <4 x float> @llvm.musa.mul.f.bst4.src1(float %a1,<4 x float> %v)
  %prior=call <4 x float> @llvm.musa.mul.f.bst4.src1(float %a2,<4 x float> %scaled)
  %vv = bitcast <4 x float> %prior to <2 x i64>
  %pair = extractelement <2 x i64> %vv, i32 %a3
  ret i64 %pair
}

define float @gdn_dot4(i64 %a0, i64 %a1, i64 %a2, i64 %a3) alwaysinline nounwind readnone {
%xl = bitcast i64 %a0 to <2 x float>
  %xh = bitcast i64 %a1 to <2 x float>
  %x = shufflevector <2 x float> %xl, <2 x float> %xh, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %yl = bitcast i64 %a2 to <2 x float>
  %yh = bitcast i64 %a3 to <2 x float>
  %y = shufflevector <2 x float> %yl, <2 x float> %yh, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %z=call <4 x float> @llvm.musa.mul.f.bst4.src01(<4 x float> %x,<4 x float> %y)
  %lo=shufflevector <4 x float> %z,<4 x float> undef,<2 x i32> <i32 0,i32 2>
  %hi=shufflevector <4 x float> %z,<4 x float> undef,<2 x i32> <i32 1,i32 3>
  %t=fadd <2 x float> %lo,%hi
  %a=extractelement <2 x float> %t,i32 0
  %b=extractelement <2 x float> %t,i32 1
  %c=fadd float %a,%b
  ret float %c
}

define i64 @gdn_updated_pair(i64 %a0, i64 %a1, i64 %a2, i64 %a3, float %a4, i32 %a5) alwaysinline nounwind readnone {
%sl = bitcast i64 %a0 to <2 x float>
  %sh = bitcast i64 %a1 to <2 x float>
  %s = shufflevector <2 x float> %sl, <2 x float> %sh, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %kl = bitcast i64 %a2 to <2 x float>
  %kh = bitcast i64 %a3 to <2 x float>
  %k = shufflevector <2 x float> %kl, <2 x float> %kh, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %v=call <4 x float> @llvm.musa.fma.f.bst4.src12(float %a4,<4 x float> %k,<4 x float> %s)
  %vv = bitcast <4 x float> %v to <2 x i64>
  %pair = extractelement <2 x i64> %vv, i32 %a5
  ret i64 %pair
}

define float @gdn_absmax4(i64 %a0, i64 %a1) alwaysinline nounwind readnone {
%vl = bitcast i64 %a0 to <2 x float>
  %vh = bitcast i64 %a1 to <2 x float>
  %v = shufflevector <2 x float> %vl, <2 x float> %vh, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %vi=bitcast <4 x float> %v to <4 x i32>
  %mag=and <4 x i32> %vi,<i32 2147483647,i32 2147483647,i32 2147483647,i32 2147483647>
  %lo=shufflevector <4 x i32> %mag,<4 x i32> undef,<2 x i32> <i32 0,i32 2>
  %hi=shufflevector <4 x i32> %mag,<4 x i32> undef,<2 x i32> <i32 1,i32 3>
  %test=icmp ugt <2 x i32> %lo,%hi
  %t=select <2 x i1> %test,<2 x i32> %lo,<2 x i32> %hi
  %a=extractelement <2 x i32> %t,i32 0
  %b=extractelement <2 x i32> %t,i32 1
  %test2=icmp ugt i32 %a,%b
  %m=select i1 %test2,i32 %a,i32 %b
  %r=bitcast i32 %m to float
  ret float %r
}

define i32 @gdn_encode4(i64 %a0, i64 %a1, float %a2) alwaysinline nounwind readnone {
%sl = bitcast i64 %a0 to <2 x float>
  %sh = bitcast i64 %a1 to <2 x float>
  %s = shufflevector <2 x float> %sl, <2 x float> %sh, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %v=call <4 x float> @llvm.musa.mul.f.bst4.src1(float %a2,<4 x float> %s)
  %q=call <4 x i8> @llvm.musa.f2e4m3.rn.bst4(<4 x float> %v)
  %i=bitcast <4 x i8> %q to i32
  ret i32 %i
}

define i64 @gdn_scaled_query_pair(i64 %word,float %iq,float %scale,i32 %bf,i32 %which) alwaysinline nounwind readnone {
  %b=bitcast i64 %word to <4 x bfloat>
  %h=bitcast i64 %word to <4 x half>
  %bv=fpext <4 x bfloat> %b to <4 x float>
  %hv=fpext <4 x half> %h to <4 x float>
  %use_bf=icmp ne i32 %bf,0
  %v=select i1 %use_bf,<4 x float> %bv,<4 x float> %hv
  %norm=call <4 x float> @llvm.musa.mul.f.bst4.src1(float %iq,<4 x float> %v)
  %scaled=call <4 x float> @llvm.musa.mul.f.bst4.src1(float %scale,<4 x float> %norm)
  %parts=bitcast <4 x float> %scaled to <2 x i64>
  %r=extractelement <2 x i64> %parts,i32 %which
  ret i64 %r
}

; Decode four contiguous E4M3 values without applying a common channel scale.
define i64 @gdn_decode_half4(i32 %bits) alwaysinline nounwind readnone {
  %v = bitcast i32 %bits to <4 x i8>
  %f = call <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8> %v)
  %r = bitcast <4 x half> %f to i64
  ret i64 %r
}
