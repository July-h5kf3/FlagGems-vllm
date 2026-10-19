; Copyright 2026 FlagOS Contributors
; SPDX-License-Identifier: Apache-2.0

declare void @llvm.mxc.arrive(i32) convergent nounwind
declare void @llvm.mxc.barrier.inst() convergent nounwind
declare void @llvm.mxc.barrier.shared() convergent nounwind

define i32 @flash_mla_int8_wait_all() alwaysinline convergent nounwind {
  call void @llvm.mxc.arrive(i32 64)
  call void @llvm.mxc.barrier.inst()
  ret i32 0
}

define i32 @flash_mla_int8_block_barrier() alwaysinline convergent nounwind {
  call void @llvm.mxc.barrier.shared()
  call void @llvm.mxc.barrier.inst()
  ret i32 0
}
