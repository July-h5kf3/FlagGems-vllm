# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# flake8: noqa: E501,F841

from __future__ import annotations

import triton
import triton.language as tl

from flaggems_vllm.ops.flash_mla_fp8.common import HAS_TLE, tle

if HAS_TLE:

    @triton.jit
    def _publish_p_fp8_sw64_cuda_stmatrix(s_p, p):
        """CUDA save_rPb_to_sP: swap uint16 packs 1/2, then two x4 STSM."""
        raw = p.to(tl.uint8, bitcast=True).to(tl.uint32)
        base = tle.gpu.local_ptr(s_p, (0, 0))
        base_u32 = tl.inline_asm_elementwise(
            asm="mov.u32 $0, $1;",
            constraints="=r,r",
            args=[base],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b32 tid, warp_off, row_off, common, tmp, phys0, phys1, addr0, addr1;\n"
                ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3, lane, src0, src1, selector, lo, hi;\n"
                ".reg .pred take_hi;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 warp_off, tid, 96;\n"
                "shl.b32 warp_off, warp_off, 5;\n"
                "and.b32 row_off, tid, 15;\n"
                "shl.b32 row_off, row_off, 6;\n"
                "or.b32 common, warp_off, row_off;\n"
                "and.b32 tmp, tid, 16;\n"
                "or.b32 common, common, tmp;\n"
                "and.b32 lane, tid, 31;\n"
                "and.b32 src0, lane, 28;\n"
                "and.b32 tmp, lane, 1;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "add.u32 src0, src0, tmp;\n"
                "add.u32 src1, src0, 1;\n"
                "and.b32 selector, lane, 2;\n"
                "setp.ne.u32 take_hi, selector, 0;\n"
                "selp.u32 selector, 0x7632, 0x5410, take_hi;\n"
                "and.b32 a0, $32, 255;\n"
                "and.b32 tmp, $33, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 a0, a0, tmp;\n"
                "and.b32 tmp, $36, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 a0, a0, tmp;\n"
                "and.b32 tmp, $37, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 a0, a0, tmp;\n"
                "shfl.sync.idx.b32 lo, a0, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, a0, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 a0, lo, hi, selector;\n"
                "and.b32 a1, $34, 255;\n"
                "and.b32 tmp, $35, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 a1, a1, tmp;\n"
                "and.b32 tmp, $38, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 a1, a1, tmp;\n"
                "and.b32 tmp, $39, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 a1, a1, tmp;\n"
                "shfl.sync.idx.b32 lo, a1, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, a1, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 a1, lo, hi, selector;\n"
                "and.b32 a2, $40, 255;\n"
                "and.b32 tmp, $41, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 a2, a2, tmp;\n"
                "and.b32 tmp, $44, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 a2, a2, tmp;\n"
                "and.b32 tmp, $45, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 a2, a2, tmp;\n"
                "shfl.sync.idx.b32 lo, a2, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, a2, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 a2, lo, hi, selector;\n"
                "and.b32 a3, $42, 255;\n"
                "and.b32 tmp, $43, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 a3, a3, tmp;\n"
                "and.b32 tmp, $46, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 a3, a3, tmp;\n"
                "and.b32 tmp, $47, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 a3, a3, tmp;\n"
                "shfl.sync.idx.b32 lo, a3, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, a3, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 a3, lo, hi, selector;\n"
                "and.b32 b0, $48, 255;\n"
                "and.b32 tmp, $49, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 b0, b0, tmp;\n"
                "and.b32 tmp, $52, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 b0, b0, tmp;\n"
                "and.b32 tmp, $53, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 b0, b0, tmp;\n"
                "shfl.sync.idx.b32 lo, b0, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, b0, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 b0, lo, hi, selector;\n"
                "and.b32 b1, $50, 255;\n"
                "and.b32 tmp, $51, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 b1, b1, tmp;\n"
                "and.b32 tmp, $54, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 b1, b1, tmp;\n"
                "and.b32 tmp, $55, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 b1, b1, tmp;\n"
                "shfl.sync.idx.b32 lo, b1, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, b1, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 b1, lo, hi, selector;\n"
                "and.b32 b2, $56, 255;\n"
                "and.b32 tmp, $57, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 b2, b2, tmp;\n"
                "and.b32 tmp, $60, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 b2, b2, tmp;\n"
                "and.b32 tmp, $61, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 b2, b2, tmp;\n"
                "shfl.sync.idx.b32 lo, b2, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, b2, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 b2, lo, hi, selector;\n"
                "and.b32 b3, $58, 255;\n"
                "and.b32 tmp, $59, 255;\n"
                "shl.b32 tmp, tmp, 8;\n"
                "or.b32 b3, b3, tmp;\n"
                "and.b32 tmp, $62, 255;\n"
                "shl.b32 tmp, tmp, 16;\n"
                "or.b32 b3, b3, tmp;\n"
                "and.b32 tmp, $63, 255;\n"
                "shl.b32 tmp, tmp, 24;\n"
                "or.b32 b3, b3, tmp;\n"
                "shfl.sync.idx.b32 lo, b3, src0, 0x1f, 0xffffffff;\n"
                "shfl.sync.idx.b32 hi, b3, src1, 0x1f, 0xffffffff;\n"
                "prmt.b32 b3, lo, hi, selector;\n"
                "shr.u32 tmp, common, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 phys0, common, tmp;\n"
                "add.u32 common, common, 32;\n"
                "shr.u32 tmp, common, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 phys1, common, tmp;\n"
                "add.u32 addr0, $64, phys0;\n"
                "add.u32 addr1, $64, phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr0], {a0, a1, a2, a3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr1], {b0, b1, b2, b3};\n"
                "fence.proxy.async.shared::cta;\n"
                "mov.u32 $0, $32;\n"
                "}"
            ),
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[raw, base_u32],
            dtype=tl.uint32,
            is_pure=False,
            pack=32,
        )

    @triton.jit
    def _publish_p_fp8_sw64_cuda_native_coupled_stmatrix(s_p, p):
        """CUDA-native P publication; V repack carries the matching K permutation."""
        base = tle.gpu.local_ptr(s_p, (0, 0))
        base_u32 = tl.inline_asm_elementwise(
            asm="mov.u32 $0, $1;",
            constraints="=r,r",
            args=[base],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15;\n"
                ".reg .b32 tid, warp_off, row_off, common, tmp, phys0, phys1, addr0, addr1;\n"
                ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 warp_off, tid, 96;\n"
                "shl.b32 warp_off, warp_off, 5;\n"
                "and.b32 row_off, tid, 15;\n"
                "shl.b32 row_off, row_off, 6;\n"
                "or.b32 common, warp_off, row_off;\n"
                "and.b32 tmp, tid, 16;\n"
                "or.b32 common, common, tmp;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h0, $33, $32;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h1, $35, $34;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h2, $37, $36;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h3, $39, $38;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h4, $41, $40;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h5, $43, $42;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h6, $45, $44;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h7, $47, $46;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h8, $49, $48;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h9, $51, $50;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h10, $53, $52;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h11, $55, $54;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h12, $57, $56;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h13, $59, $58;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h14, $61, $60;\n"
                "cvt.rn.satfinite.e4m3x2.f32 h15, $63, $62;\n"
                "mov.b32 a0, {h0, h2};\n"
                "mov.b32 a1, {h1, h3};\n"
                "mov.b32 a2, {h4, h6};\n"
                "mov.b32 a3, {h5, h7};\n"
                "mov.b32 b0, {h8, h10};\n"
                "mov.b32 b1, {h9, h11};\n"
                "mov.b32 b2, {h12, h14};\n"
                "mov.b32 b3, {h13, h15};\n"
                "shr.u32 tmp, common, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 phys0, common, tmp;\n"
                "add.u32 common, common, 32;\n"
                "shr.u32 tmp, common, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 phys1, common, tmp;\n"
                "add.u32 addr0, $64, phys0;\n"
                "add.u32 addr1, $64, phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr0], {a0, a1, a2, a3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr1], {b0, b1, b2, b3};\n"
                "fence.proxy.async.shared::cta;\n"
                "mov.u32 $0, $64;\n"
                "}"
            ),
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[p, base_u32],
            dtype=tl.uint32,
            is_pure=False,
            pack=32,
        )

    @triton.jit
    def _load_qrope_rs_fragment(s_qr, k_tile: tl.constexpr):
        # Map the physical register/lane/warp ownership bits of a 1024-wide
        # arange into the WGMMA K16 register-A layout using views only.
        physical = tl.arange(0, 1024).to(tl.bfloat16)
        ownership_bits = tl.reshape(physical, (2, 2, 2, 2, 2, 2, 2, 2, 2, 2))
        logical_bits = tl.permute(ownership_bits, (0, 1, 8, 2, 3, 4, 7, 5, 6, 9))
        carrier = tl.reshape(logical_bits, (64, 16))
        tile = carrier.to(tl.int32) * 0 + k_tile

        # Keep the shared operand visible to allocation/liveness while using
        # inline PTX only for the permitted pointer move and LDSM path.
        base = tle.gpu.local_ptr(s_qr, (0, 0))
        base_u32 = tl.inline_asm_elementwise(
            asm="mov.u32 $0, $1;",
            constraints="=r,r",
            args=[base],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b32 raw<4>;\n"
                ".reg .b32 lane, tid, x, y, tmp, off, smem;\n"
                ".reg .b32 lane16, group, src0, src1;\n"
                ".reg .b32 v00, v01, v20, v21, v02, v03, v22, v23;\n"
                ".reg .pred upper;\n"
                "mov.u32 lane, %laneid;\n"
                "mov.u32 tid, %tid.x;\n"
                "mov.u32 smem, $16;\n"
                "shl.b32 off, tid, 6;\n"
                "and.b32 off, off, 8064;\n"
                "shl.b32 x, tid, 2;\n"
                "and.b32 x, x, 56;\n"
                "shl.b32 y, tid, 3;\n"
                "and.b32 y, y, 8;\n"
                "shl.b32 tmp, $8, 4;\n"
                "or.b32 y, y, tmp;\n"
                "xor.b32 x, x, y;\n"
                "shl.b32 x, x, 1;\n"
                "add.u32 off, off, x;\n"
                "add.u32 off, smem, off;\n"
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                "{raw0, raw1, raw2, raw3}, [off];\n"
                "and.b32 lane16, lane, 15;\n"
                "and.b32 src0, lane16, 3;\n"
                "and.b32 group, lane16, 12;\n"
                "shl.b32 group, group, 1;\n"
                "add.u32 src0, src0, group;\n"
                "add.u32 src1, src0, 4;\n"
                "setp.ge.u32 upper, lane, 16;\n"
                "shfl.sync.idx.b32 v00, raw0, src0, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v01, raw1, src0, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v20, raw2, src0, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v21, raw3, src0, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v02, raw0, src1, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v03, raw1, src1, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v22, raw2, src1, 31, 0xffffffff;\n"
                "shfl.sync.idx.b32 v23, raw3, src1, 31, 0xffffffff;\n"
                "selp.b32 $0, v01, v00, upper;\n"
                "selp.b32 $1, v21, v20, upper;\n"
                "selp.b32 $2, v03, v02, upper;\n"
                "selp.b32 $3, v23, v22, upper;\n"
                "}"
            ),
            constraints=(
                "=r,=r,=r,=r," "r,r,r,r," "r,r,r,r,r,r,r,r," "r,r,r,r,r,r,r,r"
            ),
            args=[carrier, tile, base_u32],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=8,
        )

    @triton.jit
    def _zero_invalid_fp8_rows_sw128(s_src, valid_tokens):
        """Zero invalid rows of one SW128 64x128 FP8 tile with 16B stores."""
        carrier = tl.arange(0, 128).to(tl.uint32)
        src_base = tle.gpu.local_ptr(s_src, (0, 0))
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .pred invalid;\n"
                ".reg .b32 tid, row, col, logical, swz, addr, z;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 tid, tid, 127;\n"
                "shr.u32 row, tid, 1;\n"
                "and.b32 col, tid, 1;\n"
                "shl.b32 col, col, 6;\n"
                "setp.ge.u32 invalid, row, $3;\n"
                "mov.u32 z, 0;\n"
                "shl.b32 logical, row, 7;\n"
                "add.u32 logical, logical, col;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 addr, logical, swz;\n"
                "add.u32 addr, $2, addr;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 logical, logical, 16;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 addr, logical, swz;\n"
                "add.u32 addr, $2, addr;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 logical, logical, 16;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 addr, logical, swz;\n"
                "add.u32 addr, $2, addr;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 logical, logical, 16;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 addr, logical, swz;\n"
                "add.u32 addr, $2, addr;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "mov.u32 $0, $1;\n"
                "}"
            ),
            constraints="=r,r,r,r",
            args=[carrier, src_base, valid_tokens],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _zero_invalid_fp8_rows_sw128_x4(
        s_src0,
        s_src1,
        s_src2,
        s_src3,
        valid_tokens,
    ):
        """Zero the same invalid rows in four SW128 64x128 FP8 tiles.

        The four content tiles share row validity and SW128 addressing.  Keep
        the four 16B stores per tile, but compute the predicate and swizzled
        byte offset only once.
        """
        carrier = tl.arange(0, 128).to(tl.uint32)
        src0_base = tle.gpu.local_ptr(s_src0, (0, 0))
        src1_base = tle.gpu.local_ptr(s_src1, (0, 0))
        src2_base = tle.gpu.local_ptr(s_src2, (0, 0))
        src3_base = tle.gpu.local_ptr(s_src3, (0, 0))
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .pred invalid;\n"
                ".reg .b32 tid, lane, warp, row, col, logical, swz, off, addr, z;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 tid, tid, 127;\n"
                "and.b32 lane, tid, 31;\n"
                "shr.u32 warp, tid, 5;\n"
                "mov.u32 row, lane;\n"
                "shl.b32 col, warp, 4;\n"
                "setp.ge.u32 invalid, row, $6;\n"
                "mov.u32 z, 0;\n"
                "shl.b32 logical, row, 7;\n"
                "add.u32 logical, logical, col;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 off, logical, swz;\n"
                "add.u32 addr, $2, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $3, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $4, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $5, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 logical, logical, 64;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 off, logical, swz;\n"
                "add.u32 addr, $2, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $3, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $4, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $5, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 row, row, 32;\n"
                "setp.ge.u32 invalid, row, $6;\n"
                "shl.b32 logical, row, 7;\n"
                "add.u32 logical, logical, col;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 off, logical, swz;\n"
                "add.u32 addr, $2, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $3, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $4, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $5, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 logical, logical, 64;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 off, logical, swz;\n"
                "add.u32 addr, $2, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $3, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $4, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $5, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "mov.u32 $0, $1;\n"
                "}"
            ),
            constraints="=r,r,r,r,r,r,r",
            args=[
                carrier,
                src0_base,
                src1_base,
                src2_base,
                src3_base,
                valid_tokens,
            ],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _cuda_vtranspose_fp8_64x128_plain(
        s_src,
        s_dst,
        dst_row: tl.constexpr,
    ):
        """CUDA-authority SW128 -> SW64 FP8 transpose for one 64x128 tile."""
        carrier = tl.arange(0, 128).to(tl.uint32)
        src_base = tle.gpu.local_ptr(s_src, (0, 0))
        dst_base = tle.gpu.local_ptr(s_dst, (dst_row, 0))
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b32 tid, lane, warp, src_row, tmp, tmp2;\n"
                ".reg .b32 src_log, src_phys, src_addr0, src_addr1;\n"
                ".reg .b32 dst_row_r, dst_col, dst_log0, dst_log1;\n"
                ".reg .b32 dst_phys0, dst_phys1, dst_addr0, dst_addr1;\n"
                ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
                ".reg .b32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 tid, tid, 127;\n"
                "and.b32 lane, tid, 31;\n"
                "shr.u32 warp, tid, 5;\n"
                # CUDA's LDSM/STSM register order presents source-row bits as
                # [b1,b3,b2,b0] to TLE's logical SW64 view.  Apply the inverse
                # [b3,b1,b2,b0] mapping at the load boundary so PV observes
                # the same logical transpose as the tensor path.
                "and.b32 src_row, lane, 17;\n"
                "and.b32 tmp, lane, 8;\n"
                "shr.u32 tmp, tmp, 2;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 2;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 4;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "shl.b32 src_log, src_row, 7;\n"
                "shl.b32 tmp, warp, 4;\n"
                "add.u32 src_log, src_log, tmp;\n"
                "shr.u32 tmp, src_log, 7;\n"
                "and.b32 tmp, tmp, 7;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 src_phys, src_log, tmp;\n"
                "add.u32 src_addr0, $2, src_phys;\n"
                "add.u32 src_addr1, src_addr0, 4096;\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{a0, a1, a2, a3}, [src_addr0];\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{b0, b1, b2, b3}, [src_addr1];\n"
                "prmt.b32 c0, a0, a1, 0x6420;\n"
                "prmt.b32 c1, a0, a1, 0x7531;\n"
                "prmt.b32 c2, a2, a3, 0x6420;\n"
                "prmt.b32 c3, a2, a3, 0x7531;\n"
                "prmt.b32 d0, b0, b1, 0x6420;\n"
                "prmt.b32 d1, b0, b1, 0x7531;\n"
                "prmt.b32 d2, b2, b3, 0x6420;\n"
                "prmt.b32 d3, b2, b3, 0x7531;\n"
                "and.b32 dst_row_r, lane, 7;\n"
                "shl.b32 dst_row_r, dst_row_r, 1;\n"
                "shr.u32 tmp, lane, 3;\n"
                "and.b32 tmp, tmp, 1;\n"
                "add.u32 dst_row_r, dst_row_r, tmp;\n"
                "shl.b32 tmp, warp, 4;\n"
                "add.u32 dst_row_r, dst_row_r, tmp;\n"
                "shr.u32 dst_col, lane, 4;\n"
                "and.b32 dst_col, dst_col, 1;\n"
                "shl.b32 dst_col, dst_col, 4;\n"
                "shl.b32 dst_log0, dst_row_r, 6;\n"
                "add.u32 dst_log0, dst_log0, dst_col;\n"
                "add.u32 dst_log1, dst_log0, 32;\n"
                "shr.u32 tmp, dst_log0, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 dst_phys0, dst_log0, tmp;\n"
                "shr.u32 tmp2, dst_log1, 7;\n"
                "and.b32 tmp2, tmp2, 3;\n"
                "shl.b32 tmp2, tmp2, 4;\n"
                "xor.b32 dst_phys1, dst_log1, tmp2;\n"
                "add.u32 dst_addr0, $3, dst_phys0;\n"
                "add.u32 dst_addr1, $3, dst_phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr0], {c0, c1, c2, c3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr1], {d0, d1, d2, d3};\n"
                "add.u32 src_log, src_log, 64;\n"
                "shr.u32 tmp, src_log, 7;\n"
                "and.b32 tmp, tmp, 7;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 src_phys, src_log, tmp;\n"
                "add.u32 src_addr0, $2, src_phys;\n"
                "add.u32 src_addr1, src_addr0, 4096;\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{a0, a1, a2, a3}, [src_addr0];\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{b0, b1, b2, b3}, [src_addr1];\n"
                "prmt.b32 c0, a0, a1, 0x6420;\n"
                "prmt.b32 c1, a0, a1, 0x7531;\n"
                "prmt.b32 c2, a2, a3, 0x6420;\n"
                "prmt.b32 c3, a2, a3, 0x7531;\n"
                "prmt.b32 d0, b0, b1, 0x6420;\n"
                "prmt.b32 d1, b0, b1, 0x7531;\n"
                "prmt.b32 d2, b2, b3, 0x6420;\n"
                "prmt.b32 d3, b2, b3, 0x7531;\n"
                "add.u32 dst_log0, dst_log0, 4096;\n"
                "add.u32 dst_log1, dst_log1, 4096;\n"
                "shr.u32 tmp, dst_log0, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 dst_phys0, dst_log0, tmp;\n"
                "shr.u32 tmp2, dst_log1, 7;\n"
                "and.b32 tmp2, tmp2, 3;\n"
                "shl.b32 tmp2, tmp2, 4;\n"
                "xor.b32 dst_phys1, dst_log1, tmp2;\n"
                "add.u32 dst_addr0, $3, dst_phys0;\n"
                "add.u32 dst_addr1, $3, dst_phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr0], {c0, c1, c2, c3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr1], {d0, d1, d2, d3};\n"
                "mov.u32 $0, $1;\n"
                "}"
            ),
            constraints="=r,r,r,r",
            args=[carrier, src_base, dst_base],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _cuda_vtranspose_fp8_64x128_kperm(
        s_src,
        s_dst,
        dst_row: tl.constexpr,
    ):
        """CUDA-authority SW128 -> SW64 FP8 transpose for one 64x128 tile."""
        carrier = tl.arange(0, 128).to(tl.uint32)
        src_base = tle.gpu.local_ptr(s_src, (0, 0))
        dst_base = tle.gpu.local_ptr(s_dst, (dst_row, 0))
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b32 tid, lane, warp, src_row, tmp, tmp2;\n"
                ".reg .b32 src_log, src_phys, src_addr0, src_addr1;\n"
                ".reg .b32 dst_row_r, dst_col, dst_log0, dst_log1;\n"
                ".reg .b32 dst_phys0, dst_phys1, dst_addr0, dst_addr1;\n"
                ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
                ".reg .b32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 tid, tid, 127;\n"
                "and.b32 lane, tid, 31;\n"
                "shr.u32 warp, tid, 5;\n"
                # CUDA's LDSM/STSM register order presents source-row bits as
                # [b1,b3,b2,b0] to TLE's logical SW64 view.  Apply the inverse
                # [b3,b1,b2,b0] mapping at the load boundary so PV observes
                # the same logical transpose as the tensor path.
                "and.b32 src_row, lane, 17;\n"
                "and.b32 tmp, lane, 8;\n"
                "shr.u32 tmp, tmp, 2;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 2;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 4;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                # Direct CUDA STSM presents P to TLE as dest <- source pi,
                # pi=[0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] per K16.
                # Load V from pi(dest) as well, preserving the dot product
                # while removing the publication-side cross-lane shuffle.
                "mov.u32 tmp2, src_row;\n"
                "and.b32 src_row, tmp2, 17;\n"
                "and.b32 tmp, tmp2, 4;\n"
                "shr.u32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, tmp2, 8;\n"
                "shr.u32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, tmp2, 2;\n"
                "shl.b32 tmp, tmp, 2;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "shl.b32 src_log, src_row, 7;\n"
                "shl.b32 tmp, warp, 4;\n"
                "add.u32 src_log, src_log, tmp;\n"
                "shr.u32 tmp, src_log, 7;\n"
                "and.b32 tmp, tmp, 7;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 src_phys, src_log, tmp;\n"
                "add.u32 src_addr0, $2, src_phys;\n"
                "add.u32 src_addr1, src_addr0, 4096;\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{a0, a1, a2, a3}, [src_addr0];\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{b0, b1, b2, b3}, [src_addr1];\n"
                "prmt.b32 c0, a0, a1, 0x6420;\n"
                "prmt.b32 c1, a0, a1, 0x7531;\n"
                "prmt.b32 c2, a2, a3, 0x6420;\n"
                "prmt.b32 c3, a2, a3, 0x7531;\n"
                "prmt.b32 d0, b0, b1, 0x6420;\n"
                "prmt.b32 d1, b0, b1, 0x7531;\n"
                "prmt.b32 d2, b2, b3, 0x6420;\n"
                "prmt.b32 d3, b2, b3, 0x7531;\n"
                "and.b32 dst_row_r, lane, 7;\n"
                "shl.b32 dst_row_r, dst_row_r, 1;\n"
                "shr.u32 tmp, lane, 3;\n"
                "and.b32 tmp, tmp, 1;\n"
                "add.u32 dst_row_r, dst_row_r, tmp;\n"
                "shl.b32 tmp, warp, 4;\n"
                "add.u32 dst_row_r, dst_row_r, tmp;\n"
                "shr.u32 dst_col, lane, 4;\n"
                "and.b32 dst_col, dst_col, 1;\n"
                "shl.b32 dst_col, dst_col, 4;\n"
                "shl.b32 dst_log0, dst_row_r, 6;\n"
                "add.u32 dst_log0, dst_log0, dst_col;\n"
                "add.u32 dst_log1, dst_log0, 32;\n"
                "shr.u32 tmp, dst_log0, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 dst_phys0, dst_log0, tmp;\n"
                "shr.u32 tmp2, dst_log1, 7;\n"
                "and.b32 tmp2, tmp2, 3;\n"
                "shl.b32 tmp2, tmp2, 4;\n"
                "xor.b32 dst_phys1, dst_log1, tmp2;\n"
                "add.u32 dst_addr0, $3, dst_phys0;\n"
                "add.u32 dst_addr1, $3, dst_phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr0], {c0, c1, c2, c3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr1], {d0, d1, d2, d3};\n"
                "add.u32 src_log, src_log, 64;\n"
                "shr.u32 tmp, src_log, 7;\n"
                "and.b32 tmp, tmp, 7;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 src_phys, src_log, tmp;\n"
                "add.u32 src_addr0, $2, src_phys;\n"
                "add.u32 src_addr1, src_addr0, 4096;\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{a0, a1, a2, a3}, [src_addr0];\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                "{b0, b1, b2, b3}, [src_addr1];\n"
                "prmt.b32 c0, a0, a1, 0x6420;\n"
                "prmt.b32 c1, a0, a1, 0x7531;\n"
                "prmt.b32 c2, a2, a3, 0x6420;\n"
                "prmt.b32 c3, a2, a3, 0x7531;\n"
                "prmt.b32 d0, b0, b1, 0x6420;\n"
                "prmt.b32 d1, b0, b1, 0x7531;\n"
                "prmt.b32 d2, b2, b3, 0x6420;\n"
                "prmt.b32 d3, b2, b3, 0x7531;\n"
                "add.u32 dst_log0, dst_log0, 4096;\n"
                "add.u32 dst_log1, dst_log1, 4096;\n"
                "shr.u32 tmp, dst_log0, 7;\n"
                "and.b32 tmp, tmp, 3;\n"
                "shl.b32 tmp, tmp, 4;\n"
                "xor.b32 dst_phys0, dst_log0, tmp;\n"
                "shr.u32 tmp2, dst_log1, 7;\n"
                "and.b32 tmp2, tmp2, 3;\n"
                "shl.b32 tmp2, tmp2, 4;\n"
                "xor.b32 dst_phys1, dst_log1, tmp2;\n"
                "add.u32 dst_addr0, $3, dst_phys0;\n"
                "add.u32 dst_addr1, $3, dst_phys1;\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr0], {c0, c1, c2, c3};\n"
                "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "[dst_addr1], {d0, d1, d2, d3};\n"
                "mov.u32 $0, $1;\n"
                "}"
            ),
            constraints="=r,r,r,r",
            args=[carrier, src_base, dst_base],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _cuda_vtranspose_fp8_64x128(
        s_src,
        s_dst,
        dst_row: tl.constexpr,
        permute_k: tl.constexpr,
    ):
        if permute_k:
            return _cuda_vtranspose_fp8_64x128_kperm(s_src, s_dst, dst_row)
        return _cuda_vtranspose_fp8_64x128_plain(s_src, s_dst, dst_row)

else:
    _publish_p_fp8_sw64_cuda_stmatrix = None
    _publish_p_fp8_sw64_cuda_native_coupled_stmatrix = None
    _load_qrope_rs_fragment = None
    _zero_invalid_fp8_rows_sw128 = None
    _zero_invalid_fp8_rows_sw128_x4 = None
    _cuda_vtranspose_fp8_64x128_plain = None
    _cuda_vtranspose_fp8_64x128_kperm = None
    _cuda_vtranspose_fp8_64x128 = None
