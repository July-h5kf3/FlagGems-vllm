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

import triton
import triton.language as tl

from flaggems_vllm.ops.flash_mla import HAS_TLE_FLASH_MLA as HAS_TLE
from flaggems_vllm.ops.flash_mla import tle

if HAS_TLE:

    @triton.jit
    def _publish_p_fp8_sw64_coupled_stmatrix(s_p, p):
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
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",  # noqa: E501
            args=[p, base_u32],
            dtype=tl.uint32,
            is_pure=False,
            pack=32,
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
    def _vtranspose_fp8_64x128_plain(
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
    def _vtranspose_fp8_64x128_kperm(
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
    def _vtranspose_fp8_64x128(
        s_src,
        s_dst,
        dst_row: tl.constexpr,
        permute_k: tl.constexpr,
    ):
        if permute_k:
            return _vtranspose_fp8_64x128_kperm(s_src, s_dst, dst_row)
        return _vtranspose_fp8_64x128_plain(s_src, s_dst, dst_row)

else:
    _publish_p_fp8_sw64_coupled_stmatrix = None
    _zero_invalid_fp8_rows_sw128_x4 = None
    _vtranspose_fp8_64x128_plain = None
    _vtranspose_fp8_64x128_kperm = None
    _vtranspose_fp8_64x128 = None
