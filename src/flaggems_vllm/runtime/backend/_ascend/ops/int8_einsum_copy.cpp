// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

template <typename T, unsigned Rank> struct MemRef {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_fg_blockdiag_gmajor(MemRef<__gm__ int8_t, 2> *src,
                                      uint16_t actual_rows, uint16_t stride,
                                      uint16_t block_rows, uint16_t groups,
                                      MemRef<__cbuf__ int8_t, 4> *dst) {
  const uint32_t expanded_rows = block_rows * groups;
  const uint32_t bytes = expanded_rows * groups * 128;
  auto target = dst->aligned + dst->offset;
  const uint64_t config = 1 | (static_cast<uint64_t>(bytes / 32) << 16);
  create_cbuf_matrix(reinterpret_cast<__cbuf__ uint32_t *>(target), config,
                     static_cast<uint32_t>(0));
  pipe_barrier(PIPE_MTE2);
  for (uint16_t group = 0; group < groups; ++group) {
    const uint32_t offset = group * block_rows * 32 + group * 4 * expanded_rows * 32;
    copy_gm_to_cbuf_multi_nd2nz_b8(
        target + offset, src->aligned + src->offset + group * 128,
        0, 1, actual_rows, 128, 0, stride, expanded_rows, 1, 1);
  }
}

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_fg_fullk_nd2nz(MemRef<__gm__ int8_t, 2> *src,
                                 uint16_t rows, uint16_t columns,
                                 uint16_t stride,
                                 MemRef<__cbuf__ int8_t, 4> *dst) {
  copy_gm_to_cbuf_multi_nd2nz_b8(dst->aligned + dst->offset,
                                src->aligned + src->offset, 0, 1, rows,
                                columns, 0, stride, rows, 1, 1);
}
