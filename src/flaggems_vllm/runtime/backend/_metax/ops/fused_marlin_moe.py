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

"""MetaX W4A16 INT4 and W8A16 FP8 SwiGLU MoE.

The NVIDIA fast paths dequantize with Hopper PTX (lop3 / bf16x2) and require
SM90 for FP8. MetaX reports a CUDA-compatible capability, so those paths must
not be selected here. This kernel follows vLLM-metax's Triton WNA16 GEMM with
plain output-major weights and no Marlin repack: UINT4B8 packs the even K
index in the low nibble with group scales in the activation dtype; FP8 E4M3FN
stores one byte per weight with FP32 or activation-dtype group scales. FP8
bytes are decoded in registers with integer bit moves; each K tile holds one
scale group, so the scale multiplies the partial product. When there are at
most half as many routes as experts, FP8 runs per-route GEMV kernels that fuse
SiLU and the top-k sum instead of padding every route to an MMA tile. Dense
128-row FP8 batches dequantize w1 into a transient BF16/FP16 buffer of at most
FP8_DEQUANT_MAX_BYTES and run gate/up as a plain GEMM.
"""

from enum import Enum
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP8_E4M3, QUANT_TYPE_UINT4B8
from flaggems_vllm.ops.fused_marlin_moe import (
    fused_marlin_moe as generic_fused_marlin_moe,
)
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_no_tle,
    moe_align_block_size_small_grouped,
)
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.runtime.backend._metax.fused.moe_sum import moe_sum as device_moe_sum

NUM_STAGES = 4
NUM_WARPS = 4
FUSE_GATE_UP_MAX_TOKENS = 4
LARGE_EXPERT_MIN_COUNT = 128
LARGE_EXPERT_BLOCK16_MAX_TOKENS = 448
LARGE_EXPERT_BLOCK32_MAX_TOKENS = 1028
SMALL_EXPERT_BLOCK16_MAX_TOKENS = 20
SMALL_EXPERT_BLOCK32_MAX_TOKENS = 40
WIDE_K_EXPERT_COUNT = 256
WIDE_K_MIN_TOKENS = 3584
MAX_SMALL_GROUPED_EXPERTS = 1024
SMALL_GROUPED_MAX_ROUTES = 64
SMALL_EXPERT_GROUPED_MAX_ROUTES = 128
PACKED_LOAD_MIN_TOKENS = 8
MIN_GROUP_SIZE = 128
FP8_BLOCK16_MAX_ROUTES_PER_EXPERT = 20
FP8_BLOCK32_MAX_ROUTES_PER_EXPERT = 32
FP8_BLOCK64_MAX_ROUTES_PER_EXPERT = 128
FP8_DEQUANT_MAX_BYTES = 1 << 30
FP8_DEQUANT_BLOCK_M = 128
FP8_DEQUANT_BLOCK_K = 128
FP8_DEQUANT_BLOCK_ROWS = 16
FP8_GEMV_MAX_ROUTES_PER_EXPERT = 0.5
FP8_GEMV_BLOCK_K = 128
FP8_GEMV_NARROW_MAX_OUTPUTS = 8192
FP8_DECODE_SCALE = tl.constexpr(256.0)


@triton.jit
def decode_fp8_e4m3(weight):
    """Decode E4M3FN bytes (uint8) to FP16 values divided by FP8_DECODE_SCALE.

    Moving the 4 exponent and 3 mantissa bits into an FP16 pattern is exact,
    including subnormals, and is faster than the native FP8 conversion on C550.
    NaN codes (0x7F / 0xFF) decode to +-480 instead of NaN.
    """
    # Sign extension puts the sign in bit 15; clear its copy in bit 14.
    bits = weight.to(tl.int8, bitcast=True).to(tl.int16)
    half_bits = (bits << 7) & -0x4001  # 0xBFFF as int16
    return half_bits.to(tl.float16, bitcast=True)


@triton.jit
def decode_fp8_e4m3_words(
    word, ROWS: tl.constexpr, COLS: tl.constexpr, compute_type: tl.constexpr
):
    """Decode int32 words of four K-consecutive E4M3FN bytes to [ROWS, COLS].

    BF16: normal exponents are rebiased by 120, so values are exact without a
    decode scale; subnormal codes (below 2**-6) become BF16 subnormals and are
    flushed by the MMA. FP16: the bits form q / FP8_DECODE_SCALE exactly. NaN
    codes (0x7F / 0xFF) decode to +-480 instead of NaN.
    """
    sign_odd = word & -0x7FFF8000  # 0x80008000: signs of bytes 1 and 3
    sign_even = (word & 0x00800080) << 8
    if compute_type == tl.bfloat16:
        # Bit 7 of each half is set when that byte has a non-zero exponent.
        rebias_even = ((word & 0x00780078) + 0x00780078) & 0x00800080
        rebias_odd = (((word & 0x78007800) >> 8) + 0x00780078) & 0x00800080
        even = (((word & 0x007F007F) << 4) + rebias_even * 120) | sign_even
        odd = (((word >> 4) & 0x07F007F0) + rebias_odd * 120) | sign_odd
    else:
        even = ((word & 0x007F007F) << 7) | sign_even
        odd = ((word >> 1) & 0x3F803F80) | sign_odd
    byte0 = even.to(tl.int16).to(compute_type, bitcast=True)
    byte2 = (even >> 16).to(tl.int16).to(compute_type, bitcast=True)
    byte1 = odd.to(tl.int16).to(compute_type, bitcast=True)
    byte3 = (odd >> 16).to(tl.int16).to(compute_type, bitcast=True)
    # The nested join flattens to K order byte0, byte1, byte2, byte3.
    return tl.reshape(
        tl.join(tl.join(byte0, byte2), tl.join(byte1, byte3)), (ROWS, COLS)
    )


@triton.jit
def moe_wna16_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    HOIST_SCALE: tl.constexpr,
    EVEN_K: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    PACKED_LOAD: tl.constexpr,
    NAIVE_ASSIGNMENT: tl.constexpr,
    IS_FP8: tl.constexpr,
    DEQUANTIZED: tl.constexpr,
):
    """Group-wise GEMM. B is (E, N, K // 2) INT4, (E, N, K // 4) int32 words of
    FP8 bytes, or dequantized (E, N, K); scales are (E, N, K // group)."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if NAIVE_ASSIGNMENT:
        # Each route needs its own block, so sorting cannot improve tile occupancy.
        rows = tl.arange(0, BLOCK_SIZE_M)
        offs_token = tl.where(rows == 0, pid_m, num_valid_tokens).to(tl.int64)
        token_mask = rows == 0
        expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    else:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
        token_mask = offs_token < num_valid_tokens
        expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_n = offs_cn % N

    if expert == -1:
        zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        tl.store(
            c_ptrs,
            zeros,
            mask=token_mask[:, None] & (offs_cn[None, :] < N),
        )
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # Keep masked tail pointers inside the packed-weight allocation.
    packed_k = tl.minimum(offs_k // 2, K // 2 - 1)
    a_ptrs = (
        a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + expert * stride_be
        + packed_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    shifter = (offs_k[:, None] % 2) * 4
    if PACKED_LOAD:
        offs_packed_k = tl.arange(0, BLOCK_SIZE_K // 2)
        b_packed_ptrs = (
            b_ptr
            + expert * stride_be
            + offs_packed_k[None, :] * stride_bk
            + offs_n[:, None] * stride_bn
        )
    if DEQUANTIZED:
        tl.static_assert(EVEN_K and not FUSE_SILU)
        b_dense_ptrs = (
            b_ptr
            + expert * stride_be
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )
    if IS_FP8:
        # The launcher keeps K tiles inside one scale group and never fuses SiLU.
        tl.static_assert(EVEN_K and HOIST_SCALE and not FUSE_SILU)
        # K-contiguous words load faster than the K-major tile and are transposed.
        offs_word = tl.arange(0, BLOCK_SIZE_K // 4)
        b_word_ptrs = (
            b_ptr
            + expert * stride_be
            + offs_word[None, :] * stride_bk
            + offs_n[:, None] * stride_bn
        )
        if compute_type == tl.bfloat16:
            decode_scale = 1.0
        else:
            decode_scale = FP8_DECODE_SCALE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    scale_expert = b_scale_ptr + expert * stride_bse
    if FUSE_SILU:
        b_up_ptrs = b_ptrs + N * stride_bn
        scale_up_expert = scale_expert + N * stride_bsn
        accumulator_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # cpasync can overlap the next K-tile load with the current dot.
    for tile in tl.range(tl.cdiv(K, BLOCK_SIZE_K)):
        k_off = tile * BLOCK_SIZE_K
        k_mask = offs_k < K - k_off
        if DEQUANTIZED:
            activation = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            accumulator += tl.dot(activation, tl.load(b_dense_ptrs), allow_tf32=False)
            b_dense_ptrs += BLOCK_SIZE_K * stride_bk
        elif IS_FP8:
            activation = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            code = decode_fp8_e4m3_words(
                tl.load(b_word_ptrs), BLOCK_SIZE_N, BLOCK_SIZE_K, compute_type
            )
            scale = tl.load(
                scale_expert + offs_n * stride_bsn + (k_off // GROUP_SIZE) * stride_bsk
            ).to(tl.float32)
            # FP8 values are exact in the compute type, so scale the partial
            # product once per K tile instead of every weight.
            partial = tl.dot(activation, tl.trans(code), allow_tf32=False)
            accumulator += partial * (scale * decode_scale)[None, :]
            b_word_ptrs += (BLOCK_SIZE_K // 4) * stride_bk
        else:
            if EVEN_K:
                activation = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                if PACKED_LOAD:
                    packed = tl.load(b_packed_ptrs)
                    nibble = tl.trans(tl.interleave(packed & 0xF, packed >> 4))
                else:
                    packed = tl.load(b_ptrs)
                    nibble = (packed >> shifter) & 0xF
                if FUSE_SILU:
                    packed_up = tl.load(b_up_ptrs)
                    nibble_up = (packed_up >> shifter) & 0xF
            else:
                activation = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                packed = tl.load(b_ptrs, mask=k_mask[:, None], other=0)
                nibble = tl.where(k_mask[:, None], (packed >> shifter) & 0xF, 0)
                if FUSE_SILU:
                    packed_up = tl.load(b_up_ptrs, mask=k_mask[:, None], other=0)
                    nibble_up = tl.where(
                        k_mask[:, None], (packed_up >> shifter) & 0xF, 0
                    )
            if HOIST_SCALE:
                scale = tl.load(
                    scale_expert
                    + offs_n * stride_bsn
                    + (k_off // GROUP_SIZE) * stride_bsk
                ).to(tl.float32)
                weight = ((nibble.to(tl.float32) - 8.0) * scale[None, :]).to(
                    compute_type
                )
                if FUSE_SILU:
                    scale_up = tl.load(
                        scale_up_expert
                        + offs_n * stride_bsn
                        + (k_off // GROUP_SIZE) * stride_bsk
                    ).to(tl.float32)
                    weight_up = (
                        (nibble_up.to(tl.float32) - 8.0) * scale_up[None, :]
                    ).to(compute_type)
            else:
                group_id_k = (offs_k[:, None] + k_off) // GROUP_SIZE
                scale = tl.load(
                    scale_expert
                    + offs_n[None, :] * stride_bsn
                    + group_id_k * stride_bsk,
                    mask=k_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                weight = ((nibble.to(tl.float32) - 8.0) * scale).to(compute_type)
                if FUSE_SILU:
                    scale_up = tl.load(
                        scale_up_expert
                        + offs_n[None, :] * stride_bsn
                        + group_id_k * stride_bsk,
                        mask=k_mask[:, None],
                        other=0.0,
                    ).to(tl.float32)
                    weight_up = ((nibble_up.to(tl.float32) - 8.0) * scale_up).to(
                        compute_type
                    )
            accumulator = tl.dot(activation, weight, acc=accumulator, allow_tf32=False)
            if FUSE_SILU:
                accumulator_up = tl.dot(
                    activation, weight_up, acc=accumulator_up, allow_tf32=False
                )
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        if FUSE_SILU:
            b_up_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        if PACKED_LOAD:
            b_packed_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    if MUL_ROUTED_WEIGHT:
        route = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * route[:, None]
        if FUSE_SILU:
            accumulator_up = accumulator_up * route[:, None]

    if FUSE_SILU:
        gate = accumulator.to(compute_type).to(tl.float32)
        up = accumulator_up.to(compute_type).to(tl.float32)
        accumulator = (gate / (1.0 + tl.exp(-gate))) * up

    offs_store = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_store[None, :]
    tl.store(
        c_ptrs,
        accumulator.to(compute_type),
        mask=token_mask[:, None] & (offs_store[None, :] < N),
    )


@triton.jit
def fp8_moe_gemv_gate_up_kernel(
    hidden_ptr,
    w_ptr,
    scale_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    out_ptr,
    I: tl.constexpr,
    K: tl.constexpr,
    stride_hm,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_sk,
    stride_om,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """SiLU(gate) * up for one route and BLOCK_SIZE_N intermediate columns."""
    route = tl.program_id(axis=0).to(tl.int64)
    pid_n = tl.program_id(axis=1)
    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    gate_ptrs = (
        w_ptr + expert * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :]
    )
    up_ptrs = gate_ptrs + I * stride_wn
    gate_scale_ptrs = scale_ptr + expert * stride_se + offs_n * stride_sn
    up_scale_ptrs = gate_scale_ptrs + I * stride_sn
    x_ptrs = hidden_ptr + (route // top_k) * stride_hm + offs_k
    gate = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    up = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for tile in tl.range(K // BLOCK_SIZE_K):
        group = (tile * BLOCK_SIZE_K // GROUP_SIZE) * stride_sk
        x = tl.load(x_ptrs).to(tl.float32)[None, :]
        gate_code = decode_fp8_e4m3(tl.load(gate_ptrs)).to(tl.float32)
        up_code = decode_fp8_e4m3(tl.load(up_ptrs)).to(tl.float32)
        gate_scale = tl.load(gate_scale_ptrs + group).to(tl.float32)
        up_scale = tl.load(up_scale_ptrs + group).to(tl.float32)
        gate += tl.sum(gate_code * x, axis=1) * gate_scale
        up += tl.sum(up_code * x, axis=1) * up_scale
        x_ptrs += BLOCK_SIZE_K
        gate_ptrs += BLOCK_SIZE_K
        up_ptrs += BLOCK_SIZE_K
    gate *= FP8_DECODE_SCALE
    up *= FP8_DECODE_SCALE
    if MUL_ROUTED_WEIGHT:
        route_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
        gate *= route_weight
        up *= route_weight
    # Round like the stored gate/up GEMM output before the activation.
    gate = gate.to(compute_type).to(tl.float32)
    up = up.to(compute_type).to(tl.float32)
    activated = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(out_ptr + route * stride_om + offs_n, activated.to(compute_type))


@triton.jit
def fp8_moe_gemv_down_kernel(
    activated_ptr,
    w_ptr,
    scale_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    out_ptr,
    I: tl.constexpr,
    stride_am,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_sk,
    stride_om,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Sum one token's top-k down projections; this replaces moe_sum."""
    token = tl.program_id(axis=0).to(tl.int64)
    pid_n = tl.program_id(axis=1)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    total = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for slot in tl.static_range(top_k):
        route = token * top_k + slot
        expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        w_ptrs = (
            w_ptr + expert * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :]
        )
        scale_ptrs = scale_ptr + expert * stride_se + offs_n * stride_sn
        a_ptrs = activated_ptr + route * stride_am + offs_k
        accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for tile in tl.range(I // BLOCK_SIZE_K):
            group = (tile * BLOCK_SIZE_K // GROUP_SIZE) * stride_sk
            a = tl.load(a_ptrs).to(tl.float32)[None, :]
            code = decode_fp8_e4m3(tl.load(w_ptrs)).to(tl.float32)
            scale = tl.load(scale_ptrs + group).to(tl.float32)
            accumulator += tl.sum(code * a, axis=1) * scale
            a_ptrs += BLOCK_SIZE_K
            w_ptrs += BLOCK_SIZE_K
        accumulator *= FP8_DECODE_SCALE
        if MUL_ROUTED_WEIGHT:
            accumulator *= tl.load(topk_weights_ptr + route).to(tl.float32)
        # Round each route like the routed buffer that moe_sum reads.
        total += accumulator.to(compute_type).to(tl.float32)
    tl.store(out_ptr + token * stride_om + offs_n, total.to(compute_type))


@triton.jit
def dequantize_fp8_weight_kernel(
    weight_ptr,
    scale_ptr,
    out_ptr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Write E4M3 weights times their group scales in the output dtype."""
    rows = tl.program_id(axis=0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    offs_k = tl.program_id(axis=1) * BLOCK_K + tl.arange(0, BLOCK_K)
    code = decode_fp8_e4m3(tl.load(weight_ptr + rows[:, None] * K + offs_k[None, :]))
    scale = tl.load(
        scale_ptr + rows[:, None] * (K // GROUP_SIZE) + offs_k[None, :] // GROUP_SIZE
    ).to(tl.float32)
    value = code.to(tl.float32) * (scale * FP8_DECODE_SCALE)
    tl.store(
        out_ptr + rows[:, None] * K + offs_k[None, :],
        value.to(out_ptr.dtype.element_ty),
    )


def dequantize_fp8_weight(
    weight: torch.Tensor, scale: torch.Tensor, group_size: int, dtype: torch.dtype
) -> torch.Tensor:
    """Return (E, N, K) E4M3 weights times their (E, N, K // group) scales."""
    num_experts, out_features, reduction = weight.shape
    output = torch.empty(weight.shape, device=weight.device, dtype=dtype)
    block_k = 256 if reduction % 256 == 0 else 128
    grid = (
        num_experts * out_features // FP8_DEQUANT_BLOCK_ROWS,
        reduction // block_k,
    )
    dequantize_fp8_weight_kernel[grid](
        weight.view(torch.uint8),
        scale,
        output,
        reduction,
        GROUP_SIZE=group_size,
        BLOCK_ROWS=FP8_DEQUANT_BLOCK_ROWS,
        BLOCK_K=block_k,
        num_warps=NUM_WARPS,
    )
    return output


def select_fp8_gemv_block_n(num_rows: int, width: int) -> int:
    """Use narrower GEMV tiles when wide ones would leave the device idle."""
    return 16 if num_rows * width < FP8_GEMV_NARROW_MAX_OUTPUTS else 32


def run_fp8_moe_gemv(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    group_size: int,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """FP8 MoE for routes that do not share experts: no alignment or moe_sum."""
    compute_type = tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16
    num_tokens, hidden_size = hidden_states.shape
    intermediate_size = w1.shape[1] // 2
    top_k = topk_ids.shape[1]
    num_routes = num_tokens * top_k
    activated = torch.empty(
        (num_routes, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    block_n = select_fp8_gemv_block_n(num_routes, intermediate_size)
    fp8_moe_gemv_gate_up_kernel[(num_routes, intermediate_size // block_n)](
        hidden_states,
        w1.view(torch.uint8),
        w1_scale,
        topk_ids,
        topk_weights,
        activated,
        intermediate_size,
        hidden_size,
        hidden_states.stride(0),
        w1.stride(0),
        w1.stride(1),
        w1_scale.stride(0),
        w1_scale.stride(1),
        w1_scale.stride(2),
        activated.stride(0),
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=FP8_GEMV_BLOCK_K,
        GROUP_SIZE=group_size,
        top_k=top_k,
        MUL_ROUTED_WEIGHT=apply_router_weight_on_input,
        compute_type=compute_type,
        num_warps=NUM_WARPS,
    )
    block_n = select_fp8_gemv_block_n(num_tokens, hidden_size)
    fp8_moe_gemv_down_kernel[(num_tokens, hidden_size // block_n)](
        activated,
        w2.view(torch.uint8),
        w2_scale,
        topk_ids,
        topk_weights,
        output,
        intermediate_size,
        activated.stride(0),
        w2.stride(0),
        w2.stride(1),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        output.stride(0),
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=FP8_GEMV_BLOCK_K,
        GROUP_SIZE=group_size,
        top_k=top_k,
        MUL_ROUTED_WEIGHT=not apply_router_weight_on_input,
        compute_type=compute_type,
        num_warps=NUM_WARPS,
    )
    return output


def select_tile_shape(num_tokens: int, num_experts: int) -> tuple[int, int, int]:
    """Select M padding by the number of routes available per expert."""
    if num_experts >= LARGE_EXPERT_MIN_COUNT:
        small_limit = LARGE_EXPERT_BLOCK16_MAX_TOKENS
        medium_limit = LARGE_EXPERT_BLOCK32_MAX_TOKENS
    else:
        small_limit = SMALL_EXPERT_BLOCK16_MAX_TOKENS
        medium_limit = SMALL_EXPERT_BLOCK32_MAX_TOKENS
    if num_tokens <= small_limit:
        block_m = 16
    elif num_tokens <= medium_limit:
        block_m = 32
    else:
        block_m = 64
    if num_tokens == 1:
        return block_m, 32, 64
    # A 128-wide K tile helps the 16-row tier and the high-density E=256 tier.
    # Keep K=64 for 32-row tiles and intermediate 64-row batches: those spill
    # or regress with K=128 on C550. Other large expert counts are unmeasured.
    should_use_wide_k = (
        num_experts >= LARGE_EXPERT_MIN_COUNT
        and FUSE_GATE_UP_MAX_TOKENS <= num_tokens <= LARGE_EXPERT_BLOCK16_MAX_TOKENS
    ) or (num_experts == WIDE_K_EXPERT_COUNT and num_tokens >= WIDE_K_MIN_TOKENS)
    block_k = 128 if should_use_wide_k else 64
    return block_m, 64, block_k


def select_fp8_tile_shape(
    num_tokens: int, num_experts: int, top_k: int
) -> tuple[int, int, int]:
    """Select FP8 tiles by the average number of routes per expert."""
    num_routes = num_tokens * top_k
    if num_routes <= FP8_BLOCK16_MAX_ROUTES_PER_EXPERT * num_experts:
        return 16, 64, 128
    if num_routes <= FP8_BLOCK32_MAX_ROUTES_PER_EXPERT * num_experts:
        return 32, 128, 128
    if num_routes <= FP8_BLOCK64_MAX_ROUTES_PER_EXPERT * num_experts:
        return 64, 128, 128
    # K=128 is slower than K=64 for 128-row FP8 tiles on C550.
    return 128, 128, 64


def activation_name(activation: str | Enum | None) -> str:
    if activation is None:
        return "silu"
    if isinstance(activation, str):
        return activation.lower()
    if isinstance(activation, Enum):
        if isinstance(activation.value, str):
            return activation.value.lower()
        return activation.name.lower()
    return ""


def check_wna16_options(
    hidden_states: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_zeros: Optional[torch.Tensor],
    w2_zeros: Optional[torch.Tensor],
    expert_map: Optional[torch.Tensor],
    g_idx1: Optional[torch.Tensor],
    g_idx2: Optional[torch.Tensor],
    sort_indices1: Optional[torch.Tensor],
    sort_indices2: Optional[torch.Tensor],
    input_global_scale1: Optional[torch.Tensor],
    input_global_scale2: Optional[torch.Tensor],
    global_scale1: Optional[torch.Tensor],
    global_scale2: Optional[torch.Tensor],
    activation: str | Enum | None,
    activation_func: Optional[Callable],
    input_dtype: Optional[torch.dtype],
    clamp_limit: Optional[float],
    is_k_full: bool,
    group_size: int,
    global_num_experts: int,
    num_experts: int,
) -> None:
    """Reject vLLM options that the MetaX kernels do not implement."""
    if any(item is not None for item in (g_idx1, g_idx2, sort_indices1, sort_indices2)):
        raise NotImplementedError("act_order (g_idx / sort_indices) is not supported")
    if input_dtype not in (None, hidden_states.dtype):
        raise NotImplementedError("FP8 / INT8 input quantization is not supported")
    if any(item is not None for item in (bias1, bias2, w1_zeros, w2_zeros)):
        raise NotImplementedError("bias and zero points are not supported")
    if expert_map is not None or global_num_experts not in (-1, num_experts):
        raise NotImplementedError("expert maps are not supported")
    if any(
        item is not None
        for item in (
            input_global_scale1,
            input_global_scale2,
            global_scale1,
            global_scale2,
        )
    ):
        raise NotImplementedError("global scales are not supported")
    if activation_func is not None or activation_name(activation) != "silu":
        raise NotImplementedError("only the SiLU / SwiGLU activation is supported")
    if clamp_limit is not None:
        raise NotImplementedError("clamp_limit is not supported")
    if not is_k_full:
        raise NotImplementedError("partial-K expert weights are not supported")
    if group_size < MIN_GROUP_SIZE or group_size % MIN_GROUP_SIZE:
        raise NotImplementedError(f"group_size must be a multiple of {MIN_GROUP_SIZE}")


def launch_moe_wna16_gemm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    should_mul_routed_weight: bool,
    top_k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    group_size: int,
    num_valid_tokens: int,
    should_fuse_silu: bool = False,
    should_use_packed_load: bool = False,
    should_use_naive_assignment: bool = False,
    is_fp8: bool = False,
    is_dequantized: bool = False,
) -> None:
    compute_type = tl.float16 if activation.dtype == torch.float16 else tl.bfloat16
    num_rows, reduction = activation.shape
    out_features = weight.shape[1] // 2 if should_fuse_silu else weight.shape[1]
    if should_use_naive_assignment:
        problem_m = num_valid_tokens * block_m
    else:
        problem_m = sorted_token_ids.shape[0]
        if num_rows < block_m:
            problem_m = min(problem_m, num_rows * top_k * block_m)
    if output.ndim == 3:
        stride_cm = output.stride(1)
        stride_cn = output.stride(2)
    else:
        stride_cm = output.stride(0)
        stride_cn = output.stride(1)
    if is_fp8:
        # Load four K-consecutive FP8 bytes per int32 word.
        weight = weight.view(torch.int32)
    grid = (triton.cdiv(problem_m, block_m) * triton.cdiv(out_features, block_n),)
    moe_wna16_gemm_kernel[grid](
        activation,
        weight,
        output,
        scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        out_features,
        reduction,
        problem_m,
        num_valid_tokens,
        activation.stride(0),
        activation.stride(1),
        weight.stride(0),
        weight.stride(2),
        weight.stride(1),
        stride_cm,
        stride_cn,
        scale.stride(0),
        scale.stride(2),
        scale.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=1,
        GROUP_SIZE=group_size,
        MUL_ROUTED_WEIGHT=should_mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        HOIST_SCALE=group_size % block_k == 0,
        EVEN_K=reduction % block_k == 0,
        FUSE_SILU=should_fuse_silu,
        PACKED_LOAD=should_use_packed_load and not should_fuse_silu,
        NAIVE_ASSIGNMENT=should_use_naive_assignment,
        IS_FP8=is_fp8,
        DEQUANTIZED=is_dequantized,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        pipeline="cpasync",
        pipeline_load_num=-1,
        inner_stages=(0, 0),
    )


def run_moe_wna16(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    group_size: int,
    apply_router_weight_on_input: bool,
    inplace: bool,
    output: Optional[torch.Tensor],
    reducer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | None],
    is_fp8: bool,
) -> torch.Tensor:
    if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("expected rank-2 activations and rank-3 weights")
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("routing tensors must have shape [tokens, topk]")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("activations must be FP16 or BF16")
    if is_fp8:
        pack_factor = 1
        if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
            raise ValueError("FP8 W8A16 weights must be FLOAT8_E4M3FN tensors")
        if (
            w1_scale.dtype not in (torch.float32, hidden_states.dtype)
            or w2_scale.dtype != w1_scale.dtype
        ):
            raise ValueError("FP8 scales must match the activation dtype or use FP32")
    else:
        pack_factor = 2
        if w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
            raise ValueError("UINT4B8 weights must be packed UINT8 tensors")
        if (
            w1_scale.dtype != hidden_states.dtype
            or w2_scale.dtype != hidden_states.dtype
        ):
            raise ValueError("UINT4B8 scales must match the activation dtype")
    num_tokens, hidden_size = hidden_states.shape
    num_experts, fused_intermediate, packed_k = w1.shape
    intermediate_size = fused_intermediate // 2
    top_k = topk_ids.shape[1]
    if (
        num_experts == 0
        or hidden_size == 0
        or intermediate_size == 0
        or fused_intermediate % 2
        or top_k < 1
        or top_k > num_experts
    ):
        raise ValueError("invalid expert, hidden, intermediate, or topk dimension")
    if hidden_size % group_size or intermediate_size % group_size:
        raise ValueError("hidden and intermediate dimensions must be group-aligned")
    if hidden_size != packed_k * pack_factor:
        raise ValueError(f"K mismatch: hidden {hidden_size} vs weight {packed_k}")
    if w2.shape != (num_experts, hidden_size, intermediate_size // pack_factor):
        raise ValueError(f"unexpected w2 shape {tuple(w2.shape)}")
    if w1_scale.shape != (num_experts, fused_intermediate, hidden_size // group_size):
        raise ValueError(f"unexpected w1_scale shape {tuple(w1_scale.shape)}")
    if w2_scale.shape != (num_experts, hidden_size, intermediate_size // group_size):
        raise ValueError(f"unexpected w2_scale shape {tuple(w2_scale.shape)}")
    if topk_ids.shape[0] != num_tokens or topk_weights.shape != topk_ids.shape:
        raise ValueError("routing tensors must have matching [tokens, topk] shapes")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be INT32 or INT64")
    if topk_weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("topk_weights must be floating point")
    tensors = (hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids)
    if any(
        tensor.device != hidden_states.device or not tensor.is_contiguous()
        for tensor in tensors
    ):
        raise ValueError("MoE tensors must be contiguous and on the same device")
    destination = hidden_states if inplace else output
    if destination is not None and (
        destination.shape != hidden_states.shape
        or destination.dtype != hidden_states.dtype
        or destination.device != hidden_states.device
        or not destination.is_contiguous()
    ):
        raise ValueError(
            "output must match hidden_states shape, dtype, device, and layout"
        )
    if num_tokens == 0:
        return (
            destination if destination is not None else torch.empty_like(hidden_states)
        )

    num_routes = topk_ids.numel()
    if (
        is_fp8
        and reducer is device_moe_sum
        and num_routes <= FP8_GEMV_MAX_ROUTES_PER_EXPERT * num_experts
    ):
        return run_fp8_moe_gemv(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            (
                destination
                if destination is not None
                else torch.empty_like(hidden_states)
            ),
            group_size=group_size,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
    if is_fp8:
        block_m, block_n, block_k = select_fp8_tile_shape(
            num_tokens, num_experts, top_k
        )
    else:
        block_m, block_n, block_k = select_tile_shape(num_tokens, num_experts)
    # The unfused FP8 path is faster at every measured batch size.
    should_fuse_gate_up = not is_fp8 and num_tokens <= FUSE_GATE_UP_MAX_TOKENS
    # In 128-row tiles one w1 decode is shared by enough routes that a dense
    # GEMM over a transient dequantized copy is faster than decoding per tile.
    should_dequantize_w1 = (
        is_fp8
        and block_m == FP8_DEQUANT_BLOCK_M
        and w1.numel() * hidden_states.element_size() <= FP8_DEQUANT_MAX_BYTES
    )
    if not should_fuse_gate_up:
        gate_up = torch.empty(
            (num_tokens * top_k, fused_intermediate),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    activated = torch.empty(
        (num_tokens * top_k, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    routed = torch.empty(
        (num_tokens, top_k, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    # The grouped align statically unrolls every route for every expert;
    # limit it sooner for large expert banks to bound compilation time.
    max_grouped_routes = (
        SMALL_GROUPED_MAX_ROUTES
        if num_experts >= LARGE_EXPERT_MIN_COUNT
        else SMALL_EXPERT_GROUPED_MAX_ROUTES
    )
    should_use_naive_assignment = (
        LARGE_EXPERT_MIN_COUNT <= num_experts <= MAX_SMALL_GROUPED_EXPERTS
        and num_routes <= SMALL_GROUPED_MAX_ROUTES
    )
    if should_use_naive_assignment:
        # Only expert_ids is read in this specialization. Reuse a no-copy
        # view for the unused alignment pointers; the grid has one route per CTA.
        sorted_token_ids = expert_ids = num_tokens_post_padded = topk_ids.view(-1)
    elif num_routes <= max_grouped_routes and num_experts <= MAX_SMALL_GROUPED_EXPERTS:
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size_small_grouped(topk_ids, num_experts, block_m)
        )
    else:
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size_no_tle(topk_ids, block_m, num_experts)
        )
    valid_slots = topk_ids.numel()
    # Packed-byte loads help 16-row tiles but spill on wider row tiles.
    should_use_packed_load = (
        num_experts >= LARGE_EXPERT_MIN_COUNT
        and PACKED_LOAD_MIN_TOKENS <= num_tokens <= LARGE_EXPERT_BLOCK16_MAX_TOKENS
    )
    if should_fuse_gate_up:
        launch_moe_wna16_gemm(
            hidden_states,
            w1,
            w1_scale,
            activated,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            should_mul_routed_weight=apply_router_weight_on_input,
            top_k=top_k,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            group_size=group_size,
            num_valid_tokens=valid_slots,
            should_fuse_silu=True,
            should_use_naive_assignment=should_use_naive_assignment,
            is_fp8=is_fp8,
        )
    else:
        launch_moe_wna16_gemm(
            hidden_states,
            (
                dequantize_fp8_weight(w1, w1_scale, group_size, hidden_states.dtype)
                if should_dequantize_w1
                else w1
            ),
            w1_scale,
            gate_up,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            should_mul_routed_weight=apply_router_weight_on_input,
            top_k=top_k,
            block_m=block_m,
            block_n=block_n,
            block_k=FP8_DEQUANT_BLOCK_K if should_dequantize_w1 else block_k,
            group_size=group_size,
            num_valid_tokens=valid_slots,
            should_use_packed_load=should_use_packed_load,
            should_use_naive_assignment=should_use_naive_assignment,
            is_fp8=is_fp8 and not should_dequantize_w1,
            is_dequantized=should_dequantize_w1,
        )
        silu_and_mul_out(
            gate_up[:, :intermediate_size],
            gate_up[:, intermediate_size:],
            activated,
        )
    launch_moe_wna16_gemm(
        activated,
        w2,
        w2_scale,
        routed,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        should_mul_routed_weight=not apply_router_weight_on_input,
        top_k=1,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_size=group_size,
        num_valid_tokens=valid_slots,
        should_use_packed_load=should_use_packed_load,
        should_use_naive_assignment=should_use_naive_assignment,
        is_fp8=is_fp8,
    )
    if inplace:
        out_hidden_states = hidden_states
    elif output is not None:
        out_hidden_states = output
    else:
        out_hidden_states = torch.empty_like(hidden_states)
    reducer(routed, out_hidden_states)
    return out_hidden_states


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: str | Enum | None = None,
    activation_func: Optional[Callable] = None,
    moe_sum: Optional[Callable] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    clamp_limit: Optional[float] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Dispatch UINT4B8 / FP8 E4M3 to the local WNA16 path, others to the shared path."""
    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")
    if quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP8_E4M3):
        check_wna16_options(
            hidden_states,
            bias1,
            bias2,
            w1_zeros,
            w2_zeros,
            expert_map,
            g_idx1,
            g_idx2,
            sort_indices1,
            sort_indices2,
            input_global_scale1,
            input_global_scale2,
            global_scale1,
            global_scale2,
            activation,
            activation_func,
            input_dtype,
            clamp_limit,
            is_k_full,
            group_size,
            global_num_experts,
            w1.shape[0],
        )
        return run_moe_wna16(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            group_size=group_size,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
            output=output,
            reducer=device_moe_sum if moe_sum is None else moe_sum,
            is_fp8=quant_type_id == QUANT_TYPE_FP8_E4M3,
        )
    return generic_fused_marlin_moe(
        hidden_states,
        w1,
        w2,
        bias1,
        bias2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        quant_type_id,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        activation=activation,
        activation_func=activation_func,
        moe_sum=moe_sum,
        expert_map=expert_map,
        input_global_scale1=input_global_scale1,
        input_global_scale2=input_global_scale2,
        global_scale1=global_scale1,
        global_scale2=global_scale2,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=w1_zeros,
        w2_zeros=w2_zeros,
        workspace=workspace,
        intermediate_cache13=intermediate_cache13,
        intermediate_cache2=intermediate_cache2,
        is_k_full=is_k_full,
        output=output,
        input_dtype=input_dtype,
        inplace=inplace,
        clamp_limit=clamp_limit,
        group_size=group_size,
    )


def fused_marlin_moe_w8a16_fp8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    group_size: int = 128,
    inplace: bool = False,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 E4M3 [E, N, K] weights with groupwise scales and A16 activations."""
    return fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        bias1=None,
        bias2=None,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=QUANT_TYPE_FP8_E4M3,
        group_size=group_size,
        inplace=inplace,
        output=output,
    )
