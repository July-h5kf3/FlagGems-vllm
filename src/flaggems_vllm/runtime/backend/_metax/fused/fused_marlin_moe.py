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

"""MetaX SwiGLU MoE with plain output-major INT4 or E4M3FN weights.

C550 has no fast FP8 conversion, so FP8 bytes are decoded with integer bit
moves; each K tile lies in one scale group, so the scale multiplies the
partial product.
"""

import math
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP8_E4M3, QUANT_TYPE_UINT4B8
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_no_tle,
    moe_align_block_size_small_grouped,
)
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._metax.fused.moe_sum import _qwen_moe_sum_kernel

# Module import: ops.fused_moe imports this package through fused.moe_sum.
from flaggems_vllm.runtime.backend._metax.ops import fused_moe as metax_fused_moe

MIN_GROUP_SIZE = 128
LARGE_EXPERT_MIN_COUNT = 128
MAX_GROUPED_ALIGN_EXPERTS = 1024
# The grouped align unrolls every route for every expert; bound compile time.
LARGE_EXPERT_GROUPED_MAX_ROUTES = 64
SMALL_EXPERT_GROUPED_MAX_ROUTES = 128
# Max tokens for 16- and 32-row tiles, keyed by E >= LARGE_EXPERT_MIN_COUNT.
INT4_BLOCK_M_MAX_TOKENS = {True: (448, 1028), False: (20, 40)}
INT4_FUSE_GATE_UP_MAX_TOKENS = 4
INT4_PACKED_LOAD_MIN_TOKENS = 8
INT4_WIDE_K_EXPERTS = 256
INT4_WIDE_K_MIN_TOKENS = 3584
FP8_GEMV_MAX_ROUTES_PER_EXPERT = 0.5
FP8_GEMV_BLOCK_K = 128
FP8_GEMV_NARROW_MAX_OUTPUTS = 8192
FP8_DENSE_MIN_ROUTES_PER_EXPERT = 64
FP8_DEQUANT_MAX_BYTES = 1 << 30
FP8_DEQUANT_BLOCK_ROWS = 16
FP8_NARROW_MAX_INTERMEDIATE = 512
FP8_N_TILES_MAX_K = 512
FP8_DECODE_SCALE = tl.constexpr(256.0)
FP8_BF16_REBIAS = tl.constexpr(2.0**120)


class Fp8Tile(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    pipeline: str
    scenario: str = ""
    # N tiles walked per program, so a short K still keeps loads in flight.
    n_tiles: int = 1


# (max routes per expert, gate/up tile, down tile), tuned on C550.
FP8_SPARSE_TIER = (
    20,
    Fp8Tile(16, 64, 128, 4, 2, "basic"),
    Fp8Tile(16, 64, 128, 4, 2, "basic", n_tiles=8),
)
FP8_NARROW_TIERS = (
    FP8_SPARSE_TIER,
    (
        32,
        Fp8Tile(32, 64, 128, 4, 2, "cpasync", "unroll"),
        Fp8Tile(32, 128, 128, 2, 2, "cpasync", "unroll"),
    ),
    (
        math.inf,
        Fp8Tile(64, 128, 128, 4, 2, "cpasync", "unroll"),
        Fp8Tile(64, 128, 64, 4, 3, "cpasync"),
    ),
)
FP8_WIDE_TIERS = (
    FP8_SPARSE_TIER,
    (32,) + (Fp8Tile(32, 128, 128, 4, 4, "cpasync"),) * 2,
    (128,) + (Fp8Tile(64, 128, 128, 4, 4, "cpasync"),) * 2,
    # K=128 is slower than K=64 for 128-row tiles on C550.
    (math.inf,) + (Fp8Tile(128, 128, 64, 4, 4, "cpasync"),) * 2,
)
FP8_DENSE_DOWN = Fp8Tile(128, 128, 64, 4, 4, "cpasync", "unroll")
FP8_DENSE_GATE_UP = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 4,
    "pipeline": "cpasync",
}


@triton.jit
def decode_fp8_e4m3(weight):
    """E4M3FN bytes to exact FP16 values divided by FP8_DECODE_SCALE."""
    # Sign extension puts the sign in bit 15; clear its copy in bit 14.
    bits = weight.to(tl.int8, bitcast=True).to(tl.int16)
    return ((bits << 7) & -0x4001).to(tl.float16, bitcast=True)


@triton.jit
def fp32_bits_to_bf16(bits):
    # Exact: E4M3 values leave the low 16 FP32 bits zero.
    value = bits.to(tl.float32, bitcast=True) * FP8_BF16_REBIAS
    value = (value.to(tl.int32, bitcast=True) >> 16).to(tl.int16)
    return value.to(tl.bfloat16, bitcast=True)


@triton.jit
def decode_fp8_e4m3_words(
    word, ROWS: tl.constexpr, COLS: tl.constexpr, compute_type: tl.constexpr
):
    """int32 words of four K-consecutive E4M3FN bytes to [ROWS, COLS] values.

    Values are exact. BF16 puts each byte in FP32 bits without a rebias, so
    subnormal codes stay subnormal, then multiplies by 2^120. FP16 values are
    divided by FP8_DECODE_SCALE, which also keeps subnormals exact.
    """
    if compute_type == tl.bfloat16:
        # Arithmetic shifts; 0x87F00000 keeps the sign and magnitude bits.
        byte0 = fp32_bits_to_bf16(((word << 24) >> 4) & -0x78100000)
        byte1 = fp32_bits_to_bf16(((word << 16) >> 4) & -0x78100000)
        byte2 = fp32_bits_to_bf16(((word << 8) >> 4) & -0x78100000)
        byte3 = fp32_bits_to_bf16((word >> 4) & -0x78100000)
    else:
        sign_odd = word & -0x7FFF8000
        sign_even = (word & 0x00800080) << 8
        even = ((word & 0x007F007F) << 7) | sign_even
        odd = ((word >> 1) & 0x3F803F80) | sign_odd
        byte0 = even.to(tl.int16).to(compute_type, bitcast=True)
        byte2 = (even >> 16).to(tl.int16).to(compute_type, bitcast=True)
        byte1 = odd.to(tl.int16).to(compute_type, bitcast=True)
        byte3 = (odd >> 16).to(tl.int16).to(compute_type, bitcast=True)
    return tl.reshape(
        tl.join(tl.join(byte0, byte2), tl.join(byte1, byte3)), (ROWS, COLS)
    )


@triton.jit
def int4_moe_gemm_kernel(
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
    num_valid_tokens,
    stride_am,
    stride_be,
    stride_bn,
    stride_cm,
    stride_bse,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    PACKED_LOAD: tl.constexpr,
    NAIVE_ASSIGNMENT: tl.constexpr,
):
    """B is (E, N, K // 2) uint8 and scales are (E, N, K // GROUP_SIZE)."""
    compute_type = a_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)
    pid_m = pid // (N // BLOCK_SIZE_N)
    pid_n = pid % (N // BLOCK_SIZE_N)
    if NAIVE_ASSIGNMENT:
        # One route per program: sorting cannot improve tile occupancy.
        rows = tl.arange(0, BLOCK_SIZE_M)
        offs_token = tl.where(rows == 0, pid_m, num_valid_tokens).to(tl.int64)
        token_mask = rows == 0
    else:
        if pid_m * BLOCK_SIZE_M >= tl.load(num_tokens_post_padded_ptr):
            return
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
        token_mask = offs_token < num_valid_tokens
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + offs_n[None, :]
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + expert * stride_be + (offs_k // 2)[:, None]
    b_ptrs += offs_n[None, :] * stride_bn
    shifter = (offs_k[:, None] % 2) * 4
    if PACKED_LOAD:
        b_packed_ptrs = b_ptr + expert * stride_be + offs_n[:, None] * stride_bn
        b_packed_ptrs += tl.arange(0, BLOCK_SIZE_K // 2)[None, :]
    scale_ptrs = b_scale_ptr + expert * stride_bse + offs_n * stride_bsn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if FUSE_SILU:
        accumulator_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for tile in tl.range(K // BLOCK_SIZE_K):
        activation = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        if PACKED_LOAD:
            packed = tl.load(b_packed_ptrs)
            nibble = tl.trans(tl.interleave(packed & 0xF, packed >> 4))
        else:
            nibble = (tl.load(b_ptrs) >> shifter) & 0xF
        if FUSE_SILU:
            nibble_up = (tl.load(b_ptrs + N * stride_bn) >> shifter) & 0xF
        group = tile * BLOCK_SIZE_K // GROUP_SIZE
        scale = tl.load(scale_ptrs + group).to(tl.float32)
        weight = ((nibble.to(tl.float32) - 8.0) * scale[None, :]).to(compute_type)
        if FUSE_SILU:
            scale_up = tl.load(scale_ptrs + N * stride_bsn + group).to(tl.float32)
            weight_up = (nibble_up.to(tl.float32) - 8.0) * scale_up[None, :]
            weight_up = weight_up.to(compute_type)
        accumulator = tl.dot(activation, weight, acc=accumulator, allow_tf32=False)
        if FUSE_SILU:
            accumulator_up = tl.dot(
                activation, weight_up, acc=accumulator_up, allow_tf32=False
            )
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K // 2
        if PACKED_LOAD:
            b_packed_ptrs += BLOCK_SIZE_K // 2

    if MUL_ROUTED_WEIGHT:
        route = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * route[:, None]
        if FUSE_SILU:
            accumulator_up = accumulator_up * route[:, None]
    if FUSE_SILU:
        gate = accumulator.to(compute_type).to(tl.float32)
        up = accumulator_up.to(compute_type).to(tl.float32)
        accumulator = (gate / (1.0 + tl.exp(-gate))) * up
    tl.store(c_ptrs, accumulator.to(compute_type), mask=token_mask[:, None])


@triton.jit
def fp8_moe_gemm_kernel(
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
    num_valid_tokens,
    stride_am,
    stride_be,
    stride_bn,
    stride_cm,
    stride_bse,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    N_TILES: tl.constexpr,
    ALIGN_BLOCK_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
):
    """B is E4M3FN viewed as (E, N, K // 4) int32 words. Routes are padded
    per expert to ALIGN_BLOCK_M, a multiple of BLOCK_SIZE_M."""
    compute_type = a_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)
    pid_m = pid // (N // (BLOCK_SIZE_N * N_TILES))
    pid_n = pid % (N // (BLOCK_SIZE_N * N_TILES))
    if pid_m * BLOCK_SIZE_M >= tl.load(num_tokens_post_padded_ptr):
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    expert = tl.load(expert_ids_ptr + pid_m // (ALIGN_BLOCK_M // BLOCK_SIZE_M))
    expert = expert.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :]
    # K-contiguous words load faster than a K-major tile.
    b_ptrs = b_ptr + expert * stride_be + tl.arange(0, BLOCK_SIZE_K // 4)[None, :]
    scale_ptrs = b_scale_ptr + expert * stride_bse
    decode_scale = 1.0 if compute_type == tl.bfloat16 else FP8_DECODE_SCALE
    if MUL_ROUTED_WEIGHT:
        route = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
    num_k_tiles: tl.constexpr = K // BLOCK_SIZE_K
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if N_TILES == 1:
        # Pointer-increment loop; measurably faster than the flat loop below.
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        b_ptrs += offs_n[:, None] * stride_bn
        scale_ptrs += offs_n * stride_bsn
        for tile in tl.range(num_k_tiles):
            activation = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            code = decode_fp8_e4m3_words(
                tl.load(b_ptrs), BLOCK_SIZE_N, BLOCK_SIZE_K, compute_type
            )
            scale = tl.load(scale_ptrs + tile * BLOCK_SIZE_K // GROUP_SIZE)
            partial = tl.dot(activation, tl.trans(code), allow_tf32=False)
            accumulator += partial * (scale.to(tl.float32) * decode_scale)[None, :]
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K // 4
        if MUL_ROUTED_WEIGHT:
            accumulator = accumulator * route[:, None]
        tl.store(
            c_ptrs + pid_n * BLOCK_SIZE_N,
            accumulator.to(compute_type),
            mask=token_mask[:, None],
        )
        return
    for step in tl.range(N_TILES * num_k_tiles):
        n_tile = step // num_k_tiles
        tile = step % num_k_tiles
        offs_n = (pid_n * N_TILES + n_tile) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        activation = tl.load(
            a_ptrs + tile * BLOCK_SIZE_K, mask=token_mask[:, None], other=0.0
        )
        word = tl.load(b_ptrs + offs_n[:, None] * stride_bn + tile * BLOCK_SIZE_K // 4)
        code = decode_fp8_e4m3_words(word, BLOCK_SIZE_N, BLOCK_SIZE_K, compute_type)
        group = tile * BLOCK_SIZE_K // GROUP_SIZE
        scale = tl.load(scale_ptrs + offs_n * stride_bsn + group).to(tl.float32)
        partial = tl.dot(activation, tl.trans(code), allow_tf32=False)
        partial = partial * (scale * decode_scale)[None, :]
        accumulator = tl.where(tile == 0, partial, accumulator + partial)
        if tile == num_k_tiles - 1:
            result = accumulator
            if MUL_ROUTED_WEIGHT:
                result = result * route[:, None]
            n_start = (pid_n * N_TILES + n_tile) * BLOCK_SIZE_N
            tl.store(
                c_ptrs + n_start, result.to(compute_type), mask=token_mask[:, None]
            )


@triton.jit
def fp8_moe_gemv_kernel(
    a_ptr,
    w_ptr,
    scale_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    top_k: tl.constexpr,
    FIRST: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
):
    """FIRST: SiLU(gate) * up of one route. Otherwise the sum of one token's
    top-k down projections, which replaces moe_sum."""
    compute_type = a_ptr.dtype.element_ty
    row = tl.program_id(axis=0).to(tl.int64)
    offs_n = tl.program_id(axis=1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    total = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for slot in tl.static_range(1 if FIRST else top_k):
        route = row if FIRST else row * top_k + slot
        expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        w_ptrs = w_ptr + expert * stride_we + offs_n[:, None] * stride_wn
        w_ptrs += offs_k[None, :]
        scale_ptrs = scale_ptr + expert * stride_se + offs_n * stride_sn
        a_ptrs = a_ptr + (row // top_k if FIRST else route) * K + offs_k
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        up = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for tile in tl.range(K // BLOCK_SIZE_K):
            group = tile * BLOCK_SIZE_K // GROUP_SIZE
            a = tl.load(a_ptrs).to(tl.float32)[None, :]
            code = decode_fp8_e4m3(tl.load(w_ptrs)).to(tl.float32)
            scale = tl.load(scale_ptrs + group).to(tl.float32)
            acc += tl.sum(code * a, axis=1) * scale
            if FIRST:
                code_up = decode_fp8_e4m3(tl.load(w_ptrs + N * stride_wn))
                scale_up = tl.load(scale_ptrs + N * stride_sn + group).to(tl.float32)
                up += tl.sum(code_up.to(tl.float32) * a, axis=1) * scale_up
            a_ptrs += BLOCK_SIZE_K
            w_ptrs += BLOCK_SIZE_K
        acc *= FP8_DECODE_SCALE
        up *= FP8_DECODE_SCALE
        if MUL_ROUTED_WEIGHT:
            route_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
            acc *= route_weight
            up *= route_weight
        # Round like the routed buffers of the GEMM path.
        if FIRST:
            gate = acc.to(compute_type).to(tl.float32)
            up = up.to(compute_type).to(tl.float32)
            total = gate / (1.0 + tl.exp(-gate)) * up
        else:
            total += acc.to(compute_type).to(tl.float32)
    tl.store(out_ptr + row * N + offs_n, total.to(compute_type))


@triton.jit
def dequantize_fp8_kernel(
    weight_ptr,
    scale_ptr,
    out_ptr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
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


def sum_routes(routed, output):
    """MetaX's unrolled moe_sum kernel for every top-k: moe_sum sends other
    top-k values to a generic autotuner whose 1024-thread config fails on C550."""
    num_tokens, top_k, hidden = routed.shape
    _qwen_moe_sum_kernel[(num_tokens, triton.cdiv(hidden, 2048))](
        routed,
        output,
        routed,
        0,
        0,
        num_tokens,
        hidden,
        TOPK=top_k,
        APPLY_ROUTER_WEIGHT=False,
        BLOCK_SIZE=2048,
        num_warps=8,
    )


@triton.jit
def zero_workspace_kernel(x_ptr, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + offsets, 0, mask=offsets < numel)


def align_routes(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    max_grouped_routes = (
        LARGE_EXPERT_GROUPED_MAX_ROUTES
        if num_experts >= LARGE_EXPERT_MIN_COUNT
        else SMALL_EXPERT_GROUPED_MAX_ROUTES
    )
    if (
        topk_ids.numel() <= max_grouped_routes
        and num_experts <= MAX_GROUPED_ALIGN_EXPERTS
    ):
        return moe_align_block_size_small_grouped(topk_ids, num_experts, block_m)
    # Zero the aligner's cumsum and count buffers with Triton, not Torch.
    workspace = topk_ids.new_empty(((num_experts + 1) ** 2,), dtype=torch.int32)
    zero_workspace_kernel[(triton.cdiv(workspace.numel(), 1024),)](
        workspace, workspace.numel(), BLOCK=1024
    )
    cumsum, counts = workspace.split([num_experts + 1, num_experts * (num_experts + 1)])
    return moe_align_block_size_no_tle(
        topk_ids, block_m, num_experts, workspace=(cumsum, counts)
    )


def launch_int4_gemm(
    activation,
    weight,
    scale,
    output,
    topk_weights,
    alignment,
    *,
    mul_routed_weight,
    top_k,
    tile,
    group_size,
    fuse_silu=False,
    packed_load=False,
    naive_assignment=False,
):
    block_m, block_n, block_k = tile
    sorted_token_ids = alignment[0]
    num_routes = topk_weights.numel()
    out_features = weight.shape[1] // 2 if fuse_silu else weight.shape[1]
    if naive_assignment:
        problem_m = num_routes * block_m
    else:
        problem_m = min(sorted_token_ids.shape[0], num_routes * block_m)
    grid = (triton.cdiv(problem_m, block_m) * (out_features // block_n),)
    int4_moe_gemm_kernel[grid](
        activation,
        weight,
        output,
        scale,
        topk_weights,
        *alignment,
        out_features,
        activation.shape[1],
        num_routes,
        activation.stride(0),
        weight.stride(0),
        weight.stride(1),
        output.stride(-2),
        scale.stride(0),
        scale.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        FUSE_SILU=fuse_silu,
        PACKED_LOAD=packed_load and not fuse_silu,
        NAIVE_ASSIGNMENT=naive_assignment,
        num_warps=4,
        num_stages=4,
        pipeline="cpasync",
        pipeline_load_num=-1,
        inner_stages=(0, 0),
    )


def run_int4_moe(hs, w1, w2, s1, s2, topk_weights, topk_ids, output, **options):
    num_tokens, hidden_size = hs.shape
    num_experts, fused_intermediate, _ = w1.shape
    top_k = topk_ids.shape[1]
    num_routes = topk_ids.numel()
    is_large_bank = num_experts >= LARGE_EXPERT_MIN_COUNT
    small_limit, medium_limit = INT4_BLOCK_M_MAX_TOKENS[is_large_bank]
    block_m = 16 if num_tokens <= small_limit else 32
    block_m = block_m if num_tokens <= medium_limit else 64
    # K=128 helps 16-row tiles and dense E=256 batches; others spill with it.
    is_wide_k = (
        is_large_bank and INT4_FUSE_GATE_UP_MAX_TOKENS <= num_tokens <= small_limit
    ) or (num_experts == INT4_WIDE_K_EXPERTS and num_tokens >= INT4_WIDE_K_MIN_TOKENS)
    tile = (block_m, 32, 64) if num_tokens == 1 else (block_m, 64, 64)
    if num_tokens > 1 and is_wide_k:
        tile = (block_m, 64, 128)
    naive_assignment = (
        is_large_bank
        and num_experts <= MAX_GROUPED_ALIGN_EXPERTS
        and num_routes <= LARGE_EXPERT_GROUPED_MAX_ROUTES
    )
    if naive_assignment:
        alignment = (topk_ids.view(-1),) * 3
    else:
        alignment = align_routes(topk_ids, block_m, num_experts)
    fuse_silu = num_tokens <= INT4_FUSE_GATE_UP_MAX_TOKENS
    # Packed-byte loads help 16-row tiles but spill on wider ones.
    packed_load = (
        is_large_bank and INT4_PACKED_LOAD_MIN_TOKENS <= num_tokens <= small_limit
    )
    kwargs = dict(
        group_size=options["group_size"],
        tile=tile,
        naive_assignment=naive_assignment,
    )
    activated = hs.new_empty((num_routes, fused_intermediate // 2))
    router_on_input = options["apply_router_weight_on_input"]
    gate_up = activated if fuse_silu else hs.new_empty((num_routes, fused_intermediate))
    launch_int4_gemm(
        hs,
        w1,
        s1,
        gate_up,
        topk_weights,
        alignment,
        mul_routed_weight=router_on_input,
        top_k=top_k,
        fuse_silu=fuse_silu,
        packed_load=packed_load,
        **kwargs,
    )
    if not fuse_silu:
        silu_and_mul_out(*gate_up.chunk(2, dim=-1), activated)
    routed = hs.new_empty((num_tokens, top_k, hidden_size))
    launch_int4_gemm(
        activated,
        w2,
        s2,
        routed,
        topk_weights,
        alignment,
        mul_routed_weight=not router_on_input,
        top_k=1,
        packed_load=packed_load,
        **kwargs,
    )
    sum_routes(routed, output)
    return output


def launch_fp8_gemm(
    activation,
    weight,
    scale,
    output,
    topk_weights,
    alignment,
    *,
    tile,
    align_block_m,
    mul_routed_weight,
    top_k,
    group_size,
):
    out_features, reduction = weight.shape[1], activation.shape[1]
    num_routes = topk_weights.numel()
    problem_m = min(alignment[0].shape[0], num_routes * align_block_m)
    n_tiles = tile.n_tiles if reduction <= FP8_N_TILES_MAX_K else 1
    while out_features % (tile.block_n * n_tiles):
        n_tiles //= 2
    grid = (
        triton.cdiv(problem_m, tile.block_m)
        * (out_features // (tile.block_n * n_tiles)),
    )
    fp8_moe_gemm_kernel[grid](
        activation,
        weight.view(torch.int32),
        output,
        scale,
        topk_weights,
        *alignment,
        out_features,
        reduction,
        num_routes,
        activation.stride(0),
        weight.stride(0) // 4,
        weight.stride(1) // 4,
        output.stride(-2),
        scale.stride(0),
        scale.stride(1),
        BLOCK_SIZE_M=tile.block_m,
        BLOCK_SIZE_N=tile.block_n,
        BLOCK_SIZE_K=tile.block_k,
        N_TILES=n_tiles,
        ALIGN_BLOCK_M=align_block_m,
        GROUP_SIZE=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
        pipeline=tile.pipeline,
        scenario=tile.scenario,
    )


def launch_fp8_gemv(activation, weight, scale, topk_ids, topk_weights, output, **kw):
    num_rows, width = output.shape
    # Narrower tiles keep the device busy when there are few outputs.
    block_n = 16 if num_rows * width < FP8_GEMV_NARROW_MAX_OUTPUTS else 32
    fp8_moe_gemv_kernel[(num_rows, width // block_n)](
        activation,
        weight.view(torch.uint8),
        scale,
        topk_ids,
        topk_weights,
        output,
        width,
        activation.shape[1],
        weight.stride(0),
        weight.stride(1),
        scale.stride(0),
        scale.stride(1),
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=FP8_GEMV_BLOCK_K,
        num_warps=4,
        **kw,
    )


def dequantize_fp8(weight, scale, group_size, dtype):
    num_experts, out_features, reduction = weight.shape
    output = torch.empty(weight.shape, device=weight.device, dtype=dtype)
    block_k = 256 if reduction % 256 == 0 else 128
    grid = (num_experts * out_features // FP8_DEQUANT_BLOCK_ROWS, reduction // block_k)
    dequantize_fp8_kernel[grid](
        weight.view(torch.uint8),
        scale,
        output,
        reduction,
        GROUP_SIZE=group_size,
        BLOCK_ROWS=FP8_DEQUANT_BLOCK_ROWS,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output


def run_fp8_moe(hs, w1, w2, s1, s2, topk_weights, topk_ids, output, **options):
    num_tokens, hidden_size = hs.shape
    num_experts, fused_intermediate, _ = w1.shape
    intermediate_size = fused_intermediate // 2
    top_k = topk_ids.shape[1]
    num_routes = topk_ids.numel()
    routes_per_expert = num_routes / num_experts
    group_size = options["group_size"]
    router_on_input = options["apply_router_weight_on_input"]
    activated = hs.new_empty((num_routes, intermediate_size))
    if routes_per_expert <= FP8_GEMV_MAX_ROUTES_PER_EXPERT:
        common = dict(GROUP_SIZE=group_size, top_k=top_k)
        launch_fp8_gemv(
            hs,
            w1,
            s1,
            topk_ids,
            topk_weights,
            activated,
            FIRST=True,
            MUL_ROUTED_WEIGHT=router_on_input,
            num_stages=2,
            pipeline="basic",
            **common,
        )
        launch_fp8_gemv(
            activated,
            w2,
            s2,
            topk_ids,
            topk_weights,
            output,
            FIRST=False,
            MUL_ROUTED_WEIGHT=not router_on_input,
            **common,
        )
        return output
    # One w1 dequantization into a bounded buffer beats decoding it per tile.
    should_dequantize_w1 = (
        routes_per_expert > FP8_DENSE_MIN_ROUTES_PER_EXPERT
        and w1.numel() * hs.element_size() <= FP8_DEQUANT_MAX_BYTES
    )
    if should_dequantize_w1:
        align_block_m = FP8_DENSE_GATE_UP["BLOCK_SIZE_M"]
        down_tile = FP8_DENSE_DOWN
    else:
        tiers = (
            FP8_NARROW_TIERS
            if intermediate_size <= FP8_NARROW_MAX_INTERMEDIATE
            else FP8_WIDE_TIERS
        )
        _, gate_up_tile, down_tile = next(
            tier for tier in tiers if routes_per_expert <= tier[0]
        )
        align_block_m = max(gate_up_tile.block_m, down_tile.block_m)
    alignment = align_routes(topk_ids, align_block_m, num_experts)
    gate_up = hs.new_empty((num_routes, fused_intermediate))
    common = dict(top_k=top_k, align_block_m=align_block_m, group_size=group_size)
    if should_dequantize_w1:
        metax_fused_moe.invoke_fused_moe_triton_kernel(
            hs,
            dequantize_fp8(w1, s1, group_size, hs.dtype),
            gate_up.view(num_tokens, top_k, fused_intermediate),
            None,
            None,
            topk_weights,
            *alignment,
            router_on_input,
            top_k,
            FP8_DENSE_GATE_UP,
            tl.float16 if hs.dtype == torch.float16 else tl.bfloat16,
        )
    else:
        launch_fp8_gemm(
            hs,
            w1,
            s1,
            gate_up,
            topk_weights,
            alignment,
            tile=gate_up_tile,
            mul_routed_weight=router_on_input,
            **common,
        )
    silu_and_mul_out(*gate_up.chunk(2, dim=-1), activated)
    routed = hs.new_empty((num_tokens, top_k, hidden_size))
    common["top_k"] = 1
    launch_fp8_gemm(
        activated,
        w2,
        s2,
        routed,
        topk_weights,
        alignment,
        tile=down_tile,
        mul_routed_weight=not router_on_input,
        **common,
    )
    sum_routes(routed, output)
    return output


def check_inputs(hs, w1, w2, s1, s2, topk_weights, topk_ids, output, group_size, quant):
    if hs.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or topk_ids.ndim != 2:
        raise ValueError("expected rank-2 activations and routing, rank-3 weights")
    if hs.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("activations must be FP16 or BF16")
    if group_size < MIN_GROUP_SIZE or group_size % MIN_GROUP_SIZE:
        raise NotImplementedError(f"group_size must be a multiple of {MIN_GROUP_SIZE}")
    num_tokens, hidden_size = hs.shape
    num_experts, fused_intermediate, _ = w1.shape
    intermediate_size = fused_intermediate // 2
    pack = 2 if quant == QUANT_TYPE_UINT4B8 else 1
    if (
        num_experts == 0
        or intermediate_size == 0
        or hidden_size % group_size
        or intermediate_size % group_size
        or w1.shape != (num_experts, 2 * intermediate_size, hidden_size // pack)
        or w2.shape != (num_experts, hidden_size, intermediate_size // pack)
    ):
        raise ValueError("weight shapes must be group-aligned and match hs")
    if s1.shape != (
        num_experts,
        2 * intermediate_size,
        hidden_size // group_size,
    ) or s2.shape != (num_experts, hidden_size, intermediate_size // group_size):
        raise ValueError("scale shapes do not match the quantization groups")
    weight_dtype = torch.uint8 if pack == 2 else torch.float8_e4m3fn
    scale_dtypes = (hs.dtype,) if pack == 2 else (hs.dtype, torch.float32)
    if w1.dtype != weight_dtype or w2.dtype != weight_dtype:
        raise ValueError(f"weight dtype must be {weight_dtype}")
    if s1.dtype not in scale_dtypes or s2.dtype != s1.dtype:
        raise ValueError(f"scale dtype must be one of {scale_dtypes}")
    if (
        topk_weights.shape != topk_ids.shape
        or topk_ids.shape[0] != num_tokens
        or not 1 <= topk_ids.shape[1] <= num_experts
        or topk_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("routing must be [tokens, topk] with integer expert ids")
    tensors = (hs, w1, w2, s1, s2, topk_weights, topk_ids)
    if output is not None:
        tensors += (output,)
        if output.shape != hs.shape or output.dtype != hs.dtype:
            raise ValueError("output must match hidden_states shape and dtype")
    if any(t.device != hs.device or not t.is_contiguous() for t in tensors):
        raise ValueError("MoE tensors must be contiguous and on one device")


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
    activation: Any = None,
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
    """UINT4B8 or FP8 E4M3 SwiGLU MoE with A16 activations."""
    if quant_type_id not in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP8_E4M3):
        raise NotImplementedError(
            f"MetaX does not support quant_type_id {quant_type_id}"
        )
    if any(x is not None for x in (g_idx1, g_idx2, sort_indices1, sort_indices2)):
        raise NotImplementedError("act_order is not supported")
    if input_dtype is not None:
        raise NotImplementedError("FP8 / INT8 input quantization is not supported")
    name = activation.value if isinstance(activation, Enum) else activation
    if name is not None and str(name).lower() != "silu":
        raise NotImplementedError("only the SiLU activation is supported")
    unsupported = (
        bias1,
        bias2,
        activation_func,
        moe_sum,
        expert_map,
        input_global_scale1,
        input_global_scale2,
        global_scale1,
        global_scale2,
        w1_zeros,
        w2_zeros,
        clamp_limit,
    )
    if any(x is not None for x in unsupported) or not is_k_full:
        raise NotImplementedError("unsupported fused_marlin_moe option on MetaX")
    if global_num_experts not in (-1, w1.shape[0]):
        raise NotImplementedError("expert maps are not supported")
    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")
    if inplace:
        output = hidden_states
    check_inputs(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        group_size,
        quant_type_id,
    )
    if output is None:
        output = torch.empty_like(hidden_states)
    if hidden_states.shape[0] == 0:
        return output
    run = run_fp8_moe if quant_type_id == QUANT_TYPE_FP8_E4M3 else run_int4_moe
    with torch_device_fn.device(hidden_states.device):
        return run(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            output,
            group_size=group_size,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


def fused_marlin_moe_w4a16_int4(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_ids,
    *,
    activation="silu",
    group_size=128,
    apply_router_weight_on_input=False,
    inplace=False,
    swap_ab=True,
):
    return fused_marlin_moe(
        hidden_states,
        w1,
        w2,
        None,
        None,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        QUANT_TYPE_UINT4B8,
        activation=activation,
        group_size=group_size,
        apply_router_weight_on_input=apply_router_weight_on_input,
        inplace=inplace,
    )


def fused_marlin_moe_w8a16_fp8(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    *,
    w1_scale,
    w2_scale,
    group_size=128,
    inplace=False,
    output=None,
):
    return fused_marlin_moe(
        hidden_states,
        w1,
        w2,
        None,
        None,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        QUANT_TYPE_FP8_E4M3,
        group_size=group_size,
        inplace=inplace,
        output=output,
    )


__all__ = [
    "fused_marlin_moe",
    "fused_marlin_moe_w4a16_int4",
    "fused_marlin_moe_w8a16_fp8",
]
