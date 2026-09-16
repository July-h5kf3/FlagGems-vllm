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

"""Vendored baseline for benchmark-only use on PPU.

Derived (Apache-2.0) from the upstream vLLM main kernel
``vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`` with all
vLLM-only scaffolding (``VllmTritonJitKernel`` warmup integration, ``launch_pdl``,
custom-op registration, ``current_platform``) stripped, per the PPU benchmark
policy ("vLLM 仓库算子" baseline). The kernel logic is kept identical except
that the native ``tl.float8e4nv`` cast is replaced by the same verified manual
E4M3 conversion used by the PPU operator, because FlagTree on PPU has no
working native fp8e4m3fn conversion (the fairest possible copy on PPU).

Both variants are exposed through ``quantize``:
    quantize=True  -> FP8 E4M3 + UE8M0(pow2) scales (same-precision baseline)
    quantize=False -> BF16 rotated output, no scales  (BF16 baseline)
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _float_to_e4m3fn_bits(x):
    """Convert pre-clamped (|x| <= 448) f32 values to e4m3fn bits (0..255).

    Identical to the helper in
    ``flaggems_vllm/runtime/backend/_thead/fused/fused_inv_rope_fp8_quant.py``;
    bit-exact with torch's float8_e4m3fn cast (probed exhaustively).
    """
    xb = x.to(tl.int32, bitcast=True)
    s = (xb >> 31) & 1
    e8 = (xb >> 23) & 0xFF
    m23 = xb & 0x7FFFFF
    c = ((e8 << 23) | m23) - (120 << 23)
    q_norm = (c + 0x7FFFF + ((c >> 20) & 1)) >> 20
    q_norm = tl.minimum(q_norm, 0x7E)
    sh = tl.maximum(141 - e8, 1)
    n = (1 << 23) | m23
    q_sub = (n + (1 << (sh - 1)) - 1 + ((n >> sh) & 1)) >> sh
    q = tl.where(e8 >= 121, q_norm, tl.where(e8 >= 117, q_sub, 0))
    return (s << 7) | (q & 0x7F)


@triton.jit(do_not_specialize=["num_tokens", "scale_stride_k"])
def _vllm_fused_inv_rope_fp8_quant_kernel(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    out_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    out_stride_group,
    out_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    QUANTIZE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)
    o_stride_token = o_stride_token.to(tl.int64)
    o_stride_head = o_stride_head.to(tl.int64)
    cache_stride_pos = cache_stride_pos.to(tl.int64)
    out_stride_group = out_stride_group.to(tl.int64)
    out_stride_token = out_stride_token.to(tl.int64)
    scale_stride_group = scale_stride_group.to(tl.int64)
    scale_stride_k = scale_stride_k.to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD
    # Padding rows in the TMA-aligned scale buffer: fill with zero and skip quant.
    if pid_token >= num_tokens:
        if not QUANTIZE:
            return
        if TMA_ALIGNED_SCALES:
            packed_offsets = tl.arange(0, CHUNKS_PER_HEAD // 4)
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + (head_in_group * (CHUNKS_PER_HEAD // 4) + packed_offsets)
                * scale_stride_k
            )
            tl.store(scale_addr, tl.zeros((CHUNKS_PER_HEAD // 4,), dtype=tl.int32))
        else:
            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            qb_indices = qb_start + block_offsets
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = NOPE_DIM
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    if not QUANTIZE:
        out_base = (
            out_ptr
            + g * out_stride_group
            + pid_token * out_stride_token
            + qb_start * QUANT_GROUP_SIZE
        )
        tl.store(out_base + offsets, x)
        return

    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    scale_raw = block_absmax * (1.0 / fp8_max)
    # vLLM main always emits pow2 (UE8M0) scales, even for the FP32 layout.
    scales = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant_u8 = _float_to_e4m3fn_bits(tl.clamp(x / scales_exp, -fp8_max, fp8_max)).to(
        tl.uint8
    )

    out_base = (
        out_ptr
        + g * out_stride_group
        + pid_token * out_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    # out_ptr is a uint8 view of the fp8 buffer when QUANTIZE is on.
    tl.store(out_base + offsets, x_quant_u8)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    if TMA_ALIGNED_SCALES:
        scale_bits = scales.to(tl.int32, bitcast=True)
        ue8m0_bytes = (scale_bits >> 23) & 0xFF
        packed_val = tl.sum(
            tl.reshape(ue8m0_bytes, (CHUNKS_PER_HEAD // 4, 4))
            << (tl.arange(0, 4)[None, :] * 8),
            axis=1,
        )
        packed_offsets = tl.arange(0, CHUNKS_PER_HEAD // 4)
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + (head_in_group * (CHUNKS_PER_HEAD // 4) + packed_offsets) * scale_stride_k
        )
        tl.store(scale_addr, packed_val)
    else:
        scale_addrs = (
            scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, scales)


def _get_tma_aligned_size(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


def vllm_fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    tma_aligned_scales: bool = False,
    quantize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standalone copy of the vLLM main ``fused_inv_rope_fp8_quant`` entry point."""
    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    tma_aligned_T = _get_tma_aligned_size(num_tokens, 4) if quantize else num_tokens
    if quantize and tma_aligned_scales:
        assert chunks_per_head % 4 == 0
        packed_sf_k = (num_scale_blocks + 3) // 4
        scale_inner = packed_sf_k
    elif quantize:
        scale_inner = num_scale_blocks
    else:
        scale_inner = 0

    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    out_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=fp8_dtype if quantize else o.dtype,
        device=o.device,
    )
    if quantize:
        scale_buf = torch.empty(
            n_groups * scale_inner * tma_aligned_T,
            dtype=scale_dtype,
            device=o.device,
        ).as_strided(
            (n_groups, num_tokens, scale_inner),
            (scale_inner * tma_aligned_T, 1, tma_aligned_T),
        )
        # FlagTree cannot take a float8_e4m3fn pointer; store through a uint8
        # view of the fp8 buffer (the kernel emits raw e4m3fn bit patterns).
        out_arg = out_buf.view(torch.uint8)
    else:
        scale_buf = torch.empty(0, dtype=scale_dtype, device=o.device)
        out_arg = out_buf

    grid = (tma_aligned_T, n_groups * heads_per_group)
    _vllm_fused_inv_rope_fp8_quant_kernel[grid](
        o,
        positions,
        cos_sin_cache,
        out_arg,
        scale_buf,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        out_stride_group=out_buf.stride(0),
        out_stride_token=out_buf.stride(1),
        scale_stride_group=scale_buf.stride(0) if quantize else 0,
        scale_stride_k=scale_buf.stride(2) if quantize else 0,
        fp8_max=fp8_max,
        eps=1e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        NOPE_DIM=nope_dim,
        HALF_ROPE=rope_dim // 2,
        QUANTIZE=quantize,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        num_stages=1,
        num_warps=1,
    )

    output = out_buf.transpose(0, 1)
    scales = scale_buf.transpose(0, 1) if quantize else scale_buf
    return output, scales
