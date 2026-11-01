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

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry


@libentry()
@triton.jit(do_not_specialize=["scale"])
def fused_indexer_weight_scale_int8_kernel(
    weights_ptr,
    weight_scale_ptr,
    q_scale_ptr,
    output_ptr,
    num_tokens,
    scale,
    HEADS: tl.constexpr,
    WEIGHT_STRIDE_T: tl.constexpr,
    WEIGHT_STRIDE_H: tl.constexpr,
    SCALE_STRIDE_T: tl.constexpr,
    QUERY_STRIDE_T: tl.constexpr,
    QUERY_STRIDE_H: tl.constexpr,
    DENSE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    pid = tl.program_id(0)
    if DENSE:
        head_tiles: tl.constexpr = (HEADS + BLOCK_HEADS - 1) // BLOCK_HEADS
        tokens = (pid // head_tiles) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        heads = (pid % head_tiles) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
        mask = (tokens[:, None] < num_tokens) & (heads[None, :] < HEADS)
        weight_offsets = tokens[:, None] * WEIGHT_STRIDE_T + heads[None, :]
        query_offsets = tokens[:, None] * QUERY_STRIDE_T + heads[None, :]
        output_offsets = tokens[:, None] * HEADS + heads[None, :]
        weight_scale = tl.load(
            weight_scale_ptr + tokens * SCALE_STRIDE_T,
            mask=tokens < num_tokens,
            other=0.0,
        )[:, None]
    else:
        output_offsets = pid * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
        mask = output_offsets < num_tokens * HEADS
        tokens = output_offsets // HEADS
        heads = output_offsets % HEADS
        weight_offsets = tokens * WEIGHT_STRIDE_T + heads * WEIGHT_STRIDE_H
        query_offsets = tokens * QUERY_STRIDE_T + heads * QUERY_STRIDE_H
        weight_scale = tl.load(
            weight_scale_ptr + tokens * SCALE_STRIDE_T, mask=mask, other=0.0
        )
    weights = tl.load(weights_ptr + weight_offsets, mask=mask, other=0).to(tl.float32)
    query_scale = tl.load(q_scale_ptr + query_offsets, mask=mask, other=0.0)
    # Match the BF16 weights consumed by the native reference before promotion.
    dequantized = (weights * weight_scale).to(tl.bfloat16).to(tl.float32)
    output = dequantized * query_scale * scale
    tl.store(output_ptr + output_offsets, output, mask=mask)


def fused_indexer_weight_scale_w8a16_int8(
    weights: torch.Tensor,
    weight_scale: torch.Tensor,
    q_scale: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Scale symmetric INT8 [T,H] weights with per-token FP32 [T,1] scales.

    Weights are rounded to BF16 after dequantization. Query scales are FP32
    [T,H,1]; the new contiguous output is FP32 [T,H]. Forward inference only.
    """
    if weights.ndim != 2:
        raise ValueError("weights must have shape [tokens, heads]")
    tokens, heads = weights.shape
    if weight_scale.shape != (tokens, 1) or q_scale.shape != (tokens, heads, 1):
        raise ValueError("weight_scale must be [T,1] and q_scale must be [T,H,1]")
    if (
        weights.dtype != torch.int8
        or weight_scale.dtype != torch.float32
        or q_scale.dtype != torch.float32
    ):
        raise TypeError("weights must be INT8 and both scale tensors must be FP32")
    if weights.device.type != "npu":
        raise NotImplementedError("INT8 indexer weight scaling requires Ascend")
    if weight_scale.device != weights.device or q_scale.device != weights.device:
        raise ValueError("all input tensors must be on the same device")
    output = torch.empty((tokens, heads), dtype=torch.float32, device=weights.device)
    total = tokens * heads
    if total == 0:
        return output
    is_dense = (
        weights.is_contiguous()
        and weight_scale.is_contiguous()
        and q_scale.is_contiguous()
    )
    if is_dense:
        block_heads = min(128, triton.next_power_of_2(heads))
        tile_elements = 4096 if total >= 65536 else 1024
        block_rows = min(triton.next_power_of_2(tokens), tile_elements // block_heads)
        grid = (triton.cdiv(tokens, block_rows) * triton.cdiv(heads, block_heads),)
    else:
        block_heads = min(1024, triton.next_power_of_2(total))
        block_rows = 1
        grid = (triton.cdiv(total, block_heads),)
    fused_indexer_weight_scale_int8_kernel[grid](
        weights,
        weight_scale,
        q_scale,
        output,
        tokens,
        scale,
        heads,
        weights.stride(0),
        weights.stride(1),
        weight_scale.stride(0),
        q_scale.stride(0),
        q_scale.stride(1),
        is_dense,
        block_rows,
        block_heads,
    )
    return output
