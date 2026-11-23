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


def flash_mla_with_kvcache_w8a8(
    q_nope,
    q_rope,
    k_cache_lora,
    k_cache_rope,
    q_scale,
    k_scale,
    block_table,
    cache_seqlens,
    head_dim_v,
    tile_scheduler_metadata=None,
    num_splits=None,
    softmax_scale=None,
    causal=False,
    out=None,
    lse=None,
):
    """Paged dense MLA with signed INT8 content and scaled BF16 RoPE."""
    if tile_scheduler_metadata is not None or num_splits is not None:
        raise NotImplementedError("Ascend dense MLA selects its own scheduling")
    if q_nope.ndim != 4 or k_cache_lora.ndim not in (3, 4) or block_table.ndim != 2:
        raise ValueError("Expected rank-4 query, paged KV cache and rank-2 block table")
    batch, queries, heads, dim = q_nope.shape
    cache_prefix = k_cache_lora.shape[:-1]
    if (
        queries not in (1, 2)
        or heads not in (16, 32, 64, 128)
        or dim != 512
        or head_dim_v != 512
        or k_cache_lora.shape[-1] != 512
        or cache_prefix[1:] not in ((64,), (64, 1))
    ):
        raise ValueError(
            "Expected Sq=1/2, H=16/32/64/128, page=64, Dv=512 and one KV head"
        )
    tensors = (
        (q_nope, q_nope.shape, torch.int8),
        (q_rope, (*q_nope.shape[:-1], 64), torch.bfloat16),
        (q_scale, (*q_nope.shape[:-1], 1), torch.float32),
        (k_cache_lora, k_cache_lora.shape, torch.int8),
        (k_cache_rope, (*cache_prefix, 64), torch.bfloat16),
        (k_scale, (*cache_prefix, 1), torch.float32),
        (block_table, (batch, block_table.shape[1]), torch.int32),
        (cache_seqlens, (batch,), torch.int32),
    )
    device = q_nope.device
    if device.type != "npu":
        raise ValueError("Ascend dense MLA requires NPU tensors")
    for tensor, shape, dtype in tensors:
        if tensor.device != device or tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(f"Expected {shape}, {dtype} on {device}")
        if not tensor.is_contiguous():
            raise ValueError("Ascend dense MLA requires contiguous input tensors")
    if out is None:
        out = torch.empty(q_nope.shape, device=device, dtype=q_rope.dtype)
    if lse is None:
        lse = torch.empty((batch, heads, queries), device=device, dtype=torch.float32)
    for tensor, shape, dtype in (
        (out, q_nope.shape, q_rope.dtype),
        (lse, (batch, heads, queries), torch.float32),
    ):
        if (
            tensor.device != device
            or tensor.shape != shape
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"Expected a contiguous output {shape}, {dtype} on {device}"
            )
    if batch:
        # Only this operator requires the FlagTree Ascend custom-op interfaces.
        from flaggems_vllm.runtime.backend._ascend.fused import mla_w8a8

        with torch.npu.device(device.index):
            scale = 576**-0.5 if softmax_scale is None else float(softmax_scale)
            tensors = (
                q_nope,
                q_rope,
                k_cache_lora,
                k_cache_rope,
                q_scale,
                k_scale,
                block_table,
                cache_seqlens,
            )
            width = block_table.shape[1]
            if width == 0:
                out.zero_()
                lse.fill_(float("inf"))
            elif width <= 6 and mla_w8a8.requires_portable_short():
                rows = batch * queries * heads
                page_heads = max(32, heads)
                page_rows = batch * queries * page_heads
                if heads == 16:
                    q32, qr32, qs32 = (
                        t.unsqueeze(2)
                        .expand(-1, -1, 2, -1, -1)
                        .contiguous()
                        .view(batch, queries, page_heads, t.shape[-1])
                        for t in (q_nope, q_rope, q_scale)
                    )
                    tensors = (
                        q32,
                        qr32,
                        k_cache_lora,
                        k_cache_rope,
                        qs32,
                        k_scale,
                        block_table,
                        cache_seqlens,
                    )
                partial = torch.empty(
                    (page_rows, width, 512), device=device, dtype=torch.float32
                )
                stats = torch.empty(
                    (page_rows, width, 2), device=device, dtype=torch.float32
                )
                tasks = batch * queries * triton.cdiv(heads, 32) * width * 4
                mla_w8a8.portable_page[(tasks,)](
                    *tensors, partial, stats, page_heads, queries, width, scale, causal
                )
                mla_w8a8.portable_merge[(rows,)](
                    partial,
                    stats,
                    out,
                    lse,
                    heads,
                    queries,
                    width,
                    triton.next_power_of_2(width),
                    page_heads,
                )
            elif width <= 6:
                tasks = batch * queries * (heads // 16)
                mla_w8a8.launch(
                    mla_w8a8.short_decode,
                    tasks,
                    (*tensors, out, lse),
                    (heads, queries, width, scale, causal),
                )
            else:
                window_rows = min(64, heads * queries)
                tasks = batch * (heads * queries // window_rows)
                cores = min(20, tasks)
                plan_stride = (
                    max(8, triton.next_power_of_2(triton.cdiv(tasks, cores)))
                    if window_rows <= 32 and batch <= 128
                    else 0
                )
                c, r, p, pv = mla_w8a8.allocate_scratch(
                    device,
                    cores * 2 * window_rows * 384,
                    cores * (2 * window_rows * 512 + plan_stride),
                )
                mla_w8a8.launch(
                    mla_w8a8.window_pipeline,
                    cores,
                    (*tensors, c, r, p, pv, out, lse),
                    (
                        tasks,
                        width,
                        cores,
                        scale,
                        heads,
                        queries,
                        causal,
                        window_rows,
                        batch,
                        plan_stride,
                    ),
                )
    return out, lse
