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


"""Input construction and numerical references shared by MLA FP8 tests and benchmarks."""

import math

import torch

from flaggems_vllm.ops.flash_mla_fp8.flash_mla_with_kvcache_fwd_w8a8_fp8 import (
    quantize_k_ckv_per_token,
    quantize_q_ckv_per_token,
)


def make_dense_inputs(
    batch,
    heads,
    seqlen,
    *,
    page_multiple=1,
    seed=42,
    dtype=torch.bfloat16,
    device="cuda",
    magnitude=0.1,
):
    pages_per_row = math.ceil(seqlen / (64 * page_multiple)) * page_multiple
    total_pages = batch * pages_per_row
    if seed is not None:
        torch.manual_seed(seed)
    else:
        pass
    query = torch.randn(batch, 1, heads, 576, dtype=dtype, device=device) * magnitude
    cache = torch.randn(total_pages, 64, 576, dtype=dtype, device=device) * magnitude
    q_nope, q_rope, q_scale = quantize_q_ckv_per_token(query)
    k_lora, k_rope, k_scale = quantize_k_ckv_per_token(cache)
    block_table = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        batch, pages_per_row
    )
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=device)
    return dict(
        q=query,
        blocked_k=cache,
        q_nope=q_nope,
        q_rope=q_rope,
        q_scale=q_scale,
        k_lora=k_lora,
        k_rope=k_rope,
        k_scale=k_scale,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        lengths=(seqlen,) * batch,
    )


def make_sparse_inputs(batch, heads, topk, seed=42, magnitude=0.1):
    torch.manual_seed(seed)
    pages = (topk + 63) // 64 + 4
    query = (
        torch.randn(batch, 1, heads, 576, device="cuda", dtype=torch.bfloat16)
        * magnitude
    )
    cache = torch.randn(pages, 64, 576, device="cuda", dtype=torch.bfloat16) * magnitude
    q_nope, q_rope, q_scale = quantize_q_ckv_per_token(query)
    k_nope, k_rope, k_scale = quantize_k_ckv_per_token(cache)
    indices = torch.randint(
        pages * 64, (batch, 1, topk), device="cuda", dtype=torch.int32
    )
    return [q_nope, q_rope, k_nope, k_rope, q_scale, k_scale, indices], query, cache


def sparse_reference(inputs, attn_sink=None, topk_length=None, softmax_scale=None):
    q_nope, q_rope, k_nope, k_rope, q_scale, k_scale, indices = inputs
    query = torch.cat((q_nope.float(), q_rope.float()), -1) * q_scale
    cache = (torch.cat((k_nope.float(), k_rope.float()), -1) * k_scale).reshape(-1, 576)
    batch, _, heads, _ = query.shape
    output = torch.zeros((batch, 1, heads, 512), device="cuda", dtype=torch.float32)
    lse = torch.full((batch, heads, 1), float("inf"), device="cuda")
    for row in range(batch):
        length = (
            indices.shape[-1] if topk_length is None else max(0, int(topk_length[row]))
        )
        selected = indices[row, 0, :length].long()
        selected = selected[(selected >= 0) & (selected < cache.shape[0])]
        if selected.numel() == 0:
            continue
        keys = cache[selected]
        scale = 576**-0.5 if softmax_scale is None else softmax_scale
        logits = query[row, 0] @ keys.T * scale
        row_lse = torch.logsumexp(logits, -1)
        value = logits.softmax(-1) @ keys[:, :512]
        if attn_sink is not None:
            value *= torch.sigmoid(row_lse - attn_sink)[:, None]
        output[row, 0] = value
        lse[row, :, 0] = row_lse
    return output, lse


def assert_sparse_accuracy(output, lse, expected, expected_lse):
    relative_l2 = (
        output.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-12)
    assert relative_l2.item() < 0.05, relative_l2.item()
    torch.testing.assert_close(lse, expected_lse, atol=0.025, rtol=0.002)


def pack_cuda_sparse_fp8_cache(
    k_nope: torch.Tensor, k_scale: torch.Tensor, k_rope: torch.Tensor
) -> torch.Tensor:
    """Pack the CUDA sparse layout without changing the per-token NoPE scale."""
    packed = torch.empty(
        (*k_nope.shape[:2], 1, 656), device=k_nope.device, dtype=torch.uint8
    )
    token_bytes = packed[:, :, 0]
    token_bytes[..., :512].copy_(k_nope.view(torch.uint8))
    scales = k_scale.expand(*k_scale.shape[:2], 4).contiguous()
    token_bytes[..., 512:528].copy_(scales.view(torch.uint8))
    token_bytes[..., 528:].copy_(k_rope.contiguous().view(torch.uint8))
    return packed


def dense_reference(inputs):
    q = inputs["q"].float()
    blocked_k = inputs["blocked_k"].float()
    block_table = inputs["block_table"]
    cache_seqlens = inputs["cache_seqlens"]
    batch, _, h_q, _ = q.shape
    out = torch.empty(batch, 1, h_q, 512, dtype=torch.float32, device=q.device)
    lse = torch.empty(batch, h_q, 1, dtype=torch.float32, device=q.device)
    softmax_scale = 576**-0.5

    for batch_idx in range(batch):
        seqlen = int(cache_seqlens[batch_idx].item())
        page_count = math.ceil(seqlen / 64)
        page_ids = block_table[batch_idx, :page_count].long()
        kv = blocked_k.index_select(0, page_ids).reshape(-1, 576)[:seqlen]
        scores = torch.matmul(q[batch_idx, 0], kv.transpose(0, 1))
        scores *= softmax_scale
        probabilities = torch.softmax(scores, dim=-1)
        out[batch_idx, 0] = torch.matmul(probabilities, kv[:, :512])
        lse[batch_idx, :, 0] = torch.logsumexp(scores, dim=-1)
    return out, lse


def assert_dense_accuracy(out, lse, ref_out, ref_lse):
    out_f32 = out.float()
    rel_l2 = torch.linalg.vector_norm(out_f32 - ref_out) / torch.linalg.vector_norm(
        ref_out
    ).clamp_min(1e-12)
    cosine_distance = 1.0 - torch.nn.functional.cosine_similarity(
        out_f32.flatten(), ref_out.flatten(), dim=0
    )
    lse_max_abs = (lse.float() - ref_lse).abs().max()
    assert rel_l2.item() <= 5e-2
    assert cosine_distance.item() <= 1e-3
    assert lse_max_abs.item() <= 2e-2
