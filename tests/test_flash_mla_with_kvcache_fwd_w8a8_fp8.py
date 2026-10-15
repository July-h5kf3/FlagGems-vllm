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

import math

import pytest
import torch

from flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8 import (
    HAS_TLE,
    flash_mla_with_kvcache_fwd_w8a8_fp8,
    get_mla_fp8_metadata,
)

from .accuracy_utils import gems_assert_close
from .test_flash_mla_with_kvcache import VLLM_AVAILABLE, _run_cuda

HAS_FP8_MLA = (
    HAS_TLE and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
)


def _quantize_dense_fp8(x, d_nope=512):
    """Per-token FP8 content plus scaled BF16 RoPE, and its BF16 dequantization."""
    scale = x[..., :d_nope].float().abs().amax(-1, keepdim=True) / 448.0
    nope = (x[..., :d_nope].float() / scale).to(torch.float8_e4m3fn)
    rope = (x[..., d_nope:].float() / scale).to(torch.bfloat16)
    dequantized = torch.cat((nope.float(), rope.float()), -1) * scale
    return nope, rope, scale, dequantized.to(torch.bfloat16)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_fp8
@pytest.mark.skipif(not HAS_FP8_MLA, reason="Requires NVIDIA Hopper and TLE")
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM FlashMLA is required")
@pytest.mark.parametrize("seq_q", (1, 2))
@pytest.mark.parametrize("h_q", (16, 64, 128))
def test_dense_decode(seq_q, h_q):
    batch, d_qk = 4, 576
    h_k = 1
    head_dim_v = 512
    page_block_size = 64
    seqlen = 256

    torch.manual_seed(45)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device="cuda")
    max_pages_per_seq = math.ceil(seqlen / page_block_size) + 4
    total_pages = batch * max_pages_per_seq
    kv_cache = (
        torch.randn(
            total_pages, page_block_size, h_k, d_qk, dtype=torch.bfloat16, device="cuda"
        )
        * 0.1
    )
    block_table = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(
        batch, max_pages_per_seq
    )
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device="cuda")
    cache_seqlens[0] = seqlen // 2
    cache_seqlens[-1] = seqlen + page_block_size
    q_nope, q_rope, q_scale, q = _quantize_dense_fp8(q)
    k_nope, k_rope, k_scale, kv_cache = _quantize_dense_fp8(kv_cache)

    cuda_out, cuda_lse = _run_cuda(
        q, kv_cache, block_table, cache_seqlens, head_dim_v, causal=True
    )
    out, lse = flash_mla_with_kvcache_fwd_w8a8_fp8(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
        causal=True,
    )
    gems_assert_close(out, cuda_out, torch.bfloat16, atol=2e-3)
    gems_assert_close(lse, cuda_lse, torch.float32, atol=2e-4)
    # Precomputed GPU split metadata must reproduce the internal plan.
    metadata, num_splits = get_mla_fp8_metadata(cache_seqlens, seq_q * h_q)
    planned_out, planned_lse = flash_mla_with_kvcache_fwd_w8a8_fp8(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
        metadata,
        num_splits,
        causal=True,
    )
    torch.testing.assert_close(planned_out, out, rtol=0, atol=0)
    torch.testing.assert_close(planned_lse, lse, rtol=0, atol=0)
