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

import flaggems_vllm
from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import HAS_TLE

from .accuracy_utils import gems_assert_close, to_reference

try:
    from vllm.third_party.flashmla.flash_mla_interface import (
        flash_mla_with_kvcache as cuda_flash_mla,
    )
    from vllm.third_party.flashmla.flash_mla_interface import (
        get_mla_metadata as cuda_get_mla_metadata,
    )

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

OUTPUT_ATOL = 2e-3
SUPPORTED = (
    HAS_TLE and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
)
ASCEND_INT8 = flaggems_vllm.flash_mla_with_kvcache_w8a8.__module__ == (
    "flaggems_vllm.runtime.backend._ascend.ops.flash_mla_with_kvcache_w8a8"
)


def quantize_dense_fp8(x, *, dequantize=True):
    scale = x[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    nope = (x[..., :512].float() / scale).to(torch.float8_e4m3fn)
    rope = (x[..., 512:].float() / scale).to(torch.bfloat16)
    if dequantize:
        native = (torch.cat((nope.float(), rope.float()), -1) * scale).to(
            torch.bfloat16
        )
    else:
        native = None
    return nope, rope, scale, native


def quantize_dense_int8(x):
    content = x[..., :512].float()
    scale = content.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    integers = (content / scale).round().clamp(-127, 127).to(torch.int8)
    rope = (x[..., 512:].float() / scale).to(torch.bfloat16)
    return integers, rope, scale


def torch_reference(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    head_dim_v,
    *,
    softmax_scale=None,
    causal=False,
):
    batch, queries, _, dim = q.shape
    scale = dim**-0.5 if softmax_scale is None else softmax_scale
    cache = k_cache[block_table.long()].reshape(batch, -1, dim).float()
    tokens = torch.arange(cache.shape[1], device=q.device)
    valid = tokens[None, :] < cache_seqlens[:, None]
    cache = cache.masked_fill(~valid[:, :, None], 0)
    scores = torch.einsum("bqhd,bnd->bqhn", q.float(), cache) * scale
    allowed = valid[:, None, :]
    if causal:
        query_positions = (
            cache_seqlens[:, None]
            - queries
            + torch.arange(queries, device=q.device)[None, :]
        )
        allowed = allowed & (tokens[None, None, :] <= query_positions[:, :, None])
    else:
        pass
    scores = scores.masked_fill(~allowed[:, :, None, :], -torch.inf)
    probabilities = torch.where(allowed[:, :, None, :], scores.softmax(-1), 0)
    output = torch.einsum("bqhn,bnd->bqhd", probabilities, cache[..., :head_dim_v]).to(
        q.dtype
    )
    return output, torch.logsumexp(scores, -1).transpose(1, 2)


def vllm_reference(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    metadata, _ = cuda_get_mla_metadata()
    return cuda_flash_mla(
        q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, **kwargs
    )


def assert_dense_fp8_close(actual, reference):
    reference = tuple(to_reference(tensor) for tensor in reference)
    if reference[0].device.type == "cpu":
        actual = tuple(tensor.cpu() for tensor in actual)
    else:
        pass
    gems_assert_close(actual[0], reference[0], torch.bfloat16, atol=OUTPUT_ATOL)
    gems_assert_close(actual[1], reference[1], torch.float32, atol=2e-4)


REFERENCES = (torch_reference, vllm_reference) if VLLM_AVAILABLE else (torch_reference,)


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(not SUPPORTED, reason="FP8 MLA requires NVIDIA Hopper and TLE")
@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize(
    "batch,seqlen,varlen",
    [
        (4, 128, False),
        (4, 256, True),
        (2, 33280, False),
        (8, 33280, False),
        (16, 8192, False),
    ],
)
@pytest.mark.parametrize("seq_q", (1, 2))
@pytest.mark.parametrize("h_q", (16, 32, 64, 128))
def test_dense_decode(reference, seq_q, h_q, batch, seqlen, varlen):
    head_dim_v, page_block_size = 512, 64
    torch.manual_seed(45)
    q = torch.randn(
        batch, seq_q, h_q, 576, dtype=torch.bfloat16, device=flaggems_vllm.device
    )
    max_pages_per_seq = math.ceil(seqlen / page_block_size) + 4
    total_pages = batch * max_pages_per_seq
    cache = (
        torch.randn(
            total_pages,
            page_block_size,
            1,
            576,
            dtype=torch.bfloat16,
            device=flaggems_vllm.device,
        )
        * 0.1
    )
    table = torch.arange(
        total_pages, dtype=torch.int32, device=flaggems_vllm.device
    ).view(batch, max_pages_per_seq)
    host_lengths = [seqlen] * batch
    if varlen:
        host_lengths[0] = seqlen // 2
        host_lengths[-1] = seqlen + page_block_size
    else:
        pass
    lengths = torch.tensor(host_lengths, dtype=torch.int32, device=flaggems_vllm.device)
    q_nope, q_rope, q_scale, native_q = quantize_dense_fp8(q)
    k_nope, k_rope, k_scale, native_cache = quantize_dense_fp8(cache)
    actual = flaggems_vllm.flash_mla_with_kvcache_w8a8(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_scale,
        k_scale,
        table,
        lengths,
        head_dim_v,
        causal=True,
    )
    expected = reference(
        native_q, native_cache, table, lengths, head_dim_v, causal=True
    )
    assert_dense_fp8_close(actual, expected)


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(not ASCEND_INT8, reason="Requires the Ascend INT8 MLA backend")
@pytest.mark.parametrize("seq_q,causal", [(1, True), (2, True), (2, False)])
@pytest.mark.parametrize("heads", (16, 32))
def test_dense_decode_ascend(seq_q, causal, heads):
    torch.manual_seed(45)
    device = flaggems_vllm.device
    batch = 3
    q, qr, qs = quantize_dense_int8(
        torch.randn(batch, seq_q, heads, 576, device=device)
    )
    k, kr, ks = quantize_dense_int8(torch.randn(5, 64, 1, 576, device=device))
    table = torch.tensor([[3, 0], [4, 2], [1, 0]], dtype=torch.int32, device=device)
    lengths = torch.tensor([121, 65, 0], dtype=torch.int32, device=device)
    kr[2, 1:] = float("nan")
    ks[2, 1:] = float("nan")
    actual, lse = flaggems_vllm.flash_mla_with_kvcache_w8a8(
        q, qr, k, kr, qs, ks, table, lengths, 512, causal=causal
    )
    query = torch.cat((q.float(), qr.float()), -1) * qs
    cache = torch.cat((k.float(), kr.float()), -1) * ks
    expected, expected_lse = torch_reference(
        query, cache, table, lengths, 512, causal=causal
    )
    gems_assert_close(actual, expected.to(actual.dtype), actual.dtype, atol=OUTPUT_ATOL)
    gems_assert_close(lse[:2], expected_lse[:2], torch.float32, atol=2e-4)
    gems_assert_close(lse[2], torch.full_like(lse[2], float("inf")), torch.float32)


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(not ASCEND_INT8, reason="Requires the Ascend INT8 MLA backend")
@pytest.mark.parametrize("heads,seq_q", [(16, 1), (16, 2), (64, 2)])
def test_dense_decode_ascend_paged_windows(heads, seq_q):
    torch.manual_seed(45)
    device = flaggems_vllm.device
    q, qr, qs = quantize_dense_int8(torch.randn(2, seq_q, heads, 576, device=device))
    k, kr, ks = quantize_dense_int8(torch.randn(12, 64, 1, 576, device=device))
    table = torch.tensor(
        [[5, 8, 2, 0, 4, 11, 1], [5, 9, 6, 3, 10, 7, 0]],
        dtype=torch.int32,
        device=device,
    )
    lengths = torch.tensor([385, 448], dtype=torch.int32, device=device)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(2, heads, seq_q, dtype=torch.float32, device=device)
    actual, actual_lse = flaggems_vllm.flash_mla_with_kvcache_w8a8(
        q, qr, k, kr, qs, ks, table, lengths, 512, causal=True, out=out, lse=lse
    )
    query = torch.cat((q.float(), qr.float()), -1) * qs
    cache = torch.cat((k.float(), kr.float()), -1) * ks
    expected, expected_lse = torch_reference(
        query, cache, table, lengths, 512, causal=True
    )
    assert actual is out and actual_lse is lse
    torch.testing.assert_close(out.float(), expected, rtol=0, atol=0.15)
    torch.testing.assert_close(lse, expected_lse, rtol=0, atol=2e-4)
