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
import triton
import triton.language as tl

import flaggems_vllm

from . import conftest as cfg
from .accuracy_utils import gems_assert_close, gems_assert_equal


def quantize_mla_content(tensor):
    scale = (
        tensor[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    )
    content = (
        (tensor[..., :512].float() / scale).round().clamp(-127, 127).to(torch.int8)
    )
    rope = (tensor[..., 512:].float() / scale).to(tensor.dtype)
    return content, rope, scale


def make_dense_int8_inputs(
    batch,
    heads,
    length,
    dtype=torch.bfloat16,
    *,
    ragged=False,
    strided=False,
    magnitude=0.3,
    extra_pages=0,
    queries=1,
    seed=42,
):
    torch.manual_seed(seed)
    pages = max(1, math.ceil(length / 64)) + extra_pages
    query = (
        torch.randn(
            batch, queries, heads, 576, dtype=dtype, device=flaggems_vllm.device
        )
        * magnitude
    )
    cache = (
        torch.randn(batch * pages, 64, 576, dtype=dtype, device=query.device)
        * magnitude
    )
    q, qr, qs = quantize_mla_content(query)
    kv, kr, ks = quantize_mla_content(cache)
    table = (
        torch.arange(batch * pages, dtype=torch.int32, device=query.device)
        .reshape(batch, pages)
        .flip(1)
    )
    lengths = torch.full((batch,), length, dtype=torch.int32, device=query.device)
    if ragged:
        lengths = torch.tensor(
            [0 if i == 0 else max(1, length - i * 17) for i in range(batch)],
            device=query.device,
            dtype=torch.int32,
        )
    if strided:
        outer_views = []
        for tensor in (q, qr, kv, kr, qs, ks, table):
            storage = torch.empty(
                (tensor.shape[0] * 2, *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            view = storage[::2]
            view.copy_(tensor)
            outer_views.append(view)
        q, qr, kv, kr, qs, ks, table = outer_views
    return (q, qr, kv, kr, qs, ks, table, lengths)


def dense_int8_mla_reference(tensors, causal=False):
    q, qr, kv, kr, qs, ks, table, lengths = tensors
    query = torch.cat((q.float(), qr.float()), -1) * qs
    cache = torch.cat((kv.float(), kr.float()), -1) * ks
    return dense_mla_reference(query, cache, table, lengths, causal)


def dense_mla_reference(query, cache, table, lengths, causal=False):
    batch, queries, heads, _ = query.shape
    output = torch.zeros(
        (batch, queries, heads, 512), dtype=torch.float32, device=query.device
    )
    lse = torch.full(
        (batch, heads, queries), float("inf"), dtype=torch.float32, device=query.device
    )
    for row in range(batch):
        length = int(lengths[row].item())
        if length == 0:
            continue
        cache_row = cache[table[row].long()].reshape(-1, 576)[:length].float()
        for query_index in range(queries):
            visible = max(0, length - queries + query_index + 1) if causal else length
            if visible == 0:
                continue
            active_cache = cache_row[:visible]
            logits = query[row, query_index].float() @ active_cache.T * 576**-0.5
            output[row, query_index] = logits.softmax(-1) @ active_cache[:, :512]
            lse[row, :, query_index] = logits.logsumexp(-1)
    return output, lse


SUPPORTED = "flash_mla_with_kvcache_fwd_w8a8_int8" in dict(
    flaggems_vllm.runtime.backend.get_customized_ops()
)
pytestmark = [
    pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8,
    pytest.mark.skipif(
        not SUPPORTED, reason="backend has no dense INT8 MLA implementation"
    ),
]
CASES = (
    [(1, 64, 65), (3, 17, 129)]
    if cfg.QUICK_MODE
    else [
        (2, 64, 0),
        (1, 1, 1),
        (1, 64, 63),
        (1, 64, 64),
        (1, 64, 65),
        (3, 17, 129),
        (2, 128, 640),
        (1, 64, 8192),
    ]
)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.parametrize("batch,heads,length", CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splits", [1, 4])
def test_flash_mla_with_kvcache_fwd_w8a8_int8(batch, heads, length, dtype, splits):
    tensors = make_dense_int8_inputs(
        batch, heads, length, dtype, ragged=batch > 2, strided=True
    )
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, num_splits=splits
    )
    expected, expected_lse = dense_int8_mla_reference(tensors)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_mla_int8_large_head_tiles(dtype):
    tensors = make_dense_int8_inputs(128, 128, 257, dtype)
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, num_splits=1
    )
    expected, expected_lse = dense_int8_mla_reference(tensors)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


def test_flash_mla_int8_concentrated_probability():
    tensors = make_dense_int8_inputs(2, 64, 257, magnitude=2.0)
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, num_splits=4
    )
    expected, expected_lse = dense_int8_mla_reference(tensors)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.02)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.parametrize("batch,heads", [(0, 64), (2, 0)])
@pytest.mark.parametrize("queries", [1, 2])
def test_flash_mla_int8_empty_batch(batch, heads, queries):
    tensors = make_dense_int8_inputs(batch, heads, 64, queries=queries)
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(*tensors)
    assert output.shape == (batch, queries, heads, 512) and lse.shape == (
        batch,
        heads,
        queries,
    )


@pytest.mark.parametrize("queries", [1, 2])
def test_flash_mla_int8_prepared_length_update(queries):
    tensors = make_dense_int8_inputs(2, 64, 257, queries=queries)
    handle, _ = flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, causal=True
    )
    tensors[-1].fill_(65)
    output, lse = handle()
    expected, expected_lse = dense_int8_mla_reference(tensors, True)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


def test_flash_mla_int8_large_cache_address():
    tensors = make_dense_int8_inputs(1, 64, 64)
    q, qr, kv, kr, qs, ks, table, lengths = tensors
    pages = 65538
    cache = torch.empty((pages, 64, 512), dtype=kv.dtype, device=kv.device)
    rope_cache = torch.empty((pages, 64, 64), dtype=kr.dtype, device=kr.device)
    scales = torch.empty((pages, 64, 1), dtype=ks.dtype, device=ks.device)
    cache[-1].copy_(kv[0])
    rope_cache[-1].copy_(kr[0])
    scales[-1].copy_(ks[0])
    high_page = torch.full_like(table, pages - 1)
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        q, qr, cache, rope_cache, qs, scales, high_page, lengths
    )
    expected, expected_lse = dense_int8_mla_reference(tensors)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


def test_flash_mla_int8_prefill_is_unsupported():
    tensors = list(make_dense_int8_inputs(1, 64, 64))
    tensors[0] = tensors[0].expand(1, 3, 64, 512)
    with pytest.raises(NotImplementedError, match="requires Q"):
        flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(*tensors)


@triton.jit
def probability_quantization_kernel(
    Input,
    Coarse,
    Residual,
    N: tl.constexpr,
    QUANTIZER: tl.constexpr,
    USE_PTX: tl.constexpr,
):
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    probability = tl.load(Input + offsets, offsets < N, 0.0)
    if USE_PTX:
        coarse, residual = QUANTIZER(probability)
    else:
        coarse = tl.floor(probability + 0.5).to(tl.int8)
        residual = tl.floor((probability - coarse.to(tl.float32)) * 127.0 + 0.5).to(
            tl.int8
        )
    tl.store(Coarse + offsets, coarse, offsets < N)
    tl.store(Residual + offsets, residual, offsets < N)


def test_flash_mla_int8_probability_quantization():
    from flaggems_vllm.runtime.backend._thead.ops.flash_mla_with_kvcache_fwd_w8a8_int8 import (
        quantize_mla_probabilities,
        quantize_mla_probabilities_packed,
    )

    torch.manual_seed(42)
    device = flaggems_vllm.device
    halves = torch.arange(127, device=device, dtype=torch.float32) + 0.5
    boundaries = torch.cat(
        (
            torch.nextafter(halves, torch.full_like(halves, -float("inf"))),
            halves,
            torch.nextafter(halves, torch.full_like(halves, float("inf"))),
        )
    )
    residual_halves = (
        64 + (torch.arange(-63, 64, device=device, dtype=torch.float32) + 0.5) / 127
    )
    small = torch.tensor(
        [0, 1, 0x007FFFFF, 0x00800000], dtype=torch.int32, device=device
    ).view(torch.float32)
    probabilities = torch.cat(
        (
            torch.rand(262144, device=device) * 127,
            boundaries,
            residual_halves,
            small,
            torch.tensor([127.0], device=device),
        )
    )
    coarse_reference = torch.empty_like(probabilities, dtype=torch.int8)
    residual_reference = torch.empty_like(coarse_reference)
    coarse = torch.empty_like(coarse_reference)
    residual = torch.empty_like(coarse_reference)
    grid = (triton.cdiv(probabilities.numel(), 1024),)
    probability_quantization_kernel[grid](
        probabilities,
        coarse_reference,
        residual_reference,
        probabilities.numel(),
        quantize_mla_probabilities,
        False,
    )
    for quantizer in (quantize_mla_probabilities, quantize_mla_probabilities_packed):
        probability_quantization_kernel[grid](
            probabilities,
            coarse,
            residual,
            probabilities.numel(),
            quantizer,
            True,
        )
        gems_assert_equal(coarse, coarse_reference)
        gems_assert_equal(residual, residual_reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splits", [1, 4])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_mla_int8_two_queries(dtype, splits, causal):
    tensors = make_dense_int8_inputs(
        3, 17, 129, dtype, queries=2, ragged=True, strided=True
    )
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, causal=causal, num_splits=splits
    )
    expected, expected_lse = dense_int8_mla_reference(tensors, causal)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.parametrize("length", [0, 1, 2, 65])
def test_flash_mla_int8_causal_query_boundaries(length):
    tensors = make_dense_int8_inputs(1, 16, length, queries=2)
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, causal=True, num_splits=4
    )
    expected, expected_lse = dense_int8_mla_reference(tensors, True)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)


@pytest.mark.parametrize("queries", [1, 2])
def test_flash_mla_int8_nan_cache_tail(queries):
    tensors = make_dense_int8_inputs(2, 32, 65, queries=queries)
    # The page table reverses physical pages, so logical positions 65..127 are in page zero.
    for row in range(2):
        page = row * 2
        tensors[3][page, 1:].fill_(float("nan"))
        tensors[5][page, 1:].fill_(float("nan"))
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, causal=True, num_splits=4
    )
    expected, expected_lse = dense_int8_mla_reference(tensors, True)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)
