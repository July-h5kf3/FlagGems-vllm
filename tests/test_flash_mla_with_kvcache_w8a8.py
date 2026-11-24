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

from . import conftest as cfg
from .accuracy_utils import gems_assert_close, gems_assert_equal, to_reference

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
WINDOW_ATOL = 1e-2
SUPPORTED = (
    HAS_TLE and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
)
IS_ASCEND_MODULE = flaggems_vllm.flash_mla_with_kvcache_w8a8.__module__ == (
    "flaggems_vllm.runtime.backend._ascend.ops.flash_mla_with_kvcache_w8a8"
)
if IS_ASCEND_MODULE:
    from flaggems_vllm.runtime.backend._ascend.ops.flash_mla_with_kvcache_w8a8 import (
        ascend_mla_available,
    )

    ASCEND_INT8 = ascend_mla_available()
else:
    ASCEND_INT8 = False


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
    gems_assert_close(actual[0], reference[0], torch.bfloat16, atol=OUTPUT_ATOL)
    gems_assert_close(actual[1], reference[1], torch.float32, atol=2e-4)


def assert_lse_close(actual_lse, expected_lse, atol=2e-4):
    expected_inf = expected_lse.masked_fill(torch.isneginf(expected_lse), torch.inf)
    inf_mask = torch.isinf(expected_inf)
    gems_assert_equal(torch.isinf(actual_lse), inf_mask)
    finite_mask = ~inf_mask
    if finite_mask.any():
        gems_assert_close(
            actual_lse[finite_mask],
            expected_inf[finite_mask],
            torch.float32,
            atol=atol,
        )


REFERENCES = (torch_reference, vllm_reference) if VLLM_AVAILABLE else (torch_reference,)
DECODE_SHAPES = (
    [(4, 128, False), (4, 256, True)]
    if cfg.QUICK_MODE
    else [
        (4, 128, False),
        (4, 256, True),
        (2, 33280, False),
        (8, 33280, False),
        (16, 8192, False),
    ]
)
ASCEND_WINDOW_CASES = (
    [(16, 1, True), (64, 2, True)]
    if cfg.QUICK_MODE
    else [
        (16, 1, True),
        (16, 2, False),
        (32, 1, True),
        (64, 1, True),
        (64, 2, True),
        (128, 2, True),
    ]
)
SPLIT_COUNTS = [1, 4] if cfg.QUICK_MODE else [1, 2, 4, 8, 16]


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(not SUPPORTED, reason="FP8 MLA requires NVIDIA Hopper and TLE")
@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("batch,seqlen,varlen", DECODE_SHAPES)
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
@pytest.mark.skipif(not ASCEND_INT8, reason="Requires Ascend910B INT8 MLA backend")
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
    assert_lse_close(lse, expected_lse)

    empty_table = torch.empty((batch, 0), dtype=torch.int32, device=device)
    zero_lengths = torch.zeros((batch,), dtype=torch.int32, device=device)
    empty_out, empty_lse = flaggems_vllm.flash_mla_with_kvcache_w8a8(
        q, qr, k, kr, qs, ks, empty_table, zero_lengths, 512, causal=causal
    )
    gems_assert_close(
        empty_out, torch.zeros_like(empty_out), empty_out.dtype, atol=OUTPUT_ATOL
    )
    gems_assert_equal(
        torch.isinf(empty_lse), torch.ones_like(empty_lse, dtype=torch.bool)
    )


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(not ASCEND_INT8, reason="Requires Ascend910B INT8 MLA backend")
@pytest.mark.parametrize("heads,seq_q,causal", ASCEND_WINDOW_CASES)
def test_dense_decode_ascend_paged_windows(heads, seq_q, causal):
    torch.manual_seed(45)
    device = flaggems_vllm.device
    batch = 4 if cfg.QUICK_MODE else 25
    q, qr, qs = quantize_dense_int8(
        torch.randn(batch, seq_q, heads, 576, device=device)
    )
    k, kr, ks = quantize_dense_int8(torch.randn(32, 64, 1, 576, device=device))
    table = torch.randint(0, 32, (batch, 13), dtype=torch.int32, device=device)
    base_lengths = [0, 1, 385, 769]
    host_lengths = (base_lengths * ((batch + 3) // 4))[:batch]
    lengths = torch.tensor(host_lengths, dtype=torch.int32, device=device)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(batch, heads, seq_q, dtype=torch.float32, device=device)
    actual, actual_lse = flaggems_vllm.flash_mla_with_kvcache_w8a8(
        q, qr, k, kr, qs, ks, table, lengths, 512, causal=causal, out=out, lse=lse
    )
    query = torch.cat((q.float(), qr.float()), -1) * qs
    cache = torch.cat((k.float(), kr.float()), -1) * ks
    expected, expected_lse = torch_reference(
        query, cache, table, lengths, 512, causal=causal
    )
    assert actual is out and actual_lse is lse
    gems_assert_close(out, expected.to(out.dtype), out.dtype, atol=WINDOW_ATOL)
    assert_lse_close(lse, expected_lse)


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(not ASCEND_INT8, reason="Requires Ascend910B INT8 MLA backend")
def test_dense_decode_ascend_peaked_and_outlier_scale():
    device = flaggems_vllm.device
    q = torch.zeros((1, 1, 16, 512), dtype=torch.int8, device=device)
    qr = torch.zeros((1, 1, 16, 64), dtype=torch.bfloat16, device=device)
    q[:, :, :, 0] = 100
    qs = torch.ones((1, 1, 16, 1), dtype=torch.float32, device=device)
    table = torch.arange(7, dtype=torch.int32, device=device).view(1, 7)
    lengths = torch.tensor([384], dtype=torch.int32, device=device)

    k = torch.zeros((7, 64, 1, 512), dtype=torch.int8, device=device)
    kr = torch.zeros((7, 64, 1, 64), dtype=torch.bfloat16, device=device)
    ks = torch.ones((7, 64, 1, 1), dtype=torch.float32, device=device)
    k[0, 0, 0, 0] = 100
    k[:, :, 0, 1] = 100
    k[0, 0, 0, 1] = -100
    actual, lse = flaggems_vllm.flash_mla_with_kvcache_w8a8(
        q, qr, k, kr, qs, ks, table, lengths, 512, softmax_scale=0.00056, causal=True
    )
    expected, expected_lse = torch_reference(
        torch.cat((q.float(), qr.float()), -1) * qs,
        torch.cat((k.float(), kr.float()), -1) * ks,
        table,
        lengths,
        512,
        softmax_scale=0.00056,
        causal=True,
    )
    gems_assert_close(actual, expected.to(actual.dtype), actual.dtype, atol=0.5)
    assert_lse_close(lse, expected_lse)

    k.zero_()
    k[:, :, 0, 1] = 100
    k[0, 0, 0, 1] = -100
    ks.fill_(0.05)
    ks[0, 0, 0, 0] = 20.0
    actual_s, lse_s = flaggems_vllm.flash_mla_with_kvcache_w8a8(
        q, qr, k, kr, qs, ks, table, lengths, 512, causal=True
    )
    expected_s, expected_lse_s = torch_reference(
        torch.cat((q.float(), qr.float()), -1) * qs,
        torch.cat((k.float(), kr.float()), -1) * ks,
        table,
        lengths,
        512,
        causal=True,
    )
    gems_assert_close(actual_s, expected_s.to(actual_s.dtype), actual_s.dtype, atol=0.1)
    assert_lse_close(lse_s, expected_lse_s)


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(not ASCEND_INT8, reason="Requires Ascend910B INT8 MLA backend")
@pytest.mark.parametrize("splits", SPLIT_COUNTS)
def test_dense_decode_ascend_split_kv(splits):
    torch.manual_seed(45)
    device = flaggems_vllm.device
    q, qr, qs = quantize_dense_int8(torch.randn(3, 2, 16, 576, device=device))
    k, kr, ks = quantize_dense_int8(torch.randn(7, 64, 576, device=device))
    table = torch.tensor(
        [[6, 1, 3, 0, 1, 5, 2]], device=device, dtype=torch.int32
    ).repeat(3, 1)
    lengths = torch.tensor([0, 1, 385], device=device, dtype=torch.int32)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(3, 16, 2, device=device)
    result = flaggems_vllm.flash_mla_with_kvcache_w8a8(
        q,
        qr,
        k,
        kr,
        qs,
        ks,
        table,
        lengths,
        num_splits=splits,
        softmax_scale=0.03125,
        causal=True,
        out=out,
        lse=lse,
    )
    expected, logs = torch_reference(
        torch.cat((q.float(), qr.float()), -1) * qs,
        torch.cat((k.float(), kr.float()), -1) * ks,
        table,
        lengths,
        512,
        softmax_scale=0.03125,
        causal=True,
    )
    assert result[0] is out and result[1] is lse
    gems_assert_close(out, expected.to(out.dtype), out.dtype, atol=OUTPUT_ATOL)
    assert_lse_close(lse, logs)
