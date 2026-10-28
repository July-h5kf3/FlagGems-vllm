# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close, gems_assert_equal
from tests.test_flash_mla_sparse_fwd import (
    Flashmla_Sparse_Test_Param,
    FlashmlaSparseTestKit,
)

OP_NAME = "flash_mla_with_kvcache_sparse_w8a8_int8"
OVERRIDES = dict(flaggems_vllm.runtime.backend.get_customized_ops())
SUPPORTED = OP_NAME in OVERRIDES
if SUPPORTED:
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_mla_with_kvcache_sparse_w8a8_int8 import (
        prepare_flash_mla_with_kvcache_sparse_w8a8_int8,
    )
else:
    prepare_flash_mla_with_kvcache_sparse_w8a8_int8 = None
pytestmark = [
    pytest.mark.flash_mla_with_kvcache_sparse_w8a8_int8,
    pytest.mark.skipif(not SUPPORTED, reason="No registered sparse INT8 KVCache MLA"),
]


def quantize(tensor):
    scale = (
        tensor[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
    )
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    content = (tensor[..., :512].float() / scale).round().to(torch.int8)
    rope = (tensor[..., 512:].float() / scale).to(torch.bfloat16)
    native = (torch.cat((content.float(), rope.float()), -1) * scale).to(torch.bfloat16)
    return content, rope, scale, native


def make_inputs(
    batch, queries, heads, topk, pages=32, dtype=torch.bfloat16, rope_dim=64
):
    param = Flashmla_Sparse_Test_Param(
        batch * queries, pages * 64, topk, heads, d_qk=576
    )
    query, cache, indices, _, _ = FlashmlaSparseTestKit.make_input_flashmla(
        param, seed=42
    )
    q, qr, qs, native_q = quantize(query.view(batch, queries, heads, 576))
    kv, kr, ks, native_kv = quantize(cache.view(pages, 64, 1, 576))
    if rope_dim == 0:
        qr, kr = qr[..., :0].contiguous(), kr[..., :0].contiguous()
        native_q, native_kv = (
            native_q[..., :512].contiguous(),
            native_kv[..., :512].contiguous(),
        )
    inputs = [
        q,
        qr.to(dtype),
        kv,
        kr.to(dtype),
        qs,
        ks,
        indices.view(batch, queries, topk),
    ]
    return inputs, native_q, native_kv


def reference(
    inputs,
    *,
    attn_sink=None,
    topk_length=None,
    softmax_scale=None,
    reference_device="cpu",
    chunk_rows=16,
):
    q, qr, kv, kr, qs, ks, indices = (tensor.to(reference_device) for tensor in inputs)
    attn_sink = None if attn_sink is None else attn_sink.to(reference_device)
    topk_length = (
        None if topk_length is None else topk_length.to(reference_device).flatten()
    )
    batch, queries, heads, _ = q.shape
    dims = 512 + qr.shape[-1]
    query = (torch.cat((q.float(), qr.float()), -1) * qs).view(-1, heads, dims)
    cache = (torch.cat((kv.float(), kr.float()), -1) * ks).view(-1, 1, dims)
    selected = indices.view(batch * queries, 1, -1)
    output = torch.empty(
        (batch * queries, heads, 512), dtype=torch.float32, device=reference_device
    )
    lse = torch.empty(
        (batch * queries, heads), dtype=torch.float32, device=reference_device
    )
    for begin in range(0, batch * queries, chunk_rows):
        end = min(begin + chunk_rows, batch * queries)
        output[begin:end], _, lse[begin:end] = (
            FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
                end - begin,
                cache.shape[0],
                heads,
                1,
                dims,
                indices.shape[-1],
                query[begin:end],
                cache,
                selected[begin:end],
                dims**-0.5 if softmax_scale is None else softmax_scale,
                512,
                attn_sink,
                None if topk_length is None else topk_length[begin:end],
                output_dtype=torch.float32,
            )
        )
    return output.view(batch, queries, heads, 512), lse.view(
        batch, queries, heads
    ).transpose(1, 2)


def assert_accuracy(actual, expected):
    ref_output, ref_lse = expected
    output, lse = (tensor.to(ref_output.device) for tensor in actual)
    gems_assert_close(output, ref_output, output.dtype, atol=2e-3)
    gems_assert_equal(torch.isposinf(lse), torch.isposinf(ref_lse))
    finite = torch.isfinite(ref_lse)
    gems_assert_close(lse[finite], ref_lse[finite], torch.float32, atol=2e-4)


@pytest.mark.parametrize("topk", [129, 4097])
@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("queries", [1, 2])
@pytest.mark.parametrize("heads", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sparse_decode(queries, heads, dtype, rope_dim, topk):
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs, _, _ = make_inputs(4, queries, heads, topk, dtype=dtype, rope_dim=rope_dim)
    inputs[4].mul_(1.3)
    inputs[4][2].mul_(256)
    inputs[5].mul_(0.7)
    indices = inputs[-1]
    indices[0].fill_(-1)
    indices[1, :, ::3] = inputs[2].shape[0] * 64
    indices[2, :, 1] = indices[2, :, 0]
    lengths = torch.full((4, queries), 97, dtype=torch.int32, device=inputs[0].device)
    lengths[-1] = 0
    sink = torch.randn(heads, dtype=torch.float32, device=inputs[0].device)
    sink[0], sink[1] = float("inf"), -float("inf")
    out = torch.empty((4, queries, heads, 512), dtype=dtype, device=inputs[0].device)
    actual = flaggems_vllm.flash_mla_with_kvcache_sparse_w8a8_int8(
        *inputs,
        topk_length=lengths,
        attn_sink=sink,
        out=out,
        softmax_scale=0.125,
    )
    assert actual[0] is out
    assert_accuracy(
        actual,
        reference(inputs, topk_length=lengths, attn_sink=sink, softmax_scale=0.125),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads", [16, 64, 128])
@pytest.mark.parametrize("queries", [1, 2])
def test_sparse_decode_split_merge(dtype, heads, queries):
    inputs, _, _ = make_inputs(4, queries, heads, 640, pages=16, dtype=dtype)
    inputs[4].mul_(1.3)
    inputs[5].mul_(0.7)
    inputs[6][1, :, ::3] = -1
    lengths = (
        torch.tensor([640, 511, 97, 0], dtype=torch.int32, device=inputs[0].device)[
            :, None
        ]
        .expand(4, queries)
        .contiguous()
    )
    sink = torch.randn(heads, dtype=torch.float32, device=inputs[0].device)
    sink[0], sink[1] = float("inf"), -float("inf")
    actual = flaggems_vllm.flash_mla_with_kvcache_sparse_w8a8_int8(
        *inputs, topk_length=lengths, attn_sink=sink
    )
    assert_accuracy(actual, reference(inputs, topk_length=lengths, attn_sink=sink))


@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("topk", [0, 1, 64, 65, 512, 2048, 8320])
def test_sparse_decode_lengths(topk, rope_dim):
    inputs, _, _ = make_inputs(2, 1, 64, topk, rope_dim=rope_dim)
    if topk > 0:
        inputs[-1][0].fill_(-1)
        inputs[-1][1, :, ::4] = 2147483647
    if topk == 8320:
        inputs[4].mul_(256)
    handle, _ = prepare_flash_mla_with_kvcache_sparse_w8a8_int8(*inputs)
    actual = handle()
    if topk == 0:
        gems_assert_equal(actual[0], torch.zeros_like(actual[0]))
        gems_assert_equal(actual[1], torch.full_like(actual[1], float("inf")))
    else:
        assert_accuracy(actual, reference(inputs))


@pytest.mark.parametrize("empty", ["batch", "heads", "cache"])
def test_sparse_decode_empty(empty):
    inputs, _, _ = make_inputs(2, 1, 64, 65)
    if empty == "batch":
        for index in (0, 1, 4, 6):
            inputs[index] = inputs[index][:0]
    elif empty == "heads":
        for index in (0, 1, 4):
            inputs[index] = inputs[index][:, :, :0]
    else:
        for index in (2, 3, 5):
            inputs[index] = inputs[index][:0]
    output, lse = flaggems_vllm.flash_mla_with_kvcache_sparse_w8a8_int8(*inputs)
    gems_assert_equal(output, torch.zeros_like(output))
    gems_assert_equal(lse, torch.full_like(lse, float("inf")))


@pytest.mark.parametrize("queries", [1, 2])
def test_sparse_decode_batch_boundary(queries):
    batch = 512 // queries + 1
    inputs, _, _ = make_inputs(batch, queries, 16, 4097, pages=16, rope_dim=0)
    lengths = (
        torch.arange(batch * queries, dtype=torch.int32, device=inputs[0].device).view(
            batch, queries
        )
        % 129
    )
    sink = torch.randn(16, dtype=torch.float32, device=inputs[0].device)
    sink[0], sink[1] = float("inf"), -float("inf")
    actual = flaggems_vllm.flash_mla_with_kvcache_sparse_w8a8_int8(
        *inputs, topk_length=lengths, attn_sink=sink
    )
    assert_accuracy(actual, reference(inputs, topk_length=lengths, attn_sink=sink))
