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

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.flash_mla_ckv_fp8_per_token import (
    quantize_k_ckv_per_token,
    quantize_q_ckv_per_token,
)

from . import conftest as cfg

pytestmark = [
    pytest.mark.flash_mla_sparse_fwd_w8a8_fp8,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
        reason="requires Hopper",
    ),
]


def make_inputs(batch, heads, topk, seed=42, magnitude=0.1):
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


def dequantized_reference(inputs, attn_sink=None, topk_length=None, softmax_scale=None):
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


def assert_accuracy(output, lse, expected, expected_lse):
    relative_l2 = (
        output.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-12)
    assert relative_l2.item() < 0.05, relative_l2.item()
    torch.testing.assert_close(lse, expected_lse, atol=0.025, rtol=0.002)


CASES = (
    [(2, 64, 129), (4, 128, 512)]
    if cfg.QUICK_MODE
    else [
        (1, 64, 0),
        (2, 64, 1),
        (3, 128, 65),
        (4, 64, 129),
        (2, 128, 2048),
        (64, 128, 256),
        (128, 128, 128),
    ]
)


@pytest.mark.parametrize("batch,heads,topk", CASES)
@pytest.mark.parametrize("magnitude", [0.1, 1.0])
def test_sparse_fp8_accuracy(batch, heads, topk, magnitude):
    inputs, _, _ = make_inputs(batch, heads, topk, magnitude=magnitude)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, attn_sink=sink)
    reference, reference_lse = dequantized_reference(inputs, attn_sink=sink)
    assert_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_masks_lengths_and_sink():
    inputs, _, _ = make_inputs(4, 64, 1025)
    inputs[-1][0].fill_(-1)
    inputs[-1][2, :, ::2] = inputs[2].shape[0] * 64 + 10
    inputs[-1][3, :, :64] = -1
    lengths = torch.tensor([1025, 0, 129, 1025], device="cuda", dtype=torch.int32)
    sink = torch.zeros(64, device="cuda", dtype=torch.float32)
    sink[0], sink[1] = float("inf"), -float("inf")
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(
        *inputs, attn_sink=sink, topk_length=lengths
    )
    reference, reference_lse = dequantized_reference(
        inputs, attn_sink=sink, topk_length=lengths
    )
    assert_accuracy(output, lse, reference, reference_lse)
    assert output[:2].count_nonzero().item() == 0
    assert output[:, :, 0].count_nonzero().item() == 0


@pytest.mark.parametrize("batch", [4, 16])
def test_sparse_fp8_strides_and_graph_replay(batch):
    inputs, _, _ = make_inputs(batch, 128, 513)
    # Noncontiguous outer strides must not change physical token addressing.
    for index in range(6):
        tensor = inputs[index]
        storage = torch.empty(
            (tensor.shape[0] * 2,) + tensor.shape[1:], device="cuda", dtype=tensor.dtype
        )
        storage[::2].copy_(tensor)
        inputs[index] = storage[::2]
    reference, reference_lse = dequantized_reference(inputs)
    flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    graph.replay()
    expected = output.clone()
    graph.replay()
    assert torch.equal(output, expected)
    assert_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_empty_batch_and_cache():
    inputs, _, _ = make_inputs(0, 64, 128)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    assert output.shape == (0, 1, 64, 512) and lse.shape == (0, 64, 1)
    inputs, _, _ = make_inputs(1, 64, 128)
    for index in (2, 3, 5):
        inputs[index] = inputs[index][:0]
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    assert output.count_nonzero().item() == 0 and lse.isposinf().all().item()


def test_sparse_fp8_rejects_wrong_rope_dtype():
    inputs, _, _ = make_inputs(1, 64, 128)
    inputs[1] = inputs[1].to(torch.float8_e4m3fn)
    with pytest.raises(TypeError, match="RoPE"):
        flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)


def test_sparse_fp8_scaled_rope_and_custom_softmax():
    inputs, _, _ = make_inputs(2, 64, 129)
    inputs[1].mul_(3)
    inputs[3].mul_(2)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(
        *inputs, softmax_scale=0.5
    )
    reference, reference_lse = dequantized_reference(inputs, softmax_scale=0.5)
    assert_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("topk", [1, 65, 129, 513])
def test_sparse_fp8_staged_masks_and_lengths(topk):
    inputs, _, _ = make_inputs(16, 64, topk)
    inputs[-1][0].fill_(-1)
    inputs[-1][3, :, ::2] = inputs[2].shape[0] * 64
    lengths = torch.arange(16, device="cuda", dtype=torch.int32) * topk // 15
    sink = torch.randn(64, device="cuda", dtype=torch.float32)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(
        *inputs, attn_sink=sink, topk_length=lengths
    )
    reference, reference_lse = dequantized_reference(
        inputs, attn_sink=sink, topk_length=lengths
    )
    assert_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_cuda_bf16_reference():
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    inputs, query, cache = make_inputs(4, 128, 512)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    reference, _, reference_lse = flash_mla_sparse_fwd(
        query[:, 0],
        cache.reshape(-1, 1, 576),
        inputs[-1],
        576**-0.5,
        512,
    )
    assert_accuracy(output, lse, reference[:, None], reference_lse[:, :, None])
