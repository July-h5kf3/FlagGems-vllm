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
from flaggems_vllm.utils.triton_version_utils import has_triton_tle
from tests.test_flash_mla_sparse_fwd_w8a8_fp8 import (
    assert_accuracy,
    dequantized_reference,
    make_inputs,
)
from tests.test_flash_mla_sparse_fwd_w8a8_fp8 import pytestmark as sparse_marks

pytestmark = sparse_marks + [
    pytest.mark.skipif(not has_triton_tle(3, 6, 0), reason="requires TLE sparse MLA")
]


@pytest.mark.parametrize(
    "batch,heads,topk",
    [(1, 64, 512), (1, 64, 2048), (4, 128, 1025), (16, 64, 4097), (1, 64, 8193)],
)
@pytest.mark.parametrize("magnitude", [0.1, 1.0])
def test_sparse_fp8_split_accuracy(batch, heads, topk, magnitude):
    inputs, _, _ = make_inputs(batch, heads, topk, seed=123, magnitude=magnitude)
    sink = torch.randn(heads, device="cuda")
    sink[0], sink[1] = float("inf"), -float("inf")
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, attn_sink=sink)
    reference, reference_lse = dequantized_reference(inputs, attn_sink=sink)
    assert_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("length", [0, 1, 63, 65, 257, 1025])
def test_sparse_fp8_split_empty_partitions_and_replay(length):
    inputs, _, _ = make_inputs(2, 64, 1025, seed=123)
    inputs[-1][1].fill_(-1)
    lengths = torch.full((2,), length, device="cuda", dtype=torch.int32)
    reference, reference_lse = dequantized_reference(inputs, topk_length=lengths)
    flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, topk_length=lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(
            *inputs, topk_length=lengths
        )
    graph.replay()
    assert_accuracy(output, lse, reference, reference_lse)
    expected = output.clone()
    graph.replay()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("page", [0, 4, 8, 15])
def test_sparse_fp8_split_repairs_any_partition(page):
    inputs, _, _ = make_inputs(4, 64, 1024, seed=123, magnitude=1.0)
    inputs[-1].copy_(torch.arange(1024, device="cuda", dtype=torch.int32)[None, None])
    inputs[5][page].fill_(2.0**40)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    reference, reference_lse = dequantized_reference(inputs)
    assert_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_split_repairs_variable_lengths():
    inputs, _, _ = make_inputs(4, 64, 1025, seed=123, magnitude=1.0)
    inputs[-1].copy_(torch.arange(1025, device="cuda", dtype=torch.int32)[None, None])
    inputs[5][4].fill_(2.0**40)
    lengths = torch.tensor([0, 1, 513, 999], device="cuda", dtype=torch.int32)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(
        *inputs, topk_length=lengths
    )
    reference, reference_lse = dequantized_reference(inputs, topk_length=lengths)
    assert_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("batch,heads,topk", [(2, 128, 1025), (8, 64, 4096)])
def test_sparse_fp8_unaligned_padded_strides(batch, heads, topk):
    inputs, _, _ = make_inputs(batch, heads, topk, seed=123)
    for index in (0, 1, 2, 3):
        tensor = inputs[index]
        storage = torch.empty(
            (*tensor.shape[:-1], tensor.shape[-1] + 1),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        inputs[index] = storage[..., 1:]
        inputs[index].copy_(tensor)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    reference, reference_lse = dequantized_reference(inputs)
    assert_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("batch,topk", [(1, 512), (8, 4096)])
def test_sparse_fp8_subnormal_probability_scale(batch, topk):
    inputs, _, _ = make_inputs(batch, 64, topk, seed=123)
    inputs[0].view(torch.uint8).zero_()
    inputs[1].zero_()
    inputs[2].fill_(128.0)
    inputs[3].zero_()
    inputs[4].fill_(1.0)
    inputs[5].fill_(2.0**-122)
    output, _ = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    # Normalize before comparison so the tiny output cannot pass via an absolute tolerance.
    torch.testing.assert_close(
        output.float() * (2.0**115),
        torch.ones_like(output, dtype=torch.float32),
        atol=0,
        rtol=1 / 128,
    )


@pytest.mark.parametrize("batch,topk", [(4, 1024), (8, 4096)])
def test_sparse_fp8_small_path_empty_cache(batch, topk):
    inputs, _, _ = make_inputs(batch, 64, topk, seed=123)
    for index in (2, 3, 5):
        inputs[index] = inputs[index][:0]
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    assert output.count_nonzero().item() == 0
    assert lse.isposinf().all().item()


@pytest.mark.parametrize("head", [16, 63, 127])
def test_sparse_fp8_exact_qk_late_heads(head):
    inputs, _, _ = make_inputs(8, 128, 1024, seed=123, magnitude=1.0)
    inputs[4][:, :, head].mul_(2.0**20)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    reference, reference_lse = dequantized_reference(inputs)
    assert_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("batch,topk", [(1, 512), (8, 4096)])
def test_sparse_fp8_graph_switches_to_precise_qk(batch, topk):
    inputs, _, _ = make_inputs(batch, 64, topk, seed=123, magnitude=1.0)
    inputs[-1].copy_(torch.arange(topk, device="cuda", dtype=torch.int32)[None, None])
    flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    inputs[5][1].fill_(2.0**40)
    graph.replay()
    reference, reference_lse = dequantized_reference(inputs)
    assert_accuracy(output, lse, reference, reference_lse)
