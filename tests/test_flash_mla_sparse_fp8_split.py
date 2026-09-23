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
    "batch,heads,topk", [(1, 64, 512), (4, 128, 1025), (16, 64, 4097), (1, 64, 8193)]
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
