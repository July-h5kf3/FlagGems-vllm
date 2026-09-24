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


@pytest.mark.parametrize("start,stop", [(0, 16), (8, 12), (9, 10)])
def test_sparse_fp8_large_scale_winner(start, stop):
    inputs, _, _ = make_inputs(16, 64, 192, seed=123, magnitude=1.0)
    inputs[-1].copy_(torch.arange(192, device="cuda", dtype=torch.int32)[None, None])
    inputs[5][1].fill_(2.0**40)
    lengths = torch.arange(16, device="cuda", dtype=torch.int32) * 192 // 15
    sink = torch.randn(64, device="cuda")
    inputs = [
        tensor[start:stop] if index in (0, 1, 4, 6) else tensor
        for index, tensor in enumerate(inputs)
    ]
    kwargs = dict(topk_length=lengths[start:stop], attn_sink=sink)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, **kwargs)
    reference, reference_lse = dequantized_reference(inputs, **kwargs)
    assert_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_small_block_scale_keeps_accumulator_finite():
    inputs, _, _ = make_inputs(16, 64, 192, seed=123, magnitude=1.0)
    inputs[-1].copy_(torch.arange(192, device="cuda", dtype=torch.int32)[None, None])
    inputs[5].fill_(1.0)
    sink = torch.randn(64, device="cuda")
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, attn_sink=sink)
    reference, reference_lse = dequantized_reference(inputs, attn_sink=sink)
    assert_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_preserves_small_contribution_after_cancellation():
    inputs, _, _ = make_inputs(16, 64, 128, seed=123, magnitude=1.0)
    inputs[0].view(torch.uint8).zero_()
    inputs[1].zero_()
    inputs[1][..., 0] = 1.0
    inputs[2].fill_(1.0)
    inputs[2][0, 32:].fill_(-1.0)
    inputs[3].zero_()
    inputs[3][1, :, 0] = -864.0
    inputs[4].fill_(1.0)
    inputs[5].fill_(1.0)
    inputs[-1].copy_(torch.arange(128, device="cuda", dtype=torch.int32)[None, None])
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    reference, reference_lse = dequantized_reference(inputs)
    assert_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("magnitude", [0.1, 1.0, 3.0])
def test_sparse_fp8_single_request_mixed_scales(magnitude):
    inputs, _, _ = make_inputs(1, 64, 192, seed=123, magnitude=magnitude)
    inputs[-1].copy_(torch.arange(192, device="cuda", dtype=torch.int32)[None, None])
    inputs[5][0, ::2].fill_(2.0**-40)
    inputs[5][0, 1::2].fill_(2.0**40)
    length = torch.full((1,), 192, device="cuda", dtype=torch.int32)
    sink = torch.randn(64, device="cuda")
    kwargs = dict(topk_length=length, attn_sink=sink)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, **kwargs)
    reference, reference_lse = dequantized_reference(inputs, **kwargs)
    assert_accuracy(output, lse, reference, reference_lse)
