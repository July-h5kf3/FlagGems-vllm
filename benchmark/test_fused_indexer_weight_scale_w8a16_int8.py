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
from tests import test_fused_indexer_weight_scale_w8a16_int8 as indexer_tests
from tests.accuracy_utils import gems_assert_close

from . import base

native_weight_scale = indexer_tests.native_weight_scale


def input_fn(shape, dtype, device):
    if dtype != torch.bfloat16:
        raise ValueError("the native baseline requires BF16 weights")
    torch.manual_seed(0)
    tokens, heads = shape
    weights, weight_scale, q_scale, weights_bf16 = indexer_tests.make_inputs(
        tokens, heads, device
    )
    yield weights, weight_scale, q_scale, (128 * heads) ** -0.5, weights_bf16


class IndexerWeightScaleBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [(1, 16), (128, 16), (4096, 16)]
    DEFAULT_SHAPE_DESC = "tokens, heads"

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for inputs in super().get_input_iter(dtype):
            weights, weight_scale, q_scale, scale, weights_bf16 = inputs
            expected = self.torch_op(weights_bf16, q_scale, scale)
            output = self.gems_op(weights, weight_scale, q_scale, scale)
            gems_assert_close(output, expected, torch.float32, atol=1e-6)
            yield inputs

    def get_latency(self, op, weights, weight_scale, q_scale, scale, weights_bf16):
        if op is self.torch_op:
            return super().get_latency(op, weights_bf16, q_scale, scale)
        else:
            return super().get_latency(op, weights, weight_scale, q_scale, scale)


@pytest.mark.fused_indexer_weight_scale
@pytest.mark.skipif(
    indexer_tests.API_NAME not in vars(flaggems_vllm),
    reason="backend has no INT8 indexer weight scaling",
)
def test_fused_indexer_weight_scale(native_weight_scale):
    bench = IndexerWeightScaleBenchmark(
        op_name="fused_indexer_weight_scale",
        torch_op=native_weight_scale,
        input_fn=input_fn,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flaggems_vllm.fused_indexer_weight_scale_w8a16_int8)
    bench.run()
