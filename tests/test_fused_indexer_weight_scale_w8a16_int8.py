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

from .accuracy_utils import gems_assert_close
from .conftest import QUICK_MODE

API_NAME = "fused_indexer_weight_scale_w8a16_int8"
pytestmark = pytest.mark.skipif(
    API_NAME not in vars(flaggems_vllm),
    reason="backend has no INT8 indexer weight scaling",
)
SHAPES = (
    [(0, 16), (1, 16), (7, 13), (129, 64)]
    if QUICK_MODE
    else [
        (0, 16),
        (7, 0),
        (1, 1),
        (1, 16),
        (7, 13),
        (33, 64),
        (1003, 16),
        (64, 129),
        (16384, 16),
        (4096, 64),
    ]
)


@pytest.fixture(scope="module")
def native_weight_scale():
    # Ascend's native simple_compile_backend is eager, including for this leaf.
    with torch.compiler.set_stance("force_eager"):
        from vllm.models.glm5next.common.attention import _fused_indexer_weight_scale

        yield _fused_indexer_weight_scale


def make_inputs(tokens, heads, device, is_strided=False):
    width = heads * 2 if is_strided else heads
    rows = tokens * 2 if is_strided else tokens
    weights = torch.randint(
        -128, 128, (tokens, width), dtype=torch.int32, device=device
    ).to(torch.int8)
    weight_scale = torch.rand((rows, 1), device=device) * 0.02 + 0.001
    q_scale = torch.exp2(
        torch.randint(-8, 1, (tokens, width, 1), device=device).float()
    )
    if is_strided:
        weights = weights[:, ::2]
        weight_scale = weight_scale[::2]
        q_scale = q_scale[:, ::2]
    weights_bf16 = (weights.float() * weight_scale).to(torch.bfloat16)
    return weights, weight_scale, q_scale, weights_bf16


@pytest.mark.fused_indexer_weight_scale
@pytest.mark.parametrize("tokens,heads", SHAPES)
@pytest.mark.parametrize("is_strided", [False, True])
@pytest.mark.parametrize("scale", [0.0, 0.02209708691207961, -0.125])
def test_fused_indexer_weight_scale_int8(
    tokens, heads, is_strided, scale, native_weight_scale
):
    torch.manual_seed(0)
    weights, weight_scale, q_scale, weights_bf16 = make_inputs(
        tokens, heads, flaggems_vllm.device, is_strided
    )
    expected = native_weight_scale(weights_bf16, q_scale, scale)
    output = flaggems_vllm.fused_indexer_weight_scale_w8a16_int8(
        weights, weight_scale, q_scale, scale
    )
    gems_assert_close(output, expected, torch.float32, atol=1e-6)


@pytest.mark.fused_indexer_weight_scale
def test_fused_indexer_weight_scale_nonfinite(native_weight_scale):
    torch.manual_seed(0)
    weights, weight_scale, q_scale, _ = make_inputs(2, 16, flaggems_vllm.device)
    weights[0, :3] = torch.tensor(
        [-128, 0, 127], dtype=torch.int8, device=weights.device
    )
    q_scale[0, :3, 0] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], device=weights.device
    )
    weights_bf16 = (weights.float() * weight_scale).to(torch.bfloat16)
    expected = native_weight_scale(weights_bf16, q_scale, 0.125)
    output = flaggems_vllm.fused_indexer_weight_scale_w8a16_int8(
        weights, weight_scale, q_scale, 0.125
    )
    gems_assert_close(output, expected, torch.float32, atol=1e-6, equal_nan=True)
