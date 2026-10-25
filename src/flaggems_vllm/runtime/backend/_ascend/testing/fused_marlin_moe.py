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

from importlib import import_module

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT4B8

GROUP_SIZE = 128
REJECTION_MATCHES = {
    "quant_type_id": "Only symmetric uint4b8 with group_size=128",
    "act_order": "extra quantization metadata",
    "FP8": "extra quantization metadata",
}


def ascend_weights(e, k, n, dtype):
    import torch_npu

    torch.manual_seed(7)
    result = []
    for ni, ki in [(2 * n, k), (k, n)]:
        w = torch.randint(0, 256, (e, ni, ki // 2), device="npu", dtype=torch.uint8)
        s = torch.rand((e, ni, ki // 128), device="npu", dtype=dtype) * 0.03
        native = []
        for ei in range(e):
            q = w[ei].to(torch.int32)
            q = torch.stack((q & 15, q >> 4), dim=-1).reshape(ni, ki) - 8
            native.append(torch_npu.npu_convert_weight_to_int4pack(q.T.contiguous()))
        wp = torch.stack(native)
        sn = s.transpose(1, 2).contiguous()
        result.append((w, s, wp, sn, torch.zeros_like(sn)))
    return result


def ascend_baseline(x, ww, p, ids, *, vllm_dispatch=False):
    import torch_npu

    e = ww[0][0].shape[0]
    (a, idx, counts, _) = torch_npu.npu_moe_init_routing_v2(
        x,
        ids.to(torch.int32),
        expert_num=e,
        active_num=x.shape[0] * ids.shape[1],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        active_expert_range=[0, e],
        quant_mode=-1,
    )
    if vllm_dispatch:
        counts = counts.to(torch.int64)
    for j in range(2):
        (_, _, w, s, z) = ww[j]
        a = torch_npu.npu_grouped_matmul(
            x=[a],
            weight=[w],
            antiquant_scale=[s],
            antiquant_offset=[z],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=counts,
            output_dtype=x.dtype,
        )[0]
        if j == 0:
            a = torch_npu.npu_swiglu(a)
    if vllm_dispatch:
        idx = torch.abs(idx)
        p = p.to(a.dtype)
    return torch_npu.npu_moe_token_unpermute(a, idx, probs=p)


def ascend_vllm_baseline(x, ww, p, ids):
    """Single-device vLLM-Ascend W4A16 AllGather/GMM path, EP=1.

    Mirrors vllm-project/vllm-ascend c5055c8086d56ea1b2714b0f555ba67edd05e945:
    quantization/methods/wna16/w4a16.py and ops/fused_moe/token_dispatcher.py.
    Weight repacking is outside timing, as in process_weights_after_loading.
    """
    return ascend_baseline(x, ww, p, ids, vllm_dispatch=True)


def ascend_inputs(m, e, k, t, seed=7):
    torch.manual_seed(seed + m)
    x = torch.randn((m, k), device="npu", dtype=torch.bfloat16) * 0.1
    ids = torch.rand((m, e), device="npu").topk(t, -1).indices.to(torch.int32)
    p = torch.softmax(torch.randn((m, t), device="npu"), -1)
    return (x, p, ids)


def make_test_inputs(precision, router_weight_on_input, factory, *args):
    if precision != "int4" or args[-2] != torch.bfloat16 or router_weight_on_input:
        pytest.skip(
            "Ascend Marlin MoE supports BF16 INT4 with output-side router weights"
        )
    (hs, w1, w2, r1, r2, weights, ids, s1, s2) = factory(*args)
    return (hs, w1, w2, r1, r2, weights.float(), ids, s1, s2)


def supports_benchmark(precision):
    return precision == "int4"


def has_baseline(precision):
    if not supports_benchmark(precision):
        return False
    try:
        import_module("torch_npu")
    except ImportError:
        return False
    return True


def benchmark_inputs(precision, benchmark, dtype):
    from tests.accuracy_utils import gems_assert_close

    geometry = None
    weights = None
    for tokens, experts, hidden, intermediate, topk in benchmark.shapes:
        if geometry != (experts, hidden, intermediate):
            weights = ascend_weights(experts, hidden, intermediate, dtype)
            geometry = (experts, hidden, intermediate)
        (x, probabilities, ids) = ascend_inputs(tokens, experts, hidden, topk)
        inputs = (
            x,
            weights[0][0],
            weights[1][0],
            weights[0][1],
            weights[1][1],
            weights[0][2],
            weights[1][2],
            weights[0][3:],
            weights[1][3:],
            probabilities,
            ids,
        )
        actual = candidate(precision, *inputs)
        gems_assert_close(
            actual,
            ascend_baseline(x, weights, probabilities, ids),
            torch.bfloat16,
            atol=0.02,
        )
        gems_assert_close(
            actual, baseline(precision, *inputs), torch.bfloat16, atol=0.02
        )
        yield inputs


def baseline(precision, *inputs):
    (x, w1, w2, s1, s2, nw1, nw2, ns1, ns2, probabilities, ids) = inputs
    native_weights = [(w1, s1, nw1, *ns1), (w2, s2, nw2, *ns2)]
    return ascend_vllm_baseline(x, native_weights, probabilities, ids)


def candidate(precision, *inputs):
    (x, w1, w2, s1, s2, _nw1, _nw2, _ns1, _ns2, probabilities, ids) = inputs
    return flaggems_vllm.fused_marlin_moe(
        hidden_states=x,
        w1=w1,
        w2=w2,
        bias1=None,
        bias2=None,
        w1_scale=s1,
        w2_scale=s2,
        topk_weights=probabilities,
        topk_ids=ids,
        quant_type_id=QUANT_TYPE_UINT4B8,
    )
