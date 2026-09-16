# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Hygon benchmark against pre-dequantized PyTorch grouped MoE.

This is not a vLLM Marlin comparison. Routing indices are computed before
measurement, and weight decoding is excluded from the PyTorch baseline.
"""

import json

import flaggems_vllm
import pytest
import torch
import torch.nn.functional as F
import triton
from tests.marlin_moe_hygon_reference import make_case, reference

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "hygon", reason="Hygon backend benchmark"
)


def prepare_baseline(args, w1, w2):
    a, ids, weights = (args[k] for k in ("hidden_states", "topk_ids", "topk_weights"))
    m, k = a.shape
    topk = ids.shape[1]
    routes = [torch.where(ids.flatten() == e)[0] for e in range(w1.shape[0])]
    active = [(e, r, r // topk) for e, r in enumerate(routes) if r.numel()]

    def run():
        result = torch.empty((m * topk, k), dtype=a.dtype, device=a.device)
        for e, r, tokens in active:
            gu = a[tokens] @ w1[e].T
            g, u = gu.float().chunk(2, -1)
            act = (F.silu(g) * u).to(a.dtype)
            out = act @ w2[e].T
            result[r] = (out.float() * weights.flatten()[r, None]).to(a.dtype)
        return result.view(m, topk, k).float().sum(1).to(a.dtype)

    return run


@pytest.mark.parametrize("q", [0, 1, 2, 6])
@pytest.mark.parametrize(
    "shape", [(1, 8, 1024, 2048, 2), (16, 8, 1024, 2048, 2), (128, 8, 1024, 2048, 2)]
)
def test_hygon_marlin_benchmark(q, shape):
    torch.manual_seed(752)
    args, w1, w2 = make_case(*shape, q=q, dtype=torch.bfloat16)
    baseline = prepare_baseline(args, w1, w2)
    run = lambda: flaggems_vllm.fused_marlin_moe(**args)
    actual = run()
    expected = reference(args, w1, w2)
    # Validate against the FP32 oracle: the timing baseline rounds GEMM2
    # before weighting and therefore has different BF16 rounding.
    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.04)
    base_ms = triton.testing.do_bench(baseline, warmup=100, rep=200)
    op_ms = triton.testing.do_bench(run, warmup=100, rep=200)
    print(
        "HYGON_MARLIN "
        + json.dumps(
            dict(
                q=q,
                shape=shape,
                dtype="bfloat16",
                baseline="predequantized_torch",
                baseline_ms=base_ms,
                operator_ms=op_ms,
                speedup=base_ms / op_ms,
            )
        )
    )
