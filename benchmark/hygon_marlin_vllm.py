# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare the Hygon quantized operator with installed native vLLM BF16 MoE.

Run from the repository root with PYTHONPATH=src:. . Weight dequantization,
compilation and first-use INT4 layout conversion are outside timed regions.
"""

import json
from functools import partial
from statistics import mean

import flaggems_vllm
import torch
import triton
import vllm
from tests.marlin_moe_hygon_reference import make_case, reference
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


def errors(actual, expected):
    delta = actual.float() - expected.float()
    rms = (
        delta.square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1e-12)
    )
    peak = delta.abs().max() / expected.float().abs().max().clamp_min(1e-12)
    return dict(relative_rms=rms.item(), relative_peak=peak.item())


def main():
    assert fused_experts.__module__ == "vllm.model_executor.layers.fused_moe.fused_moe"
    print(
        "VLLM_BASELINE "
        + json.dumps(
            dict(
                version=vllm.__version__,
                function=fused_experts.__module__ + ".fused_experts",
            )
        ),
        flush=True,
    )
    results = []
    for q in (0, 1, 2, 6):
        for m in (1, 16):
            torch.manual_seed(752)
            shape = (m, 8, 4096, 14336, 2)
            args, w1, w2 = make_case(*shape, q=q, dtype=torch.bfloat16)
            # Native vLLM consumes BF16 weights with the same decoded values.
            baseline = partial(
                fused_experts,
                args["hidden_states"],
                w1,
                w2,
                args["topk_weights"],
                args["topk_ids"],
            )
            run = partial(flaggems_vllm.fused_marlin_moe, **args)
            expected = reference(args, w1, w2)
            actual, native = run(), baseline()
            op_error, vllm_error = errors(actual, expected), errors(native, expected)
            for check in (op_error, vllm_error):
                assert check["relative_rms"] < 0.01 and check["relative_peak"] < 0.02, (
                    q,
                    m,
                    check,
                )
            base_ms = triton.testing.do_bench(baseline, warmup=100, rep=200)
            op_ms = triton.testing.do_bench(run, warmup=100, rep=200)
            row = dict(
                q=q,
                shape=shape,
                vllm_bf16_ms=base_ms,
                operator_ms=op_ms,
                speedup=base_ms / op_ms,
                operator_error=op_error,
                vllm_error=vllm_error,
            )
            results.append(row)
            print("HYGON_VLLM " + json.dumps(row), flush=True)
            del args, w1, w2, actual, native, expected, baseline, run
            torch.cuda.empty_cache()
    print(
        "HYGON_VLLM_MEAN "
        + json.dumps(dict(arithmetic_mean=mean(row["speedup"] for row in results))),
        flush=True,
    )


if __name__ == "__main__":
    main()
