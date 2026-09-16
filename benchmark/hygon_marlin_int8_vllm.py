# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native vLLM W8A16 INT8 comparison on the shared channel-scale subset.

Our group-128 scales repeat each channel's scale. Native vLLM applies the
channel scale after the dot product; our path rounds dequantized weights to
activation dtype before the dot product. Both are checked independently.
"""

import json
from functools import partial

import torch
import triton
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

import flaggems_vllm
from benchmark.hygon_marlin_vllm import errors
from tests.marlin_moe_hygon_reference import make_case, reference


def main():
    for m in (1, 16):
        torch.manual_seed(752)
        shape = (m, 8, 4096, 14336, 2)
        args, w1, w2 = make_case(*shape, q=1, dtype=torch.bfloat16)
        del w1, w2
        native_weights, native_scales, decoded = [], [], []
        for index in (1, 2):
            packed = args[f"w{index}"]
            scale = args[f"w{index}_scale"]
            channel = scale[..., 0].contiguous()
            scale.copy_(channel[..., None].expand_as(scale))
            weight = (packed.to(torch.int16) - 128).to(torch.int8)
            native_weights.append(weight)
            native_scales.append(channel)
            decoded.append(
                (weight.float() * channel.float()[..., None]).to(torch.bfloat16)
            )
        baseline = partial(
            fused_experts,
            args["hidden_states"],
            native_weights[0],
            native_weights[1],
            args["topk_weights"],
            args["topk_ids"],
            use_int8_w8a16=True,
            w1_scale=native_scales[0],
            w2_scale=native_scales[1],
        )
        run = partial(flaggems_vllm.fused_marlin_moe, **args)
        expected = reference(args, *decoded)
        actual, native = run(), baseline()
        op_error, native_error = errors(actual, expected), errors(native, expected)
        for check in (op_error, native_error):
            assert check["relative_rms"] < 0.01 and check["relative_peak"] < 0.02, check
        native_ms = triton.testing.do_bench(baseline, warmup=100, rep=200)
        operator_ms = triton.testing.do_bench(run, warmup=100, rep=200)
        print(
            "HYGON_VLLM_INT8 "
            + json.dumps(
                dict(
                    shape=shape,
                    vllm_int8_ms=native_ms,
                    operator_ms=operator_ms,
                    speedup=native_ms / operator_ms,
                    operator_error=op_error,
                    vllm_error=native_error,
                )
            ),
            flush=True,
        )
        del (
            args,
            native_weights,
            native_scales,
            decoded,
            baseline,
            run,
            expected,
            actual,
            native,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
