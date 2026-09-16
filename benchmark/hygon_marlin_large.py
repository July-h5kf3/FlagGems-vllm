# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Additional upstream-shaped validation; run from the repository root."""
import importlib
import json
import sys
from functools import partial

import flaggems_vllm
import torch
import triton
from benchmark.test_fused_marlin_moe_hygon import prepare_baseline
from tests.marlin_moe_hygon_reference import make_case, reference


def main():
    for q in map(int, sys.argv[1].split(",")) if len(sys.argv) > 1 else (0, 1, 2, 6):
        for m in map(int, sys.argv[2].split(",")) if len(sys.argv) > 2 else (1, 16):
            torch.manual_seed(752)
            shape = (m, 8, 4096, 14336, 2)
            args, w1, w2 = make_case(*shape, q=q, dtype=torch.bfloat16)
            run = partial(flaggems_vllm.fused_marlin_moe, **args)
            actual = run()
            expected = reference(args, w1, w2)
            delta = actual.float() - expected.float()
            rms = (
                delta.square().mean().sqrt()
                / expected.float().square().mean().sqrt().clamp_min(1e-12)
            ).item()
            peak = (
                delta.abs().max() / expected.float().abs().max().clamp_min(1e-12)
            ).item()
            assert rms < 0.01 and peak < 0.02, (q, m, rms, peak)
            baseline = prepare_baseline(args, w1, w2)
            base_ms = triton.testing.do_bench(baseline, warmup=100, rep=200)
            op_ms = triton.testing.do_bench(run, warmup=100, rep=200)
            module = importlib.import_module(
                "flaggems_vllm.runtime.backend._hygon.fused.fused_marlin_moe"
            )
            tuned = []
            for name in ("_gemm", "_gemv"):
                kernel = getattr(module, name)
                for entry in kernel.kernel_cache[0].values():
                    config = dict(kernel=name, **entry[1])
                    if config not in tuned:
                        tuned.append(config)
            print("HYGON_CONFIG " + json.dumps(tuned, default=str), flush=True)
            print(
                "HYGON_LARGE "
                + json.dumps(
                    dict(
                        q=q,
                        shape=shape,
                        relative_rms=rms,
                        relative_peak=peak,
                        baseline_ms=base_ms,
                        operator_ms=op_ms,
                        speedup=base_ms / op_ms,
                    )
                ),
                flush=True,
            )
            if q == 0 and len(sys.argv) > 3 and sys.argv[3] == "cold":

                def repack_run(task_args=args, backend=module):
                    backend._TRANSPOSE_CACHE.clear()
                    return flaggems_vllm.fused_marlin_moe(**task_args)

                repack_ms = triton.testing.do_bench(repack_run, warmup=100, rep=200)
                print(
                    "HYGON_REPACK "
                    + json.dumps(dict(shape=shape, repack_ms=repack_ms, warm_ms=op_ms)),
                    flush=True,
                )
                del repack_run
            del args, w1, w2, actual, expected, delta, baseline, run
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
