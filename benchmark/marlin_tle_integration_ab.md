# Integrated TLE Marlin validation

Environment: Ascend 910B4-1, physical NPU 2, CANN 9.0.0, torch 2.10.0+cpu, torch_npu 2.10.0.post2, Triton 3.5.1.

A: frozen whole-Cube custom with the original conditional TLE Vector dequantization. Routing uses the same FlagTree primitives as B. This is not the original whole-MoE baseline.

B: production `marlin_w4a16` modules using FlagTree CompareScalar, GatherMask, INT4 Cast, cube_begin/end; TLE task/K loops and default `tl.dot`; unconditional FP32 scaling. No experimental operator import, MMA replacement or workspace-store deletion.

Fixed workload: BF16 activations/scales, packed INT4 weights, 256 experts, hidden dimension 4096, intermediate dimension 256. Three separately captured A/B graph groups, alternating capture/run order, five samples per group, 20 replays per sample. Values below are medians of 15 samples in microseconds; negative change means faster.

| M | top-k | A (us) | B (us) | Change |
| --- | --- | --- | --- | --- |
| 1 | 6 | 123.68 | 125.38 | +1.37% |
| 8 | 6 | 542.44 | 527.39 | -2.77% |
| 64 | 1 | 719.77 | 685.41 | -4.77% |
| 16 | 6 | 838.22 | 831.33 | -0.82% |
| 32 | 6 | 1304.46 | 1249.23 | -4.23% |
| 512 | 6 | 2389.51 | 2285.92 | -4.34% |
| 2048 | 6 | 2653.17 | 2592.03 | -2.30% |
| 16384 | 6 | 10375.79 | 9975.65 | -3.86% |

All A/B graph outputs match exactly, including modified activations, routing all tokens to expert zero, and restoring changed routes. Cases with M <= 32 also pass the existing reference tolerance (atol=rtol=0.02). Graph allocation and capture introduce variation; these results do not establish performance for unmeasured shapes.

Production validation: `tests/test_fused_marlin_moe_w4a16_int4.py`: 64 passed (329.03 s). CPU regression `test_custom_without_outputs_preserves_regions` passed. Relevant Black/isort/flake8 and diff checks passed. Production torch calls are uninitialized allocation or metadata only; tl.Tensor conversions are device TLE code, not PyTorch compute fallbacks.

FlagTree validation: CompareScalar 35 cases; GatherMask 25 cases; composed routing 32 cases; changed-input graph replay; mask 12 invalid signatures; Cast 20 encoding/size cases and three changed-input replays; Cast nine valid sizes and 10 invalid signatures; Cube boundaries around a TLE dot pass exactly. Boundary ordinary/mix bitcodes compile and link.

The device primitive checks use the CANN 9.0 ABI adapter in the operator. Native public custom ABI execution on CANN 9.1 remains unvalidated. The separate Cast registration check runs before installing that adapter, since registration tests assert the original public symbol name.

Ascend launch tuning (K=128, two stages, existing tile/merge choices) is retained. No NVIDIA autotune change is applicable. BF16/INT4 shape, layout and unsupported-input behavior remain governed by the existing frontend. No training/backward support is added.

Reproduction artifacts on the task server: `/data/ldc/ops_work/marlin-integration-validation/` contains the frozen baseline, environment loader, tests, benchmark script, JSON samples and logs. Run `docker exec flagtree-dev-ldc bash /data/ldc/ops_work/marlin-integration-validation/run.sh bench.py`. The launcher verifies NPU 2 is idle first. The environment loader only selects the FlagTree checkout; B calls the production operator directly.
