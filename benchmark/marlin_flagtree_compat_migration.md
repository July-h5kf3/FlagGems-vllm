# FlagTree-owned CANN 9.0 compatibility

The operator library no longer contains `common_ir.py`, `compiler.py` or `primitives.py`. It directly uses the existing public TLE raw APIs. FlagTree owns version detection, primitive ABI adaptation, compiler invocation and IR compatibility. Old Fixpipe-specific logic and last-kernel global IR state were removed.

Validation environment: physical NPU 2 (Ascend 910B4-1), CANN 9.0.0, torch 2.10.0+cpu, torch_npu 2.10.0.post2, Triton 3.5.1.

- Production MoE: 64 tests passed in 358.63 s.
- FlagTree CPU regressions: 9 passed, covering version gating, no-output calls, output-address ABI, unsupported scratch/ABI rejection, and mixed-core annotations.
- Standalone public primitives: CompareScalar 35 cases; GatherMask 25 cases; composed routes 32 cases; changed-input graph replay; 12 invalid mask signatures; INT4 Cast 20 encoding/size cases and three graph replays; nine valid and 10 invalid Cast signatures; Cube boundaries around tl.dot. All passed. The process asserted that neither flag_gems nor flaggems_vllm was imported.

Performance uses the previously frozen baseline: A = whole Cube plus original conditional TLE Vector; B = the production TLE Cube and FP32 Vector, now using FlagTree-owned compatibility. It is not a controlled before/after comparison of the compatibility-layer migration alone. Both A and B compile through the updated FlagTree backend. No legacy operator-library compiler hook is installed for the benchmark.

BF16/INT4, E=256, hidden=4096, intermediate=256. Three separately captured A/B graph groups, five alternating samples per group, 20 replays per sample. Medians of 15 samples, microseconds. A/B outputs match exactly for normal inputs, modified activations, all routes to expert zero, and restored routes. Small cases also pass the existing reference tolerance.

| M | top-k | A (us) | B (us) | B vs A |
| --- | --- | --- | --- | --- |
| 1 | 6 | 125.40 | 127.71 | +1.84% |
| 8 | 6 | 542.82 | 535.64 | -1.32% |
| 64 | 1 | 702.11 | 724.12 | +3.13% |
| 16 | 6 | 836.34 | 827.81 | -1.02% |
| 32 | 6 | 1300.53 | 1260.12 | -3.11% |
| 512 | 6 | 2390.76 | 2301.84 | -3.72% |
| 2048 | 6 | 2652.81 | 2566.98 | -3.24% |
| 16384 | 6 | 10369.58 | 10013.27 | -3.44% |

The M=64/top-k=1 case varies substantially between captures: the three A/B capture medians are retained in performance.json. Do not interpret these measurements as all cases being faster or as isolating the cost of migration.

CANN 9.1+ and unknown toolkit versions retain the native path; native CANN 9.1 device execution remains unvalidated. The CANN 9.0 path requires the packaged primitive C++ sources, CANN compiler/headers and FlagTree Template headers (discovered from a source checkout or provided by FLAGTREE_TEMPLATE_INCLUDE). Generated artifacts use Triton cache storage.

Ascend tile/merge settings, K=128 and two-stage buffering are unchanged. There is no NVIDIA autotune change, PyTorch compute fallback, MMA substitution or workspace-store deletion. Existing BF16/INT4 frontend restrictions remain unchanged.

Reproduction: /data/ldc/ops_work/flagtree-compat-migration contains the environment loader selecting the modified FlagTree Python checkout, tests, frozen baseline, benchmark, JSON samples and logs. The loader selects compiler source for the test environment; the standalone primitive test does not import the operator library. No new native compiler binary is required by these Python changes.

Additional integration check: loading FlagTree before the application, then importing the existing FlagGems dependency with its legacy compiler hook, passes fresh-cache reference checks for M=1 and M=40.
