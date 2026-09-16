# Hygon fused Marlin MoE optimization record

Target: `hygon/fused_marlin_moe`, upstream base `12bcf25`.
Environment: Hygon BW gfx936, 80 CUs, 65520 MB; PyTorch 2.4.1,
HIP 6.1.25065, Triton 3.6.0; `flagtree-dev-mixq`, GPU 0.

Repository `optimization.md` and `deep_opt.md` are absent. The available
`workflow.md` gates are followed using the measured trial record below.

Frozen baseline: pre-dequantized grouped PyTorch MoE, with expert route indices
prepared before timing. It is not vLLM Marlin. vLLM 0.6.2 has no registered
`torch.ops._moe_C.marlin_gemm_moe` in this container. Timing uses
`triton.testing.do_bench(warmup=100, rep=200)`, milliseconds.
Validation uses an independent FP32 accumulation oracle with the same stage
rounding contract. The timing baseline has an additional BF16 rounding before
router multiplication, so it is not the numerical oracle.

Core active set: Q={INT4,INT8,FP8,MXFP4}, BF16, E=8, K=1024, N=2048,
top-k=2, M={1,16,128}. Large set: same Q/dtype/E/top-k, K=4096, N=14336,
M={1,16}. Shapes and baselines are unchanged across trials.

| Trial | Hypothesis/change | Validation | Measurement/decision |
| --- | --- | --- | --- |
| 0 | Portable fused decoding plus direct/grouped GEMM | 81 tests and 13 additional checks passed | Core 12 cases: 2.06–6.41x, geometric mean 3.77x. Large INT4 M=1 0.842x; FP8 0.584/0.678x and MXFP4 0.429/0.497x. Trigger deeper optimization. |
| Profile | Locate direct INT4 M=1 time | PyTorch HIP profiler completed | Two GEMM kernels 1.158 ms (99.67% of GPU time); sum 3.84 us. No allocation/copy compute fallback. |
| 1 | Replace small-route GEMM with contiguous-K vector reduction GEMV | 16 small-route tests and 4 large numerical checks passed | INT8 M=1 improves 0.820→0.733 ms. INT4 1.169→2.015 ms and MXFP4 2.325→3.124 ms regress. Retain GEMV only for INT8. |
| 2 | Replace FP8/FP4 exponent arithmetic with exact FP32 bit decoding | 6 decode/partial-tile tests and 4 large numerical checks passed | FP8 1.688→1.504 / 5.188→4.493 ms; MXFP4 2.325→1.686 / 7.854→5.662 ms. Keep, but large cases remain below target. |
| 3 | Transpose GEMM operands so output-major weights become the left operand with contiguous reduction dimension | 91 checks passed before the new signed-zero test exposed FP8 -0 canonicalization; fixed sign injection with integer bit operations, then both exhaustive decode tests passed | FP8 improves to 1.256/3.739 ms; INT4/MXFP4 unchanged. Contiguous-K weight layout remains limiting. |

Raw logs and JSON measurements are stored in this directory. Relative RMS and
peak errors for large cases are checked against 0.01 and 0.02 respectively;
the original core benchmark tolerance remains unchanged.

| 4 | Cache a transposed quantized layout (no dequantized weights), invalidate on version/layout/storage/stream changes, bypass cache for inference tensors without a version | 18 mutation/layout/graph/tail checks passed | INT4 improves to 0.995/2.718 ms (0.999/1.294x). FP8 regresses; INT8/MXFP4 have no stable gain. Retain the cache only for INT4. |

| 5 | Split long K reductions into 8 independent FP32 partial sums; apply rounding, router weights and activation only after reduction | 16 focused tests and 8 large numerical checks passed | INT4 1.413/1.442x; INT8 1.330/1.473x; FP8 1.001/1.037x. MXFP4 still 0.708/0.727x. Keep. |

| 6 | Combine E2M1/E8M0 exponents directly into BF16, with exact exceptional-value handling | All 256 E8M0 encodings crossed with all E2M1 codes pass bit-exact comparison, including negative zero | Regressed MXFP4 to 1.556/6.139 ms; reverted. |

| 7 | Test software pipelining stages 2/3 to overlap loads with decode/dot operations | Both large MXFP4 numerical checks passed | Autotuning selected the existing stage-1 configurations. No improvement; remove additional candidates. |

Optimization stop: keep the validated improvements. Large MXFP4 remains below
the 0.9x target (approximately 0.71–0.75x). No torch fallback or relaxed tolerance
was introduced. This delivery is functionally implemented, with a performance
limitation for the large MXFP4 workload.

Final regression adjustment: FP16 MXFP4 with input-side routing weights had
one cancellation-sensitive element outside the original 2e-3/2e-2 tolerance.
It reproduced with and without split-K. Promoting FP16 dot operands to IEEE
FP32 passed the original test without tolerance changes. Keep conservative
non-split accumulation for input-side routing. The complete 142-test suite
passed, including an exact FP16-subnormal construction. BF16 uses the original
dot dtype.

Delivery correction: the 0.9x threshold above is the repository's trial
threshold, not the workspace acceptance criterion. agent.md requires vLLM,
0.95x against matching precision or 1.3x against BF16, and arithmetic averaging.
Our PyTorch surrogate cannot establish that acceptance. FP8 and MXFP4 large
cases also fall below 1.3x. Final measurements are reported separately.

Final address audit: cast pointer offset components to int64 before stride
multiplication. Add INT4 transpose and INT8 grouped tests with an expert stride
larger than 2 GiB; this avoids silent int32 overflow on large expert banks.


## Follow-up after checkpoint 52ae6d6

The final address audit exposed a JIT helper compatibility bug: Python constant
expert IDs have no `.to` method. Use `tl.cast` so `_decode` supports both
constant IDs and runtime tensor IDs, preserving int64 address arithmetic.

The full regression then reproduced the FP16 MXFP4 cancellation case, despite
IEEE FP32 dots. A separate double-precision GEMM oracle (rounded to FP32 before
router weighting, preserving the original stage contract) returned -0.07421875
at output [1,2553], matching the kernel; the old FP32 GEMM oracle returned
-0.078125. The kernel had zero tolerance failures against the double oracle,
while the FP32 oracle had one. Consequently the earlier claim that promoting
dot operands alone resolved the issue was insufficient.

A trial summing FP32 dot tiles in FP64 did not fix the comparison with the
FP32 oracle and was reverted. Production retains FP32 accumulation. FP16 test
references now use FP64 GEMMs followed by FP32 accumulator rounding, then the
same router/activation/FP16 stage rounding. BF16 references and all comparison
tolerances are unchanged. The targeted regression, exhaustive special decode,
and both >2 GiB expert-stride checks passed (5 tests).

Final verification: 144 tests passed in 61.43 seconds. All 12 small and eight
large benchmark cases passed numerical checks. Final arithmetic mean speedups
against the surrogate are 4.140x (small) and 1.143x (large). Large MXFP4 remains
0.715/0.750x. INT4 rebuilding on each call costs 5.750/7.423 ms for M=1/16;
cached-layout timings are 0.715/2.414 ms. This does not establish vLLM acceptance.
