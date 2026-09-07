# Ascend W4A16 INT4 fused Marlin MoE

This is an Ascend 910B development implementation of
`flaggems_vllm.fused_marlin_moe_w4a16_int4`. The call-count weighted 1.3x target is met; per-shape acceptance remains incomplete:
the target is 1.3x versus the same-precision vLLM-Ascend AscendC path over the
53 shapes from FlagGems-vllm PR #741. Passing correctness is not performance acceptance.

## Contract

- BF16 activations and scales; symmetric `uint4b8` weights; group size 128.
- Activations: `[M,K]`; first weights: `[E,2N,K/2]`; second weights: `[E,K,N/2]`.
- Each uint8 stores the even K code in its low nibble and the odd K code in its high nibble.
  Dequantization is `(code - 8) * scale`, rounded to BF16 before Cube multiplication.
- First scales: `[E,2N,K/128]`; second scales: `[E,K,N/128]`.
- Router IDs: contiguous INT32 or INT64 `[M,top_k]`, with values in `[0,E)`.
  Router probabilities: contiguous FP32 with the same shape.
- Output: a new BF16 `[M,K]` tensor. Both GEMMs accumulate in FP32; intermediate
  projections and SwiGLU are rounded to BF16. The weighted reduction uses FP32.
- Forward inference only. Empty M is supported. FP16 activations, bias, other
  activations, asymmetric zero points, expert-parallel maps, caller-owned
  workspaces and in-place output currently raise explicit errors.
- All inputs must be contiguous on one NPU. K and N must be multiples of 128.
  Metadata checks bound index arithmetic and the tested UB footprint.

The main acceptance geometry is `E=256, K=4096, N=256, top_k=6` with BF16
activations, FP32 probabilities and the original PR's `randn * 0.1` / `rand * 0.03`
input distributions. The Ascend baseline is the primitive sequence in
vLLM-Ascend revision `99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f`:
`npu_moe_init_routing_v2`, two INT4 antiquant `npu_grouped_matmul` calls,
`npu_swiglu`, and `npu_moe_token_unpermute`.
The benchmark starts with preselected routes, as PR #741 does, and aligns both
paths on FP32 probabilities. It does not measure router-logit selection or communication.

## Implementation

For at most 32 routed rows, one AscendC kernel packs activations and builds direct
expert/restore metadata. This reduces the small path to five launches. Larger
inputs retain grouped expert routing. Triton owns metadata, layout preparation
and launch composition for that grouped path. `al.custom`
inlines source-hashed AscendC fragments for route compression, row packing,
the Vector/Cube GEMM pipeline, SwiGLU and weighted unpermutation.
Vector cores dequantize compressed INT4 tiles into a two-slot workspace;
Cube cores consume the tiles while the next tiles are prepared. The large-M
dispatch uses BM=128 and BN=256. SwiGLU batches 16 complete rows where appropriate;
the weighted restore overlaps two input buffers with vector work and uses explicit
V-to-MTE2 lifetime fences. Cross-core
events and the final drain protect slot reuse.

Launches query the physical AIC/AIV counts. Vector fragments distribute logical
tasks inside those physical launches. Source code, all static geometry and launch
parameters participate in the bitcode cache keys.

Only the compressed layout transformation, scale permutation and per-tile precision
flags are cached. For a scale magnitude of zero or in `[2^-14, 4096]`, multiplying
a BF16 scale by an INT4 integer is exactly representable in FP16: a BF16 significand
has 8 bits and the largest non-power-of-two INT4 magnitude is 7, so the product
needs at most 11 significant bits. It stays in the normal finite FP16 range.
The implementation uses this exact half multiply before the final BF16 rounding;
other scales retain the FP32 multiplication path. Flags are computed from the
actual scales, not inferred from test shapes.

Only compressed weights are retained across calls.
Tensor identity and mutation versions invalidate the cache. Inference tensors
without version counters bypass that cache. No dequantized floating weight cache
or PyTorch compute fallback is used. Torch usage in production is limited to
allocation, metadata and no-copy views; references live only in tests/benchmarks.
Cached preparation is intended for a warmed inference stream; concurrent first use
on independent streams has not been validated.

NVIDIA autotune configuration is unchanged: this change is an Ascend specialization.
The existing generic implementation remains available through the named public entry.
Ascend configurations are selected explicitly from source/profiler experiments.

## Reproduction on the development server

Tested environment: `zhiyuan-huawei`, container `flagtree-dev-ldc`,
`/data/ldc/ops_work/.venv`, Ascend 910B4, CANN 9.0, Torch 2.10.0+cpu,
torch_npu 2.10.0.post2 and FlagTree 0.6.0+ascend.gitf56cd1bd.
The checkout contains a frozen CANN 9.0 CommonIR compatibility adapter as a fallback.
In this development environment, importing FlagGems installs its adapter first;
the active adapter path and SHA256 are recorded in the environment manifest.
First use requires `ccec` and creates ignored `_build/` artifacts beside the fragments.

```bash
# Inside flagtree-dev-ldc, from this checkout:
bash tools/run_marlin_ascend.sh -m pytest tests/test_fused_marlin_moe_w4a16_int4.py -q
bash tools/run_marlin_ascend.sh tools/bench_marlin_ascend.py --m all --iters 20 --pairs 3
bash tools/run_marlin_ascend.sh -m pytest benchmark/test_fused_marlin_moe_w4a16_int4_ascend.py --mode operator --warmup 10 --iter 50
```

Set `MARLIN_DEVICE` to an available physical NPU. The paired runner records normal
operator calls and explicit NPUGraph replay separately, validates captured output,
alternates timing order, and writes every shape and repetition to JSON. Weights are
prepared and compiled before timing on both paths. The call-count weighted speedup
is `sum(calls * baseline_us) / sum(calls * candidate_us)`.
Event timing is screening evidence; profiler records are kept separately.

## Remaining work

Performance acceptance must use all 53 shapes and expose individual regressions.
Further launch fusion and better Cube/Vector overlap may be required. A useful
local improvement or a passing precision suite does not establish the 1.3x target.
Cross-stream first use, backward, additional activation dtypes and a portable
upstream CommonIR compiler integration remain outside this initial scope.

## Measured status

The continued implementation passes **47 functional tests** and **51 CI helper
tests**. All **53 shapes** pass both normal and captured-output comparison against
the same-precision AscendC baseline, with maximum absolute difference **0** on the trace.

- Normal operator calls, call-count weighted: **1.3362x**; 47/53 points reach 1.3x.
- Explicit NPUGraph replay, call-count weighted: **1.3627x**; 52/53 points reach 1.3x.
- `torch_npu.profiler`, median summed kernel time of five calls per shape: **1.3631x**.
  The recorded counts are exactly 1325 baseline and 2075 candidate kernels.

**Weighted acceptance is reached; the every-shape gate is not.** Normal-call
points below 1.3x are M=1,2,4,8,16,16384; graph points below 1.3x
are M=16384. M=16384 is now roughly 10.9 ms versus
13.2 ms in `ce0a5fd`, but remains slightly slower than its AscendC baseline.
Do not describe these weighted results as a 1.3x improvement for every shape.

All timings use the unchanged 53 shapes and call counts. Paired event runs alternate
ordering and use medians of three pairs. Profiler runs warm each path and shape five
times before recording five calls; its recording schedule itself has no warmup steps.
Raw rows, repeated pairs and profiler results are retained in `docs/benchmarks/`.
The rejected combined-planning kernel and other screening records are in ignored
`work/r2/`. Validated baselines remain in commits `418a6dc` and `ce0a5fd`.


## Updated acceptance gate

Every individual shape must have speedup >=1.0x in each reported mode, and the
call-count weighted speedup must remain >=1.3x. Full runs of
`tools/bench_marlin_ascend.py` now exit unsuccessfully if either condition fails.
The historical `all_shapes_1_3x` field remains a stricter diagnostic, not this gate.

The current result does not pass: ordinary-call regressions are M=1,2,4,8,16384;
the NPUGraph regression is M=16384. Aggregation cannot override these regressions.

Baseline scope clarification: the benchmark reconstructs the vLLM-Ascend W4A16
primitive chain using torch_npu/CANN AscendC calls. It does not invoke the complete
vLLM-Ascend entry. Both paths use FP32 route probabilities to match PR #741;
`AscendW4A16FusedMoEMethod.apply` in the recorded vLLM-Ascend revision instead casts
probabilities to the activation dtype before calling its full execution path.
Thus the recorded comparison is an aligned operator-chain benchmark, not a full
framework end-to-end benchmark.
