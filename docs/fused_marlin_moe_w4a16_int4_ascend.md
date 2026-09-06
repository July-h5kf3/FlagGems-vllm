# Ascend W4A16 INT4 fused Marlin MoE

This is an Ascend 910B development implementation of
`flaggems_vllm.fused_marlin_moe_w4a16_int4`. Performance acceptance is pending:
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

Triton owns metadata, layout preparation and launch composition. `al.custom`
inlines source-hashed AscendC fragments for route compression, row packing,
the Vector/Cube GEMM pipeline, SwiGLU and weighted unpermutation.
Vector cores dequantize compressed INT4 tiles into a two-slot workspace;
Cube cores consume the tiles while the next tiles are prepared. Cross-core
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

The final BF16 functional suite passed 41 tests. All 53 trace shapes passed normal
and captured-output comparison against the AscendC baseline with maximum absolute
difference 0 in that workload. CI selection helper tests passed 51 tests.

Call-count weighted results over all 53 shapes:

- Normal operator calls: **1.2146x**, with **47/53** points at least 1.3x.
- Explicit NPUGraph replay: **1.2401x**, with **50/53** points at least 1.3x.
- `torch_npu.profiler`, median summed kernel durations over five invocations:
  **1.2442x**. The trace contains exactly 1325 baseline and 2120 candidate kernels.

**The requested 1.3x acceptance target has not been met.** Normal-call points below
1.3x are M=1,2,4,8,16,16384; graph points below 1.3x are M=1,2,16384.
The largest shape remains a regression, so the aggregate gain must not be presented
as a completed performance migration. The profiler was warmed by five calls of
each path and shape before recording; the profiling schedule itself had no warmup steps.

Full rows, repeated pairs, profiling results and environment fingerprints are in
`docs/benchmarks/marlin_w4a16_int4_ascend_{results,profiler,environment}.json`.
Exploratory logs and rejected implementations are retained under ignored `work/`.
The reproducible development baseline is commit `418a6dc`.
