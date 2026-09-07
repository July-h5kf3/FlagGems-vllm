# Ascend W4A16 INT4 fused Marlin MoE

The Ascend 910B implementation passes the 53-shape acceptance gate: every point
is at least 1.0x and call-count weighted speedup is at least 1.3x in both normal
operator calls and explicit NPUGraph replay. Individual results remain available
in the benchmark reports. This is an aligned operator-chain comparison.

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

For at most 64 routed rows, a single mixed al.custom launch combines direct
packing, two GEMMs, SwiGLU and weighted unpermutation. All physical AICs and their
paired AIVs enter five SyncAll<false>() hardware barriers; idle cores still
participate. Barrier flags 11/12/13 are separate from GEMM ring flags 2/3.
One allocation holds aligned GM intermediates, and stages reuse one UB scratch
area. Intermediate activations and workspaces are not cached across calls.

Larger inputs retain grouped routing with eight launches. At the largest shape,
metadata retains 128-row packing. Both projections merge adjacent 128-row tiles
belonging to the same expert into up to 256 rows, with 128-column GEMM tiles.
Odd expert tails retain 128 rows and the matching accumulator stride. The large
first projection uses a measured 19-core schedule; the second uses 20. The
20-core first-projection schedule had a less balanced strided task assignment
for this geometry.

Row packing processes 16 rows per iteration and zeros only partial blocks.
SwiGLU batches up to 32 rows and handles a final partial batch explicitly.
Weighted unpermutation prefetches two input buffers and uses a separate output
buffer. V-to-MTE2 fences protect input reuse; an MTE3-to-V fence protects output
reuse while the next row loads. FP32 Axpy performs each weighted accumulation.
Explicit drains complete every pending event before returning.

Vector cores dequantize INT4 into a two-slot GM ring; Cube cores consume each
slot while the next tile is prepared. No PyTorch compute fallback is used.

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
bash tools/run_marlin_ascend.sh tools/bench_marlin_ascend.py --m all --iters 30 --pairs 5
bash tools/run_marlin_ascend.sh tools/profile_marlin_ascend.py
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
Future optimizations must preserve the every-shape gate and the precision contract.
Cross-stream first use, backward, additional activation dtypes and a portable
upstream CommonIR compiler integration remain outside this initial scope.

## Measured status

The final version passes **56 functional tests**, including changed-input Graph
replay, partial SwiGLU batches, exact-half/fallback scales and repeated asynchronous
unpermutation. All **53 trace shapes** pass eager and captured-output comparison;
maximum absolute difference is **0.0**.

- Normal calls, weighted: **1.3939x**;
  minimum point **1.0264x**.
- NPUGraph replay, weighted: **1.4005x**;
  minimum point **1.0301x**.
- Profiler summed kernel duration, weighted: **1.4044x**;
  minimum point **1.0305x**.
- M=16384: normal **1.0264x**, Graph **1.0301x**.

**Acceptance passes.** The gate is every point >=1.0x plus weighted speedup >=1.3x,
not 1.3x at every point. The paired full run uses 30 iterations and five alternating
pairs per shape. A separate process repeats M=1,2,4,8,16384 with 50 iterations and
five pairs. Both modes pass again for each repeated point. The profiler warms each
shape five times and records five calls, totaling 1325 baseline and 1980 candidate
kernels. Profiler values are reported separately from event timing.

The baseline reconstructs the vLLM-Ascend W4A16 primitive chain with torch_npu/CANN
AscendC at revision 99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f. Both paths use FP32
router probabilities to match PR #741. The full AscendW4A16FusedMoEMethod.apply
at that revision casts them to activation dtype; it is not invoked here.
Router-logit selection, communication and one-time weight preparation are excluded
on both sides.

The branch is Ascend/fused_marlin_moe_w4a16_int4 and the preserved previous
checkpoint is c9bd34d. Rejected and intermediate experiments are retained under
ignored work/r3; they are not production dispatch paths. Source hashes and raw
timing pairs accompany this report.
