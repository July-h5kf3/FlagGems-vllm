# Hygon fused Marlin MoE

`flaggems_vllm.fused_marlin_moe` selects the Hygon implementation through the
existing vendor registrar. The reference interface and quantization tags follow
[PR #752](https://github.com/flagos-ai/FlagGems-vllm/pull/752), revision
`8b6b436af999f4b9d386399f2ebe1373bdec63f3`. The implementation uses portable Triton
loads and dot products, without PPU TLE instructions or NVIDIA PTX.

## Input and output contract

Let `M` be tokens, `E` experts, `K` hidden size, `N` intermediate size and `T`
top-k. Activations are contiguous `[M,K]` FP16/BF16. Routing IDs are contiguous
`[M,T]` int32/int64 with values in `[0,E)`; routing weights have the same shape
and FP16/BF16/FP32 dtype. Duplicate expert IDs are supported. Invalid ID values
are outside the contract; the operator does not synchronize to validate GPU data.
All tensors must be on the same device and must not require gradients.

| Quantization tag | Weight layout w1 / w2 | Scale layout / dtype | Group size |
| --- | --- | --- | --- |
| 0: INT4 uint4b8 | `[E,2N,K/2]` / `[E,K,N/2]` uint8 | `[E,2N,K/G]` / `[E,K,N/G]`, activation dtype | 128 |
| 1: INT8 uint8b128 | `[E,2N,K]` / `[E,K,N]` uint8 | same scale layout, activation dtype | 128 |
| 2: FP8 E4M3FN | `[E,2N,K]` / `[E,K,N]` uint8 or float8_e4m3fn | same scale layout, FP16/BF16/FP32 | 32/64/128 or -1 |
| 6: MXFP4 E2M1 | `[E,2N,K/2]` / `[E,K,N/2]` uint8 | same scale layout, E8M0 uint8 or float8_e8m0fnu | 32 |

`G=-1` means one scale per output channel. Four-bit codes use low nibble for
even reduction indices and high nibble for odd indices. INT4 and INT8 decode
with offsets 8 and 128. Weights/scales can have arbitrary strides. `K` and `N`
must be positive multiples of 32 and divisible by group size when positive.
This layout is **not** the int32 vLLM Marlin-repacked layout.

Output is `[M,K]`, with the activation dtype. FP16 dot operands are promoted
to IEEE FP32 inside Triton for precision;
BF16 uses BF16 dot operands. GEMMs accumulate in FP32; decoded
weights and the intermediate activation use activation dtype. GEMM1 rounds its
gate/up outputs to activation dtype before FP32 SiLU and multiplication. When
`apply_router_weight_on_input=True`, routing weights multiply both GEMM1 gate
and up accumulators before rounding/activation; otherwise they multiply GEMM2
accumulators before rounding and top-k summation. This follows the referenced
PPU implementation's placement. `inplace=True` and an explicit output tensor
are mutually exclusive. An output may alias activations; it may not share
storage with live weights, scales or routing tensors. Empty `M` is supported.

Bias, act-order, expert maps, custom activations/reductions, caller-provided
workspaces, global scaling, clamp limits, FP8 activations and backward are
explicitly unsupported. Only SiLU/SwiGLU and local expert weights are supported.
Nonempty calls support at most 16384 routes and 1024 experts. The grouped route
workspace is `E * ceil(M*T/16)*16` int32 elements.

## Implementation and tuning

Up to 2 routes for `max(K,N)>=4096`, or 16 routes for smaller weights, use direct
expert lookup, with a dedicated GEMV for INT8. Larger calls compact routes per
expert and calculate prefix sums, then use grouped GEMM. Both paths fuse weight
decoding into GEMM. With output-side routing weights, up to 32 routes and `max(K,N)>=4096`, GEMMs use
eight K partitions and FP32 partial sums; a reduction kernel applies routing
and SiLU only after combining those sums. Other shapes and input-side routing weights use the original accumulation
order and fuse SiLU into GEMM1. A final Triton kernel sums top-k outputs.
INT4 quantized weights and scales are transposed with Triton. MXFP4 weights
are decoded to exact twice-value signed integers in a transposed byte cache,
with a reserved code for negative zero; half-E8M0 scales are cached in BF16.
Both formats use weak source references. Cache keys include the tensor version, shape, strides,
dtype, device, storage address and stream. Ordinary in-place mutations invalidate
the cached layout. Mutations through `.data` or external storage writers that
bypass PyTorch version tracking must not be used with cached weights. Inference
mode tensors without version counters are repacked on every call. The INT4 cache
adds one copy of packed weights and scales. The MXFP4 cache uses one byte per
weight and two bytes per scale, with no floating-point weight expansion.

A different stream repacks rather than reading a potentially unfinished copy.
Graph capture may reuse a cache warmed on the same stream; otherwise packing is
included in the captured graph and is not published to the global cache. Warm
up before graph capture to resolve autotuning. Graphs that reuse a cached
layout must be recaptured after modifying the corresponding weights/scales.

`hygon_marlin_gemm` and `hygon_marlin_gemv` in the Hygon tuning YAML tune
`BN`, `BK`, and warp count.
The host selects `BM=64` for `R>=32*E`, `BM=32` for `R>=16*E`, and
`BM=16` otherwise, using the same tile for routing and GEMM. Neighboring N tiles
of an expert execute consecutively. Routing, prefix, transpose, split-K reduction and top-k sum
use fixed launch configurations: these are linear auxiliary kernels, not
NVIDIA tuning targets. The autotune key records GEMM dimensions, route/expert
counts, row tile, quantization, stage, direct/grouped path and split-K count. No NVIDIA kernel/config
is changed.

Production PyTorch use is limited to allocation, metadata, byte views and a
device guard. No reference implementation is imported by production code.

FP16 correctness references use FP64 GEMMs followed by FP32 accumulator
rounding to avoid FP32 oracle errors at intermediate FP16 rounding boundaries.
BF16 references use FP32 GEMMs.

## Reproduction

Use the existing `flagtree-dev-mixq` container on `zhiyuan-haiguang`, with repository
`/home/JSST/shared_workspace_mixq/ldc/ops_work/FlagGems-vllm`. Check `hy-smi` for an
idle device before setting `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`.

```bash
PYTHONPATH=src python -m pytest -q tests/test_fused_marlin_moe_hygon.py
PYTHONPATH=src python -m pytest -q -s benchmark/test_fused_marlin_moe_hygon.py
# Upstream-sized BF16 checks and timings (optional quant tags and M list):
PYTHONPATH=src:. python benchmark/hygon_marlin_large.py
# Include INT4 repacking on every call in a separate measurement:
PYTHONPATH=src:. python benchmark/hygon_marlin_large.py 0 1 cold
```

The benchmark compares with grouped PyTorch matmuls over pre-dequantized
weights. Routing indices and dequantization are excluded from baseline timing.
Autotuning and first-use layout packing are excluded from operator timing;
these are steady-state timings with version-tracked weights. Metrics are event-timed milliseconds
and `baseline_ms/operator_ms`, with arithmetic mean over the active set. This is
not a comparison with vLLM Marlin or end-to-end serving.

The verified container has PyTorch 2.4.1, Triton 3.6.0, vLLM 0.6.2 and a 64 GB
Hygon BW gfx936. `torch.ops._moe_C.marlin_gemm_moe` is not registered there, so a
vLLM Marlin performance comparison is unavailable in this environment. The
native BF16 `fused_experts` path is available and has now been measured; see
[the results record](hygon_marlin_results/README.md#native-vllm-bf16-comparison).

## Current performance checkpoint

The latest production candidate passed all 162 functional tests on gfx936
(848.06 seconds). This includes both FP16/BF16 inputs, all four public quant
formats, nonfinite/subnormal values, cache invalidation and dense/skewed routing.
Performance acceptance remains incomplete.

For BF16 inputs, `E=8,K=4096,N=14336,topk=2`, and
`M=1,4,8,16,32,64,128,256`, current steady-state results are:

| Format | vLLM baseline | Mean speedup | Minimum speedup |
|---|---|---:|---:|
| INT4 | BF16 fused_experts | 1.427x | 1.229x |
| INT8 | INT8 W8A16 fused_experts | 1.448x | 1.116x |
| FP8 | BF16 fused_experts | 1.176x | 1.062x |
| MXFP4 | BF16 fused_experts | 1.352x | 1.189x |

INT8 uses a common supported subset: one channel scale repeated across our
quantization groups. It does not validate vLLM support for arbitrary group128
scales. Each implementation is independently checked against the numerical
reference. The ratios are medians of three alternating-order measurements;
summary means are arithmetic means across shapes. Compilation and first-use
packing are excluded. Cold-cache and inference-mode repacking have separate costs.

MXFP4 BF16 inputs also passed all eight token counts for
`E=256,K=7168,N=2048,topk=8`, with mean1.158x and minimum1.104x. This large-bank
baseline uses a benchmark-process-only vLLM expert-offset int64 correction:
stock vLLM0.6.2 generates unsafe 32-bit addressing beyond2GiB on this device.
The correction passed an actual greater-than2GiB-stride GEMM check; it does
not modify the installed package. The sweep prints the exact patch and hashes.
The E8 results above need no address correction.

FP16 performance has not passed: MXFP4 E8 M1/M16 currently obtains
0.918x/0.990x against vLLM FP16. Other geometries/formats remain to be swept.
Do not interpret the BF16 subset as full acceptance. Previous checkpoint
measurements are retained in [the historical results](hygon_marlin_results/README.md);
[current measurements](hygon_marlin_results/optimization-checkpoint.json) include
source hashes, all measured samples and numerical errors.

```bash
PYTHONPATH=src:. python benchmark/hygon_marlin_sweep.py --q 6 --geometry 0
PYTHONPATH=src:. python benchmark/hygon_marlin_sweep.py --q 1 --geometry 0 --baseline int8
PYTHONPATH=src:. python benchmark/hygon_marlin_sweep.py --q 6 --geometry 1
PYTHONPATH=src:. python benchmark/hygon_marlin_sweep.py --q 6 --geometry 0 --dtype float16 --tokens 1,16
```
