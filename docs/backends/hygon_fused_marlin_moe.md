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

Up to 16 routes use direct expert lookup, with a dedicated GEMV for INT8. Larger calls compact routes per
expert and calculate prefix sums, then use grouped GEMM. Both paths fuse weight
decoding into GEMM. With output-side routing weights, up to 32 routes and `max(K,N)>=4096`, GEMMs use
eight K partitions and FP32 partial sums; a reduction kernel applies routing
and SiLU only after combining those sums. Other shapes and input-side routing weights use the original accumulation
order and fuse SiLU into GEMM1. A final Triton kernel sums top-k outputs.
INT4 quantized weights and scales are transposed with Triton and cached using weak
source references. Cache keys include the tensor version, shape, strides,
dtype, device, storage address and stream. Ordinary in-place mutations invalidate
the cached layout. Mutations through `.data` or external storage writers that
bypass PyTorch version tracking must not be used with cached weights. Inference
mode tensors without version counters are repacked on every call. Quantized
weight caches use approximately one additional copy of weights and scales;
there is no floating-point weight expansion.

A different stream repacks rather than reading a potentially unfinished copy.
Graph capture may reuse a cache warmed on the same stream; otherwise packing is
included in the captured graph and is not published to the global cache. Warm
up before graph capture to resolve autotuning. Graphs that reuse a cached
layout must be recaptured after modifying the corresponding weights/scales.

`hygon_marlin_gemm` and `hygon_marlin_gemv` in the Hygon tuning YAML tune
`BN`, `BK`, and warp count.
`BM=16` is fixed because routing depends on it. Routing, prefix, transpose, split-K reduction and top-k sum
use fixed launch configurations: these are linear auxiliary kernels, not
NVIDIA tuning targets. The autotune key records GEMM dimensions, route/expert
counts, quantization, stage, direct/grouped path and split-K count. No NVIDIA kernel/config
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

## Performance boundary

For `E=8,K=4096,N=14336,top-k=2,M=1/16` in the verified environment,
Final measurements put INT4/INT8 at approximately 1.34–1.48x,
FP8 at 1.01–1.03x and MXFP4 at 0.71–0.75x against the stated PyTorch baseline.
The workspace requires a vLLM baseline, 0.95x for matching precision or 1.3x
against BF16, summarized by the arithmetic mean. The surrogate baseline does
not satisfy that acceptance requirement; large FP8/MXFP4 also miss 1.3x.
Performance acceptance is therefore incomplete. BF16 exponent folding and stage-2/3 software pipelining trials
were rejected because they did not improve that workload. The surrogate results alone do not establish vLLM performance.

Final regression: 144 tests passed. Detailed final measurements and limitations
are in [the results record](hygon_marlin_results/README.md). INT4 layout rebuilding
on each call adds about 5 ms for the measured large expert bank; reuse the
version-tracked cache for steady-state inference.

Against native vLLM 0.6.2 BF16 `fused_experts`, the same eight large cases have
an arithmetic mean speedup of 1.029x, below the workspace target of 1.3x.
Per-format means are INT4 1.290x, INT8 1.258x, FP8 0.913x and MXFP4 0.654x.
Both implementations pass the numerical checks; no vLLM code was patched.
