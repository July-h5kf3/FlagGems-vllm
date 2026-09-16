# Hygon fused Marlin MoE checkpoint

Branch: `hygon/fused_marlin_moe`; upstream base: `12bcf25`.
Reference: https://github.com/flagos-ai/FlagGems-vllm/pull/752

This checkpoint preserves the implementation and measured results. It is not a
performance acceptance claim. Tests ran on GPU 0 in `flagtree-dev-mixq` on
`zhiyuan-haiguang` (Hygon BW gfx936, PyTorch 2.4.1, Triton 3.6.0, vLLM 0.6.2).

## Recorded measurements

Final implementation, BF16 activations, E=8, K=4096, N=14336, top-k=2.
All eight cases passed numerical checks; raw data are in `final-large.log/json`.

| Quantization | M=1 latency / speedup | M=16 latency / speedup |
| --- | --- | --- |
| INT4 | 0.715 ms / 1.369x | 2.414 ms / 1.452x |
| INT8 | 0.737 ms / 1.343x | 2.390 ms / 1.480x |
| FP8 | 0.998 ms / 1.006x | 3.395 ms / 1.030x |
| MXFP4 | 1.368 ms / 0.715x | 5.285 ms / 0.750x |

The arithmetic mean over these eight measurements is 1.143x.
The 12 smaller cases (E=8,K=1024,N=2048,M=1/16/128,top-k=2) passed, with
arithmetic mean 4.140x and range
1.687–6.413x (`final-core.log/json`).
INT4 repacking on every call, with compilation already warm, costs a total
5.750/7.423 ms for M=1/16, versus 0.715/2.414 ms with cached layouts.

The baseline is pre-dequantized grouped PyTorch MoE, not native vLLM. Route
preparation and dequantization are excluded from baseline timing; compilation
and first-use INT4 layout packing are excluded from operator timing. The
container does not register `torch.ops._moe_C.marlin_gemm_moe`.

The workspace requires vLLM comparisons and 0.95x against matching precision
or 1.3x against BF16. These measurements cannot establish that acceptance.
Large FP8/MXFP4 also fall below 1.3x against the surrogate. Only a subset of the
reference PR's shapes has been measured. Repacking measurements include the operator and layout rebuilding, not compilation.
`benchmark.log/json` and `large-split.log` retain the earlier measurements;
use the `final-*` files for the final implementation. `optimization.md` records retained and rejected trials.

## Validation state at checkpoint

The final suite passed **144 tests** in 61.43 seconds (`final-tests.log`),
including int64 pointer arithmetic, constant expert IDs and expert strides
larger than 2 GiB. FP16 references now use FP64 GEMMs with FP32 accumulator
rounding; the old FP32 oracle could cross an intermediate FP16 midpoint.
Comparison tolerances are unchanged. See `optimization.md` for the diagnostic
results and the rejected FP64 accumulation trial. Production retains FP32
accumulation and no PyTorch compute fallback.

Black, isort, flake8, Python compilation, YAML parsing, CI test selection and
diff whitespace checks passed during implementation.

See `../hygon_fused_marlin_moe.md` for layout, API limitations and reproduction.
The API takes output-major byte weights, not vLLM int32 Marlin-repacked weights.
