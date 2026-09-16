# Hygon fused Marlin MoE checkpoint

Branch: `hygon/fused_marlin_moe`; upstream base: `12bcf25`.
Reference: https://github.com/flagos-ai/FlagGems-vllm/pull/752

This checkpoint preserves the implementation and measured results. It is not a
performance acceptance claim. Tests ran on GPU 0 in `flagtree-dev-mixq` on
`zhiyuan-haiguang` (Hygon BW gfx936, PyTorch 2.4.1, Triton 3.6.0, vLLM 0.6.2).

## Recorded measurements

BF16 activations, E=8, K=4096, N=14336, top-k=2. Numbers below are from
`large-split.log`, before the final int64 address arithmetic change.

| Quantization | M=1 latency / speedup | M=16 latency / speedup |
| --- | --- | --- |
| INT4 | 0.713 ms / 1.413x | 2.421 ms / 1.442x |
| INT8 | 0.734 ms / 1.330x | 2.387 ms / 1.473x |
| FP8 | 0.994 ms / 1.001x | 3.397 ms / 1.037x |
| MXFP4 | 1.377 ms / 0.708x | 5.381 ms / 0.727x |

The arithmetic mean over these eight measurements is approximately 1.141x.
The baseline is pre-dequantized grouped PyTorch MoE, not native vLLM. Route
preparation and dequantization are excluded from baseline timing; compilation
and first-use INT4 layout packing are excluded from operator timing. The
container does not register `torch.ops._moe_C.marlin_gemm_moe`.

The workspace requires vLLM comparisons and 0.95x against matching precision
or 1.3x against BF16. These measurements cannot establish that acceptance.
Large FP8/MXFP4 also fall below 1.3x against the surrogate. Only a subset of the
reference PR's shapes has been measured. Cold/repacking cost is not yet measured.
`benchmark.log/json` contain the initial smaller-shape measurements, not final
implementation results. `optimization.md` records retained and rejected trials.

## Validation state at checkpoint

The suite passed 142 tests before the final address audit, with unchanged
numerical tolerances. The final implementation adds int64 pointer arithmetic
and two tests with an expert stride greater than 2 GiB. Its 144-case regression
was still running when this checkpoint was requested; no completed result is
claimed for that run. Black, isort, flake8, Python compilation, YAML parsing,
CI test selection and diff whitespace checks passed during implementation.
The host call audit shows allocations, metadata and views; production has no
PyTorch compute fallback.

See `../hygon_fused_marlin_moe.md` for layout, API limitations and reproduction.
The API takes output-major byte weights, not vLLM int32 Marlin-repacked weights.
