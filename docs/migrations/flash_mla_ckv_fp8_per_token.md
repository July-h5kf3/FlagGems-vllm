# H800 FP8 MLA import provenance

- Source: https://github.com/flagos-ai/FlagGems/pull/5849
- Source commit: `ec53a81ee158ae976e3acde061924bf738935d89`.
- Required compiler: https://github.com/flagos-ai/FlagTree/pull/1001 at
  `2ace55ecae1e911c46668834a93807d9f5574540`; build with TLE enabled.
- Destination branch: `h800/flash_mla_ckv_fp8_per_token`.

This change relocates the existing implementation to `flaggems_vllm.ops`,
updates test and benchmark imports, exports the four original public symbols,
and adds the operator catalog entry. Top-level imports and `_FULL_CONFIG`
are derived from `ops.__all__` by the existing package initialization.
The implementation is copied byte-for-byte; no scheduling, numerical behavior,
or tuning parameters are changed. The original main branch is retained;
this branch starts from upstream main because the fork's PPU changes conflict
with current upstream.

## Scope and limitations

This is a source migration, not new kernel development or performance acceptance.
The imported implementation retains its fixed Hopper tuning and adaptive Split-K
policy; no libtuner conversion is attempted. It also retains upstream PyTorch
quantization, metadata creation, padding/copy, and scheduler operations. Therefore
it has NOT passed the repository's no-torch-compute production-path gate.
Validation below was performed on the migrated implementation; historical
performance claims from the source PR are not treated as new measurements.
The API targets Hopper FP8 CKV caches with BF16 RoPE and is forward-only.

## Validation entry points

With a matching TLE-enabled compiler and an idle Hopper GPU:

```sh
PYTHONPATH=src:$PYTHONPATH python -m pytest -q tests/test_flash_mla_ckv_fp8_per_token.py --quick
PYTHONPATH=src:$PYTHONPATH python -m pytest -q benchmark/test_flash_mla_ckv_fp8_per_token.py --collect-only
```

The benchmark uses the 24-shape matrix and the CUDA FP8 implementation from
`meituan-longcat/FlashMLA`, branch `feature/ckv_fp8_per_token`, validated at
`a29b228de7f4152f10afc9d3ad1b95dd3aa52ec3`. Build its `flash_mla_fp8` package
and make it importable before running the benchmark; otherwise the benchmark
is skipped. Both implementations consume the same quantized tensors. CUDA
scheduler metadata and TLE preparation are performed before steady-state timing.
The CUDA cache tensors add a singleton KV-head dimension using no-copy views.
The earlier BF16 baseline is superseded by this CUDA FP8 baseline.

## Migration validation (2026-09-16)

On H800 with the TLE-enabled compiler revision above:

- The migrated operator test file passed all three tests.
- All 24 standard benchmark shapes passed output/LSE accuracy checks against
  the CUDA FP8 reference and deterministic graph replay checks.
- Maximum output relative L2 error was 0.007768; maximum LSE absolute error
  was 2.862e-6.

These results cover operator execution, not end-to-end vLLM model integration.
Keep this migration as a draft while the compiler prerequisite and the
production-path policy gaps described above remain unresolved.
