<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# MThreads sparse FP8 MLA optimization progress

Optimization is paused at the user's request on 2026-11-09. The weighted target
is **1.3×** and has **not** been achieved. The latest validated runtime code is
`dbba4fe6ded483b6813723dc6d92185b83e1e3db`; subsequent experiments described below
have not been incorporated into the operator.

## Validated implementation

The baseline is native MATE 0.2.4 MODEL1 D512 on MTT S5000, with BF16 QK/PV
computation and FP8 KV-cache storage. The environment uses vLLM 0.24.0 empty.
The frozen matrix contains 414 shapes and 13,920 original calls. Speedups are
weighted by each shape's call count; the baseline, inputs and weights remain
unchanged.

| Measurement | Result |
| --- | ---: |
| Previous full weighted speedup (`6133827`) | 1.03863030× |
| Current full weighted speedup (`dbba4fe`) | **1.06965444×** |
| Current result with previous unaffected-shape timings | 1.06813553× |
| Affected group | 266 shapes / 8,298 calls |
| Affected weighted speedup, before → after | 1.12657871× → 1.17607411× |

The retained change removes a redundant producer barrier before probability
publication and extends double probability buffering to long TopK. The original
error checks, scale handling, whole-row repair and single-buffer compilation
fallback remain.

All 414 shapes passed the original all-row FP32 comparison with output
`atol=0.002` and LSE `atol=0.0002`. Maximum absolute errors were
0.000268153846 and 0.000001907349, respectively. Private and stock FlagTree each
passed 179 functional cases and the core benchmark, with four complete latency
and speedup records per core JSON. Each individual pipeline prototype and the
combined candidate passed 432 additional eager/graph comparisons, 12 optional
workspace fallbacks and four compile/fallback comparisons. Full pre-commit,
54 CI helper tests, CI pins and diff checks passed.

## Subsequent experiments

These are diagnostics on subsets, not new full-matrix acceptance results.
Timings below are median complete prepared calls. Each pair used the same
process, device and frozen input. Unless stated otherwise, the original strict
FP32 output comparison passed. Rejected variants are excluded from runtime code.

| Experiment | Representative eager timing, before → candidate | Decision |
| --- | --- | --- |
| Centered PV: M16 common sum plus M64 residual, shared result transfer | B512/K640: 0.35844 → 0.45376 ms; B512/K8320: 2.94458 → 3.27676 ms | Rejected; more spills and slower calls |
| Separate FP32 vector common-sum pass | B512/K640: 0.35878 → 0.72072 ms; B512/K8320: 2.94608 → 5.69728 ms | Rejected; additional KV pass dominates |
| Eight consumer warps, split M64 layout | B512/K640: 0.35100 → 0.45844 ms | Rejected; additional spills |
| Eight producer warps | B512/K640: 0.35824 → 0.56670 ms; B512/K8320: 2.93982 → 5.57360 ms | Rejected |
| Paired FP8 conversion option | B512/K640: 0.35788 → 0.40704 ms; B512/K8320: 2.93938 → 3.57004 ms | Rejected despite bitwise conversion equivalence |
| FP16 producer residual | B512/K8320: 2.93708 → 2.91568 ms | Not adopted; weak gain, no stable medium-TopK gain, additional rounding budget required |
| Shared PV scale across heads | B4112/K640: 2.59114 → 2.71448 ms; B512/K8320: 2.94180 → 3.02924 ms | Rejected; early weight scheduling also regressed |
| Short-query split warp-specialized pipeline | B16/K8320: 141.36 → 261.80 µs | Rejected |
| Reordered short-query split tasks | B16/K8320: 141.72 → 214.38 µs | Rejected |
| Fused metadata with reordered split tasks | B16/K8320: 141.80 → 224.84 µs | Rejected |
| Two raw KV buffers, uncached metadata | B16/K8320: 141.72 → 175.08 µs | Rejected |
| Move next-page copy after QK issue | B16/K8320: 141.54 → 171.46 µs | Rejected |
| Aligned packed query reduction | B16/K640: 42.40 → 41.60 µs, graph 45.88 → 46.18 µs; B16/K8320: 166.60 → 181.88 µs | Not adopted; mixed results and extra layout conversion |

The native-M32 eight-consumer variant exceeded shared-memory limits and was
not launched. The explicit M64N128 QK layout also exceeded the limit. A cached
suffix variant of the two-buffer kernel required 204,800 bytes, exceeding the
196,608-byte device limit, and was not device-tested.

Several explicit-layout variants failed compilation with the scalar-reduction
layout verifier. No compiler patch or system environment replacement was made.
An initial reordered split trial silently used the legacy preparation fallback;
its timing was discarded. The corrected trial checked the actual fused-kernel
name before timing. New short-path verification also inspects the fragment
flags when repair is fused into combine, so output correctness alone cannot
be mistaken for evidence that the approximate fast path ran.

## Pending packed-query reduction work

The remaining candidates reuse the existing native four-value FP8-to-FP16
conversion to compute four absolute query values per word. FP16 subtotals are
converted to FP32, reduced and multiplied by 1.00154. Attention QK/PV arithmetic
is unchanged. The aligned short variant above is not retained.

Completed evidence:

- A device primitive check covered all 254 finite FP8 codes and 8,192 four-term
  groups. Native subtotals matched the CPU FP16 reference exactly.
- Shared query storage round-tripped its bytes exactly under an actual SQMMA
  consumer. The computed query L1 bound passed the upper-bound check.
- Exact rational analysis using the actual FP32 inflation gives a lower ratio
  of 1.00006306386 to the exact L1 sum, above the original FP32 summation upper
  ratio of 1.00003045890. Positive finite query-scale multiplication preserves
  that ordering. Four FP8 magnitudes sum to at most 1792, avoiding FP16 overflow.
- Unaligned packed preparation and a small shared-memory layout bridge compiled
  without a GPU, retaining 148,480 bytes of shared memory. This does not prove
  their device behavior or performance.

Pending validation:

- Complete operator validation and paired timing for the unaligned short
  candidate and the shared-memory bridge.
- Validation and timing for the packed reduction in the separate large-batch
  query-bound kernel.
- For any promising candidate: dynamic contents/scales/indices, tails, actual
  repair and fallback, private/stock FlagTree, then the complete 414-shape matrix
  with the original weights and tolerances.

Recent device launches were rejected by the occupancy guard while other tasks
held device allocations. Rejected launches are not completed tests. The guard
was not relaxed and other processes were not stopped. Resume only after a fresh
idle-device check or an explicitly provided test window.

## Evidence and reproduction

The development workspace retains raw logs, source prototypes, before/after IR,
exit records and result JSON files under:

- `reports/mthreads-pr17-publish-20261109`: validated runtime, full matrix and
  precision summary, private/stock core JSON files and delivery comparison.
- `reports/mthreads-pr17-centered-20261109`,
  `reports/mthreads-pr17-consumer8-20261109`,
  `reports/mthreads-pr17-burst-20261109`,
  `reports/mthreads-pr17-fixed-scale-20261109`: rejected data-flow experiments.
- `reports/mthreads-pr17-short-ws-20261109`,
  `reports/mthreads-pr17-short-prefetch-20261109`: rejected short-query pipelines.
- `reports/mthreads-pr17-packed-qnorm-20261109`: primitive checks, rounding proof
  and pending packed-reduction candidates.

Validated operator commands:

```bash
python -m pytest -q -m flash_mla_with_kvcache_sparse_w8a8_fp8 --quick \
  tests/test_flash_mla_with_kvcache_sparse_w8a8_fp8.py
python -m pytest -q -m flash_mla_with_kvcache_sparse_w8a8_fp8 \
  benchmark/test_flash_mla_with_kvcache_sparse_w8a8_fp8.py \
  --level core --warmup 1 --iter 1 --record json --output benchmark_sparse_fp8.json
python -m pytest -q -m flash_mla_with_kvcache_sparse_w8a8_fp8 \
  benchmark/test_flash_mla_with_kvcache_sparse_w8a8_fp8.py \
  --level comprehensive --warmup 25 --iter 100 --record json \
  --output benchmark_sparse_fp8_full.json
```

The fork PR base remains `6e951850a50ad176d644eb17e81ca9d3a5fb9c2a`.
