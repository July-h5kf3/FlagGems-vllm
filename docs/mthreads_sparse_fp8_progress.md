<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# MThreads sparse FP8 MLA optimization progress

Optimization resumed at the user's request on 2026-11-10. The weighted target
is **1.3×** and has **not** been achieved. The latest validated change extends
the existing eight-token-lane KV copy layout to long TopK. It builds on the
previous validated runtime `dbba4fe6ded483b6813723dc6d92185b83e1e3db`.

## Validated implementation

The baseline is native MATE 0.2.4 MODEL1 D512 on MTT S5000, with BF16 QK/PV
computation and FP8 KV-cache storage. The environment uses vLLM 0.24.0 empty.
The frozen matrix contains 414 shapes and 13,920 original calls. Speedups are
weighted by each shape's call count; the baseline, inputs and weights remain
unchanged.

| Measurement | Result |
| --- | ---: |
| Previous full weighted speedup (`dbba4fe`) | 1.06965444× |
| Current full weighted speedup | **1.07757864×** |
| Current result with previous unaffected-shape timings | 1.07813284× |
| Affected group | 133 shapes / 4,045 calls |
| Affected weighted speedup, before → after | 1.18026625× → 1.20944285× |

The current change uses eight token lanes per warp for every TopK greater than
128, extending the existing medium-TopK copy layout to long TopK. The producer
row-warp count stays at 16. At the 128-token tile size, these cover 128 distinct
token rows without duplicate copies. The long-TopK main kernel's spills drop
from 13 to zero; its 32 total warps and 196,608 shared-memory bytes are unchanged.

The previous producer-barrier removal and double probability buffering remain.
The numerical expressions, error checks, scale handling, whole-row repair and
single-buffer compilation fallback are unchanged by this copy-layout update.

All 414 shapes passed the original all-row FP32 comparison with output
`atol=0.002` and LSE `atol=0.0002`. Maximum absolute errors were
0.000268153846 and 0.000001907349, respectively. Private and stock FlagTree each
passed 179 functional cases and the core benchmark, with four complete latency
and speedup records per core JSON. The current candidate passed 504 additional
eager/graph comparisons, 14 optional workspace fallbacks and four compile/fallback
comparisons. These include the additional long tail at TopK 8193. Full pre-commit,
54 CI helper tests, CI pins and diff checks passed.

Two independent cards repeated seven alternating complete-call timing pairs
on the frozen B512/K8320 input:

| GPU | Eager before → candidate | Graph before → candidate |
| --- | --- | --- |
| 4 | 2.87494 → 2.78484 ms | 2.91086 → 2.82514 ms |
| 7 | 2.82796 → 2.73540 ms | 2.86585 → 2.77119 ms |

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
layout verifier. These initial experiments did not modify the compiler or
system environment; the later isolated compiler diagnostic is described below.
An initial reordered split trial silently used the legacy preparation fallback;
its timing was discarded. The corrected trial checked the actual fused-kernel
name before timing. New short-path verification also inspects the fragment
flags when repair is fused into combine, so output correctness alone cannot
be mistaken for evidence that the approximate fast path ran.

## Packed-query reduction work

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

The previously pending unaligned short candidate, shared-memory bridge and
large-batch query-bound candidate completed the subset device checks below.
None was adopted. Any promising future candidate still requires dynamic
contents/scales/indices, tails, actual repair and fallback, private/stock
FlagTree, and the complete 414-shape matrix with original weights and tolerances.

On 2026-11-09, device launches were rejected by the occupancy guard while other
tasks held device allocations. Rejected launches are not completed tests. On
2026-11-10, testing resumed after fresh checks found available cards. The guard
was not relaxed and other processes were not stopped.

## Resumed device experiments, 2026-11-10

Eight candidates completed 27 fixed-input paired scenarios on the existing
private runtime. The complete prepared calls were measured in alternating
eager and graph pairs. All scenarios passed the original frozen comparisons
and checked actual fast-path flags. These are BF16 subset checks, not new
414-shape or stock-runtime acceptance results. No candidate had a sufficient
stable complete-call benefit for adoption.

| Candidate | Representative eager timing, before → candidate | Result |
| --- | --- | --- |
| Unaligned packed query L1 | B16/K8320: 166.08 → 216.96 µs | Rejected; spills increase from 3 to 72 |
| Packed shared-memory bridge | B16/K8320: 168.48 → 177.76 µs | Rejected |
| Packed large query-bound prepass | B512/K640: 350.18 → 347.40 µs | Query stage improves, but complete graph timing regresses slightly |
| Outlined fast combine | B16/K640: 42.16 → 41.52 µs; B16/K8320: 166.38 → 167.44 µs | Mixed benefit; not adopted |
| Normalized FP16 split buffers | B4/K8320: 84.52 → 89.42 µs | Rejected |
| Unnormalized FP16 split buffers with added rounding/overflow guards | B16/K8320: 167.54 → 167.72 µs | No stable benefit |
| Existing separate combine/repair fallback | B16/K640: 42.70 → 54.24 µs | Rejected |
| Shared query-L1 prepass and async Q copy | B16/K8320: 140.68 → 141.82 µs | Rejected; extra launch offsets preparation savings |

Private and stock FlagTree each compiled the control and packed large-query
bound kernels for TopK 128, 640 and 8320 without exposing any GPU: 12 successful
compilation checks. These checks establish compilation only.

A separate whole-query FP32 diagnostic examined single-term FP8 PV on 32
frozen rows. Maximum observed absolute errors were approximately 0.00096 for
TopK 640 and 0.00073 for TopK 8320. However, the quantization-only range bound
was 0.00954–0.01080, exceeding the existing 0.0038 guard. Small observed errors
therefore do not justify removing the second PV term. A Cauchy–Schwarz bound
was 0.00262–0.00303, but excludes QK, accumulation and denominator errors and
requires KV column norms. Its complete numerical certificate and the cost of
computing those norms remain unverified. No single-term kernel is adopted.

## Additional diagnostics, 2026-11-11

The following results are excluded from the retained runtime change:

- Conditional second-PV execution passed frozen comparisons but raised spills
  to 116/172 and substantially slowed both tested TopK values.
- Alternative QK layouts required 229,376 shared-memory bytes, exceeding the
  196,608-byte limit. A 64-token variant fit but slowed complete calls.
- Moving a shared common PV into the producer, or storing it between pages,
  still increased spills and slowed complete calls. Removing residual PV
  entirely gave no medium-TopK benefit and only about 4% at long TopK before
  the missing omission certificate was added; it was not adopted.
- Early next-page KV copy, polynomial-only compilation, four-token-lane copy
  and normalized selected key scales had no stable benefit across their
  intended cases. The polynomial-only variant would also lose the native-exp
  fast path for some inputs. These variants were not adopted.

A report-only per-element snapshot matched every KV square on all rows/pages
of TopK 640 and 8320 over three replays each. The original cross-warp column
norm reduction still produced occasional wrong values. Moving its token-axis
reduction within a warp passed the sampled FP64 norm and inflated upper-bound
checks. Combining four adjacent columns reduced overhead, but it remained
slower than the original operator. The cross-warp root cause is unproven, and
none of this norm work is used by the retained kernel.

A separate scalar-reduction compiler reproducer identified layout inference
being applied to scalar results. A fix in an owned experimental source tree
compiled and passed four device patterns; three control executable sections
were byte-identical before and after it. The actual FlagTree checkout, private
runtime and stock installation were not changed. That isolated fix is not a
dependency or part of this PR, and establishes no operator speedup.

## Evidence and reproduction

The development workspace retains raw logs, source prototypes, before/after IR,
exit records and result JSON files under:

- `reports/mthreads-pr17-copy8-delivery-20261111`: current validated source
  manifest, four full-matrix JSON files, precision summary, private/stock core
  JSON files, mutation/tail/fallback logs and delivery comparison.
- `reports/mthreads-pr17-publish-20261109`: previous validated runtime and full
  matrix used for the before/after comparison.
- `reports/mthreads-pr17-centered-20261109`,
  `reports/mthreads-pr17-consumer8-20261109`,
  `reports/mthreads-pr17-burst-20261109`,
  `reports/mthreads-pr17-fixed-scale-20261109`: rejected data-flow experiments.
- `reports/mthreads-pr17-short-ws-20261109`,
  `reports/mthreads-pr17-short-prefetch-20261109`: rejected short-query pipelines.
- `reports/mthreads-pr17-packed-qnorm-20261109`: primitive checks, rounding proof
  and pending packed-reduction candidates.
- `reports/mthreads-pr17-resume-20261110`: resumed paired trials, CPU compilation
  matrix, single-term PV bound diagnostics and rejected source variants.
- `reports/mthreads-pr17-large-prefetch-20261111`: retained copy-layout paired
  evidence and rejected copy/exponent schedules.
- `reports/mthreads-pr17-norm-barrier-20261111`,
  `reports/mthreads-pr17-producer-common-20261111`,
  `reports/mthreads-pr17-common-only-20261111`,
  `reports/mthreads-pr17-keyscale-20261111`: additional rejected diagnostics.
- `reports/mthreads-pr17-scalar-layout-20261111`: isolated compiler reproducer,
  patch, binary manifest and before/after controls.

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
