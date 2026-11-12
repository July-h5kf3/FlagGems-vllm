<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# MThreads sparse FP8 MLA optimization progress

Optimization continues at the user's request. The weighted target is **1.3×**
and has **not** been achieved. The latest validated change moves long-TopK KV
prefetch into the PV consumer, building on runtime `2fca81f5` and documentation
checkpoint `554ccba5`.

## Current consumer-owned prefetch, 2026-11-12

For TopK>1024 with double probability buffering and the existing certified
compiler capability, the PV consumer now copies the next KV page while the
producer computes current QK and probabilities. The consumer acquires the
existing empty slot, completes the original bounded copy, then publishes a
copy-ready barrier. The producer waits that barrier before QK. Empty-slot
release still follows both PV waits, protecting KV and probability storage.
K640, short-query kernels and the single-buffer fallback retain their previous
copy schedule. No workspace, numerical arithmetic, threshold or compiler
binary changes are required.

| Measurement | Result |
| --- | ---: |
| Previous full weighted speedup | 1.11197191× |
| Current full weighted speedup, 414 shapes / 13,920 calls | **1.19331369×** |
| Full speedup with previous unaffected timings | 1.19238015× |
| Changed group | 133 shapes / 4,045 calls |
| Changed-group weighted speedup, before → after | 1.24047471× → 1.51718242× |

Two independent cards repeated seven alternating complete prepared-call pairs
on frozen B512/K8320 inputs:

| GPU | Eager before → candidate | Graph before → candidate |
| --- | --- | --- |
| 0 | 2.75810 → 2.30252 ms | 2.75952 → 2.26469 ms |
| 2 | 2.70678 → 2.25238 ms | 2.73669 → 2.25092 ms |

The main kernel retains 120 registers, 32 total warps and 196,608 shared bytes;
spills increase from zero to 18. Retention is based on complete-call gains.
A split-copy variant reduces spills only to 17 without a stable additional
gain, and is excluded. K640 regressed under consumer prefetch, so the retained
scope preserves its original executable code.

All 414 final-source cases pass the original all-row FP32 comparison, output
`atol=0.002` and LSE `atol=0.0002`. Maximum absolute errors remain
0.000268153846 and 0.000001907349. Private and stock FlagTree each pass 179
functional cases and the core benchmark, with four complete core JSON records
per environment. The actual imported vLLM remains 0.24.0+empty and MATE 0.2.4.
Full-matrix performance is measured on the existing private compiler.

Final-source checks also pass 72 repeated tail/content/scale/index eager/graph
comparisons, eight double/single-buffer comparisons, 26 even-page comparisons
and four optional-workspace fallbacks. These cover K8193 tails, K1153/K1536
even page counts, selected-only and full-cache metadata, mixed unsafe rows,
tiny scales, empty selections and raw-index fallback.

Seven executable-section comparisons pass: five unaffected configurations
match the previous runtime, and changed K8320/K8193 configurations match the
measured prototype. The 840 tracked runtime/test/other files match the tested
snapshot; the progress document is maintained separately. Stock public-path
validation passes; forced unsupported stock TLE configurations are not claimed
as executable-equality successes.

Installed MATE source and generated code were captured without changing the
baseline. The actual MODEL1 scheduled path uses BM64/BN64, 640 main threads,
BF16 QK/PV operands, FP32 accumulation and a separate combine kernel. This
evidence guided the copy-role experiment. Rejected separate-repair testing
also showed that eliminating cold-path spills alone did not improve latency.

Evidence: `reports/mthreads-pr17-consumer-delivery-20261112`,
`reports/mthreads-pr17-consumer-copy-20261112`,
`reports/mthreads-pr17-baseline-study-20261112` and
`reports/mthreads-pr17-unfused-single-20261112`.

## K=640 short-query schedule, 2026-11-12

For B16/K640, two 32-head CTAs per query now process all five pages in one
split, using 16 warps and the existing in-CTA FP32 repair. This removes the
combine launch. For B1–4/K640, the existing page splits remain and each CTA
uses 16 rather than eight warps. The frozen matrix contains B4 and B16 in
these changed routes: two shapes and 2,402 calls.

| Measurement | Result |
| --- | ---: |
| Previous full weighted speedup (`d1406fde`) | 1.09026451× |
| Final-source full weighted speedup, 414 shapes / 13,920 calls | **1.11197191×** |
| Full speedup with previous unaffected timings | 1.11049765× |
| Changed K640 shapes, before → after | 0.86164332× → 0.97889783× |
| First candidate full measurement, before host cleanup | 1.11438940× |

Seven alternating complete prepared-call pairs on frozen inputs measured:

| Shape | Eager before → candidate | Graph before → candidate |
| --- | --- | --- |
| B16/K640 | 42.32 → 37.20 µs | 46.80 → 42.70 µs |
| B4/K640 | 35.36 → 31.52 µs | 33.23 → 30.83 µs |

The B16 main kernel uses 240 registers, 418 spills and 98,816 shared-memory
bytes. The complete call improves despite the spills. The B4 main kernel
uses 89 registers, zero spills and 91,648 shared-memory bytes. Neither result
is inferred from an isolated kernel timing.

All JIT kernels, FP8 QK, two-term FP8 PV, numerical certificates, thresholds
and repair arithmetic are unchanged. The single-split repair route is gated
by persistent-path support; disabling that compiler capability retains the
existing flagged repair. BF16 and FP16 checks exercise this route with actual
ineligible rows. Host cleanup preserved the scheduling plan on 1,260 metadata
combinations and every JIT AST.

The final source passed all 414 original all-row FP32 comparisons, with output
`atol=0.002` and LSE `atol=0.0002`. Maximum absolute errors were
0.000268153846 and 0.000001907349. Private and stock FlagTree each passed 179
functional cases and the core benchmark; each core JSON has four complete
latency/speedup records. The full performance matrix uses the existing private
compiler; no stock full-matrix speedup is claimed. The vLLM empty environment,
MATE baseline, compiler binaries, fixtures, weights and tolerances are unchanged.

The first final-source repeat reached the runner's eight-minute deadline with
partial records and is excluded from aggregation. Six new disjoint shards
completed successfully with the same benchmark parameters and final source.
Their source hashes and complete shape coverage were checked before aggregation.
The runtime and tests match 840 tracked files in the tested snapshot; the
progress document is maintained separately.

Further candidates were rejected:

| Candidate | Representative complete-call eager timing, before → candidate | Result |
| --- | --- | --- |
| Force inline repair | B16/K640: 37.40 → 38.52 µs | Spills increase from 418 to 460 |
| Ordinary masked loads and Triton dot | B16/K640: 37.60 → 51.36 µs; B16/K8320: 166.94 → 212.40 µs | Slower on all four tested shapes |
| Two-stage ordinary-load loop | B16/K640: 39.50 → 51.56 µs; B16/K8320: 168.36 → 224.00 µs | Slower on all four tested shapes |
| 16 heads with 16 warps | No device launch | Compiler cannot select a supported TLE SQMMA configuration |

The ordinary-load candidates passed the frozen comparisons and actual
fast-path flag checks. These diagnostics are excluded from production.

## Follow-up deep optimization, 2026-11-12

Eleven further executable variants completed 23 paired scenarios on the
existing private runtime. Each scenario used seven alternating complete-call
eager and graph pairs, frozen inputs, the original strict output/LSE checks
and actual fast-path flags. None provided a stable improvement for adoption.
The retained runtime and its 1.11197191× full-matrix result are unchanged.

| Candidate | Representative eager timing, before → candidate | Result |
| --- | --- | --- |
| Native exponential in short persistent queries | B16/K8320: 167.56 → 169.34 µs | Mixed results across three shapes |
| More token lanes with column warp groups | B512/K8320: 2757.34 → 2793.22 µs and 2707.42 → 3058.80 µs | Both layouts regress; spills increase to two |
| BN256 with one KV slot and matched producer layout | B512/K640: 349.52 → 638.88 µs | 69 spills |
| BN256 with compact denominator partials | B512/K640: 351.38 → 617.04 µs | 58 spills; still slower |
| Per-query large-batch persistent dispatch | B512/K8320: 2739.62 → 4320.12 µs | Fewer launches do not offset the main-kernel cost |
| Two 32-head CTAs per query with uncached metadata | B512/K8320: 2788.68 → 6457.92 µs | Lower resource use does not improve complete calls |
| Ordinary async copy with a zero buffer | B512/K8320: 2740.04 → 3049.70 µs | 12 spills |
| Precomputed 64-bit source pointers | B512/K8320: 2756.18 → 2971.00 µs | Eight spills; a higher global register cap does not remove them |
| Native masked copy with shared zero stores | B512/K8320: 2710.36 → 2797.50 µs | Zero spills, but about 3% slower |

The wide-tile investigation identified an extra 64 KiB QK layout conversion.
Matching the producer layout removed it and restored the original 192 KiB
resource limit. Lowered role code retained full-CTA barriers only in the
prologue. No compiler or hardware limit was changed.

Zero-buffer, pointer-staging and native masked-copy checks covered 60 tiles
and 3,932,160 bytes exactly, including all byte encodings, mixed valid/invalid
indices and both index representations. The first two also checked their
per-call zero initialization and metadata paths. These checks establish copy
correctness; the variants were rejected on complete-call performance.

A uniform-numerator diagnostic passed no head under its conservative guard:
the combined bounds were 0.00885–0.00987, above the unchanged 0.0038 limit.
A high-level masked async-load/store candidate compiled but generated no
direct async-copy operations in the installed compiler, and was excluded
before device testing. Neither result permits dropping numerical checks.

Raw sources, exit records, IR and timings are retained under
`reports/mthreads-pr17-deepopt-20261112` and its referenced report directories.
The subset trials do not establish stock-runtime or full-matrix acceptance
for any rejected candidate. Hardware counters were not collected in this
batch; compiled resources, lowered code and device event timings supplied
the profiling evidence. The 1.3× target remains unmet.

## Earlier native-exponential implementation

The baseline is native MATE 0.2.4 MODEL1 D512 on MTT S5000, with BF16 QK/PV
computation and FP8 KV-cache storage. The environment uses vLLM 0.24.0 empty.
The frozen matrix contains 414 shapes and 13,920 original calls. Speedups are
weighted by each shape's call count; the baseline, inputs and weights remain
unchanged.

| Measurement | Result |
| --- | ---: |
| Previous full weighted speedup (`45f58771`) | 1.07757864× |
| Native-exponential full weighted speedup (`d1406fde`) | 1.09026451× |
| Previous native-long result with previous unaffected timings | 1.08606611× |
| Previous long-TopK group | 133 shapes / 4,045 calls |
| Previous long-TopK speedup, before → after | 1.20944285× → 1.23763440× |
| New large K=640 group | 133 shapes / 4,253 calls |
| New K=640 speedup, before → after | 1.17237381× → 1.18580684× |

The retained copy-layout change uses eight token lanes per warp for every TopK greater than
128, extending the existing medium-TopK copy layout to long TopK. The producer
row-warp count stays at 16. At the 128-token tile size, these cover 128 distinct
token rows without duplicate copies. The long-TopK main kernel's spills drop
from 13 to zero; its 32 total warps and 196,608 shared-memory bytes are unchanged.

The previous producer-barrier removal and double probability buffering remain.
That update uses the bounded native exponential in the large producer
for K=640 and TopK>1024. Short queries and the medium short path retain their
existing polynomial. The existing native path already uses the same FP32 LLVM exp2 intrinsic. Valid
contributing half-scores lie in [-1, 1], so its base-2 arguments stay within
approximately ±2.8854 and never need expf's subnormal-output workaround.

A primitive check covered 195,590 uniform, finite raw-bit and boundary inputs;
results were bitwise equal to the existing native exponential and had maximum
relative error 1.93e-7 against CPU FP64. The original conservative polynomial
budget is retained. FP8 QK, two-term FP8 PV, all eligibility checks, numerical
thresholds, scale handling and whole-row repair are unchanged. No fast-math
flag or compiler change is required.

The earlier seven executable-section comparisons passed: three unaffected configurations
(short TopK 640/8320 and large TopK 128) match the previous runtime, and
three scoped long-TopK configurations match the measured prototype, including
single buffering and the TopK 8193 tail. The current main kernel still uses 120 registers, zero
spills, 32 total warps and 196,608 shared-memory bytes.

All 414 shapes passed the original all-row FP32 comparison with output
`atol=0.002` and LSE `atol=0.0002`. Maximum absolute errors were
0.000268153846 and 0.000001907349, respectively. Private and stock FlagTree each
passed 179 functional cases and the core benchmark, with four complete latency
and speedup records per core JSON. The current candidate passed 432 additional
eager/graph comparisons, 12 optional workspace fallbacks and eight compile/fallback
comparisons. Full pre-commit,
54 CI helper tests, CI pins and diff checks passed.

Two independent cards repeated seven alternating complete-call timing pairs
on the frozen B512/K8320 input:

| GPU | Eager before → candidate | Graph before → candidate |
| --- | --- | --- |
| 4 | 2.78416 → 2.71810 ms | 2.82393 → 2.75449 ms |
| 7 | 2.77632 → 2.71030 ms | 2.79833 → 2.72337 ms |

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

A mixed 3/4-fragment short-query plan passed 79,518 coverage/capacity checks
and the frozen B16/K640 oracle. It used 60 rather than 48 CTAs with the same
maximum two pages per fragment, but did not speed up the main kernel and
increased combine time. It was rejected.

Removing the explicit end-of-page copy barrier was also rejected. The private
compiler retained an implicit input barrier only in double-buffered kernels;
the single-buffered versions lacked that edge and were not device-tested.
The verified double-buffer diagnostic passed frozen comparisons but showed no
speedup. The accepted runtime keeps the original explicit copy synchronization
in both buffering modes.

## Evidence and reproduction

The development workspace retains raw logs, source prototypes, before/after IR,
exit records and result JSON files under:

- `reports/mthreads-pr17-headsplit-delivery-20261112`: final source manifest,
  six complete matrix JSON files, strict precision records, before/after
  aggregation, private/stock validation and rejected prototype logs.
- `reports/mthreads-pr17-headsplit-w16-20261111`,
  `reports/mthreads-pr17-b4h32w16-20261111`: retained scheduling pairs and
  BF16/FP16 mutation, repair and compile-fallback checks.
- `reports/mthreads-pr17-regular-pages-20261112`,
  `reports/mthreads-pr17-head16-w16-20261112`: rejected ordinary-load and
  head-count diagnostics.
- `reports/mthreads-pr17-native-delivery-20261111`: previous validated source
  manifest, four full-matrix JSON files, precision summary, private/stock core
  JSON files, mutation/tail/fallback logs and delivery comparison.
- `reports/mthreads-pr17-native-exp-20261111`: primitive equivalence, executable
  scope checks and two-card prepared-call pairs for the native exponential.
- `reports/mthreads-pr17-copy8-delivery-20261111`: previous validated copy-layout
  runtime and full matrix used for the current comparison.
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
- `reports/mthreads-pr17-balanced-20261111`,
  `reports/mthreads-pr17-copy-sync-20261111`: rejected fragment/synchronization
  diagnostics; neither is included in the current runtime.

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
