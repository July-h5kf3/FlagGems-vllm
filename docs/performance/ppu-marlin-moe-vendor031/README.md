# PPU quantized fused Marlin MoE: vendor vLLM 0.31 checkpoint

This is a review checkpoint for the T-Head implementation added by [upstream PR #752](https://github.com/flagos-ai/FlagGems-vllm/pull/752). The [F225 W8 checkpoint](f225-w8-checkpoint.md) gives the current W8 results and validation status. The W8 tables below record earlier source revisions. **INT4 and MXFP4 are frozen at the user-approved native W4A16 checkpoints. INT8 meets the count-weighted 1.3x target on its valid shapes. FP8 remains below 1.3x.** The tables retain the previous native failures at INT8 M48, M56 and M64 and FP8 M64. Any additional failed rows also appear in the tables.

The [FP8 expert-window checkpoint](nov20-fp8-expert-windows.md) records the current 53-shape comparison against SAIL 0.33.0rc1 with the released DeepGEMM BF16 MoE. The [previous large-batch checkpoint](nov20-fp8-overlap.md) records the initial two-bank implementation.

## Part 1 — Operator implementation and optimization

Review status: draft. Performance comparisons use the latest native version supported on the tested PPU.

The public `fused_marlin_moe` interface and vendor registration are retained. Changes are confined to the T-Head backend and the existing functional test file. The NVIDIA implementation and tuning configuration are unchanged.

For the grouped BF16 fast path, the operator aligns flattened top-k routes by expert, gathers activations, runs the gate/up projection, fuses SiLU with gathering the down-projection input, runs the down projection, and combines the routed outputs. Input and output aliases retain the original validation and staging order.

Weights remain in their public output-major byte storage. UINT8B128 uses an offset of 128; UINT4B8 uses two offset-8 nibbles per byte; FP8 uses E4M3; MXFP4 uses E2M1 with E8M0 group scales. Every invocation reads the current weights and scales. No persistent decoded floating-point weight bank or data-dependent weight/scale cache is added.

| Path | Eligibility and parameters | Purpose |
| --- | --- | --- |
| Raw W8 BF16 | Contiguous weights, BF16 scales, group 128, K/N multiples of 128, routes <=16E; N tile 128, K tile 128, eight warps | Load original bytes into shared memory and decode register fragments directly for BF16 MMA |
| W8 eight-row / sixteen-row | Eight rows for routes <=6E, otherwise sixteen | Reduce padded work for sparse experts while retaining the larger tile at higher density |
| Raw W4 BF16 | Same K/N alignment and contiguous weights; INT4 group128/BF16 scales or MXFP4 group32; routes <=16E; N tile64, K tile128, four warps | Reuse packed bytes without the former INT32 expansion/repacking in the GEMM loop |
| W4 eight-row / sixteen-row | Eight rows for routes <=4E, otherwise sixteen | Use the smaller tile only where complete-call measurements show a benefit |
| Very sparse INT4/MXFP4/INT8 | Reuse the raw matrix path when eligible and its conservative workspace estimate fits 2GiB | Improve cases previously assigned to the direct route; keep FP8's direct-route selection |
| Large BF16 batches | Reuse the existing bounded decode-bank + BF16 grouped-GEMM implementation; gate/up columns256, down columns4096 for W8 (1024 for W4) | Decode each consumed column slice per invocation and reuse scratch, with a 1GiB decoded-bank and 2GiB temporary-workspace bound |

For eligible raw INT8/INT4/MXFP4 calls with at most two tokens, topk<=6 and M*topk<=E<=256, reuse the existing single-route alignment for M=1 and short grouped alignment for M=2. The latter retains expert sharing across tokens. Larger batches and FP8 retain their existing routing. Repeated expert IDs remain valid; no host read of routing data or new GPU arithmetic is introduced.

Eligible raw INT8 calls with at most32 routes now use w8_small_routes_kernel after the existing M=1/2 shortcuts. It computes pairwise expert matches, per-route ranks and padded expert offsets in one CTA; it preserves route order within each expert and groups experts by first appearance. This replaces the two zero-initialization and four alignment launches. The comparison matrix is bounded by32x32; larger calls retain grouped alignment. Sorted storage keeps the existing min(R*BM,R+E*(BM-1)) bound, and the GEMMs consume the matching expert metadata. That INT8 alignment change leaves W4/FP8 dispatch and dequantization unchanged; the later W8 updates below are recorded separately.

The eight-row W8 path now uses `_load_w8_activation_x2` to load an8x16 BF16 activation fragment directly from the existing AIU shared-memory swizzle. Each lane receives the same two packed words previously obtained from two scalar shared loads and per-lane address calculations. The256-byte activation row remains split into two128-byte planes; the second plane uses base+1024 and the channel coordinate stays within0..96. This changes only the fragment load: routing, scale checks, exact BF16 weight decoding, MMA accumulation, shared-buffer sizes and the W4 paths retain their prior code.

The raw MMA treats the weight fragment as N-by-K and the activation fragment as K-by-routed-rows; stores restore route-major output. Activations are permuted within groups of 16 K elements for W8 and 32 for W4. The same self-inverse permutation is used in the gather and fused SiLU/gather stages. It remains inside the original scale group and adds no separate global weight-packing pass.

Exact packed BF16 decoding is retained. The INT8 fragment decoder removes a dependent sign XOR and constructs the subtraction bias with LOP3 before the unchanged BF16 scale multiplication. W4 removes redundant low-16-bit masks before register decoding. MXFP4 combines four adjacent scale-byte loads only when group stride is one, row/expert strides are multiples of four, and the pointer is four-byte aligned; other layouts retain scalar reads. FP8 folding is selected using freshly computed per-expert scale-safety flags, with the robust whole-loop implementation retained for unsafe scales. Finite values, signed zero and Inf retain the tested bit semantics; NaN payloads are unspecified and classes are checked.

The sparse route uses a conservative bound for alignment, both staged activations and projection storage. With `R=M*topk`, padded rows are bounded by `R+7E` in the previously-direct band. The estimate is `4(E+1)^2 + 16(R+7E) + 2(R+7E)(K+N) + 2R(K+2N) + 8` bytes. Calls above the cap retain their existing direct path. Inputs and caller-visible output are outside the temporary budget. Existing peak-memory regressions cover 32768 experts for INT4 and INT8, including approximately0.8/1.6GB input storage.

FP16, unaligned/general geometries, the original direct implementation and bounded dense fallback remain available under their existing conditions. Unsupported compatibility arguments still fail explicitly. Production computation uses Triton/TLE and the existing FlagGems operators; no Torch compute fallback or vendor-native replacement of the candidate is added.

### TLE features

| API | Location | Role |
| --- | --- | --- |
| `tle_async.gpu.alloc` | Raw narrow/square W8 and W4 kernels | Allocate power-of-two shared Q/A double buffers |
| `tle_async.gpu.local_ptr` | The same kernels | Obtain shared addresses consumed by PPU copies and fragment loads |
| `tle_async.load` | `dense_w8_bf16_kernel` | Asynchronously load the decoded BF16 bank while activations retain per-tile routing |

NVIDIA autotuning is unchanged. These PPU paths use measured fixed schedules and the existing dispatch conditions; no new FlagTree custom operation or compiler patch is introduced.

### Inline assembly

- `ppu.cp.async.aiu.bulk.tensor...2d.b8`, commit/wait and CTA barriers transfer raw byte tiles and staged activations and protect buffer reuse. Padding is explicitly initialized by the current staging implementation.
- `_load_w8_activation_x2` uses `ppu.ldmatrix.sync.aligned.m8n8.x2.swzl.shared.b16` for the W8 eight-row activation fragment, reducing scalar shared address/load instructions. `ppu.ldmatrix...x4.swzl` retains the existing weight and sixteen-row fragment loads. BF16 `m16n8k16` and `m16n16k16` MMA accumulate in FP32.
- `prmt`, `lop3`, packed BF16 subtraction and `mul.rn.bf16x2` implement exact quantized fragment decoding and scale multiplication. The eight-row ISA path avoids relying on an unsupported eight-row `tl.dot` lowering in the installed backend.

Earlier matrix-x2 checkpoint (`52a2ed3b`) raw-kernel resources: W8 INT8 eight-row gate/down use62/62 registers; FP8 uses54/52. These eight-row kernels retain36KiB shared memory, compared with70/70 INT8 and68/66 FP8 registers before the change. The sixteen-row W8 kernels are unchanged (INT8 74/74 registers,40KiB shared). INT4 uses60/58 registers for both row sizes; MXFP4 uses60/60 for eight rows and60/56 for sixteen. W4 shared memory is12/16KiB. These resource records report no spills. The W4 records precede the final host-only workspace predicate and retain identical GPU kernel ASTs.

### W8 dense projection update

The large-batch W8 path now gathers activations per GEMM tile and asynchronously loads its decoded BF16 weight bank with TLE. It retains the64x256x32 tile, four warps, four stages, per-call decoding and BF16 rounding before route reduction. It adds no expanded sorted activation matrix or global workspace. INT4/MXFP4 keep the original GEMM invocation.

Complete-call M16384 A/B against the previous source reduced INT8 latency8.55179→7.94444ms and FP8 latency8.65585→8.04178ms, about7.1%. These local gains are separate from the fresh native weighted results below. Both BN128 variants passed checks but were slower and were rejected. Source pins, all105 fresh valid native rows, functional results, memory checks and archive hashes are in [w8_dense_update.json](w8_dense_update.json).

### W8 raw projection and metadata update

The W8 double-buffer prefetch now issues both independent activation-plane copies before the weight copy, retaining the original K128 geometry, buffers, commit/wait and CTA barriers. The sixteen-row raw kernel checks the ninth sorted ID: when only the first eight rows are valid, it reuses the existing eight-row accumulator and matrix loader. Fully populated tiles retain the square MMA path. Alignment and host route selection are unchanged.

Raw BF16/group128 FP8 calls in the compact safety range now initialize both expert safety arrays and both existing alignment workspaces in one `fp8_safety_and_alignment_zero` kernel, replacing four launches. It calls the original compact classifiers and writes the same workspace zeros. Flags are allocated during weight preparation and initialized by the alignment helper before either GEMM. The helper retains the original workspace lifetime; dense, direct and general cases retain their previous checks. No decoded-weight cache, new global workspace or numerical approximation is added.

The combined prototype passed all53 fixed shapes in each W8 format with bit-identical outputs. Count-weighted gains against the previous source were1.011116x INT8 and1.013440x FP8, each count10105; these are A/B gains, distinct from the fresh native ratios below. Mixed-expert/padding, all BF16 scale bit patterns, noncontiguous scales, zeroing canaries and live safe/unsafe scale updates were checked before final-source regression and native testing. The existing tile regression now includes five and nine valid rows in a sixteen-row tile. [w8_raw_update.json](w8_raw_update.json) contains pins, all A/B/native rows, checks and archive hashes.

### Bounded W8 routing update

Eligible raw W8 calls with at most4096 routes and1024 experts now build grouped metadata in one CTA. It fills sorted padding, zeros one counter per expert, uses a CTA barrier, computes a masked histogram and padded prefix, then assigns unique atomic route ranks. One route writes each expert tile ID. Sorted storage retains min(R*BM,R+E*(BM-1)); expert IDs beyond the padded prefix remain unread. Logical route/zero vectors cover at least all physical threads, avoiding replicated side effects for small inputs. Eight warps serve R<=128 and sixteen serve larger routes; compact FP8 safety scans can select sixteen. The earlier tiny INT8 shortcuts remain.

For deferred compact FP8 safety, fp8_safety_and_routes reuses both classifiers in each expert CTA and builds routing only in CTA0. Routing never reads another CTA's flags. The later GEMMs consume both arrays after kernel completion on the same stream; no grid barrier is added. This replaces the previous joint safety/zero launch plus four alignment stages with one launch and O(E) counter scratch. Larger/general calls retain the old alignment, and W4 dispatch and GPU kernels remain unchanged. No new TLE feature, inline assembly or weight arithmetic is introduced.

All53 shapes in each W8 format matched the previous source bit for bit. Count-weighted A/B gains were1.010367x INT8 and1.008721x FP8, each count10105. These gains are separate from the fresh native results below. All-last-expert routing was also checked: at4096 routes its helper regresses by0.436us, while smaller measured sizes improve. Observed route kernels use32–60 registers with zero spills. The existing whole-call update regression now covers4092 and4098 routes, on either side of the new limit. [w8_routes_update.json](w8_routes_update.json) records source pins, full A/B/native rows, safety checks, resource records and archive hashes.

### W8 scale-load and dense-bank update

The raw eight-/sixteen-row W8 loops now issue their two readonly BF16 scale loads before the current tile prefetch/wait. Casts, scale folding and MMA arithmetic remain after the wait. The K128 double buffers, copy extents, commit/wait operations and CTA barriers retain their previous ordering. This exposes independent scale reads earlier without caching weights or scales across calls. Loading the next iteration's scales was also correct but slower at every representative point, so it is not used.

In representative K4096/N128 unit shaders with routed-weight multiplication disabled, narrow INT8/FP8 register counts change62→38/54→34 and sixteen-row counts74→46/64→42, with zero spills. Shared buffers remain36/40KiB; these observations do not describe every shape.

The two FP8 safety classifiers now share is_fp8_scale_fold_overflow. For BF16, a finite scale overflows a2**120 fold exactly when its sign-cleared bits are in[0x4380,0x7f80). Unsigned(magnitude-0x4380)<0x3c00 implements that interval; Inf/NaN and signedzero retain the old classification. Other scale dtypes retain the original floating-point computation. All65536 BF16 patterns matched the old predicate on CPU and GPU, with30720 unsafe values. The compact classifier's offline shader has410→299 non-NOP instructions and60→38 vector registers, without stack storage. A one-role compact microbench improves5.506→4.600us; this is separate from complete-call/native measurements.

The dense W8 down-projection bank expands from1024 to4096 output columns, while gate/up banks stay256. The matrix tile remains64x256x32 with four warps and four stages. The same keyword tile parameters feed both the workspace budget and dense execution. For the fixed K4096 shape, the down projection processes its output columns in one pass instead of four; each pass performs the existing decode, GEMM and route reduction, while retaining the1GiB decoded-bank and2GiB temporary-workspace caps. It increases projection storage: final-source M16384 peaks including the128MiB output are1,527,456,768 bytes for INT8 and1,527,459,840 bytes for FP8, below the2GiB+output limit of2,281,701,376 bytes. W4 retains its previous256/1024 choices. The existing projection regression now uses K4224 for W8 to exercise a4096+128 tail; W4 test inputs retain K2176 and both use a full-column numerical reference.

All53 fixed shapes in each W8 format match the prior source bit for bit. Count-weighted A/B gains are1.032150x INT8 and1.020638x FP8, each count10105. These gains are distinct from the fresh native ratios below. The merged source also passed320 exact/FP32 tiles, full scale-bit checks, dynamic input/scale updates and real-memory bounds. No new TLE feature, inline assembly, weight approximation or persistent input-dependent cache is introduced. [w8_load_update.json](w8_load_update.json) contains all pins, rows, resources and evidence hashes.

### Format-specific W8 fragment scheduling

FP8 now uses paired_fp8_folded/paired_fp8_robust to decode two independent byte-packed qwords in one inline-assembly block. It interleaves the same supported32-bit AND/add/PRMT/LOP3/shift/packed-BF16-multiply operations. Each component retains its original normalization, scale multiplication and BF16 RNE order. The natural output order is q0-low/q0-high/q8-low/q8-high: the eight-row caller maps these to a0/a2/a1/a3, while the sixteen-row caller maps a0/a1/a2/a3. No new conversion instruction, quantization approximation or weight cache is introduced.

INT8 retains its existing decoder and reads each activation fragment before weight-fragment decoding, after the existing tile wait and weight-fragment read. FP8 keeps the activation read after decoding: the earlier-read/pairing combination was correct but slower than pairing alone. Copy extents, K128 buffers, commit/wait operations, CTA barriers, MMA order, route selection and dense-bank configuration retain their previous values. W4 kernels and parameter selection are unchanged. This adds no TLE feature; the new inline assembly only schedules existing b32 instructions across independent words.

Independent CPU exact-product/BF16-RNE tables and GPU tests cover all256 codes x65536 BF16 scale bit patterns x8 nonperiodic byte patterns, with independently permuted scale streams. INT8 and robust FP8 cover every scale; folded FP8 is checked against RNE only in its existing safe domain. Finite values, signedzero and Inf match bit for bit; NaN payloads remain unspecified and classes match. The merged source also passed320 exact/FP32 tiles,24 whole calls and8 live-update graph cases, before the registered-source regression and native measurements.

The preceding decoder scheduling update matched its predecessor bit for bit across all 53 fixed shapes in each W8 format. Its count-weighted A/B gains were 1.004987x INT8 and 1.013990x FP8, each with count 10105. These gains are separate from the fresh native ratios below. INT8 pairing, x4 activation fusion and five dense-pipeline variants were rejected after correct outputs but slower complete calls. [w8_pair_update.json](w8_pair_update.json) records pins, full rows, primitive reference digests, memory limits and archive hashes.

### Split K for one INT8 token

For a BF16 INT8 call with one token, the existing raw path may launch fewer than 32 gate/up CTAs. On the compatible K multiple of 1024, K-pitch and N ranges, dispatch reuses the existing split-K matvec and fused partial-reduction/SiLU kernels. Each part decodes the original byte weights and BF16 scales. The split results are reduced before the existing gate/up BF16 rounding. Down projection and route combine use the existing direct path. The workspace remains bounded by the eligibility conditions.

In count-weighted A/B against the previous source, the 53 INT8 shapes show 1.002076x (count 10105). M1 latency changes from 34.12 to 30.42 microseconds. M1 changes FP32 accumulation order and passes the existing accuracy limits. Its candidate FP32 peak error is 0.407%, and all other shapes match the previous output in the measured checks. The measured A/B gain refers to the prior implementation. The native BF16 comparison is reported separately. [w8_tiny_update.json](w8_tiny_update.json)

### Wide W8 weight copies and sparse FP8 scale checks

The 16-row raw W8 path copies each 128-column weight block as two 64-row blocks at the same K tile size of 128 bytes. Each four-warp group consumes its own 8 KiB shared-memory block through the corresponding matrix load, including the underfilled tail of a 16-row block. The separate 8-row path keeps the original 128-row copy. The selected copy forms use the same double buffers, BF16 conversion and MMA order.

When there are at most half as many routes as experts, the compact FP8 scale check launches by route. The first occurrence of each routed expert owns that expert's two flags. The alignment CTA still emits the expert tiles, and only those tiles read the flags after the launch. Eligible sparse FP8 calls that can already fill a 64-CTA grouped projection use this path for both projections.

The 53-shape count-weighted A/B gains against the preceding source are 1.004908x INT8 and 1.008149x FP8 at count 10105 each. The small FP8 M4 call drops from 105.32 to 84.43 microseconds. Its different accumulation order passes the original FP32 bounds with 0.387% RMS and 0.458% peak error. The remaining FP8 shapes and all INT8 shapes match the previous output bits. [w8_tile_routes_update.json](w8_tile_routes_update.json)

### Skip inactive raw W8 staging tiles

The activation gather and SiLU staging kernels now return when an entire raw W8 staging block starts at or above the padded route count. The projection GEMMs already skip these blocks. The raw W8 consumers explicitly opt into the guard. Other callers still receive zero-filled inactive rows, and active expert tiles keep their original data and zero-filled padding. The guard reads the current padded route count on every invocation, including CUDA graph replay.

The 53-shape A/B of the staging optimization before the explicit opt-in gives 1.016232x for INT8 and 1.012944x for FP8 against the previous implementation, at count 10105 for each. The final opt-in implementation also matches the previous output bits on all 53 shapes in each format and passes all 16 existing staging tests. [w8_stage_trim_update.json](w8_stage_trim_update.json)

### Select the FP8 schedule by route density

Grouped raw FP8 now crosses from 8 rows to 16 at five routes per expert on average. INT8 retains its six-route cutoff. The 16-row FP8 tile amortizes weight movement at the earlier crossover, where more experts occupy both halves. For intermediate batches that use generic grouped FP8, its down projection uses a 128-column tile with four warps, and staging skips rows that its consumer cannot read. In the dense FP8 bank, a CTA selects the existing decoder from its expert flag, so each bank is written once. The folded branch scales once per group before broadcasting; the original robust conversion still handles unsafe scales.

The two-component crossover and dense-bank prototype was bit-exact across all 53 FP8 shapes and gave 1.005075x local count-weighted speedup at count 10105. In the separate generic-stage trial, the 2048-token FP8 call fell from 2.391 to 2.290 ms. The final merged source is bit-exact against the previous checkpoint on all 53 shapes for each format. The official ratios below come from that final source against the latest supported native version. [w8_fp8_schedule_update.json](w8_fp8_schedule_update.json)

### Unroll two FP8 reduction groups

The raw FP8 projections unroll two 128-wide K groups when K is divisible by 256. Each group uses the existing asynchronous copy, safety selection and accumulation order. All 53 original FP8 shapes match the prior implementation bit-for-bit, with 1.004538x count-weighted speedup across count 10105 in paired runs. The entries below are from the merged source against the latest native release. [w8_fp8_unroll_update.json](w8_fp8_unroll_update.json)

## Part 2 — Performance

### Vendor baseline and full-call interface

The baseline is the official [vLLM-SAIL v0.31.0 release](https://github.com/vllm-project/vllm-sail/releases/tag/v0.31.0), published2026-11-13 and confirmed as latest during these measurements, with its [release manifest](https://github.com/vllm-project/vllm-sail/releases/download/v0.31.0/release-manifest.json).

- vLLM `0.31.1.dev0+gdb9527a46.d20261113.empty`, commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- SAIL `0.31.0`, commit `71913f22e21951b52b6ee082f06614d66e68c817`.
- Native DeepGEMM `1.1.0+v1.0.1.ppu2.2.0`, commit `8292bb50504556daceee820991158c759290a38c`, without kernel-source modifications.
- The native expert is `vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe.PPUDeepGemmExperts`, invoked through `vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernel.apply`.

There is no matching810E W8A16/group128 native contract for the tested INT8/FP8 inputs; both use the same-release unquantized BF16 native MoE. Its weights are the corresponding decoded BF16 values, prepared outside the call. INT4/MXFP4 use native W4A16. INT4 group128 scales are repeated exactly to group32 metadata without requantizing weights. Official Marlin helpers prepare static layouts only; the timed native compute is DeepGEMM, not Marlin GEMM.

Native registration and constraints: [INT4](https://github.com/vllm-project/vllm-sail/blob/71913f22e21951b52b6ee082f06614d66e68c817/vllm_sail/registry/moe_backends/int_wna16.py), [MXFP4](https://github.com/vllm-project/vllm-sail/blob/71913f22e21951b52b6ee082f06614d66e68c817/vllm_sail/registry/moe_backends/mxfp4.py), [expert implementation](https://github.com/vllm-project/vllm-sail/blob/71913f22e21951b52b6ee082f06614d66e68c817/vllm_sail/model_executor/layers/fused_moe/experts/deep_gemm_moe.py).

The BF16 factory is selected by `select_unquantized_moe_backend(config)` and constructed by `make_unquantized_moe_kernel(FusedMoEQuantConfig.make(), config, backend, experts)` from `vllm.model_executor.layers.fused_moe.oracle.unquantized`. Both native branches execute the complete interface:

```python
output = kernel.apply(
    hidden_states, w1, w2, topk_weights, topk_ids,
    MoEActivation.SILU, 256, None, False,
)
```

The configuration is E256, topk6, hidden4096, intermediate256, BF16 input, SiLU, no bias, one local rank and expert-parallel maps disabled. Router weighting is on the second projection. The candidate calls the existing public `flaggems_vllm.fused_marlin_moe` with identical quantized tensors/scales and group128, or group32 for MXFP4.

### Workload and measurement

The preserved `DeepSeek-V4-Flash-p32768d1024.txt` capture contains53 unique shapes, each `(M,256,4096,256,6)`, with sum(count)=10105. The historical capture key is `flag_gems.ops.fused_marlin_moe.fused_marlin_moe`. [shape_source.txt](shape_source.txt) contains its exact53 records; [shapes53.json](shapes53.json) is the frozen parsed histogram. [shape_provenance.json](shape_provenance.json) records hashes and the equality check against the original model dump. M covers1–512,2048 and16384. Synthetic tensor values are used. W4 reuses this geometry/count distribution as supplementary coverage, not independently traced W4 model frequencies.

Native and candidate run on the same card in isolated Python environments. They alternate vendor/candidate, candidate/vendor, vendor/candidate. Each round uses a100ms complete CUDA graph; the median of each engine's three measurements forms the ratio. Input creation, static native packing, first JIT/autotuning and graph capture are outside timing. Candidate weight/scale reads, scale checks, routing, both GEMMs, activation and reduction are inside timing. Input and weight/scale hashes must match. Startup idle/UUID checks and continuous context monitoring reject interference and only terminate this task's processes.

W8 weight seed is7 and activation/routing seed is7+M. W4 uses the frozen existing generator's seed0 and the same activation/routing generation. Accuracy uses relative RMS<1% and global relative peak<2%, preserving the existing BF16 rounding contract. Decoder proofs also check finite/zero/Inf bits and NaN classes independently.

`speedup_i = native_ms_i / candidate_ms_i` and `weighted_speedup = sum(count_i * speedup_i) / sum(count_i)` over valid samples. The weighted speedup is not a ratio of summed latencies.

| Precision | Native precision | Weighted speedup | Valid/attempted shapes | Valid count | Valid range | Measured commit |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| INT8 | BF16 | 1.314680x | 50/53 | 9589 | 0.616–1.516x | `6277aac4` |
| FP8 | BF16 | 1.182837x | 52/53 | 9933 | 0.609–1.317x | `6277aac4` |
| INT4 | INT4 | 1.223084x | 53/53 | 10105 | 0.749–1.355x | per-row CSV |
| MXFP4 | MXFP4 | 1.078587x | 53/53 | 10105 | 0.640–1.242x | per-row CSV |

INT8 M56/count172 is invalid: native BF16 peak error against independent FP32 is2.420966%, while the candidate peak is0.564822%; the paired peak is2.372263%. This native failure is preserved and was not resampled. Its count is excluded from both numerator and denominator, formal speedup is blank, and the actual timing ratio is retained only as `diagnostic_speedup` in [performance.csv](performance.csv). [m56-cpu-recheck.json](m56-cpu-recheck.json) independently rechecks the saved three outputs on CPU. Native BF16 repeat instability also appeared in prior vendor-only eager/graph tests; its root cause is unresolved. Current passing samples do not establish native repeat stability.

The original M48, M56 and M64 INT8 diagnostics and M64 FP8 diagnostics are retained without resampling. The M64 native FP32 peak errors were 9.0481% INT8 and 11.2609% FP8, confirmed from the saved outputs on CPU. Weighted speedups include only successful measurements. Details for any fresh native failures are in [w8_tile_routes_update.json](w8_tile_routes_update.json).

### Complete per-shape table

All 212 shape rows, including the invalid native samples, are retained below and in [performance.csv](performance.csv). Fixed shape parameters are E256/K4096/N256/topk6. Group size is128 for INT8/FP8/INT4 and32 for MXFP4. Units are milliseconds; error columns are percentages of the native reference. M56's independent-FP32 diagnosis is stated above.

<details>
<summary>INT8: 53 attempted shapes</summary>

| M | count | Native ms | Candidate ms | Speedup | RMS % | Peak % | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 172 | 0.032429 | 0.030962 | 1.047380x | 0.5702 | 0.6127 | passed |
| 2 | 172 | 0.053907 | 0.053582 | 1.006057x | 0.5891 | 0.6757 | passed |
| 4 | 172 | 0.101975 | 0.071098 | 1.434286x | 0.5771 | 0.4000 | passed |
| 8 | 172 | 0.159488 | 0.136691 | 1.166774x | 0.6050 | 0.6313 | passed |
| 16 | 172 | 0.263813 | 0.230130 | 1.146362x | 0.6111 | 0.9259 | passed |
| 24 | 172 | 0.368484 | 0.308177 | 1.195690x | 0.6066 | 0.7692 | passed |
| 32 | 172 | 0.492756 | 0.342109 | 1.440350x | 0.4693 | 0.5618 | passed |
| 40 | 172 | 0.604900 | 0.417390 | 1.449245x | 0.4689 | 0.3937 | passed |
| 48 | 172 | 0.636929 | 0.469529 | — | 0.4772 | 7.0376 | retained native accuracy failure; excluded |
| 56 | 172 | 0.690912 | 0.524323 | — | 0.4651 | 2.3723 | retained native accuracy failure; excluded |
| 64 | 172 | 0.718038 | 0.506473 | — | 0.4778 | 9.0417 | retained native accuracy failure; excluded |
| 72 | 172 | 0.786428 | 0.538417 | 1.460629x | 0.4328 | 0.5618 | passed |
| 80 | 172 | 0.792929 | 0.540749 | 1.466353x | 0.4330 | 0.3268 | passed |
| 88 | 172 | 0.842650 | 0.568634 | 1.481884x | 0.4383 | 0.5618 | passed |
| 96 | 172 | 0.798058 | 0.549755 | 1.451661x | 0.4391 | 0.7042 | passed |
| 104 | 172 | 0.859574 | 0.585098 | 1.469110x | 0.4391 | 0.7874 | passed |
| 112 | 172 | 0.909608 | 0.602131 | 1.510646x | 0.4296 | 0.6250 | passed |
| 120 | 172 | 0.866027 | 0.591133 | 1.465029x | 0.4393 | 0.6623 | passed |
| 128 | 172 | 0.910767 | 0.600872 | 1.515743x | 0.4389 | 0.5747 | passed |
| 136 | 172 | 0.880907 | 0.598965 | 1.470717x | 0.4300 | 0.6173 | passed |
| 144 | 172 | 0.917715 | 0.617561 | 1.486030x | 0.4381 | 0.6098 | passed |
| 152 | 172 | 0.922357 | 0.618585 | 1.491076x | 0.4316 | 0.5208 | passed |
| 160 | 172 | 0.926572 | 0.630208 | 1.470263x | 0.4342 | 0.5952 | passed |
| 168 | 172 | 0.928822 | 0.631001 | 1.471982x | 0.4354 | 0.6757 | passed |
| 176 | 172 | 0.936305 | 0.642275 | 1.457795x | 0.4338 | 0.6061 | passed |
| 184 | 172 | 0.931862 | 0.654407 | 1.423979x | 0.4371 | 0.6452 | passed |
| 192 | 172 | 0.939192 | 0.656124 | 1.431425x | 0.4366 | 0.6061 | passed |
| 200 | 172 | 0.936131 | 0.663614 | 1.410657x | 0.4350 | 0.5882 | passed |
| 208 | 172 | 0.945934 | 0.668640 | 1.414713x | 0.4333 | 0.7732 | passed |
| 216 | 172 | 0.952575 | 0.680922 | 1.398949x | 0.4375 | 0.6944 | passed |
| 224 | 172 | 0.946411 | 0.680720 | 1.390309x | 0.4379 | 0.6536 | passed |
| 232 | 172 | 0.955040 | 0.700014 | 1.364315x | 0.4350 | 0.6452 | passed |
| 240 | 172 | 0.949707 | 0.709779 | 1.338032x | 0.4339 | 0.6667 | passed |
| 248 | 172 | 0.944644 | 0.722151 | 1.308097x | 0.4355 | 0.7092 | passed |
| 256 | 172 | 0.954474 | 0.710495 | 1.343394x | 0.4316 | 0.6024 | passed |
| 272 | 172 | 0.956798 | 0.701082 | 1.364746x | 0.4348 | 0.6098 | passed |
| 288 | 172 | 0.955697 | 0.702625 | 1.360181x | 0.4376 | 0.4405 | passed |
| 304 | 172 | 0.961890 | 0.705216 | 1.363965x | 0.4366 | 0.5952 | passed |
| 320 | 172 | 0.965187 | 0.709182 | 1.360985x | 0.4372 | 0.6098 | passed |
| 336 | 172 | 0.967194 | 0.713183 | 1.356166x | 0.4329 | 0.4762 | passed |
| 352 | 172 | 0.965726 | 0.720051 | 1.341190x | 0.4401 | 0.5376 | passed |
| 368 | 172 | 0.978647 | 0.727610 | 1.345016x | 0.4359 | 0.5587 | passed |
| 384 | 172 | 0.972560 | 0.731676 | 1.329221x | 0.4361 | 0.5747 | passed |
| 400 | 172 | 0.976941 | 0.735964 | 1.327431x | 0.4390 | 0.4651 | passed |
| 416 | 172 | 0.987109 | 0.743487 | 1.327675x | 0.4356 | 0.5525 | passed |
| 432 | 172 | 0.992406 | 0.747328 | 1.327940x | 0.4376 | 0.5650 | passed |
| 448 | 172 | 0.994456 | 0.754832 | 1.317453x | 0.4342 | 0.5952 | passed |
| 464 | 172 | 0.996567 | 0.758036 | 1.314669x | 0.4344 | 0.9524 | passed |
| 480 | 172 | 0.984226 | 0.777905 | 1.265226x | 0.4380 | 0.5848 | passed |
| 496 | 344 | 0.990718 | 0.793032 | 1.249279x | 0.4345 | 0.5376 | passed |
| 512 | 344 | 1.010703 | 0.803183 | 1.258371x | 0.4345 | 0.5848 | passed |
| 2048 | 43 | 1.391391 | 2.258018 | 0.616200x | 0.4365 | 0.4484 | passed |
| 16384 | 946 | 7.152458 | 7.800194 | 0.916959x | 0.4358 | 0.6881 | passed |

</details>

<details>
<summary>FP8: 53 attempted shapes</summary>

| M | count | Native ms | Candidate ms | Speedup | RMS % | Peak % | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 172 | 0.032338 | 0.031861 | 1.014971x | 0.6900 | 0.9868 | passed |
| 2 | 172 | 0.063630 | 0.055847 | 1.139362x | 0.6183 | 0.6289 | passed |
| 4 | 172 | 0.102107 | 0.085417 | 1.195390x | 0.5943 | 0.6667 | passed |
| 8 | 172 | 0.159392 | 0.154034 | 1.034786x | 0.6078 | 0.6757 | passed |
| 16 | 172 | 0.264459 | 0.255827 | 1.033741x | 0.6034 | 0.9050 | passed |
| 24 | 172 | 0.368935 | 0.347450 | 1.061837x | 0.6007 | 0.9479 | passed |
| 32 | 172 | 0.493275 | 0.397974 | 1.239465x | 0.4799 | 0.5000 | passed |
| 40 | 172 | 0.605247 | 0.474564 | 1.275375x | 0.4629 | 0.6024 | passed |
| 48 | 172 | 0.637034 | 0.513924 | 1.239550x | 0.4710 | 0.8658 | passed |
| 56 | 172 | 0.690165 | 0.542722 | 1.271672x | 0.4712 | 0.6173 | passed |
| 64 | 172 | 0.718391 | 0.580053 | — | 0.5069 | 11.3393 | retained native accuracy failure; excluded |
| 72 | 172 | 0.786032 | 0.617991 | 1.271914x | 0.4353 | 0.6803 | passed |
| 80 | 172 | 0.793459 | 0.617399 | 1.285164x | 0.4366 | 0.5682 | passed |
| 88 | 172 | 0.842172 | 0.648100 | 1.299447x | 0.4387 | 0.6944 | passed |
| 96 | 172 | 0.798001 | 0.627891 | 1.270923x | 0.4366 | 0.6452 | passed |
| 104 | 172 | 0.859668 | 0.669247 | 1.284529x | 0.4325 | 0.7299 | passed |
| 112 | 172 | 0.907921 | 0.690649 | 1.314591x | 0.4357 | 0.5988 | passed |
| 120 | 172 | 0.867108 | 0.678260 | 1.278429x | 0.4331 | 0.6289 | passed |
| 128 | 172 | 0.911208 | 0.691723 | 1.317302x | 0.4391 | 0.7092 | passed |
| 136 | 172 | 0.880532 | 0.688336 | 1.279218x | 0.4396 | 0.6536 | passed |
| 144 | 172 | 0.916844 | 0.710411 | 1.290582x | 0.4333 | 0.6623 | passed |
| 152 | 172 | 0.921715 | 0.710291 | 1.297659x | 0.4441 | 0.7246 | passed |
| 160 | 172 | 0.927207 | 0.721400 | 1.285289x | 0.4395 | 0.7194 | passed |
| 168 | 172 | 0.928839 | 0.723904 | 1.283097x | 0.4381 | 0.5814 | passed |
| 176 | 172 | 0.936004 | 0.733489 | 1.276098x | 0.4403 | 0.4975 | passed |
| 184 | 172 | 0.930762 | 0.749466 | 1.241901x | 0.4419 | 0.8152 | passed |
| 192 | 172 | 0.937871 | 0.748611 | 1.252814x | 0.4406 | 0.6250 | passed |
| 200 | 172 | 0.935780 | 0.759952 | 1.231367x | 0.4372 | 0.5556 | passed |
| 208 | 172 | 0.946224 | 0.763020 | 1.240104x | 0.4429 | 0.5952 | passed |
| 216 | 172 | 0.951479 | 0.773519 | 1.230065x | 0.4399 | 0.6944 | passed |
| 224 | 172 | 0.942627 | 0.772138 | 1.220802x | 0.4420 | 0.6897 | passed |
| 232 | 172 | 0.954351 | 0.777322 | 1.227741x | 0.4385 | 0.5917 | passed |
| 240 | 172 | 0.946925 | 0.773313 | 1.224505x | 0.4356 | 0.5128 | passed |
| 248 | 172 | 0.944816 | 0.776260 | 1.217139x | 0.4353 | 0.6135 | passed |
| 256 | 172 | 0.953698 | 0.774519 | 1.231343x | 0.4405 | 0.5814 | passed |
| 272 | 172 | 0.956785 | 0.786793 | 1.216057x | 0.4410 | 0.9677 | passed |
| 288 | 172 | 0.953544 | 0.786303 | 1.212693x | 0.4403 | 0.8065 | passed |
| 304 | 172 | 0.961675 | 0.789360 | 1.218296x | 0.4412 | 0.6289 | passed |
| 320 | 172 | 0.965173 | 0.793655 | 1.216111x | 0.4358 | 0.5464 | passed |
| 336 | 172 | 0.966928 | 0.797231 | 1.212859x | 0.4391 | 0.4739 | passed |
| 352 | 172 | 0.965037 | 0.803868 | 1.200492x | 0.4378 | 0.7772 | passed |
| 368 | 172 | 0.979996 | 0.813975 | 1.203964x | 0.4388 | 0.6536 | passed |
| 384 | 172 | 0.972448 | 0.813231 | 1.195784x | 0.4382 | 0.5848 | passed |
| 400 | 172 | 0.976488 | 0.821188 | 1.189117x | 0.4377 | 0.5917 | passed |
| 416 | 172 | 0.988298 | 0.828011 | 1.193582x | 0.4367 | 0.5988 | passed |
| 432 | 172 | 0.993478 | 0.832570 | 1.193267x | 0.4364 | 0.6024 | passed |
| 448 | 172 | 0.994320 | 0.840524 | 1.182977x | 0.4398 | 0.5376 | passed |
| 464 | 172 | 0.997388 | 0.843229 | 1.182820x | 0.4394 | 0.6536 | passed |
| 480 | 172 | 0.985160 | 0.864756 | 1.139235x | 0.4379 | 0.6098 | passed |
| 496 | 344 | 0.992147 | 0.878885 | 1.128869x | 0.4385 | 0.5814 | passed |
| 512 | 344 | 1.010598 | 0.894781 | 1.129436x | 0.4388 | 0.5181 | passed |
| 2048 | 43 | 1.393234 | 2.288944 | 0.608680x | 0.4377 | 1.0753 | passed |
| 16384 | 946 | 7.156365 | 7.833453 | 0.913565x | 0.4392 | 0.7463 | passed |

</details>

<details>
<summary>INT4: 53 attempted shapes</summary>

| M | count | Native ms | Candidate ms | Speedup | RMS % | Peak % | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 172 | 0.048713 | 0.036363 | 1.339601x | 0.4176 | 0.5181 | passed |
| 2 | 172 | 0.060852 | 0.051142 | 1.189870x | 0.4054 | 0.5618 | passed |
| 4 | 172 | 0.088215 | 0.076382 | 1.154930x | 0.4375 | 0.4274 | passed |
| 8 | 172 | 0.152839 | 0.147621 | 1.035350x | 0.4176 | 0.5482 | passed |
| 16 | 172 | 0.295156 | 0.227194 | 1.299137x | 0.4357 | 0.7519 | passed |
| 24 | 172 | 0.398128 | 0.293867 | 1.354791x | 0.4291 | 0.5682 | passed |
| 32 | 172 | 0.447410 | 0.360947 | 1.239547x | 0.4494 | 0.7937 | passed |
| 40 | 172 | 0.536879 | 0.415353 | 1.292584x | 0.4355 | 0.5319 | passed |
| 48 | 172 | 0.578390 | 0.453127 | 1.276442x | 0.4436 | 0.6803 | passed |
| 56 | 172 | 0.615405 | 0.471438 | 1.305379x | 0.4346 | 0.6098 | passed |
| 64 | 172 | 0.649522 | 0.481500 | 1.348957x | 0.4372 | 0.6024 | passed |
| 72 | 172 | 0.696060 | 0.537921 | 1.293981x | 0.4367 | 0.5587 | passed |
| 80 | 172 | 0.699811 | 0.539270 | 1.297702x | 0.4399 | 0.5376 | passed |
| 88 | 172 | 0.730138 | 0.558238 | 1.307933x | 0.4377 | 0.5155 | passed |
| 96 | 172 | 0.709566 | 0.549567 | 1.291137x | 0.4419 | 0.6410 | passed |
| 104 | 172 | 0.757422 | 0.566720 | 1.336501x | 0.4391 | 0.6897 | passed |
| 112 | 172 | 0.790255 | 0.598316 | 1.320797x | 0.4385 | 0.7905 | passed |
| 120 | 172 | 0.764712 | 0.571790 | 1.337402x | 0.4387 | 0.7194 | passed |
| 128 | 172 | 0.790613 | 0.597447 | 1.323319x | 0.4387 | 0.6061 | passed |
| 136 | 172 | 0.775079 | 0.592739 | 1.307624x | 0.4413 | 0.5435 | passed |
| 144 | 172 | 0.795119 | 0.609762 | 1.303982x | 0.4418 | 0.5714 | passed |
| 152 | 172 | 0.800497 | 0.608818 | 1.314837x | 0.4326 | 0.5587 | passed |
| 160 | 172 | 0.808918 | 0.621356 | 1.301859x | 0.4405 | 0.6211 | passed |
| 168 | 172 | 0.807280 | 0.623785 | 1.294165x | 0.4423 | 0.6289 | passed |
| 176 | 172 | 0.817925 | 0.661415 | 1.236629x | 0.4445 | 0.5376 | passed |
| 184 | 172 | 0.818466 | 0.660534 | 1.239098x | 0.4401 | 0.5952 | passed |
| 192 | 172 | 0.822011 | 0.663432 | 1.239027x | 0.4362 | 0.5747 | passed |
| 200 | 172 | 0.820709 | 0.663415 | 1.237099x | 0.4377 | 0.5236 | passed |
| 208 | 172 | 0.826260 | 0.669597 | 1.233965x | 0.4403 | 0.7426 | passed |
| 216 | 172 | 0.831384 | 0.672543 | 1.236179x | 0.4380 | 0.6849 | passed |
| 224 | 172 | 0.826855 | 0.673052 | 1.228516x | 0.4380 | 0.5714 | passed |
| 232 | 172 | 0.835004 | 0.676244 | 1.234769x | 0.4370 | 0.5650 | passed |
| 240 | 172 | 0.828588 | 0.672374 | 1.232332x | 0.4407 | 0.6135 | passed |
| 248 | 172 | 0.828855 | 0.673386 | 1.230876x | 0.4385 | 0.4739 | passed |
| 256 | 172 | 0.828963 | 0.676398 | 1.225555x | 0.4415 | 0.5076 | passed |
| 272 | 172 | 0.836955 | 0.680874 | 1.229237x | 0.4376 | 0.5263 | passed |
| 288 | 172 | 0.837225 | 0.682696 | 1.226351x | 0.4401 | 0.5495 | passed |
| 304 | 172 | 0.835883 | 0.686199 | 1.218134x | 0.4398 | 0.7576 | passed |
| 320 | 172 | 0.841317 | 0.688296 | 1.222318x | 0.4385 | 0.5405 | passed |
| 336 | 172 | 0.843751 | 0.689936 | 1.222941x | 0.4362 | 0.4902 | passed |
| 352 | 172 | 0.844323 | 0.690953 | 1.221968x | 0.4400 | 1.0417 | passed |
| 368 | 172 | 0.855804 | 0.705220 | 1.213528x | 0.4387 | 0.5747 | passed |
| 384 | 172 | 0.851231 | 0.700544 | 1.215099x | 0.4422 | 0.5587 | passed |
| 400 | 172 | 0.854526 | 0.707168 | 1.208378x | 0.4358 | 1.0256 | passed |
| 416 | 172 | 0.859569 | 0.717107 | 1.198661x | 0.4359 | 0.5952 | passed |
| 432 | 172 | 0.860811 | 0.717918 | 1.199038x | 0.4438 | 0.8427 | passed |
| 448 | 172 | 0.870647 | 0.726906 | 1.197743x | 0.4405 | 0.5495 | passed |
| 464 | 172 | 0.873336 | 0.730260 | 1.195925x | 0.4373 | 0.5128 | passed |
| 480 | 172 | 0.887534 | 0.744434 | 1.192226x | 0.4400 | 0.5051 | passed |
| 496 | 344 | 0.900798 | 0.754417 | 1.194031x | 0.4384 | 0.5155 | passed |
| 512 | 344 | 0.910519 | 0.773588 | 1.177008x | 0.4414 | 0.5848 | passed |
| 2048 | 43 | 1.679465 | 2.243149 | 0.748709x | 0.4399 | 0.4566 | passed |
| 16384 | 946 | 8.898035 | 8.860129 | 1.004278x | 0.4387 | 0.9217 | passed |

</details>

<details>
<summary>MXFP4: 53 attempted shapes</summary>

| M | count | Native ms | Candidate ms | Speedup | RMS % | Peak % | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 172 | 0.054698 | 0.044053 | 1.241622x | 0.4415 | 0.5618 | passed |
| 2 | 172 | 0.069629 | 0.062402 | 1.115821x | 0.4412 | 0.5128 | passed |
| 4 | 172 | 0.103446 | 0.096244 | 1.074832x | 0.4618 | 0.5165 | passed |
| 8 | 172 | 0.175384 | 0.180016 | 0.974271x | 0.4474 | 0.8621 | passed |
| 16 | 172 | 0.325643 | 0.292391 | 1.113723x | 0.4306 | 0.6803 | passed |
| 24 | 172 | 0.439984 | 0.380644 | 1.155894x | 0.4275 | 0.4425 | passed |
| 32 | 172 | 0.497136 | 0.458377 | 1.084555x | 0.4428 | 0.4049 | passed |
| 40 | 172 | 0.595640 | 0.538914 | 1.105260x | 0.4418 | 0.7692 | passed |
| 48 | 172 | 0.640710 | 0.582853 | 1.099266x | 0.4416 | 0.6536 | passed |
| 56 | 172 | 0.681574 | 0.615056 | 1.108150x | 0.4381 | 0.6711 | passed |
| 64 | 172 | 0.717127 | 0.627934 | 1.142042x | 0.4345 | 0.6024 | passed |
| 72 | 172 | 0.772014 | 0.698787 | 1.104791x | 0.4360 | 0.6024 | passed |
| 80 | 172 | 0.771889 | 0.697900 | 1.106016x | 0.4342 | 0.5682 | passed |
| 88 | 172 | 0.811387 | 0.730918 | 1.110092x | 0.4425 | 0.5464 | passed |
| 96 | 172 | 0.783895 | 0.709504 | 1.104851x | 0.4373 | 0.6667 | passed |
| 104 | 172 | 0.837441 | 0.741407 | 1.129529x | 0.4352 | 0.7299 | passed |
| 112 | 172 | 0.872370 | 0.782993 | 1.114148x | 0.4366 | 0.7519 | passed |
| 120 | 172 | 0.846140 | 0.747129 | 1.132522x | 0.4436 | 0.6944 | passed |
| 128 | 172 | 0.873035 | 0.781873 | 1.116595x | 0.4375 | 0.5618 | passed |
| 136 | 172 | 0.854234 | 0.774909 | 1.102367x | 0.4408 | 0.5587 | passed |
| 144 | 172 | 0.875347 | 0.796090 | 1.099558x | 0.4389 | 0.5952 | passed |
| 152 | 172 | 0.882983 | 0.794722 | 1.111060x | 0.4348 | 0.5000 | passed |
| 160 | 172 | 0.893111 | 0.810100 | 1.102469x | 0.4406 | 0.9554 | passed |
| 168 | 172 | 0.888976 | 0.812463 | 1.094174x | 0.4430 | 0.6369 | passed |
| 176 | 172 | 0.902610 | 0.843106 | 1.070577x | 0.4386 | 0.5848 | passed |
| 184 | 172 | 0.900923 | 0.840998 | 1.071255x | 0.4394 | 0.6410 | passed |
| 192 | 172 | 0.903250 | 0.844319 | 1.069797x | 0.4419 | 0.6536 | passed |
| 200 | 172 | 0.902903 | 0.845625 | 1.067734x | 0.4395 | 0.5587 | passed |
| 208 | 172 | 0.908228 | 0.851153 | 1.067055x | 0.4394 | 0.5848 | passed |
| 216 | 172 | 0.917995 | 0.855346 | 1.073244x | 0.4379 | 0.6289 | passed |
| 224 | 172 | 0.906442 | 0.855058 | 1.060094x | 0.4396 | 0.5405 | passed |
| 232 | 172 | 0.917659 | 0.858447 | 1.068976x | 0.4360 | 0.6410 | passed |
| 240 | 172 | 0.908594 | 0.854208 | 1.063668x | 0.4418 | 0.6098 | passed |
| 248 | 172 | 0.908794 | 0.854475 | 1.063570x | 0.4384 | 0.4651 | passed |
| 256 | 172 | 0.911407 | 0.858435 | 1.061708x | 0.4362 | 0.5128 | passed |
| 272 | 172 | 0.920049 | 0.864274 | 1.064533x | 0.4380 | 0.5181 | passed |
| 288 | 172 | 0.920543 | 0.865747 | 1.063293x | 0.4365 | 0.5128 | passed |
| 304 | 172 | 0.920888 | 0.868509 | 1.060310x | 0.4405 | 0.5435 | passed |
| 320 | 172 | 0.923769 | 0.870606 | 1.061065x | 0.4382 | 0.5376 | passed |
| 336 | 172 | 0.925705 | 0.871872 | 1.061745x | 0.4387 | 0.9804 | passed |
| 352 | 172 | 0.923716 | 0.873384 | 1.057630x | 0.4383 | 0.5464 | passed |
| 368 | 172 | 0.937263 | 0.889576 | 1.053606x | 0.4356 | 0.5848 | passed |
| 384 | 172 | 0.932736 | 0.884969 | 1.053976x | 0.4390 | 0.6135 | passed |
| 400 | 172 | 0.935388 | 0.892630 | 1.047901x | 0.4404 | 0.5155 | passed |
| 416 | 172 | 0.944809 | 0.904093 | 1.045034x | 0.4392 | 0.5650 | passed |
| 432 | 172 | 0.945944 | 0.905077 | 1.045153x | 0.4365 | 0.6024 | passed |
| 448 | 172 | 0.956161 | 0.915806 | 1.044065x | 0.4396 | 0.5650 | passed |
| 464 | 172 | 0.955067 | 0.918109 | 1.040255x | 0.4390 | 0.5236 | passed |
| 480 | 172 | 0.974214 | 0.938783 | 1.037741x | 0.4362 | 1.1173 | passed |
| 496 | 344 | 0.990865 | 0.950937 | 1.041988x | 0.4377 | 0.6098 | passed |
| 512 | 344 | 1.001918 | 0.972760 | 1.029974x | 0.4389 | 0.5848 | passed |
| 2048 | 43 | 1.701534 | 2.656946 | 0.640409x | 0.4377 | 0.4425 | passed |
| 16384 | 946 | 9.653156 | 8.930081 | 1.080971x | 0.4386 | 1.0000 | passed |

</details>

## Part 3 — Validation and reproduction

Final W8 source `6277aac464d8c5ca840696bb2629b402b03ea7dc` (`d9b1bbdec7ef3036e0db3356220a099b16523b03b195b2478adf78d785072efe`) ran 873 unique functional cases with 695 passed, 178 skipped and 0 failed across six disjoint shards, each with completed GPU-context monitoring. Existing tests cover both row sizes, all quantized formats, noncontiguous/unsafe scales, graph and inference updates to IDs/weights/scales, output aliases and workspace limits. The test file is `64d2fec9eee01cf5e9859f972676dc73b9db79d141172362d788011aacc18741`; numerical bounds remain RMS<1% and global peak<2%.

The earlier matrix-x2 update (`52a2ed3b`) passed three nonperiodic byte-pattern comparisons across both activation planes/all eight warps and12 independent-FP32 tile cases, including normal/unsafe FP8 scales and K128/256/4096. Full53-shape A/B outputs were bit-identical for both formats, each count10105. Count-weighted gains against the previous source were1.005826x INT8 and1.006083x FP8; these A/B gains are distinct from the native speedups in Part2. Formatting preserves the candidate AST exactly.

On the corresponding valid shape sets, INT8 changes from 1.314846x to 1.314680x with count 9589, and FP8 changes from 1.177011x to 1.182837x with count 9933. Native repeat stability remains open. The user requested the latest supported vendor-native baseline for performance comparisons. [w8_fp8_unroll_update.json](w8_fp8_unroll_update.json) records the earlier rows, source pins and diagnostics. [w8_fp8_schedule_update.json](w8_fp8_schedule_update.json) preserves the preceding checkpoint.

Tiny raw split-K, single-Q buffering, row-major dense decoding and K64 schedules passed their stated accuracy checks but regressed in complete-call pilots and were rejected. An isolated wider-copy probe faulted and its dependent test did not run; new-process known-copy health and the same-allocation KT128 control both passed. Those research changes are absent from the submitted source; no driver reset or foreign process change was performed. Three-buffer AIU and separate pair/tail GEMM launches also passed their stated checks but regressed in complete calls and were excluded. W4 code paths and native measurements remain frozen.

| Component | Tested version |
| --- | --- |
| Accelerator | T-Head Zhenwu810E,96GB, single-card operator testing |
| Runtime | SAIL SDK2.2; existing UMD driver mode, no device/driver configuration changes |
| Driver status at integration |1.3.2-d7f5a2 |
| PyTorch |2.13.0 PPU2.2 build, common to both environments |
| Native compiler | Vendor Triton3.7.0 PPU2.2 build |
| Candidate compiler | FlagTree0.7.0+ppu.git1a96a36a, Triton3.6.0 |
| vLLM/SAIL | Versions and commits listed in Part2 |

The final 873-case regression and native rows are in [w8_fp8_unroll_update.json](w8_fp8_unroll_update.json).

Pre-commit,84 CI helper tests,10 workflow-policy tests, registry/export/affected-operator marker checks and the no-Torch-compute audit passed before publication. Whole-repository marker inspection reports25 unrelated pre-existing issues; the changed operators pass their incremental marker check. `tools/check_ci_pins.py` is absent from the current repository and is not invoked by its actual workflow. The canonical core benchmark entry check is recorded separately: **0 passed,4 skipped**, exit0. The existing device gates require NVIDIA Hopper/Hygon and, for two entries, Moore Threads; they exclude this PPU environment. It does not replace the vendor-factory measurements above. The earlier vLLM 0.24 empty environment checks are retained for diagnostics. This checkpoint compares performance only with the latest supported native baseline, as requested. Full model serving, other accelerators and a complete FP16 performance distribution are not validated.

Candidate environment variables include `UMD_PLATFORM_TYPE=1`, `HGGC_DRIVER_CANDIDATE=UMD`, `DNN_VENDOR=thead`, `VLLM_PLUGINS=ppu`, both selected-device visibility variables, and the existing host-driver/tool paths. Use an idle mapped device; launchers enforce UUID/context ownership. The recorded container mapping is ordinal0..5 to host physical0,10,11,12,13,14.

Functional reproduction from the repository root in the existing candidate environment (visibility0 is an example; select an idle mapped device):

```bash
export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
export DNN_VENDOR=thead VLLM_PLUGINS=ppu
export CUDA_VISIBLE_DEVICES=0 HGGC_VISIBLE_DEVICES=0
export PATH="/opt/host-tools:$PATH"
export LD_LIBRARY_PATH="/opt/host-driver:$LD_LIBRARY_PATH"
export PYTHONPATH="src:/work/deps:.:/work"
/opt/candidate031/bin/python -m pytest -q \
  tests/test_fused_marlin_moe.py
```

Core benchmark entry checks:

```bash
/opt/candidate031/bin/python -m pytest -q \
  benchmark/test_fused_marlin_moe_w8a16_int8.py \
  benchmark/test_fused_marlin_moe_w8a16_fp8.py \
  benchmark/test_fused_marlin_moe_w4a16_int4.py \
  benchmark/test_fused_marlin_moe_w4a16_mxfp4.py \
  --level core --warmup 1 --iter 1 --record json
```

The matched vendor measurement uses the frozen external test-host harness, which keeps native Triton3.7 and candidate FlagTree3.6 imports in separate processes. The harness, temporary shards and device scheduling stay outside the repository benchmark. Existing test-host directories are `/tmp/pr752-vendor-baseline-20261116/opt-20261116/phase135-fp8-unroll-final` for current W8 and `phase23-sparse-bounded` for W4. `run_pair.py`, `worker_diagnostic.py`, raw JSON and monitors are retained in each measurement directory. Shared input/native-W4 helpers are under `/work` in the container; the final W8 input helper is also archived as `inputs_snapshot.py`. On that host, choose idle physical/ordinal values and unused job names, then run each disjoint shard0..2:

```bash
# In the corresponding frozen test directory; example mapping only.
python3 run_pair.py --gpu 0 --device 0 --job review-int8-s0 \
  --precision int8 --shard-i 0 --shard-n 3
# W4 uses the phase23 directory and the native W4 flag.
python3 run_pair.py --gpu 0 --device 0 --job review-int4-s0 \
  --precision int4 --shard-i 0 --shard-n 3 --native-w4
```

INT8/FP8 omit `--native-w4`; INT4/MXFP4 require it. A context interruption is excluded as an entire job and retained, with a fresh idle-card retry. A numerical failure is diagnosed against FP32, preserved and excluded without resampling to obtain a pass.

Evidence archives retained in the task workspace/test logs:

| Archive | SHA256 | Verified files | Contents |
| --- | --- | ---: | --- |
| phase135-final-results.tar.gz |`7c4c1d3f6c7a9f259790b646313e3767d038a14fe8ddfa168c00dc2f341de5c5`|92|Merged source, final verification, native rows, memory and core records |
| phase125-final-results.tar.gz |`87eded2054e0db720a2473df39d7deb38d31d39990bf841f144a4a22562e8316`|97|Merged source, final verification, native rows, memory and core records |
| phase112-final-results.tar.gz |`25be75ba5a3433f6be10aa24b017639cec93a6c9037e7c0515c6b3149ac2cd4a`|97|Merged source, final verification, native rows, memory and core records |
| phase106-final-results.tar.gz |`f8e6620ba49a68119a2426d72cf21cb5e3227360bb3492b06696f733eb02995a`|85|Merged source, final verification, native rows, memory and core records |
| phase99-final-results-with-failures.tar.gz |`32e72bbd27dfb07b1657dc20e3db47f919b5d0c8522993dc8a361dc1ba229fbc`|90|871 functional cases, current native rows, M64 failure tensors, CPU diagnostics, memory and core entry checks |
| phase81-results.tar.gz |`5e4d804fb85972d95788706a2963ac0088f48638df486823267d473579bf1bf5`|50|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase82-results.tar.gz |`a66113e4261679e9474786db58e6c1a43ac534d6ed89c7e349e2cf0f3e09aa95`|27|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase83-results.tar.gz |`83e07a94f4bb8343d6114a0137f99e79279645215e8a29d88640c17422fe5108`|33|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase84-results.tar.gz |`f0f7f0f5f4460e2e146a3f45cffbf4e28dbd812c4426c95c36013a721f35de06`|23|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase85-results.tar.gz |`6a12e6409922235179c1640d9dddbc1813d574f67bc3acd3288acb1bef6749f8`|33|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase86-complete-results.tar.gz |`1c011a065e3748f6188de85e65f4b0f77a56341eb3bdbb65b7fd65ca1d39b4f3`|54|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase87-results.tar.gz |`73a55d361f06c5e33b0ad808517b17be013d96d80e097f52112937b73ee6f8d6`|67|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase88-final-results.tar.gz |`ddaa252b1626863eecaab7a36ebda59ed8d4a82510c2735158fcec8e09bf1f18`|88|Final source/tests,870 functional cases,104 fresh native rows, both retained failures, real-memory bound and four skipped core entries |
| phase73-results.tar.gz |`e4f7dd127bdd6cd1f71be8374e9a54514a60e12d8846c9c5758df1b0bcffdade`|13|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase74-results.tar.gz |`09df281cd74a0c340f2ea2af2684c848b2570ece39336f56a100693c7d62c96d`|32|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase75-results.tar.gz |`535fb5abe26a869e3c92f95263a7a37618521445623eefc72069fab28c340c55`|82|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase76-results.tar.gz |`e58edf01f14d193b85e8939bf85f00af928274bc74cc756f48cacbb19387823f`|87|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase77-results.tar.gz |`3a700c24883b7f8809333116bad34f1ff1c9070d28485ce3cefd200273e19817`|40|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase78-results.tar.gz |`610fd15de9dd710aa7484f5bfa55ed583f5816bba9ff9e2d3739fbb1f602efec`|25|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase79-results.tar.gz |`916476e84f20c101133864db5953d7b28ce94cf8acd4ed8bc890a0a434363095`|84|Component or merged research, correctness/performance evidence, source pins and monitors |
| phase80-final-results.tar.gz |`7936d3daf166e839103b18ab38eab4a79d4e1d8f35cf59853d4418e961c15b1c`|88|Final source/tests,870 functional cases,104 fresh native rows, both retained failures, real-memory bound and four skipped core entries |
| phase69-70-results.tar.gz |`952440a88a6419cdf2f95530591dd200e17ad754f483111572cd0c20ff5262e4`|40|Routing/ISA research, exact checks, measurements and device monitors |
| phase71-complete-results.tar.gz |`049177f0969f7fd69c10cac3037dba0131109969b2a958fabfd3bb2313c265b4`|70|Routing/ISA research, exact checks, measurements and device monitors |
| phase72-final-results.tar.gz |`1eeff229d04ea618084a185e8ac5dc7d71e0daaef9f9ba0ef191b5f5a8c6c23b`|80|Final source/tests,870 functional cases,105 fresh attempts,104 valid native rows, new M48 failure, retained M56 exclusion, four skipped core entries |
| phase62-evidence.tgz |`f9a65091bd9efcf407f0225d199e84a66ad0f707e88c29ab710113f5f1d5770e`|29|Validated component/combined research, source pins, measurements and device monitors |
| phase63-evidence.tgz |`7551fcad5d09d22c984175f2546f53c3dba6131b9ef5e0367f5820a2d3dab234`|30|Validated component/combined research, source pins, measurements and device monitors |
| phase64-evidence.tgz |`c360ac2bc4dbeb3a19f5e8b47ca647e028a5846b58b77c0dc5452034a090c231`|33|Validated component/combined research, source pins, measurements and device monitors |
| phase65-evidence.tgz |`f4c4c006477ce250e9a6931ab39734349ba3615556148cf715e887e6ac74e77d`|24|Validated component/combined research, source pins, measurements and device monitors |
| phase66-evidence.tgz |`6f31eb2b320ea14f1e34095d5dad41d81d5ad56e03835c0377cbc96b7af137b1`|24|Validated component/combined research, source pins, measurements and device monitors |
| phase67-evidence.tgz |`b8736170edade9b3fc578df9221beb70c604094c774eea20caee1796f0481fc0`|49|Validated component/combined research, source pins, measurements and device monitors |
| phase68-evidence.tgz |`7d43fca7b9a6bfbc8e9e730c540db1b0a4a2f6a1b22bfe6db6dd62fc91e306d2`|77|Final source/tests,690 functional cases,105 fresh valid native rows, retained M56 exclusion, four skipped core entries |
| phase55-56-evidence.tgz |`8aeea6dc67482ca412e66ab368d1a4e333c2996646a95d5ead8779c17ac72a31`|69|Validated research, measurements, sources and device monitors |
| phase57-58-evidence.tgz |`39d002ef25234bbe31ef60ea68788fecb3ec2be1194475c2b3f5940cb7c4d983`|61|Validated research, measurements, sources and device monitors |
| phase59-evidence.tgz |`1dd2897d7f517f3e613420e6c4702ac45b635427cb8b0ba30a4ecb178a9772cd`|54|Validated research, measurements, sources and device monitors |
| phase60-evidence.tgz |`58427e508f086a1df85d251d3d6ae11789d34127e2595288a6c38cb746660a89`|79|Validated research, measurements, sources and device monitors |
| phase61-evidence.tgz |`ef1896af6990038048c7474e185728fc3c1a05b5adead227d80991e2194d25d9`|77|Final source/tests,674 functional cases,105 fresh valid native rows, retained M56 exclusion, four skipped core entries |
| pr23-target13-phase43-48.tgz |`a91d9d1b40c13c7db1925a00717b8b08c19f541614139cb53947c834bd44cd3f`|173|Full53 A/B in each W8 format, load proofs, resources, accepted/rejected research and health controls |
| pr23-target13-phase49.tgz |`e240720e4a0ad88d042454c18a9be4595922034e33d14270bc5c545f4b701968`|79|Final source/tests,674 functional cases,105 fresh valid W8 native rows, retained M56 exclusion, four skipped core entries |
| phase26-evidence.tgz |`09f989cac571f01863f7747f2de76e20c9675c65aa114ef3e6f9faa233f0fc94`|91|Previous INT8 checkpoint,654 functional cases, full INT8 A/B,53 native attempts, failed M56 tensors, context interruption and legitimate retry |
| phase23-evidence.tgz |`d08e5e05cc514b13a216a32be87d4605ae1de6bf7a609a225fdbd52fd8dcb439`|106|Validated W4 checkpoint and106 native points |
| phase24-28-research.tgz |`a57899604c1bc5d4b477dad22280997f68ffa118b44fd02c4cd0c5613ee97e7c`|478|Stage/resource profiles and excluded experiments; diagnostic data are not used as formal speedups |

Later AIU source-row cropping and constant-argument experiments are not part of the submitted implementation. Their mixed or negligible complete-call results did not replace this validated checkpoint.
