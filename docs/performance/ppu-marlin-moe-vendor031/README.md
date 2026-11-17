# PPU quantized fused Marlin MoE: vendor vLLM 0.31 checkpoint

This is a review checkpoint for the T-Head implementation added by [upstream PR #752](https://github.com/flagos-ai/FlagGems-vllm/pull/752). It contains the validated optimization through `add30d25a5e09fe784241954f9261359f666f49d` and its tests. **INT4 and MXFP4 are frozen at the user-approved native W4A16 checkpoints. FP8 and INT8 remain below the count-weighted 1.3x target.** INT8 has one failed native BF16 accuracy sample, which is retained and excluded from its weighted result.

## Part 1 — Operator implementation and optimization

Review status: draft; performance and fixed-environment acceptance remain open.

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
| Large BF16 batches | Reuse the existing bounded decode-bank + BF16 grouped-GEMM implementation; gate/up columns256, down columns1024 | Decode each consumed column slice per invocation and reuse scratch, with a 1GiB decoded-bank and 2GiB temporary-workspace bound |

For eligible raw INT8/INT4/MXFP4 calls with at most two tokens, topk<=6 and M*topk<=E<=256, reuse the existing single-route alignment for M=1 and short grouped alignment for M=2. The latter retains expert sharing across tokens. Larger batches and FP8 retain their existing routing. Repeated expert IDs remain valid; no host read of routing data or new GPU arithmetic is introduced.

Eligible raw INT8 calls with at most32 routes now use w8_small_routes_kernel after the existing M=1/2 shortcuts. It computes pairwise expert matches, per-route ranks and padded expert offsets in one CTA; it preserves route order within each expert and groups experts by first appearance. This replaces the two zero-initialization and four alignment launches. The comparison matrix is bounded by32x32; larger calls retain grouped alignment. Sorted storage keeps the existing min(R*BM,R+E*(BM-1)) bound, and the GEMMs consume the matching expert metadata. W4 and FP8 dispatch, dequantization and MMA kernels are unchanged.

The raw MMA treats the weight fragment as N-by-K and the activation fragment as K-by-routed-rows; stores restore route-major output. Activations are permuted within groups of 16 K elements for W8 and 32 for W4. The same self-inverse permutation is used in the gather and fused SiLU/gather stages. It remains inside the original scale group and adds no separate global weight-packing pass.

Exact packed BF16 decoding is retained. The INT8 fragment decoder removes a dependent sign XOR and constructs the subtraction bias with LOP3 before the unchanged BF16 scale multiplication. W4 removes redundant low-16-bit masks before register decoding. MXFP4 combines four adjacent scale-byte loads only when group stride is one, row/expert strides are multiples of four, and the pointer is four-byte aligned; other layouts retain scalar reads. FP8 folding is selected using freshly computed per-expert scale-safety flags, with the robust whole-loop implementation retained for unsafe scales. Finite values, signed zero and Inf retain the tested bit semantics; NaN payloads are unspecified and classes are checked.

The sparse route uses a conservative bound for alignment, both staged activations and projection storage. With `R=M*topk`, padded rows are bounded by `R+7E` in the previously-direct band. The estimate is `4(E+1)^2 + 16(R+7E) + 2(R+7E)(K+N) + 2R(K+2N) + 8` bytes. Calls above the cap retain their existing direct path. Inputs and caller-visible output are outside the temporary budget. Existing peak-memory regressions cover 32768 experts for INT4 and INT8, including approximately0.8/1.6GB input storage.

FP16, unaligned/general geometries, the original direct implementation and bounded dense fallback remain available under their existing conditions. Unsupported compatibility arguments still fail explicitly. Production computation uses Triton/TLE and the existing FlagGems operators; no Torch compute fallback or vendor-native replacement of the candidate is added.

### TLE features

| API | Location | Role |
| --- | --- | --- |
| `tle_async.gpu.alloc` | Raw narrow/square W8 and W4 kernels | Allocate power-of-two shared Q/A double buffers |
| `tle_async.gpu.local_ptr` | The same kernels | Obtain shared addresses consumed by PPU copies and fragment loads |

NVIDIA autotuning is unchanged. These PPU paths use measured fixed schedules and the existing dispatch conditions; no new FlagTree custom operation or compiler patch is introduced.

### Inline assembly

- `ppu.cp.async.aiu.bulk.tensor...2d.b8`, commit/wait and CTA barriers transfer raw byte tiles and staged activations and protect buffer reuse. Padding is explicitly initialized by the current staging implementation.
- `ppu.ldmatrix...x4.swzl` loads the verified shared-memory fragment layout. BF16 `m16n8k16` and `m16n16k16` MMA accumulate in FP32.
- `prmt`, `lop3`, packed BF16 subtraction and `mul.rn.bf16x2` implement exact quantized fragment decoding and scale multiplication. The eight-row ISA path avoids relying on an unsupported eight-row `tl.dot` lowering in the installed backend.

Measured raw-kernel resources: W8 INT8 eight/sixteen-row gate/down use70/70 and74/74 registers with36/40KiB shared memory. INT4 uses60/58 registers for both row sizes; MXFP4 uses60/60 for eight rows and60/56 for sixteen. W4 shared memory is12/16KiB. These resource records report no spills. The W4 records precede the final host-only workspace predicate and retain identical GPU kernel ASTs.

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
| INT8 | BF16 | 1.200762x | 52/53 | 9933 | 0.617–1.406x | per-row CSV |
| FP8 | BF16 | 1.072796x | 53/53 | 10105 | 0.583–1.218x | `f4e6ceec` |
| INT4 | INT4 | 1.223084x | 53/53 | 10105 | 0.749–1.355x | per-row CSV |
| MXFP4 | MXFP4 | 1.078587x | 53/53 | 10105 | 0.640–1.242x | per-row CSV |

INT8 M56/count172 is invalid: native BF16 peak error against independent FP32 is2.420966%, while the candidate peak is0.564822%; the paired peak is2.372263%. This native failure is preserved and was not resampled. Its count is excluded from both numerator and denominator, formal speedup is blank, and the actual timing ratio is retained only as `diagnostic_speedup` in [performance.csv](performance.csv). [m56-cpu-recheck.json](m56-cpu-recheck.json) independently rechecks the saved three outputs on CPU. Native BF16 repeat instability also appeared in prior vendor-only eager/graph tests; its root cause is unresolved. Current passing samples do not establish native repeat stability.

The earlier M=1/2 INT8, INT4 and MXFP4 rows and the latest INT8 M=4 row were remeasured with their respective routing updates. Other rows retain their previous measurements because their computation and dispatch are unchanged. CSV rows record their exact measured commits and source hashes; this is a mixed-checkpoint aggregate rather than a new full 212-point run. W4 measurements are frozen by the user; FP8 and INT8 remain below1.3x.

### Complete per-shape table

All212 attempts, including the invalid native sample, are retained below and in [performance.csv](performance.csv). Fixed shape parameters are E256/K4096/N256/topk6. Group size is128 for INT8/FP8/INT4 and32 for MXFP4. Units are milliseconds; error columns are percentages of the native reference. M56's independent-FP32 diagnosis is stated above.

<details>
<summary>INT8: 53 attempted shapes</summary>

| M | count | Native ms | Candidate ms | Speedup | RMS % | Peak % | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 172 | 0.032479 | 0.036973 | 0.878451x | 0.5695 | 0.6127 | passed |
| 2 | 172 | 0.054144 | 0.055113 | 0.982417x | 0.5891 | 0.6757 | passed |
| 4 | 172 | 0.102001 | 0.075389 | 1.353004x | 0.5771 | 0.4000 | passed |
| 8 | 172 | 0.159541 | 0.152099 | 1.048929x | 0.6050 | 0.6313 | passed |
| 16 | 172 | 0.263934 | 0.246067 | 1.072608x | 0.6111 | 0.9259 | passed |
| 24 | 172 | 0.368382 | 0.335843 | 1.096885x | 0.6066 | 0.7692 | passed |
| 32 | 172 | 0.493078 | 0.374044 | 1.318237x | 0.4693 | 0.5618 | passed |
| 40 | 172 | 0.604574 | 0.455341 | 1.327739x | 0.4689 | 0.3937 | passed |
| 48 | 172 | 0.636905 | 0.487235 | 1.307182x | 0.4688 | 0.8130 | passed |
| 56 | 172 | 0.690912 | 0.524323 | — | 0.4651 | 2.3723 | native accuracy failed; excluded |
| 64 | 172 | 0.718409 | 0.546172 | 1.315353x | 0.4622 | 0.6667 | passed |
| 72 | 172 | 0.786514 | 0.584069 | 1.346612x | 0.4328 | 0.5618 | passed |
| 80 | 172 | 0.793509 | 0.584106 | 1.358503x | 0.4330 | 0.3268 | passed |
| 88 | 172 | 0.842247 | 0.618605 | 1.361527x | 0.4383 | 0.5618 | passed |
| 96 | 172 | 0.799084 | 0.597266 | 1.337903x | 0.4391 | 0.7042 | passed |
| 104 | 172 | 0.861057 | 0.632148 | 1.362113x | 0.4391 | 0.7874 | passed |
| 112 | 172 | 0.910404 | 0.647624 | 1.405761x | 0.4296 | 0.6250 | passed |
| 120 | 172 | 0.866886 | 0.637874 | 1.359024x | 0.4393 | 0.6623 | passed |
| 128 | 172 | 0.911579 | 0.649185 | 1.404192x | 0.4389 | 0.5747 | passed |
| 136 | 172 | 0.880362 | 0.644849 | 1.365221x | 0.4300 | 0.6173 | passed |
| 144 | 172 | 0.916745 | 0.666651 | 1.375149x | 0.4381 | 0.6098 | passed |
| 152 | 172 | 0.922546 | 0.667712 | 1.381652x | 0.4316 | 0.5208 | passed |
| 160 | 172 | 0.927744 | 0.683292 | 1.357756x | 0.4342 | 0.5952 | passed |
| 168 | 172 | 0.929112 | 0.681254 | 1.363827x | 0.4354 | 0.6757 | passed |
| 176 | 172 | 0.936353 | 0.695711 | 1.345893x | 0.4338 | 0.6061 | passed |
| 184 | 172 | 0.930871 | 0.709054 | 1.312836x | 0.4371 | 0.6452 | passed |
| 192 | 172 | 0.939617 | 0.710918 | 1.321696x | 0.4366 | 0.6061 | passed |
| 200 | 172 | 0.936343 | 0.717550 | 1.304917x | 0.4350 | 0.5882 | passed |
| 208 | 172 | 0.945692 | 0.724547 | 1.305219x | 0.4333 | 0.7732 | passed |
| 216 | 172 | 0.952191 | 0.732659 | 1.299636x | 0.4375 | 0.6944 | passed |
| 224 | 172 | 0.945400 | 0.731646 | 1.292155x | 0.4379 | 0.6536 | passed |
| 232 | 172 | 0.953636 | 0.753334 | 1.265888x | 0.4350 | 0.6452 | passed |
| 240 | 172 | 0.948220 | 0.765533 | 1.238641x | 0.4339 | 0.6667 | passed |
| 248 | 172 | 0.944229 | 0.776847 | 1.215464x | 0.4355 | 0.7092 | passed |
| 256 | 172 | 0.953585 | 0.762184 | 1.251122x | 0.4316 | 0.6024 | passed |
| 272 | 172 | 0.956515 | 0.802710 | 1.191607x | 0.4348 | 0.6098 | passed |
| 288 | 172 | 0.955026 | 0.804920 | 1.186486x | 0.4376 | 0.4405 | passed |
| 304 | 172 | 0.961492 | 0.810024 | 1.186991x | 0.4366 | 0.5952 | passed |
| 320 | 172 | 0.965168 | 0.811690 | 1.189085x | 0.4372 | 0.6098 | passed |
| 336 | 172 | 0.967659 | 0.816422 | 1.185244x | 0.4329 | 0.4762 | passed |
| 352 | 172 | 0.964214 | 0.818001 | 1.178744x | 0.4401 | 0.5376 | passed |
| 368 | 172 | 0.979318 | 0.826148 | 1.185403x | 0.4359 | 0.5587 | passed |
| 384 | 172 | 0.972476 | 0.828753 | 1.173421x | 0.4361 | 0.5747 | passed |
| 400 | 172 | 0.975338 | 0.833884 | 1.169632x | 0.4390 | 0.4651 | passed |
| 416 | 172 | 0.985933 | 0.839276 | 1.174743x | 0.4356 | 0.5525 | passed |
| 432 | 172 | 0.992786 | 0.843337 | 1.177212x | 0.4376 | 0.5650 | passed |
| 448 | 172 | 0.993424 | 0.848848 | 1.170321x | 0.4342 | 0.5952 | passed |
| 464 | 172 | 0.996609 | 0.851975 | 1.169764x | 0.4344 | 0.9524 | passed |
| 480 | 172 | 0.985324 | 0.870381 | 1.132060x | 0.4380 | 0.5848 | passed |
| 496 | 344 | 0.990583 | 0.888129 | 1.115359x | 0.4345 | 0.5376 | passed |
| 512 | 344 | 1.009557 | 0.902264 | 1.118915x | 0.4345 | 0.5848 | passed |
| 2048 | 43 | 1.392988 | 2.258975 | 0.616646x | 0.4365 | 0.4484 | passed |
| 16384 | 946 | 7.151798 | 8.519493 | 0.839463x | 0.4358 | 0.6881 | passed |

</details>

<details>
<summary>FP8: 53 attempted shapes</summary>

| M | count | Native ms | Candidate ms | Speedup | RMS % | Peak % | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 172 | 0.032144 | 0.031877 | 1.008363x | 0.6900 | 0.9868 | passed |
| 2 | 172 | 0.063372 | 0.055851 | 1.134652x | 0.6183 | 0.6289 | passed |
| 4 | 172 | 0.101984 | 0.106452 | 0.958025x | 0.5949 | 0.6667 | passed |
| 8 | 172 | 0.159126 | 0.179527 | 0.886360x | 0.6078 | 0.6757 | passed |
| 16 | 172 | 0.264080 | 0.287495 | 0.918555x | 0.6034 | 0.9050 | passed |
| 24 | 172 | 0.368992 | 0.382255 | 0.965304x | 0.6007 | 0.9479 | passed |
| 32 | 172 | 0.493204 | 0.435784 | 1.131763x | 0.4799 | 0.5000 | passed |
| 40 | 172 | 0.605167 | 0.525104 | 1.152469x | 0.4629 | 0.6024 | passed |
| 48 | 172 | 0.636572 | 0.563043 | 1.130591x | 0.4710 | 0.8658 | passed |
| 56 | 172 | 0.690466 | 0.592487 | 1.165369x | 0.4712 | 0.6173 | passed |
| 64 | 172 | 0.717792 | 0.623767 | 1.150739x | 0.4724 | 0.7143 | passed |
| 72 | 172 | 0.786242 | 0.671591 | 1.170714x | 0.4353 | 0.6803 | passed |
| 80 | 172 | 0.791728 | 0.670879 | 1.180136x | 0.4366 | 0.5682 | passed |
| 88 | 172 | 0.842484 | 0.701721 | 1.200596x | 0.4387 | 0.6944 | passed |
| 96 | 172 | 0.797476 | 0.684822 | 1.164500x | 0.4366 | 0.6452 | passed |
| 104 | 172 | 0.858265 | 0.727420 | 1.179877x | 0.4325 | 0.7299 | passed |
| 112 | 172 | 0.907632 | 0.746592 | 1.215700x | 0.4357 | 0.5988 | passed |
| 120 | 172 | 0.865455 | 0.735186 | 1.177192x | 0.4331 | 0.6289 | passed |
| 128 | 172 | 0.909431 | 0.746893 | 1.217619x | 0.4391 | 0.7092 | passed |
| 136 | 172 | 0.878897 | 0.744697 | 1.180207x | 0.4396 | 0.6536 | passed |
| 144 | 172 | 0.914678 | 0.769066 | 1.189335x | 0.4333 | 0.6623 | passed |
| 152 | 172 | 0.919897 | 0.770161 | 1.194420x | 0.4441 | 0.7246 | passed |
| 160 | 172 | 0.923868 | 0.782351 | 1.180887x | 0.4395 | 0.7194 | passed |
| 168 | 172 | 0.927405 | 0.784011 | 1.182898x | 0.4381 | 0.5814 | passed |
| 176 | 172 | 0.933889 | 0.795606 | 1.173808x | 0.4403 | 0.4975 | passed |
| 184 | 172 | 0.929873 | 0.812279 | 1.144770x | 0.4419 | 0.8152 | passed |
| 192 | 172 | 0.936467 | 0.810181 | 1.155873x | 0.4406 | 0.6250 | passed |
| 200 | 172 | 0.934932 | 0.824189 | 1.134367x | 0.4372 | 0.5556 | passed |
| 208 | 172 | 0.943769 | 0.827468 | 1.140551x | 0.4429 | 0.5952 | passed |
| 216 | 172 | 0.950273 | 0.846834 | 1.122147x | 0.4399 | 0.6944 | passed |
| 224 | 172 | 0.943936 | 0.841787 | 1.121348x | 0.4420 | 0.6897 | passed |
| 232 | 172 | 0.955428 | 0.868782 | 1.099732x | 0.4385 | 0.5917 | passed |
| 240 | 172 | 0.949084 | 0.880725 | 1.077616x | 0.4356 | 0.5128 | passed |
| 248 | 172 | 0.946063 | 0.890330 | 1.062599x | 0.4353 | 0.6135 | passed |
| 256 | 172 | 0.955460 | 0.879695 | 1.086127x | 0.4405 | 0.5814 | passed |
| 272 | 172 | 0.958070 | 0.885067 | 1.082484x | 0.4410 | 0.9677 | passed |
| 288 | 172 | 0.955370 | 0.887640 | 1.076303x | 0.4403 | 0.8065 | passed |
| 304 | 172 | 0.962403 | 0.890113 | 1.081214x | 0.4412 | 0.6289 | passed |
| 320 | 172 | 0.965561 | 0.892560 | 1.081788x | 0.4358 | 0.5464 | passed |
| 336 | 172 | 0.967823 | 0.898467 | 1.077194x | 0.4391 | 0.4739 | passed |
| 352 | 172 | 0.965163 | 0.901015 | 1.071194x | 0.4378 | 0.7772 | passed |
| 368 | 172 | 0.981431 | 0.914470 | 1.073224x | 0.4388 | 0.6536 | passed |
| 384 | 172 | 0.973780 | 0.912970 | 1.066607x | 0.4382 | 0.5848 | passed |
| 400 | 172 | 0.975518 | 0.919830 | 1.060542x | 0.4377 | 0.5917 | passed |
| 416 | 172 | 0.988110 | 0.929448 | 1.063115x | 0.4367 | 0.5988 | passed |
| 432 | 172 | 0.992507 | 0.933437 | 1.063282x | 0.4364 | 0.6024 | passed |
| 448 | 172 | 0.995682 | 0.939880 | 1.059372x | 0.4398 | 0.5376 | passed |
| 464 | 172 | 0.998035 | 0.942670 | 1.058732x | 0.4394 | 0.6536 | passed |
| 480 | 172 | 0.984717 | 0.962230 | 1.023370x | 0.4379 | 0.6098 | passed |
| 496 | 344 | 0.991442 | 0.982125 | 1.009486x | 0.4385 | 0.5814 | passed |
| 512 | 344 | 1.009696 | 0.994081 | 1.015708x | 0.4388 | 0.5181 | passed |
| 2048 | 43 | 1.393194 | 2.390853 | 0.582718x | 0.4377 | 1.0753 | passed |
| 16384 | 946 | 7.154914 | 8.673405 | 0.824926x | 0.4392 | 0.7463 | passed |

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

Latest W8 update: a full regression of674 unique cases passed496 and skipped178, with no failures. After the allocation-bound change and upstream merge, the complete674-case regression again passed496 and skipped178. Four busy-card launches were refused before GPU work; those shards then passed on idle cards. All53 INT8 A/B outputs were bit-identical, count10105; M=4 improved from82.995us to76.663us (7.63% less latency). Its independent FP32 RMS/peak were0.4345%/0.4601%. The subsequent allocation-bound change retains identical buffer dimensions for every frozen real shape;36 targeted public tests and144 canary cases passed on the final source. New routing kernel resources are42 registers,128B shared memory and no spills. See [w8_update.json](w8_update.json) for source pins, the full A/B record and final native M4 measurement. Only that native row is refreshed; other rows keep their recorded commits. INT8 M56 remains excluded.

FP8 trials covered four-warp execution, activation load scheduling, persistent CTA scheduling, dense N128 tiles, mixed8/16-row MMA, independent FP32 accumulator chains, paired scale-safety scans and small-route alignment. None provided a material stable full-call gain for FP8, so its implementation and measurements remain unchanged. Two INT8 research jobs aborted on foreign contexts and were excluded in full; their logs are retained. No driver or foreign process was changed. INT4/MXFP4 data and their source paths stay frozen.

The routing update has **484 passed,178 skipped,0 failed**,662 unique cases across eight disjoint shards, including repeated experts for M=1/2 in all four formats. The same 484/178/0 result was repeated after merging upstream. One busy-card launch was refused before any GPU work; its shard then passed on an idle card. The initial publication results below are retained as historical evidence. See [routing_update.json](routing_update.json) for the update's source pins, tests and six native rows.

Complete-call A/B against the previous PR implementation (microseconds; all outputs bit-identical, independent FP32 bounds passed):

| Format | M=1 before / after | M=2 before / after | M=1 latency reduction | M=2 latency reduction |
| --- | ---: | ---: | ---: | ---: |
| INT8 | 44.00 / 36.15 | 53.69 / 51.20 | 17.83% | 4.64% |
| INT4 | 43.33 / 35.53 | 52.88 / 50.47 | 18.00% | 4.56% |
| MXFP4 | 51.20 / 43.33 | 63.97 / 61.36 | 15.38% | 4.08% |

M=4/8 control calls remain bit-identical and within 0.4% timing variation. AIU zero-fill, dense bank layout and expert-first grid trials were rejected because complete-call results regressed. Fixed-vLLM0.24 empty acceptance and core-benchmark PPU coverage remain pending; no new acceptance speedups are claimed.

| Component | Tested version |
| --- | --- |
| Accelerator | T-Head Zhenwu810E,96GB, single-card operator testing |
| Runtime | SAIL SDK2.2; existing UMD driver mode, no device/driver configuration changes |
| Driver status at integration |1.3.2-d7f5a2 |
| PyTorch |2.13.0 PPU2.2 build, common to both environments |
| Native compiler | Vendor Triton3.7.0 PPU2.2 build |
| Candidate compiler | FlagTree0.7.0+ppu.git1a96a36a, Triton3.6.0 |
| vLLM/SAIL | Versions and commits listed in Part2 |

The validated optimization snapshot has **476 passed,178 skipped,0 failed**,654 unique cases over eight disjoint shards. Tests cover quantized formats, both row sizes, layouts, signed-zero/Inf/NaN handling, current weight/scale reads under inference and CUDA graphs, output aliases, dense workspace boundaries and existing FP16/general paths. An INT8 decoder proof covers67,108,864 decoded values; the packed decode tests cover all codes and BF16/E8 scale encodings. No numerical tolerance is relaxed.

Publication integration verification: **476 passed,178 skipped,0 failed after upstream/fork synchronization**,654 unique cases; tested code `54bf55beaf4f29d66e930c9b4a85bb74dc9b53a9`. Source/test hashes and eight-shard results are in [publication_validation.json](publication_validation.json).

Pre-commit,84 CI helper tests,10 workflow-policy tests, registry/export/marker checks and the no-Torch-compute audit are run before publication. `tools/check_ci_pins.py` is absent from the current repository and is not invoked by its actual workflow. The canonical core benchmark entry check is recorded separately: **0 passed,4 skipped**, exit0. The existing device gates require NVIDIA Hopper/Hygon and, for two entries, Moore Threads; they exclude this PPU environment. It does not replace the vendor-factory measurements above. Fixed-vLLM0.24 empty-mode acceptance for this checkpoint remains pending. Full model serving, other accelerators and a complete FP16 performance distribution are not validated.

Candidate environment variables include `UMD_PLATFORM_TYPE=1`, `HGGC_DRIVER_CANDIDATE=UMD`, `DNN_VENDOR=thead`, `VLLM_PLUGINS=ppu`, both selected-device visibility variables, and the existing host-driver/tool paths. Use an idle mapped device; launchers enforce UUID/context ownership. The recorded container mapping is ordinal0..5 to host physical0,10,11,12,13,14.

Functional reproduction from the repository root in the existing candidate environment (visibility0 is an example; select an idle mapped device):

```bash
export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
export DNN_VENDOR=thead VLLM_PLUGINS=ppu
export CUDA_VISIBLE_DEVICES=0 HGGC_VISIBLE_DEVICES=0
export PATH="/opt/host-tools:$PATH"
export LD_LIBRARY_PATH="/opt/host-driver:$LD_LIBRARY_PATH"
export PYTHONPATH="src:/work/deps:.:/work"
/opt/candidate031/bin/python -m pytest -q --quick \
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

The matched vendor measurement uses the frozen external test-host harness, which keeps native Triton3.7 and candidate FlagTree3.6 imports in separate processes. The harness, temporary shards and device scheduling stay outside the repository benchmark. Existing test-host directories are `/tmp/pr752-vendor-baseline-20261116/opt-20261116/phase26-int8-sparse` for current INT8 and `phase23-sparse-bounded` for W4. `run_pair.py`, `worker_diagnostic.py`, `inputs.py`, `native_w4.py`, raw JSON and monitors are retained there. On that host, choose idle physical/ordinal values and unused job names, then run each disjoint shard0..2:

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
| phase26-evidence.tgz |`09f989cac571f01863f7747f2de76e20c9675c65aa114ef3e6f9faa233f0fc94`|91|Current source/tests,654 functional cases, full INT8 A/B,53 native attempts, failed M56 tensors, context interruption and legitimate retry |
| phase23-evidence.tgz |`d08e5e05cc514b13a216a32be87d4605ae1de6bf7a609a225fdbd52fd8dcb439`|106|Validated W4 checkpoint and106 native points |
| phase24-28-research.tgz |`a57899604c1bc5d4b477dad22280997f68ffa118b44fd02c4cd0c5613ee97e7c`|478|Stage/resource profiles and excluded experiments; diagnostic data are not used as formal speedups |

Later AIU source-row cropping and constant-argument experiments are not part of the submitted implementation. Their mixed or negligible complete-call results did not replace this validated checkpoint.
