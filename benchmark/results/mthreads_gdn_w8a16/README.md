# MThreads GDN validation results

All 110 frozen shapes meet the 0.85x speedup floor: minimum 0.850418x. Prefill reaches 1.142231x (target 0.9x), and the invocation-weighted overall result is 1.348908x (target 1.3x). These are GPU activity timings, not model throughput.

## Implementation and interface

The MThreads backend implements FP8 E4M3FN persistent GDN state with FP32 per-value-channel scales. State accumulation and per-channel scales use FP32; eligible prefill matrix operands use FP16/BF16. Production computation uses Triton; Torch is limited to allocation, metadata and views/copies. MATE is used only as a test/benchmark baseline.

The existing `gdn_w8a16` interface supports decode and prefill. The backend also exports `gated_delta_rule_decode_w8a16_fp8` for the vendor decode/MTP interface and `chunk_gated_delta_rule_w8a16_fp8` for MP31 prefill. Native logical state layout is VK `[pool,HV,V,K]`. Coverage includes padding indices, slot zero, independent strided state pools, optional initialization flags, supplied output, quantized MTP histories and read-only MTP state. Prefill returns output, final FP8 state and final FP32 scales; the latter two are None when final state is disabled.

On MP31 with FlagTree, normalized single-token decode at K=V=128 retains the packed FP32 register implementation. Decode TME state reads remain available through `FLAGGEMS_MTHREADS_GDN_TME=1`. The prefill raw-gate arguments `A_log`, `a`, `dt_bias`, and `b` now fuse gates with Q/K normalization and per-token value bounds. The long-window kernel waits for K/K products, then overwrites the consumed K operand slot with Q through TME; this avoids allocating a separate Q tile. Precomputed-gate calls remain supported and do not require TME.

The first 64 tokens include the dequantized initial state; later CTAs produce 32 outputs with 32 preceding tokens as history. Device checks cover input ranges, history decay and matrix residuals. The final state uses a compensated matrix update with a quantization-aware residual check. Rejected output heads or final-state tiles replay the FP32 recurrence. Single-sequence 33–128-token calls use a triangular matrix solve with compensated final-state accumulation. Large triangular powers trigger FP32 replay to avoid cancellation for nearly collinear keys and weak forget gates. Each V tile commits only after range and finite-value checks; rejected tiles preserve the original state for replay. Stock Triton and unsupported fast-path layouts retain the recurrence. Live state is never replayed for autotuning.

The operator is forward-only, supports BF16/FP16 activations, and does not promise training/autograd. MP31 requires chunk size 64 and does not support state checkpoints. The full JSON below measures BF16 activations against the vendor's FP32-state path, not BF16 persistent state.

## Baseline and measurement

The baseline source is vendor vLLM `v0.32.0-hybrid-cache-fc928696-pd-20261105`, vendor revision `fc9286968d77c6dee46f75e6e8d9fed790204aca`, upstream revision `5df63a7203cdb9328d824a27b682e77c7b9928be`. Decode is `mate.gdn_decode.gated_delta_rule_decode`; prefill is `mate.gdn_prefill.chunk_gated_delta_rule` plus the original fused gate helper.

Acceptance was run in an isolated official vLLM 0.24.0 empty environment with Torch 2.11.0.post2+musa5.2.0, FlagTree 0.7.0+mthreads / Triton 3.6.0, MATE 0.3.0+mutlassdev, MTT S5000 MP31 and driver 3.3.5-server. Vendor vLLM packages/plugins and vLLM shared objects were absent. The user approved loading the unmodified original vendor gate file independently for this environment. Its Apache-2.0 source and SHA256 check are included under `vendor-source-review/`; `empty_gating_reference_probe.py` binds that exact function only in benchmark processes. It does not install `vllm_musa` or change the gate algorithm.

`musa_gpu_activity_union_v3_drained` measures the union of GPU kernel/copy/set activity intervals inside each sample annotation. State and Q/K restoration, a 256 MiB cache clear and synchronization occur outside samples; an additional unmeasured activity drains the last profiler event. The sampler rejects missing/incomplete activities. This is device activity duration, not end-to-end model throughput or eager Python dispatch latency. Every accepted shard had unchanged source hashes and no detected foreign GPU contention.

The 110 unique shapes and invocation counts are frozen Qwen3.5/3.6-35B-A3B TP1 captures. Sequence boundaries were not captured and are reconstructed uniformly. This is not evidence of coverage of all models or full-model quality.

| Scene | Unique shapes | Sum(count) | Weighted native / candidate |
| --- | ---: | ---: | ---: |
| Packed decode | 33 | 1,209,300 | 1.349391x |
| Separate QKV update | 25 | 3,690 | 1.430779x |
| Prefill | 52 | 4,290 | 1.142231x |
| Overall | 110 | 1,217,280 | 1.348908x |

Aggregation is `sum(count_i * speedup_i) / sum(count_i)` over all 110 records. Shapes, counts, native reference code, tolerances and the GPU-activity timing algorithm are unchanged. The candidate prefill wrapper now calls the supported raw-gate API. The previous PR head (`65d914a7`) measured 0.933753x prefill, 1.307622x overall, and a 0.366798x minimum. Two preceding checks of the new kernels measured overall 1.327723x and 1.282981x, with minima 0.861079x and 0.867643x. A later five-sample check measured 1.259167x overall and a 0.840610x minimum. The latest source-matched run used 30 samples per operation through the existing --iter option, compared with 5 in preceding runs. Its minimum is close to the target and does not establish a stable per-run margin. This latest result is used for acceptance; short decode timings account for most weighted variation. `summary.json` retains the full-run summaries, including the additional guard validation.

## Validation and reproduction

The isolated empty environment passes all 86 MThreads quick accuracy cases with zero failures; 92 upstream cases for other backends are skipped by their vendor gates. No MThreads case is skipped. The three-scene core benchmark and the full 110-shape benchmark pass with complete JSON details. Initial state is exactly dequantized before the vendor comparison. Prefill final state is checked against an independent FP32 reference; the channel-normalized FP8 error bound remains `16/448 + 5e-4`. It is not loosened to absorb native BF16 chunk error.

Required runtime dependencies are the device's Torch/MUSA and MATE packages, a compatible FlagTree, this repository, and official vLLM 0.24.0 built with `VLLM_TARGET_DEVICE=empty`. The vendor `vllm_musa` package is not needed for the commands below. Select a fresh idle card with `mthreads-gmi` before setting `MUSA_VISIBLE_DEVICES`.

```bash
export PYTHONPATH="$PWD/benchmark/results/mthreads_gdn_w8a16:$PWD/src:$PWD:$PYTHONPATH"
export VLLM_TARGET_DEVICE=empty
export VLLM_PLUGINS=
python -m pytest tests/test_gdn_w8a16.py --quick -m gdn_w8a16 --record json --output /tmp/gdn-accuracy.json -q
python -m pytest -p empty_gating_reference_probe benchmark/test_gdn_w8a16.py -m gdn_w8a16 --level core --warmup 1 --iter 1 --record json --output /tmp/gdn-core.json -q
python -m pytest -p empty_gating_reference_probe benchmark/test_gdn_w8a16.py -m gdn_w8a16 --level comprehensive --shape_file benchmark/gdn_w8a16_mthreads_shapes.yaml --warmup 1 --iter 30 --record json --output /tmp/gdn-full.json -q
```

The full measurement ran on physical device 3. Core benchmark, default accuracy and TME accuracy used device 3; stock accuracy and the independent FP32 output comparison used device 4. All testing was confined to authorized devices 2, 3 and 4. Source hashes were unchanged during measurement and no foreign GPU allocation was observed on the measured devices.

## Optimization status and limits

The independent FP32 comparison covered 2,472,521,728 BF16 output elements across all 52 prefill shapes without rounding the reference. Maximum absolute error was 5.67790121e-05; no element exceeded 2e-4. Maximum final-state error normalized by channel amax was 0.0359854437, below the unchanged limit 0.0362143. An additional 80-case short-sequence experiment varied seeds, FP16/BF16 inputs, weak/strong gates and initial-state magnitudes; its maximum output error was 1.230724e-4. These are observations on bounded test inputs, not a universal absolute-error guarantee for arbitrary values. Tests cover partial and empty sequences, unused state slots, separate pools, raw gates and large-state FP32 replay.

TLE experiments included explicit shared layouts, SQMMA, TME buffer reuse and producer/consumer warp specialization. The working warp-specialized prefill prototype took 4.864 ms versus 1.435 ms native on 16383 tokens / 16 sequences; its 256 registers, 30 spills and 94,208 bytes of shared memory erased the expected benefit. Other variants exposed compiler or runtime failures. They remain report-only experiments. Production TME uses the installed compiler without patches, while decode TME remains optional.

The default path, opt-in TME path and stock Triton fallback each passed 86 MThreads quick accuracy cases. The suite includes large finite BF16 Q and a non-default scale that can overflow if multiplied in the wrong order. A separate FP32-output comparison on the six most frequent packed decode shapes found no element above 2e-4 absolute error (380,928 elements). Separate edge checks exercised identity, shuffled and negative indices, a noncontiguous FP8 state pool and a misaligned Q view. The latter two use the original implementation. PC-sampling configuration was accepted through the MUSA-specific SDK structure, but explicit start returned `MUPTI_ERROR_NOT_SUPPORTED`; automatic sampling produced no data, so no PC-derived bottleneck claim is made.

A report-controller bug left two historical experimental sweep variables on the host instead of forwarding them into the container. Those historical configuration labels were invalidated and the sweeps rerun with worker-side configuration checks. The production kernel and the archived 110-shape acceptance run do not use these tuning variables and are unaffected.

The frozen workload meets the per-shape floor in this 30-sample measurement, with little margin on the slowest shape. General inputs that trigger FP32 replay can be slower. The default native baseline still imports the vendor gate helper; empty-mode validation uses the approved standalone-source plugin above.
