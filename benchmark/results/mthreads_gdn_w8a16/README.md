# MThreads GDN development checkpoint

This is an incomplete performance checkpoint for review. The 1.3x weighted target has not been reached. No experimental kernel with a measured regression has been integrated.

## Implementation and interface

The MThreads backend implements FP8 E4M3FN persistent GDN state with FP32 per-value-channel scales and FP32 recurrence. Production computation uses Triton; Torch is limited to allocation, metadata and views/copies. MATE is used only as a test/benchmark baseline.

The existing `gdn_w8a16` interface supports decode and prefill. The backend also exports `gated_delta_rule_decode_w8a16_fp8` for the vendor decode/MTP interface and `chunk_gated_delta_rule_w8a16_fp8` for MP31 prefill. Native logical state layout is VK `[pool,HV,V,K]`. Coverage includes padding indices, slot zero, independent strided state pools, optional initialization flags, supplied output, quantized MTP histories and read-only MTP state. Prefill returns output, final FP8 state and final FP32 scales; the latter two are None when final state is disabled.

TLE is optional and guarded by compiler capability. Its small-batch T=1 path uses explicit blocked/sliced layouts. Other paths use ordinary Triton. Prefill retains FP32 recurrence between tokens and quantizes at the call boundary; MP31 normalization stores normalized Q/K back to their activation dtype. Kernels use fixed, externally evaluated configurations because live autotuning would mutate persistent state. No NVIDIA tuning configuration is changed.

The operator is forward-only, supports BF16/FP16 activations, and does not promise training/autograd. MP31 requires chunk size 64 and does not support state checkpoints. The full JSON below measures BF16 activations against the vendor's FP32-state path, not BF16 persistent state.

## Baseline and measurement

The baseline source is vendor vLLM `v0.32.0-hybrid-cache-fc928696-pd-20261105`, vendor revision `fc9286968d77c6dee46f75e6e8d9fed790204aca`, upstream revision `5df63a7203cdb9328d824a27b682e77c7b9928be`. Decode is `mate.gdn_decode.gated_delta_rule_decode`; prefill is `mate.gdn_prefill.chunk_gated_delta_rule` plus the original fused gate helper.

Acceptance was run in an isolated official vLLM 0.24.0 empty environment with Torch 2.11.0.post2+musa5.2.0, FlagTree 0.7.0+mthreads / Triton 3.6.0, MATE 0.3.0+mutlassdev, MTT S5000 MP31 and driver 3.3.5-server. Vendor vLLM packages/plugins and vLLM shared objects were absent. The user approved loading the unmodified original vendor gate file independently for this environment. Its Apache-2.0 source and SHA256 check are included under `vendor-source-review/`; `empty_gating_reference_probe.py` binds that exact function only in benchmark processes. It does not install `vllm_musa` or change the gate algorithm.

`musa_gpu_activity_union_v3_drained` measures the union of GPU kernel/copy/set activity intervals inside each sample annotation. State and Q/K restoration, a 256 MiB cache clear and synchronization occur outside samples; an additional unmeasured activity drains the last profiler event. The sampler rejects missing/incomplete activities. This is device activity duration, not end-to-end model throughput or eager Python dispatch latency. Every accepted shard had unchanged source hashes and no detected foreign GPU contention.

The 110 unique shapes and invocation counts are frozen Qwen3.5/3.6-35B-A3B TP1 captures. Sequence boundaries were not captured and are reconstructed uniformly. This is not evidence of coverage of all models or full-model quality.

| Scene | Unique shapes | Sum(count) | Weighted native / candidate |
| --- | ---: | ---: | ---: |
| Packed decode | 33 | 1,209,300 | 0.953196x |
| Separate QKV update | 25 | 3,690 | 1.006199x |
| Prefill | 52 | 4,290 | 0.030387x |
| All | 110 | 1,217,280 | 0.950105x |

Aggregation is `sum(count_i * speedup_i) / sum(count_i)` over all 110 records, including regressions. `full-benchmark.json` contains each positive baseline/candidate latency and speedup. These values describe the final source hashes in `summary.json`, after synchronization to upstream 80ab403. The earlier 1.001669x checkpoint is retained separately in summary.json for provenance and is not the final-source result. Long prefill is substantially slower than native and requires further optimization.

## Validation and reproduction

After synchronization to upstream 80ab403, the isolated empty environment passes all 55 MThreads quick accuracy cases with zero failures; 92 upstream cases for other backends are skipped by their vendor gates. No MThreads case is skipped. The three-scene core benchmark and the full 110-shape benchmark pass with complete JSON details. Initial state is exactly dequantized before the vendor comparison. Prefill final state is checked against an independent FP32 per-token reference; the channel-normalized FP8 error bound remains `16/448 + 5e-4`. It is not loosened to absorb native BF16 chunk error.

Required runtime dependencies are the device's Torch/MUSA and MATE packages, a compatible FlagTree, this repository, and official vLLM 0.24.0 built with `VLLM_TARGET_DEVICE=empty`. The vendor `vllm_musa` package is not needed for the commands below. Select a fresh idle card with `mthreads-gmi` before setting `MUSA_VISIBLE_DEVICES`.

```bash
export PYTHONPATH="$PWD/benchmark/results/mthreads_gdn_w8a16:$PWD/src:$PWD:$PYTHONPATH"
export VLLM_TARGET_DEVICE=empty
export VLLM_PLUGINS=
python -m pytest tests/test_gdn_w8a16.py --quick -m gdn_w8a16 --record json --output /tmp/gdn-accuracy.json -q
python -m pytest -p empty_gating_reference_probe benchmark/test_gdn_w8a16.py -m gdn_w8a16 --level core --warmup 1 --iter 5 --record json --output /tmp/gdn-core.json -q
python -m pytest -p empty_gating_reference_probe benchmark/test_gdn_w8a16.py -m gdn_w8a16 --level comprehensive --shape_file benchmark/gdn_w8a16_mthreads_shapes.yaml --warmup 1 --iter 5 --record json --output /tmp/gdn-full.json -q
```

The archived run used external three-device sharding and serialized MUPTI sampling. The MThreads benchmark class itself is serial; repository `--parallel` integration remains work for a later revision.

## Optimization status and limits

Explicit TLE layouts are retained for the small-batch path. Report-only trials evaluated FP8 software/packed conversion, WMMA/shared staging, warp specialization, state liveness, V-row loops and alternative cache layouts. None established the required full-workload improvement. PC-sampling configuration was accepted through the MUSA-specific SDK structure, but explicit start returned `MUPTI_ERROR_NOT_SUPPORTED`; automatic sampling produced no data, so no PC-derived bottleneck claim is made.

A report-controller bug left two historical experimental sweep variables on the host instead of forwarding them into the container. Those historical configuration labels were invalidated and the sweeps rerun with worker-side configuration checks. The production kernel and the archived 110-shape acceptance run do not use these tuning variables and are unaffected.

This PR saves the implementation and verified results for review. Remaining work includes the 1.3x weighted performance target, long-prefill optimization and parallel benchmark integration. The default benchmark still imports the vendor gate entry point; empty-mode validation uses the explicitly approved standalone-source plugin above.

Post-merge validation: all pinned pre-commit hooks pass on the complete source tree, 83 CI helper tests pass, 10 workflow-policy tests pass, and production no-Torch-compute audit reports zero violations. Export, operator registry, marker, ATen-name and performance-reference checks pass. Upstream removed `tools/check_ci_pins.py`; current workflow-policy checks were run instead. Other-device hardware tests were not run.
