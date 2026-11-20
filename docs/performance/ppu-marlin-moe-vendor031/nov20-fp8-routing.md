# PPU FP8 W8A16 routing checkpoint

The first CTA of the existing FP8 scale-check and alignment launch now groups and pads the routes. The other CTAs check both projections' live BF16 scales for their assigned routes or experts. Both GEMMs consume the outputs after the launch completes. This removes the first scale check from the route grouping CTA's execution path. The change reuses the existing Triton grouping kernel and keeps the projection kernels, TLE instructions and FP32 accumulation unchanged.

## Performance

The reference is SAIL 0.33.0rc1 on PPU-ZW810E with vLLM 0.33.1.dev0 (4ba787ec) and DeepGEMM for SAIL (0f9756f). FP8 W8A16 uses the same activation precision and decoded BF16 weights in the native BF16 MoE because the vendor's W8 modes quantize activations. [SAIL release](https://github.com/vllm-project/vllm-sail/releases/tag/v0.33.0rc1), [installation requirements](https://github.com/vllm-project/vllm-sail/blob/v0.33.0rc1/docs/getting_started/installation.md), [DeepGEMM revision](https://github.com/t-head/DeepGEMM-for-sail/commit/0f9756f27305e97c0f4794f27ac2f2f19155b059).

The native entry point is vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernel.apply, configured with PPUDeepGemmExperts in BF16 mode. The timing excludes constructing the decoded weights.

~~~python
output = kernel.apply(
    hidden_states, gate_up_weight_bf16, down_weight_bf16,
    routing_weights, expert_ids, MoEActivation.SILU, 256, None, False,
)
~~~

All three rounds used the same 53 DeepSeek-V4-Flash p32768d1024 shapes and original call counts. Inputs and routes use synthetic data with seed 7 + M; FP8 weights use seed 7. Each latency is the median of three CUDA Graph measurements interleaved on the same card. The weighted ratio uses native latency divided by the candidate latency for each valid row. [Shape provenance](shape_provenance.json) · [All latency and accuracy rows](nov20-fp8-routing.csv).

| FP8 build | Weighted speedup | Per-shape range | Valid shapes | Valid count |
| --- | ---: | ---: | ---: | ---: |
| 6609de2e | 1.297999× | 0.608924–2.195749× | 53/53 | 10,105 |
| Dedicated routing CTA | 1.300418× | 0.607336–2.191596× | 53/53 | 10,105 |
| Dedicated routing CTA, repeat | 1.300004× | 0.608306–2.195905× | 52/53 | 9,933 |

The repeat's native BF16 result exceeded the 2% relative maximum error gate at M=56 (3.29%). The row is included with its status and errors. On the 52 shared valid shapes, the three rounds measure 1.297951×, 1.300402× and 1.300004×. Direct same-card comparison with 6609de2e produced bit-identical outputs for all 53 shapes and a weighted 1.001700× speedup. M=512 improved from 823.625 to 821.641 µs in that comparison.

## Validation

The native and optimized result comparison uses relative RMS below 0.01 and relative maximum error below 0.02. The native result is also checked against an FP32 accumulator reference. PPU functional tests passed 447 cases with 440 deselected, including Graph replay after replacing route IDs and scales. The runtime used SDK 2.2.0-663c22, vendor PyTorch 2.13.0 and FlagTree 0.7.0+ppu.git1a96a36a for the candidate, and vendor Triton 3.7.0 for the reference.

~~~bash
DNN_VENDOR=thead PYTHONPATH=src python -m pytest -q tests/test_fused_marlin_moe.py -k ppu
~~~

In the separate fixed vLLM 0.24.0+empty environment, all four repository core benchmark entries were skipped by device gating and the JSON had no latency or speedup. This environment also lacks vllm._C and vllm._C_stable_libtorch. The fixed-environment performance acceptance is still pending.
