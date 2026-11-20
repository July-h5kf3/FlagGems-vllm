# PPU FP8 expert-window checkpoint

The M=16384, E=256, K=4096, N=256, topk=6 path starts gate and up projection after decoding the first half of the experts. A higher-priority auxiliary stream prepares the next half in the other bank or in an unused slice of the current bank. The upper-projection grid uses the GPU-generated range of routed rows for each half. Each CTA processes additional tiles when the range exceeds the launch grid.

The comparison uses SAIL 0.33.0rc1, vLLM 0.33.1.dev0 and DeepGEMM 0f9756f BF16 MoE on PPU-ZW810E. All 53 DeepSeek-V4-Flash shapes passed the original native and FP32 gates, covering 10,105 calls. The measured range was **0.948345×–2.195087×**. The count-weighted mean of per-shape speedups was **1.303288×** and the ratio of total native latency to total implementation latency was **1.107439×**. [Per-shape measurements](nov20-fp8-expert-windows.csv)

| Shape and run | Physical device | Native (ms) | Implementation (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| M=2048, full shape sweep | 10 | 1.393593 | 1.448921 | 0.961815× |
| M=16384, full shape sweep | 11 | 7.151568 | 7.541104 | 0.948345× |
| M=16384, isolated comparison | 10 | 7.158583 | 7.522302 | 0.951648× |
| M=16384, isolated comparison | 12 | 7.157925 | 7.528429 | 0.950786× |

The full-sweep M=16384 measurement improved from 0.943126× and 7.587277 ms at the previous checkpoint. It remains about 13 µs above the 0.95× threshold for its measured native latency.

All 16 CUDA Graph regression cases passed. A separate stress check routed all 98,304 routes to a single expert in either half, updated the down scales during replay and compared eager and replayed outputs bit for bit with the previous checkpoint, with and without input-side router weighting. Peak temporary allocation was 1,930,109,952 bytes in eager execution and 1,930,110,976 bytes during CUDA Graph capture.

The separate vLLM 0.24.0+empty environment skipped four core benchmark cases at the PPU device gates, producing no acceptance speedup.

Measured operator SHA-256 `eab3a672abfd884137b744a65039e608bd9a2cb2e75d0ce25537798cf965db07`.
