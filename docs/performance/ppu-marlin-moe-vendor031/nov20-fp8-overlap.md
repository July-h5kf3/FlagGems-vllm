# PPU FP8 large-batch checkpoint

For eligible FP8 W8A16 batches, the T-Head backend prepares expert routes with GPU histograms and a prefix scan. Two BF16 weight banks let the current grouped GEMM run while the next projection's weights decode on a second stream. Both capture and eager execution read the current routes and scales. The path is gated on at least 8192 tokens, 128 or 256 experts, K4096, N256, topk4–8, contiguous activations and routing inputs, and a 2 GiB temporary workspace limit.

The baseline is SAIL 0.33.0rc1 with vLLM 0.33.1.dev0 and DeepGEMM 0f9756f BF16 MoE on PPU-ZW810E. Each row uses three rounds of alternating CUDA Graph measurements on the same exclusive device, identical inputs and weights, and the existing native and FP32 precision checks. All 53 DeepSeek-V4-Flash shapes passed, representing 10,105 calls. Speedup ranged from **0.943126× to 2.191583×**. The count-weighted mean of the per-shape speedups was **1.302795×**, using the same calculation as earlier checkpoints. Summing the latencies over those calls gives a native-to-candidate ratio of **1.104073×**. [Per-shape measurements](nov20-fp8-overlap.csv)

| Tokens | Physical device | Native (ms) | Candidate (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 0 | 1.391403 | 1.449613 | 0.959845× |
| 16384 | 11 | 7.155759 | 7.587277 | 0.943126× |
| 16384, independent repeat | 10 | 7.153948 | 7.563652 | 0.945833× |

M=16384 improved from 0.934683× at `4bd913f4`. Using the full-run native measurement, the candidate needs to reach approximately 7.532 ms for 0.95×. The measured candidate still needs to save about 55 µs on that device.

At M=16384, peak temporary allocation was 1,930,109,952 bytes in eager execution and 1,930,110,976 bytes during CUDA Graph capture. Replaying after in-place route and scale updates produced the same output as eager execution. The existing FP8 CUDA Graph regression passed all 14 cases, including the new M=8192 case on the overlapping path.

The separate vLLM 0.24.0+empty acceptance run skipped the four core benchmarks because their platform gates exclude PPU. It produced no acceptance speedup.

Measured operator SHA-256 `e11d30341475d2a775b800c2a5014465967b09451e8c098b9e6fa5712e7c3107`.
