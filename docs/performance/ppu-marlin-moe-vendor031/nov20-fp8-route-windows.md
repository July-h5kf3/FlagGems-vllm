# PPU FP8 route-window checkpoint

For M=16384, E=256, K=4096, N=256 and topk=6, the route alignment keeps a full histogram and prefix sum. It scatters the first expert window's routes, then records a CUDA event. The first gate GEMM can begin after that event and its weight decode finish. The auxiliary stream scatters the second window's routes while the first GEMM runs. The second gate GEMM waits for its routes and decoded weights.

The comparison uses SAIL 0.33.0rc1, vLLM 0.33.1.dev0 and DeepGEMM 0f9756f BF16 MoE on PPU-ZW810E. The complete repeat passed the original numerical checks on all 53 DeepSeek-V4-Flash shapes, covering 10,105 calls. The range was **0.951380×–2.197419×**. The call-count-weighted mean of per-shape speedups was **1.303064×**. The ratio of total native latency to total implementation latency was **1.109266×**. [Both full sweeps](nov20-fp8-route-windows.csv)

| Shape and run | Physical device | Native (ms) | Implementation (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| M=2048, complete repeat | 14 | 1.392103 | 1.448728 | 0.960914× |
| M=16384, first sweep | 11 | 7.154874 | 7.520010 | 0.951445× |
| M=16384, complete repeat | 13 | 7.153898 | 7.519497 | 0.951380× |
| M=16384, isolated comparison | 0 | 7.160348 | 7.520284 | 0.952138× |
| M=16384, isolated comparison | 10 | 7.159299 | 7.520125 | 0.952019× |

The first sweep's M=40 native result had a 6.347% peak relative error against FP32, above the 2% limit. The implementation's peak error against FP32 was 0.501%. The CSV retains the first sweep and its rejected row. The second sweep ran the complete shape set with the same implementation, weights and inputs.

All 16 CUDA Graph tests and 128 quick alignment tests passed. Replays with all 98,304 routes assigned to a single expert in either window, changes to the down scale during replay, and both input-side router weighting modes matched the previous checkpoint bit for bit. Peak temporary allocation was 1,930,109,952 bytes in eager execution and 1,930,110,976 bytes during CUDA Graph capture.

The separate vLLM 0.24.0+empty environment skipped four core benchmark cases at their PPU device gates and produced no acceptance speedup.

Measured SHA-256 for the operator is `7742c2699fdefdb30cba43b639e7811a117d40aaae35bc347d9bc37e692a8fe2`. The shared alignment module is `54ee615d3301ce9afd5a11c79121a5c7be4f1bebb2736fbd4942d5b563a601ab`.
