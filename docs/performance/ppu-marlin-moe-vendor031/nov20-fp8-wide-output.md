# PPU FP8 wide-tile output checkpoint

The 64-route tile reads the first index in its last 16-route chunk. When that index is padding, both projections use the existing FP8 decoder and BF16 MMA for the first 48 routes only. The route placement and activation packing remain aligned at 64 rows.

For the K256 down projection, the tile reuses its activation shared buffer after the last MMA. It deposits FP32 fragments using a row-dependent XOR on the channel index. The four lanes that produce the same channel for different route rows address distinct banks. Reloading by route and channel gives contiguous output stores and one router-weight load per row. Each result still rounds to BF16 after the router multiplication, before ordered route reduction. The TLE shared allocation and local pointer operations reuse the existing 64 KiB buffer. The AIU and MMA instructions are unchanged.

## Performance

The reference is SAIL 0.33.0rc1 and vLLM 0.33.1.dev0 native BF16 MoE on PPU-ZW810E. The native API, FP32 oracle, input generation, original DeepSeek-V4-Flash p32768d1024 counts and CUDA Graph timing protocol are unchanged. [Native API and protocol](nov20-fp8-routing.md) · [53-shape results](nov20-fp8-wide-output.csv).

| M=2048 | Native ms | Implementation ms | Speedup |
| --- | ---: | ---: | ---: |
| Before, 6969a2b6 | 1.391003 | 1.618424 | 0.859480× |
| Tail and shared output staging | 1.391625 | 1.449420 | 0.960125× |
| Repeat on another PPU | 1.391691 | 1.449983 | 0.959798× |

The full run produced 52 valid shapes and a count of 9,933. The count-weighted speedup is 1.302128×. The range is 0.934985× at M=16384 through 2.194896× at M=2. The native M=64 result exceeded the 2% max-relative-error limit at 3.46%. The table retains the failing row without a speedup.

## Validation

The FP8 CUDA Graph suite passed 12 tests. The M=2048 output is bit-identical to the old implementation. The test environment used SDK 2.2.0-663c22, vendor Torch 2.13.0 and FlagTree 0.7.0+ppu.git1a96a36a. The fixed vLLM 0.24.0+empty core cases remain gated on PPU.
