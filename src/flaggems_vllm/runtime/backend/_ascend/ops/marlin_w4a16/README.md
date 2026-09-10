# Ascend Marlin W4A16 implementation

The existing BF16 / packed INT4 dispatch and shape restrictions are enforced by
`fused_marlin_moe_w4a16_int4.py`. Unsupported inputs raise; this implementation
has no PyTorch compute fallback.

The required FlagTree custom primitives are:

| API | Device operation |
| --- | --- |
| `compare_scalar` | FP32 equality to a packed uint16 mask |
| `gather_mask` | Stable FP32 compaction and selected count |
| `cast_int4_to_fp16` | Signed, low-nibble-first INT4 unpacking |
| `cube_begin`, `cube_end` | CUBE-local `PipeBarrier<PIPE_ALL>` |

Routing loops, count accumulation, GM transfers, weight scaling, task scheduling,
K loops, matrix multiplication (`tl.dot`), activation and output reduction are
TLE code. The boundary primitives do not synchronize different cores; explicit
TLE `sync_block_set/wait` implements the two-stage Vector/Cube handshake.

Dequantization always multiplies in FP32 before converting to BF16. The old
FP16/FP32 branch required a scalar reduction of scale flags. With a visible TLE
Cube, that reduction triggered an unused Vector-to-Cube workspace transfer in
the tested compiler. Unconditional FP32 removes the triggering source pattern;
no MMA replacement or workspace-store deletion is performed by this package.
The prepared scale flags and existing launch parameters remain in the shared
interface to avoid changing weight preparation in this update.

Install FlagTree with the custom primitive registration and Ascend backend
compatibility from `Ascend/fused_marlin_moe_custom`. The operator directly calls
public `tle.dsa.ascend.raw` APIs; it does not compile AscendC, adapt the custom
ABI, rewrite IR, or modify compiler functions. CANN 9.0 compatibility belongs to
FlagTree and is selected there using the toolkit version. CANN 9.1+ uses the
native custom path, which has not been device-validated in this environment.

Ascend launch choices (K=128, two stages, shape-dependent tiles and merge policy)
are the existing measured settings. This Ascend-only update does not change the
NVIDIA backend or introduce NVIDIA autotune configuration.
