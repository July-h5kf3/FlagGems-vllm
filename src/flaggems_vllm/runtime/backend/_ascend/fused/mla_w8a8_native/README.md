# Ascend A2 dense MLA window stages

The public `flash_mla_with_kvcache_w8a8` entry uses ordinary Triton for short page tables and for every explicit `num_splits` value (1, 2, 4, 8 or 16). The split-KV path reads the caller's page table, computes INT8 content QK plus BF16 RoPE QK, and encodes the scale-weighted probabilities as high and residual INT8 digits for PV. Multiple KV partitions are combined using their FP32 maxima and denominators. This follows the paged INT8 split-KV algorithm used by the MetaX and PPU implementations. [MetaX INT8 PV](https://github.com/July-h5kf3/FlagGems-vllm/blob/11365a80fe172e94156f9d277700bca2ec7fc18e/src/flaggems_vllm/runtime/backend/_metax/ops/flash_mla_with_kvcache_w8a8.py#L140-L180) · [PPU decode and split selection](https://github.com/July-h5kf3/FlagGems-vllm/blob/ed89e643aaa822acb79c2949df6372602613047d/src/flaggems_vllm/runtime/backend/_thead/ops/flash_mla_with_kvcache_w8a8.py#L871-L1007)

Automatic dispatch uses one mixed AIC/AIV kernel for page-table capacities above six pages. Its TLE schedule traverses the page table, assigns work and exchanges completion events. Four registered device functions serve all three row tilings (M16/M32/M64). Each is inlined into that kernel. [TLE window schedule](../mla_w8a8_window.py) · [Portable Triton kernels](../mla_w8a8.py)

| Device function | Why it uses native instructions |
|---|---|
| `mla_window_qk` | The Cube loads discontiguous ND64 pages into NZ tiles in L1 and publishes the two QK results through Fixpipe. The latent tile stays resident for PV. |
| `mla_window_probability` | The Vector unit computes online softmax, applies token scales and packs signed INT8 probabilities in UB. Two calls to this same function bracket the previous PV completion event so score and PV transfers can overlap. |
| `mla_window_pv` | The Cube reads the resident L1 tile in the PV orientation and accumulates INT32 products. An event releases each tile for the next prefetch. |
| `mla_window_update` | The Vector unit consumes the published PV, rescales its FP32 running numerator and writes the final O and LSE when the request finishes. |

The row count and probability phase are compile-time operands. Short tables and explicitly requested split counts use the Triton path with zero custom registrations in a fresh process.

On an Ascend 910B4-1 with CANN 9.2.1, an earlier four-page TLE implementation used ordinary Triton dots and vector operations on the same input tensors. The table gives full eager-call medians over ten alternating samples per implementation, with B=128 and ordered physical pages.

| Heads | Query tokens | Mean KV length | Lengths | Four-page TLE | Native window |
|---:|---:|---:|---|---:|---:|
| 16 | 1 | 4096 | Uniform | 0.886 ms | 0.711 ms |
| 64 | 2 | 8192 | Ragged | 5.873 ms | 2.348 ms |
| 128 | 1 | 16384 | Uniform | 10.185 ms | 3.716 ms |

These measurements include grouping, scheduling and probability encoding as well as the native instructions. Both implementations passed the sampled 0.15 O/LSE checks on these inputs. To compare the two paths shipped in this package, reuse the same tensor objects and time the public call with `num_splits=None` and `num_splits=4`. The explicit split path uses two probability digits, while the native window uses one; concentrated attention can expose a measurable accuracy difference.

The wheel contains two bitcodes, `window_cube.bc` and `window_vec.bc`. To rebuild both on an AArch64 CANN development installation, run `CANN_SDK="$ASCEND_HOME_PATH" bash build.sh`. The native helpers are distributed under the [CANN Open Software License](LICENSE).
