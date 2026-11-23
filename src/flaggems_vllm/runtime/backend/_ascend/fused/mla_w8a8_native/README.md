# Ascend A2 dense MLA window stages

The public `flash_mla_with_kvcache_w8a8` entry uses ordinary Triton for short page tables and for every explicit `num_splits` value (1, 2, 4, 8 or 16). The split-KV path reads the caller's page table, computes INT8 content QK plus BF16 RoPE QK, and encodes the scale-weighted probabilities as high and residual INT8 digits for PV. Multiple KV partitions are combined using their FP32 maxima and denominators. This follows the paged INT8 split-KV algorithm used by the MetaX and PPU implementations. [MetaX INT8 PV](https://github.com/July-h5kf3/FlagGems-vllm/blob/11365a80fe172e94156f9d277700bca2ec7fc18e/src/flaggems_vllm/runtime/backend/_metax/ops/flash_mla_with_kvcache_w8a8.py#L140-L180) · [PPU decode and split selection](https://github.com/July-h5kf3/FlagGems-vllm/blob/ed89e643aaa822acb79c2949df6372602613047d/src/flaggems_vllm/runtime/backend/_thead/ops/flash_mla_with_kvcache_w8a8.py#L871-L1007)

Automatic dispatch uses the native stages for page-table capacities above six pages. Two Cube stages share each six-page latent tile in L1 across QK and PV. They load discontiguous ND64 pages into the required NZ layout, transfer operands to L0 in each product's orientation, and use Fixpipe and hardware events to publish results and release buffers. The three Vector stages apply the causal mask and per-token scales, perform online softmax and signed INT8 packing, and accumulate each window's PV contribution. Their explicit DMA fences allow the previous window's PV transfer to overlap the current score tile. The TLE kernel owns page traversal, work assignment and synchronization between Cube and Vector. Only this window path registers the bitcode entry points. [TLE window schedule](../mla_w8a8_window.py) · [Portable Triton kernels](../mla_w8a8.py)

On an Ascend 910B4-1 with CANN 9.2.1, an earlier four-page TLE implementation used ordinary Triton dots and vector operations on the same input tensors. The table gives full eager-call medians over ten alternating samples per implementation, with B=128 and ordered physical pages.

| Heads | Query tokens | Mean KV length | Lengths | Four-page TLE | Native window |
|---:|---:|---:|---|---:|---:|
| 16 | 1 | 4096 | Uniform | 0.886 ms | 0.711 ms |
| 64 | 2 | 8192 | Ragged | 5.873 ms | 2.348 ms |
| 128 | 1 | 16384 | Uniform | 10.185 ms | 3.716 ms |

These measurements include grouping, scheduling and probability encoding as well as the native instructions. Both implementations passed the sampled 0.15 O/LSE checks on these inputs. To compare the two paths shipped in this package, reuse the same tensor objects and time the public call with `num_splits=None` and `num_splits=4`. The explicit split path uses two probability digits, while the native window uses one; concentrated attention can expose a measurable accuracy difference.

The M16/M32/M64 variants share the Cube and Vector sources in this directory. To rebuild the six bitcodes on an AArch64 CANN development installation, run `CANN_SDK="$ASCEND_HOME_PATH" bash build.sh`. The native helpers are distributed under the [CANN Open Software License](LICENSE).
