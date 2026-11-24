# Ascend A2 dense MLA window stages

The public `flash_mla_with_kvcache_w8a8` entry uses ordinary Triton for short page tables and for every explicit `num_splits` value (1, 2, 4, 8 or 16). Both the split-KV path and the native mixed AIC/AIV window path read the caller's page table, compute INT8 content QK plus BF16 RoPE QK, and encode the scale-weighted probabilities as high and residual INT8 digits (`high + low / 127`) for PV before normalizing by the FP32 online softmax denominator. This matches the two-digit INT8 PV formulation in [attention.py](../attention.py) while preserving accuracy under concentrated attention and heterogeneous page-token scales.

Automatic dispatch uses one mixed AIC/AIV kernel for page-table capacities above six pages on Ascend 910B. Its TLE schedule sorts variable-length batches in snake order across all row tilings (M16/M32/M64), traverses six-page (384-token) windows and exchanges completion events. Four registered device functions are inlined into that kernel. [TLE window schedule](../mla_w8a8_window.py) · [Portable Triton kernels](../mla_w8a8.py)

| Device function | Why it uses native instructions |
|---|---|
| `mla_window_qk` | The Cube loads discontiguous ND64 pages into NZ tiles in L1 and publishes the two QK results through Fixpipe. The latent tile stays resident for PV. |
| `mla_window_probability` | The Vector unit runs hardware `vexp` online softmax, applies token scales and packs high and residual signed INT8 probability digits in UB while overlapping prior PV preloads. |
| `mla_window_pv` | The Cube reuses the transposed L0B value tile across both probability digits, accumulates INT32 products on ping-ponged L0C banks and releases L1 tiles for the next prefetch. |
| `mla_window_update` | The Vector unit overlaps residual PV DMA with primary PV accumulation, rescales its FP32 running numerator and writes the final O and LSE when the request finishes. |

The row count and probability phase are compile-time operands, while batch size, task count and page-table width remain runtime scalars so dynamic decode batches do not trigger recompilation. Short tables and explicitly requested split counts use the Triton path with zero custom registrations in a fresh process.

The wheel contains two bitcodes, `window_cube.bc` and `window_vec.bc`. To rebuild both on an AArch64 CANN development installation, run `CANN_SDK="$ASCEND_HOME_PATH" bash build.sh`. The native helpers are distributed under the [CANN Open Software License](LICENSE).
