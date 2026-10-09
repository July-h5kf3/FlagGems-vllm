# Hopper dense FP8 MLA execution contract

NoPE is FP8 E4M3 and RoPE is BF16. Both are divided by their FP32 query/token scale, matching the public per-token interface. All descriptor inputs must be contiguous and 16-byte aligned. Unsupported layouts are rejected before a descriptor is created; descriptor flattening uses view only.

The one-shot operator sizes its schedule from page-table capacity and reads current lengths on device. Prepared calls accept host initial/maximum length certificates to select the measured schedule without reading GPU tensors on the CPU. Each certificate must match the GPU data supplied by the caller. Native metadata kernels initialize integer plans, pad/copy the page table and perform length updates.

## Fixed-layout tuning exemption

The main kernel's 64-head/64-token tiles, two worker groups, one stage and base four-warp launch are structural parts of its inline PTX register/shared-memory permutations and named-barrier ABI. The compiled launch/capture path binds these layouts before replay. Changing these values through libtuner is not a valid candidate space. They retain the measured original schedule and are explicitly exempt from live autotuning. The combine widths/warp counts use the existing frozen compiled merge schedule to preserve the prepared graph.

Capacity-plan blocks are fixed by the number of splits that must be initialized, not independent performance knobs. Page-table copy is tunable through flash_mla_fp8_copy_table with BLOCK128/256 and a BATCH/COLS/PADDED key.

Only the NVIDIA Hopper backend registers this implementation. Generic compatibility functions delegate to the registered operator or report unsupported. Production code performs metadata reads, uninitialized allocation and no-copy views; it does not use Torch compute/copy/cast.

The historical B128/H128 workloads with uniform 256–4096-token power-of-two capacities use
one direct-output CTA per 64-head group. The existing single-page split route
created redundant CTAs and merge traffic despite an already full grid.
The 2048/4096-token cases likewise benefit from keeping each row in one CTA.
Measured grain selection retains all other host routes and kernel arithmetic.

Prepared handles require explicit host length certificates and contiguous,
16-byte-aligned descriptor inputs. The query and cache storage remain bound;
in-place updates are observed because descriptors use true views. One-shot
calls derive capacity from the page-table width and mask using device lengths,
without reading lengths on the CPU. Lengths must be within that table capacity
and the 33280-token implementation limit.
