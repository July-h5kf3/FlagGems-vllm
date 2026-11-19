# PR #1306 V32 performance checkpoint

This checkpoint records the latest verified results for `candidate-v32-affine-loader` (R51), based on PR head `7c4b176717f508c77f7801197cf86ac028017f2b`. The exact R51 source and incremental patch remain on the validation host. This branch contains the benchmark evidence and recovery identifiers.

## Performance

All 25 GLM-5.2 shapes passed comparison against an FP32 reference. On MTT S5000, the count-weighted speedup over MATE 0.2.7 was **1.2143×** across **3,822 calls**. The same-round R50 control measured **1.2128×**. The 20 large shapes all improved in paired measurements, with a count-weighted **1.0052×** over R50. The full-workload target is **1.3×**.

MATE ran in vLLM-MUSA 0.28.0 with image digest `sha256:5c9c48d53febe7443e100c5554a1009f19294d6365a0dd97cfbe582a3203eac2`. Each shape used the same physical card for the candidate, control, and vendor measurement. Candidate and control used four alternating graph-timing pairs. `paired` is the median of those pairwise ratios. Vendor speedups use median latencies and are weighted by the original call counts. The `mate_source` column identifies the vendor measurement round.

[Per-shape measurements](v32-affine-loader.csv) · [Weighted summary](summary.json)

## Correctness and source recovery

R51 moved the two affine scale-bound sums into the KV loader and retained the existing precision fallback for rejected rows. SQ=4033 passed 384 BF16 and 384 FP16 graph replays with mutable inputs. The forced generic SQ=1280 FP16 path passed 256 replays. A byte-identical source snapshot passed 640 threshold replays per dtype. The private and stock FlagTree test suites passed 17 cases each, and the V32 core benchmark recorded three shapes.

The validation artifacts are in the R51 directory `mthreads-pr1306-v32-opt-20261118-r51`. Recover `candidate-v32-affine-loader`, `v32-affine-loader-increment.patch`, and `affine-checksums.txt` together. The increment was verified against the R50 candidate and reconstructs the R51 pipeline byte for byte. R52 wait experiments are awaiting on-device verification.
