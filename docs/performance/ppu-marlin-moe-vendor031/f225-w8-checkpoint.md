# F225 W8 Marlin MoE draft checkpoint

Operator commit `dff9cc8b6ced06246a34f32c0924108e99dc75d1`. The tested T-Head operator file has SHA-256 `f5b2d7e7c46066701b9cc69de541d5a0967681dc92bba7cc851f73d9d5d8dc0c`.

F225 limits short-route FP8 scale checks to routed experts, gathers BF16 activations in pairs on aligned inputs, and skips padding-only activation and SwiGLU staging tiles. The tiny FP8 gate uses 128-column tiles. Eligible high-density down projections load route IDs and weights earlier, and the grouped kernels use signed 32-bit offsets when the tensor bounds allow them. The large-batch FP8 down projection uses ordinary Triton `tl.load` for a full 4096-column block with K=256. The existing TLE asynchronous load covers the tail and other dense geometries. FP8 weight decoding retains the inline assembly and BF16 rounding order. The published INT4 and MXFP4 measurements use their native W4A16 checkpoints.

## Performance

The native baseline is BF16 `PPUDeepGemmExperts` from vLLM-SAIL v0.31.0 and vLLM `0.31.1.dev0+gdb9527a46.d20261113.empty`. Both runs used PyTorch 2.13.0 on a T-Head 810E. Native used Triton 3.7.0 and F225 used FlagTree Triton 3.6.0. The [baseline interface and workload](README.md#vendor-baseline-and-full-call-interface) describe the full operator call. All shapes use E=256, K=4096, N=256 and topk=6. The [DeepSeek-V4-Flash shape capture](shape_source.txt) has 53 shapes and total count 10105 per precision.

Each latency is the median of three complete-call measurements. The ratio is native latency divided by F225 latency. The weighted mean uses the captured count for every numerically valid pair.

| Precision | Count-weighted speedup | Valid shapes | Valid count | Per-shape speedup |
| --- | ---: | ---: | ---: | ---: |
| FP8 | 1.24839x | 51/53 | 9761 | 0.60731x–1.72554x |
| INT8 | 1.31228x | 49/53 | 9417 | 0.61785x–1.51085x |

INT8 meets the 1.3x target. FP8 remains below it. At M=16384 the paired FP8 comparison against the preceding candidate reduced latency from 7.829221 ms to 7.686320 ms.

| M | Count | FP8 native (ms) | FP8 F225 (ms) | FP8 ratio | INT8 native (ms) | INT8 F225 (ms) | INT8 ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 172 | 0.032301 | 0.021898 | 1.475055x | 0.032462 | 0.030959 | 1.048549x |
| 2 | 172 | 0.063638 | 0.036880 | 1.725536x | 0.053835 | 0.053513 | 1.006014x |
| 4 | 172 | 0.102190 | 0.084778 | 1.205385x | 0.102005 | 0.071149 | 1.433690x |
| 8 | 172 | 0.159242 | 0.153582 | 1.036848x | 0.159684 | 0.137002 | 1.165556x |
| 16 | 172 | — | — | Native failed | 0.263787 | 0.230082 | 1.146491x |
| 24 | 172 | 0.368817 | 0.339276 | 1.087068x | 0.368728 | 0.307963 | 1.197315x |
| 32 | 172 | 0.493369 | 0.389970 | 1.265146x | — | — | Native failed |
| 40 | 172 | 0.605318 | 0.465237 | 1.301097x | 0.604524 | 0.416732 | 1.450630x |
| 48 | 172 | 0.636981 | 0.504055 | 1.263714x | — | — | Native failed |
| 56 | 172 | 0.690104 | 0.532775 | 1.295302x | — | — | Native failed |
| 64 | 172 | — | — | Native failed | — | — | Native failed |
| 72 | 172 | 0.785969 | 0.606737 | 1.295403x | 0.786208 | 0.539926 | 1.456140x |
| 80 | 172 | 0.792715 | 0.606628 | 1.306756x | 0.793010 | 0.538821 | 1.471752x |
| 88 | 172 | 0.843461 | 0.637283 | 1.323527x | 0.841823 | 0.567641 | 1.483019x |
| 96 | 172 | 0.798029 | 0.617084 | 1.293226x | 0.798280 | 0.550757 | 1.449424x |
| 104 | 172 | 0.859750 | 0.658229 | 1.306156x | 0.859546 | 0.585048 | 1.469188x |
| 112 | 172 | 0.908679 | 0.678763 | 1.338727x | 0.909816 | 0.602186 | 1.510854x |
| 120 | 172 | 0.866904 | 0.666710 | 1.300272x | 0.866471 | 0.590634 | 1.467018x |
| 128 | 172 | 0.911230 | 0.679257 | 1.341510x | 0.910249 | 0.602954 | 1.509649x |
| 136 | 172 | 0.879683 | 0.676477 | 1.300388x | 0.880636 | 0.598748 | 1.470796x |
| 144 | 172 | 0.916675 | 0.697558 | 1.314120x | 0.917674 | 0.617716 | 1.485593x |
| 152 | 172 | 0.921761 | 0.697445 | 1.321624x | 0.921913 | 0.619267 | 1.488718x |
| 160 | 172 | 0.927141 | 0.708237 | 1.309083x | 0.926367 | 0.628959 | 1.472857x |
| 168 | 172 | 0.928407 | 0.710557 | 1.306591x | 0.928981 | 0.630275 | 1.473930x |
| 176 | 172 | 0.935868 | 0.720006 | 1.299807x | 0.937352 | 0.641999 | 1.460051x |
| 184 | 172 | 0.931478 | 0.735503 | 1.266450x | 0.930371 | 0.654426 | 1.421660x |
| 192 | 172 | 0.938490 | 0.735151 | 1.276595x | 0.939581 | 0.655564 | 1.433241x |
| 200 | 172 | 0.936285 | 0.746130 | 1.254854x | 0.936183 | 0.663549 | 1.410873x |
| 208 | 172 | 0.945769 | 0.749028 | 1.262662x | 0.945952 | 0.669009 | 1.413959x |
| 216 | 172 | 0.951187 | 0.727058 | 1.308268x | 0.951962 | 0.682196 | 1.395438x |
| 224 | 172 | 0.943448 | 0.725535 | 1.300349x | 0.946714 | 0.682176 | 1.387786x |
| 232 | 172 | 0.954806 | 0.730510 | 1.307039x | 0.954664 | 0.700319 | 1.363186x |
| 240 | 172 | 0.947108 | 0.726515 | 1.303632x | 0.949213 | 0.710936 | 1.335160x |
| 248 | 172 | 0.945163 | 0.730397 | 1.294041x | 0.946822 | 0.720897 | 1.313394x |
| 256 | 172 | 0.955129 | 0.727637 | 1.312644x | 0.954485 | 0.710042 | 1.344266x |
| 272 | 172 | 0.956634 | 0.740512 | 1.291854x | 0.956430 | 0.701425 | 1.363551x |
| 288 | 172 | 0.953957 | 0.739744 | 1.289578x | 0.955852 | 0.703547 | 1.358619x |
| 304 | 172 | 0.961570 | 0.743037 | 1.294108x | 0.961493 | 0.704418 | 1.364948x |
| 320 | 172 | 0.965182 | 0.747857 | 1.290597x | 0.964728 | 0.709262 | 1.360186x |
| 336 | 172 | 0.966884 | 0.751557 | 1.286507x | 0.966410 | 0.712920 | 1.355566x |
| 352 | 172 | 0.964598 | 0.757870 | 1.272776x | 0.964323 | 0.720282 | 1.338812x |
| 368 | 172 | 0.979698 | 0.768914 | 1.274131x | 0.978820 | 0.727308 | 1.345812x |
| 384 | 172 | 0.972851 | 0.760148 | 1.279817x | 0.971735 | 0.732272 | 1.327014x |
| 400 | 172 | 0.976105 | 0.769708 | 1.268150x | 0.976438 | 0.735985 | 1.326709x |
| 416 | 172 | 0.988274 | 0.775627 | 1.274162x | 0.987002 | 0.742714 | 1.328912x |
| 432 | 172 | 0.992800 | 0.777487 | 1.276934x | 0.992372 | 0.747885 | 1.326904x |
| 448 | 172 | 0.994813 | 0.786390 | 1.265038x | 0.994402 | 0.754841 | 1.317365x |
| 464 | 172 | 0.997738 | 0.787266 | 1.267346x | 0.996849 | 0.757176 | 1.316537x |
| 480 | 172 | 0.984811 | 0.808084 | 1.218698x | 0.984203 | 0.778845 | 1.263670x |
| 496 | 344 | 0.991217 | 0.823193 | 1.204112x | 0.990383 | 0.792889 | 1.249081x |
| 512 | 344 | 1.011466 | 0.834769 | 1.211671x | 1.011021 | 0.802471 | 1.259885x |
| 2048 | 43 | 1.390602 | 2.289772 | 0.607310x | 1.392323 | 2.253495 | 0.617850x |
| 16384 | 946 | 7.152288 | 7.691796 | 0.929859x | 7.152728 | 7.799341 | 0.917094x |

The following comparisons retain earlier native BF16 numerical failures. Their global relative peak error against the FP32 oracle exceeded 0.02. These shapes were excluded from the weighted means without resampling. The measured pairs had no new numerical failures.

| Precision | M | Native global relative peak error |
| --- | ---: | ---: |
| FP8 | 16 | 0.04202596 |
| FP8 | 64 | 0.11260887 |
| INT8 | 32 | 0.20568949 |
| INT8 | 48 | 0.06993463 |
| INT8 | 56 | 0.02420966 |
| INT8 | 64 | 0.09048066 |

## Validation

- Full-output A/B tests matched the preceding candidate bit for bit on all 53 FP8 and all 53 INT8 shapes.
- The 4096- and 4224-column suites passed 14 functional and graph checks each, including replay, route and weight updates, mixed scales and in-place output. The latter suite exercised a 128-column tail.
- At M=16384, temporary workspace excluding output was 1,393,242,112 bytes for FP8 and 1,393,239,040 bytes for INT8. Both were below 2 GiB.
- Acceptance remains open. The last PPU core entry check had four skips. The [existing PPU commands and environment](README.md#part-3--validation-and-reproduction) apply. The F225 PPU results above were collected before syncing upstream.
