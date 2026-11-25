# PPU Marlin MoE 内联汇编逐项审计

当前审核分支已按原生 PPU 基线验收全部四种精度的加权加速比，W8 门槛为 1.30×，W4 门槛为 0.95×。[达标结果、逐 shape 数据及复核脚本](native-gate/README.md)

后续提交已继续替换第 5、17 项，当前审核分支剩 10 处。更新后的精度、延迟代价和保留 x4 的依据见[小幅回退项复查](small-regression-followup.md)。以下为上一轮的 17 项审计记录。

比较基线为 `30dc5ae3`，候选为 `23d92821`。以 Python AST 统计基线 PPU Marlin 实现中的全部 17 个 `tl.inline_asm_elementwise` 调用点。4 处完全改用编译器接口，7 处缩小汇编范围，6 处保留。两条 wait 常量分支共用一个调用点，最终源码有 12 处。新增的 `marlin_fragments.py` 有 0 处内联汇编。[代码差异](https://github.com/July-h5kf3/FlagGems-vllm/compare/30dc5ae3e1264a617446023c96da6487dc43a78c...23d928210d558d6df29147b44d127a5f8e75c8f4)

## 整算子结果

设备为 PPU-ZW810E，驱动 1.3.2-d7f5a2，HGGC SDK 2.2.1，PyTorch 2.13.0+v0.1.0.ppu2.2.0，FlagTree 0.7.0+ppu.git1a96a36a。公开入口基线与候选共用权重、激活及路由。每个 shape 在同一张卡上分别执行六轮 75 ms CUDA Graph 计时，每轮交替先后顺序，取各自中位数。加权值为逐 shape 的“基线中位耗时 / 候选中位耗时”按调用次数的算术平均。数据来自 DeepSeek-V4-Flash p32768d1024 的 53 个 token 数，E=256、K=4096、N=256、top-k=6。W4 在同一组尺寸和频次上补充验证。两侧均为 FlagGems-vllm 的 PPU 实现。[212 行原始比较](paired.csv) · [各轮计时和独立试验](measurements.json)

| 精度 | 输出逐位相同 | 调用次数 | 加权比值 | 最小比值 | 最大比值 |
|---|---:|---:|---:|---:|---:|
| FP8 | 53/53 | 10,105 | 1.00389× | 0.99849×（M=32） | 1.01057×（M=64） |
| INT8 | 53/53 | 10,105 | 1.01085× | 0.99900×（M=2） | 1.29859×（M=1） |
| INT4 | 53/53 | 10,105 | 1.00909× | 1.00181×（M=1） | 1.01159×（M=96） |
| MXFP4 | 53/53 | 10,105 | 1.00403× | 0.99805×（M=1） | 1.00622×（M=4） |

所有 shape 另对抽样 token 比较独立的 FP32 SwiGLU MoE 参考，相对 RMS 小于 1%，按参考最大幅值归一化的最大误差小于 2%。表中保留首轮结果。M=16 的 FP8 路径另外核对了实际执行的两个 matvec 特化和一个归约 kernel；三份可执行段的 SHA-256 均与基线相同。模型服务吞吐和其他硬件版本没有测量。

## 原始 17 个位置

行号链接指向基线。表中每组 A/B 都来自同次、同卡的配对实验；同时给出不同试验时分别标注。函数级微基准描述给定工作量的 kernel 总时间。编译失败和错误输出均不作为性能样本。表中反引号内的实验名是 measurements.json 的顶层键。

| 基线行 | 函数或操作 | 处理和证据，A → B 的单位为 µs |
|---:|---|---|
| [90](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L90) | `decode_int8_scaled_bf16` | **缩小范围。** 单 token BF16 matvec 沿 K 成对解码，整算子 30.54 → 23.55。Dense bank 使用 Triton 整数转换。其余融合路径保留 packed unpack，scale 乘法由 Triton 处理。在 M=768 单独换成普通转换，grouped kernel 为 715.78 → 938.00。见 `matvec-k`、`sdk-int8`、`partial-int8`。 |
| [111](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L111) | `decode_fp8_scaled_bf16_normalized` | **缩小范围。** 普通 dense bank 使用编译器生成的 BF16 对；M=4096 的整个调用为 3384.08 → 3347.73。融合 matvec 与双流 overlap 继续用 packed 版本。M=1 的矩阵向量 kernel 用原生 E4M3 cast 后为 9.16 → 185.15；单独把 FP8 scale 乘法拆出来也使 M=1 整算子从 19.21 增至 22.04。M=16384 的 overlap 中只改变 down 权重 bank，整算子 7674.05 → 7702.73。见 `fp8-bank-ordinary`、`stream-fp8`、`partial-fp`、`fp8-bank-mix`。 |
| [164](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L164) | `decode_raw_int4_bf16` | **缩小范围。** Dense bank 用 Triton 位运算和 BF16 算术。M=16384 的 K=4096 权重 bank 用 BF16 位嵌入后为 557.28 → 528.41。融合路径保留 packed 解码。M=1024 的 grouped kernel 为 540.46 → 784.38（位嵌入），普通整数展开为 978.97。M=768 单独把 scale 乘法移出该汇编，整算子 1324.48 → 1425.78。见 `bits-i4`、`sdk-i4`、`partial-i4`。 |
| [243](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L243) | `decode_raw_mxfp4_bf16` | **缩小范围。** Dense 路径使用 E2M1 位嵌入或 SDK packed 解码；M=16384 整算子 8929.70 → 8904.46。融合 kernel 保留 nibble 转 BF16，乘 scale 已交给 Triton。M=768 融合 kernel 全换成位嵌入为 706.13 → 1083.54，换成通用 E2M1 数值转换为 1442.64。见 `mxfast-dense`、`mx-extra`、`partial-mx`。 |
| [476](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L476) | `decode_int8_fragment_bf16` | **缩小范围。** 独立 Narrow 和 matvec 用 SDK/FlagTree 输入片段算术；Square 的完整块、其短尾分支及 Wide 保留短的 packed unpack，乘法走 SDK。M=512 Square 的 K=4096 投影在原版为 430.53，全 SDK 多种排序为 431.72–441.13；M=2048 Wide 整算子为 1390.94 → 1393.72–1399.14。另一次 M=64 对照，交错生成 Narrow 的两个权重字使整算子 500.51 → 493.84。见 `order2-int8`、`ship-i8-compare`、`partial-square`。 |
| [544](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L544) | `decode_fp8_pair` | **完整替换。** 使用 SDK 字节置换、FlagTree 三输入逻辑、packed BF16 乘法。普通与溢出安全路径分别安排两个权重字的计算顺序。E=8、K=7168、N=1536 的附加 case 中，调整稳健分支后整算子由 390.01 降至 385.16。见 `word-edge`、`edge-final` 和穷举结果。 |
| [556](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L556) | `fold_fp8_scale_fragment` | **完整替换。** 用 SDK 的 packed BF16 乘法折叠两个 scale。FP8 M=128 单处替换为 675.84 → 675.89，M=512 为 824.68 → 824.91。对照含有符号零、无穷和 NaN 的 scale。见 `packed-fp8` 和穷举结果。 |
| [593](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L593) | `mma_bf16_m16n8` | **保留。** 用 one-hot 操作数测得 A/B/C 的 lane 和寄存器排列，再重排成 Triton tile 调 `tl.dot`。单次 MMA 微基准为 3.32 → 11.06；16 次依赖累加为 3.82 → 103.14，有限整数输入的 FP32 结果完全相同。直接嵌回 narrow kernel 的候选出现非有限输出；该失败不作为有效性能样本。另用仓库已有的完整高层 GEMM 路径，INT8 M=64 的整算子为 498.67 → 695.17。见 `mma_bf16_m16n8-micro`、`generic-i8`。 |
| [611](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L611) | `copy_w8_tile_async` | **保留。** 单独把二维 AIU copy 换成普通 load/store，保留原 commit/wait，INT8 M=64 整算子为 493.22 → 1913.41；把整个调度换成同步拷贝为 1906.56。公开 `tle.gpu.copy` 指针形式通过了 swizzle 内容检查：1024 个 128×128 字节 tile 为 6.50 → 56.19；descriptor 形式被当前 PPU 后端显式拒绝。带 `tle.load(is_async=True)` 的候选通过单 tile 检查，但在实际 narrow kernel 的 TTGIR pass 失败。见 `copy-i8`、`group-cost` 和编译记录。 |
| [684](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L684) | `commit_w8_tile` | **保留。** 现有 AIU 流水要求将本轮拷贝提交为一组。测试镜像没有向该 Triton 前端提供 async-commit builder；SDK 函数返回 void，不能通过该版 elementwise extern 调用。可运行的 TLE 同步调度整体为 493.22 → 1906.56（M=64）。该数值衡量完整协议。见 `copy-i8`、`group-cost`。 |
| [697](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L697) | `wait_w8_tile(KEEP=True)` | **部分替换。** CTA barrier 已用 `tl.debug_barrier()`；保留仅等待至剩一组的硬件操作。同卡将两个分支都改为等待全部传输，结果逐位相同，INT8 M=64 为 493.22 → 567.12，M=512 为 800.32 → 978.99。见 `copy-i8`。 |
| [706](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L706) | `wait_w8_tile(KEEP=False)` | **部分替换。** 尾块仍须等待最后一组；CTA 同步交给 Triton，两条 wait 共用一个编译期选择的汇编调用。单 tile 探针确认机器码先执行 group wait，再执行 CTA barrier，然后读共享内存。只用最后一组的 16×128 字节 tile，AIU 完整协议与合法 TLE 同步方案为 2.55 → 4.93。该比较没有移除仍在飞行的传输所需的等待。见 `group-cost`、原版/候选 ISA。 |
| [718](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L718) | `mma_bf16_square` | **保留。** 16×16×16 的实测 fragment 重排加 `tl.dot`，单次 3.11 → 18.74，16 次依赖累加 3.71 → 197.69。INT8 M=512 在相同外层 Square kernel 中替换这一处，整算子 799.80 → 22967.16；该整算子结果在误差门槛内，未达到逐位相同。见 `mma_bf16_square-micro`、`dot16-acc`。 |
| [735](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L735) | `load_matrix_x4` | **保留。** 按已验证的 swizzle、warp/lane 位置用四个普通 shared-memory load 重建四个寄存器。探针逐字相同；INT8 M=512 Square 的上投影 428.44 → 458.34，下投影 289.60 → 325.08。见 `loads2-i8`。 |
| [750](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L750) | `decode_i4_register` | **完整替换。** 用 SDK 置换、packed BF16 减法和乘法。将两个权重字并列处理后，INT4 M=64 的整算子 450.59 → 445.92；M=512 为 691.88 → 685.49。Wide 单独实测了处理顺序。见 `ordering-int4` 和最终全量表。 |
| [769](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L769) | `decode_mx_register` | **完整替换。** SDK 置换完成 E2M1 查表与符号扩展，随后用 packed BF16 乘 scale。MXFP4 M=64 的整算子 597.49 → 594.67，M=512 为 906.10 → 902.36。Wide 保留同一套 SDK 解码，用三输入逻辑合并符号，并先处理第一组权重；最终 M=2048 的配对中位数为 1402.68 → 1403.33。见 `sdk-mx` 和最终全量表。 |
| [876](https://github.com/July-h5kf3/FlagGems-vllm/blob/30dc5ae3e1264a617446023c96da6487dc43a78c/src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py#L876) | `load_activation_fragment_x2` | **保留。** 使用与硬件 swizzle 相同的地址表达式，用两个普通 shared-memory load 读取激活。探针逐字相同；INT8 M=64 上投影 289.19 → 294.07，下投影 183.74 → 188.26，整算子 500.20 → 506.01。见 `loads2-i8`。 |

## 编译器边界

- 数值片段通过 FlagTree 的 `extern_elementwise` 调用 HGGC 的 `__ppukernel_prmt_u32`、`__ppumath_hmul2_2bf16`、`__ppumath_hsub2_2bf16` 以及编译器 intrinsic `llvm.ppu.lop3`。这些是 T-Head 后端专用能力。普通 dense 的整数解包、位转换与 CTA barrier 使用 Triton 语言。
- 在该镜像上，`tle.gpu.copy` 指针版本可以运行；descriptor 版本在 PPU lowering 报 `unsupported TLE feature on the 'ppu' backend: tle.gpu.copy.tensor_descriptor`。`raw.call` 的 PPU 路径由该版本的 primitive 支持检查拒绝。Gluon 的 async-group 封装在当前 Triton builder 上缺少 `create_async_commit_group` / `create_async_wait_group`。SDK 的 void commit/wait 用 elementwise extern 表达时未通过结果类型校验。
- 最后一组 wait 的原版与把 CTA barrier 移到 Triton 的小型候选，反汇编顺序均为 AIU load → group commit → wait_group(0) → CTA barrier → swizzled shared load。没有把直接删除 commit、最后一组 wait 或 CTA barrier 当作合法替代。这里的保留结论适用于实际测量的编译器、目标和形状。

## 精度、复测与复现

- 每个被检查的数值转换分支覆盖 256 个输入字节和全部 65,536 个 BF16 scale 位型的组合，寄存器内使用不同 lane 模式。有限数、无穷、有符号零均按位比较，NaN 比较类别。最终 12 组分支各 16,777,216 组输入，无差异。Dense bank 另外检查非零权重 storage offset、带 padding 的 scale 和尾部列，逐值差异为 0。
- 另测不在 53 行内的 M=768 融合路径及 M=4096 普通 dense 路径。四种精度的完整输出均与基线逐位相同。M=4096 的 FP8、INT8、INT4、MXFP4 整算子分别从 3436.25、3463.52、3972.30、4034.53 µs 变为 3399.68、3442.57、3918.31、4007.78 µs。记录键为 `additional-*`。
- 小专家数、长短 expert tile 混合、奇数个 K tile 和 graph replay 更新 scale/routing 均由独立边界实验验证。最终 PPU Marlin quick 为 268 passed、178 skipped。仓库的四项 core benchmark 因平台门控跳过；本表使用独立的同卡 A/B 计时。
- [下载实验脚本](experiments.tar.gz)。在上述 PPU 环境中选择空闲设备并设置 runtime 可见设备，再在空目录中解包并运行下列命令。脚本将基线导出到独立临时目录，候选通过精确提交号读取。运行前确保 checkout 能解析两个提交；改用 int8、int4 或 mxfp4 可复现其余三组。另可用 `gen.py`、`pipeline.pygen` 等生成单处对照，用 `caller.py`、`mma_unit.py` 或 `group_cost.py` 运行对应测量。

```sh
tar -xzf experiments.tar.gz
PYTHON_BIN=/opt/candidate031/bin/python bash reproduce.sh /path/to/FlagGems-vllm fp8
```
