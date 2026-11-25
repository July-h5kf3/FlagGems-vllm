# PPU Marlin MoE 小幅回退项复查

按原审计编号，第 5 项与第 17 项已完整替换。第 5 项原有汇编覆盖的 INT8 Square、短尾和 Wide 路径，现统一复用上线过的 `decode_int8_fragments`。第 17 项按 PPU AIU 的 swizzle 计算两个 word 的共享内存地址，用两次普通 Triton 读取代替 `ldmatrix.x2`。Warp/lane 布局仍由 T-Head 后端表达。该文件的内联汇编从 12 处降到 10 处。相对最初的 17 处，累计 6 处完全替换、6 处缩小范围、5 处保留。[本轮代码差异](https://github.com/July-h5kf3/FlagGems-vllm/compare/23d928210d558d6df29147b44d127a5f8e75c8f4...4bc5cd669186daee32e405ab74be0464cd71fc0b)

## 实测代价

同一张 PPU-ZW810E 上，以 `23d92821` 对比 `4bc5cd66`。每种精度 53 个 shape，E=256、K=4096、N=256、top-k=6；两版使用相同权重、激活与路由。每个 shape 两版各执行六轮 75 ms Graph 计时，交替先后次序，取中位数。平均延迟变化按每个 shape 的“新版 / 旧版 − 1”以频次加权，加权比值沿用原报告“旧版 / 新版”的计算方式。每种精度总权重均为 10,105。[逐 shape 数据](followup-paired.csv) · [各轮计时与对照](followup-raw.json)

| 精度 | 加权平均延迟变化 | 加权旧版／新版 | 最大单 shape 延迟增加 | 对应耗时 µs |
|---|---:|---:|---:|---:|
| FP8 | +0.86% | 0.99151× | +2.07%（M=48） | 500.55 → 510.91 |
| INT8 | +1.89% | 0.98157× | +2.80%（M=184） | 650.24 → 668.48 |
| INT4 | +1.38% | 0.98659× | +6.15%（M=8） | 128.82 → 136.74 |
| MXFP4 | +0.46% | 0.99542× | +1.15%（M=136） | 735.78 → 744.25 |

- **第 5 项。** 单独替换时，INT8 M=512 的整算子由 800.14 增至 803.56 µs，M=2048 由 1391.38 增至 1401.64 µs。实现删除了原来的汇编函数和按权重分块格式选择解码器的分支。
- **第 17 项。** 单独替换的候选在 INT8 M=64 上由 493.66 增至 507.53 µs，在 M=512 上由 800.14 增至 802.30 µs。最终组合结果见上表。INT4 M=8 的独立复测为 127.93 → 136.11 µs（+6.39%）。另一种按字节计算地址的写法在该点较快，但在 INT4 M=64/128 和 INT8 M=64/512 上更慢，因此保留了按 32 位 word 寻址的写法。
- **第 14 项。** 另测了用四个普通读取替换 `load_matrix_x4`。单独替换时，INT8 M=512 由 800.12 增至 853.41 µs（+6.66%），M=2048 由 1391.87 增至 2006.85 µs（+44.18%）。此次保留 `ldmatrix.x4`。

## 验证

212 组整算子输出均与审核基线逐位相同，并通过独立 FP32 抽样参考检查（相对 RMS <1%，归一化最大误差 <2%）。另对照了 128/256 线程下八行激活的全部 32 字节 sector 和各 warp/lane，两次普通读取的结果与 x2 指令逐 word 相同。四读取的对照候选亦在其余四种 AIU tile 几何下通过同样检查。四组边界测试覆盖短尾、不同 K 长度、原地更新 scale 与路由后的 Graph replay。PPU Marlin quick 为 268 passed、178 skipped，四项 core benchmark 受平台门控跳过。数据为算子 Graph 延迟，服务吞吐未测量。

## 复现

使用上一轮的 [实验包](experiments.tar.gz) 中的 `sweep.py`、`inputs.py` 与 `shapes.json`。在 PPU 环境中令 `PYTHONPATH` 指向基线 23d92821 的 `src`，把候选提交的 `src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py` 导出到运行目录为 `candidate.py`。例如运行 `python sweep.py int8 candidate int8-results` 可生成 INT8 的逐 shape 对照。对 fp8、int4、mxfp4 使用相同入口。
