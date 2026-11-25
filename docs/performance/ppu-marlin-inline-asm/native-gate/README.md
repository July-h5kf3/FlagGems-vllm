# PPU Marlin MoE 原生基线验收

以 PPU-ZW810E（驱动 1.3.2-d7f5a2）上的 SAIL 原生 `FusedMoEKernel.apply` 与 `PPUDeepGemmExperts` 为基线。FP8/INT8 W8A16 保持 BF16 激活，原生端使用相同权重解码后的 BF16 MoE。INT4/MXFP4 分别与对应的原生 W4A16 算子比较。原生进程使用 SAIL 0.33.0rc1、vLLM 0.33.1.dev0、DeepGEMM `0f9756f`。候选使用 SDK 2.2.1 和 FlagTree 0.7.0+ppu.git1a96a36a。同一 shape 的两侧在同一物理卡、独立 Python 环境下计时。

| 精度 | 调整前 `4bc5cd66` | 调整后 `57d1c47f` | 调整后独立复测 | 加权门槛 | 验收 |
|---|---:|---:|---:|---:|---|
| FP8 | 1.29080× | 1.30449× | 1.30448× | 1.30× | 达标 |
| INT8 | 1.30890× | 1.30881× | 1.30903× | 1.30× | 达标 |
| INT4 | 0.97158× | 0.97152× | — | 0.95× | 达标 |
| MXFP4 | 0.96981× | 0.97007× | — | 0.95× | 达标 |

每种精度 53 个 DeepSeek-V4-Flash p32768d1024 shape、10,105 次调用。每个 shape 双方各进行三轮 100 ms CUDA Graph 计时，交替先后，每方取中位数。表中数值为 `sum(count × native_ms / candidate_ms) / sum(count)`。FP8、INT8 各要求加权加速比 ≥1.30×，INT4、MXFP4 各要求 ≥0.95×。本次 W4 判定使用表中已有的完整测量，独立复测仅覆盖 W8。本轮没有测服务吞吐或同步 eager 性能。[逐行延迟与精度](paired.csv) · [汇总与失败记录](summary.json)

## 改动与验证

- 稀疏 FP8 grouped kernel 在指针、stride 和 K 维满足条件时，每次按 32 位读取相邻两个 K128 分组的 BF16 scale。其余布局保留单组读取。
- FP8 下投影（K=256、输出宽度 4096）将八条有效路由的两个相邻 K16 片段准备成一个 16 行 AIU tile，复用已有矩阵加载。16 行满块沿用原布局。小批量直接分段计算最多覆盖 256 条路由，平均每个 expert 至多一条，并调整小批量下投影的流水级数。PPU Marlin 的内联汇编调用点维持 10 处。[代码差异](https://github.com/July-h5kf3/FlagGems-vllm/compare/4bc5cd669186daee32e405ab74be0464cd71fc0b...57d1c47faef9d632d0662825fb00fb1a658df67e)
- 稳健性测试覆盖 17/7 条路由混合的 expert tile、带 padding 且地址偏移的 scale、K 分组数为奇数、Graph replay 的路由变化和 scale 原地更新。18 组附加检查与旧代码逐位一致，且通过独立 FP32 精度门槛。已有 Marlin quick 为 268 passed、178 skipped，新增定向用例单独运行通过。4 个仓库 core benchmark 受平台门控跳过。

## 有效测量与复测

计时前核对候选源码 SHA-256、权重和输入哈希及原生/候选的设备 UUID。精度要求候选相对原生、原生相对 FP32 的相对 RMS 均小于 1%，归一化最大误差均小于 2%。下列首测失败均保留原始记录，只对无效 shape 重启独立进程并核对相同输入。有效的原始行均沿用。

- before，INT8 M=64：原生相对 FP32 的最大误差为 7.32%，候选相对原生为 7.32%。候选相对 FP32 为 0.45%。
- after，FP8 M=64：原生相对 FP32 的最大误差为 9.94%，候选相对原生为 9.98%。候选相对 FP32 为 0.65%。

另一次 INT8 复核的 M=64 中，原生相对 FP32 的最大误差为 17.47%，同卡单行重测为 3.12%。两次候选相对 FP32 均为 0.45%。该轮仅有 52/53 个有效 shape（9,933 次调用，加权 1.30686×）。完整记录在压缩包的 `prior-confirm-int8-*` 中。表中 INT8 的独立结果取自之后重新启动的完整 53-shape 测量。

设备监控全程检查运行进程集合。发生预占后，中止的精度分片整段重跑。同一次任务中较早完成、且设备监控已确认结束的独立精度段保留。两轮 W8 各自覆盖完整 53 个 shape，结果分别计算。

## 复核

[原始配对计时、输入哈希及 shape 权重](paired-runs.tar.gz) · [验收脚本](check_native_gate.py)。在空目录中解包并运行下列命令，需要 Python 3.9 或更高版本。脚本独立计算每行和加权比值，校验精度、源码、GPU、输入、权重和覆盖范围。默认校验四种精度，缺少任一项或任一项低于对应门槛均返回非零退出码。仅复核 W8 时显式选择 FP8 和 INT8。

```sh
tar -xzf paired-runs.tar.gz
python3 check_native_gate.py --shapes shapes.json --candidate-sha256 9ec02f3c90f441a55fe5732b0b4b18c884b61063c797803e0c8fb7a0148db449 --retry main-w8-retry.json main-w8-[0-3].json main-int4-[0-3].json main-mx-[0-3].json
python3 check_native_gate.py --shapes shapes.json --candidate-sha256 9ec02f3c90f441a55fe5732b0b4b18c884b61063c797803e0c8fb7a0148db449 --require fp8 --require int8 confirm-fp8-[0-3].json confirm-int8-[0-3].json
```

## 重新计时

压缩包的 `harness/` 内是本次实际使用的主机调度脚本、容器 worker、输入生成和 W4 原生适配。原生与候选分别使用 `/opt/vendor031/bin/python` 和 `/opt/candidate031/bin/python`。在已有 SDK 2.2.1 基准目录中，将候选 checkout 放在例如 `candidate/` 下，确认选定卡空闲并重新核对主机与容器的设备编号，然后运行单卡命令。其他测试镜像需要调整容器名、挂载和解释器路径。

```sh
python3 run_pair_formal.py --gpu "$HOST_GPU" --device "$CONTAINER_GPU" \
  --job "$JOB_ID" --candidate-repo candidate --candidate-sha256 9ec02f3c90f441a55fe5732b0b4b18c884b61063c797803e0c8fb7a0148db449 \
  --deepgemm-release --native-w4 --precision fp8,int8,int4,mxfp4
```
