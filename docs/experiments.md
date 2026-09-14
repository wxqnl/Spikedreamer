# 实验协议与结果来源

本文记录 2026-09-14 已完成的 Walker seed 0 研究，不把未来实验矩阵写成既有结果。[README](../README.md) 展示主表；[结果 JSON](../results/walker_seed0.json) 保存全部评估值。

## 已完成范围

五组为 `legacy`、`stateful_t8`、`ann_gru`、`lif_t8` 和 `stateful_gatedmem32`。每组均完成 1M 原始环境帧、71,171 次训练更新和 100 个完整评估点。完整任务范围目前只有 `walker_walk`，训练种子只有 0。

框架支持 19 个视觉 DMC 任务，不等于已完成 19 任务基准。早期 TA/自适应短验证、提前停止的 context32/slowmem 和固定 T8 主表分开报告。

## 交互、训练与评估

| 项目 | 本次配置 |
|---|---|
| 环境 | 真实 DeepMind Control Suite，Walker Walk |
| 输入 | 64×64 RGB |
| 交互预算 | 1,000,000 原始环境帧，包含随机 prefill |
| action repeat / episode 上限 | 2 / 1000 原始帧 |
| 采样环境 | 4 个真实模拟器；当前路径为独立 CPU OSMesa 工作进程 |
| prefill / pretrain | 2500 agent transitions / 100 次更新 |
| batch / 学习长度 | B=56，L=64；每次更新消费 3584 个学习时间步 |
| train ratio | 每个新增 transition 对应 512 个学习时间步 |
| 想象长度 / 随机潜变量 | H=15 / 32×32 categorical |
| 精度 / 上限 | FP32，`max_updates=0`，不做更新数截断 |
| 评估频率 | 10K、20K、…、1M 帧，每个位置评估 10 个完整回合 |
| 评估随机性 | 固定 seeds 100000–100009，动作取 mode，posterior 仍采样 |
| 训练恢复 | 格式 2 检查点；恢复模型、三个 Adam、回放和 RNG，模拟器开启新 episode |

环境返回的实际 frame count 用于记账，评估交互不加入 1M 训练帧。每个非 prefill transition 累积 `512 / (56×64) = 1/7` 次更新额度；Walker 完整预算对应：

```text
100 + floor((1,000,000 / 2 - 2500) / 7) = 71,171 updates
```

五组的每个固定评估帧点具有相同更新数。Gated Memory 另取最多 32 步真实前缀做无梯度 burn-in；它增加计算，但不缩短 L64，也不计入上述学习样本预算。其他四组已完成运行的 burn-in 为 0。

评估时按回合保留独立 RNG，恢复训练 RNG 后再继续学习。CPU 渲染可以重叠执行，网络仍按独立 B1 轨迹推理；不能将并行环境数当作独立训练种子。

## 模型差异

| preset | 外围 / 核心 | deter | burn-in | 总参数 | 可训练参数 |
|---|---|---:|---:|---:|---:|
| `stateful_gatedmem32` | SNN T8 / MCN T8，输入依赖树突记忆 | 512 | 32 | 19,896,503 | 18,715,060 |
| `ann_gru` | ANN / ANN-GRU，单次递归更新 | 512 | 0 | 19,107,469 | 17,926,030 |
| `stateful_t8` | SNN T8 / 显式状态 MCN T8 | 512 | 0 | 18,846,903 | 17,665,460 |
| `lif_t8` | SNN T8 / 单房室 LIF T8 | 624 | 0 | 19,102,583 | 17,863,796 |
| `legacy` | 发布版 SNN T8 / MCN T8 + 共享训练器 | 512 | 0 | 18,846,903 | 17,665,460 |

总参数包括冻结的 slow-value 网络。LIF 的 deter=624 是实际已完成配置，不能在整理时改写成 512；ANN 使用完整 ANN 外围，不能标成只替换 RSSM 核心。

Gated Memory 相对 Stateful-T8 同时改变树突复位、保留更新、输入门控和 burn-in，并增加 1,049,600 个参数。相对 Static Slow Memory，新增门投影替代 1024 个静态保留参数，净增 1,048,576 个参数。当前结果不是参数或计算量严格配平的机制消融。

## 结果统计

主终点为恰好 1M 帧的评估均值，不选择训练过程最高点。辅助指标取 910K–1M 的末十个评估位置的均值。每点评估标准差使用 `ddof=0`，反映同一已训练策略的十个回合差异。

末十次评估来自同一训练轨迹，彼此相关；100 个评估点也不能充当 100 个独立种子。当前不报告显著性检验、跨种子置信区间或非劣性结论。旧统计工具的多任务、配对种子前提仍须满足，不能把这一份 seed 0 数据输入后放宽门槛。

完整曲线及逐回合分数保存在 [walker_seed0.json](../results/walker_seed0.json)，数值表和图由 [report.py](../spikedreamer/report.py) 生成。图中不平滑、不补点，阴影为回合标准差而非跨训练种子的置信区间。

## 运行时长与故障记录

| 方法 | 累计进程时长 / h | 首次启动至完成 / h | 解释 |
|---|---:|---:|---|
| Gated Memory-T8 | 35.61 | 35.61 | 单段完成，无训练失败或重启 |
| ANN-GRU | 11.21 | 11.21 | 单段完成 |
| Stateful-T8 | 32.00 | 33.42 | 经过两次人工恢复及吞吐路径更新 |
| LIF-T8 | 27.76 | 27.76 | 单段完成 |
| Legacy-T8 | 32.43 | 34.15 | 两次人工恢复；另有一次恢复配置检查失败 |

累计进程时长包含编译、采样、训练、评估、I/O 和恢复前未保留的尾段。首尾时长还包含暂停间隔。Legacy 的失败是恢复配置检查，不应隐去，也不能据此捏造数值发散。原始日志保留在各自档案中。

这些数值是完成实验的观测时长，不是纯 GPU-hours 或统一硬件负载下的 microbenchmark。Gated Memory 在本次记录中慢于 ANN，没有训练加速结论。旧 TA 的短 checkpoint 测量见 [历史验证](validation-20260908.md)，不能用于当前 Gated Memory 的速度或能耗主张。

## 运行环境

完整实验使用 Python 3.10.16、PyTorch 2.6.0 / CUDA 12.4、H100 80GB、dm-control 1.0.9、MuJoCo 2.3.5 和 NumPy 1.26.4。每组实际包版本及配置均在结果 JSON 的 `runtime` 与 `config` 中保留。

当前训练入口从 CPU 子进程运行真实 MuJoCo 物理和 OSMesa 渲染，使用 `runtime/with-osmesa.sh` 设置渲染变量。系统必须已提供 OSMesa 动态库。已完成运行还通过 `SPIKEDREAMER_ENV_LD_PRELOAD` 仅为子进程指定可用的 `libstdc++.so.6`；不能把这项预加载移到 CUDA 训练父进程。

2026-09-14 整理时，主代码服务器缺少系统 OSMesa，真实环境测试在那里未通过；同份代码在原训练服务器的 CPU 上通过 Walker 真实环境检查。没有为文档整理安装驱动或改动系统渲染库，详见 [验证记录](validation.md)。

命令见 [README](../README.md#训练与评估)。`configs/dmc_1m.yaml` 固定共用预算，preset 决定模型宽度、T 和 burn-in。老默认 B16/TA 只为历史兼容保留；遗漏 `--preset` 或共用配置就不是本次主协议。

## 原始证据与导出

以下路径均相对原始实验档案根目录；仓库只收录可公开的数值证据，不收录检查点、回放、内部监控状态或凭证。

| 档案目录 | 主记录 | 对应结果 |
|---|---|---|
| `spikedreamer-fixed-t8-walker-20260910/` | `FINAL_RESULTS.json` | Legacy-T8、Stateful-T8 |
| `spikedreamer-walker-gru-lif-20260911/` | `FINAL_RESULTS.json` | ANN-GRU、LIF-T8 |
| `spikedreamer-stateful-gatedmem-walker-20260912/` | `FINAL_RESULTS.json` | Gated Memory-T8 |
| `spikedreamer-stateful-context32-walker-20260912-rerun1/` | `STOPPED_BY_USER.json` | 71K 日志 / 70K 检查点，提前停止 |
| `spikedreamer-stateful-slowmem-walker-20260912/` | `STOPPED_BY_USER.json` | 98K 日志 / 90K 检查点，提前停止 |

每组原始评估来自 `runs/<preset>/walker_walk/seed-0/metrics.jsonl`。恢复后以检查点保留前缀和后续实际评估组成曲线，不把被丢弃尾段当成额外样本。导出器检查完整帧点、更新数、逐回合统计和终点档案一致性；它不替代模型或回放验收。

当前统一代码来自已完成 Gated Memory 的冻结实现，保留五组模型结构。旧曲线属于各自冻结代码和运行记录，不声称统一入口能逐位复现经过恢复与吞吐迭代的历史轨迹。未来研究计划见 [research-roadmap.md](research-roadmap.md)，本次整理没有启动新实验。
