# 设计与方法：输入依赖的树突记忆

本文对应 2026-09-14 整理后的主方法 `stateful_gatedmem32`。实现来自已完成 1M 帧训练的冻结 Gated Memory 快照。本文解释现有代码，不引入新算法。

## 1. 研究问题和实现范围

问题是如何在脉冲世界模型中管理跨环境转移的神经元状态。我们把树突状态的保留与胞体发放复位分开，并让保留比例依赖输入。训练端从真实历史恢复片段初始状态；想象端沿分支携带相同结构的状态。

该设计建立在 Spiking-WM 之上。[原论文](https://doi.org/10.1073/pnas.2513319122) 已提出 MCN 和脉冲世界模型；[发布代码](../upstream/networks.py) 保留跨环境转移的 spike-indexed recurrence。本项目所说的“显式状态”是膜状态进入轨迹，而不是给原本无递归的网络添加时间联系。

完整 ANN-GRU 是共享训练器下的另一条模型路径，不是本方法内部隐藏的 ANN 记忆支路。

## 2. 两个时间轴

令 h 表示环境决策步，k=1,...,T 表示一次转移内的脉冲仿真索引，当前 T=8。action repeat=2，因此一个完整决策步通常对应两个原始环境帧。

`StatefulFixedRSSM.img_step` 保留发布版的配对方式：第 (h,k) 个内部步使用上一环境步 **同一个 k** 的脉冲 s[h-1,k]，不是 s[h,k-1]。胞体与树突电位则在内部 k 轴连续积分，并在 k=T 后传到下一环境步。输入 LIF 在 T 个内部步重复消费当前 latent/action 输入，外围 SNN 仍使用 T8。

这种双时间轴不能与单步时间对齐 TAP 混为一谈。早期 `ta` 和 `adaptive` 分支仍保留以兼容旧记录，但不是当前主方法。

## 3. 轨迹状态

下表省略学习序列轴；B 为轨迹数，D=512，Z=C=32。

| 字段 | 形状 | 含义与边界 |
|---|---|---|
| `stoch`, `logit` | B×32×32 | 分类潜变量及其 logits |
| `deter` | B×8×512 | 上一次转移各内部索引的发放输出 |
| `soma`, `basal`, `apical` | B×512 | T8 末端的胞体和两路树突电位 |
| `input_mem` | B×input_layers×512 | 输入 LIF 边界膜状态 |
| `prior_mem` | B×output_layers×512 | 先验 LIF 边界膜状态 |
| `steps` | B×1 | 实际内部步数，当前固定为 8 |
| `gate_stats` | B×6 | 两路保留乘积的均值/最小值/最大值，仅 detach 日志 |

`deter` 在实际转移后为二值脉冲；继承的学得初始状态由 `tanh(raw.W)` 构造，不能把它也描述成已发放脉冲。树突、胞体、权重、门值、动作和分布读出均包含连续值。本实现使用稠密 GPU 运算，不声称全链路只有二值算术或自动获得事件驱动节能。

真实 episode 起点逐样本重置完整状态并屏蔽入站动作。观测 posterior 的 LIF 每次观测重置，不进入 imagination 所需的持久状态；编码器、解码器、Actor 和 Value 的局部膜状态也不跨环境转移共享。

## 4. 输入依赖的树突更新

### 4.1 门控

对输入脉冲 x[h,k] 和上一步同索引脉冲 s[h-1,k]：

```text
c[h,k] = concat(x[h,k], s[h-1,k])
g[h,k] = W_gate c[h,k] + b_gate
(logit_b, logit_a) = split(g[h,k])
rho_j[h,k] = exp(logsigmoid(logit_j[h,k]) / T)
write_j[h,k] = 1 - rho_j[h,k]
             = -expm1(logsigmoid(logit_j[h,k]) / T)
```

j∈{b,a} 分别对应基树突和顶树突。`W_gate` 是一个合并的 1024×1024 投影，偏置为 1024 维，总计 1,049,600 个参数。使用 `logsigmoid` 和 `expm1` 避免接近 1 的保留率产生明显的相消误差。

门的输入在树突积分前即可获得，因此一次批量计算 T8 门值，再执行原顺序的 T8 神经元积分。这个优化不删除内部步、不降低精度，也不截断梯度。

### 4.2 树突和胞体

候选电流仍来自原 MCN 的投影与 PopNorm，沿用上游参数名称（包括 `apcial_w` 的历史拼写），保持检查点兼容：

```text
candidate_b = basal_norm(basal_w(c))
candidate_a = apical_norm(apcial_w(c))
input_u     = soma_norm(soma_w(x))

b = b_prev + write_b * (candidate_b - b_prev)
a = a_prev + write_a * (candidate_a - a_prev)

u = u_prev + sigmoid(a) * (b - 2*u_prev + input_u) / tau
s = QGate(u - threshold)
u = u * (1 - stop_gradient(s))
```

当前 b、a 不随 s 清空；u 仍执行原硬复位。注意这里有两种不同的“门”：原 `sigmoid(a)` 调节树突对胞体的作用，新 `memory_gates` 调节树突自身的记忆保留。后者是本版本的新增模块，不能将前者写成本项目的新贡献。

对给定候选，树突更新是逐维凸组合。但候选本身来自递归网络，这不构成整个系统有界、梯度稳定或长程记忆有效的证明。

### 4.3 初始化与统计口径

设初始半衰期 H0=8 个环境决策步：

```text
r0 = 2 ** (-1 / H0)
W_gate = 0
b_gate = logit(r0)
rho0 = r0 ** (1 / T)
```

恒定初值下，一次决策的 T 步直接保留乘积为 r0，H0 次决策后为 1/2。构建门层时保留并恢复 CPU RNG，避免新增层构造消耗随机数而改变外围网络初始化。初始化时它退化为前版 Static Slow Memory 的保留设置；训练后门权重才学到输入选择性。

日志 `keep_env` 记录实际 T 个内部步的 rho 乘积。训练后各步门值可不同，这个乘积不是单一固定半衰期，更不是整网的记忆长度。候选对历史的依赖、胞体动力学、随机潜变量和后续网络都不包含在这一标量解释中。

## 5. 真实历史预热与学习

`Replay.sample(B, L, burn_in=32)` 保留原 L=64 学习窗口的采样，额外构造同一 episode 中最多 32 步先前记录。构造前缀不再消耗回放 RNG。前缀不足时只为固定张量形状重复一条真实观测，并通过 `warmup_valid` 禁止这些填充位置更新状态。

预热执行真实 encoder 和 RSSM，但包在 `torch.no_grad()` 中；结束状态 detach 后才进入完整 L64 学习。学习窗口中的真实 episode 边界仍会重置状态；后续拼接片段也有明确边界。没有从未来观测推断前缀，也没有把 32 步前缀加进 loss 或更新预算。

这能减轻随机片段零状态初始化与在线状态的不一致，但预热最多覆盖 32 步，不保证重建任意长历史下的精确在线状态。早于可用前缀的历史仍被截断；本次实验尚未单独测量这类状态误差。

## 6. 世界模型与策略学习

[model.py](../spikedreamer/model.py) 联合优化图像 NLL、离散回报 NLL、continuation NLL 和 balanced KL。当前 `spike_reg=0`，没有额外稀疏目标。

Actor 和 Value 在后验状态起点上做 H=15 的潜空间想象。行为目标沿用修正后共享训练器和发布实现的状态/奖励对齐、continuation 权重、回报归一化与慢 Value 正则；当前门控版本不额外改写这些目标。Actor 学习时冻结世界模型参数，但保留经世界模型输入的梯度；三个 Adam 分别拥有世界模型、Actor、Value 参数。

使用原二值发放与分类潜变量构造 `get_feat`。`basal`、`apical` 和 `gate_stats` 不直接拼接到预测头或策略输入，因而没有新增连续记忆特征旁路。

## 7. 对照与可归因范围

| 比较 | 保持一致的部分 | 同时变化/限制 |
|---|---|---|
| Stateful-T8 vs Legacy-T8 | 同尺寸、固定 T8、共享训练协议 | 跨转移膜状态组织；早期运行有吞吐改进和恢复分段 |
| Gated Memory vs Stateful-T8 | SNN 外围、T8、胞体与读出骨架 | burn-in32、树突不清空、输入门控及更多参数 |
| Gated Memory vs Static Slow Memory | 初始保留、外围初始化基准、burn-in、T8 | 输入依赖门替代静态保留参数；Static 只完成 90K 检查点 |
| Gated Memory vs ANN-GRU | 1M 帧、B56/L64/H15、更新预算 | 全套网络类型、T、参数、burn-in 和计算量 |
| LIF-T8 vs MCN 方法 | SNN 外围、T8、共享任务协议 | 单房室结构与 deter=624；不是逐参数匹配 |

论文需要检验的贡献是这些机制的效果及适用条件。当前单 seed 结果不能回答“收益是否主要来自更多参数”“是否仅为 burn-in 效果”或“在新任务上是否依旧成立”。

## 8. 代码入口

- [rssm.py](../spikedreamer/rssm.py)：`RSSMBase.warmup`、`StatefulFixedRSSM`、`StatefulGatedMemoryRSSM`、完整 ANN 和 LIF 核心。
- [replay.py](../spikedreamer/replay.py)：有效前缀、不可变采样与检查点引用。
- [model.py](../spikedreamer/model.py)：`WorldModel.infer/loss`、`Agent.imagine/train_batch`。
- [config.py](../spikedreamer/config.py)：命名 preset 与非法组合检查。
- [acceleration.py](../spikedreamer/acceleration.py)：区域 fullgraph 编译，参数键保持不变。

当前模型没有 Transformer attention、KV cache 或 MoE；这些机制不是本研究的组成部分。[研究计划](research-roadmap.md) 列出下一步应补齐的证据。
