# SpikeDreamer v0.1：设计与实现

## 动机与主假设

Spiking-WM 已经给出端到端脉冲世界模型。它的发布版 RSSM 每次 `obs_step/img_step` 重置模块内部神经元状态，并在该环境转移内展开 T 次；官方启动脚本取 T=8。上一转移的 deterministic spike 表示仍有递归连接，不能将它描述为“完全没有跨时间记忆”。

本项目改变的是神经元状态的时间组织：把 MCN 胞体、树突和相关 LIF 膜电位显式放入 RSSM state，沿真实转移及想象转移携带。主模型每次转移只更新一次。主假设是这能减少重复仿真，在相同环境预算下保持控制性能；这仍需完整实验。

## 状态与计算

每条轨迹持有：

| 字段 | 形状，不含序列轴 | 含义 |
|---|---|---|
| `stoch`, `logit` | B×32×32 | categorical 群体编码及其模拟值 logits |
| `deter` | B×512 | 当前 MCN spike，主配置严格为 0/1 |
| `soma`, `basal`, `apical` | B×512 | 发放及 reset 后的胞体、树突状态 |
| `input_mem`, `prior_mem` | B×层数×512 | 跨转移的输入与 prior LIF 膜电位 |
| `steps` | B×1 | 此次转移实际使用的核心更新次数 |

每个 categorical 分组选择一个类别，通过 straight-through 重参数化训练。32 组共有 32 个激活位；主配置不沿内部 T 轴复制。像素、连续动作、概率分布读出及神经元内部电位仍为模拟值，不宣称全链路只有二值运算。

MCN 保留发布源码的积分形式。令 `b_in, a_in, u_in` 为投影与 PopNorm 后的电流：

```text
b = b_prev + (b_in - b_prev) / tau_b
a = a_prev + (a_in - a_prev) / tau_a
u = u_prev + sigmoid(a) * (b - 2*u_prev + u_in) / tau
s = QGate(u - threshold)
```

默认发放后重置胞体和两条树突，reset 的 spike 分支 detach，膜电位跨时间不 detach。`reset_dendrites=False` 可单独消融树突重置。此处以发布代码为基准，不将其声称为论文公式的逐字转写。

新核心 PopNorm 初始 gain 为 `threshold × core_norm_gain`，默认 `core_norm_gain=2`。单步串联神经元在原尺度下初始活动过低，因而显式开放该尺度；它是一个需要报告的设计变化，须与 `gain=1` 及 reset 对照共同检验，不能把全部收益归因于持久状态。

编码器、posterior 观测融合、decoder、actor 和 critic 是每次调用独立的 SNN 计算；它们的内部膜电位不跨环境转移延续。持久化范围仅限上表。主配置这些读出模块也使用 T=1。

## 状态边界与训练目标

- 状态字典不在调用者侧原地修改；各 imagination 分支携带独立状态，训练与评估不依赖共享 MCN 缓存。
- `is_first` 逐样本重置所有动态状态并屏蔽入站动作。episode 时间截断不误标为环境终止。
- Replay 中 `action[t]` 表示到达 `observation[t]` 的动作；随机片段起点与跨 episode 拼接均设置 reset。可用 `burn_in` 在不回传梯度的前缀上恢复状态；基准默认 0。
- World-model loss = 图像 NLL + 两热奖励 NLL + continuation NLL + balanced KL。主配置无额外稀疏正则；`spike_reg` 仅约束最后一次核心发放的平均活动，不等于全部计算成本。
- 想象 H=15 个动作、H+1 个状态，用 `r(s[t+1])`、`V(s[t+1])` 和正确的末端 bootstrap 计算 λ-return。actor 通过冻结参数但保留输入梯度的世界模型训练；critic 使用 detach 目标和慢网络正则。
- FP32 是主协议；BF16 保留 FP32 概率读出。大而有限的梯度使用 FP64 范数后进行标准全局裁剪，真正 NaN/Inf 仍报错，绝不清零伪装成功。v0.1 使用 eager，不包含未经验证的编译或稀疏 GPU 加速声明。

## 方法与必要对照

| 设置 | 动态核心 | 外围 SNN T | 回答的问题 |
|---|---|---:|---|
| `legacy` | 发布版、内部 T=8、每转移重置 | 8 | 统一训练器下的原架构主基线 |
| `legacy` + `core_steps=1 io_steps=1` | 发布版 T=1 | 1 | 仅缩短原模型是否已足够 |
| `ta` | 持久 MCN，单步 | 1 | 主方法 |
| `ta_core` | 持久 MCN，单步 | 8 | 隔离动态核心收益 |
| `ta` + `persistent=False` | 重置膜电位，保留 spike 递归 | 1 | 神经元跨转移记忆是否有效 |
| `binary` | 无膜电位 MCN 硬阈值递归，LIF 不跨转移 | 1 | 膜电位动力学是否优于二值递归 |
| `gru` | 同宽 GRU 动态核心，仍使用脉冲外围 | 1 | GRU 动态控制，不是完整 ANN Dreamer |
| `ta` + `readout=membrane` | 下游改读胞体电位 | 1 | spike-only 读出的代价 |

MCN binary 对照保持同形状的三个投影，但不执行树突/胞体时间积分。GRU 仅保持宽度一致，不声称参数完全相同；每个 run 的 manifest 记录真实参数量。

## 自适应分支：已实现，未作为主方法

`adaptive` 在第一步后计算 categorical prior 的分组归一化熵。超过 `entropy_threshold` 的样本被实际 gather 到紧凑 batch，执行额外 MCN/prior 更新，再 scatter 回完整状态。外部 action/latent 只消费一次，额外步输入为零外部 spikes，最多 `adaptive_max_steps=4`。输入 LIF 不在额外步重复编码。

门控是硬规则，没有对整数 K 虚构可微梯度，也没有实现设计初稿中不可直接求导的 K-budget loss。需要在开发集校准阈值，报告观测及想象阶段的实际 K，并计入 gather/scatter 和 host 同步开销。

当前默认阈值 0.8 在短运行中几乎总执行 K=4，且长 BPTT 梯度很大。虽然数值裁剪和完整更新已跑通，尚不能视为稳定有效的自适应算法。优先检验固定单步模型，不为这条分支增加未经验证的机制。

本文件取代前期讨论稿中的实现假设；具体代码、计数边界与统计口径以本仓库 v0.1 为准。
