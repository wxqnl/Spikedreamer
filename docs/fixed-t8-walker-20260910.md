## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / implementation
- Origin Date: 2026-09-10
- Verification Status: UNVERIFIED (implementation and full-size update checked; full-budget result pending)
- Version Label: fixed-t8-explicit-state-walker-v1

## 本轮问题与用户决定

先在原固定T=8计算骨架上检验跨转移神经元状态的作用，再考虑降低T。
本轮只训练Walker Walk。用户批准baseline也从零重训，与新模型统一OSMesa。
旧单步TA结果、旧Legacy部分曲线保留，不能拼入本次训练或作为本次终点。

实验目录：`/data/Minko/experiments/spikedreamer-fixed-t8-walker-20260910`，真实运行服务器42（172.27.0.5）。
所有操作先经New-H100-3并进入/data/Minko，再进入42的/data/Minko。

## 实现

`spikedreamer/rssm.py: StatefulFixedRSSM`复用LegacyRSSM的原始参数模块。
`spikedreamer/config.py`新增`stateful_t8`预设；方法名为`stateful`。
不改原有Legacy、TA、binary、GRU的实现或已运行实验的冻结源码。

| 部分 | 本轮定义 |
|---|---|
| 内部计算 | 每次转移固定8步，不使用自适应退出 |
| 输入 | 每个内部步都重复原latent/action输入 |
| 递归 | 第t个内部步仍读取上一转移的第t个deter |
| MCN | 原basal/apical/soma积分顺序、阈值和脉冲后重置 |
| prior/posterior | 保留各自8步LIF及顺序求和读出；不先平均encoder后只更新一次 |
| 额外轨迹状态 | input_mem、soma、basal、apical、prior_mem |
| 真实与想象 | posterior保留完整prior膜状态，下一次obs_step/img_step继续传入 |
| 外围 | 原8步encoder、decoder、reward、continuation、actor和critic |
| 初始化 | 原learned deter及零膜状态，不新增参数 |
| 重置控制 | stateful_t8加persistent=False逐转移清零新增膜状态 |

posterior观察分支的LIF仍逐观察重置。当前观测通过posterior更新stoch，
并在下一次转移影响膜状态，没有额外的“状态一致性loss”，也未直接把所有膜值拼给actor。
episode边界重置完整轨迹状态；replay保留原L64、burn_in0的截断历史规则。
新模型不通过增加隐藏维度、归一化增益或改变行为目标来配合状态保留。

固定步配置要求io_steps与core_steps相等，使用原gain=1、spike readout和脉冲后的树突重置。
旧Legacy的core_norm_gain=2字段从未作用于原始网络，其实际初始化gain也是1。
新批次共享配置明确写gain=1，避免把配置标签误当实际计算差异。
actual manifest中两组只有method不同，其余配置字段一致。

## 已完成的必要检查

检查用旧Walker的真实280K checkpoint/replay，只读作实现验证，未用于新正式训练初始化。
完整B56×L64、512隐藏维度、32×32随机状态、T8、H15，没有替换环境或缩小正式模型。

- 同一seed下，Legacy与stateful初始state_dict的键和值逐项完全一致；均18846903参数。
- 关闭跨转移保留时，完整序列post/prior的stoch、deter、logit与Legacy最大绝对误差均为0。
- 真实prior转移与从相同posterior出发的想象转移一致；完整膜状态进入后续状态。
- 分支调用不污染原状态；episode reset与新episode一致；真实膜状态的保留与清零产生不同后续状态。
- fresh seed0的stateful完成一次完整world/actor/critic优化，loss和梯度有限。
  actual effective_steps及imag_effective_steps均为8。
  单次更新3.823秒，峰值allocated65848101888字节、reserved71772930048字节。
  这是一次全尺寸检查，不是长程吞吐基准，也不是控制性能结果。

完整数值见`/data/Minko/experiments/spikedreamer-fixed-t8-walker-20260910/validation/fixed-t8-real-walker.json`。
没有从这些验证权重或旧replay继续正式训练。正式两组都有独立的新空目录，
启动日志为frames0/updates0，resume=False。

## 新训练

| GPU | 预设 | 任务 | 训练PID | 守护PID |
|---|---|---|---:|---:|
| 4 | legacy | walker_walk | 3533182 | 3622358 |
| 7 | stateful_t8 | walker_walk | 3533181 | 3624440 |

两组于2026-09-10 15:18:16 CST从零启动。以上是启动PID，后续状态以control和实际进程为准。
共同seed0、B56、FP32、L64、H15、ratio512、四环境、action_repeat2，
prefill2500 transitions、pretrain100、1M原始帧，预计71171次完整更新；
每10K帧评估10完整回合。Adam、EMA、阈值训练和format2保存规则沿用已修正trainer。

两组均为独立CPU OSMesa渲染、CUDA训练；渲染worker专用系统libstdc++预加载，
父训练进程不设LD_PRELOAD。相同诊断设置，仅CUDA转储文件按GPU区分。
15:22:57 CST已核对两个父训练进程、8个CPU渲染worker及四个服务；
两组均完成5K随机预填充，进入初始预训练，尚无在线loss/完整评估/checkpoint。
此时日志updates0为预训练前记录，不代表预训练没有执行。
后续启动验收与进度见实验目录STARTUP.md/.json和最新巡检。

GPU5/6不增加本项目任务，现有StarVLA服务未干预。不得称它们物理空闲。

## 旧训练保留情况

旧Legacy Walker于15:14停止，末次日志289K/20385更新，最近保存点280K/19742。
旧Legacy Cheetah于15:10停止，末次日志58K/3885更新，最近保存点50K/3314。
尚未保存的尾段不能从checkpoint恢复；旧日志、checkpoint、replay没有删除。
两组主动停止不记作新失败。已完成的两个单步TA结果保持原样。

旧目录STOPPED_FOR_FIXED_T8_WALKER.json及control已标记stopped_by_user。
旧守护和调度器不再恢复或释放后续队列。新训练只由新目录的两组服务及既有两小时巡检监管。
不启动Cheetah、Cartpole、DMC19、新seed或第三个消融组。

## 结果边界

本轮能先回答单任务、单seed、同T8和同渲染设置下的初步差异。
实现一致性和成功启动不能证明跨状态保留提升控制能力，需等待同1M终点和完整学习曲线。
不根据单个最好评估点替代固定预算终点，不将10个评估回合当作10个训练seed。
