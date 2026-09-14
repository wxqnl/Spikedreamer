# 2026-09-09 共享训练器修正

本次修正纠正 Legacy 与独立发布代码的训练语义差异，并同时应用到 TA。
旧实验 spikedreamer-formal-throughput-20260909 已按用户授权停止，
所有日志、replay、checkpoint 和源码保留。旧结果不作为正确 baseline 的方法对比结果。

## 实现改动

- 在构建 Adam 前启用神经元阈值参数；Legacy 世界模型补回15个阈值，
  actor/critic 各2个。TA 的功能式 LIF/MCN 阈值也按同一规则训练。
  QGate alpha 的发布版 backward 仍不提供梯度。
- 世界模型保持 eps=1e-8、clip=1000；actor/critic 改回 eps=1e-5、clip=100。
- 行为学习对齐发布版：H=15采用14个目标、学习到的 continuation 权重、
  value 输入梯度、行为学习前的 slow-value EMA，以及相同 lambda-return 算法。
- 世界模型各预测头保留独立 feature 分支，KL 各自先归约再加权，
  以匹配发布版 FP32 的反向累加顺序。
- 评估使用随机 posterior 与 mode 动作；每回合固定环境及 posterior seed，
  结束后恢复训练 RNG。
- checkpoint 格式升级为2；禁止使用新训练器恢复或评估格式1的旧 checkpoint。
  旧 checkpoint 应使用对应归档源码。

## 已完成的检查

使用独立导入的 upstream/models.py、networks.py、tools.py，真实 B56×L64 replay、
同一权重和 RNG，以及两边各自新建的 Adam，执行完整 WM+actor+critic 更新。

六项损失完全一致。世界模型82组梯度及97组更新后参数完全一致；
critic 十组梯度及十二组更新后参数完全一致。
actor 梯度最大相对L2差为1.1011e-6，更新后参数最大绝对差为1.86265e-8，
满足预先设置的检查范围。Adam状态、slow value 和 quantile EMA 也通过。
初次检查发现一个卷积权重更新差异，修正计算图/归约顺序后通过；
失败 JSON 保留，检查容差没有放宽。

三组 H100 检查均从 seed0 新权重开始，使用完整 B56、L64、H15、FP32，
完成五次实际更新和恢复优化器/RNG后的第六次更新对照。
三组模型及 Adam 状态有限，各19组阈值均发生更新；恢复对照通过。
已有 dynamics/metrics 测试11项通过。

工程检查代码与证据保存在服务器：

- /data/Minko/experiments/spikedreamer-baseline-audit-20260909/check_training_parity.py
- 同目录 corrected-full-update-parity.json 和 corrected-full-update-parity-v2.json
- /data/Minko/experiments/spikedreamer-corrected-20260909/check_cuda_updates.py
- 同目录 validation/、PROTOCOL.md、MONITOR.md

完整更新对照命令在43的 /data/Minko 执行；输出须使用尚不存在的新文件名：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
.venvs/spikedreamer/bin/python \
experiments/spikedreamer-baseline-audit-20260909/check_training_parity.py \
--source /data/Minko/spike-dreamer \
--checkpoint /data/Minko/experiments/spikedreamer-baseline-audit-20260909/input/legacy-walker/latest.pt \
--output /data/Minko/experiments/spikedreamer-baseline-audit-20260909/parity-repeat.json
```

## 尚未证实的部分

这些检查没有复现原论文的完整学习曲线，也不能证明早先低回报只由上述错误导致。
Legacy 的名称仍是 released architecture + shared trainer；B56、环境封装、
在线调度及固定评估种子等适配仍需按实际配置披露。
新实验使用新目录从零运行，不载入诊断模型或旧 replay；只完成原定三组，
后续扩训继续等待用户决定。
