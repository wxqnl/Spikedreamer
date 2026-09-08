# 上游来源与适配边界

来源：[Brain-Cog-Lab/Spiking-WM](https://github.com/Brain-Cog-Lab/Spiking-WM)，2026-09-08 获取。原始快照直接包含在 `upstream/`，不是需要再次解析远端状态的子模块。保留原 README、LICENSE、作者信息及致谢。

原论文：Sun, Zhao, Lyu and Zeng, *Spiking world model with multicompartment neurons for model-based reinforcement learning*, PNAS 122(50), e2513319122 (2025), [DOI](https://doi.org/10.1073/pnas.2513319122)，[arXiv](https://arxiv.org/abs/2503.00713)。

## 继承与修改

| 文件 | 处理 |
|---|---|
| `upstream/` | 未修改的原始研究代码、配置、启动脚本与素材 |
| `vendor/node.py` | 继承 LIF/MCN 神经元，只改为包内 import |
| `vendor/surrogate.py` | 继承原 QGate 等 surrogate 实现，未改数学定义 |
| `vendor/normalization.py` | 继承 PopNorm；修复 repr 中不存在的属性名 |
| `vendor/networks.py` | 继承 SNN 编解码器、MLP、actor、发布版 RSSM；包内 import、配置访问、图像维度排列、任意动作维度拆分兼容修正；概率读出保留 FP32 |
| `vendor/tools.py` | 继承分布、初始化及工具；categorical 与离散回报分布在 BF16 模式下仍以 FP32 计算 |

不支持的 proprioceptive encoder 显式报错，不提供假实现。未把上游所有 Atari/DMLab/其他任务包装器搬进新训练器，v0.1 的完整实现范围是视觉连续控制 DMC。

`LegacyRSSM` 是发布版核心的适配器，统一 batch-first 状态轴、隔离原地修改，并兼容 bool episode reset。单元测试将它与相同权重的发布版核心直接比较。它不把旧模型替换成新 MCN 后仍称“官方基线”。

`neurons.py/rssm.py` 的持久状态核心、`model.py` 的统一训练器以及环境、replay、恢复、评估、计数、调度模块为本项目实现。新 MCN 沿用发布代码的非线性胞体积分形式，但状态组织、初始状态与归一化尺度有所变化，见设计文档。

## 复现命名

`legacy` 的准确名称是“Spiking-WM released architecture + shared trainer”。它与 TA 共用动作处理、replay、frame accounting、λ-return、三个优化器、评估和记录逻辑。它不是未改动原训练器的逐字复现，也不能直接拿原论文得分替代本地重跑。

要运行未经改动的原程序，请使用 `upstream/` 的环境说明和 `scripts/train.sh`；本次验证没有宣称原程序在新依赖环境下的完整复现。发布版脚本使用 spike_times=8，而原默认配置是 5，本文主基线显式选 8。

依赖锁定见 `requirements-lock.txt`；每次训练另保存实际版本、配置、GPU 和参数量。保留 Apache-2.0 与原作者致谢；新贡献不应表述为首次提出 SNN world model。
