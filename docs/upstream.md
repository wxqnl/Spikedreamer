# 上游来源与适配边界

本仓库包含继承的研究组件和本项目新增实现。原始 SNN 世界模型、门控神经元的既有思想以及当前工程改动应分别归属。

## Spiking-WM

来源：[Brain-Cog-Lab/Spiking-WM](https://github.com/Brain-Cog-Lab/Spiking-WM)，初始快照于 2026-09-08 获取。原 README、许可证、作者信息及致谢保留在 [upstream/](../upstream/)；目录为随仓库保存的文件，不是子模块。

原论文：Yinqian Sun, Feifei Zhao, Mingyang Lyu and Yi Zeng. *Spiking world model with multicompartment neurons for model-based reinforcement learning*. PNAS 122(50), e2513319122 (2025). [DOI](https://doi.org/10.1073/pnas.2513319122)，[arXiv](https://arxiv.org/abs/2503.00713)。

| 路径 | 继承与处理 |
|---|---|
| `upstream/` 中原 Spiking-WM 文件 | 原始研究代码、配置、启动脚本与素材保留 |
| `spikedreamer/vendor/node.py` | 原 LIF/MCN 神经元，包内 import 适配 |
| `spikedreamer/vendor/surrogate.py` | 原 QGate 等 surrogate，数学定义不变 |
| `spikedreamer/vendor/normalization.py` | 原 PopNorm；修正 repr 中不存在的属性 |
| `spikedreamer/vendor/networks.py` | SNN 编解码器、MLP、Actor、发布版 RSSM；包内 import、配置访问、图像轴、动作维度、概率精度及区域编译适配 |
| `spikedreamer/vendor/tools.py` | 分布、初始化和工具；概率读出精度与训练工具适配 |

`LegacyRSSM` 统一 batch-first 状态轴，隔离原地修改，并兼容 bool episode reset。它的准确名称是 “Spiking-WM released architecture + shared trainer”，不是未经修改原训练器的逐位复现。发布版启动脚本使用 T=8，原默认配置为 5；本地主表明确使用 T8。不能以原论文得分代替本地训练结果。

发布版已有跨转移的 spike-indexed recurrence 和顶树突对胞体的门控。本项目不声称原模型没有时间记忆，也不将已有的 MCN、SNN world model 或该胞体门控归为新贡献。

## 完整 ANN 基线

来源：[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)。相关原文件与 MIT 许可证保留在 [upstream/ann-dreamer/](../upstream/ann-dreamer/)，作为独立来源，不属于上面的 Spiking-WM 快照。

- `spikedreamer/vendor/ann_networks.py`：派生 ANN CNN、MLP 和 GRU/RSSM 网络，保留来源声明；适配包内工具、共同配置和编译路径。
- `spikedreamer/ann.py`：本项目的共享接口适配。
- `ann_gru`：完整 ANN 编码器、解码器、先验/后验读出、Actor、Value 与 GRU 核心，在相同的新训练器中训练。

这不是原 ANN 仓库训练入口的完整复现，也不同于仅替换递归核心、仍使用 SNN 外围的 `gru` 历史对照。当前 ANN 成绩来自本地 Walker seed 0 重跑。请保留 [ANN MIT 许可证](../upstream/ann-dreamer/LICENSE) 和 [NOTICE](../NOTICE)，不要把 ANN 派生文件重新标注为只有 Apache-2.0 的来源。

## 门控动机

Lang Qin, Ziming Wang, Runhao Jiang, Rui Yan and Huajin Tang. *GRSN: Gated Recurrent Spiking Neurons for POMDPs and MARL*. AAAI 39(2), 1483–1491 (2025). [DOI](https://doi.org/10.1609/aaai.v39i2.32139)，[论文预印本](https://arxiv.org/abs/2404.15597)。

该工作中的 GRSN 研究输入依赖遗忘与互补写入。本项目参考这一动机，具体实现为 MCN 双树突的输入依赖保留、T8 内部积分和世界模型状态传递；它不是 GRSN 方法复现。研究边界与待补比较见 [研究计划](research-roadmap.md)。

## 本项目实现

`neurons.py/rssm.py` 的显式状态组织、Stateful-T8、LIF-T8、Static Slow Memory 和 Gated Memory 适配，以及 `model.py` 的共享训练器、环境进程、真实前缀回放、恢复、评估、记录和结果导出由本项目实现或组织。新 MCN 保留发布版非线性胞体积分和 surrogate，新增状态边界、树突记忆更新和门投影，见 [设计文档](design.md)。

当前视觉连续控制范围为 DMC；没有把上游所有 Atari、DMLab 和其他任务包装器搬入新训练器。不支持的 proprioceptive encoder 会显式报错，不提供替代实现。

2026-09-14 的主仓库整理以已完成 Gated Memory 1M 的冻结代码为来源，保留原有历史 preset 和已训练参数名称。早期结果仍对应各自冻结实现，不能把代码同步说成重新完成所有实验。

## 训练语义修正与许可证

2026-09-09 共享训练器修正了阈值参数遗漏、Actor/Value 优化器配置、行为目标和评估采样；与独立发布代码的真实完整更新对照记录见 [训练器修正](training-correction-20260909.md)。该项检查不等同于全学习曲线复现。格式 1 旧 checkpoint 不允许在修正后的训练器中直接恢复。

项目主许可证为 [Apache-2.0](../LICENSE)，ANN 派生部分保留 MIT 来源。第三方组件继续遵循各自许可证和声明。经核对的论文条目见 [references.bib](references.bib)；本项目当前没有已发表论文的引用条目。
