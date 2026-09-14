# 当前实现与结果验收

更新：2026-09-14。此页区分已完成训练的终点验收、本次代码整理检查和历史测试，不把三者混成一次新的完整实验。

## 本次整理做了什么

以已完成 Gated Memory 1M 帧运行的冻结代码为来源，将模型、回放、CPU 环境进程和编译路径同步到主仓库。保留原已训练模块的参数名称与数学实现，新增独立结果导出器和 OSMesa 启动包装，更新研究文档与许可证说明。

整理前的工作树已在服务器保留副本。没有删除旧实验、改写训练检查点、恢复提前停止的探索，或启动新 GPU 训练。

## 本次实际检查

| 检查 | 结果 | 覆盖边界 |
|---|---|---|
| 主服务器 CPU 单元测试 | 11 passed，20 skipped | 使用已有测试；19 个真实 DMC 测试和 1 个完整 GPU 检查点测试未启用 |
| 主服务器 Walker + OSMesa | 未通过 | 系统未提供 OSMesa，OpenGL 初始化失败；未进入模型推理或训练 |
| 原训练服务器，同份整理代码，CPU Walker | 1 passed，19 deselected | 真实 reset、step、64×64 图像和 frame accounting；无替代环境 |
| 原训练服务器，最终 Gated 检查点 CPU 加载 | 通过 | 格式 2、completed、1M frames、71,171 updates；严格加载全部模型及三个 Adam |
| 参数量核对 | 一致 | 总参数 19,896,503；可训练参数 18,715,060 |
| 原始档案导出 | 通过 | 五组共 500 个完整评估点、5000 个逐回合分数，与终点档案一致 |
| OSMesa 包装脚本语法 | 通过 | `bash -n`，不据此声称系统库可用 |
| 学习曲线可视检查 | 通过 | 服务器渲染 SVG；坐标、五组曲线、图例及回合标准差说明无裁切 |
| 入口与配置 | 通过 | 训练 CLI 可加载；五组 preset 与共用配置保持 B56/L64、T、宽度和 burn-in 差异 |
| 文档与发布文件检查 | 通过 | 13 份 Markdown 无失效本地文件链接；差异无空白错误，未检出常见凭证模式或超过 10 MB 的候选文件 |

最终 Gated 检查点加载后，世界模型、Actor、Value 优化器分别包含 84、10、10 个参数状态条目。该检查只在 CPU 内存中加载可信文件，没有调用回放恢复、执行训练更新或回写检查点。它证明整理后的模型参数键与优化器结构兼容，不证明新的训练轨迹会逐位相同。

默认测试仍以既有核心和工程语义为主；“11 passed”不代表每个新门控机制都拥有独立单元测试，也不是机制有效性证据。没有为凑测试数量添加模拟环境或缩小网络的训练脚本。

### 可复用检查命令

在准备好依赖的仓库根目录执行：

```bash
# 默认只运行 CPU 单元测试。
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -p no:cacheprovider

# 需要系统 OSMesa；仅启用 Walker 的真实环境测试，不启用完整 GPU 更新测试。
CUDA_VISIBLE_DEVICES="" SPIKEDREAMER_TEST_DMC=1 \
  SPIKEDREAMER_ENV_LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 \
  bash runtime/with-osmesa.sh python -m pytest \
  tests/test_real_integration.py -k walker_walk -p no:cacheprovider

# 从已有公开数值重画图表，不运行模型。
python -m spikedreamer.report \
  --data results/walker_seed0.json --output results --readme README.md
```

只有路径确实存在时才设置示例中的 `libstdc++`。不要通过给训练父进程添加 `LD_PRELOAD` 绕过环境问题。依赖缺失是平台准备条件，不应隐藏成测试跳过后的“全部通过”。

## 完成实验的终点验收

五组 1M 完成记录保存了最终配置、帧数、更新数、进程退出码、完整末点评估和检查点信息。最终 Gated Memory 的独立终点验收还检查了模型、三个 Adam、slow-value、回放文件引用和 RNG，确认训练进程退出码为 0，10 个完整回合回报为 925.9576 ± 19.9881。

这些验收来自实验完成时的档案；本次整理只复核上述导出与兼容性，不重跑五组 1M，也不把旧验收描述为全套重新执行。原始文件关系见 [实验协议](experiments.md#原始证据与导出)。

Gated Memory 的外部监控程序在成功写出最终档案后发生过格式化 `TypeError`。该错误发生在训练完成、检查点和结果已写出之后。保留这一记录，但不能把它算成训练失败，也不能用监控报错否定已核对的终点结果。

## 旧记录的用途

- [2026-09-08 实现验证](validation-20260908.md)：早期 TA/Legacy 的 B16 短训练、19 任务环境检查及短 checkpoint profile；不是当前主实验。
- [2026-09-09 训练器修正](training-correction-20260909.md)：原行为目标、优化器与训练语义修正，说明格式 1 旧 checkpoint 的使用边界。
- [2026-09-10 固定 T8](fixed-t8-walker-20260910.md)：当时的结构和启动记录，后续完成数值以当前结果 JSON 为准。
- [2026-09-10 渲染诊断](gpu7-render-fix-20260910.md)、[吞吐第一阶段](throughput-acceleration-20260910.md)、[吞吐第二阶段](throughput-acceleration-v2-20260910.md)：解释历史运行路径变化，不代表当前 Gated Memory 已完成速度对照。

旧日志中的阶段性回报、估计耗时和当时“主方法”名称只适用于其日期。当前主方法为 `stateful_gatedmem32`，主实验为 B56 固定 T8 Walker seed 0。

## 尚未验收的研究主张

没有完成多训练种子、多任务泛化、同 burn-in 机制消融、参数/计算配平、记忆干预、非劣性或收敛后能耗测试。已有真实训练与检查点验收说明项目可继续研究，不说明算法贡献已经被隔离或论文已达到投稿条件。
