# Walker 第二轮吞吐优化，2026-09-10

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-10
- Verification Status: UNVERIFIED（真实执行与评估一致性已检查；1M 训练结果未完成）
- Version Label: fixed-t8-explicit-state-compiled-parallel-v2

## 已部署的改动

本轮只并行处理独立的 CPU 环境工作，保留第一轮的模型计算实现。

训练时，四个 OSMesa worker 先接收各自的动作，再按原环境顺序收取结果。replay 写入、回合重置和更新额度仍按原顺序处理；接近总帧数上限时使用原串行路径，避免多跑环境帧。

评估最多同时运行四个 CPU worker。每条轨迹仍单独调用 B1 策略，保存自己的 CPU/CUDA/Python/NumPy 随机状态，不合并后验采样。十个回合使用原种子，结果按原回合编号输出。评估入口在 finally 中恢复训练模式与随机状态。评估专用模块按需加载。

正式进程仍使用 `code-speedup-v1/`，其中已部署本轮四个文件的改动；保留该路径以复用既有模型源文件与编译路径。新目录名不是启用条件。`code-speedup-v2/` 是相同最终源码的独立副本，不是当前运行入口。`code-speedup-v1-before-v2/` 保存修改前代码，`runs-before-speedup-v2/` 保存两组 100K 检查点、replay 和日志，`plan-before-speedup-v2.json` 保存原运行计划。更早的备份继续保留。

43 的 `/data/Minko/spike-dreamer` 已同步这四个文件的改动，其他现有修改未覆盖。冻结实验 YAML 仍含 `core_norm_gain=1.0`，不要用主源码的通用 YAML 覆盖它。

## 性能实测

| 项目 | 原实现 | 最终实现 | 加速 |
| --- | ---: | ---: | ---: |
| 四个真实环境共同完成一步 | 177.08 ms | 49.52 ms | 3.58× |
| Legacy 完整十回合评估 | 234.81 s | 126.78 s | 1.85× |
| Stateful 完整十回合评估 | 323.18 s | 120.64 s | 2.68× |

环境结果来自四个 worker 每组 12 步的对照，共 192 个真实原始帧。两个版本得到的图像、奖励、终止标志和帧数完全相同。评估使用同一个 100K/6885 检查点；旧耗时取同帧训练日志至评估日志的墙钟差，新耗时直接包围完整 evaluate 调用。两者均覆盖完整回合，未计入新进程策略图预热。初次测量为 122.89/120.14 秒，最终部署版本复测为表中数值。

上述倍数不是全程训练加速倍数。完整网络更新仍约 1.1 秒，本轮没有声称加速这一计算核心。最终版六次完整更新均值为 Legacy 1.11867 秒、Stateful 1.11707 秒，来自真实检查点与 B56×64 replay batch，包含世界模型、actor、value 三次反向与 Adam 更新，不包含环境和评估。

## 效果与数值边界

两组最终版本各评估十个完整 1000 帧回合，逐回合分数相对旧日志的最大差值都是 0。Legacy 为 14.38883483 ± 6.10585666；Stateful 为 63.84583050 ± 19.09379344。评估前后全部模型与三个优化器状态张量逐项相同，训练随机状态也完全相同。这仅说明相同检查点上的评估结果一致，不是最终训练效果证明。

同一真实检查点、同一 batch 和随机状态下，连续六次完整更新均保持 FP32、T8、有限 loss/梯度、三个 Adam 的原步数加 6，以及相同的 CPU/CUDA 随机状态：

- Stateful 的各项 loss 一致；最大参数差 7.45058e-9，相对 L2 差 6.71e-10，属于舍入量级。
- Legacy 第六次 model loss 为 96.40041351，原参考为 96.39579773；六次后的最大参数差 2.41101e-5、相对 L2 差 1.01026e-6。其差异大于原实现重复运行的舍入差，因此不宣称逐位一致，也不将原因武断归于普通随机噪声。
- 本轮没有更改 model、RSSM、neurons、vendor networks/tools、Adam 实现、编译选项、replay 采样或 RNG 实现。Legacy 微小差异的底层来源尚未定位。没有为消除此差异放宽有限性检查或改变确定性、精度设置；多次更新的原始对照完整保留。

曾把相同计算代码放到新目录，Stateful 六次更新的相对参数差约 1.8e-4 至 2.0e-4；保留原模型路径后降到上述舍入量级。这支持保留既有路径的部署决定，但不能据此断言具体的底层编译原因。

训练协议仍为 T8、FP32、B56、L64、burn_in0、H15、ratio512、seed0、四环境、action_repeat2、1M 原始帧。每 10K 帧仍评估十个完整回合，预计全程 71171 次完整更新。单训练 seed 的十个评估回合不构成多 seed 证据。最终效果与包含评估的全程耗时继续由原预算训练检验。

## 未采用的方案

自动 CUDA Graph Trees 在第二次预热反向的图捕获阶段长时间停留于 gc.collect；第一次预热约 197.68 秒，探测在约 711.86 秒后结束。未建立稳态收益或数值对照，因此恢复原关闭设置。该临时进程不写正式检查点，也不算训练故障。

replay 预分配对照的数组、类型、顺序、边界标志和随机状态均一致，短窗采样约 2.5 倍快，但完整更新仍出现需要区分来源的数值差异。恢复旧采样后差异没有完全消失，所以没有认定采样是原因。本轮最终保留原采样实现，不把这项收益算入部署结果。

LP_NUM_THREADS 从 1 提到 4 的真实渲染对照未得到有意义的速度增益，保持 1。没有切换到 BF16，也没有改动原 TF32 设置，没有降低 T、batch、序列、想象长度、更新比例、回合数或总帧数。

## 续训与监控

两组在各自下一次 100K 原始帧 / 6885 更新的完整保存点停止，模型、优化器、replay 和 RNG 从该保存点恢复。正常的检查点恢复会从新物理回合开始；不声称恢复未保存的物理现场或逐帧复现不间断运行。

2026-09-10 21:16 CST 已重新启动：GPU4 Legacy PID 616987，start_ticks 1014691201，日志 `control/legacy-walker_walk-seed0.resume-3.log`；GPU7 Stateful PID 616988，start_ticks 1014691205，日志 `control/stateful_t8-walker_walk-seed0.resume-2.log`。后续核对实际 control 与 /proc 身份，不能把这些历史 PID 当永久常量。

两组由 `spikedreamer-fixed-t8-monitor-20260910-gpu4/gpu7` 直接管理。user_resumes 均为 2、auto_restarts 均为 0。Legacy 保留一次已处理的早期配置拒绝，failure_count=1；Stateful 为 0。本轮没有新增训练故障。保持 OSMesa、worker 专用 libstdc++、父进程不预加载，以及原 CUDA 诊断设置。

仅使用 42 的 GPU4/7，不启用其他任务或 seed。两小时巡检按原请求恢复；两组均完成 1M、最终检查点、十回合评估与正常退出后汇总，删除巡检，不追加实验、不归档任务。

## 原始证据

`validation/speedup-v2/` 下的主要文件：

- `replay-environment-parity.json`：真实环境与初版采样对照。
- `stateful-replay-layout-parity.json`：Stateful 完整 batch 的逐字段布局与随机状态。
- `legacy-v1.json`、`stateful-v1.json`、`*-v1-repeat.json`：原实现完整更新与重复运行。
- `legacy-v2-final.json`、`stateful-v2-final.json` 及对应 `.state.pt`：最终版六次完整更新与数值差异。
- `legacy-parallel-evaluation-final.json`、`stateful-parallel-evaluation-final.json`：最终版二十个完整回合、随机状态和训练状态检查。
- `automatic-cudagraph-probe.json`、`renderer-threads-parity.json`：未采用探测的结果。

其他中间文件保留以说明筛选过程，不把它们当作最终部署版本的收益。

## 21:22 CST 在线续训检查

两组新进程均已写入 101K、102K 两条训练指标，更新数从保存点的 6885 增至 7028。
全部新增数值指标有限，参数与配置检查仍为 T8/FP32/B56/L64/H15/ratio512/gain1/compile=True。
实际 PID、start_ticks、argv、CUDA 设备和日志目录均匹配；两个 monitor 服务 active。
GPU4/7 各只有本组训练 CUDA 进程，没有图形进程；父进程没有 LD_PRELOAD，OSMesa/LP1 保持不变。

| 训练段 | 加速前最近 30 个间隔中位数 | 加速后当前短窗 | 吞吐增加 |
| --- | ---: | ---: | ---: |
| Legacy | 9.163 fps | 10.695 fps | 16.7% |
| Stateful | 8.337 fps | 10.879 fps | 30.5% |

旧速度扣除同帧完整评估的等待，新速度取启动后的 101K→102K 窗口，分别约 93.50/91.92 秒。
新短窗尚未包含 110K 评估，样本长度也短于旧窗口，因此不把这张表当最终全程倍数或精确 ETA。
原始快照为 `validation/speedup-v2/online-resume.json`。两小时原生巡检已恢复 ACTIVE。
