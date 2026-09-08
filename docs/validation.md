# 2026-09-08 实现验证记录

环境：`New-H100-3`，工作目录 `/data/Minko/spike-dreamer`；NVIDIA H100 80GB HBM3；Python 3.10；PyTorch 2.6.0 / CUDA 12.4；dm-control 1.0.9、MuJoCo 2.3.5。没有在本机或其他服务器执行训练。

## 已完成的实际运行

所有训练使用原尺寸：B=16、L=64、512 hidden/deter、32×32 latent、H=15、64×64 RGB。短运行只设置更新上限，数据来自真实 DMC。每个表内训练均完成一次 1000-frame 的真实 episode 评估。

| Run，相对 `runs/` | 模式 | 完成的更新 |
|---|---|---:|
| `validation_ta` | Cartpole，TA FP32 | 3 |
| `validation_legacy_v2` | Cartpole，发布版 T=8，FP32 | 3 |
| `validation_walker_ta` | Walker，TA FP32 | 100，再从 checkpoint 恢复至 105 |
| `validation_ta_bf16` | Walker，TA BF16 | 50 |
| `validation_adaptive_bf16_v3` | Cartpole，自适应 BF16 | 20，再从 checkpoint 恢复至 23 |
| `validation_controls/binary/cartpole_balance/seed-0` | 无膜电位对照 | 3 |
| `validation_controls/gru/cartpole_balance/seed-0` | GRU 核心对照 | 3 |
| `validation_controls/ta_core/cartpole_balance/seed-0` | 单步核心、T=8 外围 | 3 |

Cartpole 参数量：TA 18,840,233，legacy 18,839,213；Walker TA 为 18,847,923。Walker 的图像/奖励模型、actor、critic 均进行了真实优化更新，优化器恢复后继续训练；这里不使用极短期回报证明策略有效。

完整测试命令启用真实 DMC 和真实 checkpoint 后，**31 passed，10.36 秒**。覆盖：

- 逐样本 episode reset、分支隔离、序列因果性、膜电位携带、二值通信和 surrogate 梯度。
- adaptive 提前退出、全步执行、混合 batch 压缩及与逐轨迹执行的一致性。
- 发布版核心适配一致性、λ-return 对齐、replay 不被采样修改、随机数恢复。
- 大而有限梯度裁剪、真实非有限梯度拒绝、dense MAC 不按活动率打折、fused GRU 投影计数。
- 全部 19 个真实 DMC 任务的 EGL 渲染、动作接口和 frame accounting。
- 两次独立加载真实 checkpoint 后，使用相同 replay batch/RNG 完成下一次完整更新，模型及优化结果一致。

两条 warning 来自 dm-control 1.0.9 的 finger 任务对 NumPy 数组转标量的旧写法；测试通过，依赖已锁定 NumPy 1.26.4。`pip check` 无冲突。评估与开环预测命令均已实际生成 JSON 和 MP4。统计 CLI 已确认拒绝将单 seed 短运行包装成非劣结论。

## 同卡计算测量

两个 520-frame、3-update Cartpole checkpoint，在同一物理 GPU 3 上顺序测量；FP32 eager、3 次 warm-up、每项至少 10 秒且至少 5 次。不是训练收敛后的性能或吞吐结论。

| 项目 | legacy T=8 | TA T=1 |
|---|---:|---:|
| RSSM，1024 个 posterior 起点，wall p50 | 5.861 ms | 1.454 ms |
| H=15 imagination，1024 个起点，wall p50 | 126.099 ms | 31.545 ms |
| B=1 policy，wall p50 | 16.056 ms | 4.738 ms |
| 完整训练更新，wall p50 | 2033.346 ms | 490.510 ms |
| 更新阶段 peak allocated，十进制 GB | 20.255 | 5.040 |
| 每次 imagination，原始设备 J | 29.613 | 6.217 |
| 每次训练更新，原始设备 J | 419.636 | 82.073 |
| 每批 H=15 imagination，计入的 dense MAC | 394.679 G | 56.395 G |
| 同批 imagination，神经元更新 | 314.573 M | 39.322 M |

原始数据：[TA profile](results/2026-09-08/ta_profile.json)、[legacy profile](results/2026-09-08/legacy_profile.json)。JSON 还包含 p95、重复次数、逐层计数和完整配置。

训练更新 microbenchmark 约快 4.1 倍；imagination 延迟约下降 75%。这是同尺寸实现的初步计算证据，不能解释为“相同性能下加速已成立”。当前测量包含已加载的优化器与 profiling 输入/状态缓存；显存数字是测量窗口峰值，不是模型权重大小。

设备 J 来自 NVML 累计能量计数器，不是发放率乘以经验常数。未锁定 GPU 时钟、未扣 idle，也不是整机能耗。论文需要训练后 checkpoint、多窗口重复、同卡负载控制与 energy-to-score。这里不报告神经形态器件实测或捏造能耗系数。

## 已修复的问题与剩余边界

已修复：旧核心 bool reset 类型兼容、decoder 维度排列、连续动作任意维度拆分、BF16 动作转 NumPy、概率读出精度、NVML UUID 映射、有限大梯度范数溢出，以及 Adam CPU step tensor 对加载快照的别名修改。

自适应分支的默认阈值在此短运行几乎一直触发 K=4，且裁剪前梯度范数达到约 10²⁵。范数归约修正只解决数值裁剪，不等于解决优化困难。该分支标为实验性，正式主方法仍为固定单步。

checkpoint 恢复会将未完成轨迹保留为截断片段，再开启新的模拟器 episode；不是物理状态逐位恢复。开环数据来自 replay，不能称为测试集泛化结果。

未完成、也未宣称完成：19 任务×3 seeds×1M frames、性能非劣、多 seed 学习曲线、消融有效性、收敛后的能耗与训练总时长。所有长跑输出和模型 checkpoint 留在服务器 `runs/`，未提交到 Git。
