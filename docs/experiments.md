# 实验协议与运行预算

## 固定协议

主表：DMC 19 visual-control 任务，RGB 64×64，1M 原始环境 frames，action repeat=2，seeds 0/1/2，每次评估使用固定 held-out 初始条件的 10 个完整 episodes。任务列表定义于 `spikedreamer/env.py`。

所有配对方法固定 B=16、L=64、hidden/deter=512、32×32 categorical latent、H=15、train_ratio=512、4 个环境。4 个模拟器在同一进程中轮流 step，不声称四个并行 CPU worker。正式结果 `max_updates=0`，默认 FP32。

帧数包含随机 prefill；`prefill=2500` 的单位是 agent transitions，`train_ratio` 是每个新增 transition 消费的训练时间步数量。每个梯度更新消费 B×L=1024 个时间步，通常每两个 agent transitions 更新一次；burn-in 不计入受监督 L。

## 先筛选，再扩展

| 阶段 | 任务及预算 | 退出条件 |
|---|---|---|
| 实现检查 | Cartpole / Walker，完整尺寸、少量更新 | 有限 loss/梯度、恢复正确、真实评估完成 |
| 方向筛选 | Cartpole Balance、Walker Walk、Cheetah Run；200K frames；1 seed | 没有明显学习崩溃，再增加 seeds |
| 机制开发 | DEV6；500K frames；3 seeds | 固定单步接近 T=8，同时优于 T=1/reset 对照 |
| 主结果 | 19 任务；1M frames；3 seeds | 满足预先约定的性能与计算门槛 |
| 消融 | DEV6；1M frames；3 seeds | 区分 T、持久膜电位、读出与归一化尺度的贡献 |

DEV6 为 Cartpole Balance、Cartpole Swingup Sparse、Cup Catch、Finger Spin、Walker Walk、Cheetah Run。自适应分支只有在固定单步表现不足且开发集验证有效后才扩展；当前不建议作为默认大规模训练配置。

不要用本文仓库的短验证回报选择最终算法，也不要把其中单 episode 当作 seed 统计。

## 运行方法

```bash
python -m spikedreamer.suite --root runs/screen --tasks cartpole_balance walker_walk cheetah_run \
  --presets legacy ta --seeds 0 --gpus 0 1 2 3 4 5 --frames 200000
```

重复运行需添加 `--resume`；调度器拒绝覆盖已有实验，跳过已有相同预算的 completed run，失败时停止本次调度器启动的其他子进程。日志在 `_workers/`，训练文件在 `preset/task/seed-N/`。`update_limit` 表示人为限制更新次数，不是预算完成。

关键单项消融应使用独立目录：

```bash
python -m spikedreamer.train --preset legacy --logdir runs/legacy-t1/walker/seed-0 \
  --set task=walker_walk io_steps=1 core_steps=1
python -m spikedreamer.train --preset ta --logdir runs/reset/walker/seed-0 \
  --set task=walker_walk persistent=False
python -m spikedreamer.train --preset ta --logdir runs/gain1/walker/seed-0 \
  --set task=walker_walk core_norm_gain=1
```

另跑 `ta_core`、`binary`、`gru`、`readout=membrane`。外围 T 的变化会影响 encoder/decoder/actor/critic 成本，不能把整体加速完全称为 RSSM 机制收益。官方原训练入口另保存在 `upstream/`；主对照 `legacy` 使用相同的新训练器，论文中应准确命名。

## 性能统计

主指标是任务均值等权聚合后的 candidate / baseline 比值。使用 task、训练 seed 两层配对 bootstrap 的 95% 区间；下界 ≥0.95 才满足预设 5% 相对非劣门槛。19 任务中退化超过 10% 的任务至多 2 个，作为额外 guardrail。

5% 相对 margin 不等于固定 -50 分；评估 episodes 不能当作独立训练 seeds。统计脚本要求至少每任务 3 个配对 seeds、相同任务集合与交互/训练协议，且必须存在指定帧数的实际评估。

```bash
python -m spikedreamer.aggregate \
  --baseline runs/main/legacy/*/seed-* --candidate runs/main/ta/*/seed-* \
  --frames 1000000 --output runs/main/comparison.json
```

保留完整学习曲线、任务级得分与多 seed 方差；开环诊断只评估 replay 内真实序列，默认 5 步上下文及 1/5/15/50 步预测，自动排除跨 episode 的预测。它不是 held-out 泛化或控制成功的替代指标。

## 计算与能耗

在同一张空闲 GPU 上顺序测量同尺寸、同精度的 checkpoint。先 warm-up，再独立测量 RSSM 一步、H=15 imagination、真实输入的 B=1 policy、完整训练更新；必要时使用 `--parts world_update behavior_update` 拆分。延迟测量期间不注册计数 hooks。重复时间窗并报告 p50/p95，不把单次开发测量当作论文置信区间。

三类数字分别报告：

1. Dense MAC：Linear/Conv/ConvTranspose/GRU 投影的实际形状账本，不按发放率打折；不含反向、优化器、归一化等全部 FLOPs。
2. H100 测量：CUDA/wall 时间、峰值显存、NVML 设备能耗。原始 J 包含 idle draw，且不含整机 CPU/内存；`--idle-watts` 只接受同设备另测的 idle 值。训练 microbenchmark 的优化器只修改内存副本，不保存回 checkpoint。
3. 神经形态代理：二值输入 Linear 的活动连接数，Conv 使用平均活动率近似；不是 dense GPU 的实际跳零运算。`--mac-pj`、`--ac-pj`、`--neuron-pj` 和 `--coefficient-source` 可生成指定系数的算术能耗估算，系数必须一起提供并解释来源。该估算不含访存和未计数算子，不能写成实测器件总能耗。

主研究目标：性能非劣，imagination 的 GPU 延迟或设备能耗至少下降 20%，参数增幅不超过 5%。更强目标是 energy-to-score、整段训练时间也改善；这些需要完整训练记录，短 profile 不能证明。

## 基于当前实测的预算

同卡 FP32 开发测量：完整更新约 TA 0.49 秒、legacy 2.03 秒。1M frames 的主协议约有 248,850 次更新，仅更新计算就约 **TA 34 GPU-hours、legacy 141 GPU-hours / task / seed**，还需加采样、评估和 checkpoint 时间。

这是短 checkpoint 的线性外推，不是完成训练的实测时长。显存不是主要问题，更新次数和多任务多 seed 才是。19×3×两个方法的正式矩阵约一万 GPU-hours 量级，不应依据“小模型”就直接全量启动。当前交付只进行了有明确更新上限的实现验证，没有启动该矩阵。

正式跑分前先测 10K–20K frames 的稳定吞吐；如需降低 `train_ratio`，应对所有方法统一调整、单列协议，不与原 512 协议混报。
