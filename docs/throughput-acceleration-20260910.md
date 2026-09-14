# Fixed-T8 Walker 加速与续训记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-10
- Verification Status: UNVERIFIED (1M 训练结果待完成；短窗性能、完整更新和在线执行已核对)
- Version Label: fixed-t8-walker-compiled-v1

## 范围

用户要求先停止当前训练、加速，再恢复。只处理 42（172.27.0.5）的 GPU4 Legacy 和 GPU7 Stateful-T8 Walker，均为 seed0。其他 GPU、其他项目和 41 未纳入操作。

实验目录为 /data/Minko/experiments/spikedreamer-fixed-t8-walker-20260910。
原 code/ 保持不变；加速版为 code-speedup-v1/。
停训时的全部运行目录另存为 runs-before-speedup/，原配置另存为 plan-before-speedup.json。
两组保留的合法 checkpoint 都是 10K 原始帧、457 次更新，格式为 2。
原末条训练日志分别为 Legacy 12K/599、Stateful 13K/671；这些未保存部分在续训中重做。

## 实测性能

使用各自 10K checkpoint 的模型、三个 Adam 状态和 RNG，以及真实 Walker replay 的完整 B56×64 batch。
每次计时包含 world model、actor、critic 的前向、反向、裁剪和参数更新。
基线丢弃两次预热，加速版排除首次编译；表内均为随后四次完整更新的均值。

| 模型 | 原实现秒/更新 | 加速版秒/更新 | 倍数 |
|---|---:|---:|---:|
| Legacy T8 | 3.2552 | 1.1634 | 2.80× |
| Stateful T8 | 3.4141 | 1.1694 | 2.92× |

第一次成功执行完整更新的调用分别耗时 272.1 秒和 273.5 秒，其中包含尚未完成的编译。缓存已有前序探测产物，这不是清空缓存后的冷启动测试。
这是短窗、固定真实 batch 的训练更新性能，不含环境交互、定期评估及启动时间，也不代表学习回报改善。
完整在线速度从续训后的 started 分段计算，不能拼接停训前的指标。

原 Stateful 单次更新的 profiler 记录了 150027 次 cudaLaunchKernel 和 616 次 cudaStreamSynchronize。
插桩时 self CPU 总计 4.449 秒、self CUDA 总计 2.155 秒；这些插桩时间不能替代上表的无插桩计时。

## 实现

- 一次性获取整个真实序列的 reset 标记，消除逐观察的 CUDA 条件等待。
- 把同一更新的指标合成一个标量包回传 CPU。
- 实际编译 RSSM 的 obs_step/img_step、编码器、解码器，以及各输出头的 T8 张量计算段；Python 分布对象留在图外。
- 概率参数由原 softmax、sigmoid、正尺度变换构造；编译模式关闭分布构造器的逐次参数检查，保留每次更新的 loss/梯度有限性保护。
- 保留原参数名称和 checkpoint 结构，使用原随机算子 fallback；未开启 TF32 matmul，也未更改已有 cuDNN 设置。

T8、FP32、B56、L64、H15、ratio512、损失、优化器、阈值训练、1M 帧预算及 OSMesa 均保持不变。
未加入自适应提前退出，也未减少真实或想象更新次数。

PyTorch 版本为 2.6.0+cu124。首次探测卡在 AOT 的 far-apart 重计算启发式；
读取两次临时探测进程的栈后，只关闭该编译启发式，未改 PyTorch 安装或模型公式。
探测用 py-spy 0.4.2 独立放在 /data/Minko/.tools/spikedreamer-profile，未改训练虚拟环境。

## 数值与执行检查

同一保存状态和真实 batch 下，关闭编译的重构版通过完整更新对照。
参数最大绝对差分别为 Legacy 5.82e-11、Stateful 7.45e-9。

编译版保持 FP32，但不承诺逐位一致。一次完整更新后的参数相对 L2 差：
Legacy 8.67e-6，Stateful 3.93e-6；最大绝对差分别为 5.21e-5、2.34e-5。
model_loss 为 Legacy 160.558746→160.559113、Stateful 176.362137→176.364761。
三套优化器保存的 step 都正确增加 1，所有返回指标有限，参数仍为 FP32，数量为 18846903。
各组还连续执行了五次完整更新。

每组在真实 DMC 上检查了 4 环境训练动作路径和 B1 评估动作路径，共 26 个原始环境帧。
所有动作有限。四个渲染 worker 均使用 OSMesa、CUDA_VISIBLE_DEVICES 为空且未导入 torch。
该短检查没有完整评估回合，不能计作回报结果。

## 恢复与监管

两组从 10K/457 保存点恢复模型、优化器、replay 与 RNG；物理环境按原续训规则从新回合开始。
原训练调度服务保持退出，当前训练子进程由原名称的两个 monitor 服务直接持有。
plan.json 的 runtime_by_gpu 指向 code-speedup-v1，并包含 compile=True 和编译缓存目录。
GPU4 的有效续训日志为 control/legacy-walker_walk-seed0.recovery-2.log；
GPU7 为 control/stateful_t8-walker_walk-seed0.resume-1.log。
PID 与 start_ticks 以 control 状态及事件为准。

Legacy 首次启动在加载训练状态前被严格配置校验拒绝：
新副本漏了实验冻结 YAML 的 core_norm_gain=1.0。
已补回原文件的两行，两个冻结 YAML 的 diff 为空，再走有理由的 reviewed recovery。
保留该启动错误的原日志和 failure_count=1；它不是 NaN、GPU 故障或实际更新失败。
Stateful failure_count=0；两组 auto_restarts=0。用户主动停训另记，不算失败。

后续仍只完成当前两组的 1M 帧；不追加任务、seed 或降 T 实验。
每 10K 帧继续评估十个完整回合，沿用 checkpoint 保存和守护边界。
最终分数仍属于单训练 seed，不把十个评估回合当成十个训练 seed。

## 原始记录

以下均在实验目录 validation/：

- legacy-speedup-baseline.json、stateful_t8-speedup-baseline.json
- legacy-speedup-compiled-v1.json、stateful_t8-speedup-compiled-v1.json
- legacy-acceleration-validation.json、stateful_t8-acceleration-validation.json
- stateful-speedup-baseline-trace.json

## 16:57 CST 在线核对

有效续训进程为 GPU4 PID3530840、GPU7 PID2824726，两个守护均在运行。
Legacy 已到 12K/599，近 109.98 秒为 9.093 帧/秒、0.6456 更新/秒；
Stateful 已到 14K/742，近 328.58 秒为 9.130 帧/秒、0.6513 更新/秒。
两组当前 loss 和梯度范数均有限，manifest 确认 compile=True、FP32、B56、T8、gain1 和18846903参数。
这些速率避开启动阶段，尚未包含本次续训的20K完整评估开销。
两小时巡检已恢复。此记录按 ARS 分开性能与控制结果，文字经 stop-slop 精简。
