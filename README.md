# SpikeDreamer

基于 [Spiking-WM](https://github.com/Brain-Cog-Lab/Spiking-WM) 的时间对齐脉冲世界模型：让 RSSM 的神经元状态沿环境时间持续演化，用一次脉冲更新替代每个转移内部的多次重复仿真。

研究目标是在相同交互预算、统一训练协议下保持控制性能并减少计算。当前已实现并验证训练管线与计算测量，**尚未完成 1M-frame、多任务、多 seed 的性能验证**。不是“首个端到端脉冲世界模型”的主张。

## 已实现

- 64×64 RGB → 脉冲编码器 → 显式状态 MCN RSSM → 潜空间想象 → 脉冲 actor/critic；联合训练图像、奖励、continuation、KL 和行为目标。
- `legacy`：发布版 T=8 RSSM；`ta`：持久状态、单步主模型；`ta_core`：只替换动态核心，保留 T=8 编码器及各输出头。
- `binary`、`gru` 动态核心对照；状态重置、膜电位读出消融；真实逐样本压缩执行的自适应步数分支。
- DMC 19 任务、独立任务/seed 调度、replay 与优化器恢复、评估视频、开环预测诊断、配对统计。
- H100 延迟、显存、NVML 设备能耗；独立的 dense MAC、脉冲计数与事件驱动 SynOps 代理账本。

## 安装

已验证 Linux、Python 3.10、PyTorch 2.6.0 / CUDA 12.4、H100。CPU 只适合单元测试；真实训练需要 NVIDIA GPU 和 EGL。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-lock.txt
pip install -e '.[test]'
```

本项目当前服务器位置为 `/data/Minko/spike-dreamer`，环境为 `/data/Minko/.venvs/spikedreamer`。所有项目操作在 `New-H100-3` 上进行。

```bash
ssh New-H100-3
cd /data/Minko/spike-dreamer
source /data/Minko/.venvs/spikedreamer/bin/activate
export MUJOCO_GL=egl
```

若驱动已安装但 EGL 只发现 Mesa，使用仓库的 NVIDIA vendor 配置，无需修改系统文件：

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/data/Minko/spike-dreamer/runtime/10_nvidia.json
```

## 运行

完整尺寸的短运行仅限制更新次数，不替换环境、不缩小模型：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -m spikedreamer.train \
  --preset ta --logdir runs/check-walker \
  --set task=walker_walk frames=4096 prefill=256 pretrain=1 max_updates=100 eval_episodes=1
```

正式单任务配置：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -m spikedreamer.train \
  --preset ta --config configs/dmc_1m.yaml --logdir runs/ta/walker_walk/seed-0 \
  --set task=walker_walk seed=0
```

断点恢复需使用相同配置、相同目录并添加 `resume=True`。恢复模型、三个 Adam、慢 critic、replay 和随机数状态；模拟器从新 episode 开始，不保证逐帧轨迹与不中断运行完全一致。只加载可信的 `.pt` 文件。

```bash
python -m spikedreamer.evaluate --checkpoint runs/check-walker/latest.pt --episodes 10
python -m spikedreamer.profile --checkpoint runs/check-walker/latest.pt \
  --seconds 10 --output runs/check-walker/profile.json
python -m spikedreamer.diagnose --checkpoint runs/check-walker/latest.pt \
  --output runs/check-walker/open-loop.json --video runs/check-walker/open-loop.mp4
```

调度器必须显式指定物理 GPU；每张卡运行一个独立实验，并将 CUDA 与 EGL 绑定到同一编号。先查看任务清单：

```bash
python -m spikedreamer.suite --root runs/dev --suite dev6 \
  --presets legacy ta --seeds 0 1 2 --gpus 0 1 2 3 --frames 200000 --list
```

去掉 `--list` 才会启动。完整默认更新率下，1M frames 不是分钟级实验；预算见 [实验协议](docs/experiments.md)。

## 验证与文档

```bash
python -m pytest
SPIKEDREAMER_TEST_DMC=1 SPIKEDREAMER_TEST_CHECKPOINT=runs/check-walker/latest.pt \
  python -m pytest
```

[设计与实现](docs/design.md) · [实验协议](docs/experiments.md) · [实测记录与限制](docs/validation.md) · [上游来源及修改](docs/upstream.md)

原始发布代码保存在 `upstream/`，训练共用的适配组件在 `spikedreamer/vendor/`。本仓库采用 Apache-2.0；请保留 [NOTICE](NOTICE)、[LICENSE](LICENSE) 并引用 Spiking-WM 原论文。
