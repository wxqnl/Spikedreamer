# GPU7渲染路径修复（2026-09-10）

GPU7的纯EGL渲染在不加载PyTorch时也复现GRAPHICS Xid 31及异常图像。
本项目为该组启用独立CPU OSMesa环境进程，GPU7继续进行完整CUDA训练。
默认DMC路径不变；其他正在运行的实验使用原冻结源码，没有重启。

新增spikedreamer.env_process.ProcessDMC提供同步的真实环境传输和子进程清理。
通过SPIKEDREAMER_ENV_PROCESS=1启用；软件后端需要MUJOCO_GL=osmesa、
PYOPENGL_PLATFORM=osmesa、LIBGL_ALWAYS_SOFTWARE=1、GALLIUM_DRIVER=llvmpipe和LP_NUM_THREADS=1。
本服务器Conda的libstdc++版本不足，因此仅渲染子进程设置
SPIKEDREAMER_ENV_LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6；不要给训练进程统一预加载。
运行与离线评估都应使用实验记录中的完整环境。

真实五回合检查中，同OSMesa后端的原进程与新进程逐帧输出完全一致；
四份既有真实轨迹的奖励和终止标志一致。CPU与EGL渲染像素并不相同，
平均绝对差1.357143/255，最大差100/255；不得声称严格同渲染条件的方法对照。
这项修复绕开图形故障，没有修复或重置驱动，也不是硬件健康结论。

完整授权、诊断、检查结果、启动状态及可比性限制见
/data/Minko/experiments/spikedreamer-corrected-20260909/GPU7_RENDER_FIX.md
及GPU7_RENDER_FIX_STATUS.json。前两次失败现场均已保留。

截至2026-09-10 10:36:25 CST，本次正式训练已连续到6000帧/171次更新和7000帧/242次更新，
两条在线记录的loss与梯度均有限，启动后无新GPU7 Xid，训练PID359048未变。
已通过之前预训练后的失败阶段；首个10000帧checkpoint和完整1M预算尚待继续运行。
