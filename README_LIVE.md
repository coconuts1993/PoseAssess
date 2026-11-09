# 实时 3D 骨架原型（webcam）

两台（或更多）webcam → 每路 RTMPose 2D 关键点 → 用 `calibration/Calib.toml` 加权 DLT 三角化 →
One-Euro 平滑 → 实时骨架 + 简单关节角。离线流程（4. Run：Pose2Sim + OpenSim IK）不变。

代码：`poseassess/core/realtime/`（无 Qt），GUI 页面 **3c. Live 3D**，命令行测速 `run_live.bat`。

## 1. 先测速（命令行）

```
run_live.bat --project "D:\Projects\Trial01" --show
```

* 相机来源/分辨率取自 3b. Capture 的设置（`wii/capture.json`），标定取自项目的 `Calib.toml`。
* 每秒打印一行：`fps | latency | 2D 耗时 | 3D 耗时 | 两相机时间差 | 3D 关节数`，结束时打印汇总。
* 常用参数：`--size 1280x720`、`--mode lightweight|balanced|performance`、
  `--device auto|cuda|dml|cpu`、`--det-every 3`（人体检测器每 N 帧跑一次）、
  `--seconds 30`、`--csv live.csv`（每帧时间、角度、3D 点）、`--no-3d`（只测 2D，不需要标定）、
  `--sources 0 1`（直接指定相机编号或视频文件）。

**延迟的含义**：从驱动交出最旧的那一帧，到 3D 结果算好。相机自身曝光 + USB 传输（通常 30–100 ms）
和屏幕刷新要另外加上；GUI 里的 "on screen" 包含了绘制时刻。

## 2. 让 NVIDIA 显卡跑起来（重要）

便携包里装的是 **CPU 版 onnxruntime**，所以默认只会用 CPU。两种方式二选一（在有网的电脑上，
在便携包目录里执行）：

**A. DirectML（最简单，任何 DX12 显卡，无需装 CUDA）**
```
venv\Scripts\python.exe -m pip uninstall -y onnxruntime
venv\Scripts\python.exe -m pip install onnxruntime-directml
```

**B. CUDA（NVIDIA 上通常最快）**
```
venv\Scripts\python.exe -m pip uninstall -y onnxruntime
venv\Scripts\python.exe -m pip install onnxruntime-gpu nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
```
（CUDA/cuDNN 的 DLL 由这些 pip 包提供，程序会自动加载；也可以改为系统安装 CUDA 12 + cuDNN 9。）

装好后 `run_live.bat --help` 上方会打印 `onnxruntime devices: cuda, cpu`（或 `dml, cpu`）。
如果选了 GPU 但实际回退到 CPU，命令行和 GUI 都会给出警告。
这两个包同样能被 Pose2Sim 的离线 2D 阶段使用，不影响原有功能。

## 3. GUI：3c. Live 3D

1. 先在 2. Calibration 完成标定，在 3b. Capture 设好相机来源。
2. 打开 3c. Live 3D，选模型/设备/分辨率，点 Start。
   * 如果 3b. Capture 已经打开了相机，会直接复用这些视频流；否则本页自己打开相机
     （一个 webcam 不能同时被打开两次，录制前请先在本页 Stop）。
3. 左侧显示 fps、延迟、每路相机实际帧率和关节角，右侧是 2D 预览和 3D 骨架（鼠标拖动旋转）。
4. 勾选 "Save CSV" 会把每帧结果写到项目的 `live/` 目录。

## 4. 已知限制（原型）

* 同步方式是"取每台相机的最新帧"，自由运行的 webcam 之间最多差一帧（30 fps 时约 33 ms），
  统计里的 `skew` 就是这个差值；快速动作时会带来几厘米的三角化误差。
* 只跟踪一个人（每台相机里离上一帧最近/最大最可信的人）。
* 关节角是几何角度，用于实时反馈；临床角度仍以 OpenSim IK 为准。
* 直播分辨率与标定分辨率不同时按比例缩放（宽高比必须相同，否则会警告）。
