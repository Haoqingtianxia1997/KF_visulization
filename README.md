# Offline KF Trajectory Replay

该项目用于离线重放 `trajectory_data` 下的轨迹 JSON，调用 `perception.py` 中的 `BallTracker` / `KalmanFilter3D` 进行滤波，并导出结果与可视化图。

## 目标脚本

- [offline_kf_from_trajectory.py](offline_kf_from_trajectory.py)

## 功能说明

脚本会对每条轨迹执行以下流程：

1. **主 KF**
   - 有 `detection_pos`：执行 `update`
   - 无 `detection_pos`：执行 `predict`
2. **未来轨迹（每帧前向预测）**
   - 主 KF 在每帧完成当帧 `update/predict` 后
   - 从当前状态向前 `predict N` 步（`predict_n`），记录该未来时刻的位置与速度

## 输入与输出

### 输入

- 轨迹目录（默认 `trajectory_data/`）中的 `trajectory_tracker*_*.json`
- 配置文件（默认 `Tracker_config.yaml`）中的 `tracker` 参数

### 输出

默认输出到 `trajectory_data/offline_kf_outputs/`：

- `*_offline_kf.json`：每帧记录检测点、主 KF、future KF（N步前向预测）的位置/速度
- `*_offline_kf_3d.png`：3D 轨迹图（可含速度箭头与数值）

## 环境安装

建议 Python 3.9+。

```bash
pip install -r requirements.txt
```

## 运行方法

在项目根目录运行：

```bash
python offline_kf_from_trajectory.py
```

常用参数示例：

```bash
python offline_kf_from_trajectory.py \
  --trajectory-dir trajectory_data \
  --config Tracker_config.yaml \
   --predict-n 4 \
  --annotate-every 4 \
  --display-scale 1.0 \
  --tick-step 0.1
```

仅导出结果、不弹交互窗口：

```bash
python offline_kf_from_trajectory.py --no-interactive
```

## 主要参数

- `--trajectory-dir`：轨迹 JSON 所在目录
- `--config`：配置文件路径（读取 `tracker` 段）
- `--predict-n`：每帧从主 KF 状态向前预测 N 步
- `--annotate-every`：速度文本标注帧间隔
- `--no-interactive`：仅导出 JSON/PNG，不打开交互窗口
- `--output-dir`：输出目录
- `--display-scale`：可视化坐标缩放系数（仅显示缩放）
- `--tick-step`：坐标轴刻度步长

## 交互界面操作

当未使用 `--no-interactive` 时：

- `A` / 左方向键：上一条轨迹
- `D` / 右方向键：下一条轨迹
- `Q`：切换显示模式
- `ESC`：退出

## 备注

- `offline_kf_from_trajectory.py` 依赖 [perception.py](perception.py) 中的 `BallTracker`。
- 若缺少 OpenCV、SciPy 等依赖，导入 `perception.py` 会失败。请确保按 `requirements.txt` 安装。