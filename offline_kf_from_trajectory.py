#!/usr/bin/env python3
"""
离线重放 trajectory_data 中的 detection 轨迹，使用 perception.py 的 KalmanFilter3D 进行滤波。

规则：
1) 主 KF：
   - 有 detection -> update
   - 无 detection -> predict
2) 第二条轨迹（每帧前向预测）：
    - 在每一帧主 KF 完成当帧 update/predict 后
    - 从当前状态向前 predict N 步，记录该时刻的未来位置与速度

输出：
- 每条轨迹一个 *_offline_kf.json，包含两条 KF 的位置与速度序列
- 每条轨迹一个 *_offline_kf_3d.png，3D 轨迹图并标注速度
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

from perception import BallTracker


def _load_tracker_params(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["tracker"]


def _state_or_none(state):
    s = state
    if s is None:
        return None, None
    return s["position"].copy(), s["velocity"].copy()


def _as_list_or_none(v):
    if v is None:
        return None
    return [float(x) for x in v]


def run_one_trajectory(json_path: Path, tracker_params: dict[str, Any], predict_n: int):
    with open(json_path, "r", encoding="utf-8") as f:
        traj = json.load(f)

    frames = traj.get("frames", [])
    frames = sorted(frames, key=lambda x: x.get("frame_index", 0))

    tracker = BallTracker(
        num_balls=1,
        dt=tracker_params["dt"],
        g=tracker_params["g"],
        process_noise=tracker_params.get("process_noise", 0.001),
        measurement_noise=tracker_params.get("measurement_noise", 0.001),
        max_distance=tracker_params.get("max_distance", 0.5),
        drag_coefficient=tracker_params.get("drag_coefficient", 0.0),
        verbose=False,
    )
    ground_z_threshold = tracker_params.get("ground_z_threshold", -0.2)

    main_pos_hist, main_vel_hist = [], []
    det_pos_hist = []
    did_upgrade_hist = []

    future_pos_hist = [None] * len(frames)
    future_vel_hist = [None] * len(frames)

    # 与 zed_tracker_deploy 流程一致，保留清理调用
    kf_obs = [None]
    kf_obs_body = [None]

    for i, fr in enumerate(frames):
        det = fr.get("detection_pos", None)

        # 1) predict（仅对已验证tracker）
        if tracker.is_validated(0):
            tracker.predict_all(ground_z_threshold=ground_z_threshold)

        # 2) cleanup grounded
        tracker.cleanup_grounded_balls(kf_obs, kf_obs_body)

        # 3) detection update（复用 BallTracker 的验证/匹配/更新逻辑）
        before_update_count = tracker.kf_filters[0].update_count
        if det is not None:
            tracker.update([np.asarray(det, dtype=float)])
        after_update_count = tracker.kf_filters[0].update_count
        did_upgrade = after_update_count > before_update_count

        main_state = tracker.get_state(0)
        m_pos, m_vel = _state_or_none(main_state)
        main_pos_hist.append(m_pos)
        main_vel_hist.append(m_vel)
        did_upgrade_hist.append(bool(did_upgrade))

        # 4) 每一帧都从当前主KF状态向前 predict N 步，作为第二条轨迹
        if tracker.kf_filters[0].initialized:
            future_kf = copy.deepcopy(tracker.kf_filters[0])
            for _ in range(max(0, int(predict_n))):
                future_kf.predict()
            f_pos, f_vel = _state_or_none(future_kf.get_state())
            tgt_idx = i + max(0, int(predict_n))
            if 0 <= tgt_idx < len(frames):
                future_pos_hist[tgt_idx] = f_pos
                future_vel_hist[tgt_idx] = f_vel

        det_pos_hist.append(None if det is None else np.asarray(det, dtype=float))

    return {
        "meta": {
            "tracker_id": traj.get("tracker_id"),
            "source_json": str(json_path),
            "num_frames": len(frames),
            "predict_n": int(predict_n),
            "dt": float(tracker_params["dt"]),
            "g": float(tracker_params["g"]),
            "process_noise": tracker_params.get("process_noise", 0.001),
            "measurement_noise": tracker_params.get("measurement_noise", 0.001),
            "drag_coefficient": float(tracker_params.get("drag_coefficient", 0.0)),
            "max_distance": float(tracker_params.get("max_distance", 0.5)),
            "ground_z_threshold": float(ground_z_threshold),
        },
        "frames": [
            {
                "frame_index": int(fr.get("frame_index", i)),
                "timestamp": fr.get("timestamp", None),
                "detection_pos": _as_list_or_none(det_pos_hist[i]),
                "kf_main_pos": _as_list_or_none(main_pos_hist[i]),
                "kf_main_vel": _as_list_or_none(main_vel_hist[i]),
                "main_did_upgrade": bool(did_upgrade_hist[i]),
                "kf_future_pos": _as_list_or_none(future_pos_hist[i]),
                "kf_future_vel": _as_list_or_none(future_vel_hist[i]),
            }
            for i, fr in enumerate(frames)
        ],
    }


def _extract_xyz(seq):
    valid = [v for v in seq if v is not None]
    if not valid:
        return None
    arr = np.asarray(valid, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _annotate_speed(ax, pos_seq, vel_seq, every: int, color: str):
    every = max(1, int(every))
    for i, (p, v) in enumerate(zip(pos_seq, vel_seq)):
        if p is None or v is None:
            continue
        if i % every != 0:
            continue
        speed = float(np.linalg.norm(v))
        ax.text(p[0], p[1], p[2], f"{speed:.2f}", color=color, fontsize=7)


def _draw_speed_and_arrows(
    ax,
    pos_seq,
    vel_seq,
    color: str,
    arrow_len: float = 0.03,
    every: int = 1,
):
    """按步长标注速度并批量绘制方向箭头，避免 3D 绘图对象过多导致卡顿。"""
    every = max(1, int(every))

    arrow_pos = []
    arrow_dir = []
    for i, (p, v) in enumerate(zip(pos_seq, vel_seq)):
        if p is None or v is None:
            continue
        if i % every != 0:
            continue

        speed = float(np.linalg.norm(v))
        ax.text(p[0], p[1], p[2], f"{speed:.2f}", color=color, fontsize=7)

        if speed > 1e-9:
            arrow_pos.append(np.asarray(p, dtype=float))
            arrow_dir.append(np.asarray(v, dtype=float) / speed)

    if arrow_pos:
        pos = np.asarray(arrow_pos, dtype=float)
        direc = np.asarray(arrow_dir, dtype=float)
        ax.quiver(
            pos[:, 0], pos[:, 1], pos[:, 2],
            direc[:, 0], direc[:, 1], direc[:, 2],
            length=arrow_len,
            normalize=True,
            color=color,
            linewidth=1.2,
            alpha=0.9,
        )


def _annotate_position_values(ax, pos_seq, color: str, prefix: str, every: int = 1):
    every = max(1, int(every))
    for i, p in enumerate(pos_seq):
        if p is None:
            continue
        if i % every != 0:
            continue
        ax.text(
            p[0], p[1], p[2],
            f"{prefix}({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})",
            color=color,
            fontsize=7,
        )


def _first_valid_point(seq):
    for p in seq:
        if p is not None:
            return p
    return None


def _scale_seq(seq, scale: float):
    out = []
    for p in seq:
        out.append(None if p is None else (np.asarray(p, dtype=float) * scale))
    return out


def _set_equal_aspect_3d(ax, points: list[np.ndarray]):
    valid = [p for p in points if p is not None]
    if not valid:
        return
    arr = np.asarray(valid, dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _set_axis_ticks(ax, tick_step: float):
    if tick_step <= 0:
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    z0, z1 = ax.get_zlim()

    def _ticks(lo, hi, step):
        start = np.floor(lo / step) * step
        end = np.ceil(hi / step) * step
        return np.arange(start, end + 0.5 * step, step)

    ax.set_xticks(_ticks(x0, x1, tick_step))
    ax.set_yticks(_ticks(y0, y1, tick_step))
    ax.set_zticks(_ticks(z0, z1, tick_step))


def _draw_result_on_ax(
    ax,
    result: dict[str, Any],
    annotate_every: int,
    view_mode: int = 0,
    display_scale: float = 2.0,
    tick_step: float = 0.02,
):
    frames = result["frames"]

    det_seq = [None if f["detection_pos"] is None else np.asarray(f["detection_pos"], dtype=float) for f in frames]
    main_pos_seq = [None if f["kf_main_pos"] is None else np.asarray(f["kf_main_pos"], dtype=float) for f in frames]
    main_vel_seq = [None if f["kf_main_vel"] is None else np.asarray(f["kf_main_vel"], dtype=float) for f in frames]
    main_upgrade_seq = [bool(f.get("main_did_upgrade", False)) for f in frames]
    future_pos_seq = [None if f.get("kf_future_pos") is None else np.asarray(f["kf_future_pos"], dtype=float) for f in frames]
    future_vel_seq = [None if f.get("kf_future_vel") is None else np.asarray(f["kf_future_vel"], dtype=float) for f in frames]

    # main / future / detection 三类信息均按全点显示

    # 仅用于可视化显示的缩放（不改变速度数值）
    det_seq_vis = _scale_seq(det_seq, display_scale)
    main_pos_seq_vis = _scale_seq(main_pos_seq, display_scale)
    future_pos_seq_vis = _scale_seq(future_pos_seq, display_scale)

    ax.clear()

    # mode 0: speed value + speed arrows + positions
    # mode 1: positions + position numeric values
    # mode 2: positions only
    show_positions = True
    show_speed = view_mode == 0
    show_pos_values = view_mode == 1

    det_xyz = _extract_xyz(det_seq_vis)
    if show_positions and det_xyz is not None:
        ax.plot(*det_xyz, "k.", alpha=0.5, label="detection")
        if show_pos_values:
            _annotate_position_values(ax, det_seq_vis, color="black", prefix="D", every=1)

    main_xyz = _extract_xyz(main_pos_seq_vis)
    if show_positions and main_xyz is not None:
        ax.plot(*main_xyz, color="tab:blue", linewidth=2.0, label="KF main (update/predict)")
        # 用点型区分 main 的 update/predict，并批量绘制降低 3D artist 数量
        up_points = [p for i, p in enumerate(main_pos_seq_vis) if p is not None and main_upgrade_seq[i]]
        pred_points = [p for i, p in enumerate(main_pos_seq_vis) if p is not None and (not main_upgrade_seq[i])]
        if up_points:
            up_arr = np.asarray(up_points, dtype=float)
            ax.scatter(up_arr[:, 0], up_arr[:, 1], up_arr[:, 2], color="tab:blue", marker="o", s=26, alpha=0.95)
        if pred_points:
            pred_arr = np.asarray(pred_points, dtype=float)
            ax.scatter(pred_arr[:, 0], pred_arr[:, 1], pred_arr[:, 2], color="tab:blue", marker="x", s=22, alpha=0.9)

        if show_speed:
            _draw_speed_and_arrows(
                ax,
                main_pos_seq_vis,
                main_vel_seq,
                color="tab:blue",
                arrow_len=0.03 * max(display_scale, 1e-6),
                every=1,
            )
        elif show_pos_values:
            _annotate_position_values(ax, main_pos_seq_vis, color="tab:blue", prefix="M", every=1)

    future_xyz = _extract_xyz(future_pos_seq_vis)
    if show_positions and future_xyz is not None:
        horizon_n = int(result.get("meta", {}).get("predict_n", 0))
        ax.plot(*future_xyz, "--", color="tab:orange", linewidth=2.0, label=f"KF predict +{horizon_n} step")
        future_points = [p for p in future_pos_seq_vis if p is not None]
        if future_points:
            future_arr = np.asarray(future_points, dtype=float)
            ax.scatter(future_arr[:, 0], future_arr[:, 1], future_arr[:, 2], color="tab:orange", marker="^", s=24, alpha=0.9)

        if show_speed:
            _draw_speed_and_arrows(
                ax,
                future_pos_seq_vis,
                future_vel_seq,
                color="tab:orange",
                arrow_len=0.03 * max(display_scale, 1e-6),
                every=1,
            )
        elif show_pos_values:
            _annotate_position_values(ax, future_pos_seq_vis, color="tab:orange", prefix="F", every=1)

    # 起点标记
    start_main = _first_valid_point(main_pos_seq_vis)
    if show_positions and start_main is not None:
        ax.scatter(start_main[0], start_main[1], start_main[2], color="lime", marker="*", s=180, label="start_main")
        ax.text(start_main[0], start_main[1], start_main[2], "START_MAIN", color="lime", fontsize=9)

    start_future = _first_valid_point(future_pos_seq_vis)
    if show_positions and start_future is not None:
        ax.scatter(start_future[0], start_future[1], start_future[2], color="gold", marker="*", s=160, label="start_future")
        ax.text(start_future[0], start_future[1], start_future[2], "START_FUTURE", color="gold", fontsize=9)

    # 机器人体坐标系：X前, Y左, Z上
    ax.set_xlabel("X_forward (m)")
    ax.set_ylabel("Y_left (m)")
    ax.set_zlabel("Z_up (m)")

    # 原点坐标轴参考
    axis_len = 0.2
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="r", linewidth=2)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="g", linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="b", linewidth=2)

    all_points = det_seq_vis + main_pos_seq_vis + future_pos_seq_vis
    _set_equal_aspect_3d(ax, all_points)
    _set_axis_ticks(ax, tick_step)

    ax.set_title(
        f"Offline KF Replay | tracker={result['meta']['tracker_id']} | "
        f"predict+N={result['meta'].get('predict_n', 0)}"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def plot_result(
    result: dict[str, Any],
    png_path: Path,
    annotate_every: int,
    display_scale: float = 5.0,
    tick_step: float = 0.05,
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_result_on_ax(
        ax,
        result,
        annotate_every,
        view_mode=0,
        display_scale=display_scale,
        tick_step=tick_step,
    )

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _seq_to_xyz_arrays(seq):
    n = len(seq)
    x = np.full(n, np.nan, dtype=float)
    y = np.full(n, np.nan, dtype=float)
    z = np.full(n, np.nan, dtype=float)
    for i, p in enumerate(seq):
        if p is None:
            continue
        x[i], y[i], z[i] = float(p[0]), float(p[1]), float(p[2])
    return x, y, z


def plot_timeseries_components(result: dict[str, Any], png_path: Path):
    """按真实时间步对比 main/future 的位置与速度分量。"""
    frames = result["frames"]
    n = len(frames)
    t = np.arange(n, dtype=int)

    main_pos_seq = [None if f["kf_main_pos"] is None else np.asarray(f["kf_main_pos"], dtype=float) for f in frames]
    main_vel_seq = [None if f["kf_main_vel"] is None else np.asarray(f["kf_main_vel"], dtype=float) for f in frames]
    future_pos_seq = [None if f["kf_future_pos"] is None else np.asarray(f["kf_future_pos"], dtype=float) for f in frames]
    future_vel_seq = [None if f["kf_future_vel"] is None else np.asarray(f["kf_future_vel"], dtype=float) for f in frames]

    mpx, mpy, mpz = _seq_to_xyz_arrays(main_pos_seq)
    mvx, mvy, mvz = _seq_to_xyz_arrays(main_vel_seq)
    fpx, fpy, fpz = _seq_to_xyz_arrays(future_pos_seq)
    fvx, fvy, fvz = _seq_to_xyz_arrays(future_vel_seq)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    ax_px, ax_py, ax_pz = axes[0, 0], axes[0, 1], axes[0, 2]
    ax_vx, ax_vy, ax_vz = axes[1, 0], axes[1, 1], axes[1, 2]

    def _plot_pair(ax, y_main, y_future, title: str, ylabel: str):
        # 同一时间轴叠加比较，点+线同时显示。
        ax.plot(t, y_main, "-o", color="tab:blue", markersize=3.5, linewidth=1.4, label="main")
        ax.plot(t, y_future, "--^", color="tab:orange", markersize=3.5, linewidth=1.4, label="future")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    _plot_pair(ax_px, mpx, fpx, "Position X", "x (m)")
    _plot_pair(ax_py, mpy, fpy, "Position Y", "y (m)")
    _plot_pair(ax_pz, mpz, fpz, "Position Z", "z (m)")
    _plot_pair(ax_vx, mvx, fvx, "Velocity X", "vx (m/s)")
    _plot_pair(ax_vy, mvy, fvy, "Velocity Y", "vy (m/s)")
    _plot_pair(ax_vz, mvz, fvz, "Velocity Z", "vz (m/s)")

    for ax in axes[1, :]:
        ax.set_xlabel("Time Step")

    if n > 0:
        step = max(1, n // 12)
        xticks = np.arange(0, n, step)
        if (n - 1) not in xticks:
            xticks = np.append(xticks, n - 1)
        for ax in axes.reshape(-1):
            ax.set_xticks(xticks)
            ax.set_xlim(0, max(1, n - 1))

    tracker_id = result.get("meta", {}).get("tracker_id", "?")
    predict_n = result.get("meta", {}).get("predict_n", "?")
    fig.suptitle(f"KF Components vs Time Step | tracker={tracker_id} | predict+N={predict_n}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


class InteractiveTrajectoryBrowser:
    """Matplotlib 3D 交互浏览器：鼠标拖动旋转，A/D 切换轨迹。"""

    def __init__(self, items: list[dict[str, Any]], annotate_every: int, display_scale: float, tick_step: float):
        self.items = items
        self.annotate_every = annotate_every
        self.display_scale = display_scale
        self.tick_step = tick_step
        self.idx = 0
        self.view_mode = 0  # 0: full, 1: positions, 2: none

        # 禁用 Matplotlib 默认 q=quit，避免与自定义 q 切换模式冲突
        if "q" in mpl.rcParams.get("keymap.quit", []):
            mpl.rcParams["keymap.quit"] = [k for k in mpl.rcParams["keymap.quit"] if k != "q"]

        self.fig = plt.figure(figsize=(11, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._redraw()

    def _redraw(self):
        item = self.items[self.idx]
        result = item["result"]
        src = Path(item["source_json"]).name

        _draw_result_on_ax(
            self.ax,
            result,
            self.annotate_every,
            view_mode=self.view_mode,
            display_scale=self.display_scale,
            tick_step=self.tick_step,
        )
        mode_text = {
            0: "SPEED+ARROWS+POSITIONS",
            1: "POSITIONS+VALUES",
            2: "POSITIONS ONLY",
        }[self.view_mode]
        self.ax.set_title(
            f"[{self.idx + 1}/{len(self.items)}] {src}\n"
            f"A: Previous  D: Next  Q: Toggle View ({mode_text})  ESC: Quit"
        )
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ("d", "right"):
            self.idx = (self.idx + 1) % len(self.items)
            self._redraw()
        elif event.key in ("a", "left"):
            self.idx = (self.idx - 1) % len(self.items)
            self._redraw()
        elif event.key == "q":
            self.view_mode = (self.view_mode + 1) % 3
            self._redraw()
        elif event.key in ("escape",):
            plt.close(self.fig)

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Offline KF replay for trajectory_data")
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default=str(Path(__file__).parent / "trajectory_data"),
        help="trajectory_data 目录",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "Tracker_config.yaml"),
        help="Tracker_config.yaml 路径",
    )
    parser.add_argument(
        "--predict-n",
        type=int,
        default=None,
        help="For each frame, predict main KF N steps ahead to draw the second line",
    )
    parser.add_argument(
        "--annotate-every",
        type=int,
        default=4,
        help="Speed label interval in frames",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Export only json/png, do not open interactive 3D window",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: trajectory_data/offline_kf_outputs)",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1,
        help="Visualization scale factor for positions (larger -> points farther apart)",
    )
    parser.add_argument(
        "--tick-step",
        type=float,
        default=0.1,
        help="Axis tick step in visualization",
    )
    args = parser.parse_args()

    # 导出图片阶段统一使用无界面后端，避免 Windows/Tk 在批量绘图时卡住。
    plt.switch_backend("Agg")

    traj_dir = Path(args.trajectory_dir)
    config_path = Path(args.config)

    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    output_dir = Path(args.output_dir) if args.output_dir else (traj_dir / "offline_kf_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker_params = _load_tracker_params(config_path)
    predict_n = int(
        args.predict_n
        if args.predict_n is not None
        else tracker_params.get("predict_n", 8)
    )

    json_files = sorted(traj_dir.glob("trajectory_tracker*_*.json"))
    if not json_files:
        print(f"No trajectory JSON found in: {traj_dir}")
        return

    print(f"Found {len(json_files)} trajectories. Start offline replay...")
    print(f"predict_n = {predict_n}")
    print(f"output_dir = {output_dir}")
    all_items: list[dict[str, Any]] = []
    for jp in json_files:
        result = run_one_trajectory(jp, tracker_params, predict_n)

        main_points = sum(1 for fr in result["frames"] if fr.get("kf_main_pos") is not None)
        future_points = sum(1 for fr in result["frames"] if fr.get("kf_future_pos") is not None)

        out_json = output_dir / f"{jp.stem}_offline_kf.json"
        out_png = output_dir / f"{jp.stem}_offline_kf_3d.png"
        out_png_ts = output_dir / f"{jp.stem}_offline_kf_timeseries.png"

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        plot_result(
            result,
            out_png,
            args.annotate_every,
            display_scale=args.display_scale,
            tick_step=args.tick_step,
        )
        plot_timeseries_components(result, out_png_ts)

        all_items.append({
            "source_json": str(jp),
            "result": result,
        })

        print(f"完成: {jp.name}")
        print(f"  输出轨迹: {out_json.name}")
        print(f"  输出图像: {out_png.name}")
        print(f"  输出时序图: {out_png_ts.name}")
        print(f"  轨迹点数: main={main_points}, future(+{predict_n})={future_points}")

    if not args.no_interactive and all_items:
        print("\nOpen interactive 3D browser: A/D switch, Q toggle view mode, ESC quit.")
        try:
            plt.switch_backend("TkAgg")
            browser = InteractiveTrajectoryBrowser(
                all_items,
                args.annotate_every,
                display_scale=args.display_scale,
                tick_step=args.tick_step,
            )
            browser.show()
        except Exception as e:
            print(f"[warning] Interactive browser unavailable on current backend: {e}")


if __name__ == "__main__":
    main()
