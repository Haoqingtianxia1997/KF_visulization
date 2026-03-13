#!/usr/bin/env python3
"""
离线重放 trajectory_data 中的 detection 轨迹，使用 perception.py 的 KalmanFilter3D 进行滤波。

规则：
1) 主 KF：
   - 有 detection -> update
   - 无 detection -> predict
2) 影子 KF（克隆 KF）：
   - 当主 KF 的 update 次数达到 N 时克隆当前状态
   - 克隆后只做 predict，不再 update

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


def run_one_trajectory(json_path: Path, tracker_params: dict[str, Any], clone_update_n: int):
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

    clone_kf = None
    main_upgrade_count = 0

    main_pos_hist, main_vel_hist = [], []
    clone_pos_hist, clone_vel_hist = [], []
    det_pos_hist = []
    did_upgrade_hist = []

    # 与 zed_tracker_deploy 流程一致，保留清理调用
    kf_obs = [None]
    kf_obs_body = [None]

    for fr in frames:
        det = fr.get("detection_pos", None)

        # 1) predict（仅对已验证tracker）
        if tracker.is_validated(0):
            tracker.predict_all(ground_z_threshold=ground_z_threshold)

        # 2) cleanup grounded
        tracker.cleanup_grounded_balls(kf_obs, kf_obs_body)

        clone_exists_before = clone_kf is not None

        # 3) detection update（复用 BallTracker 的验证/匹配/更新逻辑）
        before_update_count = tracker.kf_filters[0].update_count
        if det is not None:
            tracker.update([np.asarray(det, dtype=float)])
        after_update_count = tracker.kf_filters[0].update_count
        did_upgrade = after_update_count > before_update_count
        if did_upgrade:
            main_upgrade_count += 1

        # 4) 达到N次upgrade后克隆（之后只predict）
        if clone_kf is None and main_upgrade_count >= clone_update_n and tracker.kf_filters[0].initialized:
            clone_kf = copy.deepcopy(tracker.kf_filters[0])

        main_state = tracker.get_state(0)
        m_pos, m_vel = _state_or_none(main_state)
        main_pos_hist.append(m_pos)
        main_vel_hist.append(m_vel)
        did_upgrade_hist.append(bool(did_upgrade))

        if clone_kf is None:
            clone_pos_hist.append(None)
            clone_vel_hist.append(None)
        else:
            if clone_exists_before:
                clone_kf.predict()
            c_pos, c_vel = _state_or_none(clone_kf.get_state())
            clone_pos_hist.append(c_pos)
            clone_vel_hist.append(c_vel)

        det_pos_hist.append(None if det is None else np.asarray(det, dtype=float))

    return {
        "meta": {
            "tracker_id": traj.get("tracker_id"),
            "source_json": str(json_path),
            "num_frames": len(frames),
            "clone_update_n": int(clone_update_n),
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
                "kf_clone_pos": _as_list_or_none(clone_pos_hist[i]),
                "kf_clone_vel": _as_list_or_none(clone_vel_hist[i]),
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


def _draw_speed_and_arrows(ax, pos_seq, vel_seq, color: str, arrow_len: float = 0.03):
    """在每个有效点标注速度大小并绘制方向箭头。"""
    for p, v in zip(pos_seq, vel_seq):
        if p is None or v is None:
            continue
        speed = float(np.linalg.norm(v))
        # 速度数值
        ax.text(p[0], p[1], p[2], f"{speed:.2f}", color=color, fontsize=7)

        # 方向箭头
        if speed > 1e-9:
            d = v / speed
            ax.quiver(
                p[0], p[1], p[2],
                d[0], d[1], d[2],
                length=arrow_len,
                normalize=True,
                color=color,
                linewidth=1.2,
                alpha=0.9
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
    clone_pos_seq = [None if f["kf_clone_pos"] is None else np.asarray(f["kf_clone_pos"], dtype=float) for f in frames]
    clone_vel_seq = [None if f["kf_clone_vel"] is None else np.asarray(f["kf_clone_vel"], dtype=float) for f in frames]

    # 仅用于可视化显示的缩放（不改变速度数值）
    det_seq_vis = _scale_seq(det_seq, display_scale)
    main_pos_seq_vis = _scale_seq(main_pos_seq, display_scale)
    clone_pos_seq_vis = _scale_seq(clone_pos_seq, display_scale)

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
        # 用点型区分 main 的 update/predict
        for p, is_up in zip(main_pos_seq_vis, main_upgrade_seq):
            if p is None:
                continue
            if is_up:
                ax.scatter(p[0], p[1], p[2], color="tab:blue", marker="o", s=26, alpha=0.95)
            else:
                ax.scatter(p[0], p[1], p[2], color="tab:blue", marker="x", s=22, alpha=0.9)

        if show_speed:
            _draw_speed_and_arrows(
                ax,
                main_pos_seq_vis,
                main_vel_seq,
                color="tab:blue",
                arrow_len=0.03 * max(display_scale, 1e-6),
            )
        elif show_pos_values:
            _annotate_position_values(ax, main_pos_seq_vis, color="tab:blue", prefix="M", every=1)

    clone_xyz = _extract_xyz(clone_pos_seq_vis)
    if show_positions and clone_xyz is not None:
        ax.plot(*clone_xyz, "--", color="tab:orange", linewidth=2.0, label="KF clone (predict only)")
        for p in clone_pos_seq_vis:
            if p is not None:
                ax.scatter(p[0], p[1], p[2], color="tab:orange", marker="^", s=24, alpha=0.9)

        if show_speed:
            _draw_speed_and_arrows(
                ax,
                clone_pos_seq_vis,
                clone_vel_seq,
                color="tab:orange",
                arrow_len=0.03 * max(display_scale, 1e-6),
            )
        elif show_pos_values:
            _annotate_position_values(ax, clone_pos_seq_vis, color="tab:orange", prefix="C", every=1)

    # 起点标记
    start_main = _first_valid_point(main_pos_seq_vis)
    if show_positions and start_main is not None:
        ax.scatter(start_main[0], start_main[1], start_main[2], color="lime", marker="*", s=180, label="start_main")
        ax.text(start_main[0], start_main[1], start_main[2], "START_MAIN", color="lime", fontsize=9)

    start_clone = _first_valid_point(clone_pos_seq_vis)
    if show_positions and start_clone is not None:
        ax.scatter(start_clone[0], start_clone[1], start_clone[2], color="gold", marker="*", s=160, label="start_clone")
        ax.text(start_clone[0], start_clone[1], start_clone[2], "START_CLONE", color="gold", fontsize=9)

    # 机器人体坐标系：X前, Y左, Z上
    ax.set_xlabel("X_forward (m)")
    ax.set_ylabel("Y_left (m)")
    ax.set_zlabel("Z_up (m)")

    # 原点坐标轴参考
    axis_len = 0.2
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="r", linewidth=2)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="g", linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="b", linewidth=2)

    all_points = det_seq_vis + main_pos_seq_vis + clone_pos_seq_vis
    _set_equal_aspect_3d(ax, all_points)
    _set_axis_ticks(ax, tick_step)

    ax.set_title(
        f"Offline KF Replay | tracker={result['meta']['tracker_id']} | "
        f"clone@update={result['meta']['clone_update_n']}"
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
        "--clone-update-n",
        type=int,
        default=None,
        help="Clone KF when main KF update count reaches N (default from YAML tracker.clone_update_n)",
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

    traj_dir = Path(args.trajectory_dir)
    config_path = Path(args.config)

    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    output_dir = Path(args.output_dir) if args.output_dir else (traj_dir / "offline_kf_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker_params = _load_tracker_params(config_path)
    clone_update_n = int(
        args.clone_update_n
        if args.clone_update_n is not None
        else tracker_params.get("clone_update_n", 8)
    )

    json_files = sorted(traj_dir.glob("trajectory_tracker*_*.json"))
    if not json_files:
        print(f"No trajectory JSON found in: {traj_dir}")
        return

    print(f"Found {len(json_files)} trajectories. Start offline replay...")
    print(f"clone_update_n = {clone_update_n}")
    print(f"output_dir = {output_dir}")
    all_items: list[dict[str, Any]] = []
    for jp in json_files:
        result = run_one_trajectory(jp, tracker_params, clone_update_n)

        out_json = output_dir / f"{jp.stem}_offline_kf.json"
        out_png = output_dir / f"{jp.stem}_offline_kf_3d.png"

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        plot_result(
            result,
            out_png,
            args.annotate_every,
            display_scale=args.display_scale,
            tick_step=args.tick_step,
        )

        all_items.append({
            "source_json": str(jp),
            "result": result,
        })

        print(f"完成: {jp.name}")
        print(f"  输出轨迹: {out_json.name}")
        print(f"  输出图像: {out_png.name}")

    if not args.no_interactive and all_items:
        print("\nOpen interactive 3D browser: A/D switch, Q toggle view mode, ESC quit.")
        browser = InteractiveTrajectoryBrowser(
            all_items,
            args.annotate_every,
            display_scale=args.display_scale,
            tick_step=args.tick_step,
        )
        browser.show()


if __name__ == "__main__":
    main()
