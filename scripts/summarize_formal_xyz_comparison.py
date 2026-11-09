#!/usr/bin/env python3
"""Validate convergence and summarize the formal sitting_xyz comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCENE = "rgbd_dataset_freiburg3_sitting_xyz"


def run_names(max_steps: int) -> dict[str, str]:
    return {
        "MIRAGE pose warm-up": (
            f"formal_xyz_v3_mirage_pose_warm_adaptive_s0_umax{max_steps}"
        ),
        "MIRAGE-3D": (
            f"formal_xyz_v3_mirage_joint_warm_adaptive_s0_umax{max_steps}"
        ),
        "SPOO stage 1": (
            f"formal_xyz_v3_spoo_stage1_adaptive_s17_umax{max_steps}"
        ),
        "SPOO": f"formal_xyz_v3_spoo_stage2_adaptive_s17_umax{max_steps}",
    }


def load_meta(output_root: Path, run: str) -> dict:
    path = output_root / run / "geometry_patch" / "geometry_patch_meta.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def convergence_rows(output_root: Path, max_steps: int) -> list[dict]:
    rows = []
    for label, run in run_names(max_steps).items():
        meta = load_meta(output_root, run)
        state = (meta.get("adaptive_scheduler") or {}).get("state") or {}
        rows.append({
            "label": label,
            "run": run,
            "updates": int(meta.get("completed_updates", -1)),
            "stopped": bool(state.get("stopped", False)),
            "best_update": int(state.get("best_update", -1)),
            "best_score": float(state.get("best_score", float("nan"))),
            "final_lr": float(state.get("lr", float("nan"))),
        })
    return rows


def require_converged(rows: list[dict]) -> None:
    failures = [row for row in rows if not row["stopped"]]
    for row in rows:
        print(
            f"{row['label']}: updates={row['updates']} stopped={row['stopped']} "
            f"best_update={row['best_update']} best_score={row['best_score']:+.6f} "
            f"final_lr={row['final_lr']:.8g}"
        )
    if failures:
        names = ", ".join(row["label"] for row in failures)
        raise SystemExit(f"not converged; refusing formal evaluation: {names}")


def load_metrics(output_root: Path, run: str) -> dict[str, str]:
    path = (
        output_root / run / "heldout_w30_s20_v5_controlled" / "four_tasks"
        / "four_task_by_scene.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["scene"] == SCENE]
    if len(selected) != 1:
        raise RuntimeError(f"expected one {SCENE} row in {path}, got {len(selected)}")
    return selected[0]


def pct_worse(clean: float, attack: float, *, lower_is_worse: bool = False) -> float:
    if abs(clean) < 1e-12:
        return float("nan")
    sign = -1.0 if lower_is_worse else 1.0
    return sign * (attack - clean) / clean * 100.0


def f(value: float) -> str:
    return f"{value:.6f}"


def write_report(root: Path, max_steps: int, convergence: list[dict]) -> Path:
    output_root = root / "outputs_attack_geometry_aware_tum10"
    names = run_names(max_steps)
    methods = {
        "MIRAGE-3D": load_metrics(output_root, names["MIRAGE-3D"]),
        "SPOO": load_metrics(output_root, names["SPOO"]),
    }
    metrics = [
        ("ATE (m)", "clean_ATE", "attack_ATE", False),
        ("RPE trans (m)", "clean_RPE_trans", "attack_RPE_trans", False),
        ("RPE rot (deg)", "clean_RPE_rot_deg", "attack_RPE_rot_deg", False),
        ("Depth AbsRel", "clean_depth_absrel", "attack_depth_absrel", False),
        ("Point Acc", "clean_point_acc", "attack_point_acc", False),
        ("Point Comp", "clean_point_comp", "attack_point_comp", False),
        ("Track EPE (px)", "clean_track_epe_mean", "attack_track_epe_mean", False),
        ("Track vis-F1", "clean_track_vis_f1", "attack_track_vis_f1", True),
    ]
    lines = [
        "# sitting_xyz：MIRAGE-3D 与 SPOO 的收敛对比",
        "",
        "- 训练集：该序列前半段 608 帧；每步随机抽 10 帧。",
        "- 测试集：后半段 608 帧，完全不参与训练和早停。",
        "- 位姿协议：30 帧窗口、步长 20、相邻窗口重叠 10 帧；先拼成一条完整轨迹，再做一次全局 Sim(3) 对齐。",
        "- 载体：两种方法使用同一个 v5 显示器三维四边形和相同物理面积。",
        "- 停止规则：固定训练审计集停滞后降低学习率；到达最低学习率后继续停滞才 early-stop，并恢复最佳审计纹理。",
        "- 旧的 10 帧窗口／步长 5 协议已删除，不参与本表。",
        "",
        "## 收敛证据",
        "",
        "| 阶段 | 实际更新数 | 最佳更新 | 最佳审计分数 | 最终学习率 | early-stop |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in convergence:
        lines.append(
            f"| {row['label']} | {row['updates']} | {row['best_update']} | "
            f"{row['best_score']:+.6f} | {row['final_lr']:.8g} | "
            f"{'是' if row['stopped'] else '否'} |"
        )
    lines += [
        "",
        "## 后半段完整四任务结果",
        "",
        "表中“恶化”为相对各方法同一路径 Clean 的变化；正数表示攻击方向变差。",
        "",
        "| 指标 | MIRAGE Clean | MIRAGE Attack | MIRAGE 恶化 | SPOO Clean | SPOO Attack | SPOO 恶化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, clean_key, attack_key, lower_is_worse in metrics:
        cells = []
        for method in ("MIRAGE-3D", "SPOO"):
            row = methods[method]
            clean = float(row[clean_key])
            attack = float(row[attack_key])
            delta = pct_worse(clean, attack, lower_is_worse=lower_is_worse)
            cells.extend([f(clean), f(attack), f"{delta:+.1f}%"])
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "## 绝对路径", ""]
    for label in ("MIRAGE-3D", "SPOO"):
        run = names[label]
        patch = output_root / run / "geometry_patch"
        result = output_root / run / "heldout_w30_s20_v5_controlled"
        lines += [
            f"- {label} 初始纹理：`{patch / 'initial_patch.png'}`",
            f"- {label} 优化纹理：`{patch / 'optimized_patch.png'}`",
            f"- {label} 30/20 结果：`{result}`",
        ]

    report_dir = root / "reports" / "formal_xyz_comparison"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "mirage_vs_spoo_w30_s20.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "report"))
    parser.add_argument("--root", type=Path, default=Path("/mnt/data/wangqq/vggt"))
    parser.add_argument("--max_steps", type=int, default=6000)
    args = parser.parse_args()
    output_root = args.root / "outputs_attack_geometry_aware_tum10"
    rows = convergence_rows(output_root, args.max_steps)
    require_converged(rows)
    if args.mode == "report":
        write_report(args.root, args.max_steps, rows)


if __name__ == "__main__":
    main()
