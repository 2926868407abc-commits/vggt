#!/usr/bin/env python3
"""Aggregate the 16 per-scene split runs into CSV, JSON, and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SCENES = (
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_sitting_rpy",
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
    "rgbd_dataset_freiburg3_walking_rpy",
    "rgbd_dataset_freiburg3_walking_halfsphere",
)
SPLITS = ("firsthalf", "evenodd")
ERROR_METRICS = (
    "ATE", "RPE_trans", "RPE_rot_deg", "depth_absrel",
    "point_acc", "point_comp", "track_epe_mean",
)
SUCCESS_METRICS = ("depth_d1", "track_vis_f1")


def load_rows(root: Path, steps: int) -> list[dict]:
    rows = []
    for split in SPLITS:
        for scene in SCENES:
            short = scene.removeprefix("rgbd_dataset_freiburg3_")
            run = f"perseq_{split}_{short}_joint_cold_s0_u{steps}"
            run_root = root / run
            path = run_root / "heldout_w30_s20/four_tasks/four_task_by_scene.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open(newline="", encoding="utf-8") as handle:
                found = list(csv.DictReader(handle))
            if len(found) != 1 or found[0]["scene"] != scene:
                raise ValueError(f"Unexpected result rows in {path}")
            row = dict(found[0])
            row["split"] = split
            row["run"] = run
            row["initial_patch"] = str(run_root / "geometry_patch/initial_patch.png")
            row["optimized_patch"] = str(run_root / "geometry_patch/optimized_patch.png")
            rows.append(row)
    return rows


def macro(rows: list[dict], split: str) -> dict:
    group = [row for row in rows if row["split"] == split]
    result = {"split": split, "n_scenes": len(group)}
    for metric in (*ERROR_METRICS, *SUCCESS_METRICS):
        clean = float(np.mean([float(row[f"clean_{metric}"]) for row in group]))
        attack = float(np.mean([float(row[f"attack_{metric}"]) for row in group]))
        result[f"clean_{metric}"] = clean
        result[f"attack_{metric}"] = attack
        if metric in ERROR_METRICS:
            damage = 100.0 * (attack / max(clean, 1e-12) - 1.0)
        else:
            damage = 100.0 * (1.0 - attack / max(clean, 1e-12))
        result[f"damage_{metric}_pct"] = damage
    result["total_unique_frames"] = int(
        sum(int(row["n_unique_frames"]) for row in group))
    return result


def pair(row: dict, metric: str, digits: int = 4) -> str:
    clean = float(row[f"clean_{metric}"])
    attack = float(row[f"attack_{metric}"])
    return f"{clean:.{digits}f} → {attack:.{digits}f}"


def write_markdown(path: Path, rows: list[dict], macros: list[dict], steps: int) -> None:
    lines = [
        "# 逐序列两种帧划分实验",
        "",
        "- 每条 TUM 序列独立训练一张 patch；",
        f"- 两种划分均为 {steps} 次更新、seed 0、64×64 warning 冷启动、同一显示器世界四边形、no-EOT；",
        "- `firsthalf`：前半段训练，只在后半段测试；",
        "- `evenodd`：偶数帧训练，只在奇数帧测试；",
        "- 每次训练随机抽 10 张训练帧；测试用 30 帧窗口、步长 20；完整留出轨迹只做一次最终 Sim(3)；",
        "- ATE/RPE/AbsRel/Point Acc/Comp/EPE 越大越差，δ1/visF1 越小越差。",
        "",
    ]
    for split in SPLITS:
        title = "前半段训练 → 后半段测试" if split == "firsthalf" else "偶数帧训练 → 奇数帧测试"
        lines.extend([
            f"## {title}",
            "",
            "| 序列 | ATE | RPE-r | AbsRel | Point Acc | Track EPE | visF1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in (item for item in rows if item["split"] == split):
            short = row["scene"].removeprefix("rgbd_dataset_freiburg3_")
            lines.append(
                f"| {short} | {pair(row, 'ATE', 6)} | {pair(row, 'RPE_rot_deg', 4)} | "
                f"{pair(row, 'depth_absrel', 4)} | {pair(row, 'point_acc', 4)} | "
                f"{pair(row, 'track_epe_mean', 3)} | {pair(row, 'track_vis_f1', 4)} |"
            )
        m = next(item for item in macros if item["split"] == split)
        lines.extend([
            "",
            "宏平均攻击方向变化（正数表示变差）：",
            "",
            f"- ATE {m['damage_ATE_pct']:+.2f}%；RPE-t {m['damage_RPE_trans_pct']:+.2f}%；RPE-r {m['damage_RPE_rot_deg_pct']:+.2f}%；",
            f"- AbsRel {m['damage_depth_absrel_pct']:+.2f}%；δ1 {m['damage_depth_d1_pct']:+.2f}%；",
            f"- Point Acc {m['damage_point_acc_pct']:+.2f}%；Point Comp {m['damage_point_comp_pct']:+.2f}%；",
            f"- Track EPE {m['damage_track_epe_mean_pct']:+.2f}%；visF1 {m['damage_track_vis_f1_pct']:+.2f}%。",
            "",
        ])
    lines.extend([
        "## 结果文件",
        "",
        f"- 合并 CSV：`{path.parent / 'per_scene_split_results.csv'}`",
        f"- 宏平均 JSON：`{path.parent / 'per_scene_split_macro.json'}`",
        "- 每个 run 的初始与优化后纹理绝对路径记录在合并 CSV 的 `initial_patch` / `optimized_patch` 列。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_root", default="/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10")
    parser.add_argument(
        "--output_dir", default="/mnt/data/wangqq/vggt/reports/per_scene_split_comparison")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.run_root), args.steps)
    csv_path = output / "per_scene_split_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    macros = [macro(rows, split) for split in SPLITS]
    (output / "per_scene_split_macro.json").write_text(
        json.dumps(macros, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output / "per_scene_split_report.md", rows, macros, args.steps)
    print(json.dumps(macros, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
