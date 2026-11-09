#!/usr/bin/env python3
"""Summarize the first 300 updates of adaptive five-task patch runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TASKS = ("pose_trans", "pose_rot", "depth", "point", "track")


def load_jsonl(path: Path, max_update: int) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["update"]) <= max_update:
                records.append(record)
    return records


def values(records: list[dict], key: str, lo: int, hi: int) -> np.ndarray:
    return np.asarray([
        float(record[key]) for record in records
        if lo <= int(record["update"]) <= hi and key in record
    ], dtype=np.float64)


def median(values_: np.ndarray) -> float:
    return float(np.median(values_)) if values_.size else float("nan")


def mean(values_: np.ndarray) -> float:
    return float(np.mean(values_)) if values_.size else float("nan")


def ratio(late: float, early: float) -> float:
    return late / early if np.isfinite(early) and abs(early) > 1e-15 else float("nan")


def summarize_run(root: Path, max_update: int) -> dict:
    patch = root / "geometry_patch"
    train = load_jsonl(patch / "training_history.jsonl", max_update)
    audit = load_jsonl(patch / "audit_history.jsonl", max_update)
    if not train or int(train[-1]["update"]) < max_update:
        raise RuntimeError(f"{root.name} has only {len(train)} updates through {max_update}")

    windows = {}
    for name, lo, hi in (("u001_100", 1, 100), ("u101_200", 101, 200),
                         ("u201_300", 201, 300)):
        windows[name] = {
            "attack_metric_median": median(values(train, "attack_metric", lo, hi)),
            "regularization_median": median(values(train, "regularization_total", lo, hi)),
            "min_gain_median": median(values(train, "projected_min_gain", lo, hi)),
            "direction_scale_median": median(
                values(train, "projected_direction_scale", lo, hi)
            ),
            "lr_median": median(values(train, "lr", lo, hi)),
        }
        for task in TASKS:
            windows[name][f"{task}_score_median"] = median(
                values(train, f"projected_{task}_score", lo, hi)
            )

    gradients = {}
    for task in TASKS:
        early_norm = median(values(train, f"projected_{task}_grad_norm", 1, 100))
        late_norm = median(values(train, f"projected_{task}_grad_norm", 201, 300))
        early_weight = mean(values(train, f"projected_{task}_weight", 1, 100))
        late_weight = mean(values(train, f"projected_{task}_weight", 201, 300))
        early_gain = median(values(train, f"projected_{task}_gain", 1, 100))
        late_gain = median(values(train, f"projected_{task}_gain", 201, 300))
        gradients[task] = {
            "grad_norm_early": early_norm,
            "grad_norm_late": late_norm,
            "grad_norm_ratio": ratio(late_norm, early_norm),
            "weight_early": early_weight,
            "weight_late": late_weight,
            "gain_early": early_gain,
            "gain_late": late_gain,
        }

    audits = [{
        "update": int(record["update"]),
        "score": float(record["score"]),
        "task_ratios": {key: float(value)
                        for key, value in record["task_ratios"].items()},
    } for record in audit]
    return {
        "run": root.name,
        "updates": len(train),
        "windows": windows,
        "gradients": gradients,
        "audits": audits,
    }


def f(value: float, digits: int = 4) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.{digits}g}"


def render_markdown(summaries: list[dict]) -> str:
    lines = [
        "# 自适应训练前 300 步诊断",
        "",
        "训练历史中的 `attack_metric` 是动态随机帧组上的 pose 汇总项，不能单独作为五任务收敛结论；固定审计分数才是调学习率的主依据。所有分数均为 larger-is-worse。",
        "",
        "## 固定审计",
        "",
        "| 序列 | update | 最弱 log 比率 | depth | point | pose rot | pose trans | track |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        scene = summary["run"].split("firsthalf_", 1)[-1].split("_joint", 1)[0]
        for audit in summary["audits"]:
            ratios = audit["task_ratios"]
            lines.append(
                f"| {scene} | {audit['update']} | {audit['score']:+.4f} | "
                f"{ratios['depth']:.3f}× | {ratios['point']:.3f}× | "
                f"{ratios['pose_rot']:.3f}× | {ratios['pose_trans']:.3f}× | "
                f"{ratios['track']:.3f}× |"
            )

    lines += [
        "",
        "## 随机训练帧组上的前后变化",
        "",
        "下表比较 1–100 与 201–300 步的中位数。倍率大于 1 表示后 100 步数值更大。",
        "",
        "| 序列 | pose trans | pose rot | depth | point | track | min gain | 正则项 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        scene = summary["run"].split("firsthalf_", 1)[-1].split("_joint", 1)[0]
        early = summary["windows"]["u001_100"]
        late = summary["windows"]["u201_300"]
        cells = []
        for task in TASKS:
            cells.append(ratio(late[f"{task}_score_median"],
                               early[f"{task}_score_median"]))
        min_gain_ratio = ratio(late["min_gain_median"], early["min_gain_median"])
        reg_ratio = ratio(late["regularization_median"],
                          early["regularization_median"])
        lines.append(
            f"| {scene} | " + " | ".join(f"{value:.2f}×" for value in cells) +
            f" | {min_gain_ratio:.2f}× | {reg_ratio:.2f}× |"
        )

    lines += [
        "",
        "## 梯度结构",
        "",
        "梯度范数倍率是 `median(201–300) / median(1–100)`；权重是 max-min 求解器分配给该任务的平均权重。",
        "",
        "| 序列 | 任务 | 梯度范数倍率 | 早期权重 | 后期权重 | 早期范数 | 后期范数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        scene = summary["run"].split("firsthalf_", 1)[-1].split("_joint", 1)[0]
        for task in TASKS:
            grad = summary["gradients"][task]
            lines.append(
                f"| {scene} | {task} | {grad['grad_norm_ratio']:.2f}× | "
                f"{grad['weight_early']:.3f} | {grad['weight_late']:.3f} | "
                f"{f(grad['grad_norm_early'])} | {f(grad['grad_norm_late'])} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--max-update", type=int, default=300)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    summaries = [summarize_run(root, args.max_update) for root in args.roots]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render_markdown(summaries), encoding="utf-8")
    print(args.markdown_out)


if __name__ == "__main__":
    main()
