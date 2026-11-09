#!/usr/bin/env python3
"""Plot convergence traces for the formal sitting_xyz MIRAGE-3D/SPOO pair.

The formal MIRAGE run already logs its projected update-direction norm.  The
formal SPOO run predates common gradient logging, so its non-invasive replay is
used for both SPOO loss and gradient curves after checking that its loss trace
matches the formal run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def values(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def ema(array: np.ndarray, span: int = 50) -> np.ndarray:
    if array.size == 0:
        return array
    alpha = 2.0 / (span + 1.0)
    output = np.empty_like(array)
    output[0] = array[0]
    for index in range(1, array.size):
        output[index] = alpha * array[index] + (1.0 - alpha) * output[index - 1]
    return output


def concat_stage(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steps = np.arange(1, first.size + second.size + 1, dtype=np.int64)
    return steps, np.concatenate([first, second])


def style_axis(axis, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#d7dce2", linewidth=0.7, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def draw_trace(axis, steps, raw, color, label, log_y=False) -> None:
    axis.plot(steps, raw, color=color, alpha=0.13, linewidth=0.65)
    axis.plot(steps, ema(raw), color=color, linewidth=1.8, label=label)
    if log_y:
        axis.set_yscale("log")


def phase_line(axis, boundary: int, left: str, right: str) -> None:
    axis.axvline(boundary, color="#626b75", linestyle="--", linewidth=1.0)
    axis.text(
        0.01, 0.96, left, transform=axis.transAxes, ha="left", va="top",
        fontsize=9, color="#39424c",
    )
    axis.text(
        0.99, 0.96, right, transform=axis.transAxes, ha="right", va="top",
        fontsize=9, color="#39424c",
    )


def compare_replay(formal: np.ndarray, replay: np.ndarray) -> dict:
    length = min(formal.size, replay.size)
    if length == 0:
        raise RuntimeError("empty formal/replay SPOO trace")
    delta = replay[:length] - formal[:length]
    correlation = float(np.corrcoef(formal[:length], replay[:length])[0, 1])
    return {
        "formal_updates": int(formal.size),
        "replay_updates": int(replay.size),
        "compared_updates": int(length),
        "mean_absolute_difference": float(np.mean(np.abs(delta))),
        "max_absolute_difference": float(np.max(np.abs(delta))),
        "correlation": correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt-root", type=Path, default=Path("/mnt/data/wangqq/vggt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=6000)
    args = parser.parse_args()

    root = args.vggt_root
    runs = root / "outputs_attack_geometry_aware_tum10"
    formal = {
        "mirage_pose": runs / f"formal_xyz_v3_mirage_pose_warm_adaptive_s0_umax{args.max_steps}",
        "mirage_joint": runs / f"formal_xyz_v3_mirage_joint_warm_adaptive_s0_umax{args.max_steps}",
        "spoo_stage1": runs / f"formal_xyz_v3_spoo_stage1_adaptive_s17_umax{args.max_steps}",
        "spoo_stage2": runs / f"formal_xyz_v3_spoo_stage2_adaptive_s17_umax{args.max_steps}",
    }
    replay = {
        "spoo_stage1": runs / f"curvegrad_xyz_v3_spoo_stage1_adaptive_s17_umax{args.max_steps}",
        "spoo_stage2": runs / f"curvegrad_xyz_v3_spoo_stage2_adaptive_s17_umax{args.max_steps}",
    }

    def history(run: Path) -> list[dict]:
        path = run / "geometry_patch" / "training_history.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        return read_jsonl(path)

    mirage_pose = history(formal["mirage_pose"])
    mirage_joint = history(formal["mirage_joint"])
    spoo_formal_1 = history(formal["spoo_stage1"])
    spoo_formal_2 = history(formal["spoo_stage2"])
    spoo_replay_1 = history(replay["spoo_stage1"])
    spoo_replay_2 = history(replay["spoo_stage2"])

    for rows, name in ((spoo_replay_1, "SPOO stage 1"), (spoo_replay_2, "SPOO stage 2")):
        missing = [index for index, row in enumerate(rows) if "texture_grad_norm" not in row]
        if missing:
            raise RuntimeError(f"{name} has {len(missing)} rows without texture_grad_norm")

    validation = {
        "stage1": compare_replay(
            values(spoo_formal_1, "attack_metric"), values(spoo_replay_1, "attack_metric")
        ),
        "stage2": compare_replay(
            values(spoo_formal_2, "attack_metric"), values(spoo_replay_2, "attack_metric")
        ),
    }

    m_steps, m_score = concat_stage(
        values(mirage_pose, "attack_metric"), values(mirage_joint, "attack_metric")
    )
    _, m_grad = concat_stage(
        values(mirage_pose, "projected_direction_scale"),
        values(mirage_joint, "projected_direction_scale"),
    )
    s_loss_steps, s_loss = concat_stage(
        values(spoo_formal_1, "attack_metric"), values(spoo_formal_2, "attack_metric")
    )
    s_grad_steps, s_grad = concat_stage(
        values(spoo_replay_1, "texture_grad_norm"),
        values(spoo_replay_2, "texture_grad_norm"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), constrained_layout=True)
    figure.suptitle("Training dynamics on TUM sitting_xyz", fontsize=14, fontweight="semibold")

    draw_trace(axes[0, 0], m_steps, m_score, "#2166ac", "EMA (50 updates)")
    style_axis(axes[0, 0], "Optimizer update", "Logged pose damage score (higher is stronger)")
    axes[0, 0].set_title("MIRAGE-3D: training objective")
    phase_line(axes[0, 0], len(mirage_pose), "Pose warm-up", "Five-task stage")

    draw_trace(axes[1, 0], m_steps, np.maximum(m_grad, 1e-30), "#2166ac", "EMA (50 updates)", True)
    style_axis(axes[1, 0], "Optimizer update", "Projected attack-direction norm (L2)")
    axes[1, 0].set_title("MIRAGE-3D: gradient norm")
    phase_line(axes[1, 0], len(mirage_pose), "Pose warm-up", "Five-task stage")

    draw_trace(axes[0, 1], s_loss_steps, s_loss, "#b2182b", "Formal run EMA (50 updates)")
    style_axis(axes[0, 1], "L-BFGS update", "Depth training loss (lower is better)")
    axes[0, 1].set_title("SPOO: training loss")
    phase_line(axes[0, 1], len(spoo_replay_1), "Texture stage", "Region/style stage")

    draw_trace(axes[1, 1], s_grad_steps, np.maximum(s_grad, 1e-30), "#b2182b", "EMA (50 updates)", True)
    style_axis(axes[1, 1], "L-BFGS update", "Texture gradient norm (L2)")
    axes[1, 1].set_title("SPOO: gradient norm")
    phase_line(axes[1, 1], len(spoo_replay_1), "Texture stage", "Region/style stage")

    for axis in axes.flat:
        axis.legend(loc="lower right", frameon=False, fontsize=8)

    png = args.output_dir / "formal_spoo_vs_mirage_training_loss_gradnorm.png"
    pdf = args.output_dir / "formal_spoo_vs_mirage_training_loss_gradnorm.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    task_names = ("pose_trans", "pose_rot", "depth", "point", "track")
    task_colors = {
        "pose_trans": "#2166ac", "pose_rot": "#67a9cf", "depth": "#1b7837",
        "point": "#762a83", "track": "#e08214",
    }
    joint_steps = np.arange(1, len(mirage_joint) + 1)
    task_figure, task_axes = plt.subplots(2, 1, figsize=(9.0, 7.2), constrained_layout=True)
    task_figure.suptitle("MIRAGE-3D five-task optimization dynamics", fontsize=14, fontweight="semibold")
    for task in task_names:
        score = values(mirage_joint, f"projected_{task}_score")
        baseline = max(float(np.median(score[: min(50, score.size)])), 1e-30)
        task_axes[0].plot(joint_steps, ema(score / baseline), color=task_colors[task], linewidth=1.5, label=task)
        gradient = np.maximum(values(mirage_joint, f"projected_{task}_grad_norm"), 1e-30)
        task_axes[1].plot(joint_steps, ema(gradient), color=task_colors[task], linewidth=1.5, label=task)
    style_axis(task_axes[0], "Five-task optimizer update", "Task score / initial 50-update median")
    task_axes[0].set_title("Normalized task damage scores (higher is stronger)")
    style_axis(task_axes[1], "Five-task optimizer update", "Per-task texture gradient norm (L2)")
    task_axes[1].set_yscale("log")
    task_axes[1].set_title("Raw per-task gradient norms")
    for axis in task_axes:
        axis.legend(ncol=5, loc="best", frameon=False, fontsize=8)
    task_figure.savefig(args.output_dir / "mirage_five_task_scores_gradnorm.png", dpi=220, bbox_inches="tight")
    task_figure.savefig(args.output_dir / "mirage_five_task_scores_gradnorm.pdf", bbox_inches="tight")
    plt.close(task_figure)

    with (args.output_dir / "formal_spoo_vs_mirage_training_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "source", "stage", "update", "quantity", "value"])
        for index, (score, grad) in enumerate(zip(m_score, m_grad), start=1):
            stage = "pose_warmup" if index <= len(mirage_pose) else "five_task"
            writer.writerow(["MIRAGE-3D", "formal", stage, index, "logged_pose_damage_score", score])
            writer.writerow(["MIRAGE-3D", "formal", stage, index, "projected_direction_grad_norm", grad])
        for index, loss in enumerate(s_loss, start=1):
            stage = "texture" if index <= len(spoo_formal_1) else "region_style"
            writer.writerow(["SPOO", "formal", stage, index, "depth_training_loss", loss])
        for index, grad in enumerate(s_grad, start=1):
            stage = "texture" if index <= len(spoo_replay_1) else "region_style"
            writer.writerow(["SPOO", "gradient_replay", stage, index, "texture_grad_norm", grad])

    summary = {
        "mirage": {
            "pose_warmup_updates": len(mirage_pose),
            "five_task_updates": len(mirage_joint),
            "gradient_field": "projected_direction_scale",
            "gradient_definition": "L2 norm of the projected attack direction before regularizer addition",
        },
        "spoo": {
            "texture_stage_updates": len(spoo_replay_1),
            "region_style_stage_updates": len(spoo_replay_2),
            "gradient_field": "texture_grad_norm",
            "gradient_definition": "L2 norm of the actual accumulated texture gradient immediately before L-BFGS.step",
            "loss_source": "original formal run",
            "gradient_source": "same-configuration replay with identical sampled subsets",
            "formal_replay_validation": validation,
        },
        "smoothing": "EMA with span 50; faint lines are raw per-update observations",
    }
    (args.output_dir / "training_curve_provenance.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    # Compact, smoothed data for the in-conversation interactive rendering.
    # The publication PNG/PDF retain the full-resolution raw observations.
    stride_m = max(1, int(np.ceil(m_steps.size / 220)))
    common_s = min(s_loss_steps.size, s_grad_steps.size)
    stride_s = max(1, int(np.ceil(common_s / 220)))
    sampled = {
        "mirage": [
            {"step": int(step), "objective": float(score), "grad": float(grad)}
            for step, score, grad in zip(
                m_steps[::stride_m], ema(m_score)[::stride_m], ema(m_grad)[::stride_m]
            )
        ],
        "spoo": [
            {"step": int(step), "objective": float(loss), "grad": float(grad)}
            for step, loss, grad in zip(
                s_loss_steps[:common_s:stride_s],
                ema(s_loss[:common_s])[::stride_s],
                ema(s_grad[:common_s])[::stride_s],
            )
        ],
        "boundaries": {"mirage": len(mirage_pose), "spoo": len(spoo_replay_1)},
    }
    (args.output_dir / "training_curve_sampled.json").write_text(
        json.dumps(sampled, separators=(",", ":")), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(png)


if __name__ == "__main__":
    main()
