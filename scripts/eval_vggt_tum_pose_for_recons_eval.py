#!/usr/bin/env python3
"""Evaluate VGGT official batch camera poses on recons_eval TUM-dynamics.

This is a bridge for the workflow:

    VGGT batch_vggt_inference.py -> vggt_outputs.npz -> recons_eval pose metrics

It reads VGGT's saved extrinsic matrices (world-to-camera), converts them to
camera-to-world trajectories, then calls recons_eval's evo-based metrics.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def default_recons_eval_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "recons_eval" and (parent / "datasets").exists():
            return parent
    for parent in here.parents:
        candidate = parent / "recons_eval"
        if (candidate / "datasets").exists():
            return candidate
    return here.parents[1]


def add_recons_eval_to_path(root: Path) -> None:
    sys.path.insert(0, str(root.resolve()))


def extrinsic_w2c_to_c2w(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsic shape (N, 3, 4), got {extrinsic.shape}")
    w2c = np.tile(np.eye(4, dtype=np.float64), (extrinsic.shape[0], 1, 1))
    w2c[:, :3, :4] = extrinsic.astype(np.float64)
    return np.linalg.inv(w2c)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VGGT official TUM pose outputs with recons_eval metrics.")
    parser.add_argument("--vggt_output_root", required=True, help="VGGT output root containing seq/vggt_outputs.npz")
    parser.add_argument("--recons_eval_root", default=str(default_recons_eval_root()))
    parser.add_argument("--data_root", help="TUM data root; default: <recons_eval_root>/data/tum")
    parser.add_argument("--model_name", default="vggt_tum_official_l3")
    parser.add_argument("--scene_pattern", default="rgbd_dataset_freiburg3_*")
    parser.add_argument("--pose_eval_stride", type=int, default=1)
    parser.add_argument("--output_root", help="Output root for per-sequence artifacts")
    parser.add_argument("--metric_csv", help="Output metric CSV path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite seq_metrics.csv if present")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recons_eval_root = Path(args.recons_eval_root)
    add_recons_eval_to_path(recons_eval_root)

    from relpose.evo_utils import calculate_averages, eval_metrics, get_tum_poses, load_traj, save_tum_poses

    vggt_root = Path(args.vggt_output_root)
    data_root = Path(args.data_root) if args.data_root else recons_eval_root / "data" / "tum"
    output_root = (
        Path(args.output_root)
        if args.output_root
        else recons_eval_root / "outputs" / "relpose-distance" / args.model_name / "tum"
    )
    metric_csv = (
        Path(args.metric_csv)
        if args.metric_csv
        else recons_eval_root / "outputs" / "relpose-distance" / f"tum-metric-{args.model_name}.csv"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    metric_csv.parent.mkdir(parents=True, exist_ok=True)

    seq_metrics_csv = output_root / "seq_metrics.csv"
    if args.overwrite and seq_metrics_csv.exists():
        seq_metrics_csv.unlink()

    results = []
    scene_dirs = sorted([p for p in vggt_root.glob(args.scene_pattern) if p.is_dir()])
    if not scene_dirs:
        raise SystemExit(f"No sequence folders matched {vggt_root / args.scene_pattern}")

    for seq_dir in scene_dirs:
        seq = seq_dir.name
        npz_path = seq_dir / "vggt_outputs.npz"
        gt_path = data_root / seq / "groundtruth_90.txt"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)

        with np.load(npz_path) as data:
            if "extrinsic" not in data:
                raise KeyError(f"`extrinsic` not found in {npz_path}")
            c2w = extrinsic_w2c_to_c2w(data["extrinsic"])

        seq_out = output_root / seq
        seq_out.mkdir(parents=True, exist_ok=True)
        np.save(seq_out / "pred_poses.npy", c2w[:, :3, :4].astype(np.float32))

        pred_traj = get_tum_poses(c2w)
        save_tum_poses(pred_traj, seq_out / "pred_traj.txt")
        gt_traj = load_traj(str(gt_path), traj_format="tum", stride=args.pose_eval_stride)
        ate, rpe_trans, rpe_rot = eval_metrics(
            pred_traj,
            gt_traj,
            seq=seq,
            filename=str(seq_out / "eval_metric.txt"),
            verbose=False,
        )
        results.append((seq, ate, rpe_trans, rpe_rot))

        row = {
            "model": args.model_name,
            "dataset": "tum",
            "seq": seq,
            "ATE": ate,
            "RPE trans": rpe_trans,
            "RPE rot": rpe_rot,
        }
        with seq_metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(f"{seq}: ATE={ate:.6f}, RPE trans={rpe_trans:.6f}, RPE rot={rpe_rot:.6f}")

    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)
    metrics = {
        "model": args.model_name,
        "ATE": avg_ate,
        "RPE trans": avg_rpe_trans,
        "RPE rot": avg_rpe_rot,
    }
    with metric_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"\nSaved metric CSV -> {metric_csv}")
    print(metrics)


if __name__ == "__main__":
    main()
