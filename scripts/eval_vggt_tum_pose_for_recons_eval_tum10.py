"""Evaluate VGGT TUM-10 pose outputs without modifying recons_eval files.

The attacked VGGT output contains only the selected 10 frames. This helper reads
each scene's attack_summary.json, extracts the corresponding rows from
groundtruth_90.txt, rewrites their timestamps to match the prediction order, and
then calls recons_eval's evo utilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def extrinsic_w2c_to_c2w(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsic shape (N,3,4), got {extrinsic.shape}")
    w2c = np.tile(np.eye(4, dtype=np.float64), (extrinsic.shape[0], 1, 1))
    w2c[:, :3, :4] = extrinsic.astype(np.float64)
    return np.linalg.inv(w2c)


def read_tum_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(stripped.split())
    return rows


def write_selected_gt(gt_rows: list[list[str]], frame_indices: list[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for order, frame_idx in enumerate(frame_indices):
            if frame_idx >= len(gt_rows):
                raise IndexError(f"frame index {frame_idx} is out of range for {out_path}")
            row = list(gt_rows[frame_idx])
            row[0] = f"{float(order):.6f}"
            f.write(" ".join(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vggt_output_root", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--recons_root", default="/mnt/data/wangqq/recons_eval")
    parser.add_argument("--tum_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--metric_csv", default=None)
    parser.add_argument("--gt_name", default="groundtruth_90.txt")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recons_root = Path(args.recons_root)
    sys.path.insert(0, str(recons_root.resolve()))

    from relpose.evo_utils import calculate_averages, eval_metrics, get_tum_poses, load_traj, save_tum_poses

    vggt_root = Path(args.vggt_output_root)
    tum_root = Path(args.tum_root) if args.tum_root else recons_root / "data/tum"
    output_root = (
        Path(args.output_root)
        if args.output_root
        else recons_root / "outputs/relpose-distance" / args.model_name / "tum10"
    )
    metric_csv = (
        Path(args.metric_csv)
        if args.metric_csv
        else recons_root / "outputs/relpose-distance" / f"tum10-metric-{args.model_name}.csv"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    metric_csv.parent.mkdir(parents=True, exist_ok=True)

    seq_metrics_csv = output_root / "seq_metrics.csv"
    if args.overwrite and seq_metrics_csv.exists():
        seq_metrics_csv.unlink()

    results = []
    scene_dirs = sorted(path for path in vggt_root.glob("rgbd_dataset_freiburg3_*") if path.is_dir())
    if not scene_dirs:
        raise SystemExit(f"No TUM outputs found under {vggt_root}")

    for seq_dir in scene_dirs:
        seq = seq_dir.name
        npz_path = seq_dir / "vggt_outputs.npz"
        summary_path = seq_dir / "attack_summary.json"
        gt_path = tum_root / seq / args.gt_name
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        frame_indices = [int(idx) for idx in summary["frame_indices"]]

        with np.load(npz_path) as data:
            c2w = extrinsic_w2c_to_c2w(data["extrinsic"])
        if len(frame_indices) != c2w.shape[0]:
            raise ValueError(f"{seq}: {len(frame_indices)} frame indices but {c2w.shape[0]} predicted poses")

        seq_out = output_root / seq
        seq_out.mkdir(parents=True, exist_ok=True)
        np.save(seq_out / "pred_poses.npy", c2w[:, :3, :4].astype(np.float32))
        pred_traj = get_tum_poses(c2w)
        save_tum_poses(pred_traj, seq_out / "pred_traj.txt")

        selected_gt_path = seq_out / "groundtruth_selected.txt"
        write_selected_gt(read_tum_rows(gt_path), frame_indices, selected_gt_path)
        gt_traj = load_traj(str(selected_gt_path), traj_format="tum", stride=1)

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
            "dataset": "tum10",
            "seq": seq,
            "n_frames": len(frame_indices),
            "ATE": ate,
            "RPE trans": rpe_trans,
            "RPE rot": rpe_rot,
        }
        with seq_metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(f"{args.model_name} {seq}: ATE={ate:.6f}, RPE trans={rpe_trans:.6f}, RPE rot={rpe_rot:.6f}")

    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)
    with metric_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "ATE", "RPE trans", "RPE rot"])
        writer.writeheader()
        writer.writerow(
            {
                "model": args.model_name,
                "ATE": avg_ate,
                "RPE trans": avg_rpe_trans,
                "RPE rot": avg_rpe_rot,
            }
        )
    print(f"Saved metric CSV -> {metric_csv}")


if __name__ == "__main__":
    main()
