#!/usr/bin/env python3
"""Evaluate VGGT official batch point maps on recons_eval Neural-RGBD.

This bridges:

    VGGT batch_vggt_inference.py -> vggt_outputs.npz -> recons_eval mv_recon metrics

The metric implementation, Neural-RGBD GT point construction, Sim(3) alignment,
ICP refinement, and accuracy/completion calculations are reused from recons_eval.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


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


def default_seq_map(root: Path, dataset_name: str) -> Path:
    if dataset_name == "NRGBD-sparse":
        filename = "NRGBD_mv-recon_seq-id-map-kf500.json"
    elif dataset_name == "NRGBD-dense":
        filename = "NRGBD_mv-recon_seq-id-map-kf100.json"
    else:
        raise ValueError("dataset_name must be NRGBD-sparse or NRGBD-dense unless --seq_id_map is set")
    return root / "datasets" / "seq-id-maps" / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VGGT official Neural-RGBD point maps with recons_eval.")
    parser.add_argument("--vggt_output_root", required=True, help="VGGT output root containing seq/vggt_outputs.npz")
    parser.add_argument("--recons_eval_root", default=str(default_recons_eval_root()))
    parser.add_argument("--dataset_name", default="NRGBD-sparse", choices=["NRGBD-sparse", "NRGBD-dense"])
    parser.add_argument("--model_name", default="vggt_nrgbd_sparse_official_l3")
    parser.add_argument("--nrgbd_dir", help="Neural-RGBD root; default: <recons_eval_root>/data/nrgbd")
    parser.add_argument("--seq_id_map", help="Path to NRGBD seq-id-map JSON")
    parser.add_argument("--cache_file", help="Dataset cache file")
    parser.add_argument("--output_root", help="Output root for artifacts")
    parser.add_argument("--metric_csv", help="Output metric CSV path")
    parser.add_argument("--pred_key", default="point_cloud_unproj", choices=["point_cloud_unproj", "point_map"])
    parser.add_argument("--load_img_size", type=int, default=518)
    parser.add_argument("--icp_threshold", type=float, default=0.1)
    parser.add_argument("--no_save_ply", action="store_true", help="Do not write pred/gt PLY files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite _all_samples.csv if present")
    return parser.parse_args()


def write_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def load_pred_points(npz_path: Path, pred_key: str) -> np.ndarray:
    with np.load(npz_path) as data:
        if pred_key not in data:
            raise KeyError(f"`{pred_key}` not found in {npz_path}")
        pred_pts = np.asarray(data[pred_key])
    if pred_pts.ndim != 4 or pred_pts.shape[-1] != 3:
        raise ValueError(f"Expected {pred_key} shape (N,H,W,3), got {pred_pts.shape} from {npz_path}")
    return pred_pts.astype(np.float32)


def main() -> None:
    args = parse_args()
    recons_eval_root = Path(args.recons_eval_root)
    add_recons_eval_to_path(recons_eval_root)

    from datasets.nrgbd import NRGBD
    from mv_recon.utils import accuracy, completion, umeyama

    vggt_root = Path(args.vggt_output_root)
    nrgbd_dir = Path(args.nrgbd_dir) if args.nrgbd_dir else recons_eval_root / "data" / "nrgbd"
    seq_id_map_path = Path(args.seq_id_map) if args.seq_id_map else default_seq_map(recons_eval_root, args.dataset_name)
    cache_file = (
        Path(args.cache_file)
        if args.cache_file
        else recons_eval_root / "data" / "dataset_cache" / "nrgbd_mv_recon_cache.npy"
    )
    output_root = (
        Path(args.output_root)
        if args.output_root
        else recons_eval_root / "outputs" / "mv_recon" / args.model_name / args.dataset_name
    )
    metric_csv = (
        Path(args.metric_csv)
        if args.metric_csv
        else recons_eval_root / "outputs" / "mv_recon" / f"{args.dataset_name}-metric-{args.model_name}.csv"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    metric_csv.parent.mkdir(parents=True, exist_ok=True)

    all_samples_csv = output_root / "_all_samples.csv"
    if args.overwrite and all_samples_csv.exists():
        all_samples_csv.unlink()

    dataset = NRGBD(
        NRGBD_DIR=str(nrgbd_dir),
        load_img_size=args.load_img_size,
        cache_file=str(cache_file),
    )
    with seq_id_map_path.open("r", encoding="utf-8") as f:
        seq_id_map = json.load(f)

    sums = {
        "Acc-mean": 0.0,
        "Acc-med": 0.0,
        "Comp-mean": 0.0,
        "Comp-med": 0.0,
        "NC-mean": 0.0,
        "NC-med": 0.0,
        "NC1-mean": 0.0,
        "NC1-med": 0.0,
        "NC2-mean": 0.0,
        "NC2-med": 0.0,
    }

    for seq_name, ids in seq_id_map.items():
        data = dataset.get_data(sequence_name=seq_name, ids=ids)
        gt_pts = data["pointclouds"]
        valid_mask = data["valid_mask"]
        images = data["images"]

        npz_path = vggt_root / seq_name / "vggt_outputs.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        pred_pts = load_pred_points(npz_path, args.pred_key)
        if pred_pts.shape != gt_pts.shape:
            raise ValueError(f"{seq_name}: pred {pred_pts.shape} != gt {gt_pts.shape}")

        colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)

        c, r, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
        pred_pts = c * np.einsum("nhwj,ij->nhwi", pred_pts, r) + t.T
        pred_flat = pred_pts[valid_mask].reshape(-1, 3)
        gt_flat = gt_pts[valid_mask].reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred_flat)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_flat)
        pcd_gt.colors = o3d.utility.Vector3dVector(colors)

        reg = o3d.pipelines.registration.registration_icp(
            pcd,
            pcd_gt,
            args.icp_threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        pcd.transform(reg.transformation)

        pcd.estimate_normals()
        pcd_gt.estimate_normals()
        pred_normal = np.asarray(pcd.normals)
        gt_normal = np.asarray(pcd_gt.normals)

        acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
        comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)

        row = {
            "seq": seq_name,
            "Acc-mean": acc,
            "Acc-med": acc_med,
            "Comp-mean": comp,
            "Comp-med": comp_med,
            "NC1-mean": nc1,
            "NC1-med": nc1_med,
            "NC2-mean": nc2,
            "NC2-med": nc2_med,
        }
        write_csv_row(all_samples_csv, row)

        sums["Acc-mean"] += acc
        sums["Acc-med"] += acc_med
        sums["Comp-mean"] += comp
        sums["Comp-med"] += comp_med
        sums["NC1-mean"] += nc1
        sums["NC1-med"] += nc1_med
        sums["NC2-mean"] += nc2
        sums["NC2-med"] += nc2_med
        sums["NC-mean"] += (nc1 + nc2) / 2
        sums["NC-med"] += (nc1_med + nc2_med) / 2

        if not args.no_save_ply:
            o3d.io.write_point_cloud(str(output_root / f"{seq_name}-pred.ply"), pcd)
            o3d.io.write_point_cloud(str(output_root / f"{seq_name}-gt.ply"), pcd_gt)

        print(f"{seq_name}: Acc={acc:.6f}, Comp={comp:.6f}, NC={(nc1 + nc2) / 2:.6f}")

    denominator = len(dataset)
    metrics = {"model": args.model_name, **{key: value / denominator for key, value in sums.items()}}
    with metric_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"\nSaved metric CSV -> {metric_csv}")
    print(metrics)


if __name__ == "__main__":
    main()
