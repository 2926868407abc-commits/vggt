"""Compare differentiable budget proxies with the calibrated clean statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG))
import attack_vggt_geometry_tum10 as A  # noqa: E402


def build(scene: str, axis: int) -> argparse.Namespace:
    argv = [
        "attack.py", "--tum_root", "/mnt/data/wangqq/recons_eval/data/tum",
        "--scene_pattern", scene, "--output_dir", "/tmp/budget_metric_diag",
        "--frame_manifest", str(VG / "data/tum_dynamics_10frame_individual_scenes"
                                    "/tum10_frame_manifest.json"),
        "--ckpt", str(VG / "checkpoints/VGGT-1B"), "--iterations", "1",
        "--texture_size", "128", "--attack_loss", "geometry_joint_gauge_budgeted",
        "--piecewise_gauge_family", "orthogonal_mode", "--piecewise_gauge_magnitude", "3",
        "--orthogonal_mode_axis", str(axis), "--joint_track_grid_rows", "4",
        "--joint_track_grid_cols", "6", "--joint_track_iters", "2",
        "--filter_budget_json", str(VG / "configs/tum10_filter_budgets.json"),
        "--filter_budget_fraction", "0.8", "--budget_reproj_stride", "4",
        "--plane_mode", "fused_depth_surface", "--plane_width", "0.30",
        "--plane_height", "0.20", "--clean_vggt_output_root",
        str(VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"),
        "--surface_score_mode", "coverage", "--surface_coverage_min", "0.005",
        "--surface_coverage_max", "0.06", "--surface_min_visible_frames", "8",
        "--surface_min_visibility_ratio", "0.80", "--surface_min_support_ratio", "0.50",
        "--surface_support_abs_tolerance", "0.10", "--surface_support_rel_tolerance", "0.06",
        "--fused_max_plane_residual", "0.12", "--visibility_depth_margin", "0.08",
        "--seed", "0", "--use_depth_visibility", "--optimize_geometry",
        "--surface_support_check",
    ]
    saved, sys.argv = sys.argv, argv
    try:
        return A.parse_args()
    finally:
        sys.argv = saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_xyz")
    ap.add_argument("--axis", type=int, default=0)
    cli = ap.parse_args()
    args = build(cli.scene, cli.axis)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = A.load_model(args, device)
    for p in model.parameters():
        p.requires_grad_(False)
    seq = A.list_scene_dirs(Path(args.tum_root), args.scene_pattern)[0]
    manifest = A.load_frame_manifest(args.frame_manifest)
    intr = np.asarray([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]],
                      dtype=np.float64)
    with torch.no_grad():
        item = A.load_tum_sequence(seq, manifest[seq.name], args.gt_name, intr,
                                   args.texture_size, args, device, model=model, dtype=dtype)
        track_target = item["joint_track_targets"]
        preds = A.forward_all_geometry_heads(
            model, item["images"], dtype, query_points=track_target["query_points"],
            track_iters=args.joint_track_iters)
        metrics = A.differentiable_filter_metrics(preds, item, args)
        track = A.budget_track_error_pixels(preds, track_target,
                                            args.joint_track_min_visibility)
    print(f"scene {seq.name}")
    for name, value in metrics.items():
        spec = item["filter_budget"]["metrics"][name]
        safe, violation = A.budget_safe_value_and_violation(
            value, spec, args.filter_budget_fraction)
        print(f"{name:<20} proxy={float(value):.8g} clean={spec['clean']:.8g} "
              f"safe={float(safe):.8g} threshold={spec['threshold']:.8g} "
              f"violation={float(violation):.5g}")
    print(f"{'track_px':<20} proxy={float(track):.8g} safe={0.8*args.budget_track_pixels:.8g}")


if __name__ == "__main__":
    main()
