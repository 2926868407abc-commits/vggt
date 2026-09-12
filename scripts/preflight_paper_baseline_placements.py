"""Renderer-exact, attack-score-free preflight for paper baseline placements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


VG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VG))
sys.path.insert(0, str(VG / "scripts"))

import attack_vggt_geometry_tum10 as A  # noqa: E402
from build_paper_baseline_placements import (  # noqa: E402
    plane_quad,
    quad_dimensions,
    quad_string,
)
from diag_gauge_invariance import (  # noqa: E402
    match_depth_paths,
    preprocess_tum_depth_to_vggt_grid,
)
from run_position_scan import build_args  # noqa: E402


TUM = Path("/mnt/data/wangqq/recons_eval/data/tum")
CLEAN = VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"


def metrics_for_quad(
    quad: np.ndarray,
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
) -> dict:
    old = getattr(args, "_active_depth_maps", None)
    args._active_depth_maps = depth_maps
    try:
        _, _, meta = A.build_geometry_arrays_for_plane(
            quad, c2w, image_paths, tensor_hw, 64, intrinsics, args
        )
    finally:
        args._active_depth_maps = old
    return {
        key: meta[key]
        for key in (
            "raw_projected_coverage_per_frame",
            "raw_projected_coverage_mean",
            "mask_coverage_per_frame",
            "mask_coverage_mean",
            "visibility_ratio_per_frame",
            "visibility_ratio_mean",
            "surface_support_ratio_per_frame",
            "surface_support_ratio_mean",
        )
    }


def fit_vo_scale(
    plane: dict,
    target: float,
    metric_fn,
) -> tuple[float, np.ndarray, dict]:
    candidates = []
    # A dense deterministic one-dimensional search is safer than assuming exact
    # monotonicity after image clipping and z-buffer visibility.
    for scale in np.linspace(0.05, 1.0, 40):
        quad = plane_quad(plane, scale_u=float(scale), scale_v=float(scale))
        metrics = metric_fn(quad)
        candidates.append((abs(metrics["raw_projected_coverage_mean"] - target), scale, quad, metrics))
    _, scale, quad, metrics = min(candidates, key=lambda item: item[0])
    return float(scale), quad, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--tum_root", type=Path, default=TUM)
    parser.add_argument("--clean_root", type=Path, default=CLEAN)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min_visible_frames", type=int, default=6)
    parser.add_argument("--min_support_ratio", type=float, default=0.50)
    args_cli = parser.parse_args()

    spec = json.loads(args_cli.placements.read_text(encoding="utf-8"))
    scene = spec["scene"]
    planes_path = Path(spec["source_planes"])
    planes_meta = json.loads(planes_path.read_text(encoding="utf-8"))
    planes_npz = np.load(planes_path.with_suffix(".npz"), allow_pickle=True)
    c2w = planes_npz["gt_c2w"]
    tensor_hw = tuple(int(value) for value in planes_npz["tensor_hw"])

    summary = json.loads(
        (args_cli.clean_root / scene / "attack_summary.json").read_text(encoding="utf-8")
    )
    image_paths = [str(value) for value in summary["image_paths"]]
    depth_paths = match_depth_paths(args_cli.tum_root / scene, image_paths)
    depth_maps = [
        preprocess_tum_depth_to_vggt_grid(path, Path(image), tensor_hw)
        for path, image in zip(depth_paths, image_paths)
    ]
    render_args = build_args(
        scene,
        64,
        str(VG / "assets/hazard_textures/mde_attack_warnning.png"),
        tum_root=args_cli.tum_root,
        clean_root=args_cli.clean_root,
        checkpoint=VG / "checkpoints/VGGT-1B",
        frame_manifest=(
            VG / "data/tum_dynamics_10frame_individual_scenes/tum10_frame_manifest.json"
        ),
        output_dir=Path("/tmp/paper_baseline_preflight"),
        surface_support_check=True,
        support_abs_tolerance=0.06,
        support_rel_tolerance=0.02,
    )
    intrinsics = np.asarray(
        [[render_args.fx, 0.0, render_args.cx],
         [0.0, render_args.fy, render_args.cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    def evaluate(quad: np.ndarray) -> dict:
        return metrics_for_quad(
            quad, c2w, image_paths, tensor_hw, intrinsics, render_args, depth_maps
        )

    for row in spec["profiles"]:
        if row["attack_loss"] == "baseline_vo_mprms":
            plane = planes_meta["candidates"][int(row["source_plane_index"])]
            target = float(row["target_projected_fov"])
            scale, quad, preflight = fit_vo_scale(plane, target, evaluate)
            row["partition_linear_scale"] = scale
            row["explicit_quad_world"] = quad_string(quad)
            width, height, area = quad_dimensions(quad)
            row.update(
                patch_width_m=width,
                patch_height_m=height,
                outer_quad_area_m2=area,
            )
        else:
            quad = np.asarray(
                [float(value) for value in row["explicit_quad_world"].split(",")]
            ).reshape(4, 3)
            preflight = evaluate(quad)
        visible_frames = int(
            np.count_nonzero(np.asarray(preflight["mask_coverage_per_frame"]) > 0.0)
        )
        supported = np.asarray(preflight["surface_support_ratio_per_frame"])
        support_active = supported[supported > 0.0]
        support_mean = float(support_active.mean()) if support_active.size else 0.0
        row["preflight"] = {
            **preflight,
            "visible_frames": visible_frames,
            "active_surface_support_mean": support_mean,
            "passed": bool(
                visible_frames >= args_cli.min_visible_frames
                and support_mean >= args_cli.min_support_ratio
            ),
        }
        print(
            f"{row['name']:<38} raw={preflight['raw_projected_coverage_mean']:.4f} "
            f"visible={preflight['mask_coverage_mean']:.4f} frames={visible_frames}/10 "
            f"support={support_mean:.3f} pass={row['preflight']['passed']}"
        )

    spec["preflight"] = {
        "renderer_exact": True,
        "uses_attack_scores": False,
        "min_visible_frames": args_cli.min_visible_frames,
        "min_support_ratio": args_cli.min_support_ratio,
    }
    output = args_cli.out or args_cli.placements
    output.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {output}")


if __name__ == "__main__":
    main()
