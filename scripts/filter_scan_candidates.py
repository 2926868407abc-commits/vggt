"""Renderer-exact preflight for automatic patch-location candidates.

The plane extractor reports whether *some part of a plane* is visible.  That is
not enough for a concrete patch rectangle: a corner candidate may be outside all
ten images, strongly foreshortened, or extend beyond the physical carrier even
though its parent plane is visible.  This stage runs the production renderer's
projection, z-buffer and depth-support checks for every rectangle before any
texture training is scheduled.
"""

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
from diag_gauge_invariance import (  # noqa: E402
    match_depth_paths,
    preprocess_tum_depth_to_vggt_grid,
)
from run_position_scan import build_args  # noqa: E402


TUM = Path("/mnt/data/wangqq/recons_eval/data/tum")
PLANES = VG / "outputs/candidate_planes"
CLEAN = VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"


def candidate_rejection_reasons(
    meta: dict,
    *,
    min_visible_frames: int,
    min_supported_frames: int,
    min_frame_coverage: float,
    min_raw_frame_coverage: float,
    min_mean_coverage: float,
    min_visibility_ratio: float,
    min_support_ratio: float,
) -> tuple[list[str], dict]:
    """Return hard rejection reasons and compact metrics for one rectangle."""
    raw = np.asarray(meta["raw_projected_coverage_per_frame"], dtype=np.float64)
    visible = np.asarray(meta["mask_coverage_per_frame"], dtype=np.float64)
    visibility = np.asarray(meta["visibility_ratio_per_frame"], dtype=np.float64)
    support = np.asarray(meta["surface_support_ratio_per_frame"], dtype=np.float64)

    raw_frames = raw >= min_raw_frame_coverage
    visible_frames = visible >= min_frame_coverage
    supported_frames = raw_frames & (support >= min_support_ratio)
    active_visibility = visibility[raw_frames]
    active_support = support[raw_frames]

    metrics = {
        "raw_projected_coverage_per_frame": raw.tolist(),
        "mask_coverage_per_frame": visible.tolist(),
        "visibility_ratio_per_frame": visibility.tolist(),
        "surface_support_ratio_per_frame": support.tolist(),
        "raw_projected_coverage_mean": float(raw.mean()),
        "mask_coverage_mean": float(visible.mean()),
        "visible_frames": int(visible_frames.sum()),
        "supported_frames": int(supported_frames.sum()),
        "visibility_ratio_active_mean": (
            float(active_visibility.mean()) if active_visibility.size else 0.0
        ),
        "surface_support_ratio_active_mean": (
            float(active_support.mean()) if active_support.size else 0.0
        ),
    }

    reasons: list[str] = []
    if metrics["visible_frames"] < min_visible_frames:
        reasons.append("too_few_visible_frames")
    if metrics["supported_frames"] < min_supported_frames:
        reasons.append("too_few_supported_frames")
    if metrics["mask_coverage_mean"] < min_mean_coverage:
        reasons.append("mean_coverage_too_small")
    if metrics["visibility_ratio_active_mean"] < min_visibility_ratio:
        reasons.append("visibility_ratio_too_small")
    if metrics["surface_support_ratio_active_mean"] < min_support_ratio:
        reasons.append("surface_support_too_small")
    return reasons, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    parser.add_argument("--area", type=float, default=0.2)
    parser.add_argument("--tum_root", type=Path, default=TUM)
    parser.add_argument("--planes_dir", type=Path, default=PLANES)
    parser.add_argument("--clean_root", type=Path, default=CLEAN)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min_visible_frames", type=int, default=6)
    parser.add_argument("--min_supported_frames", type=int, default=6)
    parser.add_argument("--min_frame_coverage", type=float, default=0.002)
    parser.add_argument("--min_raw_frame_coverage", type=float, default=0.002)
    parser.add_argument("--min_mean_coverage", type=float, default=0.005)
    parser.add_argument("--min_visibility_ratio", type=float, default=0.30)
    parser.add_argument("--min_support_ratio", type=float, default=0.60)
    parser.add_argument("--support_abs_tolerance", type=float, default=0.06)
    parser.add_argument("--support_rel_tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    source = cli.input or cli.planes_dir / f"{cli.scene}_scan_a{cli.area}.json"
    spec = json.loads(source.read_text(encoding="utf-8"))
    plane_npz = np.load(cli.planes_dir / f"{cli.scene}_planes.npz", allow_pickle=True)
    c2w = plane_npz["gt_c2w"]
    tensor_hw = tuple(int(value) for value in plane_npz["tensor_hw"])

    summary = json.loads(
        (cli.clean_root / cli.scene / "attack_summary.json").read_text(encoding="utf-8")
    )
    image_paths = [str(path) for path in summary["image_paths"]]
    sequence_dir = cli.tum_root / cli.scene
    depth_paths = match_depth_paths(sequence_dir, image_paths)
    depth_maps = [
        preprocess_tum_depth_to_vggt_grid(path, Path(image_path), tensor_hw)
        for path, image_path in zip(depth_paths, image_paths)
    ]

    args = build_args(
        cli.scene,
        64,
        str(VG / "assets/hazard_textures/mde_attack_warnning.png"),
        tum_root=cli.tum_root,
        clean_root=cli.clean_root,
        checkpoint=VG / "checkpoints/VGGT-1B",
        frame_manifest=(VG / "data/tum_dynamics_10frame_individual_scenes/"
                        "tum10_frame_manifest.json"),
        output_dir=Path("/tmp/candidate_preflight"),
        surface_support_check=True,
        support_abs_tolerance=cli.support_abs_tolerance,
        support_rel_tolerance=cli.support_rel_tolerance,
    )
    intrinsics = np.asarray(
        [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    accepted: list[dict] = []
    rejected: list[dict] = []
    old_depth_maps = getattr(args, "_active_depth_maps", None)
    args._active_depth_maps = depth_maps
    try:
        for index, candidate in enumerate(spec["candidates"]):
            quad = np.asarray(candidate["quad"], dtype=np.float64).reshape(4, 3)
            _, _, meta = A.build_geometry_arrays_for_plane(
                quad,
                c2w,
                image_paths,
                tensor_hw,
                64,
                intrinsics,
                args,
            )
            reasons, metrics = candidate_rejection_reasons(
                meta,
                min_visible_frames=cli.min_visible_frames,
                min_supported_frames=cli.min_supported_frames,
                min_frame_coverage=cli.min_frame_coverage,
                min_raw_frame_coverage=cli.min_raw_frame_coverage,
                min_mean_coverage=cli.min_mean_coverage,
                min_visibility_ratio=cli.min_visibility_ratio,
                min_support_ratio=cli.min_support_ratio,
            )
            record = {**candidate, "validation": metrics}
            if reasons:
                record["rejection_reasons"] = reasons
                rejected.append(record)
            else:
                accepted.append(record)
            if (index + 1) % 20 == 0 or index + 1 == len(spec["candidates"]):
                print(f"validated {index + 1}/{len(spec['candidates'])}")
    finally:
        args._active_depth_maps = old_depth_maps

    planes: dict[int, dict] = {}
    for candidate in accepted:
        plane = int(candidate["plane"])
        planes.setdefault(plane, {"plane": plane, "accepted": 0, "rejected": 0})
        planes[plane]["accepted"] += 1
    for candidate in rejected:
        plane = int(candidate["plane"])
        planes.setdefault(plane, {"plane": plane, "accepted": 0, "rejected": 0})
        planes[plane]["rejected"] += 1

    thresholds = {
        "min_visible_frames": cli.min_visible_frames,
        "min_supported_frames": cli.min_supported_frames,
        "min_frame_coverage": cli.min_frame_coverage,
        "min_raw_frame_coverage": cli.min_raw_frame_coverage,
        "min_mean_coverage": cli.min_mean_coverage,
        "min_visibility_ratio": cli.min_visibility_ratio,
        "min_support_ratio": cli.min_support_ratio,
        "support_abs_tolerance": cli.support_abs_tolerance,
        "support_rel_tolerance": cli.support_rel_tolerance,
    }
    output = cli.out or cli.planes_dir / f"{cli.scene}_validated_a{cli.area}.json"
    output.write_text(
        json.dumps(
            {
                **{key: value for key, value in spec.items() if key != "candidates"},
                "source_spec": str(source),
                "thresholds": thresholds,
                "planes": sorted(planes.values(), key=lambda item: item["plane"]),
                "candidates": accepted,
                "rejected_candidates": rejected,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"accepted {len(accepted)}/{len(spec['candidates'])}")
    for plane in sorted(planes):
        row = planes[plane]
        print(f"plane {plane}: accepted={row['accepted']} rejected={row['rejected']}")
    print(f"-> {output}")


if __name__ == "__main__":
    main()
