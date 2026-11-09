#!/usr/bin/env python3
"""Re-stitch saved pose windows using camera centres only.

The four-task evaluator may use dense points to estimate the relative scale of
adjacent windows.  Camera-pose evaluation must not depend on another VGGT head,
so the main-table trajectory is rebuilt here from the shared camera predictions
alone.  Ground truth is copied only after stitching and is never exposed to the
window-to-window alignment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from eval_stitched_pose import stitch_windows

    source = args.input_root / args.scene
    records = []
    for path in sorted((source / "pose_windows").glob("w*.npz")):
        with np.load(path) as data:
            records.append(
                {
                    "frames": data["frames"].astype(np.int64).tolist(),
                    "clean_c2w": data["clean_c2w"].astype(np.float64),
                    "attack_c2w": data["attack_c2w"].astype(np.float64),
                }
            )
    if not records:
        raise FileNotFoundError(f"No pose windows under {source / 'pose_windows'}")

    clean_frames, clean_c2w, clean_diag = stitch_windows(records, "clean_c2w")
    attack_frames, attack_c2w, attack_diag = stitch_windows(records, "attack_c2w")
    if not np.array_equal(clean_frames, attack_frames):
        raise AssertionError("Clean/attack stitched frame indices differ")

    with np.load(source / "stitched_trajectories.npz") as previous:
        previous_frames = previous["frame_indices"].astype(np.int64)
        if not np.array_equal(clean_frames, previous_frames):
            raise AssertionError("Camera-only and source stitched frame indices differ")
        gt_c2w = previous["gt_c2w"].astype(np.float32)

    destination = args.output_root / args.scene
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "stitched_trajectories.npz",
        frame_indices=clean_frames,
        clean_c2w=clean_c2w.astype(np.float32),
        attack_c2w=attack_c2w.astype(np.float32),
        gt_c2w=gt_c2w,
    )
    (destination / "stitch_diagnostics.json").write_text(
        json.dumps(
            {
                "scene": args.scene,
                "scale_source": "camera_centres_only",
                "gt_used_for_stitching": False,
                "clean": clean_diag,
                "attack": attack_diag,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{args.scene}: {len(clean_frames)} frames from {len(records)} windows")


if __name__ == "__main__":
    main()
