#!/usr/bin/env python3
"""Convert one-frame VGGT Bonn outputs to recons_eval monodepth layout.

Input scene names should be produced by prepare_bonn_monodepth_scenes.py:
    vggt_output_root/rgbd_bonn_balloon2__<frame_stem>/vggt_outputs.npz

Output:
    pred_out/rgbd_bonn_balloon2/<frame_stem>.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def progress(items, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def extract_depth(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as data:
        if "depth" not in data:
            raise KeyError(f"`depth` not found in {npz_path}")
        depth = np.asarray(data["depth"])
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected one 2D depth map, got {depth.shape} from {npz_path}")
    return depth.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Bonn VGGT monodepth predictions for recons_eval.")
    parser.add_argument("--vggt_output_root", required=True, help="Folder containing one-frame VGGT scene outputs")
    parser.add_argument("--pred_out", required=True, help="Output folder, e.g. outputs/monodepth/model/bonn")
    parser.add_argument("--scene_pattern", default="rgbd_bonn_*__*", help="Scene glob under vggt_output_root")
    parser.add_argument("--separator", default="__", help="Separator between sequence name and frame stem")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vggt_output_root = Path(args.vggt_output_root)
    pred_out = Path(args.pred_out)
    pred_out.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([p for p in vggt_output_root.glob(args.scene_pattern) if p.is_dir()])
    if not scene_dirs:
        raise SystemExit(f"No scene folders matched {vggt_output_root / args.scene_pattern}")

    n = 0
    for scene_dir in progress(scene_dirs, "Bonn VGGT monodepth"):
        if args.separator not in scene_dir.name:
            continue
        seq, frame_stem = scene_dir.name.split(args.separator, 1)
        npz_path = scene_dir / "vggt_outputs.npz"
        if not npz_path.exists():
            continue
        out_path = pred_out / seq / f"{frame_stem}.npy"
        if out_path.exists() and not args.overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, extract_depth(npz_path))
        n += 1

    print(f"Prepared {n} Bonn monodepth prediction files in {pred_out}")


if __name__ == "__main__":
    main()
