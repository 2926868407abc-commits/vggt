#!/usr/bin/env python3
"""Convert VGGT batch sequence outputs to recons_eval depth prediction layout.

VGGT batch output:
    vggt_output_root/rgbd_bonn_balloon2/vggt_outputs.npz

recons_eval video/monodepth prediction layout for video datasets:
    recons_eval/outputs/videodepth/<model_name>/bonn/rgbd_bonn_balloon2/<frame>.npy
    or
    recons_eval/outputs/monodepth/<model_name>/bonn/rgbd_bonn_balloon2/<frame>.npy
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


def scene_sort_key(path: Path) -> tuple:
    name = path.name
    tail = name.rsplit("_", 1)[-1]
    if tail.isdigit():
        return (name[: -len(tail)], int(tail))
    return (name, -1)


def load_depths(npz_path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(npz_path) as data:
        if "depth" not in data:
            raise KeyError(f"`depth` not found in {npz_path}")
        depth = np.asarray(data["depth"])
        image_paths = data["image_paths"].tolist() if "image_paths" in data else None

    depth = np.squeeze(depth)
    if depth.ndim == 2:
        depth = depth[None]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape (N,H,W), got {depth.shape} from {npz_path}")

    if image_paths is None:
        names = [f"frame_{idx:04d}.npy" for idx in range(depth.shape[0])]
    else:
        names = [f"{Path(str(p)).stem}.npy" for p in image_paths]
        if len(names) != depth.shape[0]:
            raise ValueError(f"image_paths count != depth count in {npz_path}")
    return depth.astype(np.float32), names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VGGT sequence depth predictions for recons_eval.")
    parser.add_argument("--vggt_output_root", required=True, help="Folder containing sequence/vggt_outputs.npz")
    parser.add_argument("--pred_out", required=True, help="Output root, e.g. outputs/videodepth/model/bonn")
    parser.add_argument("--scene_pattern", default="*", help="Sequence glob under vggt_output_root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vggt_output_root = Path(args.vggt_output_root)
    pred_out = Path(args.pred_out)
    pred_out.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted(
        [p for p in vggt_output_root.glob(args.scene_pattern) if p.is_dir()],
        key=scene_sort_key,
    )
    if not scene_dirs:
        raise SystemExit(f"No scene folders matched {vggt_output_root / args.scene_pattern}")

    total = 0
    for scene_dir in progress(scene_dirs, "VGGT sequence depth"):
        npz_path = scene_dir / "vggt_outputs.npz"
        if not npz_path.exists():
            continue
        depths, names = load_depths(npz_path)
        scene_out = pred_out / scene_dir.name
        scene_out.mkdir(parents=True, exist_ok=True)
        for depth, name in zip(depths, names):
            out_path = scene_out / name
            if out_path.exists() and not args.overwrite:
                continue
            np.save(out_path, depth)
            total += 1

    print(f"Prepared {total} frame depth files in {pred_out}")


if __name__ == "__main__":
    main()
