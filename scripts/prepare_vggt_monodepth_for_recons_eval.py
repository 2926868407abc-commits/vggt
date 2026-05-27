#!/usr/bin/env python3
"""Convert saved VGGT batch outputs to recons_eval monodepth layout.

VGGT batch output:
    vggt_output_root/nyu_000001/vggt_outputs.npz
    vggt_output_root/nyu_000002/vggt_outputs.npz

recons_eval monodepth prediction layout:
    recons_eval/outputs/monodepth/<model_name>/nyu-v2/nyu_000001.npy
    recons_eval/outputs/monodepth/<model_name>/nyu-v2/nyu_000002.npy

Optionally this also flattens GT depths exported by export_nyu_v2_scenes.py:
    nyu_scenes_root/nyu_000001/depth/000000.npy
to:
    recons_eval/data/nyu-v2/val/nyu_depths/nyu_000001.npy
"""

from __future__ import annotations

import argparse
import shutil
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


def extract_first_depth(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as data:
        if "depth" not in data:
            raise KeyError(f"`depth` not found in {npz_path}")
        depth = np.asarray(data["depth"])

    # Common VGGT saved shape for one frame: (1, H, W, 1). Accept nearby forms.
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected a single 2D depth map after squeeze, got {depth.shape} from {npz_path}")
    return depth.astype(np.float32)


def copy_or_link(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VGGT depth predictions for recons_eval monodepth eval.")
    parser.add_argument("--vggt_output_root", required=True, help="Folder containing nyu_*/vggt_outputs.npz")
    parser.add_argument("--pred_out", required=True, help="Flat prediction output folder for recons_eval")
    parser.add_argument("--scene_pattern", default="nyu_*", help="Scene glob under vggt_output_root")
    parser.add_argument("--nyu_scenes_root", help="Optional exported NYU scene root containing GT depth")
    parser.add_argument("--gt_out", help="Optional flat GT depth output folder for recons_eval")
    parser.add_argument("--img_out", help="Optional flat image output folder for recons_eval")
    parser.add_argument("--symlink_gt", action="store_true", help="Symlink GT/images instead of copying")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction .npy files")
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

    n_pred = 0
    for scene_dir in progress(scene_dirs, "VGGT pred"):
        npz_path = scene_dir / "vggt_outputs.npz"
        if not npz_path.exists():
            continue
        pred_path = pred_out / f"{scene_dir.name}.npy"
        if pred_path.exists() and not args.overwrite:
            continue
        depth = extract_first_depth(npz_path)
        np.save(pred_path, depth)
        n_pred += 1

    print(f"Prepared {n_pred} prediction depth files in {pred_out}")

    if args.nyu_scenes_root is None:
        return
    if args.gt_out is None and args.img_out is None:
        return

    nyu_scenes_root = Path(args.nyu_scenes_root)
    n_gt = 0
    n_img = 0
    for scene_dir in progress(scene_dirs, "NYU gt/img"):
        src_scene = nyu_scenes_root / scene_dir.name
        if args.gt_out is not None:
            src_depth = src_scene / "depth" / "000000.npy"
            if src_depth.exists():
                copy_or_link(src_depth, Path(args.gt_out) / f"{scene_dir.name}.npy", args.symlink_gt)
                n_gt += 1
        if args.img_out is not None:
            src_img = src_scene / "images" / "000000.png"
            if src_img.exists():
                copy_or_link(src_img, Path(args.img_out) / f"{scene_dir.name}.png", args.symlink_gt)
                n_img += 1

    if args.gt_out is not None:
        print(f"Prepared {n_gt} GT depth files in {args.gt_out}")
    if args.img_out is not None:
        print(f"Prepared {n_img} image files in {args.img_out}")


if __name__ == "__main__":
    main()
