"""Prepare single-frame TUM-10 pseudo-scenes for VLA-style patch training.

This does not modify the original TUM folders. It creates a new scene root where
each selected TUM frame is represented as one VGGT scene:

    <out_root>/<sequence>__frame_<idx>/images/<image>

The selected frames are uniformly sampled from each prepared TUM sequence. A
manifest is written so attack/evaluation can reuse exactly the same 10 frames.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from attack_vggt_new1 import find_images


def select_uniform_indices(length: int, frame_count: int) -> np.ndarray:
    if frame_count <= 0 or length <= frame_count:
        return np.arange(length, dtype=int)
    return np.unique(np.rint(np.linspace(0, length - 1, frame_count)).astype(int))


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum_root", required=True, help="Prepared TUM root with rgb_90/images sequence folders.")
    parser.add_argument("--out_root", required=True, help="Output root for single-frame pseudo-scenes.")
    parser.add_argument("--scene_pattern", default="rgbd_dataset_freiburg3_*")
    parser.add_argument("--frame_count", type=int, default=10)
    parser.add_argument("--sampling", choices=("uniform",), default="uniform")
    parser.add_argument("--copy", action="store_true", help="Copy image files instead of creating symlinks.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tum_root = Path(args.tum_root)
    out_root = Path(args.out_root)
    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    manifest: dict[str, dict] = {}
    for seq_dir in sorted(path for path in tum_root.glob(args.scene_pattern) if path.is_dir()):
        all_images = find_images(seq_dir)
        if not all_images:
            raise ValueError(f"No images found for {seq_dir}")
        frame_indices = select_uniform_indices(len(all_images), args.frame_count)
        image_paths = [all_images[int(idx)] for idx in frame_indices]

        manifest[seq_dir.name] = {
            "sampling": args.sampling,
            "frame_count_requested": args.frame_count,
            "frame_indices": [int(idx) for idx in frame_indices],
            "image_names": [Path(path).name for path in image_paths],
        }

        for image_path, frame_idx in zip(image_paths, frame_indices):
            src = Path(image_path).resolve()
            pseudo_scene = out_root / f"{seq_dir.name}__frame_{int(frame_idx):03d}"
            dst = pseudo_scene / "images" / src.name
            link_or_copy(src, dst, args.copy)
            total += 1

    manifest_path = out_root / "tum10_frame_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Prepared {total} single-frame TUM pseudo-scenes in {out_root}")
    print(f"Saved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
