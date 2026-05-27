#!/usr/bin/env python3
"""Create one-frame VGGT scenes from recons_eval Bonn RGB-D sequences.

Input expected by recons_eval after datasets/preprocess/prepare_bonn.py:
    bonn_root/rgbd_bonn_balloon2/rgb_110/*.png
    bonn_root/rgbd_bonn_balloon2/depth_110/*.png

Output expected by batch_vggt_inference.py for monocular depth:
    out_root/rgbd_bonn_balloon2__<frame_stem>/images/<frame>.png
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def progress(items, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def copy_or_link(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Bonn one-frame VGGT scenes for monodepth.")
    parser.add_argument("--bonn_root", required=True, help="Path to rgbd_bonn_dataset")
    parser.add_argument("--out_root", required=True, help="Output root for one-frame VGGT scenes")
    parser.add_argument("--seq_pattern", default="rgbd_bonn_*", help="Sequence glob under bonn_root")
    parser.add_argument("--seqs", nargs="*", help="Optional explicit sequence names to export")
    parser.add_argument("--image_dir", default="rgb_110", help="RGB frame directory inside each sequence")
    parser.add_argument("--image_ext", default="png", help="RGB frame extension")
    parser.add_argument("--separator", default="__", help="Separator between sequence name and frame stem")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bonn_root = Path(args.bonn_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.seqs:
        seq_dirs = [bonn_root / seq for seq in args.seqs]
    else:
        seq_dirs = sorted([p for p in bonn_root.glob(args.seq_pattern) if p.is_dir()])
    if not seq_dirs:
        raise SystemExit(f"No Bonn sequences matched {bonn_root / args.seq_pattern}")
    missing = [str(p) for p in seq_dirs if not p.is_dir()]
    if missing:
        raise SystemExit(f"Missing Bonn sequence folders: {missing}")

    n = 0
    for seq_dir in progress(seq_dirs, "Bonn monodepth scenes"):
        frame_paths = sorted((seq_dir / args.image_dir).glob(f"*.{args.image_ext}"))
        for frame_path in frame_paths:
            scene_name = f"{seq_dir.name}{args.separator}{frame_path.stem}"
            scene_dir = out_root / scene_name
            dst_img = scene_dir / "images" / frame_path.name
            copy_or_link(frame_path, dst_img, args.copy)
            meta = {
                "dataset": "bonn",
                "mode": "monodepth",
                "scene": scene_name,
                "sequence": seq_dir.name,
                "frame": frame_path.name,
                "source_image": str(frame_path),
            }
            with (scene_dir / "meta.json").open("w", encoding="utf-8") as fp:
                json.dump(meta, fp, indent=2)
            n += 1

    print(f"Prepared {n} one-frame scenes in {out_root}")


if __name__ == "__main__":
    main()
