#!/usr/bin/env python3
"""Create one-frame VGGT scenes from a flat monodepth image directory.

Example input:
    data/nyu-v2/val/nyu_images/000000.png
    data/nyu-v2/val/nyu_images/000001.png

Example output:
    out_root/000000/images/000000.png
    out_root/000001/images/000001.png
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
    parser = argparse.ArgumentParser(description="Prepare flat monodepth images as one-frame VGGT scenes.")
    parser.add_argument("--image_dir", required=True, help="Flat image directory")
    parser.add_argument("--out_root", required=True, help="Output root for VGGT scenes")
    parser.add_argument("--image_ext", default="png", help="Image extension")
    parser.add_argument("--scene_prefix", default="", help="Optional prefix for scene folder names")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to export")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob(f"*.{args.image_ext}"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No *.{args.image_ext} files found in {image_dir}")

    for image_path in progress(image_paths, "Monodepth scenes"):
        scene_name = f"{args.scene_prefix}{image_path.stem}"
        scene_dir = out_root / scene_name
        dst_img = scene_dir / "images" / image_path.name
        copy_or_link(image_path, dst_img, args.copy)
        meta = {
            "mode": "monodepth",
            "scene": scene_name,
            "source_image": str(image_path),
        }
        with (scene_dir / "meta.json").open("w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)

    print(f"Prepared {len(image_paths)} one-frame scenes in {out_root}")


if __name__ == "__main__":
    main()
