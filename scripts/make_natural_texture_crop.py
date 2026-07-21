#!/usr/bin/env python3
"""Crop a natural texture from a TUM RGB frame for patch initialization.

The default crop targets the wall/poster area in
rgbd_dataset_freiburg3_sitting_static and writes a 128x128 texture image.
Coordinates are normalized to the source image size by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def find_image_dir(sequence_dir: Path) -> Path:
    for name in ("images", "rgb_90", "rgb"):
        image_dir = sequence_dir / name
        if image_dir.exists():
            return image_dir
    raise FileNotFoundError(f"No images/rgb_90/rgb directory found under {sequence_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_dir",
        default="/mnt/data/wangqq/recons_eval/data/tum/rgbd_dataset_freiburg3_sitting_static",
    )
    parser.add_argument(
        "--out",
        default="/mnt/data/wangqq/vggt/assets/natural_textures/tum_sitting_static_wall_natural_crop.png",
    )
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--cx", type=float, default=0.50, help="Crop center x, normalized unless --pixel_coords is set.")
    parser.add_argument("--cy", type=float, default=0.25, help="Crop center y, normalized unless --pixel_coords is set.")
    parser.add_argument("--crop_w", type=float, default=0.18, help="Crop width, normalized unless --pixel_coords is set.")
    parser.add_argument("--crop_h", type=float, default=0.25, help="Crop height, normalized unless --pixel_coords is set.")
    parser.add_argument("--texture_size", type=int, default=128)
    parser.add_argument("--pixel_coords", action="store_true", help="Treat cx/cy/crop_w/crop_h as pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir)
    image_dir = find_image_dir(sequence_dir)
    image_paths = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No png/jpg images found in {image_dir}")
    frame_index = min(max(args.frame_index, 0), len(image_paths) - 1)
    image_path = image_paths[frame_index]

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if args.pixel_coords:
        cx, cy = args.cx, args.cy
        crop_w, crop_h = args.crop_w, args.crop_h
    else:
        cx, cy = args.cx * width, args.cy * height
        crop_w, crop_h = args.crop_w * width, args.crop_h * height

    x0 = max(0, int(round(cx - crop_w / 2)))
    x1 = min(width, int(round(cx + crop_w / 2)))
    y0 = max(0, int(round(cy - crop_h / 2)))
    y1 = min(height, int(round(cy + crop_h / 2)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop box {(x0, y0, x1, y1)} for image size {(width, height)}")

    crop = image.crop((x0, y0, x1, y1)).resize(
        (args.texture_size, args.texture_size),
        Image.Resampling.BICUBIC,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out)

    print(f"source: {image_path}")
    print(f"image_size: {(width, height)}")
    print(f"crop_box: {(x0, y0, x1, y1)}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
