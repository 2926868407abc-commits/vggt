#!/usr/bin/env python3
"""Export NYU Depth V2 labeled .mat into VGGT scene folders.

The official nyu_depth_v2_labeled.mat is a MATLAB v7.3/HDF5 file whose RGB
images are commonly stored as (N, 3, 640, 480). This script writes each NYU
sample as a single-frame VGGT scene:

    out_root/nyu_000001/images/000000.png
    out_root/nyu_000001/depth/000000.npy
    out_root/nyu_000001/depth/000000.png
    out_root/nyu_000001/meta.json

Depth values in .npy are meters. Depth PNG is uint16 millimeters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: install h5py with `pip install h5py`.") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


NYU_RGB_INTRINSICS = {
    "fx": 518.8579011745019,
    "fy": 519.4696111212749,
    "cx": 325.58244941119034,
    "cy": 253.73616633400465,
    "width": 640,
    "height": 480,
}


def iter_progress(items: Iterable[int], total: int):
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc="Export NYUv2")


def inspect_h5(mat_path: Path) -> None:
    with h5py.File(mat_path, "r") as f:
        print(f"File: {mat_path}")

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")

        f.visititems(visit)


def read_sample(dataset, idx: int) -> np.ndarray:
    """Read a sample from common NYU HDF5 or MATLAB axis layouts."""
    shape = dataset.shape
    if len(shape) in (3, 4) and shape[0] > 16:
        return np.asarray(dataset[idx])
    if len(shape) in (3, 4) and shape[-1] > 16:
        return np.asarray(dataset[..., idx])
    raise ValueError(f"Cannot infer sample axis for dataset shape {shape}")


def sample_count(dataset) -> int:
    shape = dataset.shape
    if len(shape) in (3, 4) and shape[0] > 16:
        return shape[0]
    if len(shape) in (3, 4) and shape[-1] > 16:
        return shape[-1]
    raise ValueError(f"Cannot infer sample count for dataset shape {shape}")


def image_to_hwc_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert common NYU image layouts to HWC uint8 RGB."""
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D image sample, got shape {arr.shape}")

    if arr.shape[0] == 3:
        # Official HDF5 layout: C, W, H. Other exports may be C, H, W.
        arr = arr.transpose(2, 1, 0) if arr.shape[1] > arr.shape[2] else arr.transpose(1, 2, 0)
    elif arr.shape[-1] == 3:
        pass
    elif arr.shape[1] == 3:
        arr = arr.transpose(0, 2, 1)
    else:
        raise ValueError(f"Cannot infer RGB channel axis for image shape {arr.shape}")

    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def depth_to_hw_float32(arr: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Convert common NYU depth layouts to HW float32 meters."""
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D depth sample, got shape {arr.shape}")

    if arr.shape == image_hw:
        depth = arr
    elif arr.T.shape == image_hw:
        depth = arr.T
    else:
        # Official HDF5 layout is often W, H, so transpose when ambiguous.
        depth = arr.T if arr.shape[0] > arr.shape[1] else arr

    return np.asarray(depth, dtype=np.float32)


def save_depth_png_mm(depth_m: np.ndarray, path: Path) -> None:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(np.round(depth_m[valid] * 1000.0), 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NYU Depth V2 labeled .mat to VGGT scene folders.")
    parser.add_argument("--mat", required=True, help="Path to nyu_depth_v2_labeled.mat")
    parser.add_argument("--out_root", help="Output root for VGGT scenes")
    parser.add_argument("--image_key", default="images", help="HDF5 dataset key for RGB images")
    parser.add_argument("--depth_key", default="depths", help="HDF5 dataset key for GT depth")
    parser.add_argument("--scene_prefix", default="nyu", help="Scene folder prefix")
    parser.add_argument("--start", type=int, default=0, help="0-based start index in the .mat file")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to export")
    parser.add_argument("--stride", type=int, default=1, help="Export every Nth sample")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing scene files")
    parser.add_argument("--no_depth_npy", action="store_true", help="Do not write float32 depth .npy")
    parser.add_argument("--no_depth_png", action="store_true", help="Do not write uint16 millimeter depth .png")
    parser.add_argument("--inspect", action="store_true", help="Print HDF5 keys/shapes and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mat_path = Path(args.mat)

    if args.inspect:
        inspect_h5(mat_path)
        return
    if args.out_root is None:
        raise SystemExit("--out_root is required unless --inspect is used")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with h5py.File(mat_path, "r") as f:
        if args.image_key not in f:
            raise KeyError(f"Missing image dataset key `{args.image_key}`. Run with --inspect to list keys.")
        images = f[args.image_key]
        depths = f.get(args.depth_key)

        n_samples = sample_count(images)
        indices = list(range(args.start, n_samples, args.stride))
        if args.limit is not None:
            indices = indices[: args.limit]

        manifest_path = out_root / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for idx in iter_progress(indices, total=len(indices)):
                scene_name = f"{args.scene_prefix}_{idx + 1:06d}"
                scene_dir = out_root / scene_name
                image_dir = scene_dir / "images"
                depth_dir = scene_dir / "depth"
                image_path = image_dir / "000000.png"

                if image_path.exists() and not args.overwrite:
                    continue

                image_dir.mkdir(parents=True, exist_ok=True)
                depth_dir.mkdir(parents=True, exist_ok=True)

                image = image_to_hwc_uint8(read_sample(images, idx))
                Image.fromarray(image).save(image_path)

                meta = {
                    "dataset": "nyu_depth_v2_labeled",
                    "scene": scene_name,
                    "mat_index": idx,
                    "mat_index_1based": idx + 1,
                    "image": "images/000000.png",
                    "intrinsics": NYU_RGB_INTRINSICS,
                }

                if depths is not None:
                    depth = depth_to_hw_float32(read_sample(depths, idx), image.shape[:2])
                    if not args.no_depth_npy:
                        np.save(depth_dir / "000000.npy", depth)
                        meta["depth_npy"] = "depth/000000.npy"
                    if not args.no_depth_png:
                        save_depth_png_mm(depth, depth_dir / "000000.png")
                        meta["depth_png_mm"] = "depth/000000.png"

                with (scene_dir / "meta.json").open("w", encoding="utf-8") as fp:
                    json.dump(meta, fp, indent=2)
                manifest.write(json.dumps(meta) + "\n")

    with (out_root / "nyu_intrinsics.json").open("w", encoding="utf-8") as fp:
        json.dump(NYU_RGB_INTRINSICS, fp, indent=2)
    print(f"Exported {len(indices)} requested samples to {out_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
