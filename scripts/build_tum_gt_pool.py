#!/usr/bin/env python3
"""Build a disk-backed TUM depth/point/track GT pool for dynamic frame draws."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VG))
sys.path.insert(0, str(VG / "scripts"))

from attack_vggt_new1 import find_images  # noqa: E402
from build_tum_gt_point_track import (  # noqa: E402
    TUM_CX,
    TUM_CY,
    TUM_FX,
    TUM_FY,
    project_world_to_pixels,
    sample_depth,
)
from diag_gauge_invariance import (  # noqa: E402
    build_gt_world_points,
    match_depth_paths,
    preprocess_tum_depth_to_vggt_grid,
    projection_params,
    read_tum_rows,
    tum_rows_to_c2w,
)
from tum_gt_pool import (  # noqa: E402
    BASE_ARRAY_FILES,
    POOL_ARRAY_FILES,
    POOL_META_NAME,
)


DEFAULT_TUM_ROOT = Path("/mnt/data/wangqq/recons_eval/data/tum")


def parse_frames(value: str, pool_json: Path, pool_key: str) -> list[int]:
    if value.strip():
        frames = [int(part) for part in value.split(",") if part.strip()]
    else:
        payload = json.loads(pool_json.read_text(encoding="utf-8"))
        frames = [int(frame) for frame in payload[pool_key]]
    if not frames:
        raise ValueError("The GT pool frame list is empty")
    if len(frames) != len(set(frames)):
        raise ValueError("The GT pool frame list contains duplicates")
    return sorted(frames)


def open_array(scene_dir: Path, name: str, dtype: np.dtype, shape: tuple[int, ...]):
    return np.lib.format.open_memmap(
        scene_dir / POOL_ARRAY_FILES[name], mode="w+", dtype=dtype, shape=shape
    )


def build_pool(args: argparse.Namespace) -> dict:
    scene = args.scene
    seq_dir = Path(args.tum_root) / scene
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"TUM scene not found: {seq_dir}")
    frames = parse_frames(args.frames, Path(args.pool_json), args.pool_key)

    all_images = find_images(seq_dir)
    gt_rows = read_tum_rows(seq_dir / args.gt_name)
    if max(frames) >= len(all_images) or max(frames) >= len(gt_rows):
        raise IndexError(
            f"Frame {max(frames)} exceeds image/pose lengths "
            f"{len(all_images)}/{len(gt_rows)}"
        )
    image_paths = [all_images[frame] for frame in frames]
    gt_c2w = tum_rows_to_c2w(gt_rows, frames)
    depth_paths = match_depth_paths(seq_dir, image_paths)
    missing = [frames[index] for index, path in enumerate(depth_paths) if path is None]
    if missing:
        raise RuntimeError(f"Depth association failed for frames {missing}")

    tensor_hw = (int(args.tensor_height), int(args.tensor_width))
    projections = [projection_params(Path(path), tensor_hw) for path in image_paths]
    proj = projections[0]
    for frame, other in zip(frames[1:], projections[1:]):
        if any(abs(float(other[key]) - float(proj[key])) > 1e-9 for key in proj):
            raise RuntimeError(f"Frame {frame} has inconsistent preprocessing geometry")

    scene_out = Path(args.out_dir) / scene
    meta_path = scene_out / POOL_META_NAME
    array_files = BASE_ARRAY_FILES if args.lazy_tracks else POOL_ARRAY_FILES
    existing = [scene_out / filename for filename in array_files.values()]
    if (meta_path.exists() or any(path.exists() for path in existing)) and not args.overwrite:
        raise FileExistsError(
            f"GT pool already exists or is partial at {scene_out}; pass --overwrite explicitly"
        )
    scene_out.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        # Metadata is the completion marker. Remove it before an explicitly
        # requested rebuild so a concurrent reader cannot accept partial arrays.
        meta_path.unlink()

    n = len(frames)
    h, w = tensor_hw
    ys = np.linspace(args.margin * h, (1.0 - args.margin) * h, args.rows)
    xs = np.linspace(args.margin * w, (1.0 - args.margin) * w, args.cols)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    # Preserve the old independent builder's arithmetic exactly: projection is
    # computed from a float64 grid and only the stored query/track arrays are
    # rounded to float32.
    query_grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    p = int(query_grid.shape[0])

    arrays = {
        "gt_c2w": open_array(scene_out, "gt_c2w", np.float64, (n, 4, 4)),
        "depth": open_array(scene_out, "depth", np.float32, (n, h, w)),
        "point_map": open_array(scene_out, "point_map", np.float32, (n, h, w, 3)),
        "point_valid": open_array(scene_out, "point_valid", np.bool_, (n, h, w)),
        "query_points": open_array(scene_out, "query_points", np.float32, (n, p, 2)),
        "query_valid": open_array(scene_out, "query_valid", np.bool_, (n, p)),
    }
    if not args.lazy_tracks:
        arrays.update({
            "tracks": open_array(scene_out, "tracks", np.float32, (n, n, p, 2)),
            "track_known": open_array(scene_out, "track_known", np.bool_, (n, n, p)),
            "track_visible": open_array(scene_out, "track_visible", np.bool_, (n, n, p)),
        })
    arrays["gt_c2w"][:] = gt_c2w
    arrays["query_points"][:] = query_grid[None]
    arrays["query_valid"][:] = False
    if not args.lazy_tracks:
        arrays["tracks"][:] = 0.0
        arrays["track_known"][:] = False
        arrays["track_visible"][:] = False

    print(f"[{scene}] building {n} depth/point-map frames at {h}x{w}", flush=True)
    for pos, (frame, depth_path, image_path) in enumerate(
        zip(frames, depth_paths, image_paths)
    ):
        depth = preprocess_tum_depth_to_vggt_grid(
            Path(depth_path), Path(image_path), tensor_hw
        )
        points, valid = build_gt_world_points([depth], gt_c2w[pos:pos + 1], proj)
        arrays["depth"][pos] = depth.astype(np.float32, copy=False)
        arrays["point_map"][pos] = points[0].astype(np.float32, copy=False)
        arrays["point_valid"][pos] = valid[0]
        query_depth, inside = sample_depth(depth, query_grid)
        arrays["query_valid"][pos] = inside & np.isfinite(query_depth) & (query_depth > 1e-6)
        print(
            f"  frame {frame:>3}: depth valid {float(valid[0].mean()):.1%}, "
            f"queries {int(arrays['query_valid'][pos].sum())}/{p}",
            flush=True,
        )

    if args.lazy_tracks:
        print(f"[{scene}] validating lazy reprojection tracks", flush=True)
    else:
        print(f"[{scene}] building {n}x{n} reprojection tracks", flush=True)
    roundtrip_max = 0.0
    for q_pos, q_frame in enumerate(frames):
        q_keep = np.asarray(arrays["query_valid"][q_pos], dtype=bool)
        query = query_grid[q_keep]
        qz, _ = sample_depth(np.asarray(arrays["depth"][q_pos]), query)
        u = query[:, 0] / proj["scale_x"]
        v = (query[:, 1] + proj["crop_y"]) / proj["scale_y"]
        cam = np.stack(
            [
                (u - TUM_CX) / TUM_FX * qz,
                (v - TUM_CY) / TUM_FY * qz,
                qz,
            ],
            axis=-1,
        )
        world = cam @ gt_c2w[q_pos, :3, :3].T + gt_c2w[q_pos, :3, 3]

        target_positions = [q_pos] if args.lazy_tracks else range(n)
        for t_pos in target_positions:
            xy, z = project_world_to_pixels(world, gt_c2w[t_pos], proj)
            scene_z, inside = sample_depth(np.asarray(arrays["depth"][t_pos]), xy)
            has_depth = inside & np.isfinite(scene_z) & (scene_z > 1e-6)
            in_front = np.isfinite(z) & (z > 1e-6)
            not_occluded = z <= scene_z + float(args.occl_tol)
            if not args.lazy_tracks:
                arrays["tracks"][q_pos, t_pos, q_keep] = xy.astype(
                    np.float32, copy=False
                )
                arrays["track_known"][q_pos, t_pos, q_keep] = has_depth & in_front
                arrays["track_visible"][q_pos, t_pos, q_keep] = (
                    has_depth & in_front & not_occluded
                )

        if query.size:
            if args.lazy_tracks:
                error = float(np.max(np.abs(xy.astype(np.float32) - query)))
            else:
                error = float(
                    np.max(
                        np.abs(
                            np.asarray(arrays["tracks"][q_pos, q_pos, q_keep])
                            - query
                        )
                    )
                )
            roundtrip_max = max(roundtrip_max, error)
        else:
            error = float("nan")
        print(
            f"  query frame {q_frame:>3}: {int(q_keep.sum())} points, "
            f"round-trip {error:.6f}px",
            flush=True,
        )

    for array in arrays.values():
        array.flush()

    query_counts = np.asarray(arrays["query_valid"]).sum(axis=1).astype(int)
    visible_counts = (
        None
        if args.lazy_tracks
        else np.asarray(arrays["track_visible"]).sum(axis=2).astype(int)
    )
    metadata = {
        "format_version": 2 if args.lazy_tracks else 1,
        "scene": scene,
        "track_storage": "lazy_reprojection" if args.lazy_tracks else "dense",
        "frame_indices": frames,
        "image_names": [Path(path).name for path in image_paths],
        "image_paths": image_paths,
        "tensor_hw": list(tensor_hw),
        "query_grid": {
            "rows": int(args.rows),
            "cols": int(args.cols),
            "margin": float(args.margin),
            "points": p,
        },
        "projection": {key: float(value) for key, value in proj.items()},
        "camera_intrinsics": {
            "fx": float(TUM_FX),
            "fy": float(TUM_FY),
            "cx": float(TUM_CX),
            "cy": float(TUM_CY),
        },
        "occl_tol": float(args.occl_tol),
        "array_files": dict(array_files),
        "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "array_dtypes": {name: str(array.dtype) for name, array in arrays.items()},
        "query_counts": query_counts.tolist(),
        "target_visible_counts_min": (
            None if visible_counts is None else int(visible_counts.min())
        ),
        "target_visible_counts_max": (
            None if visible_counts is None else int(visible_counts.max())
        ),
        "roundtrip_max_px": roundtrip_max,
    }
    if not np.isfinite(roundtrip_max) or roundtrip_max > 1e-3:
        raise RuntimeError(f"Round-trip check failed: {roundtrip_max:.6f}px")
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "scene": scene,
        "frames": n,
        "tensor_hw": list(tensor_hw),
        "query_count_min": int(query_counts.min()),
        "query_count_max": int(query_counts.max()),
        "roundtrip_max_px": roundtrip_max,
        "output": str(scene_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--tum_root", default=str(DEFAULT_TUM_ROOT))
    parser.add_argument("--gt_name", default="groundtruth_90.txt")
    parser.add_argument("--pool_json", default=str(VG / "data/tum10_split/eight_seq_pool.json"))
    parser.add_argument("--pool_key", default="dynamic_pool")
    parser.add_argument("--frames", default="")
    parser.add_argument("--out_dir", default=str(VG / "outputs/tum_gt_pool"))
    parser.add_argument("--tensor_height", type=int, default=392)
    parser.add_argument("--tensor_width", type=int, default=518)
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--occl_tol", type=float, default=0.05)
    parser.add_argument(
        "--lazy_tracks",
        action="store_true",
        help="Store per-frame geometry only and reproject tracks for each sampled set.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_pool(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
