#!/usr/bin/env python3
"""Prepare Neural-RGBD sampled scenes for geometry-aware VGGT patch attacks.

The geometry-aware attack script was originally written around TUM-style scene
folders. This converter creates lightweight TUM-like scene folders from the
Neural-RGBD sparse/dense frame id maps used by recons_eval:

    scene/images/*.png       sampled RGB frames
    scene/depth_90/*.png     sampled depth maps, indexed by local frame order
    scene/groundtruth_90.txt sampled camera-to-world poses in TUM text format

The original Neural-RGBD frame ids are preserved in nrgbd_geometry_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrgbd_root", required=True, help="Path to recons_eval/data/nrgbd")
    parser.add_argument("--seq_id_map", required=True, help="Path to NRGBD seq-id-map JSON")
    parser.add_argument("--out_root", required=True, help="Output root for prepared VGGT geometry scenes")
    parser.add_argument(
        "--manifest_out",
        default=None,
        help="Output frame manifest. Default: <out_root>/nrgbd_geometry_frame_manifest.json",
    )
    parser.add_argument(
        "--source_manifest_out",
        default=None,
        help="Output source-id manifest. Default: <out_root>/nrgbd_geometry_source_manifest.json",
    )
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing per-scene folders first")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def read_poses(path: Path) -> np.ndarray:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) % 4 != 0:
        raise ValueError(f"{path} has {len(rows)} pose rows, expected a multiple of 4")
    poses = []
    for start in range(0, len(rows), 4):
        mat = [[float(x) for x in rows[start + offset].split()] for offset in range(4)]
        poses.append(mat)
    poses_np = np.asarray(poses, dtype=np.float64)
    poses_np[:, :, 1:3] *= -1.0  # Match recons_eval.datasets.nrgbd gl-to-cv conversion.
    return poses_np


def rotation_to_quat_xyzw(rot: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-12)
    return tuple(float(v) for v in quat)


def write_tum_pose_file(path: Path, poses_c2w: np.ndarray, frame_ids: list[int]) -> None:
    lines = []
    for local_idx, frame_id in enumerate(frame_ids):
        pose = poses_c2w[int(frame_id)]
        qx, qy, qz, qw = rotation_to_quat_xyzw(pose[:3, :3])
        tx, ty, tz = pose[:3, 3]
        lines.append(
            f"{local_idx:.6f} {tx:.9f} {ty:.9f} {tz:.9f} "
            f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    nrgbd_root = Path(args.nrgbd_root)
    out_root = Path(args.out_root)
    manifest_out = Path(args.manifest_out) if args.manifest_out else out_root / "nrgbd_geometry_frame_manifest.json"
    source_manifest_out = (
        Path(args.source_manifest_out)
        if args.source_manifest_out
        else out_root / "nrgbd_geometry_source_manifest.json"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    with Path(args.seq_id_map).open("r", encoding="utf-8") as f:
        seq_id_map: dict[str, list[int]] = json.load(f)

    frame_manifest: dict[str, list[int]] = {}
    source_manifest: dict[str, list[int]] = {}
    n_frames = 0
    for seq_name, frame_ids in seq_id_map.items():
        seq_root = nrgbd_root / seq_name
        if not seq_root.exists():
            raise FileNotFoundError(seq_root)
        out_seq = out_root / seq_name
        if args.overwrite and out_seq.exists():
            shutil.rmtree(out_seq)
        (out_seq / "images").mkdir(parents=True, exist_ok=True)
        (out_seq / "depth_90").mkdir(parents=True, exist_ok=True)

        poses = read_poses(seq_root / "poses.txt")
        for local_idx, frame_id in enumerate(frame_ids):
            image_src = seq_root / "images" / f"img{frame_id}.png"
            depth_src = seq_root / "depth" / f"depth{frame_id}.png"
            if not image_src.exists():
                raise FileNotFoundError(image_src)
            if not depth_src.exists():
                raise FileNotFoundError(depth_src)
            link_or_copy(image_src, out_seq / "images" / f"{local_idx:06d}_img{frame_id}.png", args.copy)
            link_or_copy(depth_src, out_seq / "depth_90" / f"{local_idx:06d}_depth{frame_id}.png", args.copy)

        write_tum_pose_file(out_seq / "groundtruth_90.txt", poses, frame_ids)
        frame_manifest[seq_name] = list(range(len(frame_ids)))
        source_manifest[seq_name] = [int(frame_id) for frame_id in frame_ids]
        n_frames += len(frame_ids)

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(frame_manifest, indent=2), encoding="utf-8")
    source_manifest_out.write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(frame_manifest)} NRGBD geometry scenes and {n_frames} frames in {out_root}")
    print(f"Saved local-frame manifest -> {manifest_out}")
    print(f"Saved source-frame manifest -> {source_manifest_out}")


if __name__ == "__main__":
    main()
