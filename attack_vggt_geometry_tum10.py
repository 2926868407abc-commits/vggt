"""GT-geometry-aware planar patch attack for VGGT on TUM-10.

This is a GT-geometry-consistent physical patch pipeline:

* one learnable texture is shared by all frames and sequences
* for each TUM sequence, a 3D planar patch is either placed in front of the
  first selected camera or automatically anchored on a depth-observed surface
* the same 3D plane is projected into all selected frames with GT poses and RGB
  intrinsics, then differentiably sampled into the VGGT input images
* optional TUM depth visibility acts as a z-buffer for occlusion
* optional geometry search chooses patch position, surface orientation, and size
  by visible cross-view coverage before texture optimization
* optional physical EOT applies printable color clamping, lighting jitter, and
  camera noise during texture optimization
* the objective is still label-free feature L1: maximize
  L1(feature_adv, feature_clean)

The script writes official-style vggt_outputs.npz plus attack_summary.json so
scripts/eval_vggt_tum_pose_for_recons_eval_tum10.py can evaluate ATE/RPE.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from attack_vggt_new1 import (
    detach_predictions,
    extract_features,
    feature_l1_loss,
    find_images,
    forward_vggt,
    load_model,
    save_official_style_npz,
    set_random_seeds,
)
from attack_vggt_vla_style import load_frame_manifest, scheduled_lr
from vggt.utils.load_fn import load_and_preprocess_images


def list_scene_dirs(root: Path, pattern: str) -> list[Path]:
    scene_dirs = sorted(path for path in root.glob(pattern) if path.is_dir())
    if not scene_dirs:
        raise ValueError(f"No TUM scene folders matched {root / pattern}")
    return scene_dirs


def read_tum_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(stripped.split())
    return rows


def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def tum_rows_to_c2w(rows: list[list[str]], frame_indices: list[int]) -> np.ndarray:
    poses = []
    for frame_idx in frame_indices:
        if frame_idx >= len(rows):
            raise IndexError(f"Frame index {frame_idx} out of range for {len(rows)} GT rows")
        row = rows[int(frame_idx)]
        tx, ty, tz = map(float, row[1:4])
        qx, qy, qz, qw = map(float, row[4:8])
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = quat_xyzw_to_rot(qx, qy, qz, qw)
        c2w[:3, 3] = [tx, ty, tz]
        poses.append(c2w)
    return np.stack(poses, axis=0)


def preprocess_projection_params(image_path: str, tensor_hw: tuple[int, int]) -> dict[str, float]:
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
    tensor_h, tensor_w = tensor_hw
    new_w = 518
    new_h = round(orig_h * (new_w / orig_w) / 14) * 14
    crop_y = max(0, (new_h - 518) // 2)
    return {
        "orig_w": float(orig_w),
        "orig_h": float(orig_h),
        "scale_x": float(new_w) / float(orig_w),
        "scale_y": float(new_h) / float(orig_h),
        "crop_y": float(crop_y),
        "tensor_w": float(tensor_w),
        "tensor_h": float(tensor_h),
    }


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def tensor_xy_to_original_uv(xy: np.ndarray, proj: dict[str, float]) -> np.ndarray:
    uv = np.empty_like(xy, dtype=np.float64)
    uv[..., 0] = xy[..., 0] / proj["scale_x"]
    uv[..., 1] = (xy[..., 1] + proj["crop_y"]) / proj["scale_y"]
    return uv


def unproject_tensor_xy(xy: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray, proj: dict[str, float]) -> np.ndarray:
    uv = tensor_xy_to_original_uv(xy, proj)
    z = np.asarray(depth_m, dtype=np.float64)
    points = np.empty((*uv.shape[:-1], 3), dtype=np.float64)
    points[..., 0] = (uv[..., 0] - intrinsics[0, 2]) / intrinsics[0, 0] * z
    points[..., 1] = (uv[..., 1] - intrinsics[1, 2]) / intrinsics[1, 1] * z
    points[..., 2] = z
    return points


def read_depth_rows(path: Path) -> list[tuple[float, Path]]:
    rows: list[tuple[float, Path]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), Path(parts[1])))
            except ValueError:
                continue
    return rows


def image_timestamp(path: str) -> float | None:
    try:
        return float(Path(path).stem)
    except ValueError:
        return None


def match_depth_paths(
    seq_dir: Path,
    image_paths: list[str],
    frame_indices: list[int],
    args: argparse.Namespace,
) -> list[Path | None]:
    depth90_dir = seq_dir / "depth_90"
    if depth90_dir.exists():
        depth90_images = find_images(depth90_dir)
        if depth90_images and max(frame_indices) < len(depth90_images):
            return [Path(depth90_images[int(idx)]) for idx in frame_indices]

    rows = read_depth_rows(seq_dir / args.depth_txt_name)
    if not rows:
        return [None for _ in image_paths]

    timestamps = np.asarray([row[0] for row in rows], dtype=np.float64)
    matched = []
    for image_path in image_paths:
        ts = image_timestamp(image_path)
        if ts is None:
            matched.append(None)
            continue
        best_idx = int(np.argmin(np.abs(timestamps - ts)))
        best_dt = abs(float(timestamps[best_idx]) - ts)
        if best_dt > args.depth_match_max_dt:
            matched.append(None)
        else:
            matched.append(seq_dir / rows[best_idx][1])
    return matched


def load_depth_preprocessed(
    depth_path: Path | None,
    image_path: str,
    tensor_hw: tuple[int, int],
    args: argparse.Namespace,
) -> np.ndarray | None:
    if depth_path is None or not depth_path.exists():
        return None
    with Image.open(image_path) as image:
        orig_w, orig_h = image.size
    tensor_h, tensor_w = tensor_hw
    new_w = 518
    new_h = round(orig_h * (new_w / orig_w) / 14) * 14
    crop_y = max(0, (new_h - 518) // 2)

    depth_image = Image.open(depth_path)
    depth_image = depth_image.resize((new_w, new_h), Image.Resampling.NEAREST)
    if new_h > tensor_h:
        depth_image = depth_image.crop((0, crop_y, new_w, crop_y + tensor_h))
    depth = np.asarray(depth_image, dtype=np.float32) / float(args.depth_scale)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def load_depth_maps(
    seq_dir: Path,
    image_paths: list[str],
    frame_indices: list[int],
    tensor_hw: tuple[int, int],
    args: argparse.Namespace,
) -> list[np.ndarray | None]:
    if not args.use_depth_visibility and args.plane_mode != "auto_depth_surface":
        return [None for _ in image_paths]
    depth_paths = match_depth_paths(seq_dir, image_paths, frame_indices, args)
    return [
        load_depth_preprocessed(depth_path, image_path, tensor_hw, args)
        for depth_path, image_path in zip(depth_paths, image_paths)
    ]


def camera_to_tensor_xy(points_cam: np.ndarray, intrinsics: np.ndarray, proj: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((points_cam.shape[0], 2), dtype=np.float64)
    uv[:, 0] = intrinsics[0, 0] * (points_cam[:, 0] / np.maximum(z, 1e-6)) + intrinsics[0, 2]
    uv[:, 1] = intrinsics[1, 1] * (points_cam[:, 1] / np.maximum(z, 1e-6)) + intrinsics[1, 2]
    xy = np.empty_like(uv)
    xy[:, 0] = uv[:, 0] * proj["scale_x"]
    xy[:, 1] = uv[:, 1] * proj["scale_y"] - proj["crop_y"]
    valid &= xy[:, 0] >= 0
    valid &= xy[:, 0] <= proj["tensor_w"] - 1
    valid &= xy[:, 1] >= 0
    valid &= xy[:, 1] <= proj["tensor_h"] - 1
    return xy, valid


def homography_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    a = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(a)
    h = vh[-1].reshape(3, 3)
    return h / h[2, 2]


def build_fixed_patch_plane_world(first_c2w: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    z = args.plane_distance
    w = args.plane_width
    h = args.plane_height
    cx = args.plane_center_x
    cy = args.plane_center_y
    corners_cam = np.asarray(
        [
            [cx - w / 2, cy - h / 2, z, 1.0],
            [cx + w / 2, cy - h / 2, z, 1.0],
            [cx + w / 2, cy + h / 2, z, 1.0],
            [cx - w / 2, cy + h / 2, z, 1.0],
        ],
        dtype=np.float64,
    )
    return (first_c2w @ corners_cam.T).T[:, :3], {
        "placement_mode": "fixed_first_camera",
        "center_first_camera": [cx, cy, z],
        "width_m": w,
        "height_m": h,
    }


def rotate_axes(axis_u: np.ndarray, axis_v: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    radians = math.radians(degrees)
    cos_v = math.cos(radians)
    sin_v = math.sin(radians)
    return cos_v * axis_u + sin_v * axis_v, -sin_v * axis_u + cos_v * axis_v


def estimate_surface_axes(
    depth_map: np.ndarray,
    xy: np.ndarray,
    intrinsics: np.ndarray,
    proj: dict[str, float],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    x, y = float(xy[0]), float(xy[1])
    radius = int(args.normal_estimation_radius)
    h, w = depth_map.shape
    samples = np.asarray(
        [
            [x - radius, y],
            [x + radius, y],
            [x, y - radius],
            [x, y + radius],
        ],
        dtype=np.float64,
    )
    if np.any(samples[:, 0] < 0) or np.any(samples[:, 0] >= w) or np.any(samples[:, 1] < 0) or np.any(samples[:, 1] >= h):
        return None

    depths = np.asarray([depth_map[int(round(py)), int(round(px))] for px, py in samples], dtype=np.float64)
    if np.any(depths <= args.depth_min_m) or np.any(depths >= args.depth_max_m):
        return None

    points = unproject_tensor_xy(samples, depths, intrinsics, proj)
    p_left, p_right, p_up, p_down = points
    normal = np.cross(p_right - p_left, p_down - p_up)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None
    normal = normal / norm
    center_depth = depth_map[int(round(y)), int(round(x))]
    center = unproject_tensor_xy(np.asarray([x, y], dtype=np.float64), np.asarray(center_depth), intrinsics, proj)
    if np.dot(normal, center) > 0:
        normal = -normal

    axis_u = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    axis_u = axis_u - np.dot(axis_u, normal) * normal
    axis_norm = np.linalg.norm(axis_u)
    if axis_norm < 1e-8:
        axis_u = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        axis_u = axis_u - np.dot(axis_u, normal) * normal
        axis_norm = np.linalg.norm(axis_u)
    axis_u = axis_u / axis_norm
    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)
    return axis_u, axis_v, normal


def build_plane_world_from_camera_axes(
    first_c2w: np.ndarray,
    center_cam: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    corners_cam = np.asarray(
        [
            center_cam - axis_u * width / 2 - axis_v * height / 2,
            center_cam + axis_u * width / 2 - axis_v * height / 2,
            center_cam + axis_u * width / 2 + axis_v * height / 2,
            center_cam - axis_u * width / 2 + axis_v * height / 2,
        ],
        dtype=np.float64,
    )
    corners_h = np.concatenate([corners_cam, np.ones((4, 1), dtype=np.float64)], axis=1)
    return (first_c2w @ corners_h.T).T[:, :3]


def plane_depth_map(
    corners_cam: np.ndarray,
    intrinsics: np.ndarray,
    proj: dict[str, float],
    tensor_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    tensor_h, tensor_w = tensor_hw
    ys, xs = np.meshgrid(
        np.arange(tensor_h, dtype=np.float64),
        np.arange(tensor_w, dtype=np.float64),
        indexing="ij",
    )
    xy = np.stack([xs, ys], axis=-1)
    uv = tensor_xy_to_original_uv(xy, proj)
    rays = np.empty((tensor_h, tensor_w, 3), dtype=np.float64)
    rays[..., 0] = (uv[..., 0] - intrinsics[0, 2]) / intrinsics[0, 0]
    rays[..., 1] = (uv[..., 1] - intrinsics[1, 2]) / intrinsics[1, 1]
    rays[..., 2] = 1.0

    p0, p1, _, p3 = corners_cam
    normal = np.cross(p1 - p0, p3 - p0)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-10:
        return np.zeros((tensor_h, tensor_w), dtype=np.float64), np.zeros((tensor_h, tensor_w), dtype=bool)
    normal = normal / normal_norm
    denom = rays @ normal
    plane_d = -float(np.dot(normal, p0))
    valid = np.abs(denom) > 1e-8
    t = np.zeros((tensor_h, tensor_w), dtype=np.float64)
    t[valid] = -plane_d / denom[valid]
    valid &= t > 1e-6
    return t, valid


def build_geometry_arrays_for_plane(
    patch_world: np.ndarray,
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    tensor_h, tensor_w = tensor_hw
    ys, xs = torch.meshgrid(
        torch.arange(tensor_h, dtype=torch.float32),
        torch.arange(tensor_w, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3).numpy()

    src = np.asarray(
        [
            [0.0, 0.0],
            [texture_size - 1.0, 0.0],
            [texture_size - 1.0, texture_size - 1.0],
            [0.0, texture_size - 1.0],
        ],
        dtype=np.float64,
    )

    grids = []
    masks = []
    coverages = []
    raw_coverages = []
    visibility_ratios = []
    projected_corners = []
    depth_maps = getattr(args, "_active_depth_maps", None)
    for frame_idx, pose in enumerate(c2w):
        w2c = np.linalg.inv(pose)
        corners_h = np.concatenate([patch_world, np.ones((4, 1), dtype=np.float64)], axis=1)
        corners_cam = (w2c @ corners_h.T).T[:, :3]
        proj = preprocess_projection_params(image_paths[frame_idx], tensor_hw)
        dst, corner_valid = camera_to_tensor_xy(corners_cam, intrinsics, proj)
        projected_corners.append(dst.astype(float).tolist())

        if not bool(corner_valid.all()):
            grid = np.zeros((tensor_h, tensor_w, 2), dtype=np.float32)
            mask = np.zeros((1, tensor_h, tensor_w), dtype=np.float32)
            grids.append(grid)
            masks.append(mask)
            coverages.append(0.0)
            raw_coverages.append(0.0)
            visibility_ratios.append(0.0)
            continue

        h_mat = homography_from_points(src, dst)
        h_inv = np.linalg.inv(h_mat)
        src_h = (h_inv @ pix.T).T
        src_xy = src_h[:, :2] / src_h[:, 2:3]
        inside = (
            (src_xy[:, 0] >= 0)
            & (src_xy[:, 0] <= texture_size - 1)
            & (src_xy[:, 1] >= 0)
            & (src_xy[:, 1] <= texture_size - 1)
        )
        grid = np.zeros((tensor_h * tensor_w, 2), dtype=np.float32)
        grid[:, 0] = 2.0 * (src_xy[:, 0] / (texture_size - 1.0)) - 1.0
        grid[:, 1] = 2.0 * (src_xy[:, 1] / (texture_size - 1.0)) - 1.0
        grid = grid.reshape(tensor_h, tensor_w, 2)
        raw_inside = inside.reshape(tensor_h, tensor_w)
        mask_bool = raw_inside
        if args.use_depth_visibility and depth_maps is not None and depth_maps[frame_idx] is not None:
            patch_depth, depth_valid = plane_depth_map(corners_cam, intrinsics, proj, tensor_hw)
            scene_depth = depth_maps[frame_idx]
            scene_valid = (scene_depth > args.depth_min_m) & (scene_depth < args.depth_max_m)
            visible = depth_valid & scene_valid & (patch_depth <= scene_depth + args.visibility_depth_margin)
            mask_bool = raw_inside & visible
        mask = mask_bool.astype(np.float32).reshape(1, tensor_h, tensor_w)
        grids.append(grid)
        masks.append(mask)
        raw_coverage = float(raw_inside.mean())
        visible_coverage = float(mask.mean())
        raw_coverages.append(raw_coverage)
        coverages.append(visible_coverage)
        visibility_ratios.append(visible_coverage / raw_coverage if raw_coverage > 0 else 0.0)

    meta = {
        "plane_corners_world": patch_world.astype(float).tolist(),
        "projected_corners": projected_corners,
        "mask_coverage_per_frame": coverages,
        "mask_coverage_mean": float(np.mean(coverages)),
        "raw_projected_coverage_per_frame": raw_coverages,
        "raw_projected_coverage_mean": float(np.mean(raw_coverages)),
        "visibility_ratio_per_frame": visibility_ratios,
        "visibility_ratio_mean": float(np.mean(visibility_ratios)),
        "uses_depth_visibility": bool(args.use_depth_visibility),
    }
    return np.stack(grids, axis=0), np.stack(masks, axis=0), meta


def geometry_score(meta: dict) -> float:
    coverages = np.asarray(meta["mask_coverage_per_frame"], dtype=np.float64)
    if coverages.size == 0:
        return 0.0
    return float(coverages.mean() + 0.25 * coverages.min())


def choose_auto_depth_surface_plane(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
) -> tuple[np.ndarray, dict]:
    first_depth = depth_maps[0] if depth_maps else None
    if first_depth is None:
        patch_world, meta = build_fixed_patch_plane_world(c2w[0], args)
        meta["placement_mode"] = "fixed_fallback_no_depth"
        return patch_world, meta

    proj = preprocess_projection_params(image_paths[0], tensor_hw)
    tensor_h, tensor_w = tensor_hw
    margin_x = args.surface_search_margin * tensor_w
    margin_y = args.surface_search_margin * tensor_h
    grid_n = max(1, int(args.surface_candidate_grid))
    xs = np.linspace(margin_x, tensor_w - 1 - margin_x, grid_n)
    ys = np.linspace(margin_y, tensor_h - 1 - margin_y, grid_n)

    size_scales = [1.0]
    roll_degrees = [0.0]
    if args.optimize_geometry:
        size_scales = parse_float_list(args.geometry_size_scales)
        roll_degrees = parse_float_list(args.geometry_roll_degrees)

    best: tuple[float, np.ndarray, dict] | None = None
    tried = 0
    old_depth_maps = getattr(args, "_active_depth_maps", None)
    args._active_depth_maps = depth_maps
    try:
        for y in ys:
            for x in xs:
                xi = int(round(x))
                yi = int(round(y))
                depth = float(first_depth[yi, xi])
                if depth <= args.depth_min_m or depth >= args.depth_max_m:
                    continue
                axes = estimate_surface_axes(first_depth, np.asarray([x, y], dtype=np.float64), intrinsics, proj, args)
                if axes is None:
                    continue
                axis_u, axis_v, normal = axes
                center_cam = unproject_tensor_xy(
                    np.asarray([x, y], dtype=np.float64),
                    np.asarray(depth, dtype=np.float64),
                    intrinsics,
                    proj,
                )
                for scale in size_scales:
                    for roll in roll_degrees:
                        tried += 1
                        rolled_u, rolled_v = rotate_axes(axis_u, axis_v, roll)
                        patch_world = build_plane_world_from_camera_axes(
                            c2w[0],
                            center_cam,
                            rolled_u,
                            rolled_v,
                            args.plane_width * scale,
                            args.plane_height * scale,
                        )
                        _, _, geom_meta = build_geometry_arrays_for_plane(
                            patch_world,
                            c2w,
                            image_paths,
                            tensor_hw,
                            texture_size,
                            intrinsics,
                            args,
                        )
                        score = geometry_score(geom_meta)
                        if best is None or score > best[0]:
                            placement = {
                                "placement_mode": "auto_depth_surface",
                                "score": score,
                                "candidate_count": tried,
                                "center_tensor_xy": [float(x), float(y)],
                                "center_depth_m": depth,
                                "center_first_camera": center_cam.astype(float).tolist(),
                                "surface_normal_first_camera": normal.astype(float).tolist(),
                                "width_m": args.plane_width * scale,
                                "height_m": args.plane_height * scale,
                                "size_scale": float(scale),
                                "roll_degrees": float(roll),
                                "optimize_geometry": bool(args.optimize_geometry),
                            }
                            geom_meta.update(placement)
                            best = (score, patch_world, geom_meta)
    finally:
        args._active_depth_maps = old_depth_maps

    if best is None:
        patch_world, meta = build_fixed_patch_plane_world(c2w[0], args)
        meta["placement_mode"] = "fixed_fallback_no_valid_surface"
        meta["candidate_count"] = tried
        return patch_world, meta
    best[2]["candidate_count"] = tried
    return best[1], best[2]


def choose_patch_plane_world(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
) -> tuple[np.ndarray, dict]:
    if args.plane_mode == "auto_depth_surface":
        return choose_auto_depth_surface_plane(c2w, image_paths, tensor_hw, texture_size, intrinsics, args, depth_maps)
    return build_fixed_patch_plane_world(c2w[0], args)


def build_geometry_grids(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    depth_maps: list[np.ndarray | None],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    patch_world, placement_meta = choose_patch_plane_world(
        c2w,
        image_paths,
        tensor_hw,
        texture_size,
        intrinsics,
        args,
        depth_maps,
    )
    old_depth_maps = getattr(args, "_active_depth_maps", None)
    args._active_depth_maps = depth_maps
    try:
        grids, masks, meta = build_geometry_arrays_for_plane(
            patch_world,
            c2w,
            image_paths,
            tensor_hw,
            texture_size,
            intrinsics,
            args,
        )
    finally:
        args._active_depth_maps = old_depth_maps
    meta.update(placement_meta)
    grids_t = torch.from_numpy(grids).to(device)
    masks_t = torch.from_numpy(masks).to(device)
    return grids_t, masks_t, meta


def prepare_texture_for_render(texture: torch.Tensor, args: argparse.Namespace | None, training: bool) -> torch.Tensor:
    if args is None:
        return texture.clamp(0.0, 1.0)

    rendered = texture.clamp(args.print_min, args.print_max)
    if not training or not args.physical_eot:
        return rendered

    if args.eot_brightness > 0:
        factor = torch.empty((), device=texture.device).uniform_(1.0 - args.eot_brightness, 1.0 + args.eot_brightness)
        rendered = rendered * factor
    if args.eot_contrast > 0:
        factor = torch.empty((), device=texture.device).uniform_(1.0 - args.eot_contrast, 1.0 + args.eot_contrast)
        mean = rendered.mean(dim=(-2, -1), keepdim=True)
        rendered = (rendered - mean) * factor + mean
    if args.eot_gamma > 0:
        gamma = torch.empty((), device=texture.device).uniform_(1.0 - args.eot_gamma, 1.0 + args.eot_gamma)
        rendered = rendered.clamp(1e-5, 1.0).pow(gamma)
    return rendered.clamp(args.print_min, args.print_max)


def apply_geometry_patch(
    images: torch.Tensor,
    texture: torch.Tensor,
    grids: torch.Tensor,
    masks: torch.Tensor,
    args: argparse.Namespace | None = None,
    training: bool = False,
) -> torch.Tensor:
    n_frames = images.shape[0]
    rendered_texture = prepare_texture_for_render(texture, args, training)
    sampled = F.grid_sample(
        rendered_texture.expand(n_frames, -1, -1, -1),
        grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    patched = (images * (1.0 - masks) + sampled * masks).clamp(0.0, 1.0)
    if args is not None and training and args.physical_eot and args.eot_noise_std > 0:
        patched = (patched + torch.randn_like(patched) * args.eot_noise_std).clamp(0.0, 1.0)
    return patched


def load_tum_sequence(
    seq_dir: Path,
    frame_indices: list[int],
    gt_name: str,
    intrinsics: np.ndarray,
    texture_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    all_images = find_images(seq_dir)
    image_paths = [all_images[int(idx)] for idx in frame_indices]
    images = load_and_preprocess_images(image_paths).to(device)
    tensor_hw = tuple(int(v) for v in images.shape[-2:])
    c2w = tum_rows_to_c2w(read_tum_rows(seq_dir / gt_name), frame_indices)
    depth_maps = load_depth_maps(seq_dir, image_paths, frame_indices, tensor_hw, args)
    grids, masks, geom_meta = build_geometry_grids(
        c2w,
        image_paths,
        tensor_hw,
        texture_size,
        intrinsics,
        args,
        device,
        depth_maps,
    )
    return {
        "seq": seq_dir.name,
        "seq_dir": seq_dir,
        "images": images,
        "image_paths": image_paths,
        "image_names": [Path(path).name for path in image_paths],
        "frame_indices": frame_indices,
        "c2w_gt": c2w,
        "depth_available": [depth is not None for depth in depth_maps],
        "grids": grids,
        "masks": masks,
        "geometry": geom_meta,
        "tensor_hw": tensor_hw,
    }


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_geometry_patch(
    model: torch.nn.Module,
    scene_dirs: list[Path],
    manifest: dict[str, list[int]],
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> tuple[torch.Tensor, dict]:
    rng = np.random.default_rng(args.seed)
    texture = torch.rand(
        (1, 3, args.texture_size, args.texture_size),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    optimizer = torch.optim.AdamW([texture], lr=args.patch_lr)

    patch_dir = output_dir / "geometry_patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    history_path = patch_dir / "training_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    total_updates = args.iterations * args.inner_loop
    warmup_updates = args.warmup_iterations * args.inner_loop
    update_idx = 0
    started = time.time()
    last_loss = None

    with history_path.open("a", encoding="utf-8") as history_file:
        for iteration in range(args.iterations):
            scene_indices = rng.choice(
                len(scene_dirs),
                size=min(args.scenes_per_iteration, len(scene_dirs)),
                replace=False,
            )
            batch = []
            for idx in scene_indices:
                seq_dir = scene_dirs[int(idx)]
                item = load_tum_sequence(
                    seq_dir,
                    manifest[seq_dir.name],
                    args.gt_name,
                    intrinsics,
                    args.texture_size,
                    args,
                    device,
                )
                with torch.no_grad():
                    item["clean_features"] = [
                        feature.detach()
                        for feature in extract_features(model, item["images"], dtype, args.feature_layer)
                    ]
                batch.append(item)

            for inner_step in range(args.inner_loop):
                optimizer.zero_grad(set_to_none=True)
                current_lr = scheduled_lr(
                    args.patch_lr,
                    update_idx,
                    total_updates,
                    warmup_updates,
                    args.scheduler,
                )
                set_optimizer_lr(optimizer, current_lr)
                losses = []
                coverages = []
                for item in batch:
                    adv_images = apply_geometry_patch(
                        item["images"],
                        texture,
                        item["grids"],
                        item["masks"],
                        args,
                        training=True,
                    )
                    adv_features = extract_features(
                        model,
                        adv_images,
                        dtype,
                        args.feature_layer,
                        args.activation_checkpoint,
                    )
                    loss, terms = feature_l1_loss(adv_features, item["clean_features"])
                    (-loss / len(batch)).backward()
                    losses.append(terms["feature_l1"])
                    coverages.append(item["geometry"]["mask_coverage_mean"])

                if texture.grad is None:
                    raise RuntimeError("Geometry patch gradient is None.")
                optimizer.step()
                with torch.no_grad():
                    texture.clamp_(args.print_min, args.print_max)

                update_idx += 1
                last_loss = float(np.mean(losses))
                record = {
                    "iteration": iteration + 1,
                    "inner_step": inner_step + 1,
                    "update": update_idx,
                    "lr": current_lr,
                    "feature_l1": last_loss,
                    "mask_coverage_mean": float(np.mean(coverages)),
                    "scenes": [item["seq"] for item in batch],
                }
                history_file.write(json.dumps(record) + "\n")
                if update_idx == 1 or update_idx % args.log_every == 0 or update_idx == total_updates:
                    print(
                        f"[train-geometry] update {update_idx:06d}/{total_updates:06d} "
                        f"scenes={len(batch)} feature_l1={last_loss:.6f} "
                        f"coverage={record['mask_coverage_mean']:.6f}"
                    )

            del batch

    texture_npz = patch_dir / "geometry_patch_texture.npz"
    np.savez_compressed(texture_npz, texture=texture.detach().float().cpu().numpy())
    to_pil_image(texture.squeeze(0).detach().float().cpu().clamp(0, 1)).save(
        patch_dir / "geometry_patch_texture.png"
    )
    metadata = {
        "mode": "gt_geometry_aware_planar_patch",
        "attack_target": "feature_l1_clean_vs_adversarial",
        "feature_layer": args.feature_layer,
        "texture_shape": list(texture.shape),
        "patch_lr": args.patch_lr,
        "scheduler": args.scheduler,
        "warmup_iterations": args.warmup_iterations,
        "iterations": args.iterations,
        "inner_loop": args.inner_loop,
        "scenes_per_iteration": args.scenes_per_iteration,
        "total_updates": total_updates,
        "elapsed_seconds": time.time() - started,
        "last_logged_feature_l1": last_loss,
        "depth_visibility": {
            "enabled": bool(args.use_depth_visibility),
            "depth_txt_name": args.depth_txt_name,
            "depth_scale": args.depth_scale,
            "visibility_depth_margin": args.visibility_depth_margin,
        },
        "physical_eot": {
            "enabled": bool(args.physical_eot),
            "print_min": args.print_min,
            "print_max": args.print_max,
            "brightness": args.eot_brightness,
            "contrast": args.eot_contrast,
            "gamma": args.eot_gamma,
            "noise_std": args.eot_noise_std,
        },
        "plane": {
            "mode": args.plane_mode,
            "center_first_camera": [args.plane_center_x, args.plane_center_y, args.plane_distance],
            "width_m": args.plane_width,
            "height_m": args.plane_height,
            "surface_candidate_grid": args.surface_candidate_grid,
            "surface_search_margin": args.surface_search_margin,
            "optimize_geometry": bool(args.optimize_geometry),
            "geometry_size_scales": args.geometry_size_scales,
            "geometry_roll_degrees": args.geometry_roll_degrees,
        },
        "intrinsics": intrinsics.astype(float).tolist(),
        "frame_manifest": args.frame_manifest,
        "gt_name": args.gt_name,
        "texture_path": str(texture_npz),
    }
    with (patch_dir / "geometry_patch_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train done] saved geometry patch -> {texture_npz}")
    return texture.detach(), metadata


def load_texture(path: str, device: torch.device) -> torch.Tensor:
    with np.load(path) as data:
        texture = torch.from_numpy(np.asarray(data["texture"]).astype(np.float32))
    return texture.to(device)


def evaluate_geometry_patch(
    model: torch.nn.Module,
    texture: torch.Tensor,
    scene_dirs: list[Path],
    manifest: dict[str, list[int]],
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
    patch_metadata: dict,
) -> list[dict]:
    summaries = []
    for seq_dir in scene_dirs:
        out_dir = output_dir / seq_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_existing_outputs and (out_dir / "vggt_outputs.npz").exists() and (out_dir / "attack_summary.json").exists():
            continue

        item = load_tum_sequence(
            seq_dir,
            manifest[seq_dir.name],
            args.gt_name,
            intrinsics,
            int(texture.shape[-1]),
            args,
            device,
        )
        adv_images = apply_geometry_patch(item["images"], texture, item["grids"], item["masks"], args, training=False).detach()
        with torch.no_grad():
            clean_features = extract_features(model, item["images"], dtype, args.feature_layer)
            adv_features = extract_features(model, adv_images, dtype, args.feature_layer)
            final_loss, _ = feature_l1_loss(adv_features, clean_features)
            preds = detach_predictions(forward_vggt(model, adv_images, dtype))

        save_official_style_npz(out_dir / "vggt_outputs.npz", preds, item["image_names"], item["tensor_hw"])
        summary = {
            "scene": str(seq_dir),
            "dataset": "tum-dynamics-10frame-geometry-aware",
            "mode": "gt_geometry_aware_planar_patch",
            "n_frames": len(item["image_paths"]),
            "image_paths": [str(path) for path in item["image_paths"]],
            "frame_indices": [int(idx) for idx in item["frame_indices"]],
            "frame_manifest": args.frame_manifest,
            "ckpt": args.ckpt,
            "feature_layer": args.feature_layer,
            "final_feature_l1": float(final_loss.detach().cpu()),
            "geometry": item["geometry"],
            "depth_available": item["depth_available"],
            "geometry_patch_metadata": patch_metadata,
            "outputs": {"attacked_vggt_outputs": "vggt_outputs.npz"},
        }
        with (out_dir / "attack_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        summaries.append(summary)
        print(
            f"[apply-geometry] {seq_dir.name}: feature_l1={summary['final_feature_l1']:.6f} "
            f"coverage={item['geometry']['mask_coverage_mean']:.6f} -> {out_dir / 'vggt_outputs.npz'}"
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum_root", required=True)
    parser.add_argument("--scene_pattern", default="rgbd_dataset_freiburg3_*")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frame_manifest", required=True)
    parser.add_argument("--ckpt", default="facebook/VGGT-1B")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--gt_name", default="groundtruth_90.txt")
    parser.add_argument("--texture_path", default=None, help="Reuse an existing geometry_patch_texture.npz")
    parser.add_argument("--texture_size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--inner_loop", type=int, default=10)
    parser.add_argument("--scenes_per_iteration", type=int, default=1)
    parser.add_argument("--patch_lr", type=float, default=0.001)
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--warmup_iterations", type=int, default=20)
    parser.add_argument("--feature_layer", default="aggregator_final")
    parser.add_argument("--activation_checkpoint", action="store_true")
    parser.add_argument("--skip_existing_outputs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--fx", type=float, default=535.4)
    parser.add_argument("--fy", type=float, default=539.2)
    parser.add_argument("--cx", type=float, default=320.1)
    parser.add_argument("--cy", type=float, default=247.6)
    parser.add_argument("--plane_mode", choices=("fixed", "auto_depth_surface"), default="fixed")
    parser.add_argument("--plane_width", type=float, default=0.6)
    parser.add_argument("--plane_height", type=float, default=0.6)
    parser.add_argument("--plane_distance", type=float, default=2.0)
    parser.add_argument("--plane_center_x", type=float, default=0.0)
    parser.add_argument("--plane_center_y", type=float, default=0.0)
    parser.add_argument("--use_depth_visibility", action="store_true")
    parser.add_argument("--depth_txt_name", default="depth.txt")
    parser.add_argument("--depth_scale", type=float, default=5000.0)
    parser.add_argument("--depth_match_max_dt", type=float, default=0.05)
    parser.add_argument("--depth_min_m", type=float, default=0.2)
    parser.add_argument("--depth_max_m", type=float, default=8.0)
    parser.add_argument("--visibility_depth_margin", type=float, default=0.05)
    parser.add_argument("--surface_candidate_grid", type=int, default=5)
    parser.add_argument("--surface_search_margin", type=float, default=0.18)
    parser.add_argument("--normal_estimation_radius", type=int, default=5)
    parser.add_argument("--optimize_geometry", action="store_true")
    parser.add_argument("--geometry_size_scales", default="0.8,1.0,1.2")
    parser.add_argument("--geometry_roll_degrees", default="-15,0,15")
    parser.add_argument("--physical_eot", action="store_true")
    parser.add_argument("--print_min", type=float, default=0.0)
    parser.add_argument("--print_max", type=float, default=1.0)
    parser.add_argument("--eot_brightness", type=float, default=0.15)
    parser.add_argument("--eot_contrast", type=float, default=0.15)
    parser.add_argument("--eot_gamma", type=float, default=0.10)
    parser.add_argument("--eot_noise_std", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seeds(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_frame_manifest(args.frame_manifest)
    tum_root = Path(args.tum_root)
    scene_dirs = [path for path in list_scene_dirs(tum_root, args.scene_pattern) if path.name in manifest]
    if not scene_dirs:
        raise ValueError("No TUM scenes have entries in the frame manifest.")

    intrinsics = np.asarray(
        [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(
        f"[cfg] device={device} dtype={dtype} scenes={len(scene_dirs)} "
        f"texture_size={args.texture_size} iterations={args.iterations} inner_loop={args.inner_loop} "
        f"plane_mode={args.plane_mode} depth_visibility={args.use_depth_visibility} "
        f"physical_eot={args.physical_eot}"
    )

    model = load_model(args, device)
    for param in model.parameters():
        param.requires_grad_(False)

    if args.texture_path:
        texture = load_texture(args.texture_path, device)
        patch_metadata = {
            "mode": "gt_geometry_aware_planar_patch",
            "loaded_texture_path": str(Path(args.texture_path)),
            "frame_manifest": args.frame_manifest,
            "intrinsics": intrinsics.astype(float).tolist(),
            "plane_mode": args.plane_mode,
            "use_depth_visibility": bool(args.use_depth_visibility),
            "physical_eot": bool(args.physical_eot),
        }
        print(f"[patch] loaded geometry texture -> {args.texture_path}")
    else:
        texture, patch_metadata = train_geometry_patch(
            model,
            scene_dirs,
            manifest,
            intrinsics,
            args,
            device,
            dtype,
            output_dir,
        )

    summaries = evaluate_geometry_patch(
        model,
        texture,
        scene_dirs,
        manifest,
        intrinsics,
        args,
        device,
        dtype,
        output_dir,
        patch_metadata,
    )
    with (output_dir / "geometry_attack_batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"[done] generated {len(summaries)} geometry-aware TUM-10 outputs in {output_dir}")


if __name__ == "__main__":
    main()
