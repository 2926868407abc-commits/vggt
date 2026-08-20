"""GT-geometry-aware planar patch attack for VGGT on TUM-10.

This is a GT-geometry-consistent physical patch pipeline:

* one learnable texture is shared by all frames and sequences
* for each TUM sequence, a 3D planar patch is either placed in front of the
  first selected camera or automatically anchored on a depth-observed surface
* the same 3D plane is projected into all selected frames with GT poses and RGB
  intrinsics, then differentiably sampled into the VGGT input images
* optional TUM depth visibility acts as a z-buffer for occlusion
* optional geometry search chooses patch position, surface orientation, and size
  by visible cross-view coverage, or by a natural-sticker coverage range, before
  texture optimization
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
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from attack_vggt_new1 import (
    autocast_context,
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
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


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


def tensor_projection_params(tensor_hw: tuple[int, int]) -> dict[str, float]:
    tensor_h, tensor_w = tensor_hw
    return {
        "orig_w": float(tensor_w),
        "orig_h": float(tensor_h),
        "scale_x": 1.0,
        "scale_y": 1.0,
        "crop_y": 0.0,
        "tensor_w": float(tensor_w),
        "tensor_h": float(tensor_h),
    }


def projection_params_for_frame(args: argparse.Namespace, image_path: str, tensor_hw: tuple[int, int]) -> dict[str, float]:
    if getattr(args, "_intrinsics_in_tensor_space", False):
        return tensor_projection_params(tensor_hw)
    return preprocess_projection_params(image_path, tensor_hw)


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


def parse_manual_quad_xy(value: str, tensor_hw: tuple[int, int], coordinates: str) -> np.ndarray:
    parts = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    if len(parts) != 8:
        raise ValueError(
            "--manual_quad_xy expects 8 numbers: x0,y0,x1,y1,x2,y2,x3,y3 "
            "ordered top-left, top-right, bottom-right, bottom-left."
        )
    quad = np.asarray([float(item) for item in parts], dtype=np.float64).reshape(4, 2)
    tensor_h, tensor_w = tensor_hw
    if coordinates == "normalized":
        quad[:, 0] *= tensor_w - 1
        quad[:, 1] *= tensor_h - 1
    elif coordinates != "pixel":
        raise ValueError(f"Unsupported manual_quad_coordinates={coordinates!r}")
    quad[:, 0] = np.clip(quad[:, 0], 0.0, tensor_w - 1.0)
    quad[:, 1] = np.clip(quad[:, 1], 0.0, tensor_h - 1.0)
    return quad


def manual_quad_mask(quad_xy: np.ndarray, tensor_hw: tuple[int, int]) -> np.ndarray:
    tensor_h, tensor_w = tensor_hw
    mask_img = Image.new("L", (tensor_w, tensor_h), 0)
    polygon = [(float(x), float(y)) for x, y in quad_xy]
    ImageDraw.Draw(mask_img).polygon(polygon, fill=1)
    return np.asarray(mask_img, dtype=bool)


def shrink_quad_xy(quad_xy: np.ndarray, ratio: float) -> np.ndarray:
    ratio = float(np.clip(ratio, 0.05, 1.0))
    center = quad_xy.mean(axis=0, keepdims=True)
    return center + (quad_xy - center) * ratio


def fit_camera_plane_pca(points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    center_cam = points_cam.mean(axis=0)
    centered = points_cam - center_cam[None, :]
    cov = centered.T @ centered / max(1, points_cam.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    normal_cam = eigvecs[:, order[0]]
    if np.dot(normal_cam, -center_cam) < 0:
        normal_cam = -normal_cam
    residual = float(eigvals[order[0]] / max(float(eigvals.sum()), 1e-12))
    return center_cam, normal_cam, eigvals[order], residual


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
    if not args.use_depth_visibility and args.plane_mode not in (
        "auto_depth_surface",
        "fused_depth_surface",
        "depth_manual_anchor_surface",
        "depth_manual_quad_surface",
    ):
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


def build_plane_world_from_world_axes(
    center_world: np.ndarray,
    axis_u_world: np.ndarray,
    axis_v_world: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    axis_u_world = axis_u_world / np.linalg.norm(axis_u_world)
    axis_v_world = axis_v_world / np.linalg.norm(axis_v_world)
    return np.asarray(
        [
            center_world - axis_u_world * width / 2 - axis_v_world * height / 2,
            center_world + axis_u_world * width / 2 - axis_v_world * height / 2,
            center_world + axis_u_world * width / 2 + axis_v_world * height / 2,
            center_world - axis_u_world * width / 2 + axis_v_world * height / 2,
        ],
        dtype=np.float64,
    )


def depth_map_to_world_points(
    depth_map: np.ndarray | None,
    c2w: np.ndarray,
    image_path: str,
    tensor_hw: tuple[int, int],
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    stride: int,
) -> np.ndarray:
    if depth_map is None:
        return np.empty((0, 3), dtype=np.float64)
    tensor_h, tensor_w = tensor_hw
    ys, xs = np.meshgrid(
        np.arange(0, tensor_h, max(1, stride), dtype=np.float64),
        np.arange(0, tensor_w, max(1, stride), dtype=np.float64),
        indexing="ij",
    )
    xy = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)
    depths = depth_map[ys.astype(int).reshape(-1), xs.astype(int).reshape(-1)].astype(np.float64)
    valid = (depths > args.depth_min_m) & (depths < args.depth_max_m)
    if not bool(valid.any()):
        return np.empty((0, 3), dtype=np.float64)
    proj = projection_params_for_frame(args, image_path, tensor_hw)
    points_cam = unproject_tensor_xy(xy[valid], depths[valid], intrinsics, proj)
    points_h = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float64)], axis=1)
    return (c2w @ points_h.T).T[:, :3]


def fused_depth_point_cloud(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
) -> np.ndarray:
    clouds = [
        depth_map_to_world_points(depth, pose, image_path, tensor_hw, intrinsics, args, args.fused_point_stride)
        for depth, pose, image_path in zip(depth_maps, c2w, image_paths)
    ]
    clouds = [cloud for cloud in clouds if cloud.size > 0]
    if not clouds:
        return np.empty((0, 3), dtype=np.float64)
    points = np.concatenate(clouds, axis=0)
    if points.shape[0] > args.fused_max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(points.shape[0], size=args.fused_max_points, replace=False)
        points = points[keep]
    return points


def extrinsics_to_c2w(extrinsics: np.ndarray) -> np.ndarray:
    c2w_list = []
    for extrinsic in extrinsics:
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = np.asarray(extrinsic, dtype=np.float64)
        c2w_list.append(np.linalg.inv(w2c))
    return np.stack(c2w_list, axis=0)


def load_clean_vggt_geometry(
    seq_name: str,
    tensor_hw: tuple[int, int],
    args: argparse.Namespace,
) -> dict | None:
    if not args.clean_vggt_output_root:
        return None
    npz_path = Path(args.clean_vggt_output_root) / seq_name / "vggt_outputs.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path) as data:
        if "extrinsic" not in data or "intrinsic" not in data:
            return None
        c2w = extrinsics_to_c2w(np.asarray(data["extrinsic"]))
        intrinsics = np.asarray(data["intrinsic"], dtype=np.float64)
        if intrinsics.ndim == 2:
            intrinsics = np.repeat(intrinsics[None], c2w.shape[0], axis=0)

        point_keys = ("point_map", "point_cloud_unproj")
        point_map = None
        for key in point_keys:
            if key in data:
                point_map = np.asarray(data[key], dtype=np.float64)
                break
        if point_map is None:
            return None

        if point_map.ndim == 4 and point_map.shape[-1] == 3:
            point_map_grid = point_map.copy()
            points = point_map.reshape(-1, 3)
        else:
            return None
        valid = np.isfinite(points).all(axis=1)
        valid_grid = valid.reshape(point_map_grid.shape[:-1])

        if "point_conf" in data:
            conf = np.asarray(data["point_conf"]).reshape(-1)
            valid &= np.isfinite(conf)
            if args.vggt_point_conf_percentile > 0 and valid.any():
                threshold = np.percentile(conf[valid], args.vggt_point_conf_percentile)
                valid &= conf >= threshold
            valid_grid = valid.reshape(point_map_grid.shape[:-1])

        points = points[valid]
        if points.shape[0] > args.fused_max_points:
            rng = np.random.default_rng(args.seed)
            keep = rng.choice(points.shape[0], size=args.fused_max_points, replace=False)
            points = points[keep]

        depth_maps = None
        if "depth" in data:
            depth = np.asarray(data["depth"], dtype=np.float32)
            if depth.ndim == 4 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim == 3:
                depth_maps = [depth[idx] for idx in range(depth.shape[0])]

    tensor_h, tensor_w = tensor_hw
    if intrinsics.shape[1:] != (3, 3):
        return None
    if c2w.shape[0] != intrinsics.shape[0]:
        return None
    return {
        "c2w": c2w,
        "intrinsics": intrinsics,
        "points": points,
        "point_map_grid": point_map_grid,
        "point_valid_grid": valid_grid,
        "depth_maps": depth_maps,
        "source": str(npz_path),
        "tensor_hw": [tensor_h, tensor_w],
    }


def estimate_world_surface_axes(
    points: np.ndarray,
    center_world: np.ndarray,
    first_c2w: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None:
    if points.shape[0] < 6:
        return None
    diff = points - center_world[None, :]
    distances = np.linalg.norm(diff, axis=1)
    radius_mask = distances <= args.fused_normal_radius
    if int(radius_mask.sum()) < args.fused_min_neighbors:
        neighbor_count = min(max(args.fused_min_neighbors, 6), points.shape[0])
        nearest = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
        neighbors = points[nearest]
    else:
        neighbors = points[radius_mask]
        if neighbors.shape[0] > args.fused_max_neighbors:
            nearest = np.argsort(np.linalg.norm(neighbors - center_world[None, :], axis=1))[: args.fused_max_neighbors]
            neighbors = neighbors[nearest]

    centered = neighbors - neighbors.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, neighbors.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    normal = eigvecs[:, order[0]]
    axis_u = eigvecs[:, order[2]]
    axis_v = eigvecs[:, order[1]]

    camera_center = first_c2w[:3, 3]
    if np.dot(normal, camera_center - center_world) < 0:
        normal = -normal
        axis_v = -axis_v
    if np.linalg.norm(np.cross(axis_u, axis_v)) < 1e-8:
        return None
    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)
    residual = float(eigvals[order[0]] / max(float(eigvals.sum()), 1e-12))
    return axis_u, axis_v, normal, {
        "neighbor_count": int(neighbors.shape[0]),
        "pca_eigenvalues": eigvals[order].astype(float).tolist(),
        "plane_residual_ratio": residual,
    }


def surface_candidate_first_camera_meta(
    center_first_camera: np.ndarray,
    normal_first_camera: np.ndarray,
    args: argparse.Namespace,
) -> tuple[bool, dict]:
    center_first_camera = np.asarray(center_first_camera, dtype=np.float64)
    normal_first_camera = np.asarray(normal_first_camera, dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal_first_camera))
    if normal_norm < 1e-8 or not np.all(np.isfinite(center_first_camera)):
        return False, {}
    normal_first_camera = normal_first_camera / normal_norm
    depth = float(center_first_camera[2])

    meta = {
        "surface_center_depth_first_camera": depth,
        "surface_normal_first_camera": normal_first_camera.astype(float).tolist(),
        "surface_orientation_filter": args.surface_orientation_filter,
        "surface_min_center_depth": args.surface_min_center_depth,
        "surface_max_center_depth": args.surface_max_center_depth,
        "surface_max_tilt_degrees": args.surface_max_tilt_degrees,
    }

    if args.surface_min_center_depth > 0 and depth < args.surface_min_center_depth:
        return False, meta
    if args.surface_max_center_depth > 0 and depth > args.surface_max_center_depth:
        return False, meta

    if args.surface_orientation_filter == "none":
        return True, meta

    min_cos = math.cos(math.radians(args.surface_max_tilt_degrees))
    fronto = abs(float(normal_first_camera[2])) >= min_cos
    tabletop = abs(float(normal_first_camera[1])) >= min_cos
    side = abs(float(normal_first_camera[0])) >= min_cos
    meta.update(
        {
            "surface_is_fronto": fronto,
            "surface_is_tabletop": tabletop,
            "surface_is_side": side,
        }
    )

    if args.surface_orientation_filter == "fronto":
        return fronto, meta
    if args.surface_orientation_filter == "tabletop":
        return tabletop, meta
    if args.surface_orientation_filter == "side":
        return side, meta
    if args.surface_orientation_filter == "fronto_or_tabletop":
        return fronto or tabletop, meta
    if args.surface_orientation_filter == "axis_aligned":
        return fronto or tabletop or side, meta
    raise ValueError(f"Unknown surface_orientation_filter: {args.surface_orientation_filter}")


def world_surface_candidate_first_camera_meta(
    center_world: np.ndarray,
    normal_world: np.ndarray,
    first_c2w: np.ndarray,
    args: argparse.Namespace,
) -> tuple[bool, dict]:
    world_to_first = np.linalg.inv(first_c2w)
    center_h = world_to_first @ np.asarray(
        [center_world[0], center_world[1], center_world[2], 1.0],
        dtype=np.float64,
    )
    normal_first_camera = first_c2w[:3, :3].T @ np.asarray(normal_world, dtype=np.float64)
    return surface_candidate_first_camera_meta(center_h[:3], normal_first_camera, args)


def store_surface_candidates(args: argparse.Namespace, candidates: list[tuple[float, np.ndarray, dict]]) -> None:
    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    keep = max(1, int(args.surface_strength_candidates))
    args._last_patch_plane_candidates = [
        (float(score), patch_world.copy(), copy.deepcopy(meta))
        for score, patch_world, meta in candidates[:keep]
    ]


def choose_fused_depth_surface_plane(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
) -> tuple[np.ndarray, dict]:
    points = fused_depth_point_cloud(c2w, image_paths, tensor_hw, intrinsics, args, depth_maps)
    if points.shape[0] == 0:
        patch_world, meta = build_fixed_patch_plane_world(c2w[0], args)
        meta["placement_mode"] = "fixed_fallback_no_fused_points"
        return patch_world, meta

    rng = np.random.default_rng(args.seed)
    candidate_count = min(args.fused_surface_candidates, points.shape[0])
    candidate_ids = rng.choice(points.shape[0], size=candidate_count, replace=False)

    size_scales = [1.0]
    roll_degrees = [0.0]
    if args.optimize_geometry:
        size_scales = parse_float_list(args.geometry_size_scales)
        roll_degrees = parse_float_list(args.geometry_roll_degrees)

    best: tuple[float, np.ndarray, dict] | None = None
    candidates: list[tuple[float, np.ndarray, dict]] = []
    tried = 0
    old_depth_maps = getattr(args, "_active_depth_maps", None)
    args._active_depth_maps = depth_maps
    try:
        for point_id in candidate_ids:
            center_world = points[int(point_id)]
            axes = estimate_world_surface_axes(points, center_world, c2w[0], args)
            if axes is None:
                continue
            axis_u, axis_v, normal, pca_meta = axes
            if pca_meta["plane_residual_ratio"] > args.fused_max_plane_residual:
                continue
            candidate_ok, candidate_meta = world_surface_candidate_first_camera_meta(
                center_world, normal, c2w[0], args
            )
            if not candidate_ok:
                continue
            for scale in size_scales:
                for roll in roll_degrees:
                    tried += 1
                    rolled_u, rolled_v = rotate_axes(axis_u, axis_v, roll)
                    patch_world = build_plane_world_from_world_axes(
                        center_world,
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
                    score = geometry_score(geom_meta, args)
                    if not np.isfinite(score):
                        continue
                    center_first_cam_h = np.linalg.inv(c2w[0]) @ np.asarray(
                        [center_world[0], center_world[1], center_world[2], 1.0],
                        dtype=np.float64,
                    )
                    placement = {
                        "placement_mode": "fused_depth_surface",
                        "score": score,
                        "candidate_count": int(candidate_count),
                        "tried_geometry_count": tried,
                        "fused_point_count": int(points.shape[0]),
                        "center_world": center_world.astype(float).tolist(),
                        "center_first_camera": center_first_cam_h[:3].astype(float).tolist(),
                        "surface_normal_world": normal.astype(float).tolist(),
                        "width_m": args.plane_width * scale,
                        "height_m": args.plane_height * scale,
                        "size_scale": float(scale),
                        "roll_degrees": float(roll),
                        "optimize_geometry": bool(args.optimize_geometry),
                        "fused_point_stride": int(args.fused_point_stride),
                        "fused_normal_radius": float(args.fused_normal_radius),
                        **candidate_meta,
                        **pca_meta,
                    }
                    candidate_meta_full = copy.deepcopy(geom_meta)
                    candidate_meta_full.update(placement)
                    candidates.append((score, patch_world.copy(), candidate_meta_full))
                    if best is None or score > best[0]:
                        best = (score, patch_world, candidate_meta_full)
    finally:
        args._active_depth_maps = old_depth_maps

    if best is None:
        store_surface_candidates(args, [])
        if args.surface_score_mode == "natural":
            raise RuntimeError(
                "No fused-depth surface satisfied the natural sticker constraints. "
                "Relax surface_coverage_min/max or surface_min_visible_frames."
            )
        patch_world, meta = build_fixed_patch_plane_world(c2w[0], args)
        meta["placement_mode"] = "fixed_fallback_no_valid_fused_surface"
        meta["fused_point_count"] = int(points.shape[0])
        meta["candidate_count"] = int(candidate_count)
        meta["tried_geometry_count"] = tried
        return patch_world, meta
    best[2]["tried_geometry_count"] = tried
    store_surface_candidates(args, candidates)
    return best[1], best[2]


def choose_vggt_pointmap_surface_plane(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
) -> tuple[np.ndarray, dict]:
    points = getattr(args, "_active_vggt_points", None)
    source = getattr(args, "_active_vggt_geometry_source", None)
    if points is None or points.shape[0] == 0:
        patch_world, meta = build_fixed_patch_plane_world(c2w[0], args)
        meta["placement_mode"] = "fixed_fallback_no_vggt_pointmap"
        return patch_world, meta

    rng = np.random.default_rng(args.seed)
    candidate_count = min(args.fused_surface_candidates, points.shape[0])
    candidate_ids = rng.choice(points.shape[0], size=candidate_count, replace=False)

    size_scales = [1.0]
    roll_degrees = [0.0]
    if args.optimize_geometry:
        size_scales = parse_float_list(args.geometry_size_scales)
        roll_degrees = parse_float_list(args.geometry_roll_degrees)

    best: tuple[float, np.ndarray, dict] | None = None
    candidates: list[tuple[float, np.ndarray, dict]] = []
    tried = 0
    old_depth_maps = getattr(args, "_active_depth_maps", None)
    args._active_depth_maps = depth_maps
    try:
        for point_id in candidate_ids:
            center_world = points[int(point_id)]
            axes = estimate_world_surface_axes(points, center_world, c2w[0], args)
            if axes is None:
                continue
            axis_u, axis_v, normal, pca_meta = axes
            if pca_meta["plane_residual_ratio"] > args.fused_max_plane_residual:
                continue
            candidate_ok, candidate_meta = world_surface_candidate_first_camera_meta(
                center_world, normal, c2w[0], args
            )
            if not candidate_ok:
                continue
            for scale in size_scales:
                for roll in roll_degrees:
                    tried += 1
                    rolled_u, rolled_v = rotate_axes(axis_u, axis_v, roll)
                    patch_world = build_plane_world_from_world_axes(
                        center_world,
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
                    score = geometry_score(geom_meta, args)
                    if not np.isfinite(score):
                        continue
                    center_first_cam_h = np.linalg.inv(c2w[0]) @ np.asarray(
                        [center_world[0], center_world[1], center_world[2], 1.0],
                        dtype=np.float64,
                    )
                    placement = {
                        "placement_mode": "vggt_pointmap_surface",
                        "geometry_source": "clean_vggt_pointmap",
                        "clean_vggt_output": source,
                        "score": score,
                        "candidate_count": int(candidate_count),
                        "tried_geometry_count": tried,
                        "vggt_point_count": int(points.shape[0]),
                        "center_world": center_world.astype(float).tolist(),
                        "center_first_camera": center_first_cam_h[:3].astype(float).tolist(),
                        "surface_normal_world": normal.astype(float).tolist(),
                        "width_m": args.plane_width * scale,
                        "height_m": args.plane_height * scale,
                        "size_scale": float(scale),
                        "roll_degrees": float(roll),
                        "optimize_geometry": bool(args.optimize_geometry),
                        "vggt_point_conf_percentile": float(args.vggt_point_conf_percentile),
                        **candidate_meta,
                        **pca_meta,
                    }
                    candidate_meta_full = copy.deepcopy(geom_meta)
                    candidate_meta_full.update(placement)
                    candidates.append((score, patch_world.copy(), candidate_meta_full))
                    if best is None or score > best[0]:
                        best = (score, patch_world, candidate_meta_full)
    finally:
        args._active_depth_maps = old_depth_maps

    if best is None:
        store_surface_candidates(args, [])
        if args.surface_score_mode == "natural":
            raise RuntimeError(
                "No clean-VGGT pointmap surface satisfied the natural sticker constraints. "
                "Relax surface_coverage_min/max or surface_min_visible_frames."
            )
        patch_world, meta = build_fixed_patch_plane_world(c2w[0], args)
        meta["placement_mode"] = "fixed_fallback_no_valid_vggt_surface"
        meta["geometry_source"] = "clean_vggt_pointmap"
        meta["clean_vggt_output"] = source
        meta["vggt_point_count"] = int(points.shape[0])
        meta["candidate_count"] = int(candidate_count)
        meta["tried_geometry_count"] = tried
        return patch_world, meta
    best[2]["tried_geometry_count"] = tried
    store_surface_candidates(args, candidates)
    return best[1], best[2]


def choose_vggt_manual_anchor_surface_plane(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
) -> tuple[np.ndarray, dict]:
    points = getattr(args, "_active_vggt_points", None)
    point_map_grid = getattr(args, "_active_vggt_point_map_grid", None)
    point_valid_grid = getattr(args, "_active_vggt_point_valid_grid", None)
    source = getattr(args, "_active_vggt_geometry_source", None)
    if points is None or point_map_grid is None or point_valid_grid is None:
        raise RuntimeError("Manual VGGT anchor requires a structured clean VGGT point map.")

    n_frames, grid_h, grid_w, _ = point_map_grid.shape
    frame_idx = min(max(int(args.manual_anchor_frame), 0), n_frames - 1)
    if args.manual_anchor_coordinates == "normalized":
        target_x = float(args.manual_anchor_x) * (grid_w - 1)
        target_y = float(args.manual_anchor_y) * (grid_h - 1)
    else:
        target_x = float(args.manual_anchor_x)
        target_y = float(args.manual_anchor_y)
    target_x = min(max(target_x, 0.0), grid_w - 1.0)
    target_y = min(max(target_y, 0.0), grid_h - 1.0)

    valid = point_valid_grid[frame_idx]
    radius = max(0, int(args.manual_anchor_search_radius))
    x0 = max(0, int(round(target_x)) - radius)
    x1 = min(grid_w, int(round(target_x)) + radius + 1)
    y0 = max(0, int(round(target_y)) - radius)
    y1 = min(grid_h, int(round(target_y)) + radius + 1)
    local_valid = np.argwhere(valid[y0:y1, x0:x1])
    if local_valid.size == 0:
        raise RuntimeError(
            f"No valid VGGT point was found near manual anchor ({target_x:.1f}, {target_y:.1f}) "
            f"within radius {radius}."
        )
    local_y = local_valid[:, 0] + y0
    local_x = local_valid[:, 1] + x0
    distances = (local_x - target_x) ** 2 + (local_y - target_y) ** 2
    best_local = int(np.argmin(distances))
    anchor_x = int(local_x[best_local])
    anchor_y = int(local_y[best_local])
    center_world = np.asarray(point_map_grid[frame_idx, anchor_y, anchor_x], dtype=np.float64)

    axes = estimate_world_surface_axes(points, center_world, c2w[frame_idx], args)
    if axes is None:
        raise RuntimeError("Could not estimate a local plane at the manual VGGT anchor.")
    axis_u, axis_v, normal, pca_meta = axes
    if pca_meta["plane_residual_ratio"] > args.fused_max_plane_residual:
        raise RuntimeError(
            f"Manual anchor is not planar enough: residual={pca_meta['plane_residual_ratio']:.6f}."
        )

    rolled_u, rolled_v = rotate_axes(axis_u, axis_v, args.manual_anchor_roll_degrees)
    patch_world = build_plane_world_from_world_axes(
        center_world,
        rolled_u,
        rolled_v,
        args.plane_width,
        args.plane_height,
    )
    old_depth_maps = getattr(args, "_active_depth_maps", None)
    args._active_depth_maps = depth_maps
    try:
        _, _, geom_meta = build_geometry_arrays_for_plane(
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

    support_ratio = float(geom_meta.get("surface_support_ratio_mean", 0.0))
    if args.surface_support_check and support_ratio < args.surface_min_support_ratio:
        raise RuntimeError(
            f"Manual anchor patch is not sufficiently supported by the carrier surface: "
            f"support={support_ratio:.4f}, required={args.surface_min_support_ratio:.4f}. "
            "Adjust manual_anchor_x/y or reduce plane_width/height."
        )

    center_first_cam_h = np.linalg.inv(c2w[frame_idx]) @ np.asarray(
        [center_world[0], center_world[1], center_world[2], 1.0], dtype=np.float64
    )
    placement = {
        "placement_mode": "vggt_manual_anchor_surface",
        "geometry_source": "clean_vggt_pointmap",
        "clean_vggt_output": source,
        "manual_anchor_coordinates": args.manual_anchor_coordinates,
        "manual_anchor_requested": [float(args.manual_anchor_x), float(args.manual_anchor_y)],
        "manual_anchor_frame": frame_idx,
        "manual_anchor_tensor_xy": [anchor_x, anchor_y],
        "manual_anchor_search_radius": radius,
        "center_world": center_world.astype(float).tolist(),
        "center_anchor_camera": center_first_cam_h[:3].astype(float).tolist(),
        "surface_normal_world": normal.astype(float).tolist(),
        "width_m": args.plane_width,
        "height_m": args.plane_height,
        "roll_degrees": float(args.manual_anchor_roll_degrees),
        **pca_meta,
    }
    geom_meta.update(placement)
    args._last_patch_plane_candidates = [(0.0, patch_world.copy(), copy.deepcopy(geom_meta))]
    return patch_world, geom_meta


def choose_depth_manual_anchor_surface_plane(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
) -> tuple[np.ndarray, dict]:
    n_frames = len(image_paths)
    frame_idx = min(max(int(args.manual_anchor_frame), 0), n_frames - 1)
    depth = depth_maps[frame_idx] if depth_maps else None
    if depth is None:
        raise RuntimeError("Manual depth anchor requires a matched TUM depth map.")

    tensor_h, tensor_w = tensor_hw
    if args.manual_anchor_coordinates == "normalized":
        target_x = float(args.manual_anchor_x) * (tensor_w - 1)
        target_y = float(args.manual_anchor_y) * (tensor_h - 1)
    else:
        target_x = float(args.manual_anchor_x)
        target_y = float(args.manual_anchor_y)
    target_x = min(max(target_x, 0.0), tensor_w - 1.0)
    target_y = min(max(target_y, 0.0), tensor_h - 1.0)

    radius = max(0, int(args.manual_anchor_search_radius))
    x0 = max(0, int(round(target_x)) - radius)
    x1 = min(tensor_w, int(round(target_x)) + radius + 1)
    y0 = max(0, int(round(target_y)) - radius)
    y1 = min(tensor_h, int(round(target_y)) + radius + 1)

    local_depth = depth[y0:y1, x0:x1]
    local_valid = np.argwhere(
        np.isfinite(local_depth)
        & (local_depth >= float(args.depth_min_m))
        & (local_depth <= float(args.depth_max_m))
    )
    if local_valid.size == 0:
        raise RuntimeError(
            f"No valid TUM depth point was found near manual anchor ({target_x:.1f}, {target_y:.1f}) "
            f"within radius {radius}."
        )

    local_y = local_valid[:, 0] + y0
    local_x = local_valid[:, 1] + x0
    distances = (local_x - target_x) ** 2 + (local_y - target_y) ** 2
    best_local = int(np.argmin(distances))
    anchor_x = int(local_x[best_local])
    anchor_y = int(local_y[best_local])

    proj = projection_params_for_frame(args, image_paths[frame_idx], tensor_hw)
    frame_intrinsics = intrinsics[frame_idx] if np.asarray(intrinsics).ndim == 3 else intrinsics
    xy = np.stack([local_x.astype(np.float64), local_y.astype(np.float64)], axis=-1)
    z = depth[local_y, local_x].astype(np.float64)
    local_cam = unproject_tensor_xy(xy, z, frame_intrinsics, proj)
    local_h = np.concatenate([local_cam, np.ones((local_cam.shape[0], 1), dtype=np.float64)], axis=1)
    local_world = (c2w[frame_idx] @ local_h.T).T[:, :3]

    center_cam = unproject_tensor_xy(
        np.asarray([[anchor_x, anchor_y]], dtype=np.float64),
        np.asarray([depth[anchor_y, anchor_x]], dtype=np.float64),
        frame_intrinsics,
        proj,
    )[0]
    center_world = (c2w[frame_idx] @ np.asarray([center_cam[0], center_cam[1], center_cam[2], 1.0])).T[:3]

    axes = estimate_world_surface_axes(local_world, center_world, c2w[frame_idx], args)
    if axes is None:
        raise RuntimeError("Could not estimate a local plane at the manual TUM-depth anchor.")
    axis_u, axis_v, normal, pca_meta = axes
    if pca_meta["plane_residual_ratio"] > args.fused_max_plane_residual:
        raise RuntimeError(
            f"Manual depth anchor is not planar enough: residual={pca_meta['plane_residual_ratio']:.6f}."
        )

    rolled_u, rolled_v = rotate_axes(axis_u, axis_v, args.manual_anchor_roll_degrees)
    patch_world = build_plane_world_from_world_axes(
        center_world,
        rolled_u,
        rolled_v,
        args.plane_width,
        args.plane_height,
    )
    old_depth_maps = getattr(args, "_active_depth_maps", None)
    args._active_depth_maps = depth_maps
    try:
        _, _, geom_meta = build_geometry_arrays_for_plane(
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

    support_ratio = float(geom_meta.get("surface_support_ratio_mean", 0.0))
    if args.surface_support_check and support_ratio < args.surface_min_support_ratio:
        raise RuntimeError(
            f"Manual depth anchor patch is not sufficiently supported by the carrier surface: "
            f"support={support_ratio:.4f}, required={args.surface_min_support_ratio:.4f}. "
            "Adjust manual_anchor_x/y or reduce plane_width/height."
        )
    if args.surface_score_mode == "natural" and not np.isfinite(geometry_score(geom_meta, args)):
        coverage = float(geom_meta.get("mask_coverage_mean", 0.0))
        raw_coverage = float(geom_meta.get("raw_projected_coverage_mean", 0.0))
        visibility_ratio = float(geom_meta.get("visibility_ratio_mean", 0.0))
        raise RuntimeError(
            "Manual depth anchor patch violates natural sticker constraints: "
            f"coverage={coverage:.6f}, raw_coverage={raw_coverage:.6f}, "
            f"visibility={visibility_ratio:.4f}, support={support_ratio:.4f}. "
            "Adjust manual_anchor_x/y, reduce plane_width/height, or relax "
            "surface_coverage_min/max and visibility/support thresholds."
        )

    center_first_cam_h = np.linalg.inv(c2w[frame_idx]) @ np.asarray(
        [center_world[0], center_world[1], center_world[2], 1.0], dtype=np.float64
    )
    placement = {
        "placement_mode": "depth_manual_anchor_surface",
        "geometry_source": "tum_depth_manual_anchor",
        "manual_anchor_coordinates": args.manual_anchor_coordinates,
        "manual_anchor_requested": [float(args.manual_anchor_x), float(args.manual_anchor_y)],
        "manual_anchor_frame": frame_idx,
        "manual_anchor_tensor_xy": [anchor_x, anchor_y],
        "manual_anchor_search_radius": radius,
        "local_depth_point_count": int(local_world.shape[0]),
        "center_world": center_world.astype(float).tolist(),
        "center_anchor_camera": center_first_cam_h[:3].astype(float).tolist(),
        "surface_normal_world": normal.astype(float).tolist(),
        "width_m": args.plane_width,
        "height_m": args.plane_height,
        "roll_degrees": float(args.manual_anchor_roll_degrees),
        **pca_meta,
    }
    geom_meta.update(placement)
    args._last_patch_plane_candidates = [(0.0, patch_world.copy(), copy.deepcopy(geom_meta))]
    return patch_world, geom_meta


def choose_depth_manual_quad_surface_plane(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
) -> tuple[np.ndarray, dict]:
    n_frames = len(image_paths)
    frame_idx = min(max(int(args.manual_anchor_frame), 0), n_frames - 1)
    depth = depth_maps[frame_idx] if depth_maps else None
    if depth is None:
        raise RuntimeError("Manual quad surface requires a matched TUM depth map.")
    if not str(args.manual_quad_xy).strip():
        raise RuntimeError(
            "--manual_quad_xy is required for depth_manual_quad_surface. "
            "Use x0,y0,x1,y1,x2,y2,x3,y3 in TL,TR,BR,BL order."
        )

    tensor_h, tensor_w = tensor_hw
    quad_xy = parse_manual_quad_xy(args.manual_quad_xy, tensor_hw, args.manual_quad_coordinates)
    fit_quad_xy = shrink_quad_xy(quad_xy, args.manual_quad_fit_shrink)
    fit_mask = manual_quad_mask(fit_quad_xy, tensor_hw)
    full_mask = manual_quad_mask(quad_xy, tensor_hw)
    valid = (
        fit_mask
        & np.isfinite(depth)
        & (depth >= float(args.depth_min_m))
        & (depth <= float(args.depth_max_m))
    )
    ys, xs = np.nonzero(valid)
    stride = max(1, int(args.manual_quad_depth_sample_stride))
    if stride > 1 and xs.size > 0:
        keep = ((xs + ys) % stride) == 0
        xs = xs[keep]
        ys = ys[keep]
    min_depth_samples = max(6, int(args.fused_min_neighbors))
    if xs.size < min_depth_samples and args.manual_quad_fit_shrink < 1.0:
        valid = (
            full_mask
            & np.isfinite(depth)
            & (depth >= float(args.depth_min_m))
            & (depth <= float(args.depth_max_m))
        )
        ys, xs = np.nonzero(valid)
        if stride > 1 and xs.size > 0:
            keep = ((xs + ys) % stride) == 0
            xs = xs[keep]
            ys = ys[keep]
        fit_quad_xy = quad_xy
    if xs.size < min_depth_samples:
        raise RuntimeError(
            f"Manual quad has too few valid depth samples: {xs.size}. "
            "Move the quad inside a depth-valid surface or reduce --manual_quad_depth_sample_stride."
        )

    proj = projection_params_for_frame(args, image_paths[frame_idx], tensor_hw)
    frame_intrinsics = intrinsics[frame_idx] if np.asarray(intrinsics).ndim == 3 else intrinsics
    sample_xy = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=-1)
    sample_z = depth[ys, xs].astype(np.float64)
    sample_cam = unproject_tensor_xy(sample_xy, sample_z, frame_intrinsics, proj)

    center_cam, normal_cam, eigvals_ordered, initial_residual = fit_camera_plane_pca(sample_cam)
    initial_distances = np.abs((sample_cam - center_cam[None, :]) @ normal_cam)
    tolerance = max(
        float(args.manual_quad_plane_inlier_tolerance),
        float(np.median(sample_z)) * float(args.surface_support_rel_tolerance),
    )
    inlier_mask = initial_distances <= tolerance
    min_inliers = max(
        min_depth_samples,
        int(math.ceil(float(args.manual_quad_min_inlier_ratio) * sample_cam.shape[0])),
    )
    if int(inlier_mask.sum()) < min_inliers:
        keep_count = min(sample_cam.shape[0], min_inliers)
        keep_idx = np.argsort(initial_distances)[:keep_count]
        inlier_mask = np.zeros(sample_cam.shape[0], dtype=bool)
        inlier_mask[keep_idx] = True
    inlier_cam = sample_cam[inlier_mask]
    center_cam, normal_cam, eigvals_ordered, residual = fit_camera_plane_pca(inlier_cam)
    final_distances = np.abs((sample_cam - center_cam[None, :]) @ normal_cam)
    inlier_distances = final_distances[inlier_mask]
    inlier_ratio = float(inlier_cam.shape[0] / max(1, sample_cam.shape[0]))
    inlier_p95 = float(np.percentile(inlier_distances, 95)) if inlier_distances.size else float("inf")
    if residual > args.fused_max_plane_residual and inlier_p95 > tolerance:
        raise RuntimeError(
            f"Manual quad is not planar enough after robust fitting: "
            f"initial_residual={initial_residual:.6f}, residual={residual:.6f}, "
            f"inlier_ratio={inlier_ratio:.4f}, p95_dist={inlier_p95:.4f}m, "
            f"tolerance={tolerance:.4f}m, limit={args.fused_max_plane_residual:.6f}."
        )

    quad_uv = tensor_xy_to_original_uv(quad_xy, proj)
    rays = np.empty((4, 3), dtype=np.float64)
    rays[:, 0] = (quad_uv[:, 0] - frame_intrinsics[0, 2]) / frame_intrinsics[0, 0]
    rays[:, 1] = (quad_uv[:, 1] - frame_intrinsics[1, 2]) / frame_intrinsics[1, 1]
    rays[:, 2] = 1.0
    denom = rays @ normal_cam
    if np.any(np.abs(denom) < 1e-8):
        raise RuntimeError("Manual quad corners are nearly parallel to the fitted depth plane.")
    scale = float(np.dot(normal_cam, center_cam)) / denom
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-6):
        raise RuntimeError("Manual quad corners do not intersect the fitted depth plane in front of the camera.")
    corners_cam = rays * scale[:, None]
    corners_h = np.concatenate([corners_cam, np.ones((4, 1), dtype=np.float64)], axis=1)
    patch_world = (c2w[frame_idx] @ corners_h.T).T[:, :3]

    old_depth_maps = getattr(args, "_active_depth_maps", None)
    args._active_depth_maps = depth_maps
    try:
        _, _, geom_meta = build_geometry_arrays_for_plane(
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

    support_ratio = float(geom_meta.get("surface_support_ratio_mean", 0.0))
    if args.surface_support_check and support_ratio < args.surface_min_support_ratio:
        raise RuntimeError(
            f"Manual quad patch is not sufficiently supported by the carrier surface: "
            f"support={support_ratio:.4f}, required={args.surface_min_support_ratio:.4f}. "
            "Move the quad corners inside the carrier surface or relax support thresholds."
        )
    if args.surface_score_mode == "natural" and not np.isfinite(geometry_score(geom_meta, args)):
        coverage = float(geom_meta.get("mask_coverage_mean", 0.0))
        raw_coverage = float(geom_meta.get("raw_projected_coverage_mean", 0.0))
        visibility_ratio = float(geom_meta.get("visibility_ratio_mean", 0.0))
        raise RuntimeError(
            "Manual quad patch violates natural sticker constraints: "
            f"coverage={coverage:.6f}, raw_coverage={raw_coverage:.6f}, "
            f"visibility={visibility_ratio:.4f}, support={support_ratio:.4f}. "
            "Adjust manual_quad_xy or relax surface_coverage_min/max and visibility/support thresholds."
        )

    center_world = patch_world.mean(axis=0)
    center_first_cam_h = np.linalg.inv(c2w[frame_idx]) @ np.asarray(
        [center_world[0], center_world[1], center_world[2], 1.0], dtype=np.float64
    )
    width_top = float(np.linalg.norm(patch_world[1] - patch_world[0]))
    width_bottom = float(np.linalg.norm(patch_world[2] - patch_world[3]))
    height_left = float(np.linalg.norm(patch_world[3] - patch_world[0]))
    height_right = float(np.linalg.norm(patch_world[2] - patch_world[1]))
    normal_world = c2w[frame_idx][:3, :3] @ normal_cam
    placement = {
        "placement_mode": "depth_manual_quad_surface",
        "geometry_source": "tum_depth_manual_quad",
        "manual_quad_coordinates": args.manual_quad_coordinates,
        "manual_quad_requested": [float(v) for v in parse_float_list(args.manual_quad_xy.replace(";", ","))],
        "manual_quad_tensor_xy": quad_xy.astype(float).tolist(),
        "manual_quad_fit_tensor_xy": fit_quad_xy.astype(float).tolist(),
        "manual_quad_frame": frame_idx,
        "manual_quad_fit_shrink": float(args.manual_quad_fit_shrink),
        "manual_quad_depth_sample_stride": stride,
        "manual_quad_depth_sample_count": int(sample_cam.shape[0]),
        "manual_quad_inlier_count": int(inlier_cam.shape[0]),
        "manual_quad_inlier_ratio": inlier_ratio,
        "manual_quad_plane_inlier_tolerance_m": tolerance,
        "manual_quad_initial_plane_residual_ratio": initial_residual,
        "manual_quad_inlier_distance_p95_m": inlier_p95,
        "center_world": center_world.astype(float).tolist(),
        "center_anchor_camera": center_first_cam_h[:3].astype(float).tolist(),
        "surface_normal_camera": normal_cam.astype(float).tolist(),
        "surface_normal_world": normal_world.astype(float).tolist(),
        "width_m": float((width_top + width_bottom) / 2.0),
        "height_m": float((height_left + height_right) / 2.0),
        "width_top_m": width_top,
        "width_bottom_m": width_bottom,
        "height_left_m": height_left,
        "height_right_m": height_right,
        "pca_eigenvalues": eigvals_ordered.astype(float).tolist(),
        "plane_residual_ratio": residual,
    }
    geom_meta.update(placement)
    args._last_patch_plane_candidates = [(0.0, patch_world.copy(), copy.deepcopy(geom_meta))]
    return patch_world, geom_meta


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
    support_ratios = []
    support_coverages = []
    projected_corners = []
    depth_maps = getattr(args, "_active_depth_maps", None)
    for frame_idx, pose in enumerate(c2w):
        frame_intrinsics = intrinsics[frame_idx] if np.asarray(intrinsics).ndim == 3 else intrinsics
        w2c = np.linalg.inv(pose)
        corners_h = np.concatenate([patch_world, np.ones((4, 1), dtype=np.float64)], axis=1)
        corners_cam = (w2c @ corners_h.T).T[:, :3]
        proj = projection_params_for_frame(args, image_paths[frame_idx], tensor_hw)
        dst, corner_valid = camera_to_tensor_xy(corners_cam, frame_intrinsics, proj)
        projected_corners.append(dst.astype(float).tolist())

        if not bool(corner_valid.all()):
            grid = np.zeros((tensor_h, tensor_w, 2), dtype=np.float32)
            mask = np.zeros((1, tensor_h, tensor_w), dtype=np.float32)
            grids.append(grid)
            masks.append(mask)
            coverages.append(0.0)
            raw_coverages.append(0.0)
            visibility_ratios.append(0.0)
            support_ratios.append(0.0)
            support_coverages.append(0.0)
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
        support_ratio = 1.0 if not args.surface_support_check else 0.0
        support_coverage = float(raw_inside.mean()) if not args.surface_support_check else 0.0
        if args.use_depth_visibility and depth_maps is not None and depth_maps[frame_idx] is not None:
            patch_depth, depth_valid = plane_depth_map(corners_cam, frame_intrinsics, proj, tensor_hw)
            scene_depth = depth_maps[frame_idx]
            scene_valid = (scene_depth > args.depth_min_m) & (scene_depth < args.depth_max_m)
            visible = depth_valid & scene_valid & (patch_depth <= scene_depth + args.visibility_depth_margin)
            mask_bool = raw_inside & visible
            if args.surface_support_check:
                occluded_by_foreground = scene_valid & (
                    scene_depth < patch_depth - args.visibility_depth_margin
                )
                evaluable = raw_inside & depth_valid & scene_valid & ~occluded_by_foreground
                support_tolerance = np.maximum(
                    args.surface_support_abs_tolerance,
                    np.abs(patch_depth) * args.surface_support_rel_tolerance,
                )
                supported = evaluable & (np.abs(scene_depth - patch_depth) <= support_tolerance)
                evaluable_count = int(evaluable.sum())
                support_ratio = float(supported.sum() / evaluable_count) if evaluable_count > 0 else 0.0
                support_coverage = float(supported.mean())
        mask = mask_bool.astype(np.float32).reshape(1, tensor_h, tensor_w)
        grids.append(grid)
        masks.append(mask)
        raw_coverage = float(raw_inside.mean())
        visible_coverage = float(mask.mean())
        raw_coverages.append(raw_coverage)
        coverages.append(visible_coverage)
        visibility_ratios.append(visible_coverage / raw_coverage if raw_coverage > 0 else 0.0)
        support_ratios.append(support_ratio)
        support_coverages.append(support_coverage)

    meta = {
        "plane_corners_world": patch_world.astype(float).tolist(),
        "projected_corners": projected_corners,
        "mask_coverage_per_frame": coverages,
        "mask_coverage_mean": float(np.mean(coverages)),
        "raw_projected_coverage_per_frame": raw_coverages,
        "raw_projected_coverage_mean": float(np.mean(raw_coverages)),
        "visibility_ratio_per_frame": visibility_ratios,
        "visibility_ratio_mean": float(np.mean(visibility_ratios)),
        "surface_support_ratio_per_frame": support_ratios,
        "surface_support_ratio_mean": float(np.mean(support_ratios)),
        "surface_support_coverage_per_frame": support_coverages,
        "surface_support_coverage_mean": float(np.mean(support_coverages)),
        "uses_surface_support_check": bool(args.surface_support_check),
        "uses_depth_visibility": bool(args.use_depth_visibility),
    }
    return np.stack(grids, axis=0), np.stack(masks, axis=0), meta


def geometry_score(meta: dict, args: argparse.Namespace) -> float:
    coverages = np.asarray(meta["mask_coverage_per_frame"], dtype=np.float64)
    if coverages.size == 0:
        return -float("inf")
    visible_frames = int(np.count_nonzero(coverages > 1e-8))
    visibility_ratio = float(meta.get("visibility_ratio_mean", 0.0))
    support_ratio = float(meta.get("surface_support_ratio_mean", 0.0))
    mean_coverage = float(coverages.mean())

    if args.surface_score_mode == "natural":
        if mean_coverage < args.surface_coverage_min:
            return -float("inf")
        if args.surface_coverage_max > 0 and mean_coverage > args.surface_coverage_max:
            return -float("inf")
        if visible_frames < args.surface_min_visible_frames:
            return -float("inf")
        if visibility_ratio < args.surface_min_visibility_ratio:
            return -float("inf")
        if args.surface_support_check and support_ratio < args.surface_min_support_ratio:
            return -float("inf")

    support_bonus = 0.1 * support_ratio if args.surface_support_check else 0.0
    return float(coverages.mean() + 0.25 * coverages.min() + support_bonus)


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

    proj = projection_params_for_frame(args, image_paths[0], tensor_hw)
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
                candidate_ok, candidate_meta = surface_candidate_first_camera_meta(center_cam, normal, args)
                if not candidate_ok:
                    continue
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
                        score = geometry_score(geom_meta, args)
                        if not np.isfinite(score):
                            continue
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
                            **candidate_meta,
                        }
                        candidate_meta_full = copy.deepcopy(geom_meta)
                        candidate_meta_full.update(placement)
                        candidates.append((score, patch_world.copy(), candidate_meta_full))
                        if best is None or score > best[0]:
                            best = (score, patch_world, candidate_meta_full)
    finally:
        args._active_depth_maps = old_depth_maps

    if best is None:
        store_surface_candidates(args, [])
        if args.surface_score_mode == "natural":
            raise RuntimeError(
                "No depth surface satisfied the natural sticker constraints. "
                "Relax surface_coverage_min/max or surface_min_visible_frames."
            )
        patch_world, meta = build_fixed_patch_plane_world(c2w[0], args)
        meta["placement_mode"] = "fixed_fallback_no_valid_surface"
        meta["candidate_count"] = tried
        return patch_world, meta
    best[2]["candidate_count"] = tried
    store_surface_candidates(args, candidates)
    return best[1], best[2]


def run_patch_plane_selector(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
) -> tuple[np.ndarray, dict]:
    if args.plane_mode == "vggt_pointmap_surface":
        return choose_vggt_pointmap_surface_plane(
            c2w, image_paths, tensor_hw, texture_size, intrinsics, args, depth_maps
        )
    if args.plane_mode == "vggt_manual_anchor_surface":
        return choose_vggt_manual_anchor_surface_plane(
            c2w, image_paths, tensor_hw, texture_size, intrinsics, args, depth_maps
        )
    if args.plane_mode == "depth_manual_anchor_surface":
        return choose_depth_manual_anchor_surface_plane(
            c2w, image_paths, tensor_hw, texture_size, intrinsics, args, depth_maps
        )
    if args.plane_mode == "depth_manual_quad_surface":
        return choose_depth_manual_quad_surface_plane(
            c2w, image_paths, tensor_hw, texture_size, intrinsics, args, depth_maps
        )
    if args.plane_mode == "fused_depth_surface":
        return choose_fused_depth_surface_plane(
            c2w, image_paths, tensor_hw, texture_size, intrinsics, args, depth_maps
        )
    if args.plane_mode == "auto_depth_surface":
        return choose_auto_depth_surface_plane(
            c2w, image_paths, tensor_hw, texture_size, intrinsics, args, depth_maps
        )
    result = build_fixed_patch_plane_world(c2w[0], args)
    args._last_patch_plane_candidates = [(0.0, result[0].copy(), copy.deepcopy(result[1]))]
    return result


def select_patch_plane_with_natural_relaxation(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
) -> tuple[np.ndarray, dict]:
    constraint_names = (
        "surface_coverage_min",
        "surface_coverage_max",
        "surface_min_visible_frames",
        "surface_min_visibility_ratio",
        "surface_orientation_filter",
        "surface_max_tilt_degrees",
        "surface_min_center_depth",
        "surface_max_center_depth",
        "surface_min_support_ratio",
    )
    original = {name: getattr(args, name) for name in constraint_names}
    profiles = [
        {
            "level": 0,
            "name": "strict",
        }
    ]
    if args.natural_auto_relax and args.surface_score_mode == "natural":
        profiles.extend(
            [
                {
                    "level": 1,
                    "name": "gentle",
                    "surface_coverage_min": min(original["surface_coverage_min"], 0.002),
                    "surface_coverage_max": max(original["surface_coverage_max"], 0.06),
                    "surface_min_visible_frames": max(
                        args.natural_relax_min_visible_frames,
                        int(original["surface_min_visible_frames"]) - 1,
                    ),
                    "surface_min_visibility_ratio": max(
                        args.natural_relax_min_visibility_ratio,
                        float(original["surface_min_visibility_ratio"]) - 0.15,
                    ),
                    "surface_max_tilt_degrees": min(
                        args.natural_relax_max_tilt_degrees,
                        float(original["surface_max_tilt_degrees"]) + 10.0,
                    ),
                    "surface_min_center_depth": max(
                        args.natural_relax_min_center_depth,
                        float(original["surface_min_center_depth"]) - 0.3,
                    ),
                    "surface_min_support_ratio": max(
                        args.natural_relax_min_support_ratio,
                        float(original["surface_min_support_ratio"]) - 0.15,
                    ),
                },
                {
                    "level": 2,
                    "name": "broad_rigid_surface",
                    "surface_coverage_min": min(original["surface_coverage_min"], 0.001),
                    "surface_coverage_max": max(
                        original["surface_coverage_max"], args.natural_relax_max_coverage
                    ),
                    "surface_min_visible_frames": args.natural_relax_min_visible_frames,
                    "surface_min_visibility_ratio": args.natural_relax_min_visibility_ratio,
                    "surface_orientation_filter": args.natural_relax_orientation_filter,
                    "surface_max_tilt_degrees": args.natural_relax_max_tilt_degrees,
                    "surface_min_center_depth": args.natural_relax_min_center_depth,
                    "surface_min_support_ratio": args.natural_relax_min_support_ratio,
                },
            ]
        )

    errors = []
    try:
        for profile in profiles:
            for name, value in original.items():
                setattr(args, name, value)
            for name, value in profile.items():
                if name not in ("level", "name"):
                    setattr(args, name, value)
            try:
                patch_world, meta = run_patch_plane_selector(
                    c2w,
                    image_paths,
                    tensor_hw,
                    texture_size,
                    intrinsics,
                    args,
                    depth_maps,
                )
                used = {name: getattr(args, name) for name in constraint_names}
                meta.update(
                    {
                        "natural_relaxation_level": int(profile["level"]),
                        "natural_relaxation_name": str(profile["name"]),
                        "natural_constraints_original": original,
                        "natural_constraints_used": used,
                    }
                )
                candidates = getattr(args, "_last_patch_plane_candidates", [])
                for _, _, candidate_meta in candidates:
                    candidate_meta.update(
                        {
                            "natural_relaxation_level": int(profile["level"]),
                            "natural_relaxation_name": str(profile["name"]),
                            "natural_constraints_original": original,
                            "natural_constraints_used": used,
                        }
                    )
                if int(profile["level"]) > 0:
                    print(
                        f"[natural-relax] level={profile['level']} name={profile['name']} "
                        f"coverage=[{used['surface_coverage_min']},{used['surface_coverage_max']}] "
                        f"visible_frames={used['surface_min_visible_frames']} "
                        f"visibility={used['surface_min_visibility_ratio']} "
                        f"support={used['surface_min_support_ratio']} "
                        f"orientation={used['surface_orientation_filter']}"
                    )
                return patch_world, meta
            except RuntimeError as exc:
                errors.append(f"level {profile['level']} ({profile['name']}): {exc}")
                continue
    finally:
        for name, value in original.items():
            setattr(args, name, value)

    raise RuntimeError(
        "No physically plausible natural surface was found after all relaxation levels. "
        + " | ".join(errors)
    )


def choose_patch_plane_world(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
) -> tuple[np.ndarray, dict]:
    cache = getattr(args, "_plane_world_cache", None)
    if cache is None:
        cache = {}
        args._plane_world_cache = cache
    cache_key = (
        args.plane_mode,
        tuple(image_paths),
        texture_size,
        args.plane_width,
        args.plane_height,
        args.plane_distance,
        args.plane_center_x,
        args.plane_center_y,
        args.use_depth_visibility,
        args.optimize_geometry,
        args.geometry_size_scales,
        args.geometry_roll_degrees,
        args.fused_point_stride,
        args.fused_surface_candidates,
        args.fused_normal_radius,
        args.fused_max_plane_residual,
        args.clean_vggt_output_root,
        args.vggt_point_conf_percentile,
        args.surface_score_mode,
        args.surface_coverage_min,
        args.surface_coverage_max,
        args.surface_min_visible_frames,
        args.surface_min_visibility_ratio,
        args.surface_orientation_filter,
        args.surface_max_tilt_degrees,
        args.surface_min_center_depth,
        args.surface_max_center_depth,
        args.surface_strength_search,
        args.surface_strength_candidates,
        args.surface_strength_steps,
        args.surface_strength_lr,
        args.surface_strength_texture_init,
        args.surface_strength_regularization_weight,
        args.tv_weight,
        args.printability_weight,
        args.low_frequency_weight,
        args.low_frequency_kernel,
        args.texture_init,
        args.natural_auto_relax,
        args.natural_relax_max_coverage,
        args.natural_relax_min_visible_frames,
        args.natural_relax_min_visibility_ratio,
        args.natural_relax_orientation_filter,
        args.natural_relax_max_tilt_degrees,
        args.natural_relax_min_center_depth,
        args.surface_support_check,
        args.surface_support_abs_tolerance,
        args.surface_support_rel_tolerance,
        args.surface_min_support_ratio,
        args.natural_relax_min_support_ratio,
        args.manual_anchor_coordinates,
        args.manual_anchor_x,
        args.manual_anchor_y,
        args.manual_anchor_frame,
        args.manual_anchor_search_radius,
        args.manual_anchor_roll_degrees,
        args.manual_quad_coordinates,
        args.manual_quad_xy,
        args.manual_quad_depth_sample_stride,
        args.manual_quad_fit_shrink,
        args.manual_quad_plane_inlier_tolerance,
        args.manual_quad_min_inlier_ratio,
    )
    if cache_key in cache:
        cached = cache[cache_key]
        if len(cached) == 3:
            patch_world, meta, candidates = cached
            args._last_patch_plane_candidates = [
                (float(score), candidate_patch.copy(), copy.deepcopy(candidate_meta))
                for score, candidate_patch, candidate_meta in candidates
            ]
        else:
            patch_world, meta = cached
            args._last_patch_plane_candidates = [(float(meta.get("score", 0.0)), patch_world.copy(), copy.deepcopy(meta))]
        return patch_world.copy(), copy.deepcopy(meta)

    args._last_patch_plane_candidates = []
    result = select_patch_plane_with_natural_relaxation(
        c2w,
        image_paths,
        tensor_hw,
        texture_size,
        intrinsics,
        args,
        depth_maps,
    )

    candidates = getattr(args, "_last_patch_plane_candidates", [])
    cache[cache_key] = (
        result[0].copy(),
        copy.deepcopy(result[1]),
        [(float(score), patch.copy(), copy.deepcopy(meta)) for score, patch, meta in candidates],
    )
    return result


def build_geometry_grids(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    depth_maps: list[np.ndarray | None],
    images: torch.Tensor | None = None,
    clean_features: list[torch.Tensor] | None = None,
    model: torch.nn.Module | None = None,
    dtype: torch.dtype | None = None,
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
    candidates = getattr(args, "_last_patch_plane_candidates", [])
    if (
        args.surface_strength_search
        and model is not None
        and dtype is not None
        and images is not None
        and clean_features is not None
        and len(candidates) > 1
    ):
        strength_cache = getattr(args, "_strength_plane_cache", None)
        if strength_cache is None:
            strength_cache = {}
            args._strength_plane_cache = strength_cache
        strength_cache_key = (
            tuple(image_paths),
            texture_size,
            args.surface_strength_candidates,
            args.surface_strength_steps,
            args.surface_strength_lr,
            args.surface_strength_texture_init,
            args.surface_strength_regularization_weight,
            args.tv_weight,
            args.printability_weight,
            args.low_frequency_weight,
            args.low_frequency_kernel,
        )
        if strength_cache_key in strength_cache:
            patch_world, placement_meta = strength_cache[strength_cache_key]
            patch_world = patch_world.copy()
            placement_meta = copy.deepcopy(placement_meta)
        else:
            patch_world, placement_meta = select_candidate_by_feature_probe(
                candidates,
                images,
                clean_features,
                c2w,
                image_paths,
                tensor_hw,
                texture_size,
                intrinsics,
                args,
                depth_maps,
                model,
                dtype,
                device,
            )
            strength_cache[strength_cache_key] = (patch_world.copy(), copy.deepcopy(placement_meta))
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


def select_candidate_by_feature_probe(
    candidates: list[tuple[float, np.ndarray, dict]],
    images: torch.Tensor,
    clean_features: list[torch.Tensor],
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    depth_maps: list[np.ndarray | None],
    model: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    if not candidates:
        raise RuntimeError("surface_strength_search requested but no candidate planes were available.")

    best: tuple[float, np.ndarray, dict] | None = None
    old_depth_maps = getattr(args, "_active_depth_maps", None)
    args._active_depth_maps = depth_maps
    try:
        for rank, (geometry_score_value, patch_world, placement_meta) in enumerate(candidates, start=1):
            grids_np, masks_np, meta = build_geometry_arrays_for_plane(
                patch_world,
                c2w,
                image_paths,
                tensor_hw,
                texture_size,
                intrinsics,
                args,
            )
            meta.update(copy.deepcopy(placement_meta))
            grids = torch.from_numpy(grids_np).to(device)
            masks = torch.from_numpy(masks_np).to(device)

            probe_texture = initialize_texture(
                args,
                device,
                init=args.surface_strength_texture_init,
                requires_grad=args.surface_strength_steps > 0,
            )
            optimizer = (
                torch.optim.AdamW([probe_texture], lr=args.surface_strength_lr)
                if args.surface_strength_steps > 0
                else None
            )
            last_feature_l1 = None
            last_reg = 0.0
            for _ in range(max(1, args.surface_strength_steps)):
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                adv_images = apply_geometry_patch(images, probe_texture, grids, masks, args, training=True)
                adv_features = extract_features(
                    model,
                    adv_images,
                    dtype,
                    args.feature_layer,
                    args.activation_checkpoint,
                )
                feature_loss, terms = feature_l1_loss(adv_features, clean_features)
                reg_terms = patch_regularization_terms(probe_texture, args)
                score_loss = feature_loss - args.surface_strength_regularization_weight * reg_terms["regularization_total"]
                last_feature_l1 = float(terms["feature_l1"])
                last_reg = float(reg_terms["regularization_total"].detach().cpu())
                if optimizer is not None:
                    (-score_loss).backward()
                    optimizer.step()
                    with torch.no_grad():
                        probe_texture.clamp_(args.print_min, args.print_max)

            final_score = float(last_feature_l1 if last_feature_l1 is not None else 0.0) - (
                args.surface_strength_regularization_weight * last_reg
            )
            meta.update(
                {
                    "strength_search_enabled": True,
                    "strength_candidate_rank": rank,
                    "strength_candidate_count": len(candidates),
                    "strength_geometry_score": float(geometry_score_value),
                    "strength_probe_steps": int(args.surface_strength_steps),
                    "strength_probe_texture_init": args.surface_strength_texture_init,
                    "strength_probe_feature_l1": float(last_feature_l1 if last_feature_l1 is not None else 0.0),
                    "strength_probe_regularization": float(last_reg),
                    "strength_probe_score": final_score,
                }
            )
            if best is None or final_score > best[0]:
                best = (final_score, patch_world.copy(), meta)
    finally:
        args._active_depth_maps = old_depth_maps

    if best is None:
        raise RuntimeError("surface_strength_search did not produce a valid candidate.")
    return best[1], best[2]


def prepare_texture_for_render(texture: torch.Tensor, args: argparse.Namespace | None,
                               training: bool, n_frames: int = 1) -> torch.Tensor:
    """Photometric EOT, sampled independently per frame.

    The factors used to be single scalars applied before the texture was expanded
    across frames, so every frame in a sequence got the identical perturbation.
    That is not expectation over transformation, it is one transformation per step:
    the attack could tune to whatever brightness happened to be drawn instead of
    having to work across the range, which is how a large patch ends up leaning on
    the augmentation rather than being made robust by it. Sampling per frame makes
    each step average over ten draws instead of one.
    """
    if args is None:
        return texture.clamp(0.0, 1.0).expand(n_frames, -1, -1, -1)

    rendered = texture.clamp(args.print_min, args.print_max)
    rendered = rendered.expand(n_frames, -1, -1, -1)
    if not training or not args.physical_eot:
        return rendered

    strength = float(np.clip(getattr(args, "_eot_strength", 1.0), 0.0, 1.0))

    def per_frame(spread: float) -> torch.Tensor:
        spread *= strength
        return torch.empty((n_frames, 1, 1, 1), device=texture.device).uniform_(
            1.0 - spread, 1.0 + spread)

    if args.eot_brightness > 0:
        rendered = rendered * per_frame(args.eot_brightness)
    if args.eot_contrast > 0:
        mean = rendered.mean(dim=(-2, -1), keepdim=True)
        rendered = (rendered - mean) * per_frame(args.eot_contrast) + mean
    if args.eot_gamma > 0:
        rendered = rendered.clamp(1e-5, 1.0).pow(per_frame(args.eot_gamma))
    return rendered.clamp(args.print_min, args.print_max)


def jitter_sampling_grids(grids: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Per-frame near-identity homography on the texture sampling grid.

    The rendering pipeline had no geometric EOT at all: the homography H_t was
    reproduced exactly every step, so the optimised texture only ever had to work
    at one precise alignment. Physically the printed patch sits on the surface with
    some placement and orientation error, the surface is not perfectly planar, and
    the camera calibration is approximate. Perturbing the grid in normalised texture
    coordinates covers all three without touching the mask, i.e. the patch occupies
    the same image region but its content shifts slightly within it.
    """
    n = grids.shape[0]
    device, dtype = grids.device, grids.dtype
    strength = float(np.clip(getattr(args, "_eot_strength", 1.0), 0.0, 1.0))

    def uniform(spread: float) -> torch.Tensor:
        spread *= strength
        if spread <= 0:
            return torch.zeros((n,), device=device, dtype=dtype)
        return torch.empty((n,), device=device, dtype=dtype).uniform_(-spread, spread)

    angle = uniform(math.radians(args.eot_geo_rotate_degrees))
    scale = 1.0 + uniform(args.eot_geo_scale)
    tx, ty = uniform(args.eot_geo_translate), uniform(args.eot_geo_translate)
    px, py = uniform(args.eot_geo_perspective), uniform(args.eot_geo_perspective)

    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    h = torch.zeros((n, 3, 3), device=device, dtype=dtype)
    h[:, 0, 0], h[:, 0, 1], h[:, 0, 2] = cos, -sin, tx
    h[:, 1, 0], h[:, 1, 1], h[:, 1, 2] = sin, cos, ty
    h[:, 2, 0], h[:, 2, 1], h[:, 2, 2] = px, py, 1.0

    flat = grids.reshape(n, -1, 2)
    homo = torch.cat([flat, torch.ones_like(flat[..., :1])], dim=-1)
    warped = torch.bmm(homo, h.transpose(1, 2))
    denom = warped[..., 2:3]
    denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
    return (warped[..., :2] / denom).reshape_as(grids)


def apply_geometry_patch(
    images: torch.Tensor,
    texture: torch.Tensor,
    grids: torch.Tensor,
    masks: torch.Tensor,
    args: argparse.Namespace | None = None,
    training: bool = False,
) -> torch.Tensor:
    n_frames = images.shape[0]
    rendered_texture = prepare_texture_for_render(texture, args, training, n_frames)
    if args is not None and training and args.physical_eot:
        grids = jitter_sampling_grids(grids, args)
    sampled = F.grid_sample(
        rendered_texture,
        grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    patched = (images * (1.0 - masks) + sampled * masks).clamp(0.0, 1.0)
    if args is not None and training and args.physical_eot and args.eot_noise_std > 0:
        # Whole-image white noise changes millions of pixels that the patch cannot
        # control.  On VGGT-1B its stochastic backbone gradient overwhelmed the
        # expected patch gradient and all three 1000-step EOT runs stayed at their
        # clean-to-target distance (~0.93).  Model print/surface noise where it
        # belongs: on the rendered patch.  Camera-wide noise remains an evaluation
        # stress test, not an optimisation variable.
        strength = float(np.clip(getattr(args, "_eot_strength", 1.0), 0.0, 1.0))
        noise = torch.randn_like(sampled) * (args.eot_noise_std * strength)
        patched = (patched + noise * masks).clamp(0.0, 1.0)
    return patched


def scheduled_eot_strength(update_idx: int, total_updates: int,
                           warmup_fraction: float) -> float:
    """Start from the solvable clean objective, then progressively robustify it."""
    if total_updates <= 1:
        return 1.0
    progress = float(update_idx) / float(total_updates - 1)
    start = float(np.clip(warmup_fraction, 0.0, 0.95))
    if progress <= start:
        return 0.0
    return float(np.clip((progress - start) / max(1.0 - start, 1e-12), 0.0, 1.0))


def c2w_numpy_to_relative_tensor(c2w: np.ndarray, device: torch.device) -> torch.Tensor:
    c2w_t = torch.from_numpy(np.asarray(c2w, dtype=np.float32)).to(device)
    if c2w_t.ndim != 3:
        raise ValueError(f"Expected c2w with shape [T,4,4], got {tuple(c2w_t.shape)}")
    c2w_t = c2w_t.unsqueeze(0)
    return normalize_c2w_to_first(c2w_t).detach()


def w2c_3x4_to_c2w(extrinsic: torch.Tensor) -> torch.Tensor:
    extrinsic = extrinsic.float()
    batch_shape = extrinsic.shape[:-2]
    bottom = torch.zeros(*batch_shape, 1, 4, device=extrinsic.device, dtype=extrinsic.dtype)
    bottom[..., 0, 3] = 1.0
    w2c = torch.cat([extrinsic, bottom], dim=-2)
    return torch.linalg.inv(w2c)


def pose_predictions_to_relative_c2w(
    preds: dict[str, torch.Tensor],
    image_hw: tuple[int, int],
) -> torch.Tensor:
    if "pose_enc" not in preds:
        raise RuntimeError("VGGT output did not contain pose_enc for pose-output attack loss.")
    extrinsic, _ = pose_encoding_to_extri_intri(preds["pose_enc"], image_hw)
    return normalize_c2w_to_first(w2c_3x4_to_c2w(extrinsic))


def forward_camera_pose_only(
    model: torch.nn.Module,
    images: torch.Tensor,
    dtype: torch.dtype,
    activation_checkpoint: bool = False,
) -> dict[str, torch.Tensor]:
    """Run only VGGT's aggregator and camera head for pose-loss attacks."""
    if model.camera_head is None:
        raise RuntimeError("VGGT model has no camera_head.")
    images_b = images.unsqueeze(0) if images.ndim == 4 else images
    with torch.cuda.amp.autocast(dtype=dtype):
        old_checkpoint = getattr(model.aggregator, "gradient_checkpointing", False)
        if hasattr(model.aggregator, "gradient_checkpointing"):
            model.aggregator.gradient_checkpointing = activation_checkpoint
        try:
            aggregated_tokens_list, _ = model.aggregator(images_b)
        finally:
            if hasattr(model.aggregator, "gradient_checkpointing"):
                model.aggregator.gradient_checkpointing = old_checkpoint
    with torch.cuda.amp.autocast(enabled=False):
        pose_enc_list = model.camera_head(aggregated_tokens_list)
    return {"pose_enc": pose_enc_list[-1], "pose_enc_list": pose_enc_list}


def make_track_query_points(image_hw: tuple[int, int], rows: int, cols: int,
                            margin: float, device: torch.device) -> torch.Tensor:
    """Deterministic first-frame queries shared by clean and attacked forwards."""
    h, w = image_hw
    margin = float(np.clip(margin, 0.0, 0.45))
    ys = torch.linspace(margin * (h - 1), (1.0 - margin) * (h - 1), rows,
                        device=device)
    xs = torch.linspace(margin * (w - 1), (1.0 - margin) * (w - 1), cols,
                        device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).float()


def forward_tracking_only(model: torch.nn.Module, images: torch.Tensor,
                          query_points: torch.Tensor, dtype: torch.dtype,
                          iters: int) -> dict[str, torch.Tensor]:
    if model.track_head is None:
        raise RuntimeError("VGGT model has no track_head")
    images_b = images if images.ndim == 5 else images[None]
    query_b = query_points if query_points.ndim == 3 else query_points[None]
    with autocast_context(images_b.device, dtype):
        tokens, patch_start_idx = model.aggregator(images_b)
    track_list, vis, conf = model.track_head(
        tokens, images=images_b, patch_start_idx=patch_start_idx,
        query_points=query_b, iters=iters)
    return {"track": track_list[-1], "track_vis": vis, "track_conf": conf}


def clean_track_targets(seq_name: str, images: torch.Tensor, image_hw: tuple[int, int],
                        model: torch.nn.Module, dtype: torch.dtype,
                        args: argparse.Namespace) -> dict[str, torch.Tensor]:
    """Cache the clean track-head output; a gauge leaves 2-D correspondences fixed.

    For every frame i, projecting ``g_i X`` through camera ``g_i T_i`` cancels
    ``g_i`` exactly.  The fourth head therefore cannot have a non-trivial gauge
    target: its correct role is to stay at the clean tracks while pose/point-map
    move.  Calling this an attacked tracking target would be mathematically false.
    """
    key = (seq_name, tuple(image_hw), args.joint_track_grid_rows,
           args.joint_track_grid_cols, args.joint_track_iters,
           float(args.joint_track_query_margin))
    cache = getattr(args, "_joint_track_target_cache", None)
    if cache is None:
        cache = {}
        args._joint_track_target_cache = cache
    if key not in cache:
        query = make_track_query_points(
            image_hw, args.joint_track_grid_rows, args.joint_track_grid_cols,
            args.joint_track_query_margin, images.device)
        with torch.no_grad():
            pred = forward_tracking_only(model, images, query, dtype, args.joint_track_iters)
        cache[key] = {
            "query_points": query.detach(),
            "track": pred["track"].detach(),
            "vis": pred["track_vis"].detach(),
            "conf": pred["track_conf"].detach() if pred["track_conf"] is not None else None,
        }
    return cache[key]


def normalize_c2w_to_first(c2w: torch.Tensor) -> torch.Tensor:
    first_inv = torch.linalg.inv(c2w[:, :1])
    return torch.matmul(first_inv, c2w)


def invert_relative_trajectory(rel_c2w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(rel_c2w)


def yaw_rotation_matrices(
    angles_rad: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cos = torch.cos(angles_rad).to(device=device, dtype=dtype)
    sin = torch.sin(angles_rad).to(device=device, dtype=dtype)
    mats = torch.zeros(*angles_rad.shape, 4, 4, device=device, dtype=dtype)
    mats[..., 0, 0] = cos
    mats[..., 0, 2] = sin
    mats[..., 1, 1] = 1.0
    mats[..., 2, 0] = -sin
    mats[..., 2, 2] = cos
    mats[..., 3, 3] = 1.0
    return mats


def make_pose_bad_target(rel_c2w: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float | str]]:
    target = rel_c2w.detach().clone()
    device = target.device
    dtype = target.dtype
    n_frames = target.shape[1]
    if n_frames <= 1:
        ramp = torch.zeros((1,), device=device, dtype=dtype)
    else:
        ramp = torch.linspace(0.0, 1.0, n_frames, device=device, dtype=dtype)

    meta: dict[str, float | str] = {"pose_reference": args.pose_bad_reference}
    if args.attack_loss == "pose_drift_targeted":
        drift = torch.zeros((n_frames, 3), device=device, dtype=dtype)
        drift[:, 0] = ramp * float(args.pose_drift_x_m)
        drift[:, 1] = ramp * float(args.pose_drift_y_m)
        drift[:, 2] = ramp * float(args.pose_drift_z_m)
        target[:, :, :3, 3] = target[:, :, :3, 3] + drift[None]
        if abs(float(args.pose_drift_yaw_degrees)) > 1e-8:
            angles = ramp * math.radians(float(args.pose_drift_yaw_degrees))
            yaw = yaw_rotation_matrices(angles, device=device, dtype=dtype)
            target = torch.matmul(target, yaw[None])
        meta.update(
            {
                "pose_target": "translation_drift",
                "pose_drift_x_m": float(args.pose_drift_x_m),
                "pose_drift_y_m": float(args.pose_drift_y_m),
                "pose_drift_z_m": float(args.pose_drift_z_m),
                "pose_drift_yaw_degrees": float(args.pose_drift_yaw_degrees),
            }
        )
        return target, meta

    if args.attack_loss == "pose_scale_targeted":
        target[:, :, :3, 3] = target[:, :, :3, 3] * float(args.pose_translation_scale)
        meta.update(
            {
                "pose_target": "translation_scale",
                "pose_translation_scale": float(args.pose_translation_scale),
            }
        )
        return target, meta

    if args.attack_loss == "pose_yaw_targeted":
        angles = ramp * math.radians(float(args.pose_yaw_degrees))
        yaw = yaw_rotation_matrices(angles, device=device, dtype=dtype)
        target = torch.matmul(target, yaw[None])
        meta.update(
            {
                "pose_target": "yaw_bias",
                "pose_yaw_degrees": float(args.pose_yaw_degrees),
            }
        )
        return target, meta

    raise ValueError(f"Unsupported bad pose target for attack_loss={args.attack_loss}")


def pose_relative_mse(
    pred_rel: torch.Tensor,
    target_rel: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_rel = target_rel.detach().to(device=pred_rel.device, dtype=pred_rel.dtype)
    if pred_rel.shape != target_rel.shape:
        raise RuntimeError(f"Pose target shape mismatch: pred={tuple(pred_rel.shape)} target={tuple(target_rel.shape)}")
    rot_diff = pred_rel[..., :3, :3] - target_rel[..., :3, :3]
    trans_diff = pred_rel[..., :3, 3] - target_rel[..., :3, 3]
    rot_mse = rot_diff.pow(2).mean()
    trans_scale = target_rel[..., :3, 3].norm(dim=-1).mean().clamp_min(1e-3)
    trans_mse = (trans_diff / trans_scale).pow(2).mean()
    total = args.pose_rotation_weight * rot_mse + args.pose_translation_weight * trans_mse
    return total, {
        "pose_loss": float(total.detach().cpu()),
        "pose_rot_mse": float(rot_mse.detach().cpu()),
        "pose_trans_mse": float(trans_mse.detach().cpu()),
        "pose_trans_scale": float(trans_scale.detach().cpu()),
    }


def pose_scale_invariant_mse(
    pred_rel: torch.Tensor,
    target_rel: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Scale-invariant variant of pose_relative_mse.

    pose_relative_mse divides both trajectories by one constant taken from the
    *target* (mean GT relative-translation norm). That fixes the unit but leaves
    the global pred/GT scale difference inside the loss. VGGT's reconstruction
    scale is arbitrary per sequence (measured 2.07..4.15 across TUM-10 by
    scripts/diag_gauge_invariance.py), and the evaluator aligns with Sim(3)
    before measuring, so that residual scale term is gradient the metric throws
    away.

    Here each trajectory is normalized by its *own* relative-translation RMS
    before the comparison, which removes the scale gauge from the loss. The
    rotation term is deliberately untouched: normalize_c2w_to_first already
    makes the relative rotations invariant to a global rotation and translation.

    Side effect worth knowing: after normalization both translation fields have
    unit RMS, so the translation term is bounded by 4/3 instead of growing
    without limit.
    """
    target_rel = target_rel.detach().to(device=pred_rel.device, dtype=pred_rel.dtype)
    if pred_rel.shape != target_rel.shape:
        raise RuntimeError(f"Pose target shape mismatch: pred={tuple(pred_rel.shape)} target={tuple(target_rel.shape)}")
    eps = float(getattr(args, "pose_scale_invariant_eps", 1e-6))

    rot_diff = pred_rel[..., :3, :3] - target_rel[..., :3, :3]
    rot_mse = rot_diff.pow(2).mean()

    pred_trans = pred_rel[..., :3, 3]
    target_trans = target_rel[..., :3, 3]
    # RMS over the frame axis, keeping every leading (batch) axis independent so
    # that a batch of sequences with different scales is still handled per-sequence.
    pred_rms = pred_trans.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True).clamp_min(eps * eps).sqrt()
    target_rms = target_trans.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True).clamp_min(eps * eps).sqrt()
    trans_diff = pred_trans / pred_rms.unsqueeze(-1) - target_trans / target_rms.unsqueeze(-1)
    trans_mse = trans_diff.pow(2).mean()

    total = args.pose_rotation_weight * rot_mse + args.pose_translation_weight * trans_mse
    return total, {
        "pose_loss": float(total.detach().cpu()),
        "pose_rot_mse": float(rot_mse.detach().cpu()),
        "pose_trans_mse": float(trans_mse.detach().cpu()),
        "pose_pred_trans_rms": float(pred_rms.detach().mean().cpu()),
        "pose_target_trans_rms": float(target_rms.detach().mean().cpu()),
    }


def pairwise_relative_c2w(rel: torch.Tensor) -> torch.Tensor:
    """T_ij = T_i^{-1} T_j for every unordered pair i<j.

    (..., N, 4, 4) -> (..., N*(N-1)/2, 4, 4). For the 10-frame TUM setting that
    is 45 pairs.
    """
    n = rel.shape[-3]
    pairs = torch.triu_indices(n, n, offset=1, device=rel.device)
    left = torch.linalg.inv(rel[..., pairs[0], :, :])
    return torch.matmul(left, rel[..., pairs[1], :, :])


def pose_pairwise_relative_mse(
    pred_rel: torch.Tensor,
    target_rel: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pairwise relative-pose loss: no privileged frame.

    `pose_relative_mse` and `pose_scale_invariant_mse` both measure everything
    relative to frame 0, which gives frame 0 leverage the metric does not grant
    it: perturbing T_0 alone sends every relative pose to delta^-1 * rel_i, so all
    N-1 terms move at once, while the Sim(3)-aligned ATE just sees one outlier
    camera. Comparing all N(N-1)/2 pairs instead removes that asymmetry -- a
    corrupted frame 0 now touches only the N-1 pairs that contain it (9 of 45
    here) instead of all of them.

    Gauge behaviour, for g = (s*R_g, t_g) acting on the trajectory:
        T_i'^-1 T_j' = [ R_i^T R_j | s * R_i^T (p_j - p_i) ]
    so the rotation block is exactly invariant to the whole of Sim(3), and the
    translation block only picks up the scale s, which the per-trajectory RMS
    normalisation below removes -- same treatment as pose_scale_invariant_mse,
    so the scale leak is not reintroduced.
    """
    target_rel = target_rel.detach().to(device=pred_rel.device, dtype=pred_rel.dtype)
    if pred_rel.shape != target_rel.shape:
        raise RuntimeError(f"Pose target shape mismatch: pred={tuple(pred_rel.shape)} target={tuple(target_rel.shape)}")
    if pred_rel.shape[-3] < 2:
        raise RuntimeError("pose_pairwise_relative_mse needs at least 2 frames")
    eps = float(getattr(args, "pose_scale_invariant_eps", 1e-6))

    pred_pair = pairwise_relative_c2w(pred_rel)
    target_pair = pairwise_relative_c2w(target_rel)

    rot_mse = (pred_pair[..., :3, :3] - target_pair[..., :3, :3]).pow(2).mean()

    pred_trans = pred_pair[..., :3, 3]
    target_trans = target_pair[..., :3, 3]
    pred_rms = pred_trans.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True).clamp_min(eps * eps).sqrt()
    target_rms = target_trans.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True).clamp_min(eps * eps).sqrt()
    trans_mse = (pred_trans / pred_rms.unsqueeze(-1) - target_trans / target_rms.unsqueeze(-1)).pow(2).mean()

    total = args.pose_rotation_weight * rot_mse + args.pose_translation_weight * trans_mse
    return total, {
        "pose_loss": float(total.detach().cpu()),
        "pose_rot_mse": float(rot_mse.detach().cpu()),
        "pose_trans_mse": float(trans_mse.detach().cpu()),
        "pose_n_pairs": int(pred_pair.shape[-3]),
        "pose_pred_trans_rms": float(pred_rms.detach().mean().cpu()),
        "pose_target_trans_rms": float(target_rms.detach().mean().cpu()),
    }


def umeyama_sim3_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Least-squares Sim(3) with  c * x @ R^T + t  ~=  y.  x, y: (..., N, 3).

    Same estimator as `mv_recon/utils.py::umeyama` and the one evo uses inside
    `align(correct_scale=True)`, written in torch so it can sit inside a loss.
    """
    n = x.shape[-2]
    mu_x = x.mean(dim=-2, keepdim=True)
    mu_y = y.mean(dim=-2, keepdim=True)
    xc = x - mu_x
    yc = y - mu_y
    var_x = xc.pow(2).sum(dim=-1).mean(dim=-1)
    cov = torch.matmul(yc.transpose(-1, -2), xc) / n
    u, d, vh = torch.linalg.svd(cov)
    sign = torch.sign(torch.det(u) * torch.det(vh))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    s_diag = torch.ones_like(d)
    s_diag[..., -1] = sign
    rot = torch.matmul(torch.matmul(u, torch.diag_embed(s_diag)), vh)
    scale = (d * s_diag).sum(dim=-1) / var_x.clamp_min(eps * eps)
    trans = mu_y - scale[..., None, None] * torch.matmul(mu_x, rot.transpose(-1, -2))
    return scale, rot, trans


def pose_aligned_residual_mse(
    pred_rel: torch.Tensor,
    target_rel: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Differentiable ATE: residual measured *after* the evaluator's own alignment.

    The evaluator fits the best Sim(3) before it measures anything, so a loss that
    scores residuals in a fixed gauge is optimising a different quantity. This one
    reproduces the alignment step: fit Umeyama(pred -> target) exactly as
    `evo.core.geometry.umeyama_alignment(..., with_scale=True)` does, apply it, and
    take the RMS residual. It is then divided by the target's own RMS radius, which
    is the largest residual any prediction can produce (the alignment can always
    collapse the prediction to a point, and Umeyama *minimises* the residual). So
    the translation term equals ATE / ate_ceiling and lives in [0, 1]: it reaches 1
    exactly when the ATE is maximal, unlike the fixed-gauge losses whose ceiling
    sits at an arbitrary point unrelated to the metric.

    The alignment parameters are solved under `no_grad` on purpose. They minimise
    the very residual being differentiated, so by the envelope theorem their
    contribution to dL/d(pred) vanishes at the optimum, and detaching them is
    first-order exact while completely avoiding the ill-conditioned SVD gradient
    when the covariance has near-equal singular values.
    `tests/test_pose_gauge_losses.py` checks this against full autograd.
    """
    target_rel = target_rel.detach().to(device=pred_rel.device, dtype=pred_rel.dtype)
    if pred_rel.shape != target_rel.shape:
        raise RuntimeError(f"Pose target shape mismatch: pred={tuple(pred_rel.shape)} target={tuple(target_rel.shape)}")
    eps = float(getattr(args, "pose_scale_invariant_eps", 1e-6))

    pred_pos = pred_rel[..., :3, 3]
    target_pos = target_rel[..., :3, 3]

    with torch.no_grad():
        scale, rot, trans = umeyama_sim3_torch(pred_pos, target_pos, eps)

    aligned = scale[..., None, None] * torch.matmul(pred_pos, rot.transpose(-1, -2)) + trans
    resid_ms = (aligned - target_pos).pow(2).sum(dim=-1).mean(dim=-1)
    rmse = resid_ms.clamp_min(0.0).sqrt()

    target_centred = target_pos - target_pos.mean(dim=-2, keepdim=True)
    ceiling = target_centred.pow(2).sum(dim=-1).mean(dim=-1).clamp_min(eps * eps).sqrt()
    trans_term = (rmse / ceiling).mean()

    # rotation measured in the same aligned gauge, so it is Sim(3)-invariant too
    aligned_rot = torch.matmul(rot.unsqueeze(-3), pred_rel[..., :3, :3])
    rot_mse = (aligned_rot - target_rel[..., :3, :3]).pow(2).mean()

    total = args.pose_rotation_weight * rot_mse + args.pose_translation_weight * trans_term
    return total, {
        "pose_loss": float(total.detach().cpu()),
        "pose_rot_mse": float(rot_mse.detach().cpu()),
        "pose_trans_mse": float(trans_term.detach().cpu()),
        "pose_aligned_ate": float(rmse.detach().mean().cpu()),
        "pose_ate_ceiling": float(ceiling.detach().mean().cpu()),
        "pose_align_scale": float(scale.detach().mean().cpu()),
    }


def sim3_action_basis(positions: np.ndarray) -> np.ndarray:
    """The 7 displacement directions one Sim(3) can produce, as (7, 3N).

    Three translations, three rotations about the centroid, one scale. Any target
    displacement lying inside this span is erased by the evaluator's alignment no
    matter how large it is, which is why constant gauges score exactly zero.
    """
    cols = []
    for k in range(3):
        d = np.zeros_like(positions); d[:, k] = 1.0
        cols.append(d.reshape(-1))
    centred = positions - positions.mean(0)
    for k in range(3):
        axis = np.zeros(3); axis[k] = 1.0
        cols.append(np.cross(axis[None, :], centred).reshape(-1))
    cols.append(centred.reshape(-1))
    return np.stack(cols)


def orthogonal_mode_displacement(positions: np.ndarray, order: int, axis: int,
                                 magnitude: float) -> np.ndarray:
    """Low-frequency camera displacement with the Sim(3) span projected out.

    The hand-built families saturate because raising their magnitude scales the
    absorbed and surviving parts together, leaving the ratio fixed: trans_ramp
    stops at 76% of the ATE ceiling and scale_sine never grows at all. Choosing the
    direction instead of the size fixes that. Take a cosine profile over the frame
    index, subtract its projection onto the seven Sim(3) directions, and what is
    left cannot be absorbed at all.

    Measured in scripts/diag_optimal_gauge.py: order 2 reaches 89-93% of ceiling on
    all three sequences at a frame-to-frame jump of 0.30-0.37, against 68-76% for
    the best hand-built family at a comparable jump. Higher orders are slightly
    less absorbable but noticeably less smooth (order 5 jumps 0.69-0.77), and the
    ATE saturates either way, so order 2 is the useful setting.
    """
    n = positions.shape[0]
    basis = sim3_action_basis(positions)
    q, _ = np.linalg.qr(basis.T)
    profile = np.cos(np.pi * order * np.arange(n, dtype=np.float64) / max(n - 1, 1))
    field = np.zeros((n, 3), dtype=np.float64)
    field[:, axis] = profile
    flat = field.reshape(-1)
    perp = flat - q @ (q.T @ flat)
    norm = float(np.linalg.norm(perp))
    if norm < 1e-12:
        raise ValueError(f"orthogonal mode order={order} axis={axis} is fully absorbed "
                         "by Sim(3); pick a different order or axis")
    delta = (perp / norm * math.sqrt(n)).reshape(n, 3)
    radius = float(np.sqrt(((positions - positions.mean(0)) ** 2).sum(1).mean()))
    return magnitude * radius * delta


def piecewise_gauge_schedule(family: str, n: int, magnitude: float,
                             positions: np.ndarray | None = None,
                             order: int = 2, axis: int = 0
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame similarity (scale_i, yaw_i in degrees, translation_i).

    A constant gauge is removed exactly by the evaluator's single Sim(3) fit and
    scores zero ATE no matter how large it is -- verified up to 109x the scene
    radius in scripts/diag_piecewise_gauge.py. Only a gauge that *varies* with the
    frame leaves anything behind. These are the families measured there; the
    schedules must stay in sync with that script or the targets stop matching the
    feasibility numbers.
    """
    idx = np.arange(n, dtype=np.float64)
    span = max(n - 1, 1)
    ones, zeros = np.ones(n), np.zeros(n)
    zero_t = np.zeros((n, 3))
    unit = np.asarray([1.0, 1.0, 1.0]) / math.sqrt(3.0)

    if family == "scale_ramp":
        return 1.0 + magnitude * idx / span, zeros, zero_t
    if family == "scale_sine":
        return 1.0 + magnitude * np.sin(2 * math.pi * idx / span), zeros, zero_t
    if family == "scale_split":
        return np.where(idx < n // 2, 1.0, 1.0 + magnitude), zeros, zero_t
    if family == "yaw_ramp":
        return ones, 30.0 * magnitude * idx / span, zero_t
    if family == "yaw_sine":
        return ones, 30.0 * magnitude * np.sin(2 * math.pi * idx / span), zero_t
    if family == "trans_ramp":
        return ones, zeros, np.outer(magnitude * idx / span, unit)
    if family == "trans_sine":
        return ones, zeros, np.outer(magnitude * np.sin(2 * math.pi * idx / span), unit)
    if family == "orthogonal_mode":
        if positions is None:
            raise ValueError("orthogonal_mode needs the trajectory positions")
        # A pure per-frame translation, so the joint-head derivation still holds:
        # pose shifts by delta_i, depth is unchanged, the point cloud shifts with it.
        return ones, zeros, orthogonal_mode_displacement(positions, order, axis, magnitude)
    raise ValueError(f"Unknown piecewise gauge family: {family}")


def apply_piecewise_gauge(c2w: np.ndarray, family: str, magnitude: float,
                          order: int = 2, axis: int = 0) -> np.ndarray:
    """Act on a camera-to-world trajectory with a per-frame similarity.

    Applied in the world frame, matching scripts/diag_piecewise_gauge.py. Applying
    it after first-frame normalisation instead would be a different target, because
    a per-frame gauge does not commute with the change of reference frame.
    """
    scales, yaws, trans = piecewise_gauge_schedule(
        family, c2w.shape[0], magnitude, positions=c2w[:, :3, 3], order=order, axis=axis)
    out = np.array(c2w, dtype=np.float64, copy=True)
    for i in range(c2w.shape[0]):
        rot = quat_xyzw_to_rot(0.0, math.sin(math.radians(yaws[i]) / 2.0), 0.0,
                               math.cos(math.radians(yaws[i]) / 2.0))
        out[i, :3, :3] = rot @ c2w[i, :3, :3]
        out[i, :3, 3] = scales[i] * (rot @ c2w[i, :3, 3]) + trans[i]
    return out


def build_joint_gauge_targets(
    seq_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    """One per-frame gauge, expressed as a target for every geometry head.

    Applying g_i: X -> s_i R_i X + t_i to the world as frame i sees it moves each
    head in a determined way. A world point at camera-frame depth d appears at
    T_i'^-1 g_i(X) = s_i (R_i^c)^T (X - p_i), i.e. the camera-frame coordinates
    scale by s_i and nothing else changes, so

        pose_i  -> g_i . T_i          depth_i -> s_i . d_i
        point_i -> g_i(P_i)

    These are built from the *clean prediction*, not from ground truth, so that the
    three targets inherit the model's own head agreement rather than introducing a
    fresh inconsistency: the clean prediction already disagrees between depth and
    point heads by ~0.9% of the scene radius, and g_i carries that along scaled by
    s_i while the radius scales the same way. Verified: the relative disagreement
    of the target is 0.50-0.99x the clean value across families and magnitudes, so
    an attack that lands on this target is no more detectable by a cross-head check
    than the clean output is.
    """
    if not args.clean_vggt_output_root or not args.piecewise_gauge_family:
        return None
    npz_path = Path(args.clean_vggt_output_root) / seq_name / "vggt_outputs.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path) as data:
        if not {"extrinsic", "depth", "point_map"} <= set(data.files):
            return None
        clean_c2w = extrinsics_to_c2w(np.asarray(data["extrinsic"]))
        clean_depth = np.asarray(data["depth"], dtype=np.float64)
        clean_points = np.asarray(data["point_map"], dtype=np.float64)
        clean_depth_conf = (np.asarray(data["depth_conf"], dtype=np.float64)
                            if "depth_conf" in data else None)
        clean_point_conf = (np.asarray(data["point_conf"], dtype=np.float64)
                            if "point_conf" in data else None)
    if clean_depth.ndim == 4 and clean_depth.shape[-1] == 1:
        clean_depth = clean_depth[..., 0]

    n = clean_c2w.shape[0]
    scales, yaws, trans = piecewise_gauge_schedule(
        args.piecewise_gauge_family, n, args.piecewise_gauge_magnitude,
        positions=clean_c2w[:, :3, 3],
        order=args.orthogonal_mode_order, axis=args.orthogonal_mode_axis)

    pose_t = np.array(clean_c2w, dtype=np.float64, copy=True)
    depth_t = np.array(clean_depth, dtype=np.float64, copy=True)
    points_t = np.array(clean_points, dtype=np.float64, copy=True)
    for i in range(n):
        rot = quat_xyzw_to_rot(0.0, math.sin(math.radians(yaws[i]) / 2.0), 0.0,
                               math.cos(math.radians(yaws[i]) / 2.0))
        pose_t[i, :3, :3] = rot @ clean_c2w[i, :3, :3]
        pose_t[i, :3, 3] = scales[i] * (rot @ clean_c2w[i, :3, 3]) + trans[i]
        depth_t[i] = scales[i] * clean_depth[i]
        points_t[i] = scales[i] * (clean_points[i] @ rot.T) + trans[i]

    targets = {
        "pose_rel": c2w_numpy_to_relative_tensor(pose_t, device),
        "depth": torch.from_numpy(depth_t).to(device=device, dtype=torch.float32),
        "points": torch.from_numpy(points_t).to(device=device, dtype=torch.float32),
    }
    # Confidence is a property of the prediction, not a geometric coordinate, so
    # a gauge transformation must leave it unchanged. Preserve it explicitly:
    # otherwise a head can reduce confidence while approaching the geometry target.
    if clean_depth_conf is not None:
        targets["depth_conf"] = torch.from_numpy(clean_depth_conf).to(
            device=device, dtype=torch.float32)
    if clean_point_conf is not None:
        targets["point_conf"] = torch.from_numpy(clean_point_conf).to(
            device=device, dtype=torch.float32)
    return targets


def scale_invariant_depth_loss(pred: torch.Tensor, target: torch.Tensor,
                               eps: float = 1e-6) -> torch.Tensor:
    """Depth distance that ignores one global scale, because the evaluator does.

    `utils/depth.py::depth_evaluation` fits a single scale (median or IRLS) before
    scoring, so a uniform factor on the prediction is invisible to it and chasing
    one would be the same wasted gradient the pose losses used to spend. Dividing
    each side by its own median leaves only the per-frame pattern, which is what a
    frame-varying gauge actually asks for.
    """
    pred_med = pred.detach().flatten().median().clamp_min(eps)
    tgt_med = target.flatten().median().clamp_min(eps)
    p = pred / pred_med
    t = target / tgt_med
    return ((p - t).abs() / t.clamp_min(eps)).mean()


def confidence_preservation_loss(pred: torch.Tensor, target: torch.Tensor,
                                 eps: float = 1e-6) -> torch.Tensor:
    """Preserve clean head confidence without letting extreme pixels dominate."""
    target = target.detach().to(device=pred.device, dtype=pred.dtype)
    if pred.shape != target.shape:
        raise RuntimeError(
            f"Confidence target shape mismatch: pred={tuple(pred.shape)} "
            f"target={tuple(target.shape)}")
    pred_log = pred.float().clamp_min(eps).log()
    target_log = target.float().clamp_min(eps).log()
    return F.smooth_l1_loss(pred_log, target_log)


def dense_scalar_head(tensor: torch.Tensor) -> torch.Tensor:
    """Drop VGGT's optional batch and singleton-channel axes."""
    if tensor.ndim == 5:
        tensor = tensor[0]
    # Confidence heads return [B,T,H,W], unlike depth's [B,T,H,W,1].
    if tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[-1] != 1:
        tensor = tensor[0]
    if tensor.ndim == 4 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    return tensor


def batched_scalar_head(tensor: torch.Tensor) -> torch.Tensor:
    """Return a scalar VGGT head as [B,T,H,W] without losing batch semantics."""
    if tensor.ndim == 5 and tensor.shape[-1] == 1:
        return tensor[..., 0]
    if tensor.ndim == 4 and tensor.shape[-1] == 1:
        return tensor[..., 0].unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    raise RuntimeError(f"Expected scalar head with 3-5 dims, got {tuple(tensor.shape)}")


def differentiable_median(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """NumPy-compatible median (average the middle pair for even lengths)."""
    ordered = values.sort(dim=dim).values
    n = ordered.shape[dim]
    if n == 0:
        raise RuntimeError("median of an empty tensor")
    if n % 2:
        return ordered.select(dim, n // 2)
    return 0.5 * (ordered.select(dim, n // 2 - 1) + ordered.select(dim, n // 2))


def torch_unproject_depth_world(depth: torch.Tensor, extrinsic: torch.Tensor,
                                intrinsic: torch.Tensor) -> torch.Tensor:
    """Differentiable equivalent of VGGT's official depth-to-world export."""
    b, t, h, w = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype), indexing="ij")
    xs = xs.view(1, 1, h, w)
    ys = ys.view(1, 1, h, w)
    fx = intrinsic[..., 0, 0, None, None]
    fy = intrinsic[..., 1, 1, None, None]
    cx = intrinsic[..., 0, 2, None, None]
    cy = intrinsic[..., 1, 2, None, None]
    cam = torch.stack(((xs - cx) * depth / fx.clamp_min(1e-6),
                       (ys - cy) * depth / fy.clamp_min(1e-6), depth), dim=-1)
    c2w = w2c_3x4_to_c2w(extrinsic)
    world = torch.einsum("bthwj,btij->bthwi", cam, c2w[..., :3, :3])
    return world + c2w[..., None, None, :3, 3]


def differentiable_reprojection_metric(depth: torch.Tensor, extrinsic: torch.Tensor,
                                       intrinsic: torch.Tensor, stride: int = 4,
                                       eps: float = 1e-6) -> torch.Tensor:
    """Adjacent-frame depth reprojection with bilinear sampling for pose gradients."""
    b, t, h, w = depth.shape
    c2w = w2c_3x4_to_c2w(extrinsic)
    ys, xs = torch.meshgrid(
        torch.arange(0, h, stride, device=depth.device, dtype=depth.dtype),
        torch.arange(0, w, stride, device=depth.device, dtype=depth.dtype), indexing="ij")
    xs = xs[None]
    ys = ys[None]
    pair_errors = []
    for i in range(t - 1):
        j = i + 1
        z = depth[:, i, ::stride, ::stride]
        ki, kj = intrinsic[:, i], intrinsic[:, j]
        x = (xs - ki[:, 0, 2, None, None]) * z / ki[:, 0, 0, None, None].clamp_min(eps)
        y = (ys - ki[:, 1, 2, None, None]) * z / ki[:, 1, 1, None, None].clamp_min(eps)
        cam_i = torch.stack((x, y, z), dim=-1)
        world = torch.einsum("bhwj,bij->bhwi", cam_i, c2w[:, i, :3, :3])
        world = world + c2w[:, i, None, None, :3, 3]
        cam_j = torch.einsum("bhwj,bij->bhwi", world, extrinsic[:, j, :3, :3])
        cam_j = cam_j + extrinsic[:, j, None, None, :3, 3]
        zj = cam_j[..., 2]
        u = kj[:, 0, 0, None, None] * cam_j[..., 0] / zj.clamp_min(eps) + kj[:, 0, 2, None, None]
        v = kj[:, 1, 1, None, None] * cam_j[..., 1] / zj.clamp_min(eps) + kj[:, 1, 2, None, None]
        grid = torch.stack((2.0 * u / max(w - 1, 1) - 1.0,
                            2.0 * v / max(h - 1, 1) - 1.0), dim=-1)
        sampled = F.grid_sample(depth[:, j, None], grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)[:, 0]
        # Visibility is a discrete selection rule in the evaluator. Detaching the
        # mask keeps the selected support fixed while retaining gradients through
        # projected depth and sub-pixel coordinates.
        valid = ((z.detach() > eps) & (zj.detach() > eps) &
                 (u.detach() >= 0) & (u.detach() < w - 1) &
                 (v.detach() >= 0) & (v.detach() < h - 1) &
                 (sampled.detach() > eps))
        err = (zj - sampled).abs() / sampled.clamp_min(eps)
        if valid.any():
            pair_errors.append(differentiable_median(err[valid].flatten()))
    if not pair_errors:
        return depth.new_zeros(())
    return differentiable_median(torch.stack(pair_errors))


def differentiable_filter_metrics(preds: dict, item: dict,
                                  args: argparse.Namespace) -> dict[str, torch.Tensor]:
    """Differentiable sequence summaries matching eval_output_filters.py."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], item["tensor_hw"])
    extrinsic = extrinsic.float()
    intrinsic = intrinsic.float()
    depth = batched_scalar_head(preds["depth"]).float()
    conf = batched_scalar_head(preds["depth_conf"]).float()
    points = preds["world_points"].float()
    if points.ndim == 4:
        points = points.unsqueeze(0)

    flat_conf = conf.flatten(2)
    conf_std = differentiable_median(flat_conf.std(dim=2, unbiased=False), dim=1).mean()
    floor = conf.new_tensor(float(item["filter_budget"]["depth_conf_floor"]))
    temperature = max(float(args.budget_conf_floor_temperature), 1e-6)
    soft_floor = torch.sigmoid((floor - conf) / temperature)
    hard_floor = (conf <= floor).to(conf.dtype)
    floor_st = hard_floor.detach() + soft_floor - soft_floor.detach()
    conf_frac = differentiable_median(floor_st.flatten(2).mean(dim=2), dim=1).mean()

    unprojected = torch_unproject_depth_world(depth, extrinsic, intrinsic)
    diff = (points - unprojected).norm(dim=-1).flatten(2)
    center = points.flatten(2, 3).mean(dim=2, keepdim=True)
    radius = (points.flatten(2, 3) - center).pow(2).sum(-1).mean(-1).clamp_min(1e-12).sqrt()
    head = differentiable_median(
        differentiable_median(diff, dim=2) / radius, dim=1).mean()
    reproj = differentiable_reprojection_metric(
        depth, extrinsic, intrinsic, stride=args.budget_reproj_stride)
    return {
        "conf_std": conf_std,
        "conf_frac_floor": conf_frac,
        "head_disagree_rel": head,
        "reproj_rel_err": reproj,
    }


def load_filter_budget(seq_name: str, args: argparse.Namespace) -> dict:
    path = Path(args.filter_budget_json)
    cache = getattr(args, "_filter_budget_cache", None)
    if cache is None:
        if not path.exists():
            raise RuntimeError(f"filter budget JSON not found: {path}")
        cache = json.loads(path.read_text(encoding="utf-8"))
        args._filter_budget_cache = cache
    try:
        return cache["sequences"][seq_name]
    except KeyError as exc:
        raise RuntimeError(f"no matched filter budget for sequence {seq_name}") from exc


def budget_safe_value_and_violation(value: torch.Tensor, spec: dict,
                                    fraction: float, eps: float = 1e-8
                                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an empirical clean-to-threshold interval into a normalized hinge."""
    clean = value.new_tensor(float(spec["clean"]))
    threshold = value.new_tensor(float(spec["threshold"]))
    fraction = float(np.clip(fraction, 0.0, 1.0))
    safe = clean + fraction * (threshold - clean)
    span = (threshold - clean).abs().clamp_min(eps)
    if spec["worse"] == "high":
        violation = F.relu((value - safe) / span)
    elif spec["worse"] == "low":
        violation = F.relu((safe - value) / span)
    else:
        raise ValueError(f"Unknown budget direction: {spec['worse']}")
    return safe, violation


def budget_track_error_pixels(preds: dict, target: dict, min_visibility: float) -> torch.Tensor:
    mask = target["vis"][:, 1:] >= min_visibility
    error = (preds["track"][:, 1:] - target["track"][:, 1:]).pow(2).sum(-1).sqrt()
    return error[mask].mean() if mask.any() else error.mean()


def aligned_point_residual(pred: torch.Tensor, target: torch.Tensor,
                           stride: int = 4, eps: float = 1e-6) -> torch.Tensor:
    """Point-map distance measured after the Sim(3) that mv_recon's eval would fit.

    `mv_recon/eval.py` runs Umeyama then ICP before computing Acc/Comp, so the
    global placement of the point cloud is not part of the score. Aligning here
    first means the loss only sees the part of the disagreement that survives.
    Sub-sampled because the full map is 2M points per sequence.
    """
    p = pred.reshape(-1, 3)[::stride]
    t = target.reshape(-1, 3)[::stride]
    ok = torch.isfinite(p).all(-1) & torch.isfinite(t).all(-1)
    p, t = p[ok], t[ok]
    if p.shape[0] < 16:
        return pred.new_zeros(())
    with torch.no_grad():
        scale, rot, trans = umeyama_sim3_torch(p, t, eps)
    aligned = scale * (p @ rot.transpose(-1, -2)) + trans
    radius = (t - t.mean(0, keepdim=True)).pow(2).sum(-1).mean().clamp_min(eps * eps).sqrt()
    return (aligned - t).pow(2).sum(-1).mean().clamp_min(0.0).sqrt() / radius


def forward_all_geometry_heads(
    model: torch.nn.Module,
    images: torch.Tensor,
    dtype: torch.dtype,
    query_points: torch.Tensor | None = None,
    track_iters: int = 4,
) -> dict[str, torch.Tensor]:
    """Geometry heads in one pass, with memory-heavy heads checkpointed.

    Running the whole model the way forward_vggt does needs more memory than the
    camera-only path and overflows a 46 GiB card at 10 frames: the DPT decoders keep
    full-resolution activations for both depth and point maps on top of the
    aggregator's. Checkpointing recomputes those two heads during the backward pass
    instead of storing them, which is numerically identical and brings the peak back
    under the card. The aggregator itself is shared and runs once either way.
    """
    images_b = images if images.ndim == 5 else images[None]
    # The aggregator's own activations are the bulk of the cost -- the camera-only
    # path already sits near 32 GiB with them -- so checkpointing only the two DPT
    # heads is not enough and still overflows. Checkpointing the aggregator as well
    # trades one recomputation in the backward pass for the memory, and
    # patch_start_idx is a fixed model attribute (1 + num_register_tokens) rather
    # than something the forward discovers, so nothing is lost by dropping it from
    # the checkpointed return.
    patch_start_idx = model.aggregator.patch_start_idx

    def run_aggregator(imgs):
        with autocast_context(imgs.device, dtype):
            tokens_list, _ = model.aggregator(imgs)
        return tuple(tokens_list)

    tokens = list(torch.utils.checkpoint.checkpoint(
        run_aggregator, images_b, use_reentrant=False))

    out: dict[str, torch.Tensor] = {}
    with torch.cuda.amp.autocast(enabled=False):
        out["pose_enc"] = model.camera_head(tokens)[-1]

        def run_head(head):
            def fn(*tensors):
                return head(list(tensors), images=images_b, patch_start_idx=patch_start_idx)
            return torch.utils.checkpoint.checkpoint(fn, *tokens, use_reentrant=False)

        depth, depth_conf = run_head(model.depth_head)
        out["depth"], out["depth_conf"] = depth, depth_conf
        points, points_conf = run_head(model.point_head)
        out["world_points"], out["world_points_conf"] = points, points_conf

        if query_points is not None:
            if model.track_head is None:
                raise RuntimeError("joint track loss requested but model has no track_head")
            query_b = query_points if query_points.ndim == 3 else query_points[None]

            def run_track(*tensors):
                coord_list, vis, conf = model.track_head(
                    list(tensors), images=images_b, patch_start_idx=patch_start_idx,
                    query_points=query_b, iters=track_iters)
                return coord_list[-1], vis, conf

            track, vis, conf = torch.utils.checkpoint.checkpoint(
                run_track, *tokens, use_reentrant=False)
            out["track"], out["track_vis"], out["track_conf"] = track, vis, conf
    return out


def track_consistency_loss(pred_track: torch.Tensor, target_track: torch.Tensor,
                           pred_vis: torch.Tensor, target_vis: torch.Tensor,
                           pred_conf: torch.Tensor | None, target_conf: torch.Tensor | None,
                           image_hw: tuple[int, int], min_visibility: float,
                           visibility_weight: float, confidence_weight: float,
                           eps: float = 1e-6) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep gauge-invariant 2-D tracks and their confidence at the clean output."""
    h, w = image_hw
    diagonal = math.sqrt(float(h * h + w * w))
    # Frame zero is forcibly equal to the query by the track head, so including it
    # would dilute the useful temporal constraint.
    mask = target_vis[:, 1:] >= min_visibility
    coord_error = (pred_track[:, 1:] - target_track[:, 1:]).pow(2).sum(-1).sqrt()
    if mask.any():
        coord = coord_error[mask].mean() / max(diagonal, eps)
    else:
        coord = coord_error.mean() / max(diagonal, eps)
    vis = F.smooth_l1_loss(pred_vis[:, 1:], target_vis[:, 1:])
    conf = pred_track.new_zeros(())
    if pred_conf is not None and target_conf is not None:
        conf = F.smooth_l1_loss(pred_conf[:, 1:], target_conf[:, 1:])
    total = coord + visibility_weight * vis + confidence_weight * conf
    return total, {"track_coord": coord, "track_vis": vis, "track_conf": conf}


def geometry_joint_gauge_loss(
    preds: dict,
    item: dict,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Push pose, depth and point map at one per-frame gauge target together."""
    targets = item.get("joint_gauge_targets")
    if targets is None:
        raise RuntimeError("geometry_joint_gauge_targeted needs --piecewise_gauge_family "
                           "and --clean_vggt_output_root")
    pred_rel = pose_predictions_to_relative_c2w(preds, item["tensor_hw"])
    pose_loss, pose_terms = pose_aligned_residual_mse(pred_rel, targets["pose_rel"], args)

    depth_pred = dense_scalar_head(preds["depth"])
    depth_loss = scale_invariant_depth_loss(depth_pred.float(), targets["depth"])

    depth_conf_loss = pose_loss.new_zeros(())
    if args.joint_depth_conf_weight > 0:
        if "depth_conf" not in targets:
            raise RuntimeError("joint_depth_conf_weight > 0 needs clean depth_conf")
        depth_conf_loss = confidence_preservation_loss(
            dense_scalar_head(preds["depth_conf"]),
            dense_scalar_head(targets["depth_conf"]),
        )

    points_pred = preds["world_points"]
    if points_pred.ndim == 5:
        points_pred = points_pred[0]
    point_loss = aligned_point_residual(points_pred.float(), targets["points"],
                                        args.joint_point_stride)

    point_conf_loss = pose_loss.new_zeros(())
    if args.joint_point_conf_weight > 0:
        if "point_conf" not in targets:
            raise RuntimeError("joint_point_conf_weight > 0 needs clean point_conf")
        point_conf_loss = confidence_preservation_loss(
            dense_scalar_head(preds["world_points_conf"]),
            dense_scalar_head(targets["point_conf"]),
        )

    track_loss = pose_loss.new_zeros(())
    track_terms = {"track_coord": track_loss, "track_vis": track_loss,
                   "track_conf": track_loss}
    if args.joint_track_weight > 0:
        track_target = item.get("joint_track_targets")
        if track_target is None or "track" not in preds:
            raise RuntimeError("joint_track_weight > 0 needs cached clean tracks and track output")
        track_loss, track_terms = track_consistency_loss(
            preds["track"], track_target["track"], preds["track_vis"], track_target["vis"],
            preds.get("track_conf"), track_target.get("conf"), item["tensor_hw"],
            args.joint_track_min_visibility, args.joint_track_visibility_weight,
            args.joint_track_confidence_weight)

    total = (args.joint_pose_weight * pose_loss
             + args.joint_depth_weight * depth_loss
             + args.joint_point_weight * point_loss
             + args.joint_depth_conf_weight * depth_conf_loss
             + args.joint_point_conf_weight * point_conf_loss
             + args.joint_track_weight * track_loss)
    return total, {
        "pose_loss": float(total.detach().cpu()),
        "joint_pose_term": float(pose_loss.detach().cpu()),
        "joint_depth_term": float(depth_loss.detach().cpu()),
        "joint_point_term": float(point_loss.detach().cpu()),
        "joint_depth_conf_term": float(depth_conf_loss.detach().cpu()),
        "joint_point_conf_term": float(point_conf_loss.detach().cpu()),
        "joint_track_term": float(track_loss.detach().cpu()),
        "joint_track_coord": float(track_terms["track_coord"].detach().cpu()),
        "joint_track_vis": float(track_terms["track_vis"].detach().cpu()),
        "joint_track_conf": float(track_terms["track_conf"].detach().cpu()),
        "pose_rot_mse": pose_terms["pose_rot_mse"],
        "pose_trans_mse": pose_terms["pose_trans_mse"],
    }


def geometry_joint_gauge_budgeted_loss(
    preds: dict,
    item: dict,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Optimise pose damage while spending only the calibrated consistency budget.

    Unlike geometry_joint_gauge_loss, this does not pull every auxiliary output
    back to its exact clean value. Each output may move freely inside the matching
    clean sequence's empirical acceptance interval. A normalized hinge activates
    only near that interval's boundary, and dual ascent raises the multiplier only
    for constraints the current patch actually violates.
    """
    pred_rel = pose_predictions_to_relative_c2w(preds, item["tensor_hw"])
    if args.budget_pose_objective == "untargeted_gt":
        pose_score, pose_terms = pose_aligned_residual_mse(pred_rel, item["pose_gt_rel"], args)
        primary = -pose_score
    else:
        targets = item.get("joint_gauge_targets")
        if targets is None:
            raise RuntimeError("targeted_gauge budget objective needs joint gauge targets")
        pose_score, pose_terms = pose_aligned_residual_mse(pred_rel, targets["pose_rel"], args)
        primary = pose_score

    metrics = differentiable_filter_metrics(preds, item, args)
    selected = {s.strip() for s in args.budget_constraints.split(",") if s.strip()}
    duals = getattr(args, "_budget_duals", None)
    if duals is None:
        duals = {}
        args._budget_duals = duals
    penalty = primary.new_zeros(())
    terms: dict[str, float] = {
        "pose_loss": 0.0,
        "budget_primary_pose": float(pose_score.detach().cpu()),
        "pose_rot_mse": pose_terms["pose_rot_mse"],
        "pose_trans_mse": pose_terms["pose_trans_mse"],
    }
    specs = item["filter_budget"]["metrics"]
    for name, value in metrics.items():
        terms[f"budget_metric_{name}"] = float(value.detach().cpu())
        if name not in selected:
            continue
        safe, violation = budget_safe_value_and_violation(
            value, specs[name], args.filter_budget_fraction)
        dual = float(duals.setdefault(name, args.budget_dual_init))
        penalty = penalty + dual * violation + 0.5 * args.budget_rho * violation.pow(2)
        terms[f"budget_safe_{name}"] = float(safe.detach().cpu())
        terms[f"budget_violation_{name}"] = float(violation.detach().cpu())
        terms[f"budget_dual_{name}"] = dual

    track_target = item.get("joint_track_targets")
    if "track" in selected:
        if track_target is None or "track" not in preds:
            raise RuntimeError("track budget needs clean track targets and track output")
        track_px = budget_track_error_pixels(
            preds, track_target, args.joint_track_min_visibility)
        track_spec = {"clean": 0.0, "threshold": args.budget_track_pixels, "worse": "high"}
        safe, violation = budget_safe_value_and_violation(
            track_px, track_spec, args.filter_budget_fraction)
        dual = float(duals.setdefault("track", args.budget_dual_init))
        penalty = penalty + dual * violation + 0.5 * args.budget_rho * violation.pow(2)
        terms["budget_metric_track"] = float(track_px.detach().cpu())
        terms["budget_safe_track"] = float(safe.detach().cpu())
        terms["budget_violation_track"] = float(violation.detach().cpu())
        terms["budget_dual_track"] = dual

    total = primary + penalty
    terms["budget_penalty"] = float(penalty.detach().cpu())
    terms["pose_loss"] = float(total.detach().cpu())
    return total, terms


def update_budget_duals(args: argparse.Namespace,
                        attack_term_values: dict[str, list[float]]) -> None:
    """One projected dual-ascent update after each patch optimiser update."""
    if args.attack_loss != "geometry_joint_gauge_budgeted":
        return
    duals = getattr(args, "_budget_duals", None)
    if duals is None:
        return
    for key, values in attack_term_values.items():
        prefix = "budget_violation_"
        if not key.startswith(prefix) or not values:
            continue
        name = key[len(prefix):]
        old = float(duals.get(name, args.budget_dual_init))
        duals[name] = float(np.clip(
            old + args.budget_dual_lr * float(np.mean(values)), 0.0, args.budget_dual_max))


def should_cache_clean_pose(args: argparse.Namespace) -> bool:
    return args.attack_loss == "pose_clean_untargeted" or (
        args.attack_loss == "pose_reverse_targeted" and args.pose_reverse_reference == "clean"
    ) or (
        args.attack_loss in ("pose_drift_targeted", "pose_scale_targeted", "pose_yaw_targeted")
        and args.pose_bad_reference == "clean"
    )


def attack_objective_loss(
    model: torch.nn.Module,
    adv_images: torch.Tensor,
    item: dict,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float], bool]:
    if args.attack_loss == "feature_l1":
        adv_features = extract_features(
            model,
            adv_images,
            dtype,
            args.feature_layer,
            args.activation_checkpoint,
        )
        loss, terms = feature_l1_loss(adv_features, item["clean_features"])
        return loss, terms, True

    if args.attack_loss in ("geometry_joint_gauge_targeted", "geometry_joint_gauge_budgeted"):
        # Needs all requested heads.  Handle it before the camera-only forward;
        # the old ordering ran the 1B aggregator twice per update for no reason.
        track_target = item.get("joint_track_targets")
        query = track_target["query_points"] if track_target is not None else None
        preds = forward_all_geometry_heads(
            model, adv_images, dtype, query_points=query,
            track_iters=args.joint_track_iters)
        if args.attack_loss == "geometry_joint_gauge_budgeted":
            loss, terms = geometry_joint_gauge_budgeted_loss(preds, item, args)
        else:
            loss, terms = geometry_joint_gauge_loss(preds, item, args)
        terms["pose_reference"] = "piecewise_gauge_clean"
        terms["piecewise_gauge_family"] = args.piecewise_gauge_family
        terms["piecewise_gauge_magnitude"] = args.piecewise_gauge_magnitude
        return loss, terms, False

    preds = forward_camera_pose_only(
        model,
        adv_images,
        dtype,
        args.activation_checkpoint,
    )
    pred_rel = pose_predictions_to_relative_c2w(preds, item["tensor_hw"])

    if args.attack_loss == "pose_gt_untargeted":
        target_rel = item["pose_gt_rel"]
        loss, terms = pose_relative_mse(pred_rel, target_rel, args)
        terms["pose_reference"] = "gt"
        return loss, terms, True

    if args.attack_loss == "pose_scale_invariant_mse":
        target_rel = item["pose_gt_rel"]
        loss, terms = pose_scale_invariant_mse(pred_rel, target_rel, args)
        terms["pose_reference"] = "gt"
        return loss, terms, True

    if args.attack_loss == "pose_piecewise_gauge_targeted":
        target_rel = item.get("pose_piecewise_rel")
        if target_rel is None:
            raise RuntimeError("pose_piecewise_gauge_targeted needs --piecewise_gauge_family")
        # Scored through the aligned residual, i.e. the evaluator's own Sim(3) fit,
        # so the objective is "match this target up to the gauge the metric removes"
        # rather than "match it exactly in an arbitrary frame". Minimised, not
        # maximised: this is a targeted attack.
        loss, terms = pose_aligned_residual_mse(pred_rel, target_rel, args)
        terms["pose_reference"] = "piecewise_gauge_gt"
        terms["piecewise_gauge_family"] = args.piecewise_gauge_family
        terms["piecewise_gauge_magnitude"] = args.piecewise_gauge_magnitude
        return loss, terms, False

    if args.attack_loss == "pose_pairwise_relative_mse":
        target_rel = item["pose_gt_rel"]
        loss, terms = pose_pairwise_relative_mse(pred_rel, target_rel, args)
        terms["pose_reference"] = "gt"
        return loss, terms, True

    if args.attack_loss == "pose_aligned_residual_mse":
        target_rel = item["pose_gt_rel"]
        loss, terms = pose_aligned_residual_mse(pred_rel, target_rel, args)
        terms["pose_reference"] = "gt"
        return loss, terms, True

    if args.attack_loss == "pose_clean_untargeted":
        target_rel = item["pose_clean_rel"]
        loss, terms = pose_relative_mse(pred_rel, target_rel, args)
        terms["pose_reference"] = "clean"
        return loss, terms, True

    if args.attack_loss == "pose_reverse_targeted":
        source_key = "pose_gt_rel" if args.pose_reverse_reference == "gt" else "pose_clean_rel"
        target_rel = invert_relative_trajectory(item[source_key])
        loss, terms = pose_relative_mse(pred_rel, target_rel, args)
        terms["pose_reference"] = args.pose_reverse_reference
        terms["pose_target"] = "inverse_relative_trajectory"
        return loss, terms, False

    if args.attack_loss in ("pose_drift_targeted", "pose_scale_targeted", "pose_yaw_targeted"):
        source_key = "pose_gt_rel" if args.pose_bad_reference == "gt" else "pose_clean_rel"
        target_rel, target_meta = make_pose_bad_target(item[source_key], args)
        loss, terms = pose_relative_mse(pred_rel, target_rel, args)
        terms.update(target_meta)
        return loss, terms, False

    raise ValueError(f"Unknown attack_loss: {args.attack_loss}")


def load_tum_sequence(
    seq_dir: Path,
    frame_indices: list[int],
    gt_name: str,
    intrinsics: np.ndarray,
    texture_size: int,
    args: argparse.Namespace,
    device: torch.device,
    model: torch.nn.Module | None = None,
    dtype: torch.dtype | None = None,
) -> dict:
    all_images = find_images(seq_dir)
    image_paths = [all_images[int(idx)] for idx in frame_indices]
    images = load_and_preprocess_images(image_paths).to(device)
    tensor_hw = tuple(int(v) for v in images.shape[-2:])
    gt_c2w = tum_rows_to_c2w(read_tum_rows(seq_dir / gt_name), frame_indices)
    c2w = gt_c2w.copy()
    pose_gt_rel = c2w_numpy_to_relative_tensor(gt_c2w, device)
    # Built here rather than in the loss because the per-frame gauge acts in the
    # world frame, and only the absolute GT trajectory is available at this point.
    pose_piecewise_rel = c2w_numpy_to_relative_tensor(
        apply_piecewise_gauge(gt_c2w, args.piecewise_gauge_family,
                              args.piecewise_gauge_magnitude),
        device,
    ) if getattr(args, "piecewise_gauge_family", None) else None
    local_intrinsics = intrinsics
    old_vggt_points = getattr(args, "_active_vggt_points", None)
    old_vggt_point_map_grid = getattr(args, "_active_vggt_point_map_grid", None)
    old_vggt_point_valid_grid = getattr(args, "_active_vggt_point_valid_grid", None)
    old_vggt_source = getattr(args, "_active_vggt_geometry_source", None)
    old_tensor_intrinsics = getattr(args, "_intrinsics_in_tensor_space", False)
    args._active_vggt_points = None
    args._active_vggt_point_map_grid = None
    args._active_vggt_point_valid_grid = None
    args._active_vggt_geometry_source = None
    args._intrinsics_in_tensor_space = False
    clean_features = None
    clean_pose_rel = None
    if args.surface_strength_search and model is not None and dtype is not None:
        with torch.no_grad():
            clean_features = [
                feature.detach()
                for feature in extract_features(model, images, dtype, args.feature_layer)
            ]
    if should_cache_clean_pose(args):
        if model is None or dtype is None:
            raise RuntimeError(f"{args.attack_loss} needs model/dtype to cache clean VGGT pose reference.")
        with torch.no_grad():
            clean_preds = forward_vggt(model, images, dtype)
            clean_pose_rel = pose_predictions_to_relative_c2w(clean_preds, tensor_hw).detach()
    try:
        depth_maps = load_depth_maps(seq_dir, image_paths, frame_indices, tensor_hw, args)
        if args.plane_mode in ("vggt_pointmap_surface", "vggt_manual_anchor_surface"):
            clean_geometry = load_clean_vggt_geometry(seq_dir.name, tensor_hw, args)
            if clean_geometry is not None:
                c2w = clean_geometry["c2w"]
                local_intrinsics = clean_geometry["intrinsics"]
                args._active_vggt_points = clean_geometry["points"]
                args._active_vggt_point_map_grid = clean_geometry["point_map_grid"]
                args._active_vggt_point_valid_grid = clean_geometry["point_valid_grid"]
                args._active_vggt_geometry_source = clean_geometry["source"]
                args._intrinsics_in_tensor_space = True
                if args.use_depth_visibility and clean_geometry["depth_maps"] is not None:
                    depth_maps = clean_geometry["depth_maps"]
        grids, masks, geom_meta = build_geometry_grids(
            c2w,
            image_paths,
            tensor_hw,
            texture_size,
            local_intrinsics,
            args,
            device,
            depth_maps,
            images=images,
            clean_features=clean_features,
            model=model,
            dtype=dtype,
        )
    finally:
        args._active_vggt_points = old_vggt_points
        args._active_vggt_point_map_grid = old_vggt_point_map_grid
        args._active_vggt_point_valid_grid = old_vggt_point_valid_grid
        args._active_vggt_geometry_source = old_vggt_source
        args._intrinsics_in_tensor_space = old_tensor_intrinsics
    joint_track_targets = None
    needs_strict_track = (args.attack_loss == "geometry_joint_gauge_targeted"
                          and args.joint_track_weight > 0)
    needs_budget_track = (args.attack_loss == "geometry_joint_gauge_budgeted"
                          and "track" in {s.strip() for s in args.budget_constraints.split(",")})
    if needs_strict_track or needs_budget_track:
        if model is None or dtype is None:
            raise RuntimeError("four-head joint loss needs model/dtype to cache clean tracks")
        joint_track_targets = clean_track_targets(
            seq_dir.name, images, tensor_hw, model, dtype, args)
    return {
        "seq": seq_dir.name,
        "seq_dir": seq_dir,
        "images": images,
        "image_paths": image_paths,
        "image_names": [Path(path).name for path in image_paths],
        "frame_indices": frame_indices,
        "c2w_gt": gt_c2w,
        "c2w_geometry": c2w,
        "pose_gt_rel": pose_gt_rel,
        "pose_piecewise_rel": pose_piecewise_rel,
        "joint_gauge_targets": build_joint_gauge_targets(seq_dir.name, args, device)
        if args.attack_loss in ("geometry_joint_gauge_targeted",
                                "geometry_joint_gauge_budgeted") else None,
        "joint_track_targets": joint_track_targets,
        "filter_budget": load_filter_budget(seq_dir.name, args)
        if args.attack_loss == "geometry_joint_gauge_budgeted" else None,
        "pose_clean_rel": clean_pose_rel,
        "geometry_intrinsics": np.asarray(local_intrinsics).astype(float).tolist(),
        "geometry_source": geom_meta.get("geometry_source", "tum_gt_geometry"),
        "depth_available": [depth is not None for depth in depth_maps],
        "grids": grids,
        "masks": masks,
        "geometry": geom_meta,
        "tensor_hw": tensor_hw,
        "clean_features": clean_features,
    }


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def assert_texture_only_optimizer(optimizer: torch.optim.Optimizer | None,
                                  texture: torch.Tensor,
                                  model: torch.nn.Module) -> None:
    """Make the threat model executable: victim weights must never be optimised."""
    if optimizer is None:
        return
    optimised = {id(param) for group in optimizer.param_groups for param in group["params"]}
    model_params = {id(param) for param in model.parameters()}
    if optimised != {id(texture)}:
        raise RuntimeError("Adversarial-patch optimiser must contain only the texture tensor")
    if optimised & model_params:
        raise RuntimeError("Victim-model parameter leaked into the patch optimiser")


def initialize_texture(
    args: argparse.Namespace,
    device: torch.device,
    *,
    init: str | None = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    init = init or args.texture_init
    shape = (1, 3, args.texture_size, args.texture_size)
    if init == "random":
        texture = torch.rand(shape, device=device, dtype=torch.float32)
    elif init == "gray":
        texture = torch.full(shape, 0.5, device=device, dtype=torch.float32)
    elif init == "white":
        texture = torch.ones(shape, device=device, dtype=torch.float32)
    elif init == "black":
        texture = torch.zeros(shape, device=device, dtype=torch.float32)
    elif init == "checker":
        yy, xx = torch.meshgrid(
            torch.arange(args.texture_size, device=device),
            torch.arange(args.texture_size, device=device),
            indexing="ij",
        )
        checker = ((xx // 16 + yy // 16) % 2).float()
        texture = checker[None, None].repeat(1, 3, 1, 1)
    elif init == "image":
        if not args.texture_init_image:
            raise ValueError("--texture_init image requires --texture_init_image")
        texture = load_texture_image(args.texture_init_image, args.texture_size, device)
    else:
        raise ValueError(f"Unknown texture_init: {init}")
    texture = texture.clamp(args.print_min, args.print_max)
    texture.requires_grad_(requires_grad)
    return texture


def load_texture_image(image_path: str | Path, texture_size: int, device: torch.device) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((texture_size, texture_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor.clamp(0.0, 1.0)


def natural_reference_texture(args: argparse.Namespace, device: torch.device) -> torch.Tensor | None:
    path = args.natural_reference_image or (
        args.texture_init_image if args.texture_init == "image" else None
    )
    if not path or args.natural_reference_weight <= 0:
        return None
    cached = getattr(args, "_natural_reference_texture", None)
    cached_path = getattr(args, "_natural_reference_texture_path", None)
    if cached is not None and cached_path == str(path):
        return cached
    reference = load_texture_image(path, args.texture_size, device).detach()
    args._natural_reference_texture = reference
    args._natural_reference_texture_path = str(path)
    return reference


def total_variation_loss(texture: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(texture[..., :, 1:] - texture[..., :, :-1]).mean()
    dy = torch.abs(texture[..., 1:, :] - texture[..., :-1, :]).mean()
    return dx + dy


def low_frequency_loss(texture: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return texture.new_zeros(())
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = F.avg_pool2d(texture, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return F.mse_loss(texture, blurred)


def printable_color_loss(texture: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.printable_color_levels <= 1:
        return texture.new_zeros(())
    levels = torch.linspace(
        args.print_min,
        args.print_max,
        int(args.printable_color_levels),
        device=texture.device,
        dtype=texture.dtype,
    )
    distances = torch.abs(texture.unsqueeze(-1) - levels.view(1, 1, 1, 1, -1))
    return distances.min(dim=-1).values.mean()


def patch_regularization_terms(texture: torch.Tensor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    tv = total_variation_loss(texture)
    low_freq = low_frequency_loss(texture, args.low_frequency_kernel)
    printable = printable_color_loss(texture, args)
    reference = natural_reference_texture(args, texture.device)
    natural_reference = (
        F.mse_loss(texture, reference) if reference is not None else texture.new_zeros(())
    )
    total = (
        args.tv_weight * tv
        + args.low_frequency_weight * low_freq
        + args.printability_weight * printable
        + args.natural_reference_weight * natural_reference
    )
    return {
        "regularization_total": total,
        "tv": tv,
        "low_frequency": low_freq,
        "printability": printable,
        "natural_reference": natural_reference,
    }


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
    # Only the texture is optimised, but the model's 1.257B parameters still carry
    # requires_grad=True by default, so every backward also computes and stores
    # gradients for them (4.68 GiB in fp32) and autograd keeps the activations those
    # gradients need. Freezing removes that without touching the texture's own
    # gradient path. --no_freeze_model_parameters restores the old behaviour for A/B.
    if getattr(args, "freeze_model_parameters", True):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    rng = np.random.default_rng(args.seed)
    texture = initialize_texture(args, device, requires_grad=not args.freeze_texture)
    optimizer = None if args.freeze_texture else torch.optim.AdamW([texture], lr=args.patch_lr)
    assert_texture_only_optimizer(optimizer, texture, model)

    patch_dir = output_dir / "geometry_patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    initial_texture_npz = patch_dir / "initial_texture.npz"
    np.savez_compressed(initial_texture_npz, texture=texture.detach().float().cpu().numpy())
    to_pil_image(texture.squeeze(0).detach().float().cpu().clamp(0, 1)).save(
        patch_dir / "initial_texture.png"
    )
    history_path = patch_dir / "training_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    total_updates = args.iterations * args.inner_loop
    warmup_updates = args.warmup_iterations * args.inner_loop
    update_idx = 0
    started = time.time()
    last_loss = None

    if args.freeze_texture:
        last_loss = 0.0
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(
                json.dumps(
                    {
                        "update": 0,
                        "feature_l1": last_loss,
                        "frozen_texture": True,
                        "texture_init": args.texture_init,
                    }
                )
                + "\n"
            )
    with history_path.open("a", encoding="utf-8") as history_file:
        if args.freeze_texture:
            pass
        for iteration in range(args.iterations):
            if args.freeze_texture:
                break
            args._eot_strength = (
                scheduled_eot_strength(update_idx, total_updates, args.eot_warmup_fraction)
                if args.physical_eot else 0.0
            )
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
                    model=model,
                    dtype=dtype,
                )
                if args.attack_loss == "feature_l1" and item["clean_features"] is None:
                    with torch.no_grad():
                        item["clean_features"] = [
                            feature.detach()
                            for feature in extract_features(model, item["images"], dtype, args.feature_layer)
                        ]
                batch.append(item)

            for inner_step in range(args.inner_loop):
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                current_lr = scheduled_lr(
                    args.patch_lr,
                    update_idx,
                    total_updates,
                    warmup_updates,
                    args.scheduler,
                )
                args._eot_strength = (
                    scheduled_eot_strength(update_idx, total_updates, args.eot_warmup_fraction)
                    if args.physical_eot else 0.0
                )
                set_optimizer_lr(optimizer, current_lr)
                losses = []
                reg_totals = []
                tv_terms = []
                low_freq_terms = []
                printable_terms = []
                natural_reference_terms = []
                coverages = []
                attack_term_values: dict[str, list[float]] = {}
                for item in batch:
                    adv_images = apply_geometry_patch(
                        item["images"],
                        texture,
                        item["grids"],
                        item["masks"],
                        args,
                        training=True,
                    )
                    loss, terms, maximize_loss = attack_objective_loss(
                        model,
                        adv_images,
                        item,
                        args,
                        dtype,
                    )
                    reg_terms = patch_regularization_terms(texture, args)
                    objective = (-loss if maximize_loss else loss) + reg_terms["regularization_total"]
                    (objective / len(batch)).backward()
                    metric_value = terms.get("feature_l1")
                    if metric_value is None:
                        metric_value = terms.get("pose_loss")
                    if metric_value is None:
                        metric_value = terms["total"]
                    losses.append(float(metric_value))
                    for key, value in terms.items():
                        if isinstance(value, (int, float)):
                            attack_term_values.setdefault(key, []).append(float(value))
                    reg_totals.append(float(reg_terms["regularization_total"].detach().cpu()))
                    tv_terms.append(float(reg_terms["tv"].detach().cpu()))
                    low_freq_terms.append(float(reg_terms["low_frequency"].detach().cpu()))
                    printable_terms.append(float(reg_terms["printability"].detach().cpu()))
                    natural_reference_terms.append(float(reg_terms["natural_reference"].detach().cpu()))
                    coverages.append(item["geometry"]["mask_coverage_mean"])

                if texture.grad is None:
                    raise RuntimeError("Geometry patch gradient is None.")
                optimizer.step()
                with torch.no_grad():
                    texture.clamp_(args.print_min, args.print_max)
                update_budget_duals(args, attack_term_values)

                update_idx += 1
                last_loss = float(np.mean(losses))
                record = {
                    "iteration": iteration + 1,
                    "inner_step": inner_step + 1,
                    "update": update_idx,
                    "lr": current_lr,
                    "attack_loss": args.attack_loss,
                    "attack_metric": last_loss,
                    "regularization_total": float(np.mean(reg_totals)) if reg_totals else 0.0,
                    "tv": float(np.mean(tv_terms)) if tv_terms else 0.0,
                    "low_frequency": float(np.mean(low_freq_terms)) if low_freq_terms else 0.0,
                    "printability": float(np.mean(printable_terms)) if printable_terms else 0.0,
                    "natural_reference": float(np.mean(natural_reference_terms)) if natural_reference_terms else 0.0,
                    "mask_coverage_mean": float(np.mean(coverages)),
                    "eot_strength": float(getattr(args, "_eot_strength", 0.0)),
                    "scenes": [item["seq"] for item in batch],
                }
                for key, values in attack_term_values.items():
                    record[key] = float(np.mean(values))
                if args.attack_loss == "geometry_joint_gauge_budgeted":
                    for name, value in getattr(args, "_budget_duals", {}).items():
                        record[f"budget_dual_after_{name}"] = float(value)
                history_file.write(json.dumps(record) + "\n")
                if update_idx == 1 or update_idx % args.log_every == 0 or update_idx == total_updates:
                    print(
                        f"[train-geometry] update {update_idx:06d}/{total_updates:06d} "
                        f"scenes={len(batch)} {args.attack_loss}={last_loss:.6f} "
                        f"reg={record['regularization_total']:.6f} "
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
        "attack_target": args.attack_loss,
        "pose_reverse_reference": args.pose_reverse_reference,
        "pose_bad_reference": args.pose_bad_reference,
        "pose_bad_target": {
            "drift_x_m": args.pose_drift_x_m,
            "drift_y_m": args.pose_drift_y_m,
            "drift_z_m": args.pose_drift_z_m,
            "drift_yaw_degrees": args.pose_drift_yaw_degrees,
            "translation_scale": args.pose_translation_scale,
            "yaw_degrees": args.pose_yaw_degrees,
        },
        "pose_rotation_weight": args.pose_rotation_weight,
        "pose_translation_weight": args.pose_translation_weight,
        "pose_scale_invariant_eps": args.pose_scale_invariant_eps,
        "joint_gauge": {
            "family": args.piecewise_gauge_family,
            "magnitude": args.piecewise_gauge_magnitude,
            "orthogonal_mode_order": args.orthogonal_mode_order,
            "orthogonal_mode_axis": args.orthogonal_mode_axis,
            "pose_weight": args.joint_pose_weight,
            "depth_weight": args.joint_depth_weight,
            "point_weight": args.joint_point_weight,
            "depth_conf_weight": args.joint_depth_conf_weight,
            "point_conf_weight": args.joint_point_conf_weight,
            "point_stride": args.joint_point_stride,
            "track_weight": args.joint_track_weight,
            "track_grid_rows": args.joint_track_grid_rows,
            "track_grid_cols": args.joint_track_grid_cols,
            "track_query_margin": args.joint_track_query_margin,
            "track_iters": args.joint_track_iters,
            "track_min_visibility": args.joint_track_min_visibility,
            "track_visibility_weight": args.joint_track_visibility_weight,
            "track_confidence_weight": args.joint_track_confidence_weight,
        },
        "filter_budget": {
            "json": args.filter_budget_json,
            "fraction": args.filter_budget_fraction,
            "constraints": args.budget_constraints,
            "pose_objective": args.budget_pose_objective,
            "dual_init": args.budget_dual_init,
            "dual_lr": args.budget_dual_lr,
            "dual_max": args.budget_dual_max,
            "rho": args.budget_rho,
            "conf_floor_temperature": args.budget_conf_floor_temperature,
            "reproj_stride": args.budget_reproj_stride,
            "track_pixels": args.budget_track_pixels,
            "final_duals": dict(getattr(args, "_budget_duals", {})),
        },
        "feature_layer": args.feature_layer,
        "texture_shape": list(texture.shape),
        "texture_init": args.texture_init,
        "texture_init_image": args.texture_init_image,
        "freeze_texture": bool(args.freeze_texture),
        "optimizer_target": "texture_only",
        "victim_model_weights_updated": False,
        "patch_lr": args.patch_lr,
        "scheduler": args.scheduler,
        "warmup_iterations": args.warmup_iterations,
        "iterations": args.iterations,
        "inner_loop": args.inner_loop,
        "scenes_per_iteration": args.scenes_per_iteration,
        "total_updates": total_updates,
        "elapsed_seconds": time.time() - started,
        "last_logged_attack_metric": last_loss,
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
            "noise_scope": "patch_only",
            "warmup_fraction": args.eot_warmup_fraction,
            "geo_translate": args.eot_geo_translate,
            "geo_scale": args.eot_geo_scale,
            "geo_rotate_degrees": args.eot_geo_rotate_degrees,
            "geo_perspective": args.eot_geo_perspective,
        },
        "regularization": {
            "tv_weight": args.tv_weight,
            "printability_weight": args.printability_weight,
            "printable_color_levels": args.printable_color_levels,
            "low_frequency_weight": args.low_frequency_weight,
            "low_frequency_kernel": args.low_frequency_kernel,
            "natural_reference_image": args.natural_reference_image,
            "natural_reference_weight": args.natural_reference_weight,
        },
        "plane": {
            "mode": args.plane_mode,
            "clean_vggt_output_root": args.clean_vggt_output_root,
            "vggt_point_conf_percentile": args.vggt_point_conf_percentile,
            "center_first_camera": [args.plane_center_x, args.plane_center_y, args.plane_distance],
            "width_m": args.plane_width,
            "height_m": args.plane_height,
            "surface_candidate_grid": args.surface_candidate_grid,
            "surface_search_margin": args.surface_search_margin,
            "optimize_geometry": bool(args.optimize_geometry),
            "geometry_size_scales": args.geometry_size_scales,
            "geometry_roll_degrees": args.geometry_roll_degrees,
            "fused_point_stride": args.fused_point_stride,
            "fused_max_points": args.fused_max_points,
            "fused_surface_candidates": args.fused_surface_candidates,
            "fused_normal_radius": args.fused_normal_radius,
            "fused_min_neighbors": args.fused_min_neighbors,
            "fused_max_neighbors": args.fused_max_neighbors,
            "fused_max_plane_residual": args.fused_max_plane_residual,
            "surface_score_mode": args.surface_score_mode,
            "surface_coverage_min": args.surface_coverage_min,
            "surface_coverage_max": args.surface_coverage_max,
            "surface_min_visible_frames": args.surface_min_visible_frames,
            "surface_min_visibility_ratio": args.surface_min_visibility_ratio,
            "surface_orientation_filter": args.surface_orientation_filter,
            "surface_max_tilt_degrees": args.surface_max_tilt_degrees,
            "surface_min_center_depth": args.surface_min_center_depth,
            "surface_max_center_depth": args.surface_max_center_depth,
            "surface_support_check": bool(args.surface_support_check),
            "surface_support_abs_tolerance": args.surface_support_abs_tolerance,
            "surface_support_rel_tolerance": args.surface_support_rel_tolerance,
            "surface_min_support_ratio": args.surface_min_support_ratio,
            "manual_anchor": {
                "coordinates": args.manual_anchor_coordinates,
                "x": args.manual_anchor_x,
                "y": args.manual_anchor_y,
                "frame": args.manual_anchor_frame,
                "search_radius": args.manual_anchor_search_radius,
                "roll_degrees": args.manual_anchor_roll_degrees,
            },
            "manual_quad": {
                "coordinates": args.manual_quad_coordinates,
                "xy": args.manual_quad_xy,
                "frame": args.manual_anchor_frame,
                "depth_sample_stride": args.manual_quad_depth_sample_stride,
                "fit_shrink": args.manual_quad_fit_shrink,
                "plane_inlier_tolerance": args.manual_quad_plane_inlier_tolerance,
                "min_inlier_ratio": args.manual_quad_min_inlier_ratio,
            },
            "surface_strength_search": bool(args.surface_strength_search),
            "surface_strength_candidates": args.surface_strength_candidates,
            "surface_strength_steps": args.surface_strength_steps,
            "surface_strength_lr": args.surface_strength_lr,
            "surface_strength_texture_init": args.surface_strength_texture_init,
            "surface_strength_regularization_weight": args.surface_strength_regularization_weight,
            "natural_auto_relax": bool(args.natural_auto_relax),
            "natural_relax_max_coverage": args.natural_relax_max_coverage,
            "natural_relax_min_visible_frames": args.natural_relax_min_visible_frames,
            "natural_relax_min_visibility_ratio": args.natural_relax_min_visibility_ratio,
            "natural_relax_orientation_filter": args.natural_relax_orientation_filter,
            "natural_relax_max_tilt_degrees": args.natural_relax_max_tilt_degrees,
            "natural_relax_min_center_depth": args.natural_relax_min_center_depth,
            "natural_relax_min_support_ratio": args.natural_relax_min_support_ratio,
        },
        "intrinsics": intrinsics.astype(float).tolist(),
        "frame_manifest": args.frame_manifest,
        "gt_name": args.gt_name,
        "initial_texture_path": str(initial_texture_npz),
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
            model=model,
            dtype=dtype,
        )
        adv_images = apply_geometry_patch(item["images"], texture, item["grids"], item["masks"], args, training=False).detach()
        visibility_mask_path = out_dir / "geometry_visibility_masks.npz"
        np.savez_compressed(
            visibility_mask_path,
            masks=item["masks"].detach().float().cpu().numpy().astype(np.float16),
        )
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
            "outputs": {
                "attacked_vggt_outputs": "vggt_outputs.npz",
                "visibility_masks": visibility_mask_path.name,
            },
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
    parser.add_argument(
        "--texture_init",
        choices=("random", "gray", "white", "black", "checker", "image"),
        default="random",
    )
    parser.add_argument(
        "--texture_init_image",
        default=None,
        help="RGB image used when --texture_init=image, e.g. a natural/style-transferred sticker texture.",
    )
    parser.add_argument("--freeze_texture", action="store_true")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--inner_loop", type=int, default=10)
    parser.add_argument("--scenes_per_iteration", type=int, default=1)
    parser.add_argument("--patch_lr", type=float, default=0.001)
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--warmup_iterations", type=int, default=20)
    parser.add_argument("--feature_layer", default="aggregator_final")
    parser.add_argument(
        "--attack_loss",
        choices=(
            "feature_l1",
            "pose_gt_untargeted",
            "pose_scale_invariant_mse",
            "pose_pairwise_relative_mse",
            "pose_aligned_residual_mse",
            "pose_piecewise_gauge_targeted",
            "geometry_joint_gauge_targeted",
            "geometry_joint_gauge_budgeted",
            "pose_clean_untargeted",
            "pose_reverse_targeted",
            "pose_drift_targeted",
            "pose_scale_targeted",
            "pose_yaw_targeted",
        ),
        default="feature_l1",
        help=(
            "feature_l1 maximizes aggregator feature distance; pose_* losses operate on VGGT pose output. "
            "pose_reverse/drift/scale/yaw targeted losses minimize distance to a constructed bad trajectory. "
            "pose_scale_invariant_mse is pose_gt_untargeted with each trajectory normalized by its own "
            "translation RMS, so the loss no longer rewards the global scale gauge that the Sim(3)-aligned "
            "ATE discards (note: this is scale-INVARIANT, unrelated to the scale-TARGETED pose_scale_targeted). "
            "pose_pairwise_relative_mse compares all N(N-1)/2 relative poses instead of normalizing to frame 0, "
            "removing frame 0's privileged position. pose_aligned_residual_mse reproduces the evaluator's own "
            "Umeyama Sim(3) alignment inside the loss and scores the residual after it, i.e. a differentiable "
            "ATE normalized by its ceiling."
        ),
    )
    parser.add_argument("--pose_reverse_reference", choices=("gt", "clean"), default="gt")
    parser.add_argument("--pose_bad_reference", choices=("gt", "clean"), default="gt")
    parser.add_argument("--pose_drift_x_m", type=float, default=0.5)
    parser.add_argument("--pose_drift_y_m", type=float, default=0.0)
    parser.add_argument("--pose_drift_z_m", type=float, default=0.0)
    parser.add_argument("--pose_drift_yaw_degrees", type=float, default=0.0)
    parser.add_argument("--pose_translation_scale", type=float, default=2.0)
    parser.add_argument("--pose_yaw_degrees", type=float, default=30.0)
    parser.add_argument("--pose_rotation_weight", type=float, default=1.0)
    parser.add_argument("--pose_translation_weight", type=float, default=1.0)
    parser.add_argument(
        "--piecewise_gauge_family",
        default=None,
        choices=("scale_ramp", "scale_sine", "scale_split", "yaw_ramp", "yaw_sine",
                 "trans_ramp", "trans_sine", "orthogonal_mode"),
        help="per-frame gauge used to build the target for pose_piecewise_gauge_targeted. "
        "A constant gauge is removed exactly by the evaluator, so only frame-varying "
        "ones leave measurable error; see scripts/diag_piecewise_gauge.py for what each "
        "family is worth. scale_sine is the most consistent across sequences "
        "(68-87% of the ATE ceiling) at a frame-to-frame jump of 0.35 versus 1.0 for a "
        "hard split.",
    )
    parser.add_argument("--piecewise_gauge_magnitude", type=float, default=1.0)
    parser.add_argument("--orthogonal_mode_order", type=int, default=2,
                        help="cosine order for --piecewise_gauge_family orthogonal_mode. "
                             "Order 2 is the sweet spot: 89-93%% of the ATE ceiling at a "
                             "jump of 0.30-0.37, while order 5 buys almost nothing in ATE "
                             "and jumps 0.69-0.77")
    parser.add_argument("--orthogonal_mode_axis", type=int, default=0, choices=(0, 1, 2),
                        help="which world axis the displacement runs along; the best "
                             "choice is sequence dependent (x for sitting_xyz, y for "
                             "halfsphere and static)")
    parser.add_argument("--joint_pose_weight", type=float, default=1.0)
    parser.add_argument("--joint_depth_weight", type=float, default=1.0)
    parser.add_argument("--joint_point_weight", type=float, default=1.0)
    parser.add_argument("--joint_depth_conf_weight", type=float, default=0.0,
                        help="preserve clean depth-head confidence; zero reproduces old runs")
    parser.add_argument("--joint_point_conf_weight", type=float, default=0.0,
                        help="preserve clean point-head confidence; zero reproduces old runs")
    parser.add_argument("--joint_track_weight", type=float, default=0.0,
                        help="preserve the gauge-invariant clean track head; zero keeps "
                             "the historical three-head objective")
    parser.add_argument("--joint_track_grid_rows", type=int, default=6)
    parser.add_argument("--joint_track_grid_cols", type=int, default=8)
    parser.add_argument("--joint_track_query_margin", type=float, default=0.10)
    parser.add_argument("--joint_track_iters", type=int, default=2,
                        help="tracker refinements; 2 is the memory-conscious training default")
    parser.add_argument("--joint_track_min_visibility", type=float, default=0.20)
    parser.add_argument("--joint_track_visibility_weight", type=float, default=0.10)
    parser.add_argument("--joint_track_confidence_weight", type=float, default=0.10)
    parser.add_argument("--joint_point_stride", type=int, default=4,
                        help="sub-sampling for the point-map term; the full map is 2M "
                             "points per sequence")
    parser.add_argument("--filter_budget_json", default=None,
                        help="matched-clean thresholds emitted by eval_output_filters.py")
    parser.add_argument("--filter_budget_fraction", type=float, default=0.80,
                        help="fraction of the clean-to-filter interval the attack may spend")
    parser.add_argument("--budget_constraints",
                        default="conf_std,conf_frac_floor,head_disagree_rel,reproj_rel_err,track")
    parser.add_argument("--budget_pose_objective",
                        choices=("targeted_gauge", "untargeted_gt"), default="untargeted_gt")
    parser.add_argument("--budget_dual_init", type=float, default=0.0)
    parser.add_argument("--budget_dual_lr", type=float, default=1.0)
    parser.add_argument("--budget_dual_max", type=float, default=100.0)
    parser.add_argument("--budget_rho", type=float, default=1.0,
                        help="quadratic augmented-Lagrangian weight on normalized violation")
    parser.add_argument("--budget_conf_floor_temperature", type=float, default=0.02)
    parser.add_argument("--budget_reproj_stride", type=int, default=8)
    parser.add_argument("--budget_track_pixels", type=float, default=2.0)
    parser.add_argument(
        "--pose_scale_invariant_eps",
        type=float,
        default=1e-6,
        help="floor on the per-trajectory translation RMS used by pose_scale_invariant_mse, "
        "so a degenerate (near-static) sequence cannot blow the normalization up",
    )
    parser.add_argument("--activation_checkpoint", action="store_true")
    parser.add_argument(
        "--no_freeze_model_parameters",
        dest="freeze_model_parameters",
        action="store_false",
        help="keep the VGGT parameters trainable during patch optimization (the old "
        "behaviour). Only the texture is ever stepped, so this just costs memory and "
        "time; kept for A/B comparison against runs made before the freeze.",
    )
    parser.set_defaults(freeze_model_parameters=True)
    parser.add_argument("--skip_existing_outputs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--fx", type=float, default=535.4)
    parser.add_argument("--fy", type=float, default=539.2)
    parser.add_argument("--cx", type=float, default=320.1)
    parser.add_argument("--cy", type=float, default=247.6)
    parser.add_argument(
        "--plane_mode",
        choices=(
            "fixed",
            "auto_depth_surface",
            "fused_depth_surface",
            "vggt_pointmap_surface",
            "vggt_manual_anchor_surface",
            "depth_manual_anchor_surface",
            "depth_manual_quad_surface",
        ),
        default="fixed",
    )
    parser.add_argument("--clean_vggt_output_root", default=None)
    parser.add_argument("--vggt_point_conf_percentile", type=float, default=40.0)
    parser.add_argument("--plane_width", type=float, default=0.6)
    parser.add_argument("--plane_height", type=float, default=0.6)
    parser.add_argument("--plane_distance", type=float, default=2.0)
    parser.add_argument("--plane_center_x", type=float, default=0.0)
    parser.add_argument("--plane_center_y", type=float, default=0.0)
    parser.add_argument("--manual_anchor_coordinates", choices=("normalized", "pixel"), default="normalized")
    parser.add_argument("--manual_anchor_x", type=float, default=0.5)
    parser.add_argument("--manual_anchor_y", type=float, default=0.5)
    parser.add_argument("--manual_anchor_frame", type=int, default=0)
    parser.add_argument("--manual_anchor_search_radius", type=int, default=12)
    parser.add_argument("--manual_anchor_roll_degrees", type=float, default=0.0)
    parser.add_argument("--manual_quad_coordinates", choices=("normalized", "pixel"), default="normalized")
    parser.add_argument(
        "--manual_quad_xy",
        default="",
        help=(
            "Quad for depth_manual_quad_surface, ordered TL,TR,BR,BL. "
            "Format: x0,y0,x1,y1,x2,y2,x3,y3; values are normalized unless "
            "--manual_quad_coordinates=pixel."
        ),
    )
    parser.add_argument("--manual_quad_depth_sample_stride", type=int, default=2)
    parser.add_argument(
        "--manual_quad_fit_shrink",
        type=float,
        default=0.70,
        help="Fit the carrier plane using an inner quad, while keeping the full quad as the rendered patch boundary.",
    )
    parser.add_argument(
        "--manual_quad_plane_inlier_tolerance",
        type=float,
        default=0.06,
        help="Absolute point-to-plane tolerance in meters for robust manual-quad plane fitting.",
    )
    parser.add_argument(
        "--manual_quad_min_inlier_ratio",
        type=float,
        default=0.25,
        help="Minimum fraction of quad depth samples kept for robust manual-quad plane refitting.",
    )
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
    parser.add_argument("--fused_point_stride", type=int, default=28)
    parser.add_argument("--fused_max_points", type=int, default=6000)
    parser.add_argument("--fused_surface_candidates", type=int, default=64)
    parser.add_argument("--fused_normal_radius", type=float, default=0.25)
    parser.add_argument("--fused_min_neighbors", type=int, default=24)
    parser.add_argument("--fused_max_neighbors", type=int, default=256)
    parser.add_argument("--fused_max_plane_residual", type=float, default=0.08)
    parser.add_argument(
        "--surface_score_mode",
        choices=("coverage", "natural"),
        default="coverage",
        help="coverage keeps the old max-visible-area search; natural restricts candidates to a sticker-like coverage range.",
    )
    parser.add_argument("--surface_coverage_min", type=float, default=0.005)
    parser.add_argument("--surface_coverage_max", type=float, default=0.06)
    parser.add_argument("--surface_min_visible_frames", type=int, default=3)
    parser.add_argument("--surface_min_visibility_ratio", type=float, default=0.5)
    parser.add_argument(
        "--surface_orientation_filter",
        choices=("none", "fronto", "tabletop", "side", "fronto_or_tabletop", "axis_aligned"),
        default="none",
    )
    parser.add_argument("--surface_max_tilt_degrees", type=float, default=35.0)
    parser.add_argument("--surface_min_center_depth", type=float, default=0.0)
    parser.add_argument("--surface_max_center_depth", type=float, default=0.0)
    parser.add_argument("--surface_support_check", action="store_true")
    parser.add_argument("--surface_support_abs_tolerance", type=float, default=0.08)
    parser.add_argument("--surface_support_rel_tolerance", type=float, default=0.05)
    parser.add_argument("--surface_min_support_ratio", type=float, default=0.6)
    parser.add_argument("--surface_strength_search", action="store_true")
    parser.add_argument("--surface_strength_candidates", type=int, default=1)
    parser.add_argument("--surface_strength_steps", type=int, default=0)
    parser.add_argument("--surface_strength_lr", type=float, default=0.002)
    parser.add_argument("--surface_strength_texture_init", choices=("random", "gray", "white", "black", "checker"), default="random")
    parser.add_argument("--surface_strength_regularization_weight", type=float, default=1.0)
    parser.add_argument("--natural_auto_relax", action="store_true")
    parser.add_argument("--natural_relax_max_coverage", type=float, default=0.08)
    parser.add_argument("--natural_relax_min_visible_frames", type=int, default=2)
    parser.add_argument("--natural_relax_min_visibility_ratio", type=float, default=0.25)
    parser.add_argument(
        "--natural_relax_orientation_filter",
        choices=("none", "fronto", "tabletop", "side", "fronto_or_tabletop", "axis_aligned"),
        default="fronto_or_tabletop",
    )
    parser.add_argument("--natural_relax_max_tilt_degrees", type=float, default=50.0)
    parser.add_argument("--natural_relax_min_center_depth", type=float, default=1.0)
    parser.add_argument("--natural_relax_min_support_ratio", type=float, default=0.3)
    parser.add_argument("--tv_weight", type=float, default=0.0)
    parser.add_argument("--printability_weight", type=float, default=0.0)
    parser.add_argument("--printable_color_levels", type=int, default=8)
    parser.add_argument("--low_frequency_weight", type=float, default=0.0)
    parser.add_argument("--low_frequency_kernel", type=int, default=9)
    parser.add_argument(
        "--natural_reference_image",
        default=None,
        help="Optional natural/style-transferred texture reference preserved by an MSE regularizer.",
    )
    parser.add_argument("--natural_reference_weight", type=float, default=0.0)
    parser.add_argument("--physical_eot", action="store_true")
    parser.add_argument("--print_min", type=float, default=0.0)
    parser.add_argument("--print_max", type=float, default=1.0)
    parser.add_argument("--eot_brightness", type=float, default=0.15)
    parser.add_argument("--eot_contrast", type=float, default=0.15)
    parser.add_argument("--eot_gamma", type=float, default=0.10)
    parser.add_argument("--eot_noise_std", type=float, default=0.01)
    parser.add_argument("--eot_warmup_fraction", type=float, default=0.25,
                        help="fraction of updates trained without EOT before its "
                             "strength ramps linearly to full; avoids burying the "
                             "initial targeted gradient in augmentation noise")
    # Geometric EOT. Previously absent entirely, which left the texture optimised
    # for one exact alignment; the MDE attack literature randomises viewpoint and
    # placement and reports EOT *helping* by ~40%, against the 64% train/test drop
    # the photometric-only version produced here on a large patch.
    parser.add_argument("--eot_geo_translate", type=float, default=0.02,
                        help="patch-relative placement jitter, in normalised texture "
                             "coordinates where the patch spans [-1, 1]")
    parser.add_argument("--eot_geo_scale", type=float, default=0.03,
                        help="relative size jitter of the printed patch")
    parser.add_argument("--eot_geo_rotate_degrees", type=float, default=2.0,
                        help="in-plane orientation error of the printed patch")
    parser.add_argument("--eot_geo_perspective", type=float, default=0.01,
                        help="slight out-of-plane tilt, covering surface non-planarity "
                             "and camera calibration error")
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
        f"attack_loss={args.attack_loss} "
        f"plane_mode={args.plane_mode} depth_visibility={args.use_depth_visibility} "
        f"physical_eot={args.physical_eot} strength_search={args.surface_strength_search}"
    )

    model = load_model(args, device)
    for param in model.parameters():
        param.requires_grad_(False)

    if args.texture_path:
        texture = load_texture(args.texture_path, device)
        patch_metadata = {
            "mode": "gt_geometry_aware_planar_patch",
            "loaded_texture_path": str(Path(args.texture_path)),
            "attack_target": args.attack_loss,
            "pose_reverse_reference": args.pose_reverse_reference,
            "pose_bad_reference": args.pose_bad_reference,
            "pose_bad_target": {
                "drift_x_m": args.pose_drift_x_m,
                "drift_y_m": args.pose_drift_y_m,
                "drift_z_m": args.pose_drift_z_m,
                "drift_yaw_degrees": args.pose_drift_yaw_degrees,
                "translation_scale": args.pose_translation_scale,
                "yaw_degrees": args.pose_yaw_degrees,
            },
            "pose_rotation_weight": args.pose_rotation_weight,
            "pose_translation_weight": args.pose_translation_weight,
            "pose_scale_invariant_eps": args.pose_scale_invariant_eps,
            "frame_manifest": args.frame_manifest,
            "intrinsics": intrinsics.astype(float).tolist(),
            "plane_mode": args.plane_mode,
            "clean_vggt_output_root": args.clean_vggt_output_root,
            "vggt_point_conf_percentile": args.vggt_point_conf_percentile,
            "use_depth_visibility": bool(args.use_depth_visibility),
            "physical_eot": bool(args.physical_eot),
            "surface_strength_search": bool(args.surface_strength_search),
            "surface_strength_candidates": args.surface_strength_candidates,
            "surface_strength_steps": args.surface_strength_steps,
            "natural_auto_relax": bool(args.natural_auto_relax),
            "regularization": {
                "tv_weight": args.tv_weight,
                "printability_weight": args.printability_weight,
                "printable_color_levels": args.printable_color_levels,
                "low_frequency_weight": args.low_frequency_weight,
                "low_frequency_kernel": args.low_frequency_kernel,
            },
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
