"""GT-geometry-aware planar patch attack for VGGT on TUM-10.

This is the first minimal geometry-consistent physical patch pipeline:

* one learnable texture is shared by all frames and sequences
* for each TUM sequence, a 3D planar patch is placed in front of the first
  selected camera using GT camera pose
* the same 3D plane is projected into all selected frames with GT poses and RGB
  intrinsics, then differentiably sampled into the VGGT input images
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


def build_patch_plane_world(first_c2w: np.ndarray, args: argparse.Namespace) -> np.ndarray:
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
    return (first_c2w @ corners_cam.T).T[:, :3]


def build_geometry_grids(
    c2w: np.ndarray,
    image_paths: list[str],
    tensor_hw: tuple[int, int],
    texture_size: int,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    patch_world = build_patch_plane_world(c2w[0], args)
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
    projected_corners = []
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
        mask = inside.astype(np.float32).reshape(1, tensor_h, tensor_w)
        grids.append(grid)
        masks.append(mask)
        coverages.append(float(mask.mean()))

    grids_t = torch.from_numpy(np.stack(grids, axis=0)).to(device)
    masks_t = torch.from_numpy(np.stack(masks, axis=0)).to(device)
    meta = {
        "plane_corners_world": patch_world.astype(float).tolist(),
        "projected_corners": projected_corners,
        "mask_coverage_per_frame": coverages,
        "mask_coverage_mean": float(np.mean(coverages)),
    }
    return grids_t, masks_t, meta


def apply_geometry_patch(images: torch.Tensor, texture: torch.Tensor, grids: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    n_frames = images.shape[0]
    sampled = F.grid_sample(
        texture.expand(n_frames, -1, -1, -1),
        grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return (images * (1.0 - masks) + sampled * masks).clamp(0.0, 1.0)


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
    grids, masks, geom_meta = build_geometry_grids(
        c2w,
        image_paths,
        tensor_hw,
        texture_size,
        intrinsics,
        args,
        device,
    )
    return {
        "seq": seq_dir.name,
        "seq_dir": seq_dir,
        "images": images,
        "image_paths": image_paths,
        "image_names": [Path(path).name for path in image_paths],
        "frame_indices": frame_indices,
        "c2w_gt": c2w,
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
                    adv_images = apply_geometry_patch(item["images"], texture, item["grids"], item["masks"])
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
                    texture.clamp_(0.0, 1.0)

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
        "plane": {
            "center_first_camera": [args.plane_center_x, args.plane_center_y, args.plane_distance],
            "width_m": args.plane_width,
            "height_m": args.plane_height,
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
        adv_images = apply_geometry_patch(item["images"], texture, item["grids"], item["masks"]).detach()
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
    parser.add_argument("--plane_width", type=float, default=0.6)
    parser.add_argument("--plane_height", type=float, default=0.6)
    parser.add_argument("--plane_distance", type=float, default=2.0)
    parser.add_argument("--plane_center_x", type=float, default=0.0)
    parser.add_argument("--plane_center_y", type=float, default=0.0)
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
        f"texture_size={args.texture_size} iterations={args.iterations} inner_loop={args.inner_loop}"
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
