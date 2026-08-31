"""Diagnose whether one physical texture has a joint four-task ascent direction.

This is deliberately a diagnostic, not another weighted-sum attack.  Every term
is a differentiable proxy for the production metric and is oriented so that a
larger value means a worse prediction:

* pose: Sim(3)-aligned residual against TUM GT;
* depth: median-scale-aligned AbsRel against TUM GT depth;
* point: Sim(3)-aligned valid-pixel point residual against the GT point map;
* track: capped mean EPE against reprojection-GT tracks.

For each supplied texture state the script reports the four texture gradients,
their cosine matrix, and how much of every auxiliary gradient remains after its
component that conflicts with the pose gradient is removed.  A sizeable residual
means a pose-preserving auxiliary update exists locally; a near-zero residual
means that task is locally antiparallel to pose and gradient projection cannot
rescue it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG))

import attack_vggt_geometry_tum10 as A  # noqa: E402


SCENE = "rgbd_dataset_freiburg3_sitting_halfsphere"
MONITOR_QUAD = "0.3603,0.2939,0.5682,0.3063,0.5644,0.4943,0.3603,0.4795"
DEFAULT_STATES = (
    "init=init",
    "l64_al_s0=/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/"
    "l64_al_s0/geometry_patch/geometry_patch_texture.npz",
    "wsv_200=/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/"
    "wsv_200/geometry_patch/geometry_patch_texture.npz",
)


def build_args(scene: str, texture_size: int, track_iters: int) -> argparse.Namespace:
    argv = [
        "attack.py",
        "--tum_root", "/mnt/data/wangqq/recons_eval/data/tum",
        "--scene_pattern", scene,
        "--output_dir", "/tmp/multitask_gradient_diag",
        "--frame_manifest", str(
            VG / "data/tum_dynamics_10frame_individual_scenes/tum10_frame_manifest.json"
        ),
        "--ckpt", str(VG / "checkpoints/VGGT-1B"),
        "--texture_size", str(texture_size),
        "--texture_init", "image",
        "--texture_init_image", str(
            VG / "assets/hazard_textures/mde_attack_warnning.png"
        ),
        "--iterations", "1",
        "--inner_loop", "1",
        "--scenes_per_iteration", "1",
        "--attack_loss", "pose_aligned_residual_mse",
        "--pose_rotation_weight", "5.0",
        "--pose_translation_weight", "1.0",
        "--joint_track_iters", str(track_iters),
        "--joint_point_stride", "8",
        "--plane_mode", "depth_manual_quad_surface",
        "--manual_quad_xy", MONITOR_QUAD,
        "--manual_quad_coordinates", "normalized",
        "--manual_quad_depth_sample_stride", "1",
        "--manual_quad_fit_shrink", "0.75",
        "--manual_quad_plane_inlier_tolerance", "0.06",
        "--manual_quad_min_inlier_ratio", "0.60",
        "--surface_support_rel_tolerance", "0.01",
        "--surface_support_abs_tolerance", "0.10",
        "--fused_max_plane_residual", "0.02",
        "--surface_min_support_ratio", "0.50",
        "--surface_score_mode", "coverage",
        "--surface_coverage_min", "0.002",
        "--surface_coverage_max", "0.08",
        "--surface_min_visible_frames", "4",
        "--surface_min_visibility_ratio", "0.30",
        "--visibility_depth_margin", "0.08",
        "--seed", "0",
        "--use_depth_visibility",
        "--optimize_geometry",
        "--surface_support_check",
    ]
    saved, sys.argv = sys.argv, argv
    try:
        return A.parse_args()
    finally:
        sys.argv = saved


def load_texture(path: str, args: argparse.Namespace,
                 device: torch.device) -> torch.Tensor:
    if path == "init":
        return A.initialize_texture(args, device, requires_grad=True)
    with np.load(path) as data:
        arrays = [(key, data[key]) for key in data.files]
    candidates = []
    for key, value in arrays:
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[0] == 3:
            array = array[None]
        if array.ndim == 4 and array.shape[1] == 3:
            candidates.append((key, array))
    if not candidates:
        shapes = {key: tuple(np.asarray(value).shape) for key, value in arrays}
        raise RuntimeError(f"No [1,3,H,W] texture in {path}; arrays={shapes}")
    key, array = candidates[0]
    if tuple(array.shape[-2:]) != (args.texture_size, args.texture_size):
        raise RuntimeError(
            f"Texture {path}:{key} is {array.shape}, expected size {args.texture_size}"
        )
    return torch.from_numpy(array.astype(np.float32)).to(device).requires_grad_(True)


def gt_depth_from_world_points(gt: dict[str, np.ndarray]) -> np.ndarray:
    points = np.asarray(gt["point_map"], dtype=np.float64)
    c2w = np.asarray(gt["gt_c2w"], dtype=np.float64)
    valid = np.asarray(gt["point_valid"], dtype=bool)
    depth = np.zeros(valid.shape, dtype=np.float32)
    for frame in range(points.shape[0]):
        rotation = c2w[frame, :3, :3]
        translation = c2w[frame, :3, 3]
        camera = (points[frame] - translation) @ rotation
        z = camera[..., 2]
        keep = valid[frame] & np.isfinite(z) & (z > 1e-6)
        depth[frame, keep] = z[keep]
    return depth


def depth_absrel_score(pred: torch.Tensor, target: torch.Tensor,
                       valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    keep = valid & torch.isfinite(pred) & (pred > eps) & (target > eps)
    if int(keep.sum()) < 100:
        raise RuntimeError(f"Only {int(keep.sum())} valid depth pixels")
    p, g = pred[keep].float(), target[keep].float()
    # The production metric fits this global median scale.  Detaching it avoids
    # routing all gradients through the two selected median pixels.
    with torch.no_grad():
        scale = g.median() / p.median().clamp_min(eps)
    return ((p * scale - g).abs() / g.clamp_min(eps)).mean()


def capped_track_epe_score(pred: torch.Tensor, target: torch.Tensor,
                           visible: torch.Tensor, known: torch.Tensor,
                           cap_px: float) -> torch.Tensor:
    if pred.ndim == 4:
        pred = pred[0]
    keep = visible[1:] & known[1:]
    distance = (pred[1:] - target[1:]).float().pow(2).sum(-1).clamp_min(1e-12).sqrt()
    values = distance[keep]
    if values.numel() == 0:
        raise RuntimeError("No visible reprojection-GT track points")
    # tanh is linear around the clean 4-5 px error but suppresses the incentive
    # to spend the whole attack on a few 50+ px outliers.
    cap = max(float(cap_px), 1e-6)
    return cap * torch.tanh(values / cap).mean()


def task_scores(preds: dict[str, torch.Tensor], item: dict, gt: dict[str, np.ndarray],
                args: argparse.Namespace, device: torch.device,
                track_cap_px: float) -> dict[str, torch.Tensor]:
    pred_rel = A.pose_predictions_to_relative_c2w(preds, item["tensor_hw"])
    pose, _ = A.pose_aligned_residual_mse(pred_rel, item["pose_gt_rel"], args)

    depth_pred = A.dense_scalar_head(preds["depth"]).float()
    depth_gt_np = gt_depth_from_world_points(gt)
    depth_gt = torch.from_numpy(depth_gt_np).to(device)
    depth_valid = torch.from_numpy(np.asarray(gt["point_valid"], dtype=bool)).to(device)
    depth = depth_absrel_score(depth_pred, depth_gt, depth_valid)

    point_pred = preds["world_points"]
    if point_pred.ndim == 5:
        point_pred = point_pred[0]
    point_gt = torch.from_numpy(np.asarray(gt["point_map"], dtype=np.float32)).to(device)
    point_valid = torch.from_numpy(np.asarray(gt["point_valid"], dtype=bool)).to(device)
    # aligned_point_residual filters non-finite target rows; NaN is therefore an
    # explicit validity mask and prevents missing TUM depth from becoming (0,0,0).
    point_gt = point_gt.clone()
    point_gt[~point_valid] = float("nan")
    point = A.aligned_point_residual(
        point_pred.float(), point_gt, stride=args.joint_point_stride
    )

    track_gt = torch.from_numpy(np.asarray(gt["tracks"], dtype=np.float32)).to(device)
    track_visible = torch.from_numpy(
        np.asarray(gt["track_visible"], dtype=bool)
    ).to(device)
    track_known = torch.from_numpy(np.asarray(gt["track_known"], dtype=bool)).to(device)
    track = capped_track_epe_score(
        preds["track"], track_gt, track_visible, track_known, track_cap_px
    )
    return {"pose": pose, "depth": depth, "point": point, "track": track}


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-30) -> float:
    denom = a.norm() * b.norm()
    if float(denom) <= eps:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def diagnose_state(name: str, path: str, model: torch.nn.Module, item: dict,
                   gt: dict[str, np.ndarray], args: argparse.Namespace,
                   device: torch.device, dtype: torch.dtype,
                   track_cap_px: float) -> tuple[list[dict], dict]:
    texture = load_texture(path, args, device)
    adv = A.apply_geometry_patch(
        item["images"], texture, item["grids"], item["masks"], args, training=False
    )
    query = torch.from_numpy(np.asarray(gt["query_points"], dtype=np.float32)).to(device)
    preds = A.forward_all_geometry_heads(
        model, adv, dtype, query_points=query[None], track_iters=args.joint_track_iters
    )
    scores = task_scores(preds, item, gt, args, device, track_cap_px)
    names = list(scores)
    grads: dict[str, torch.Tensor] = {}
    for index, task in enumerate(names):
        grad = torch.autograd.grad(
            scores[task], texture, retain_graph=index < len(names) - 1
        )[0]
        grads[task] = grad.detach().float().flatten()

    cosines = {
        left: {right: cosine(grads[left], grads[right]) for right in names}
        for left in names
    }
    pose_grad = grads["pose"]
    pose_norm_sq = pose_grad.square().sum().clamp_min(1e-30)
    projected: dict[str, dict[str, float]] = {}
    projected_units = []
    for task in names[1:]:
        aux = grads[task]
        dot = torch.dot(aux, pose_grad)
        protected = aux
        if float(dot) < 0.0:
            protected = aux - dot / pose_norm_sq * pose_grad
        fraction = float(protected.norm() / aux.norm().clamp_min(1e-30))
        gain = cosine(aux, protected)
        projected[task] = {
            "original_cos_pose": cosines["pose"][task],
            "remaining_norm_fraction": fraction,
            "aux_gain_cos_after_projection": gain,
        }
        if float(protected.norm()) > 1e-30:
            projected_units.append(protected / protected.norm())

    pose_unit = pose_grad / pose_grad.norm().clamp_min(1e-30)
    combined = pose_unit.clone()
    if projected_units:
        combined = combined + torch.stack(projected_units).mean(0)
    combined /= combined.norm().clamp_min(1e-30)
    combined_gains = {task: cosine(grads[task], combined) for task in names}

    rows = []
    for task in names:
        rows.append({
            "state": name,
            "texture_path": path,
            "task": task,
            "score": float(scores[task].detach()),
            "grad_norm": float(grads[task].norm()),
            "cos_pose": cosines["pose"][task],
            "projected_remaining": (
                1.0 if task == "pose" else projected[task]["remaining_norm_fraction"]
            ),
            "combined_gain_cos": combined_gains[task],
        })
    detail = {
        "state": name,
        "texture_path": path,
        "scores": {task: float(value.detach()) for task, value in scores.items()},
        "gradient_norms": {task: float(value.norm()) for task, value in grads.items()},
        "cosines": cosines,
        "pose_protected_aux": projected,
        "combined_direction_gain_cos": combined_gains,
    }

    del preds, scores, grads, texture, adv
    torch.cuda.empty_cache()
    return rows, detail


def print_detail(detail: dict) -> None:
    names = ["pose", "depth", "point", "track"]
    print(f"\n=== {detail['state']} ===")
    print(f"{'task':<9}{'score':>12}{'|grad|':>14}{'cos(pose)':>13}"
          f"{'remain':>11}{'combined':>12}")
    for task in names:
        score = detail["scores"][task]
        norm = detail["gradient_norms"][task]
        cpose = detail["cosines"]["pose"][task]
        remain = 1.0 if task == "pose" else detail["pose_protected_aux"][task][
            "remaining_norm_fraction"
        ]
        gain = detail["combined_direction_gain_cos"][task]
        print(f"{task:<9}{score:>12.5g}{norm:>14.4e}{cpose:>13.3f}"
              f"{remain:>11.3f}{gain:>12.3f}")
    print("\ncosine matrix (all scores are ascent / larger-is-worse):")
    print(f"{'':<9}" + "".join(f"{name:>10}" for name in names))
    for left in names:
        print(f"{left:<9}" + "".join(
            f"{detail['cosines'][left][right]:>10.3f}" for right in names
        ))


def parse_state(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("state must be NAME=NPZ_PATH or NAME=init")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("state must be NAME=NPZ_PATH or NAME=init")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=SCENE)
    parser.add_argument("--texture_size", type=int, default=64)
    parser.add_argument("--track_iters", type=int, default=4)
    parser.add_argument("--track_cap_px", type=float, default=20.0)
    parser.add_argument("--state", action="append", default=None,
                        help="NAME=NPZ_PATH; repeat. Defaults: init, l64_al_s0, wsv_200")
    parser.add_argument("--out_dir", default=str(VG / "outputs/diag_multitask_gradients"))
    cli = parser.parse_args()

    states = [parse_state(spec) for spec in (cli.state or DEFAULT_STATES)]
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    args = build_args(cli.scene, cli.texture_size, cli.track_iters)

    model = A.load_model(args, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    scene_dirs = A.list_scene_dirs(Path(args.tum_root), args.scene_pattern)
    if len(scene_dirs) != 1:
        raise RuntimeError(f"Expected one scene, found {[p.name for p in scene_dirs]}")
    manifest = A.load_frame_manifest(args.frame_manifest)
    intrinsics = np.asarray(
        [[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]], dtype=np.float64
    )
    sequence = scene_dirs[0]
    with torch.no_grad():
        item = A.load_tum_sequence(
            sequence, manifest[sequence.name], args.gt_name, intrinsics,
            args.texture_size, args, device, model=model, dtype=dtype
        )
    gt_path = VG / "outputs/tum_gt_point_track" / f"{sequence.name}_gt.npz"
    with np.load(gt_path, allow_pickle=True) as data:
        gt = {key: data[key] for key in data.files}
    if tuple(gt["tensor_hw"].tolist()) != tuple(item["tensor_hw"]):
        raise RuntimeError(
            f"GT grid {tuple(gt['tensor_hw'])} != model grid {item['tensor_hw']}"
        )

    print(f"scene={sequence.name} grid={item['tensor_hw']} "
          f"coverage={item['geometry']['mask_coverage_mean']:.6f}")
    print(f"states={[name for name, _ in states]} track_points={len(gt['query_points'])}")

    all_rows, details = [], []
    for name, path in states:
        rows, detail = diagnose_state(
            name, path, model, item, gt, args, device, dtype, cli.track_cap_px
        )
        all_rows.extend(rows)
        details.append(detail)
        print_detail(detail)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gradient_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    json_path = out_dir / "gradient_details.json"
    json_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"\nSaved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
