"""Evaluate a physical patch on a complete TUM trajectory.

VGGT predicts every input window in an arbitrary local similarity gauge.  This
script runs overlapping windows and uses only poses predicted for shared frames
to map every new window into one trajectory.  Ground truth is deliberately not
available to the stitcher.  After stitching, Clean and Attack are each aligned
to GT exactly once by recons_eval's production Sim(3) metric.

The two stages use different Python environments::

    # VGGT inference (GPU)
    .../vggt/bin/python3 scripts/eval_stitched_pose.py infer ...

    # canonical ATE/RPE (CPU, recons_eval environment)
    .../recons_eval/bin/python3 scripts/eval_stitched_pose.py metrics ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCENES = (
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_sitting_rpy",
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
    "rgbd_dataset_freiburg3_walking_rpy",
    "rgbd_dataset_freiburg3_walking_halfsphere",
)


def make_windows(frame_indices: Iterable[int], window: int, stride: int) -> list[list[int]]:
    frames = [int(v) for v in frame_indices]
    if window < 2:
        raise ValueError("window must be >= 2")
    if not 0 < stride < window:
        raise ValueError("stride must satisfy 0 < stride < window so windows overlap")
    if len(frames) < window:
        raise ValueError(f"need at least {window} frames, got {len(frames)}")
    starts = list(range(0, len(frames) - window + 1, stride))
    final_start = len(frames) - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return [frames[start : start + window] for start in starts]


def load_planned_windows(path: str | Path, scene: str) -> tuple[list[int], list[list[int]]]:
    """Load one scene's frozen held-out windows and recover first-seen frame order."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = [entry for entry in raw if str(entry.get("scene")) == scene]
    if not entries:
        raise ValueError(f"Frame plan {path} has no entries for {scene}")
    windows = []
    selected = []
    seen = set()
    for entry in entries:
        if bool(entry.get("contains_training_frames", False)):
            raise ValueError(f"Frame plan entry {entry.get('name')} contains training frames")
        frames = [int(value) for value in entry["frames"]]
        if len(frames) != len(set(frames)):
            raise ValueError(f"Frame plan entry {entry.get('name')} contains duplicates")
        windows.append(frames)
        for frame in frames:
            if frame not in seen:
                selected.append(frame)
                seen.add(frame)
    for left, right in zip(windows, windows[1:]):
        if not set(left) & set(right):
            raise ValueError("Adjacent planned windows do not overlap and cannot be stitched")
    if selected != sorted(selected):
        raise ValueError("Planned held-out frames are not in chronological order")
    return selected, windows


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def estimate_prediction_sim3(
    local_c2w: np.ndarray,
    global_c2w: np.ndarray,
    fallback_scale: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Estimate local->global Sim(3) using predicted overlap poses only."""
    local = np.asarray(local_c2w, dtype=np.float64)
    global_ = np.asarray(global_c2w, dtype=np.float64)
    if local.shape != global_.shape or local.ndim != 3 or local.shape[1:] != (4, 4):
        raise ValueError(f"overlap pose shape mismatch: {local.shape} vs {global_.shape}")
    if len(local) < 1:
        raise ValueError("at least one overlapping pose is required")

    # A world-gauge rotation acts on every camera orientation as R_g = Q R_l.
    q_candidates = global_[:, :3, :3] @ np.swapaxes(local[:, :3, :3], -1, -2)
    q = nearest_rotation(q_candidates.sum(axis=0))

    x = local[:, :3, 3]
    y = global_[:, :3, 3]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    qx_centered = (q @ (x - x_mean).T).T
    y_centered = y - y_mean
    denominator = float(np.sum(qx_centered * qx_centered))
    numerator = float(np.sum(qx_centered * y_centered))
    scale_observable = denominator > 1e-10 and numerator > 1e-12
    scale = numerator / denominator if scale_observable else float(fallback_scale)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(fallback_scale)
        scale_observable = False
    # Prevent one nearly static overlap from exploding an otherwise valid chain.
    scale = float(np.clip(scale, 0.05, 20.0))
    translation = y_mean - scale * (q @ x_mean)

    mapped = apply_sim3_to_c2w(local, scale, q, translation)
    pos_residual = np.linalg.norm(mapped[:, :3, 3] - global_[:, :3, 3], axis=-1)
    rot_delta = mapped[:, :3, :3] @ np.swapaxes(global_[:, :3, :3], -1, -2)
    cos_angle = np.clip((np.trace(rot_delta, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    rot_residual = np.degrees(np.arccos(cos_angle))
    diagnostic = {
        "overlap": int(len(local)),
        "scale": scale,
        "scale_observable": bool(scale_observable),
        "local_overlap_rms": math.sqrt(denominator / max(len(local), 1)),
        "position_residual_rmse": float(np.sqrt(np.mean(pos_residual**2))),
        "rotation_residual_rmse_deg": float(np.sqrt(np.mean(rot_residual**2))),
    }
    return scale, q, translation, diagnostic


def apply_sim3_to_c2w(c2w: np.ndarray, scale: float, rotation: np.ndarray,
                      translation: np.ndarray) -> np.ndarray:
    poses = np.asarray(c2w, dtype=np.float64)
    out = np.array(poses, copy=True)
    out[:, :3, :3] = np.einsum("ij,njk->nik", rotation, poses[:, :3, :3])
    out[:, :3, 3] = scale * np.einsum("ij,nj->ni", rotation, poses[:, :3, 3])
    out[:, 3, :] = np.asarray([0.0, 0.0, 0.0, 1.0])
    return out


def stitch_windows(window_records: list[dict], pose_key: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Sequentially stitch windows; GT is neither accepted nor referenced."""
    if not window_records:
        raise ValueError("no windows to stitch")
    first_frames = [int(v) for v in window_records[0]["frames"]]
    first_poses = np.asarray(window_records[0][pose_key], dtype=np.float64)
    if len(first_frames) != len(first_poses):
        raise ValueError("first window frame/pose length mismatch")
    stitched = {frame: pose.copy() for frame, pose in zip(first_frames, first_poses)}
    ordered_frames = list(first_frames)
    diagnostics = [{"window": 0, "frames": first_frames, "overlap": 0, "scale": 1.0}]
    previous_scale = 1.0

    for window_idx, record in enumerate(window_records[1:], start=1):
        frames = [int(v) for v in record["frames"]]
        poses = np.asarray(record[pose_key], dtype=np.float64)
        if len(frames) != len(poses):
            raise ValueError(f"window {window_idx}: frame/pose length mismatch")
        overlap_frames = [frame for frame in frames if frame in stitched]
        if not overlap_frames:
            raise ValueError(f"window {window_idx} has no overlap with the stitched trajectory")
        local_lookup = {frame: i for i, frame in enumerate(frames)}
        local_overlap = np.stack([poses[local_lookup[frame]] for frame in overlap_frames])
        global_overlap = np.stack([stitched[frame] for frame in overlap_frames])
        scale, rotation, translation, diagnostic = estimate_prediction_sim3(
            local_overlap, global_overlap, fallback_scale=previous_scale)
        previous_scale = scale
        mapped = apply_sim3_to_c2w(poses, scale, rotation, translation)
        for frame, pose in zip(frames, mapped):
            if frame not in stitched:
                stitched[frame] = pose
                ordered_frames.append(frame)
        diagnostic.update({
            "window": window_idx,
            "frames": frames,
            "overlap_frames": overlap_frames,
        })
        diagnostics.append(diagnostic)

    if len(ordered_frames) != len(set(ordered_frames)):
        raise AssertionError("duplicate stitched frame")
    return (
        np.asarray(ordered_frames, dtype=np.int64),
        np.stack([stitched[frame] for frame in ordered_frames]),
        diagnostics,
    )


def build_attack_args(attack, args: argparse.Namespace):
    old_argv = sys.argv
    try:
        sys.argv = [
            "attack_vggt_geometry_tum10.py",
            "--tum_root", args.tum_root,
            "--output_dir", args.output_root,
            "--frame_manifest", args.protocol,
            "--ckpt", args.ckpt,
            "--gt_name", args.gt_name,
            "--attack_loss", "feature_l1",
            "--plane_mode", "explicit_world_quad",
            "--explicit_quad_world=" + Path(args.quad_file).read_text().strip(),
            "--use_depth_visibility",
            "--visibility_depth_margin", str(args.visibility_depth_margin),
            "--local_files_only",
        ]
        render_args = attack.parse_args()
    finally:
        sys.argv = old_argv
    # Explicit-world rendering must use synchronized TUM GT, never an old
    # fixed-ten-frame clean output.  No training-only branch is needed here.
    render_args.clean_vggt_output_root = None
    render_args.surface_strength_search = False
    render_args.physical_eot = False
    return render_args


def infer(args: argparse.Namespace) -> None:
    import torch
    import attack_vggt_geometry_tum10 as attack

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("VGGT inference requires a CUDA device")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    protocol = json.loads(Path(args.protocol).read_text())
    render_args = build_attack_args(attack, args)
    model = attack.load_model(render_args, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    texture = attack.load_texture(args.texture, device, render_args)
    intrinsics = np.asarray(
        [[render_args.fx, 0.0, render_args.cx],
         [0.0, render_args.fy, render_args.cy],
         [0.0, 0.0, 1.0]], dtype=np.float64)

    scene_names = args.scenes or list(protocol["scenes"])
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for scene_name in scene_names:
        scene_dir = Path(args.tum_root) / scene_name
        all_images = attack.find_images(scene_dir)
        count = len(all_images)
        split = count // 2
        if args.frame_plan:
            selected, windows = load_planned_windows(args.frame_plan, scene_name)
            if selected[-1] >= count:
                raise IndexError(
                    f"Planned frame {selected[-1]} exceeds {scene_name} length {count}"
                )
            if args.max_frames:
                raise ValueError("--max_frames cannot be combined with --frame_plan")
        else:
            if args.split == "unseen":
                selected = list(range(split, count))
            elif args.split == "train":
                selected = list(range(0, split))
            else:
                selected = list(range(count))
            if args.max_frames:
                selected = selected[: args.max_frames]
            windows = make_windows(selected, args.window, args.stride)
        scene_out = output_root / scene_name
        windows_dir = scene_out / "windows"
        windows_dir.mkdir(parents=True, exist_ok=True)
        print(f"[scene] {scene_name}: {len(selected)} frames, {len(windows)} windows", flush=True)

        for window_idx, frames in enumerate(windows):
            window_path = windows_dir / f"w{window_idx:04d}.npz"
            if args.resume and window_path.exists():
                with np.load(window_path) as old:
                    if old["frames"].tolist() == frames:
                        print(f"[window] {scene_name} {window_idx + 1}/{len(windows)} cached", flush=True)
                        continue
            item = attack.load_tum_sequence(
                scene_dir, frames, args.gt_name, intrinsics, int(texture.shape[-1]),
                render_args, device, model=model, dtype=dtype)
            adv_images = attack.apply_geometry_patch(
                item["images"], texture, item["grids"], item["masks"],
                render_args, training=False).detach()
            with torch.inference_mode():
                clean_pred = attack.forward_camera_pose_only(model, item["images"], dtype)
                attack_pred = attack.forward_camera_pose_only(model, adv_images, dtype)
                clean_c2w = attack.pose_predictions_to_relative_c2w(
                    clean_pred, item["tensor_hw"])[0].float().cpu().numpy()
                attack_c2w = attack.pose_predictions_to_relative_c2w(
                    attack_pred, item["tensor_hw"])[0].float().cpu().numpy()
            np.savez_compressed(
                window_path,
                frames=np.asarray(frames, dtype=np.int64),
                clean_c2w=clean_c2w.astype(np.float32),
                attack_c2w=attack_c2w.astype(np.float32),
                coverage=np.asarray(float(item["masks"].float().mean().cpu())),
            )
            print(
                f"[window] {scene_name} {window_idx + 1}/{len(windows)} "
                f"frames={frames[0]}..{frames[-1]} coverage={float(item['masks'].float().mean()):.5f}",
                flush=True,
            )
            del item, adv_images, clean_pred, attack_pred

        records = []
        for window_idx, frames in enumerate(windows):
            with np.load(windows_dir / f"w{window_idx:04d}.npz") as data:
                actual = data["frames"].astype(np.int64).tolist()
                if actual != frames:
                    raise ValueError(f"cached window {window_idx} frame mismatch")
                records.append({
                    "frames": actual,
                    "clean_c2w": data["clean_c2w"].astype(np.float64),
                    "attack_c2w": data["attack_c2w"].astype(np.float64),
                    "coverage": float(data["coverage"]),
                })
        clean_frames, clean_c2w, clean_diag = stitch_windows(records, "clean_c2w")
        attack_frames, attack_c2w, attack_diag = stitch_windows(records, "attack_c2w")
        if not np.array_equal(clean_frames, attack_frames) or clean_frames.tolist() != selected:
            raise AssertionError("stitched Clean/Attack frames do not exactly cover the selected split")
        gt_c2w = attack.tum_rows_to_c2w(
            attack.read_tum_rows(scene_dir / args.gt_name), clean_frames.tolist())
        np.savez_compressed(
            scene_out / "stitched_trajectories.npz",
            frame_indices=clean_frames,
            clean_c2w=clean_c2w.astype(np.float32),
            attack_c2w=attack_c2w.astype(np.float32),
            gt_c2w=gt_c2w.astype(np.float32),
        )
        diagnostics = {
            "scene": scene_name,
            "split": args.split,
            "frame_plan": str(args.frame_plan) if args.frame_plan else None,
            "window": args.window,
            "stride": args.stride,
            "n_frames": len(selected),
            "n_windows": len(windows),
            "gt_used_for_stitching": False,
            "final_gt_alignments_per_condition": 1,
            "mean_patch_coverage": float(np.mean([r["coverage"] for r in records])),
            "clean": clean_diag,
            "attack": attack_diag,
        }
        (scene_out / "stitch_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8")
        print(f"[stitched] {scene_name}: {len(selected)} poses -> {scene_out}", flush=True)


def write_selected_gt(gt_rows: list[list[str]], frame_indices: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for order, frame_idx in enumerate(frame_indices):
            row = list(gt_rows[int(frame_idx)])
            row[0] = f"{float(order):.6f}"
            handle.write(" ".join(row) + "\n")


def read_tum_rows(path: Path) -> list[list[str]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(line.split())
    return rows


def metrics(args: argparse.Namespace) -> None:
    recons_root = Path(args.recons_root)
    sys.path.insert(0, str(recons_root.resolve()))
    from relpose.evo_utils import eval_metrics, get_tum_poses, load_traj

    root = Path(args.output_root)
    scene_dirs = [root / name for name in (args.scenes or SCENES)]
    rows = []
    for scene_dir in scene_dirs:
        traj_path = scene_dir / "stitched_trajectories.npz"
        if not traj_path.exists():
            print(f"[metrics] skip missing {traj_path}", flush=True)
            continue
        with np.load(traj_path) as data:
            frames = data["frame_indices"].astype(np.int64).tolist()
            clean_c2w = data["clean_c2w"].astype(np.float64)
            attack_c2w = data["attack_c2w"].astype(np.float64)
            gt_c2w = data["gt_c2w"].astype(np.float64)
        selected_gt = scene_dir / "groundtruth_selected.txt"
        write_selected_gt(
            read_tum_rows(Path(args.tum_root) / scene_dir.name / args.gt_name),
            frames, selected_gt)
        gt_traj = load_traj(str(selected_gt), traj_format="tum", stride=1)
        center = gt_c2w[:, :3, 3]
        ceiling = float(np.sqrt(np.mean(np.sum((center - center.mean(axis=0)) ** 2, axis=1))))
        condition_results = {}
        for condition, c2w in (("clean", clean_c2w), ("attack", attack_c2w)):
            ate, rpe_t, rpe_r = eval_metrics(
                get_tum_poses(c2w), gt_traj, seq=f"{scene_dir.name}/{condition}",
                filename=str(scene_dir / f"{condition}_eval_metric.txt"), verbose=False)
            condition_results[condition] = {
                "ATE": float(ate),
                "ATE_ceiling_pct": float(100.0 * ate / max(ceiling, 1e-12)),
                "RPE_trans": float(rpe_t),
                "RPE_rot_deg": float(rpe_r),
            }
        row = {
            "scene": scene_dir.name,
            "n_frames": len(frames),
            "ATE_ceiling": ceiling,
            "clean_ATE": condition_results["clean"]["ATE"],
            "attack_ATE": condition_results["attack"]["ATE"],
            "clean_ATE_ceiling_pct": condition_results["clean"]["ATE_ceiling_pct"],
            "attack_ATE_ceiling_pct": condition_results["attack"]["ATE_ceiling_pct"],
            "clean_RPE_trans": condition_results["clean"]["RPE_trans"],
            "attack_RPE_trans": condition_results["attack"]["RPE_trans"],
            "clean_RPE_rot_deg": condition_results["clean"]["RPE_rot_deg"],
            "attack_RPE_rot_deg": condition_results["attack"]["RPE_rot_deg"],
        }
        row["ATE_ratio"] = row["attack_ATE"] / max(row["clean_ATE"], 1e-12)
        row["RPE_trans_ratio"] = row["attack_RPE_trans"] / max(row["clean_RPE_trans"], 1e-12)
        row["RPE_rot_ratio"] = row["attack_RPE_rot_deg"] / max(row["clean_RPE_rot_deg"], 1e-12)
        rows.append(row)
        (scene_dir / "pose_metrics.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8")
        print(
            f"[metrics] {scene_dir.name}: ATE {row['clean_ATE']:.6f}->{row['attack_ATE']:.6f} "
            f"({row['clean_ATE_ceiling_pct']:.2f}%->{row['attack_ATE_ceiling_pct']:.2f}%), "
            f"RPE-r {row['clean_RPE_rot_deg']:.4f}->{row['attack_RPE_rot_deg']:.4f}",
            flush=True,
        )
    if not rows:
        raise SystemExit("no stitched trajectories found")
    csv_path = root / "stitched_pose_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    macro = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key not in ("scene", "n_frames")
    }
    macro["n_scenes"] = len(rows)
    macro["total_frames"] = int(sum(row["n_frames"] for row in rows))
    (root / "stitched_pose_macro.json").write_text(
        json.dumps(macro, indent=2), encoding="utf-8")
    print(f"[metrics] saved {csv_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output_root", required=True)
    common.add_argument("--tum_root", default="/mnt/data/wangqq/recons_eval/data/tum_all_sync")
    common.add_argument("--gt_name", default="groundtruth.txt")
    common.add_argument("--scenes", nargs="*", choices=SCENES)

    p_infer = sub.add_parser("infer", parents=[common])
    p_infer.add_argument("--protocol", default="/mnt/data/wangqq/vggt/data/tum_universal_all/protocol.json")
    p_infer.add_argument("--texture", required=True)
    p_infer.add_argument("--quad_file", default="/mnt/data/wangqq/vggt/data/tum10_split/plane_quad_world.txt")
    p_infer.add_argument("--ckpt", default="facebook/VGGT-1B")
    p_infer.add_argument("--device", default="cuda:0")
    p_infer.add_argument("--split", choices=("unseen", "train", "all"), default="unseen")
    p_infer.add_argument(
        "--window", type=int, default=30,
        help="frames jointly sent to VGGT at test time (formal protocol: 30)")
    p_infer.add_argument(
        "--stride", type=int, default=20,
        help="window step (formal protocol: 20, hence at least 10 shared frames)")
    p_infer.add_argument("--visibility_depth_margin", type=float, default=0.08)
    p_infer.add_argument("--max_frames", type=int, default=0)
    p_infer.add_argument("--resume", action="store_true")
    p_infer.add_argument(
        "--frame_plan", default="",
        help="Optional frozen JSON eval plan; overrides the built-in chronological split.")

    p_metrics = sub.add_parser("metrics", parents=[common])
    p_metrics.add_argument("--recons_root", default="/mnt/data/wangqq/recons_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "infer":
        infer(args)
    else:
        metrics(args)


if __name__ == "__main__":
    main()
