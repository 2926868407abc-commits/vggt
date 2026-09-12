#!/usr/bin/env python3
"""Evaluate one already-trained patch on all four VGGT tasks.

This is the dense-task companion to ``eval_stitched_pose.py``.  It deliberately
does not train or update a patch.  Clean and attacked images are inferred in the
same 30-frame windows used by the stitched-pose protocol.  Depth, point-map and
reprojection-track GT are rebuilt from the exact TUM frames in each window.

With ``--frame_plan``, the all-head forward also supplies pose for prediction-only
window stitching.  Thus Clean and Attack each run VGGT only once per window while
the complete held-out trajectory still receives exactly one final Sim(3).
The older ``--pose_root`` path remains available for cached legacy evaluations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# open3d lives in the evaluation environment.  Both environments use CPython
# 3.10 and NumPy 2.2.6; append rather than prepend so torch keeps its own stack.
RECONS_SITE = Path("/mnt/data/wangqq/conda_envs/recons_eval/lib/python3.10/site-packages")
if RECONS_SITE.is_dir():
    sys.path.append(str(RECONS_SITE))
RECONS_ROOT = Path("/mnt/data/wangqq/recons_eval")
if RECONS_ROOT.is_dir():
    sys.path.append(str(RECONS_ROOT))

import attack_vggt_geometry_tum10 as attack  # noqa: E402
from diag_gauge_invariance import (  # noqa: E402
    TUM_CX,
    TUM_CY,
    TUM_FX,
    TUM_FY,
    build_gt_world_points,
    match_depth_paths,
    preprocess_tum_depth_to_vggt_grid,
    projection_params,
)
from eval_stitched_pose import (  # noqa: E402
    SCENES,
    build_attack_args,
    load_planned_windows,
    stitch_windows,
)
from mv_recon.utils import accuracy, completion, umeyama  # noqa: E402


def _seed(scene: str, window_index: int) -> int:
    digest = hashlib.sha256(f"{scene}:{window_index}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def make_query_grid(tensor_hw: tuple[int, int], rows: int, cols: int,
                    margin: float) -> np.ndarray:
    """Match build_tum_gt_pool.py exactly (including h/w rather than h-1/w-1)."""
    h, w = tensor_hw
    ys = np.linspace(margin * h, (1.0 - margin) * h, rows)
    xs = np.linspace(margin * w, (1.0 - margin) * w, cols)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)


def sample_depth(depth: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    x = np.rint(xy[:, 0]).astype(np.int64)
    y = np.rint(xy[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    values = np.zeros(len(xy), dtype=np.float64)
    values[inside] = depth[y[inside], x[inside]]
    return values, inside


def project_world(world: np.ndarray, c2w: np.ndarray,
                  proj: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(c2w)
    camera = world @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = TUM_FX * camera[:, 0] / safe + TUM_CX
    v = TUM_FY * camera[:, 1] / safe + TUM_CY
    xy = np.stack(
        [u * proj["scale_x"], v * proj["scale_y"] - proj["crop_y"]], axis=-1
    )
    return xy, z


def build_window_gt(item: dict, rows: int, cols: int, margin: float,
                    occl_tol: float) -> dict[str, np.ndarray]:
    tensor_hw = tuple(item["tensor_hw"])
    depth_paths = match_depth_paths(item["seq_dir"], item["image_paths"])
    if any(path is None for path in depth_paths):
        missing = [item["frame_indices"][i] for i, path in enumerate(depth_paths)
                   if path is None]
        raise RuntimeError(f"depth association failed for frames {missing}")
    depth = np.stack([
        preprocess_tum_depth_to_vggt_grid(path, Path(image), tensor_hw)
        for path, image in zip(depth_paths, item["image_paths"])
    ]).astype(np.float32)
    proj = projection_params(Path(item["image_paths"][0]), tensor_hw)
    point_map, point_valid = build_gt_world_points(
        [frame for frame in depth], np.asarray(item["c2w_gt"]), proj
    )

    query_all = make_query_grid(tensor_hw, rows, cols, margin)
    query_depth, query_inside = sample_depth(depth[0], query_all)
    query_keep = query_inside & np.isfinite(query_depth) & (query_depth > 1e-6)
    query = query_all[query_keep]
    query_depth = query_depth[query_keep]
    if len(query) < 20:
        raise RuntimeError(f"query frame has only {len(query)} valid track points")

    u = query[:, 0] / proj["scale_x"]
    v = (query[:, 1] + proj["crop_y"]) / proj["scale_y"]
    camera = np.stack([
        (u - TUM_CX) / TUM_FX * query_depth,
        (v - TUM_CY) / TUM_FY * query_depth,
        query_depth,
    ], axis=-1)
    q_c2w = np.asarray(item["c2w_gt"][0], dtype=np.float64)
    world = camera @ q_c2w[:3, :3].T + q_c2w[:3, 3]

    tracks = np.zeros((len(depth), len(query), 2), dtype=np.float32)
    known = np.zeros((len(depth), len(query)), dtype=bool)
    visible = np.zeros((len(depth), len(query)), dtype=bool)
    for i, c2w in enumerate(item["c2w_gt"]):
        xy, z = project_world(world, np.asarray(c2w), proj)
        scene_depth, inside = sample_depth(depth[i], xy)
        has_depth = inside & np.isfinite(scene_depth) & (scene_depth > 1e-6)
        in_front = np.isfinite(z) & (z > 1e-6)
        tracks[i] = xy.astype(np.float32)
        known[i] = has_depth & in_front
        visible[i] = has_depth & in_front & (z <= scene_depth + occl_tol)

    roundtrip = float(np.max(np.abs(tracks[0] - query)))
    if not np.isfinite(roundtrip) or roundtrip > 1e-3:
        raise RuntimeError(f"track GT round-trip failed: {roundtrip:.6g}px")
    return {
        "depth": depth,
        "point_map": point_map,
        "point_valid": point_valid,
        "query_points": query.astype(np.float32),
        "tracks": tracks,
        "track_known": known,
        "track_visible": visible,
        "roundtrip_max_px": np.asarray(roundtrip),
    }


def depth_stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    valid = (gt > 1e-6) & (pred > 1e-6) & np.isfinite(gt) & np.isfinite(pred)
    count = int(valid.sum())
    if count < 100:
        raise RuntimeError(f"only {count} valid depth pixels")
    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    scale = float(np.median(g) / max(float(np.median(p)), 1e-9))
    p *= scale
    absrel = np.abs(p - g) / g
    ratio = np.maximum(p / g, g / p)
    return {
        "absrel": float(absrel.mean()),
        "d1": float(np.mean(ratio < 1.25)),
        "valid": count,
        "scale": scale,
    }


def point_stats(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray,
                max_points: int, rng: np.random.Generator,
                icp_threshold: float) -> dict[str, float | int]:
    pred_flat = np.asarray(pred).reshape(-1, 3)
    gt_flat = np.asarray(gt).reshape(-1, 3)
    keep = np.asarray(valid).reshape(-1)
    keep &= np.isfinite(pred_flat).all(axis=1) & np.isfinite(gt_flat).all(axis=1)
    indices = np.flatnonzero(keep)
    if len(indices) > max_points:
        indices = np.sort(rng.choice(indices, max_points, replace=False))
    if len(indices) < 100:
        raise RuntimeError(f"only {len(indices)} valid point-map points")
    p = pred_flat[indices].astype(np.float64)
    g = gt_flat[indices].astype(np.float64)
    scale, rotation, translation = umeyama(p.T, g.T)
    p = float(scale) * (p @ rotation.T) + translation[:, 0]

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(g)
    reg = o3d.pipelines.registration.registration_icp(
        pcd, pcd_gt, icp_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pcd.transform(reg.transformation)
    acc, acc_med = accuracy(pcd_gt.points, pcd.points)
    comp, comp_med = completion(pcd_gt.points, pcd.points)
    return {
        "acc": float(acc),
        "acc_median": float(acc_med),
        "comp": float(comp),
        "comp_median": float(comp_med),
        "points": int(len(indices)),
        "umeyama_scale": float(scale),
        "icp_fitness": float(reg.fitness),
    }


def track_stats(pred_xy: np.ndarray, pred_vis: np.ndarray, gt: dict,
                min_points: int) -> dict[str, float | int]:
    visible = gt["track_visible"]
    known = gt["track_known"]
    selected = visible[1:] & known[1:]
    per_frame = selected.sum(axis=1)
    usable = per_frame >= min_points
    if not usable.any():
        return {"epe_mean": math.nan, "epe_median": math.nan, "points": 0,
                "frames": 0, "vis_f1": math.nan, "tp": 0, "fp": 0, "fn": 0}
    errors = np.linalg.norm(pred_xy[1:] - gt["tracks"][1:], axis=-1)
    values = errors[usable][selected[usable]]
    k = known[1:][usable]
    pv = (pred_vis[1:][usable] > 0.5) & k
    gv = selected[usable]
    tp = int((pv & gv).sum())
    fp = int((pv & ~gv & k).sum())
    fn = int((~pv & gv).sum())
    return {
        "epe_mean": float(values.mean()),
        "epe_median": float(np.median(values)),
        "points": int(len(values)),
        "frames": int(usable.sum()),
        "vis_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def extract_predictions(model: torch.nn.Module, images: torch.Tensor,
                        query: np.ndarray, device: torch.device,
                        dtype: torch.dtype,
                        tensor_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    query_t = torch.as_tensor(query, device=device, dtype=torch.float32)
    with torch.inference_mode(), attack.autocast_context(device, dtype):
        pred = model(images, query_points=query_t)
    result = {
        "c2w": attack.pose_predictions_to_relative_c2w(
            pred, tensor_hw)[0].float().cpu().numpy(),
        "depth": pred["depth"][0, ..., 0].float().cpu().numpy(),
        "point_map": pred["world_points"][0].float().cpu().numpy(),
        "track": pred["track"][0].float().cpu().numpy(),
        "track_vis": pred["vis"][0].float().cpu().numpy(),
    }
    del pred
    return result


def score_condition(pred: dict[str, np.ndarray], gt: dict, args: argparse.Namespace,
                    rng: np.random.Generator) -> dict:
    return {
        "depth": depth_stats(pred["depth"], gt["depth"]),
        "point": point_stats(pred["point_map"], gt["point_map"], gt["point_valid"],
                             args.max_point_samples, rng, args.icp_threshold),
        "track": track_stats(pred["track"], pred["track_vis"], gt,
                             args.min_track_points),
    }


def infer(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    render_args = build_attack_args(attack, args)
    model = attack.load_model(render_args, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    texture = attack.load_texture(args.texture, device, render_args)
    intrinsics = np.asarray([
        [render_args.fx, 0.0, render_args.cx],
        [0.0, render_args.fy, render_args.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    pose_root = Path(args.pose_root) if args.pose_root else None
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        if args.frame_plan:
            selected_frames, planned_windows = load_planned_windows(
                args.frame_plan, scene)
            if args.max_windows:
                planned_windows = planned_windows[:args.max_windows]
                selected_frames = []
                seen = set()
                for frames in planned_windows:
                    for frame in frames:
                        if frame not in seen:
                            selected_frames.append(frame)
                            seen.add(frame)
            source_windows = [None] * len(planned_windows)
        else:
            if pose_root is None:
                raise ValueError("Either --frame_plan or --pose_root is required")
            source_windows = sorted((pose_root / scene / "windows").glob("w*.npz"))
            if args.max_windows:
                source_windows = source_windows[:args.max_windows]
            if not source_windows:
                raise FileNotFoundError(f"no pose windows for {scene}")
            planned_windows = []
            selected_frames = []
            seen = set()
            for source in source_windows:
                with np.load(source) as stored:
                    frames = [int(v) for v in stored["frames"]]
                planned_windows.append(frames)
                for frame in frames:
                    if frame not in seen:
                        selected_frames.append(frame)
                        seen.add(frame)
        scene_out = output_root / scene
        windows_out = scene_out / "windows"
        pose_windows_out = scene_out / "pose_windows"
        windows_out.mkdir(parents=True, exist_ok=True)
        pose_windows_out.mkdir(parents=True, exist_ok=True)
        print(f"[scene] {scene}: {len(source_windows)} windows", flush=True)
        for window_index, (source, frames) in enumerate(
                zip(source_windows, planned_windows)):
            out = windows_out / f"w{window_index:04d}.json"
            pose_out = pose_windows_out / f"w{window_index:04d}.npz"
            if args.resume and out.exists() and pose_out.exists():
                old = json.loads(out.read_text(encoding="utf-8"))
                with np.load(pose_out) as old_pose:
                    pose_frames = old_pose["frames"].astype(np.int64).tolist()
                if old.get("frames") == frames and pose_frames == frames:
                    print(f"[window] {scene} {window_index + 1}/{len(source_windows)} cached",
                          flush=True)
                    continue
            item = attack.load_tum_sequence(
                Path(args.tum_root) / scene, frames, args.gt_name, intrinsics,
                int(texture.shape[-1]), render_args, device, model=model, dtype=dtype,
            )
            gt = build_window_gt(item, args.query_rows, args.query_cols,
                                 args.query_margin, args.occl_tol)
            adv = attack.apply_geometry_patch(
                item["images"], texture, item["grids"], item["masks"],
                render_args, training=False,
            ).detach()
            clean_pred = extract_predictions(
                model, item["images"], gt["query_points"], device, dtype,
                tuple(item["tensor_hw"]))
            attack_pred = extract_predictions(
                model, adv, gt["query_points"], device, dtype,
                tuple(item["tensor_hw"]))
            seed = _seed(scene, window_index)
            record = {
                "scene": scene,
                "window": window_index,
                "frames": frames,
                "contains_training_frames": False,
                "coverage": float(item["masks"].float().mean().cpu()),
                "query_points": int(len(gt["query_points"])),
                "gt_roundtrip_max_px": float(gt["roundtrip_max_px"]),
                "clean": score_condition(
                    clean_pred, gt, args, np.random.default_rng(seed)),
                "attack": score_condition(
                    attack_pred, gt, args, np.random.default_rng(seed)),
            }
            out.write_text(json.dumps(record, indent=2), encoding="utf-8")
            np.savez_compressed(
                pose_out,
                frames=np.asarray(frames, dtype=np.int64),
                clean_c2w=clean_pred["c2w"].astype(np.float32),
                attack_c2w=attack_pred["c2w"].astype(np.float32),
                coverage=np.asarray(record["coverage"], dtype=np.float32),
            )
            print(
                f"[window] {scene} {window_index + 1}/{len(source_windows)} "
                f"frames={frames[0]}..{frames[-1]} "
                f"AbsRel={record['clean']['depth']['absrel']:.4f}->"
                f"{record['attack']['depth']['absrel']:.4f} "
                f"Acc={record['clean']['point']['acc']:.4f}->"
                f"{record['attack']['point']['acc']:.4f} "
                f"EPE={record['clean']['track']['epe_mean']:.2f}->"
                f"{record['attack']['track']['epe_mean']:.2f}", flush=True,
            )
            del item, gt, adv, clean_pred, attack_pred
            torch.cuda.empty_cache()

        pose_records = []
        for window_index, frames in enumerate(planned_windows):
            with np.load(pose_windows_out / f"w{window_index:04d}.npz") as stored:
                actual = stored["frames"].astype(np.int64).tolist()
                if actual != frames:
                    raise ValueError(f"cached pose window {window_index} frame mismatch")
                pose_records.append({
                    "frames": actual,
                    "clean_c2w": stored["clean_c2w"].astype(np.float64),
                    "attack_c2w": stored["attack_c2w"].astype(np.float64),
                    "coverage": float(stored["coverage"]),
                })
        clean_frames, clean_c2w, clean_diag = stitch_windows(
            pose_records, "clean_c2w")
        attack_frames, attack_c2w, attack_diag = stitch_windows(
            pose_records, "attack_c2w")
        if (not np.array_equal(clean_frames, attack_frames)
                or clean_frames.tolist() != selected_frames):
            raise AssertionError(
                "Unified all-head pose windows do not exactly cover the held-out split")
        gt_c2w = attack.tum_rows_to_c2w(
            attack.read_tum_rows(Path(args.tum_root) / scene / args.gt_name),
            clean_frames.tolist())
        np.savez_compressed(
            scene_out / "stitched_trajectories.npz",
            frame_indices=clean_frames,
            clean_c2w=clean_c2w.astype(np.float32),
            attack_c2w=attack_c2w.astype(np.float32),
            gt_c2w=gt_c2w.astype(np.float32),
        )
        (scene_out / "stitch_diagnostics.json").write_text(
            json.dumps({
                "scene": scene,
                "frame_plan": str(args.frame_plan) if args.frame_plan else None,
                "pose_source": "same all-head forward as depth/point/track",
                "n_frames": len(selected_frames),
                "n_windows": len(planned_windows),
                "gt_used_for_stitching": False,
                "final_gt_alignments_per_condition": 1,
                "clean": clean_diag,
                "attack": attack_diag,
            }, indent=2), encoding="utf-8")
        print(
            f"[stitched] {scene}: {len(selected_frames)} all-head poses -> {scene_out}",
            flush=True)


def weighted(rows: list[dict], section: str, value: str, weight: str) -> float:
    pairs = [(float(row[section][value]), int(row[section][weight])) for row in rows]
    pairs = [(v, w) for v, w in pairs if np.isfinite(v) and w > 0]
    return float(sum(v * w for v, w in pairs) / sum(w for _, w in pairs))


def aggregate_condition(records: list[dict], condition: str) -> dict[str, float | int]:
    rows = [record[condition] for record in records]
    tp = sum(int(row["track"]["tp"]) for row in rows)
    fp = sum(int(row["track"]["fp"]) for row in rows)
    fn = sum(int(row["track"]["fn"]) for row in rows)
    return {
        "depth_absrel": weighted(rows, "depth", "absrel", "valid"),
        "depth_d1": weighted(rows, "depth", "d1", "valid"),
        "point_acc": weighted(rows, "point", "acc", "points"),
        "point_comp": weighted(rows, "point", "comp", "points"),
        "track_epe_mean": weighted(rows, "track", "epe_mean", "points"),
        "track_vis_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "depth_pixels": sum(int(row["depth"]["valid"]) for row in rows),
        "point_samples": sum(int(row["point"]["points"]) for row in rows),
        "track_points": sum(int(row["track"]["points"]) for row in rows),
    }


def summarize(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    pose_rows = {}
    with Path(args.pose_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pose_rows[row["scene"]] = row
    rows = []
    for scene in args.scenes:
        files = sorted((root / scene / "windows").glob("w*.json"))
        if not files:
            raise FileNotFoundError(f"no dense-task results for {scene}")
        records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        clean = aggregate_condition(records, "clean")
        attacked = aggregate_condition(records, "attack")
        pose = pose_rows[scene]
        row = {
            "scene": scene,
            "n_windows": len(records),
            "n_unique_frames": len({f for record in records for f in record["frames"]}),
            "mean_coverage": float(np.mean([record["coverage"] for record in records])),
            "clean_ATE": float(pose["clean_ATE"]),
            "attack_ATE": float(pose["attack_ATE"]),
            "clean_RPE_trans": float(pose["clean_RPE_trans"]),
            "attack_RPE_trans": float(pose["attack_RPE_trans"]),
            "clean_RPE_rot_deg": float(pose["clean_RPE_rot_deg"]),
            "attack_RPE_rot_deg": float(pose["attack_RPE_rot_deg"]),
        }
        for key in ("depth_absrel", "depth_d1", "point_acc", "point_comp",
                    "track_epe_mean", "track_vis_f1"):
            row[f"clean_{key}"] = clean[key]
            row[f"attack_{key}"] = attacked[key]
        rows.append(row)
        (root / scene / "four_task_summary.json").write_text(
            json.dumps({"scene": scene, "clean": clean, "attack": attacked,
                        "pose": pose}, indent=2), encoding="utf-8")

    csv_path = root / "four_task_by_scene.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [key for key in rows[0] if key.startswith("clean_") or key.startswith("attack_")]
    macro = {key: float(np.mean([float(row[key]) for row in rows])) for key in numeric}
    macro["n_scenes"] = len(rows)
    macro["total_unique_frames"] = int(sum(row["n_unique_frames"] for row in rows))
    (root / "four_task_macro.json").write_text(
        json.dumps(macro, indent=2), encoding="utf-8")
    print(f"[summary] {csv_path}", flush=True)
    for row in rows:
        print(
            f"{row['scene']}: ATE {row['clean_ATE']:.4f}->{row['attack_ATE']:.4f}; "
            f"AbsRel {row['clean_depth_absrel']:.4f}->{row['attack_depth_absrel']:.4f}; "
            f"Acc {row['clean_point_acc']:.4f}->{row['attack_point_acc']:.4f}; "
            f"EPE {row['clean_track_epe_mean']:.2f}->{row['attack_track_epe_mean']:.2f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output_root", required=True)
    common.add_argument("--scenes", nargs="+", choices=SCENES, required=True)

    p = sub.add_parser("infer", parents=[common])
    p.add_argument(
        "--pose_root", default="",
        help="Legacy pose-window root. Not needed when --frame_plan is supplied.")
    p.add_argument(
        "--frame_plan", default="",
        help="Frozen held-out window plan; enables one all-head forward per condition.")
    p.add_argument("--texture", required=True)
    p.add_argument("--protocol", default=str(REPO_ROOT / "data/tum_universal_all/protocol.json"))
    p.add_argument("--tum_root", default="/mnt/data/wangqq/recons_eval/data/tum_all_sync")
    p.add_argument("--gt_name", default="groundtruth.txt")
    p.add_argument("--quad_file", default=str(REPO_ROOT / "data/tum10_split/plane_quad_world.txt"))
    p.add_argument("--ckpt", default=str(REPO_ROOT / "checkpoints/VGGT-1B"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--visibility_depth_margin", type=float, default=0.08)
    p.add_argument("--query_rows", type=int, default=12)
    p.add_argument("--query_cols", type=int, default=16)
    p.add_argument("--query_margin", type=float, default=0.10)
    p.add_argument("--occl_tol", type=float, default=0.05)
    p.add_argument("--min_track_points", type=int, default=20)
    p.add_argument("--max_point_samples", type=int, default=200000)
    p.add_argument("--icp_threshold", type=float, default=0.10)
    p.add_argument("--max_windows", type=int, default=0)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("summarize", parents=[common])
    p.add_argument("--pose_csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "infer":
        infer(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
