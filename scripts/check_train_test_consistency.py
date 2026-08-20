"""Did the attack that was trained survive into the prediction that gets evaluated?

Training renders the patch with `prepare_texture_for_render(..., training=True)`,
which applies the photometric EOT jitter; the saved `vggt_outputs.npz` is produced
with `training=False`, i.e. no jitter. If the attack learned to exploit the
injected noise rather than the geometry, the training loss keeps climbing while
the evaluated prediction barely moves, and nothing in the existing outputs makes
that visible.

Measured case: on sitting_halfsphere `pose_aligned_residual_mse` ended training at
translation term 0.7366 but scored 0.2671 when the same loss was recomputed on the
saved prediction -- a 63.7% gap. Rerunning with PHYSICAL_EOT=0 closed it to 0.0%
and the ATE went from 26.7% to 44.2% of ceiling. On sitting_xyz and sitting_static
the same loss transferred within 7% and 1%, so the failure is configuration
dependent and silent.

This recomputes the run's own attack loss on its saved prediction and reports the
gap. It reads which loss to use from attack_summary.json, so it needs no
configuration. Exits non-zero when the gap exceeds --max_gap, so it can gate a
pipeline if wanted (default: report only).

Usage:
    /mnt/data/wangqq/conda_envs/vggt/bin/python3 \
        scripts/check_train_test_consistency.py \
        --vggt_output_root outputs_attack_geometry_aware_tum10/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

VGGT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VGGT_ROOT))

import attack_vggt_geometry_tum10 as A  # noqa: E402

LOSS_FNS = {
    "pose_gt_untargeted": A.pose_relative_mse,
    "pose_clean_untargeted": A.pose_relative_mse,
    "pose_scale_invariant_mse": A.pose_scale_invariant_mse,
    "pose_pairwise_relative_mse": A.pose_pairwise_relative_mse,
    "pose_aligned_residual_mse": A.pose_aligned_residual_mse,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vggt_output_root", required=True)
    p.add_argument("--tum_root", default="/mnt/data/wangqq/recons_eval/data/tum")
    p.add_argument("--scene_pattern", default="rgbd_dataset_freiburg3_*")
    p.add_argument("--gt_name", default="groundtruth_90.txt")
    p.add_argument("--max_gap", type=float, default=None,
                   help="if set, exit 1 when the relative gap exceeds this (e.g. 0.20)")
    p.add_argument("--out_csv", default=None)
    return p.parse_args()


def load_pred_c2w(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as d:
        ext = d["extrinsic"].astype(np.float64)
    w2c = np.tile(np.eye(4), (ext.shape[0], 1, 1))
    w2c[:, :3, :4] = ext
    return np.linalg.inv(w2c)


def load_pred_heads(npz_path: Path) -> dict[str, torch.Tensor]:
    with np.load(npz_path) as d:
        return {key: torch.from_numpy(np.asarray(d[key], dtype=np.float32))
                for key in ("depth", "depth_conf", "point_map", "point_conf")}


def relative_gap(train: float, final: float) -> float:
    return ((train - final) / train
            if train and np.isfinite(train) and np.isfinite(final) else float("nan"))


def tail_stats(tail: list[dict], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in tail if key in row]
    return ((float(np.mean(values)), float(np.std(values))) if values
            else (float("nan"), float("nan")))


def recompute_joint_loss(scene: str, npz_path: Path, meta: dict,
                         pred_rel: torch.Tensor, tail: list[dict]) -> tuple[dict, dict]:
    """Recompute every saved four-head term (tracking is not in the official NPZ)."""
    cfg = meta.get("joint_gauge") or {}
    clean_root = (meta.get("plane") or {}).get("clean_vggt_output_root")
    if not cfg or not clean_root:
        raise RuntimeError("joint run metadata lacks joint_gauge/clean_vggt_output_root")
    target_args = argparse.Namespace(
        clean_vggt_output_root=clean_root,
        piecewise_gauge_family=cfg["family"],
        piecewise_gauge_magnitude=float(cfg["magnitude"]),
        orthogonal_mode_order=int(cfg["orthogonal_mode_order"]),
        orthogonal_mode_axis=int(cfg["orthogonal_mode_axis"]),
    )
    targets = A.build_joint_gauge_targets(scene, target_args, torch.device("cpu"))
    if targets is None:
        raise RuntimeError("could not rebuild joint targets from clean output")
    loss_args = argparse.Namespace(
        pose_rotation_weight=float(meta.get("pose_rotation_weight", 1.0)),
        pose_translation_weight=float(meta.get("pose_translation_weight", 1.0)),
        pose_scale_invariant_eps=float(meta.get("pose_scale_invariant_eps", 1e-6)),
    )
    pose, pose_parts = A.pose_aligned_residual_mse(pred_rel, targets["pose_rel"], loss_args)
    pred = load_pred_heads(npz_path)
    depth = A.scale_invariant_depth_loss(A.dense_scalar_head(pred["depth"]), targets["depth"])
    point = A.aligned_point_residual(pred["point_map"], targets["points"],
                                     int(cfg.get("point_stride", 4)))
    depth_conf = A.confidence_preservation_loss(
        A.dense_scalar_head(pred["depth_conf"]), A.dense_scalar_head(targets["depth_conf"]))
    point_conf = A.confidence_preservation_loss(
        A.dense_scalar_head(pred["point_conf"]), A.dense_scalar_head(targets["point_conf"]))

    final_terms = {
        "pose": float(pose), "depth": float(depth), "point": float(point),
        "depth_conf": float(depth_conf), "point_conf": float(point_conf),
    }
    hist_keys = {
        "pose": "joint_pose_term", "depth": "joint_depth_term",
        "point": "joint_point_term", "depth_conf": "joint_depth_conf_term",
        "point_conf": "joint_point_conf_term", "track": "joint_track_term",
    }
    train_terms = {name: tail_stats(tail, key) for name, key in hist_keys.items()}
    weights = {
        "pose": float(cfg.get("pose_weight", 1.0)),
        "depth": float(cfg.get("depth_weight", 1.0)),
        "point": float(cfg.get("point_weight", 1.0)),
        "depth_conf": float(cfg.get("depth_conf_weight", 0.0)),
        "point_conf": float(cfg.get("point_conf_weight", 0.0)),
    }
    available_history = []
    for row in tail:
        if all(hist_keys[k] in row for k in weights):
            available_history.append(sum(weights[k] * float(row[hist_keys[k]]) for k in weights))
    train_available = (float(np.mean(available_history)) if available_history
                       else float("nan"))
    train_available_std = (float(np.std(available_history)) if available_history
                           else float("nan"))
    final_available = sum(weights[k] * final_terms[k] for k in weights)
    details = {
        "pose_parts": pose_parts, "train_terms": train_terms,
        "final_terms": final_terms, "weights": weights,
        "track_saved": False,
    }
    return {
        "train": train_available, "train_std": train_available_std,
        "final": final_available,
        "gap": relative_gap(train_available, final_available),
    }, details


def load_gt_c2w(tum_root: Path, scene: str, frame_indices: list[int], gt_name: str) -> np.ndarray:
    rows = [l.split() for l in (tum_root / scene / gt_name).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    out = []
    for i in frame_indices:
        r = rows[int(i)]
        m = np.eye(4)
        m[:3, :3] = A.quat_xyzw_to_rot(*map(float, r[4:8]))
        m[:3, 3] = list(map(float, r[1:4]))
        out.append(m)
    return np.stack(out)


def main() -> None:
    args = parse_args()
    run_root = Path(args.vggt_output_root)
    tum_root = Path(args.tum_root)

    history = run_root / "geometry_patch" / "training_history.jsonl"
    if not history.exists():
        raise SystemExit(f"no training history at {history}")
    hist = [json.loads(l) for l in history.read_text(encoding="utf-8").splitlines() if l.strip()]
    tail = hist[-min(20, len(hist)):]

    scene_dirs = sorted(p for p in run_root.glob(args.scene_pattern) if p.is_dir())
    if not scene_dirs:
        raise SystemExit(f"no scene outputs under {run_root}")

    rows, worst = [], 0.0
    for scene_dir in scene_dirs:
        scene = scene_dir.name
        summary = json.loads((scene_dir / "attack_summary.json").read_text(encoding="utf-8"))
        meta = summary.get("geometry_patch_metadata", {})
        loss_name = meta.get("attack_target", "")
        pred = torch.from_numpy(load_pred_c2w(scene_dir / "vggt_outputs.npz")).unsqueeze(0)

        detail_json = ""
        track_recomputed = True
        if loss_name == "geometry_joint_gauge_budgeted":
            loss_args = argparse.Namespace(
                pose_rotation_weight=float(meta.get("pose_rotation_weight", 1.0)),
                pose_translation_weight=float(meta.get("pose_translation_weight", 1.0)),
                pose_scale_invariant_eps=float(meta.get("pose_scale_invariant_eps", 1e-6)),
            )
            budget_cfg = meta.get("filter_budget") or {}
            objective_name = budget_cfg.get("pose_objective", "untargeted_gt")
            if objective_name == "untargeted_gt":
                target = torch.from_numpy(load_gt_c2w(
                    tum_root, scene, [int(i) for i in summary["frame_indices"]], args.gt_name
                )).unsqueeze(0)
                target = A.normalize_c2w_to_first(target)
            else:
                cfg = meta.get("joint_gauge") or {}
                target_args = argparse.Namespace(
                    clean_vggt_output_root=(meta.get("plane") or {}).get(
                        "clean_vggt_output_root"),
                    piecewise_gauge_family=cfg.get("family"),
                    piecewise_gauge_magnitude=float(cfg.get("magnitude", 1.0)),
                    orthogonal_mode_order=int(cfg.get("orthogonal_mode_order", 2)),
                    orthogonal_mode_axis=int(cfg.get("orthogonal_mode_axis", 0)),
                )
                rebuilt = A.build_joint_gauge_targets(
                    scene, target_args, torch.device("cpu"))
                if rebuilt is None:
                    print(f"{scene}: cannot rebuild budget pose target; skipping")
                    continue
                target = rebuilt["pose_rel"]
            final_loss_tensor, terms = A.pose_aligned_residual_mse(
                A.normalize_c2w_to_first(pred), target, loss_args)
            final_loss = float(final_loss_tensor)
            train_loss, train_std = tail_stats(tail, "budget_primary_pose")
            gap = relative_gap(train_loss, final_loss)
            track_recomputed = False
            detail_json = json.dumps({
                "scope": "budget_primary_pose_only",
                "pose_objective": objective_name,
                "constraints_checked_separately_by": "eval_output_filters.py",
            }, sort_keys=True)
            print("    note: budgeted consistency compares the primary pose score; "
                  "deploy-time constraints are checked by eval_output_filters.py")
        elif loss_name == "geometry_joint_gauge_targeted":
            try:
                joint, details = recompute_joint_loss(
                    scene, scene_dir / "vggt_outputs.npz", meta,
                    A.normalize_c2w_to_first(pred), tail)
            except RuntimeError as exc:
                print(f"{scene}: cannot recompute joint loss: {exc}; skipping")
                continue
            final_loss = joint["final"]
            train_loss = joint["train"]
            train_std = joint["train_std"]
            gap = joint["gap"]
            terms = details["pose_parts"]
            track_recomputed = False
            detail_json = json.dumps({
                key: {
                    "train_mean": details["train_terms"][key][0],
                    "train_std": details["train_terms"][key][1],
                    "final": details["final_terms"].get(key),
                    "weight": details["weights"].get(key),
                }
                for key in details["train_terms"]
            }, sort_keys=True)
            print("    note: official vggt_outputs.npz has no tracks; consistency total "
                  "covers pose/depth/point and both confidence maps, while the tracking "
                  "term is reported from training only")
        elif loss_name not in LOSS_FNS:
            print(f"{scene}: attack_loss '{loss_name}' has no pose loss to recompute, skipping")
            continue
        else:
            loss_args = argparse.Namespace(
                pose_rotation_weight=float(meta.get("pose_rotation_weight", 1.0)),
                pose_translation_weight=float(meta.get("pose_translation_weight", 1.0)),
                pose_scale_invariant_eps=float(meta.get("pose_scale_invariant_eps", 1e-6)),
            )
            gt = torch.from_numpy(
                load_gt_c2w(tum_root, scene, [int(i) for i in summary["frame_indices"]], args.gt_name)
            ).unsqueeze(0)
            final_loss_tensor, terms = LOSS_FNS[loss_name](
                A.normalize_c2w_to_first(pred), A.normalize_c2w_to_first(gt), loss_args
            )
            final_loss = float(final_loss_tensor)
            train_loss, train_std = tail_stats(tail, "pose_loss")
            gap = relative_gap(train_loss, final_loss)
        # A percentage gap on its own is not evidence: an unconverged run oscillates,
        # and then any single comparison point differs from the tail mean by chance.
        # Measured case: sitting_static + aligned_residual showed an 18.2% "gap" whose
        # tail std was 1.254 on a mean of 3.366, i.e. the gap was 0.5 sigma -- pure
        # noise -- while the eleven other runs had std <= 0.03. Only flag when the gap
        # is both large in relative terms and large against the run's own spread.
        gap_sigma = ((train_loss - float(final_loss)) / train_std
                     if train_std and np.isfinite(train_std) and train_std > 0
                     else float("nan"))
        if np.isfinite(gap) and np.isfinite(gap_sigma) and abs(gap_sigma) > 2.0:
            worst = max(worst, abs(gap))

        row = {
            "run": run_root.name, "seq": scene, "attack_loss": loss_name,
            "train_loss_mean_last20": train_loss,
            "train_loss_std_last20": train_std,
            "final_loss_on_saved_prediction": float(final_loss),
            "relative_gap": gap,
            "gap_sigma": gap_sigma,
            "train_trans": float(np.mean([r["pose_trans_mse"] for r in tail
                                          if "pose_trans_mse" in r])),
            "final_trans": terms["pose_trans_mse"],
            "train_rot": float(np.mean([r["pose_rot_mse"] for r in tail if "pose_rot_mse" in r])),
            "final_rot": terms["pose_rot_mse"],
            "physical_eot": bool((meta.get("physical_eot") or {}).get("enabled", False))
            if isinstance(meta.get("physical_eot"), dict) else None,
            "track_recomputed": track_recomputed,
            "term_details_json": detail_json,
        }
        rows.append(row)
        flag = ""
        if np.isfinite(gap) and abs(gap) > 0.20 and np.isfinite(gap_sigma) and abs(gap_sigma) > 2.0:
            flag = ("   <-- WARNING: the trained attack did not transfer to the evaluated "
                    "prediction; suspect EOT (try PHYSICAL_EOT=0)")
        elif np.isfinite(gap) and abs(gap) > 0.20:
            flag = (f"   (gap is only {abs(gap_sigma):.1f} sigma of this run's own tail "
                    f"spread -- the optimisation has not converged, not a transfer failure)")
        print(f"{scene}: {loss_name}  train={train_loss:.4f}+-{train_std:.4f}  "
              f"final={float(final_loss):.4f}  gap={gap*100:+.1f}% ({gap_sigma:+.1f} sigma)"
              f"{flag}")
        print(f"    trans {row['train_trans']:.4f} -> {row['final_trans']:.4f}   "
              f"rot {row['train_rot']:.4f} -> {row['final_rot']:.4f}   "
              f"physical_eot={row['physical_eot']}")

    if args.out_csv and rows:
        import csv

        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved -> {out}")

    if args.max_gap is not None and worst > args.max_gap:
        print(f"FAIL: worst gap {worst*100:.1f}% exceeds --max_gap {args.max_gap*100:.1f}%")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
