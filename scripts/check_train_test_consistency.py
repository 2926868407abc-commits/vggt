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
        if loss_name not in LOSS_FNS:
            print(f"{scene}: attack_loss '{loss_name}' has no pose loss to recompute, skipping")
            continue

        loss_args = argparse.Namespace(
            pose_rotation_weight=float(meta.get("pose_rotation_weight", 1.0)),
            pose_translation_weight=float(meta.get("pose_translation_weight", 1.0)),
            pose_scale_invariant_eps=float(meta.get("pose_scale_invariant_eps", 1e-6)),
        )
        pred = torch.from_numpy(load_pred_c2w(scene_dir / "vggt_outputs.npz")).unsqueeze(0)
        gt = torch.from_numpy(
            load_gt_c2w(tum_root, scene, [int(i) for i in summary["frame_indices"]], args.gt_name)
        ).unsqueeze(0)
        final_loss, terms = LOSS_FNS[loss_name](
            A.normalize_c2w_to_first(pred), A.normalize_c2w_to_first(gt), loss_args
        )

        train_vals = [r["pose_loss"] for r in tail if "pose_loss" in r]
        train_loss = float(np.mean(train_vals)) if train_vals else float("nan")
        gap = (train_loss - float(final_loss)) / train_loss if train_loss else float("nan")
        worst = max(worst, abs(gap)) if np.isfinite(gap) else worst

        row = {
            "run": run_root.name, "seq": scene, "attack_loss": loss_name,
            "train_loss_mean_last20": train_loss,
            "final_loss_on_saved_prediction": float(final_loss),
            "relative_gap": gap,
            "train_trans": float(np.mean([r["pose_trans_mse"] for r in tail
                                          if "pose_trans_mse" in r])),
            "final_trans": terms["pose_trans_mse"],
            "train_rot": float(np.mean([r["pose_rot_mse"] for r in tail if "pose_rot_mse" in r])),
            "final_rot": terms["pose_rot_mse"],
            "physical_eot": bool((meta.get("physical_eot") or {}).get("enabled", False))
            if isinstance(meta.get("physical_eot"), dict) else None,
        }
        rows.append(row)
        flag = ""
        if np.isfinite(gap) and abs(gap) > 0.20:
            flag = ("   <-- WARNING: the trained attack did not transfer to the evaluated "
                    "prediction; suspect EOT (try PHYSICAL_EOT=0)")
        print(f"{scene}: {loss_name}  train={train_loss:.4f}  final={float(final_loss):.4f}  "
              f"gap={gap*100:+.1f}%{flag}")
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
