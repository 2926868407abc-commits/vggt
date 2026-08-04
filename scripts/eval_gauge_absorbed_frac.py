"""Per-run gauge diagnostics to sit alongside the ATE/RPE evaluation.

Answers, for one attacked run: how much of what the attack did to the trajectory
is a global Sim(3) that the aligned ATE discards?

    gauge_absorbed_frac = 1 - RMS(deviation from clean after best-fit Sim(3))
                              / RMS(raw deviation from clean)

A value near 1 means the attack mostly moved / rescaled / rotated the whole
reconstruction, which `evo_utils.eval_metrics(align=True, correct_scale=True)`
removes before it measures anything.

This script does NOT modify or re-implement the evaluation. It imports
recons_eval's own functions through scripts/diag_gauge_invariance.py, and writes
only its own CSV. The ATE column here is the same estimator the production eval
uses (verified to reproduce it on 77/77 recorded runs).

Wired into run_geometry_aware_tum10.sh; run standalone as:

    /mnt/data/wangqq/conda_envs/recons_eval/bin/python3 \
        scripts/eval_gauge_absorbed_frac.py \
        --vggt_output_root outputs_attack_geometry_aware_tum10/<run> \
        --model_name vggt_<run>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_gauge_invariance import (  # noqa: E402
    ReconsEval,
    gt_traj_for,
    load_run,
    sim3_decomposition,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vggt_output_root", required=True,
                   help="attacked run directory (contains <scene>/vggt_outputs.npz)")
    p.add_argument("--model_name", required=True)
    p.add_argument("--recons_root", default="/mnt/data/wangqq/recons_eval")
    p.add_argument("--tum_root", default=None)
    p.add_argument("--clean_vggt_output_root",
                   default="/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/"
                           "tum10_clean_uniform_l3")
    p.add_argument("--scene_pattern", default="rgbd_dataset_freiburg3_*")
    p.add_argument("--gt_name", default="groundtruth_90.txt")
    p.add_argument("--out_csv", default=None,
                   help="default: <recons_root>/outputs/relpose-distance/"
                        "tum10-gauge-<model_name>.csv")
    p.add_argument("--work_dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    recons_root = Path(args.recons_root)
    tum_root = Path(args.tum_root) if args.tum_root else recons_root / "data/tum"
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else recons_root / "outputs/relpose-distance" / f"tum10-gauge-{args.model_name}.csv"
    )
    work_dir = Path(args.work_dir) if args.work_dir else out_csv.parent / "_gauge_work" / args.model_name
    work_dir.mkdir(parents=True, exist_ok=True)

    rec = ReconsEval(recons_root)
    umeyama, _, _ = rec.pointmap_fns()

    attacked_root = Path(args.vggt_output_root)
    clean_root = Path(args.clean_vggt_output_root)
    scene_dirs = sorted(p for p in attacked_root.glob(args.scene_pattern) if p.is_dir())
    if not scene_dirs:
        raise SystemExit(f"No scene outputs under {attacked_root}")

    rows: list[dict] = []
    for scene_dir in scene_dirs:
        scene = scene_dir.name
        run = load_run(attacked_root, scene)
        clean = load_run(clean_root, scene)
        if run["frame_indices"] != clean["frame_indices"]:
            raise SystemExit(f"{scene}: frame_indices differ between run and clean reference")

        gt_traj, gt_c2w = gt_traj_for(rec, tum_root, scene, run["frame_indices"],
                                      args.gt_name, work_dir)
        traj = rec.evo_utils.get_tum_poses(run["c2w"])
        rpe_trans, rpe_rot = rec.rpe(traj, gt_traj, True, True)

        # scale of the CLEAN prediction w.r.t. GT, so the run's own scale gauge is on record
        c_clean, _, _ = umeyama(clean["c2w"][:, :3, 3].T, gt_c2w[:, :3, 3].T)

        row = {
            "model": args.model_name,
            "dataset": "tum10",
            "seq": scene,
            "n_frames": len(run["frame_indices"]),
            "ATE_align_scale": rec.ate(traj, gt_traj, True, True),
            "ATE_align_noscale": rec.ate(traj, gt_traj, True, False),
            "ATE_noalign": rec.ate(traj, gt_traj, False, False),
            "RPE_trans": rpe_trans,
            "RPE_rot_deg": rpe_rot,
            "clean_umeyama_scale_pred_to_gt": float(c_clean),
        }
        row.update(sim3_decomposition(rec, run["c2w"], clean["c2w"]))
        rows.append(row)
        print(
            f"{args.model_name} {scene}: "
            f"ATE={row['ATE_align_scale']:.6f} "
            f"(no-scale {row['ATE_align_noscale']:.6f}, no-align {row['ATE_noalign']:.6f})  "
            f"gauge_absorbed={row['gauge_absorbed_frac']:.4f}  "
            f"sim3_scale_vs_clean={row['sim3_scale_vs_clean']:.4f}"
        )

    if len(rows) > 1:
        mean_row = {"model": args.model_name, "dataset": "tum10", "seq": "MEAN",
                    "n_frames": rows[0]["n_frames"]}
        for key in rows[0]:
            if key in mean_row:
                continue
            vals = [r[key] for r in rows if isinstance(r[key], float) and np.isfinite(r[key])]
            mean_row[key] = float(np.mean(vals)) if vals else float("nan")
        rows.append(mean_row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_csv, rows)
    print(f"Saved gauge diagnostics -> {out_csv}")


if __name__ == "__main__":
    main()
