"""Milestone M-A: one table of pose / depth / point-map / tracking degradation.

The runbook gates every multi-task claim on this. Until now pose had ground truth,
point map had a GT builder that nothing called routinely, depth had only AbsRel
buried in a diagnostic, and tracking was scored against VGGT's own clean prediction
-- which can say "it moved" but never "it is wrong".

Track ground truth now comes from build_tum_gt_point_track.py (reprojection GT).

Two things this checks rather than assumes:

  render fidelity  the attacked images are not saved, so they are re-rendered here.
                   If that render differed from the one the run actually used, every
                   number below would be measuring a different patch. So the
                   re-rendered poses are compared against the run's stored extrinsic
                   and the row is marked if they disagree.

  thin frames      GT visibility varies from 100% to 7% per frame. A frame with a
                   handful of visible points gives a meaningless EPE, so frames
                   below --min_track_points are dropped and the count is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG))
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import (  # noqa: E402
    ReconsEval, gt_traj_for, load_run, match_depth_paths,
    pointmap_metrics, preprocess_tum_depth_to_vggt_grid,
)

R = VG / "outputs_attack_geometry_aware_tum10"
RECONS = Path("/mnt/data/wangqq/recons_eval")
TUM = RECONS / "data/tum"
GT_DIR = VG / "outputs/tum_gt_point_track"
CLEAN = "tum10_clean_uniform_l3"


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Median-scale-aligned depth error. VGGT depth is only defined up to scale, so
    comparing raw values would report the gauge rather than the error."""
    valid = (gt > 1e-6) & (pred > 1e-6) & np.isfinite(gt) & np.isfinite(pred)
    if valid.sum() < 100:
        return {"depth_absrel": np.nan, "depth_rmse": np.nan, "depth_d1": np.nan}
    p, g = pred[valid], gt[valid]
    p = p * (np.median(g) / max(np.median(p), 1e-9))
    ratio = np.maximum(p / g, g / p)
    return {
        "depth_absrel": float(np.mean(np.abs(p - g) / g)),
        "depth_rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "depth_d1": float(np.mean(ratio < 1.25)),
    }


def track_metrics(pred_xy, pred_vis, gt, min_pts: int) -> dict[str, float]:
    tracks, visible, known = gt["tracks"], gt["track_visible"], gt["track_known"]
    sel = visible[1:] & known[1:]          # frame 0 is the query frame itself
    per_frame_n = sel.sum(axis=1)
    usable = per_frame_n >= min_pts
    if not usable.any():
        return {"track_epe": np.nan, "track_epe_mean": np.nan, "track_vis_f1": np.nan,
                "track_frames_used": 0, "track_points": 0}
    err = np.linalg.norm(pred_xy[1:] - tracks[1:], axis=-1)
    vals = err[usable][sel[usable]]
    # both statistics: on the warm-start patches the median moved 4.49 -> 4.92 px
    # while the mean moved 5.69 -> 19.31 px, so reporting either alone misleads
    epe = float(np.median(vals))
    epe_mean = float(vals.mean())

    # visibility F1 against the GT flag, on points whose GT status is known
    k = known[1:][usable]
    pv = (pred_vis[1:][usable] > 0.5) & k
    gv = sel[usable]
    tp = float((pv & gv).sum())
    fp = float((pv & ~gv & k).sum())
    fn = float((~pv & gv).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
    return {"track_epe": epe, "track_epe_mean": epe_mean, "track_vis_f1": float(f1),
            "track_frames_used": int(usable.sum()),
            "track_points": int(sel[usable].sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    ap.add_argument("--track_dir", default="/tmp/four_task_tracks",
                    help="dump_tracks.py 的输出目录")
    ap.add_argument("--min_track_points", type=int, default=20)
    ap.add_argument("--icp_threshold", type=float, default=0.1)
    ap.add_argument("--out_csv", default="/tmp/four_tasks.csv")
    ap.add_argument("--gt_dir", default=str(GT_DIR),
                    help="帧子集需指向该子集自己的 GT")
    ap.add_argument("--clean", default=CLEAN,
                    help="帧子集需指向该子集自己的 clean run 名")
    cli = ap.parse_args()

    scene = cli.scene
    gt = dict(np.load(Path(cli.gt_dir) / f"{scene}_gt.npz", allow_pickle=True))
    rec = ReconsEval(RECONS)
    work = Path("/tmp/four_task_work")
    work.mkdir(parents=True, exist_ok=True)
    clean_run = load_run(R / cli.clean, scene)
    gt_traj, _ = gt_traj_for(rec, TUM, scene, clean_run["frame_indices"],
                             "groundtruth_90.txt", work)

    rows = []

    for run in [r.strip() for r in cli.runs.split(",") if r.strip()]:
        summary_p = R / run / scene / "attack_summary.json"
        if not summary_p.exists():
            print(f"{run}: 缺 attack_summary")
            continue
        summary = json.loads(summary_p.read_text(encoding="utf-8"))
        saved = np.load(R / run / scene / "vggt_outputs.npz")
        image_paths = [str(p) for p in summary["image_paths"]]

        # --- pose / depth / point map come from the stored outputs
        traj = rec.evo_utils.get_tum_poses(np.asarray(load_run(R / run, scene)["c2w"],
                                                      dtype=np.float64))
        ate = float(rec.ate(traj, gt_traj, True, True))
        _, rpe_rot = rec.rpe(traj, gt_traj, True, True)

        tensor_hw = (saved["depth"].shape[1], saved["depth"].shape[2])
        dpaths = match_depth_paths(TUM / scene, image_paths)
        gt_depth = np.stack([preprocess_tum_depth_to_vggt_grid(p, Path(ip), tensor_hw)
                             for p, ip in zip(dpaths, image_paths)])
        dm = depth_metrics(saved["depth"][..., 0].astype(np.float64),
                           gt_depth.astype(np.float64))

        pm = pointmap_metrics(rec, saved["point_map"].astype(np.float64),
                              gt["point_map"].astype(np.float64),
                              gt["point_valid"], cli.icp_threshold)

        # --- tracking comes from dump_tracks.py, which owns the GPU half
        dump_p = Path(cli.track_dir) / f"{run}__{scene}.npz"
        if not dump_p.exists():
            print(f"{run}: 缺 track 导出（先跑 dump_tracks.py），跳过")
            continue
        dump = np.load(dump_p, allow_pickle=True)
        render_err = float(np.abs(dump["rerendered_extrinsic"] - saved["extrinsic"]).max())
        tm = track_metrics(dump["track"], dump["track_vis"], gt, cli.min_track_points)

        row = {"run": run, "scene": scene, "ATE": ate, "RPE_rot_deg": float(rpe_rot),
               **dm,
               "point_acc": pm["Acc_mean"], "point_comp": pm["Comp_mean"],
               "point_acc_med": pm["Acc_med"], "point_nc": pm["NC1"], **tm,
               "render_max_extrinsic_err": render_err,
               "render_ok": bool(render_err < 1e-3)}
        rows.append(row)
        flag = "" if row["render_ok"] else "  ⚠️ 重渲染与存档不符"
        print(f"{run:<20} ATE {ate:.4f}  RPE {rpe_rot:5.2f}°  "
              f"AbsRel {dm['depth_absrel']:.4f}  d1 {dm['depth_d1']:.3f}  "
              f"Acc {row['point_acc']:.4f}  "
              f"EPE中位 {tm['track_epe']:.2f} 均值 {tm['track_epe_mean']:.2f}px  "
              f"visF1 {tm['track_vis_f1']:.3f}{flag}")

    if rows:
        import csv
        out = Path(cli.out_csv)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {out}")
        bad = [r["run"] for r in rows if not r["render_ok"]]
        if bad:
            print(f"⚠️ 重渲染未复现存档位姿的 run: {bad}  —— 这些行的 track 指标不可信")
        used = {r["track_frames_used"] for r in rows}
        print(f"track 使用帧数（共 9 个非查询帧，阈值 {cli.min_track_points} 点）: {used}")


if __name__ == "__main__":
    main()
