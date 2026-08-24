"""Figure 13-1: what the piecewise target does at each resolution.

Left panel is the trajectories; right is the training curve that explains them. The
pair matters more than either alone -- the trajectory plot shows 64x64 sitting on
top of clean, and the loss curve shows why: the objective never descended.

All trajectories Sim(3)-aligned, as everywhere else in this document.
Labels are English -- the server has no CJK font.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import ReconsEval, gt_traj_for, load_run  # noqa: E402

R = VG / "outputs_attack_geometry_aware_tum10"
RECONS = Path("/mnt/data/wangqq/recons_eval")
OUT = VG / "outputs/figures/current"
OUT.mkdir(parents=True, exist_ok=True)
SCENE = "rgbd_dataset_freiburg3_sitting_halfsphere"
CEILING = 0.251932

rec = ReconsEval(RECONS)
umeyama, _, _ = rec.pointmap_fns()
work = Path("/tmp/f131_work")
work.mkdir(parents=True, exist_ok=True)
clean = load_run(R / "tum10_clean_uniform_l3", SCENE)
gt_traj, gt_c2w = gt_traj_for(rec, RECONS / "data/tum", SCENE,
                              clean["frame_indices"], "groundtruth_90.txt", work)
gt_xyz = gt_c2w[:, :3, 3]


def aligned(c2w):
    p = np.asarray(c2w)[:, :3, 3].T
    c, Rm, t = umeyama(p, gt_xyz.T)
    return (c * Rm @ p + t).T


def ate_frac(run):
    t = rec.evo_utils.get_tum_poses(np.asarray(load_run(R / run, SCENE)["c2w"],
                                               dtype=np.float64))
    return float(rec.ate(t, gt_traj, True, True)) / CEILING


RUNS = [("clean", "tum10_clean_uniform_l3", "#898781", "--"),
        ("global target, 64x64", "l64_al_s0", "#2a78d6", "-"),
        ("piecewise, 64x64", "p64_piece_s0", "#eb6834", "-"),
        ("piecewise, 128x128", "p128_piece_s0", "#1baf7a", "-")]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))

ax[0].plot(gt_xyz[:, 0], gt_xyz[:, 2], "-o", color="#0b0b0b", lw=2.6, ms=4.5,
           mew=0, label="GT")
for label, run, col, ls in RUNS:
    p = aligned(load_run(R / run, SCENE)["c2w"])
    frac = ate_frac(run)
    ax[0].plot(p[:, 0], p[:, 2], ls, color=col, lw=2.3, marker="o", ms=4.5, mew=0,
               label=f"{label}  ({frac:.0%})")
ax[0].set_title("trajectories, top view, Sim(3)-aligned", fontsize=11.5)
ax[0].set_xlabel("X (m)", fontsize=9.5)
ax[0].set_ylabel("Z (m)", fontsize=9.5)
ax[0].grid(alpha=0.18, lw=0.8)
ax[0].set_aspect("equal", adjustable="datalim")
ax[0].legend(fontsize=9, framealpha=0.93, title="(% of ATE ceiling)",
             title_fontsize=8.5)

CURVES = [("global target, 64x64 (maximised)", "l64_al_s0", "#2a78d6"),
          ("piecewise, 64x64 (minimised)", "p64_piece_s0", "#eb6834"),
          ("piecewise, 128x128 (minimised)", "p128_piece_s0", "#1baf7a")]
for label, run, col in CURVES:
    h = R / run / "geometry_patch/training_history.jsonl"
    v = np.array([json.loads(l)["attack_metric"]
                  for l in h.read_text(encoding="utf-8").splitlines() if l.strip()])
    ax[1].plot(v / abs(v[0]), color=col, lw=2.2, label=label)
ax[1].axhline(1.0, color="#898781", ls=":", lw=1.2)
ax[1].text(500, 1.03, "no progress", fontsize=9, color="#52514e", ha="center")
ax[1].set_yscale("log")
ax[1].set_title("training objective, normalised by its own start", fontsize=11.5)
ax[1].set_xlabel("iteration", fontsize=9.5)
ax[1].set_ylabel("objective / initial value", fontsize=9.5)
ax[1].grid(alpha=0.18, lw=0.8, which="both")
ax[1].legend(fontsize=9, framealpha=0.93)

fig.suptitle("The piecewise target needs 128x128: at 64x64 the objective never moves",
             fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "mon64_8_piecewise.png", dpi=170, facecolor="white")
print(f"wrote {OUT/'mon64_8_piecewise.png'}")
for label, run, _, _ in RUNS:
    print(f"    {label:<24} ATE {ate_frac(run):.1%} of ceiling")
