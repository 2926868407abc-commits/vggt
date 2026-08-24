"""The two figures the doc names but nobody made.

fig 7-2  distribution of gauge_absorbed over every recorded run. The doc quotes a
         median of 90.4% from 76 patches; there are 127 now, and the median has
         moved, because the gauge-aware losses added since then waste less.

fig 8-2  trajectories before and after the evaluator's Sim(3), side by side. The
         doc asks for both panels specifically -- the aligned-only version I made
         earlier cannot show how much the metric removes.

Labels are English -- the server has no CJK font.
"""

from __future__ import annotations

import csv
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
GAUGE = RECONS / "outputs/relpose-distance"
CUR = VG / "outputs/figures/current"
METH = VG / "outputs/figures/method"
for d in (CUR, METH):
    d.mkdir(parents=True, exist_ok=True)
SCENE = "rgbd_dataset_freiburg3_sitting_halfsphere"

# ---------------------------------------------------------------- fig 7-2
vals, by_family = [], {"旧/其他": [], "gauge-aware 64x64": []}
for p in GAUGE.glob("tum10-gauge-*.csv"):
    for row in csv.DictReader(p.open(encoding="utf-8")):
        v = row.get("gauge_absorbed_frac")
        if not row.get("seq", "").startswith("rgbd") or v in (None, "", "nan"):
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        vals.append(f)
        key = ("gauge-aware 64x64" if ("l64_" in p.name or "g64_" in p.name)
               else "旧/其他")
        by_family[key].append(f)

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 1, 26)
ax.hist([by_family["旧/其他"], by_family["gauge-aware 64x64"]], bins=bins,
        stacked=True, color=["#898781", "#2a78d6"], edgecolor="white", linewidth=0.6,
        label=[f"earlier runs (n={len(by_family['旧/其他'])})",
               f"gauge-aware 64x64 (n={len(by_family['gauge-aware 64x64'])})"])
med = float(np.median(vals))
ax.axvline(med, color="#e34948", lw=2)
ax.text(med, ax.get_ylim()[1] * 0.95, f"  median {med:.1%}", color="#e34948",
        fontsize=10.5, fontweight="bold", va="top")
ax.set_xlabel("fraction of the attack's trajectory damage absorbed by one global Sim(3)",
              fontsize=10)
ax.set_ylabel("runs", fontsize=10)
ax.set_title(f"Effort the evaluator's alignment throws away  (n={len(vals)} runs)\n"
             "higher = more of the optimisation went into a transform the metric cancels",
             fontsize=11.5)
ax.legend(fontsize=9.5)
ax.grid(axis="y", alpha=0.18, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(METH / "method_gauge_absorbed_hist.png", dpi=170, facecolor="white")
print(f"wrote {METH/'method_gauge_absorbed_hist.png'}   n={len(vals)} median={med:.3f}")

# ---------------------------------------------------------------- fig 8-2
rec = ReconsEval(RECONS)
umeyama, _, _ = rec.pointmap_fns()
work = Path("/tmp/fig_work")
work.mkdir(parents=True, exist_ok=True)
clean = load_run(R / "tum10_clean_uniform_l3", SCENE)
gt_traj, gt_c2w = gt_traj_for(rec, RECONS / "data/tum", SCENE,
                              clean["frame_indices"], "groundtruth_90.txt", work)
gt_xyz = gt_c2w[:, :3, 3]


def raw_xyz(c2w):
    return np.asarray(c2w)[:, :3, 3]


def aligned_xyz(c2w):
    p = raw_xyz(c2w).T
    c, Rm, t = umeyama(p, gt_xyz.T)
    return (c * Rm @ p + t).T


SERIES = [("GT", gt_c2w, "#0b0b0b", "-"),
          ("clean", load_run(R / "tum10_clean_uniform_l3", SCENE)["c2w"], "#898781", "--"),
          ("old absolute-pose", load_run(R / "l64_old_s0", SCENE)["c2w"], "#2a78d6", "-"),
          ("Sim(3)-aligned loss", load_run(R / "l64_al_s0", SCENE)["c2w"], "#eb6834", "-")]

fig2, ax2 = plt.subplots(1, 2, figsize=(12.5, 5.4))
for j, (fn, title) in enumerate([(raw_xyz, "raw prediction (no alignment)"),
                                 (aligned_xyz, "after the evaluator's Sim(3)")]):
    for name, c2w, col, ls in SERIES:
        p = gt_xyz if name == "GT" else fn(c2w)
        ax2[j].plot(p[:, 0], p[:, 2], ls, color=col, lw=2.3, marker="o", ms=4.5,
                    mew=0, label=name)
    ax2[j].set_title(title, fontsize=11.5)
    ax2[j].set_xlabel("X (m)", fontsize=9.5)
    ax2[j].grid(alpha=0.18, lw=0.8)
    ax2[j].set_aspect("equal", adjustable="datalim")
ax2[0].set_ylabel("Z (m)", fontsize=9.5)
ax2[0].legend(fontsize=9.5, framealpha=0.93)
fig2.suptitle("Most of what the attack does to the trajectory is a global similarity "
              "the metric removes\n"
              "64x64 · monitor placement · sitting_halfsphere · seed 0", fontsize=12)
fig2.tight_layout(rect=(0, 0, 1, 0.92))
fig2.savefig(CUR / "mon64_7_alignment_effect.png", dpi=170, facecolor="white")
print(f"wrote {CUR/'mon64_7_alignment_effect.png'}")
for name, c2w, _, _ in SERIES[1:]:
    r = np.linalg.norm(raw_xyz(c2w) - gt_xyz, axis=1).mean()
    a = np.linalg.norm(aligned_xyz(c2w) - gt_xyz, axis=1).mean()
    print(f"    {name:<22} 未对齐 {r:.3f} m -> 对齐后 {a:.4f} m  "
          f"(消掉 {1 - a/max(r,1e-9):.1%})")
