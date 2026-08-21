"""Four losses at 64x64, both metrics, with the seed spread shown.

Two panels rather than one because the two metrics rank the losses differently:
pairwise does the most damage to trajectory position, scale-invariant does the most
to camera orientation. A single-panel version would support whichever conclusion
its author picked.

Every bar carries its value as text -- three of the four hues sit below 3:1 against
the light surface, so identity must not rest on colour alone.

Labels are English -- the server has no CJK font.
"""

from __future__ import annotations

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
CEILING, CLEAN_ATE, CLEAN_ROT, RANDOM_FRAC = 0.251932, 0.00816, 0.446, 0.905

LOSSES = [("old", "gt_untargeted\n(old)", "#2a78d6"),
          ("si", "scale_invariant", "#eb6834"),
          ("pw", "pairwise_relative", "#1baf7a"),
          ("al", "aligned_residual", "#eda100")]
SEEDS = (0, 1, 2)

rec = ReconsEval(RECONS)
work = Path("/tmp/loss64_work")
work.mkdir(parents=True, exist_ok=True)
clean = load_run(R / "tum10_clean_uniform_l3", SCENE)
gt_traj, _ = gt_traj_for(rec, RECONS / "data/tum", SCENE, clean["frame_indices"],
                         "groundtruth_90.txt", work)


def score(run: str):
    if not (R / run / SCENE / "attack_summary.json").exists():
        return None
    t = rec.evo_utils.get_tum_poses(np.asarray(load_run(R / run, SCENE)["c2w"],
                                               dtype=np.float64))
    _, rot = rec.rpe(t, gt_traj, True, True)
    return float(rec.ate(t, gt_traj, True, True)) / CEILING, float(rot)


data = {}
for tag, _, _ in LOSSES:
    vals = [v for v in (score(f"l64_{tag}_s{s}") for s in SEEDS) if v is not None]
    data[tag] = (np.array([v[0] for v in vals]), np.array([v[1] for v in vals]))

fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
x = np.arange(len(LOSSES))

# panel 0: ATE as a fraction of its ceiling
a = ax[0]
a.axhspan(RANDOM_FRAC, 1.02, color="#0b0b0b", alpha=0.05, zorder=0)
a.axhline(RANDOM_FRAC, color="#898781", ls="--", lw=1.4)
a.axhline(CLEAN_ATE / CEILING, color="#898781", ls="--", lw=1.4)
# reference labels go in the left margin: every x position carries a bar, and the
# tallest one reaches the random line
a.text(-0.94, RANDOM_FRAC, f"random {RANDOM_FRAC:.0%} — saturated above",
       ha="left", va="bottom", fontsize=9, color="#52514e")
a.text(-0.94, CLEAN_ATE / CEILING, f"clean {CLEAN_ATE/CEILING:.1%}",
       ha="left", va="bottom", fontsize=9, color="#52514e")
for i, (tag, label, col) in enumerate(LOSSES):
    fr = data[tag][0]
    sd = fr.std(ddof=1) if len(fr) > 1 else 0.0
    a.bar(i, fr.mean(), 0.62, color=col, zorder=3)
    a.errorbar(i, fr.mean(), yerr=sd, color="#0b0b0b", capsize=5, lw=1.4, zorder=4)
    a.scatter([i] * len(fr), fr, s=16, color="#0b0b0b", alpha=0.55, zorder=5)
    a.text(i, fr.mean() + sd + 0.04, f"{fr.mean():.1%}", ha="center",
           fontsize=10.5, fontweight="bold")
a.set_ylim(0, 1.06)
a.set_ylabel("ATE as a fraction of its ceiling", fontsize=10.5)
a.set_title("Trajectory position damage", fontsize=11.5)

# panel 1: RPE rotation, which a global Sim(3) does not cancel
b = ax[1]
b.axhline(CLEAN_ROT, color="#898781", ls="--", lw=1.4)
b.text(-0.94, CLEAN_ROT, f"clean {CLEAN_ROT:.2f}°", ha="left", va="bottom",
       fontsize=9, color="#52514e")
for i, (tag, label, col) in enumerate(LOSSES):
    ro = data[tag][1]
    sd = ro.std(ddof=1) if len(ro) > 1 else 0.0
    b.bar(i, ro.mean(), 0.62, color=col, zorder=3)
    b.errorbar(i, ro.mean(), yerr=sd, color="#0b0b0b", capsize=5, lw=1.4, zorder=4)
    b.scatter([i] * len(ro), ro, s=16, color="#0b0b0b", alpha=0.55, zorder=5)
    b.text(i, ro.mean() + sd + 0.35, f"{ro.mean():.2f}°", ha="center",
           fontsize=10.5, fontweight="bold")
b.set_ylabel("RPE rotation (degrees)", fontsize=10.5)
b.set_title("Camera orientation damage", fontsize=11.5)

for a_ in ax:
    a_.set_xlim(-1.02, len(LOSSES) - 0.35)
    a_.set_xticks(x)
    a_.set_xticklabels([l for _, l, _ in LOSSES], fontsize=9.5)
    a_.grid(axis="y", alpha=0.18, lw=0.8)
    a_.set_axisbelow(True)

fig.suptitle("The two metrics rank the four losses differently — report both\n"
             "64x64 · monitor placement · sitting_halfsphere · 1000 steps · EOT off · "
             "3 seeds each (dots), error bars ±1 s.d.", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "mon64_6_loss_comparison.png", dpi=170, facecolor="white")
print(f"wrote {OUT/'mon64_6_loss_comparison.png'}")
for tag, label, _ in LOSSES:
    fr, ro = data[tag]
    print(f"    {label.replace(chr(10),' '):<24} ATE {fr.mean():.1%} ± "
          f"{fr.std(ddof=1) if len(fr)>1 else 0:.1%}   "
          f"rot {ro.mean():.2f} ± {ro.std(ddof=1) if len(ro)>1 else 0:.2f}°  (n={len(fr)})")
