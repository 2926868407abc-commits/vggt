"""Does the 64x64 loss ranking survive a change of sequence?

Prints the four losses on all three sequences side by side, on both metrics, and
states whether the orderings agree. Ceilings and random baselines are recomputed
per sequence rather than assumed, because they differ by 8x between static and
halfsphere and that is exactly what decides whether ATE means anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import ReconsEval, gt_traj_for, load_run  # noqa: E402

R = VG / "outputs_attack_geometry_aware_tum10"
RECONS = Path("/mnt/data/wangqq/recons_eval")
LOSSES = [("old", "gt_untargeted 旧"), ("si", "scale_invariant"),
          ("pw", "pairwise_relative"), ("al", "aligned_residual")]
# label, scene, run-name builder
SEQS = [
    ("halfsphere", "rgbd_dataset_freiburg3_sitting_halfsphere",
     lambda t: [f"l64_{t}_s{s}" for s in (0, 1, 2)]),
    ("xyz", "rgbd_dataset_freiburg3_sitting_xyz", lambda t: [f"g64_xyz_{t}"]),
    ("static", "rgbd_dataset_freiburg3_sitting_static", lambda t: [f"g64_st_{t}"]),
]

rec = ReconsEval(RECONS)
work = Path("/tmp/cross_work")
work.mkdir(parents=True, exist_ok=True)


def ceiling_of(gt_c2w):
    p = gt_c2w[:, :3, 3]
    return float(np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, axis=1))))


ctx = {}
for label, scene, _ in SEQS:
    clean = load_run(R / "tum10_clean_uniform_l3", scene)
    gt_traj, gt_c2w = gt_traj_for(rec, RECONS / "data/tum", scene,
                                  clean["frame_indices"], "groundtruth_90.txt", work)
    ct = rec.evo_utils.get_tum_poses(np.asarray(clean["c2w"], dtype=np.float64))
    _, crot = rec.rpe(ct, gt_traj, True, True)
    ctx[label] = {"gt": gt_traj, "ceil": ceiling_of(gt_c2w), "scene": scene,
                  "clean_ate": float(rec.ate(ct, gt_traj, True, True)),
                  "clean_rot": float(crot)}


def score(run, label):
    scene = ctx[label]["scene"]
    if not (R / run / scene / "attack_summary.json").exists():
        return None
    t = rec.evo_utils.get_tum_poses(np.asarray(load_run(R / run, scene)["c2w"],
                                               dtype=np.float64))
    _, rot = rec.rpe(t, ctx[label]["gt"], True, True)
    return float(rec.ate(t, ctx[label]["gt"], True, True)) / ctx[label]["ceil"], float(rot)


print("每序列的评测底数")
print(f"{'序列':<12}{'ATE 上限':>10}{'干净 ATE 占比':>14}{'干净 RPE 旋转':>14}")
print("-" * 52)
for label, _, _ in SEQS:
    c = ctx[label]
    print(f"{label:<12}{c['ceil']:>10.4f}{c['clean_ate']/c['ceil']:>13.1%}"
          f"{c['clean_rot']:>13.2f}°")

for metric, mi, unit in [("ATE 占上限", 0, "%"), ("RPE 旋转", 1, "°")]:
    print(f"\n\n=== {metric} ===")
    print(f"{'损失':<20}" + "".join(f"{l:>16}" for l, _, _ in SEQS))
    print("-" * 68)
    order = {}
    for tag, name in LOSSES:
        cells = []
        for label, _, mk in SEQS:
            vals = [v for v in (score(r, label) for r in mk(tag)) if v is not None]
            if not vals:
                cells.append(None)
                continue
            arr = np.array([v[mi] for v in vals])
            cells.append((arr.mean(), arr.std(ddof=1) if len(arr) > 1 else None, len(arr)))
        order[tag] = cells
        txt = ""
        for c in cells:
            if c is None:
                txt += f"{'—':>16}"
            elif unit == "%":
                txt += (f"{c[0]:>10.1%}" + (f"±{c[1]:.1%}" if c[1] else "     ")).rjust(16)
            else:
                txt += (f"{c[0]:>10.2f}" + (f"±{c[1]:.2f}" if c[1] else "     ")).rjust(16)
        print(f"{name:<20}{txt}")

    print("\n  各序列名次（强→弱）")
    rankings = []
    for j, (label, _, _) in enumerate(SEQS):
        avail = [(order[t][j][0], t) for t, _ in LOSSES if order[t][j] is not None]
        if len(avail) < len(LOSSES):
            print(f"    {label:<12} 数据不全，跳过")
            rankings.append(None)
            continue
        rank = [t for _, t in sorted(avail, reverse=True)]
        rankings.append(rank)
        print(f"    {label:<12} " + " > ".join(rank))
    done = [r for r in rankings if r]
    if len(done) > 1:
        if all(r == done[0] for r in done):
            print(f"    -> 三序列名次一致，该指标下排名可推广")
        else:
            print(f"    -> 名次不一致，**不能跨序列推广**；需按序列分别报告")
