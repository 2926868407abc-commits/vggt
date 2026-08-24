"""§15.2 -- the two comparisons, each isolating one variable.

  global vs piecewise target : l64_al_*   vs  p64_piece_*   (target shape changes)
  pose-only vs multi-head    : p64_piece_* vs  p64_joint_*  (heads change)

Chained on purpose: comparing the joint loss straight to the global-target baseline
would move both variables at once.

Adjacent comparisons are tested against the pooled s.d. of the two cells, the same
rule §15.1 used, so "different" means separable rather than merely unequal.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import ReconsEval, gt_traj_for, load_run  # noqa: E402

R = VG / "outputs_attack_geometry_aware_tum10"
RECONS = Path("/mnt/data/wangqq/recons_eval")
GAUGE = RECONS / "outputs/relpose-distance"
SCENE = "rgbd_dataset_freiburg3_sitting_halfsphere"
CEILING, CLEAN_ATE, CLEAN_ROT, RANDOM = 0.251932, 0.00816, 0.446, 0.910

ARMS = [("l64_al", "全局目标 · 仅 pose", "aligned_residual"),
        ("p64_piece", "分段目标 · 仅 pose", "piecewise_targeted"),
        ("p64_joint", "分段目标 · 多头", "joint_multihead")]

rec = ReconsEval(RECONS)
work = Path("/tmp/t152_work")
work.mkdir(parents=True, exist_ok=True)
clean = load_run(R / "tum10_clean_uniform_l3", SCENE)
gt_traj, _ = gt_traj_for(rec, RECONS / "data/tum", SCENE, clean["frame_indices"],
                         "groundtruth_90.txt", work)


def gauge_of(model):
    p = GAUGE / f"tum10-gauge-{model}.csv"
    if not p.exists():
        return None
    for row in csv.DictReader(p.open(encoding="utf-8")):
        if row.get("seq", "").startswith("rgbd"):
            return row
    return None


def measure(run):
    if not (R / run / SCENE / "attack_summary.json").exists():
        return None
    t = rec.evo_utils.get_tum_poses(np.asarray(load_run(R / run, SCENE)["c2w"],
                                               dtype=np.float64))
    _, rot = rec.rpe(t, gt_traj, True, True)
    ate = float(rec.ate(t, gt_traj, True, True))
    g = gauge_of(f"vggt_{run}")
    gab = float(g["gauge_absorbed_frac"]) if g and g.get("gauge_absorbed_frac") else np.nan
    return ate / CEILING, float(rot), gab


data = {}
print(f"halfsphere · 64x64 · 显示器放置 · 1000 步 · 3 seeds · 已标定")
print(f"干净 ATE {CLEAN_ATE/CEILING:.1%} of ceiling / RPE 旋转 {CLEAN_ROT:.2f}°"
      f"   随机基线 {RANDOM:.1%}\n")
print(f"{'配置':<22}{'ATE 占上限':>18}{'RPE 旋转 (°)':>20}{'gauge absorbed':>18}")
print("-" * 80)
for tag, label, _ in ARMS:
    vals = [v for v in (measure(f"{tag}_s{s}") for s in (0, 1, 2)) if v is not None]
    if len(vals) < 3:
        print(f"{label:<22}  只有 {len(vals)} 个种子")
        continue
    a = np.array([v[0] for v in vals])
    r = np.array([v[1] for v in vals])
    g = np.array([v[2] for v in vals])
    data[tag] = (a, r, g)
    print(f"{label:<22}{a.mean():>11.1%} ± {a.std(ddof=1):>4.1%}"
          f"{r.mean():>13.2f} ± {r.std(ddof=1):>4.2f}"
          f"{np.nanmean(g):>12.3f} ± {np.nanstd(g, ddof=1):>4.3f}")


def compare(a_tag, b_tag, title):
    if a_tag not in data or b_tag not in data:
        print(f"\n{title}: 数据不全")
        return
    print(f"\n=== {title} ===")
    for idx, (mname, fmt) in enumerate([("ATE 占上限", "{:.1%}"), ("RPE 旋转", "{:.2f}°")]):
        A, B = data[a_tag][idx], data[b_tag][idx]
        gap = B.mean() - A.mean()
        pooled = float(np.sqrt((A.var(ddof=1) + B.var(ddof=1)) / 2))
        verdict = "有差异" if abs(gap) > pooled else "**判不出差异**"
        print(f"    {mname:<10} {fmt.format(A.mean())} -> {fmt.format(B.mean())}   "
              f"差 {fmt.format(gap)}   合并标准差 {fmt.format(pooled)}   {verdict}")


compare("l64_al", "p64_piece", "全局目标 vs 分段目标（仅 pose）")
compare("p64_piece", "p64_joint", "仅 pose vs 多头（都用分段目标）")

print("\n注：差异判据与 §15.1 一致 —— 差距需大于两格的合并标准差才算可分。")
