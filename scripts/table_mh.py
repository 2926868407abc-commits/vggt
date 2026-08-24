"""Does adding output-consistency constraints cost attack strength?

Both arms maximise the same objective -- pose_aligned_residual_mse against GT --
at 64x64 on the monitor quad. The only difference is that one carries five budget
constraints on the auxiliary heads. §13.2 could not ask this because its shared
piecewise base produced nothing; this base reaches 83.4% of ceiling.

The verdict rule adds the precondition §13.2 was missing: a difference between two
arms only means something if at least one arm is meaningfully above clean. Two
cells that both sit on the clean baseline can be statistically separable and still
carry no information about the thing being compared.
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
CLEAN_FRAC = CLEAN_ATE / CEILING

ARMS = [("l64_al", "仅 pose（无约束）"), ("mh64", "pose + 五项一致性约束")]

rec = ReconsEval(RECONS)
work = Path("/tmp/mh_table_work")
work.mkdir(parents=True, exist_ok=True)
clean = load_run(R / "tum10_clean_uniform_l3", SCENE)
gt_traj, _ = gt_traj_for(rec, RECONS / "data/tum", SCENE, clean["frame_indices"],
                         "groundtruth_90.txt", work)


def gauge_row(model):
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
    g = gauge_row(f"vggt_{run}")
    gab = float(g["gauge_absorbed_frac"]) if g and g.get("gauge_absorbed_frac") else np.nan
    return float(rec.ate(t, gt_traj, True, True)) / CEILING, float(rot), gab


data = {}
print("halfsphere · 64x64 · 显示器放置 · 1000 步 · 关 EOT · 3 seeds")
print(f"干净基线 ATE {CLEAN_FRAC:.1%} of ceiling / RPE 旋转 {CLEAN_ROT:.2f}°"
      f"   随机基线 {RANDOM:.1%}\n")
print(f"{'配置':<26}{'ATE 占上限':>18}{'RPE 旋转 (°)':>20}{'gauge absorbed':>17}")
print("-" * 82)
for tag, label in ARMS:
    vals = [v for v in (measure(f"{tag}_s{s}") for s in (0, 1, 2)) if v is not None]
    if len(vals) < 3:
        print(f"{label:<26}  只有 {len(vals)} 个种子")
        continue
    a = np.array([v[0] for v in vals])
    r = np.array([v[1] for v in vals])
    g = np.array([v[2] for v in vals])
    data[tag] = (a, r)
    print(f"{label:<26}{a.mean():>11.1%} ± {a.std(ddof=1):>4.1%}"
          f"{r.mean():>13.2f} ± {r.std(ddof=1):>4.2f}"
          f"{np.nanmean(g):>11.3f} ± {np.nanstd(g, ddof=1):>4.3f}")

if len(data) == 2:
    A, B = data["l64_al"], data["mh64"]
    print("\n=== 判定 ===")
    for idx, (mname, fmt, clean_v) in enumerate(
            [("ATE 占上限", "{:.1%}", CLEAN_FRAC), ("RPE 旋转", "{:.2f}°", CLEAN_ROT)]):
        a, b = A[idx], B[idx]
        pooled = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
        # precondition: is either arm actually attacking?
        live = [n for n, v in (("仅 pose", a), ("多头", b))
                if (v.mean() - clean_v) > 3 * max(v.std(ddof=1), 1e-9)]
        gap = b.mean() - a.mean()
        print(f"\n  {mname}")
        print(f"    仅 pose {fmt.format(a.mean())}   多头 {fmt.format(b.mean())}   "
              f"差 {fmt.format(gap)}   合并标准差 {fmt.format(pooled)}")
        if not live:
            print(f"    -> 两臂都未显著高于干净基线，**此对照无意义**")
            continue
        print(f"    -> 高于干净基线的臂: {', '.join(live)}")
        if abs(gap) > pooled:
            direction = "约束降低了攻击强度" if gap < 0 else "约束反而提高了攻击强度"
            cost = abs(gap) / max(a.mean() - clean_v, 1e-12)
            print(f"    -> 可分：{direction}，代价为基线超出量的 {cost:.1%}")
        else:
            print(f"    -> 差距小于噪声：**约束没有可测量的代价**")
