"""Four losses x three seeds at 64x64, with the spread across seeds.

Reported on the metrics the monitor-placement work established: ATE as a fraction
of its ceiling (raw ATE is not comparable across sequences and saturates), and RPE
rotation, which a global Sim(3) does not cancel and which therefore still separates
runs once ATE is pinned near the ceiling.

The seed spread is the point. A same-config repeat was measured at ~7%, so any gap
between losses smaller than that is not a ranking -- it is noise, and this prints
the numbers needed to say so rather than inviting a reading off the means.
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
SCENE = "rgbd_dataset_freiburg3_sitting_halfsphere"
CEILING, CLEAN_ATE, CLEAN_ROT, RANDOM_FRAC = 0.251932, 0.00816, 0.446, 0.905
LOSSES = [("old", "pose_gt_untargeted 旧损失"),
          ("si", "pose_scale_invariant_mse"),
          ("pw", "pose_pairwise_relative_mse"),
          ("al", "pose_aligned_residual_mse")]
SEEDS = (0, 1, 2)

rec = ReconsEval(RECONS)
work = Path("/tmp/loss64_work")
work.mkdir(parents=True, exist_ok=True)
clean = load_run(R / "tum10_clean_uniform_l3", SCENE)
gt_traj, _ = gt_traj_for(rec, RECONS / "data/tum", SCENE, clean["frame_indices"],
                         "groundtruth_90.txt", work)


def score(run: str):
    p = R / run / SCENE / "attack_summary.json"
    if not p.exists():
        return None
    t = rec.evo_utils.get_tum_poses(np.asarray(load_run(R / run, SCENE)["c2w"],
                                               dtype=np.float64))
    _, rot = rec.rpe(t, gt_traj, True, True)
    return float(rec.ate(t, gt_traj, True, True)), float(rot)


print(f"64x64 · 显示器放置 · {SCENE.replace('rgbd_dataset_freiburg3_','')} · 1000 步 · EOT 关")
print(f"干净基线 ATE {CLEAN_ATE:.4f} m / RPE旋转 {CLEAN_ROT:.2f}°   "
      f"ATE 上限 {CEILING:.4f} m（随机 {RANDOM_FRAC:.1%}）\n")
hdr = (f"{'损失':<30}{'ATE 占上限':>22}{'RPE 旋转 (°)':>22}{'种子数':>7}")
print(hdr)
print("-" * 82)
table = {}
for tag, name in LOSSES:
    vals = [score(f"l64_{tag}_s{s}") for s in SEEDS]
    ok = [v for v in vals if v is not None]
    if not ok:
        print(f"{name:<30}{'全部缺失':>22}")
        continue
    fr = np.array([v[0] / CEILING for v in ok])
    ro = np.array([v[1] for v in ok])
    table[tag] = (fr, ro)
    sd_f = fr.std(ddof=1) if len(fr) > 1 else 0.0
    sd_r = ro.std(ddof=1) if len(ro) > 1 else 0.0
    print(f"{name:<30}{fr.mean():>13.1%} ± {sd_f:>5.1%}"
          f"{ro.mean():>15.2f} ± {sd_r:>4.2f}{len(ok):>7}")

print(f"\n逐种子明细")
for tag, name in LOSSES:
    row = []
    for s in SEEDS:
        v = score(f"l64_{tag}_s{s}")
        row.append("缺失" if v is None else f"{v[0]/CEILING:.1%}/{v[1]:.1f}°")
    print(f"    {name:<30}" + "   ".join(f"s{s}={r}" for s, r in zip(SEEDS, row)))

if len(table) > 1:
    print(f"\n判定")
    fr_means = {t: v[0].mean() for t, v in table.items()}
    fr_sds = [v[0].std(ddof=1) for v in table.values() if len(v[0]) > 1]
    spread = max(fr_means.values()) - min(fr_means.values())
    typ_sd = float(np.mean(fr_sds)) if fr_sds else float("nan")
    best = max(fr_means, key=fr_means.get)
    print(f"    四个损失均值跨度 {spread:.1%}，单元格内种子标准差平均 {typ_sd:.1%}")
    if spread < 2 * typ_sd:
        print(f"    -> 跨度小于两倍种子噪声，**不能据此排名**。"
              f"结论是：在物理可实现的 64x64 上，损失选择不重要。")
    else:
        print(f"    -> 跨度超过两倍种子噪声，排名可信；最强为 {best}")
    ro_means = {t: v[1].mean() for t, v in table.items()}
    print(f"    RPE 旋转最高 {max(ro_means, key=ro_means.get)} "
          f"({max(ro_means.values()):.2f}°，干净 {CLEAN_ROT:.2f}°)")
