"""Apply §15.1's pre-registered criterion to the three xyz seeds.

The criterion, written before the runs: if xyz still ranks si > pw > al > old on RPE
rotation AND every adjacent gap exceeds the seed s.d., §12's cross-sequence claim
stands; otherwise it is rewritten as "only halfsphere has a reliable ranking".

Adjacent gaps are tested against the pooled s.d. of the two cells being compared,
which is the relevant scale for "can these two be ordered", not the grand mean s.d.
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
SCENE = "rgbd_dataset_freiburg3_sitting_xyz"
CEILING, CLEAN_ROT = 0.200452, 0.383853
LOSSES = [("si", "scale_invariant"), ("pw", "pairwise_relative"),
          ("al", "aligned_residual"), ("old", "gt_untargeted 旧")]

rec = ReconsEval(RECONS)
work = Path("/tmp/xyz_verdict_work")
work.mkdir(parents=True, exist_ok=True)
clean = load_run(R / "tum10_clean_uniform_l3", SCENE)
gt_traj, _ = gt_traj_for(rec, RECONS / "data/tum", SCENE, clean["frame_indices"],
                         "groundtruth_90.txt", work)


def score(run):
    if not (R / run / SCENE / "attack_summary.json").exists():
        return None
    t = rec.evo_utils.get_tum_poses(np.asarray(load_run(R / run, SCENE)["c2w"],
                                               dtype=np.float64))
    _, rot = rec.rpe(t, gt_traj, True, True)
    return float(rec.ate(t, gt_traj, True, True)) / CEILING, float(rot)


# seed 0 has no suffix (it came from the gen64 batch); 1 and 2 do
def runs_for(tag):
    return [f"g64_xyz_{tag}"] + [f"g64_xyz_{tag}_s{s}" for s in (1, 2)]


data = {}
print(f"sitting_xyz · 64x64 · 3 seeds · 干净 RPE 旋转 {CLEAN_ROT:.2f}°  "
      f"ATE 上限 {CEILING:.4f}\n")
print(f"{'损失':<22}{'RPE 旋转 (°)':>20}{'ATE 占上限':>18}{'逐种子旋转':>26}")
print("-" * 86)
for tag, name in LOSSES:
    vals = [v for v in (score(r) for r in runs_for(tag)) if v is not None]
    if len(vals) < 3:
        print(f"{name:<22}  只有 {len(vals)} 个种子，跳过")
        continue
    ate = np.array([v[0] for v in vals])
    rot = np.array([v[1] for v in vals])
    data[tag] = rot
    print(f"{name:<22}{rot.mean():>13.2f} ± {rot.std(ddof=1):>4.2f}"
          f"{ate.mean():>12.1%} ± {ate.std(ddof=1):>4.1%}"
          f"{'  '.join(f'{v:.2f}' for v in rot):>26}")

if len(data) == len(LOSSES):
    order = sorted(data, key=lambda t: data[t].mean(), reverse=True)
    expected = ["si", "pw", "al", "old"]
    print(f"\n实测名次: {' > '.join(order)}")
    print(f"预期名次: {' > '.join(expected)}")
    same = order == expected
    print(f"名次{'一致' if same else '不一致'}")

    print("\n相邻两档能否分开（差距 vs 两者合并标准差）")
    separable = True
    for a, b in zip(order, order[1:]):
        gap = data[a].mean() - data[b].mean()
        pooled = float(np.sqrt((data[a].var(ddof=1) + data[b].var(ddof=1)) / 2))
        ok = gap > pooled
        separable &= ok
        print(f"    {a:>4} vs {b:<4} 差距 {gap:5.2f}°   合并标准差 {pooled:5.2f}°   "
              f"{'可分' if ok else '不可分'}")

    print("\n=== 判据结论 ===")
    if same and separable:
        print("    §12 的跨序列结论成立：halfsphere 与 xyz 在 RPE 旋转上名次一致，")
        print("    且每一对相邻档位都能分开。")
    elif same:
        print("    名次与 halfsphere 一致，但**并非每一对都能分开**。")
        print("    按预注册判据，§12 需改写：只能说部分档位可排序，不能宣称完整名次。")
        merged, i = [], 0
        while i < len(order):
            grp = [order[i]]
            while i + 1 < len(order):
                gap = data[grp[-1]].mean() - data[order[i + 1]].mean()
                pooled = float(np.sqrt((data[grp[-1]].var(ddof=1)
                                        + data[order[i + 1]].var(ddof=1)) / 2))
                if gap > pooled:
                    break
                i += 1
                grp.append(order[i])
            merged.append(grp)
            i += 1
        print("    可辩护的分组（组内不可分）: "
              + " > ".join("{" + ",".join(g) + "}" if len(g) > 1 else g[0]
                           for g in merged))
    else:
        print("    名次与 halfsphere 不一致。按预注册判据，§12 应改写为")
        print("    「仅 halfsphere 有可靠排名」。")
