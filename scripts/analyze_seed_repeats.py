"""Is the RPE-rotation ranking on sitting_halfsphere stable across seeds?

Three runs per loss (seeds 0,1,2). The question is not whether the numbers repeat
to the digit -- bf16 on GPU alone guarantees they will not -- but whether the
spread within a loss is small compared to the gaps between losses. If it is not,
the ranking is not a result.
"""
import csv
from pathlib import Path

import numpy as np

GAUGE = Path("/mnt/data/wangqq/recons_eval/outputs/relpose-distance")
HALF = "rgbd_dataset_freiburg3_sitting_halfsphere"
LOSSES = [("old", "旧 pose_gt"), ("si", "尺度不变"), ("pw", "成对相对"), ("al", "对齐残差")]


def row_for(model):
    p = GAUGE / f"tum10-gauge-{model}.csv"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["seq"] == HALF:
                return r
    return None


def runs_for(tag):
    """seed 0 came from the original ablation, seeds 1-2 from the repeat launcher."""
    out = []
    for model in (f"vggt_mon_monhalf_{tag}_1000",
                  f"vggt_rep_monhalf_{tag}_s1",
                  f"vggt_rep_monhalf_{tag}_s2"):
        r = row_for(model)
        if r is not None:
            out.append(r)
    return out


print("sitting_halfsphere · 显示器 · 每格 3 次重复")
print(f"{'损失':<10}{'RPE旋转 三次':>28}{'均值':>9}{'标准差':>9}{'变异':>8}{'占上限均值':>11}")
print("-" * 78)
stats = []
for tag, llabel in LOSSES:
    rs = runs_for(tag)
    if not rs:
        print(f"{llabel:<10}  没有结果")
        continue
    rot = np.array([float(r["RPE_rot_deg"]) for r in rs])
    frac = np.array([float(r["ate_frac_of_ceiling"]) for r in rs])
    cv = rot.std(ddof=1) / rot.mean() if len(rot) > 1 and rot.mean() > 0 else float("nan")
    stats.append((llabel, rot, cv))
    trip = "  ".join(f"{v:7.2f}" for v in rot)
    print(f"{llabel:<10}{trip:>28}{rot.mean():>9.2f}"
          f"{(rot.std(ddof=1) if len(rot) > 1 else float('nan')):>9.2f}"
          f"{cv:>8.1%}{frac.mean():>11.1%}")

if len(stats) >= 2:
    print("\n排序是否稳：看最强那格的最低值有没有超过第二名的最高值")
    order = sorted(stats, key=lambda s: -s[1].mean())
    (l1, r1, _), (l2, r2, _) = order[0], order[1]
    print(f"  第一 {l1}: 最低 {r1.min():.2f}°")
    print(f"  第二 {l2}: 最高 {r2.max():.2f}°")
    print("  → " + ("区间不重叠，排序稳" if r1.min() > r2.max()
                    else "区间重叠，这两名分不开"))
