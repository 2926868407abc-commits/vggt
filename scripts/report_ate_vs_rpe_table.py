"""Final table: ATE fraction saturates, RPE rotation does not.

RPE compares frame-to-frame relative motion, so a global Sim(3) does not cancel in
it the way it does in the aligned ATE. That makes it the tie-breaker for the cells
sitting at the random-noise level.
"""
import csv
from pathlib import Path

GAUGE = Path("/mnt/data/wangqq/recons_eval/outputs/relpose-distance")
XYZ = "rgbd_dataset_freiburg3_sitting_xyz"
HALF = "rgbd_dataset_freiburg3_sitting_halfsphere"
LOSSES = [("old", "旧 pose_gt"), ("si", "尺度不变"), ("pw", "成对相对"), ("al", "对齐残差")]
GROUPS = [
    ("显示器 · xyz", "mon_monxyz", XYZ, 0.938),
    ("显示器 · half", "mon_monhalf", HALF, 0.905),
    ("海报 · half", "mon_posthalf", HALF, 0.905),
    ("自动 · xyz（旧）", "cal_sitting_xyz", XYZ, 0.938),
    ("自动 · half（旧）", "cal_sitting_halfsphere", HALF, 0.905),
]


def rows_of(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {r["seq"]: r for r in csv.DictReader(fh)}


clean = rows_of(GAUGE / "tum10-gauge-cleanref.csv")
print("干净基线（攻击前 VGGT 自己的误差）")
for s in (XYZ, HALF):
    c = clean.get(s, {})
    print(f"  {s.replace('rgbd_dataset_freiburg3_', ''):<14} "
          f"ATE {float(c.get('ATE_align_scale', 'nan')):.4f} m  "
          f"RPE平移 {float(c.get('RPE_trans', 'nan')):.4f}  "
          f"RPE旋转 {float(c.get('RPE_rot_deg', 'nan')):.3f}°")

for glabel, prefix, scene, rand in GROUPS:
    cr = clean.get(scene, {})
    base_rot = float(cr.get("RPE_rot_deg", "nan"))
    print(f"\n=== {glabel}   随机基线 {rand:.1%}   干净 RPE旋转 {base_rot:.2f}°")
    print(f"{'损失':<10}{'占上限':>9}{'饱和':>6}{'RPE旋转°':>11}{'相对干净':>10}{'RPE平移':>10}")
    print("-" * 58)
    ranked = []
    for tag, llabel in LOSSES:
        p = GAUGE / f"tum10-gauge-vggt_{prefix}_{tag}_1000.csv"
        rs = rows_of(p)
        r = rs.get(scene) or (next(iter(rs.values())) if rs else None)
        if r is None:
            print(f"{llabel:<10}  缺失")
            continue
        frac = float(r["ate_frac_of_ceiling"])
        rot = float(r["RPE_rot_deg"])
        sat = "★" if frac >= rand - 0.05 else ""
        ranked.append((rot, llabel))
        print(f"{llabel:<10}{frac:>9.1%}{sat:>6}{rot:>11.2f}"
              f"{rot / base_rot:>9.0f}x{float(r['RPE_trans']):>10.4f}")
    if ranked:
        best = max(ranked)
        print(f"    → 按 RPE 旋转最强: {best[1]} ({best[0]:.1f}°)")
