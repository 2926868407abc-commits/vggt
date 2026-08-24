"""The strength / consistency frontier, and whether the dual-decay fix moves it.

Reports, per budget fraction and per decay setting: how much damage the attack did,
and whether the output filters would have caught it. A configuration is only useful
if it is simultaneously well above the clean baseline and inside every filter.

Filter status is read from the run's own recorded metrics against the calibrated
thresholds, not from whether the optimiser's internal hinge fired -- a run can end
inside its own scaled budget while still being outside the clean-calibrated one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import ReconsEval, gt_traj_for, load_run  # noqa: E402

R = VG / "outputs_attack_geometry_aware_tum10"
RECONS = Path("/mnt/data/wangqq/recons_eval")
SCENE = "rgbd_dataset_freiburg3_sitting_halfsphere"
CEILING, CLEAN_ATE, CLEAN_ROT, RANDOM = 0.251932, 0.00816, 0.446, 0.910
CLEAN_FRAC = CLEAN_ATE / CEILING

cfg = json.loads((VG / "configs/tum10_filter_budgets.json").read_text(encoding="utf-8"))
spec = cfg["sequences"][SCENE]["metrics"]
WORSE = cfg["worse_is"]

rec = ReconsEval(RECONS)
work = Path("/tmp/pareto_work")
work.mkdir(parents=True, exist_ok=True)
clean = load_run(R / "tum10_clean_uniform_l3", SCENE)
gt_traj, _ = gt_traj_for(rec, RECONS / "data/tum", SCENE, clean["frame_indices"],
                         "groundtruth_90.txt", work)


def outside(name: str, value: float) -> bool:
    """Is this metric past its clean-calibrated threshold?"""
    th = spec[name]["threshold"]
    return value < th if WORSE.get(name) == "low" else value > th


def evaluate(tag: str):
    p = R / tag / SCENE / "attack_summary.json"
    if not p.exists():
        return None
    t = rec.evo_utils.get_tum_poses(np.asarray(load_run(R / tag, SCENE)["c2w"],
                                               dtype=np.float64))
    _, rot = rec.rpe(t, gt_traj, True, True)
    ate = float(rec.ate(t, gt_traj, True, True)) / CEILING
    h = R / tag / "geometry_patch/training_history.jsonl"
    rows = [json.loads(l) for l in h.read_text(encoding="utf-8").splitlines() if l.strip()]
    last = rows[-1]
    metrics = {k[len("budget_metric_"):]: float(v) for k, v in last.items()
               if k.startswith("budget_metric_")}
    duals = {k[len("budget_dual_"):]: float(v) for k, v in last.items()
             if k.startswith("budget_dual_") and not k.startswith("budget_dual_after")}
    breached = [n for n, v in metrics.items() if n in spec and outside(n, v)]
    return ate, float(rot), metrics, duals, breached


print(f"halfsphere · 64x64 · 显示器放置 · 1000 步 · seed 0")
print(f"干净 ATE {CLEAN_FRAC:.1%} of ceiling / RPE 旋转 {CLEAN_ROT:.2f}°"
      f"   随机基线 {RANDOM:.1%}")
print(f"conf_std 阈值 {spec['conf_std']['threshold']:.3f}（越低越差，干净 "
      f"{spec['conf_std']['clean']:.3f}）\n")

hdr = f"{'档位':>6}{'衰减':>7}{'ATE 占上限':>13}{'RPE 旋转':>11}{'conf_std':>11}{'越界的过滤器':>22}"
print(hdr)
print("-" * 74)
rows_out = []
for frac in ("050", "080", "095", "120"):
    for dec in ("100", "099"):
        tag = f"pf{frac}_d{dec}"
        r = evaluate(tag)
        f_lbl = f"{int(frac)/100:.2f}"
        d_lbl = f"{int(dec)/100:.2f}"
        if r is None:
            print(f"{f_lbl:>6}{d_lbl:>7}   缺失")
            continue
        ate, rot, metrics, duals, breached = r
        rows_out.append((f_lbl, d_lbl, ate, rot, breached, duals))
        print(f"{f_lbl:>6}{d_lbl:>7}{ate:>12.1%}{rot:>10.2f}°"
              f"{metrics.get('conf_std', float('nan')):>11.3f}"
              f"{(','.join(breached) if breached else '无'):>22}")

print("\n=== 有没有「又强又合规」的点 ===")
STRONG = 3 * CLEAN_FRAC   # 至少是干净基线的三倍才算真攻击
usable = [r for r in rows_out if r[2] > STRONG and not r[4]]
print(f"判据：ATE > {STRONG:.1%}（干净的 3 倍）且所有过滤器均未越界")
if usable:
    for f_lbl, d_lbl, ate, rot, _, _ in usable:
        print(f"    ✅ 档位 {f_lbl} 衰减 {d_lbl}: ATE {ate:.1%}, 旋转 {rot:.2f}°")
else:
    print("    ❌ 没有。前沿是断的：要么攻击接近干净基线，要么过滤器越界。")

print("\n=== 衰减修复有没有作用 ===")
for frac in ("050", "080", "095", "120"):
    a = next((r for r in rows_out if r[0] == f"{int(frac)/100:.2f}" and r[1] == "1.00"), None)
    b = next((r for r in rows_out if r[0] == f"{int(frac)/100:.2f}" and r[1] == "0.99"), None)
    if a and b:
        print(f"    档位 {a[0]}: ATE {a[2]:.1%} -> {b[2]:.1%}   "
              f"（差 {b[2]-a[2]:+.1%}）")

print("\n=== 最终对偶乘子（>0 = 该约束曾长期绑定）===")
for f_lbl, d_lbl, _, _, _, duals in rows_out:
    live = {k: v for k, v in duals.items() if v > 1e-6}
    print(f"    档位 {f_lbl} 衰减 {d_lbl}: "
          + (", ".join(f"{k}={v:.2f}" for k, v in live.items()) if live else "全为 0"))
