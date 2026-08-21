"""Strength vs tolerance, one row per patch.

Tolerance is reported as the largest displacement at which the attack still keeps
half its un-jittered damage -- a single number that can be compared across arms,
rather than a curve the reader has to eyeball. Both metrics are shown because they
disagree: the 128 patch's headline RPE rotation is the fragile part, while a
coarser patch keeps its ATE damage much further out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import ReconsEval, gt_traj_for  # noqa: E402

RECONS = Path("/mnt/data/wangqq/recons_eval")
TUM = RECONS / "data/tum"
CLEAN_ROT, CLEAN_ATE, CEILING = 0.446, 0.00816, 0.251932


def half_life(mm, vals, base, floor):
    """Largest displacement where the attack keeps >=50% of its damage above floor."""
    excess = np.asarray(vals) - floor
    b = base - floor
    if b <= 1e-9:
        return None
    keep = excess / b
    ok = [m for m, k in zip(mm, keep) if k >= 0.5]
    return max(ok) if ok else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/jitter_eval/res_tolerance.npz")
    cli = ap.parse_args()
    d = np.load(cli.npz, allow_pickle=True)
    mags, pw = d["mags"], float(d["patch_width_m"])
    # grid_sample coordinates span [-1, 1] across the texture, so a jitter
    # magnitude m displaces the pattern by m/2 of the patch width. Offsets are
    # uniform in +/-m per axis, so these millimetres are per-axis maxima.
    mm = [m / 2.0 * pw * 1000 for m in mags]
    runs = str(d["runs"]).split(",")

    rec = ReconsEval(RECONS)
    work = Path("/tmp/jitter_eval/work")
    work.mkdir(parents=True, exist_ok=True)
    gt_traj, _ = gt_traj_for(rec, TUM, str(d["scene"]),
                             [int(i) for i in d["frame_indices"]],
                             "groundtruth_90.txt", work)

    def score(c2w):
        t = rec.evo_utils.get_tum_poses(np.asarray(c2w, dtype=np.float64))
        _, rot = rec.rpe(t, gt_traj, True, True)
        return float(rec.ate(t, gt_traj, True, True)), float(rot)

    print(f"贴图 {pw*1000:.0f} mm 宽   干净 RPE旋转 {CLEAN_ROT:.2f}度 / ATE {CLEAN_ATE:.4f} m"
          f"   ATE 上限 {CEILING:.4f} m（随机 90.5%）\n")
    hdr = (f"{'配置':<16}{'mm/纹素':>9}{'RPE旋转':>9}{'倍干净':>8}"
           f"{'ATE':>9}{'占上限':>8}{'RPE半衰':>9}{'ATE半衰':>9}")
    print(hdr)
    print("-" * 79)
    for run in runs:
        size = int(run.replace("res", "").split("_")[0])
        rots, ates = [], []
        for mi in range(len(mags)):
            key = f"{run}|{mi}"
            if key not in d:
                rots.append(np.nan), ates.append(np.nan)
                continue
            v = np.array([score(c) for c in d[key]])
            ates.append(v[:, 0].mean())
            rots.append(v[:, 1].mean())
        hr = half_life(mm, rots, rots[0], CLEAN_ROT)
        ha = half_life(mm, ates, ates[0], CLEAN_ATE)
        print(f"{run:<16}{pw*1000/size:>9.1f}{rots[0]:>9.2f}{rots[0]/CLEAN_ROT:>7.0f}x"
              f"{ates[0]:>9.4f}{ates[0]/CEILING:>8.1%}"
              f"{(f'{hr:.1f}mm' if hr is not None else '-'):>9}"
              f"{(f'{ha:.1f}mm' if ha is not None else '-'):>9}")

    print("\n半衰 = 仍保住一半「超出干净基线的破坏量」的最大位移；"
          "0.0mm 表示 0.3mm 就已经掉过半，>25.6mm 表示扫到头都没掉过半。")


if __name__ == "__main__":
    main()
