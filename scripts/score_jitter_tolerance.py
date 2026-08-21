"""Score the tolerance sweep (recons_eval env)."""

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/jitter_eval/tolerance.npz")
    cli = ap.parse_args()
    d = np.load(cli.npz, allow_pickle=True)
    scene = str(d["scene"])
    mags = d["mags"]
    pw = float(d["patch_width_m"])
    runs = str(d["runs"]).split(",")

    rec = ReconsEval(RECONS)
    work = Path("/tmp/jitter_eval/work")
    work.mkdir(parents=True, exist_ok=True)
    gt_traj, _ = gt_traj_for(rec, TUM, scene, [int(i) for i in d["frame_indices"]],
                             "groundtruth_90.txt", work)

    def score(c2w):
        t = rec.evo_utils.get_tum_poses(np.asarray(c2w, dtype=np.float64))
        _, rot = rec.rpe(t, gt_traj, True, True)
        return float(rec.ate(t, gt_traj, True, True)), float(rot)

    print(f"{scene}   贴图宽 {pw:.3f} m   干净基线 RPE旋转 0.45 度 / ATE 0.0082 m\n")
    for run in runs:
        print(f"=== {run}")
        print(f"{'平移抖动':>12}{'实际位移':>10}{'RPE旋转':>10}{'±':>8}{'ATE':>10}{'占无抖动':>10}")
        print("-" * 62)
        base = None
        for mi, mag in enumerate(mags):
            key = f"{run}|{mi}"
            if key not in d:
                continue
            vals = np.array([score(c) for c in d[key]])
            rot = vals[:, 1]
            ate = vals[:, 0]
            if base is None:
                base = rot.mean()
            sd = rot.std(ddof=1) if len(rot) > 1 else 0.0
            print(f"{mag:>12.4f}{mag * pw * 1000:>8.1f}mm{rot.mean():>10.2f}"
                  f"{sd:>8.2f}{ate.mean():>10.4f}{rot.mean() / base:>9.0%}")
        print()


if __name__ == "__main__":
    main()
