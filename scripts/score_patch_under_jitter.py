"""Score the trajectories eval_patch_under_jitter.py produced (recons_eval env)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sys

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import ReconsEval, gt_traj_for  # noqa: E402

RECONS = Path("/mnt/data/wangqq/recons_eval")
TUM = RECONS / "data/tum"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/jitter_eval/trajectories.npz")
    ap.add_argument("--gt_name", default="groundtruth_90.txt")
    cli = ap.parse_args()

    d = np.load(cli.npz, allow_pickle=True)
    scene = str(d["scene"])
    run = str(d["run"])
    idx = [int(i) for i in d["frame_indices"]]

    rec = ReconsEval(RECONS)
    work = Path("/tmp/jitter_eval/work")
    work.mkdir(parents=True, exist_ok=True)
    gt_traj, _ = gt_traj_for(rec, TUM, scene, idx, cli.gt_name, work)

    def score(c2w):
        traj = rec.evo_utils.get_tum_poses(np.asarray(c2w, dtype=np.float64))
        _, rot = rec.rpe(traj, gt_traj, True, True)
        return float(rec.ate(traj, gt_traj, True, True)), float(rot)

    ate0, rot0 = score(d["nojitter"][0])
    print(f"{run}  {scene}")
    print(f"无抖动（训练时的那个对齐）: ATE {ate0:.4f} m   RPE旋转 {rot0:.2f} 度\n")

    print(f"{'抖动类型':<14}{'RPE旋转均值':>13}{'标准差':>9}{'最小':>8}{'最大':>8}"
          f"{'保留':>8}{'ATE均值':>10}")
    print("-" * 71)
    for key, label in (("geo", "几何"), ("photo", "光度"), ("full", "全开")):
        if key not in d:
            continue
        vals = np.array([score(c) for c in d[key]])
        ate, rot = vals[:, 0], vals[:, 1]
        print(f"{label:<14}{rot.mean():>13.2f}{rot.std(ddof=1):>9.2f}{rot.min():>8.2f}"
              f"{rot.max():>8.2f}{rot.mean() / rot0:>7.0%}{ate.mean():>10.4f}")

    print("\n保留 = 抖动后 RPE旋转均值 / 无抖动值。该序列干净基线 0.45 度。")


if __name__ == "__main__":
    main()
