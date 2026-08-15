"""Which frame-varying gauge g_i produces error a single Sim(3) cannot absorb?

The evaluator fits ONE similarity to the whole trajectory before measuring, so a
constant g = (s R, t) is removed exactly and scores zero. A g_i that varies with
the frame cannot be removed by one transform, and whatever it leaves behind is
real ATE. This is the target design question: which families of g_i convert into
the most surviving error, and how smooth can they be while doing it.

Smoothness matters because a target that jumps between neighbouring frames is
both harder for a patch to realise and trivially visible to any temporal check.
So each family is scored on both axes:

  ate_frac   what fraction of the ATE ceiling survives the alignment
  absorbed   how much of the displacement one global Sim(3) explains (lower = better)
  jump       largest frame-to-frame change in the transform, relative to its own
             span; a hard split is 1.0, a linear ramp is 1/(N-1)

Nothing here involves the model or the patch: it operates on the ground-truth
trajectory and asks what an attack would have to achieve. Whether a patch can
actually drive VGGT to such a target is the next question, not this one.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/data/wangqq/vggt/scripts")
from diag_gauge_invariance import ReconsEval, gt_traj_for, load_run  # noqa: E402

RECONS = Path("/mnt/data/wangqq/recons_eval")
CLEAN = Path("/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3")


def yaw(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def apply_per_frame(c2w: np.ndarray, scales, yaws, trans) -> np.ndarray:
    """Apply a different similarity to each frame."""
    out = np.array(c2w, dtype=np.float64, copy=True)
    for i in range(c2w.shape[0]):
        R = yaw(yaws[i])
        out[i, :3, :3] = R @ c2w[i, :3, :3]
        out[i, :3, 3] = scales[i] * (R @ c2w[i, :3, 3]) + trans[i]
    return out


def ramp(n, lo, hi):
    return np.linspace(lo, hi, n)


def split(n, a, b, k=None):
    k = k if k is not None else n // 2
    return np.array([a] * k + [b] * (n - k), dtype=np.float64)


def sine(n, base, amp, cycles=1.0):
    return base + amp * np.sin(2 * math.pi * cycles * np.arange(n) / max(n - 1, 1))


def families(n: int, mag: float):
    """(name, scales, yaws_deg, translations) for one magnitude setting."""
    zero_t = np.zeros((n, 3))
    unit = np.ones(n)
    d = np.asarray([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    return [
        ("constant scale (control)", unit * (1 + mag), np.zeros(n), zero_t),
        ("constant yaw (control)", unit, np.full(n, 30.0 * mag), zero_t),
        ("constant trans (control)", unit, np.zeros(n), np.tile(d * mag, (n, 1))),
        ("scale ramp", ramp(n, 1.0, 1.0 + mag), np.zeros(n), zero_t),
        ("scale split", split(n, 1.0, 1.0 + mag), np.zeros(n), zero_t),
        ("scale sine", sine(n, 1.0, mag), np.zeros(n), zero_t),
        ("yaw ramp", unit, ramp(n, 0.0, 30.0 * mag), zero_t),
        ("yaw split", unit, split(n, 0.0, 30.0 * mag), zero_t),
        ("trans ramp", unit, np.zeros(n), np.outer(ramp(n, 0.0, mag), d)),
        ("trans split", unit, np.zeros(n), np.outer(split(n, 0.0, mag), d)),
        ("scale ramp + yaw ramp", ramp(n, 1.0, 1.0 + mag), ramp(n, 0.0, 30.0 * mag), zero_t),
    ]


def jumpiness(scales, yaws, trans) -> float:
    """Largest frame-to-frame step, normalised by the total span of each component."""
    out = 0.0
    for v in (np.asarray(scales), np.asarray(yaws)):
        span = float(np.ptp(v))
        if span > 1e-12:
            out = max(out, float(np.max(np.abs(np.diff(v)))) / span)
    t = np.asarray(trans)
    span = float(np.linalg.norm(t.max(0) - t.min(0)))
    if span > 1e-12:
        out = max(out, float(np.max(np.linalg.norm(np.diff(t, axis=0), axis=1))) / span)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_xyz")
    p.add_argument("--mags", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    p.add_argument("--work_dir", default="/tmp/piecewise_work")
    args = p.parse_args()

    rec = ReconsEval(RECONS)
    umeyama, _, _ = rec.pointmap_fns()
    clean = load_run(CLEAN, args.scene)
    gt_traj, gt_c2w = gt_traj_for(rec, RECONS / "data/tum", args.scene,
                                  clean["frame_indices"], "groundtruth_90.txt",
                                  Path(args.work_dir))
    n = gt_c2w.shape[0]
    p_gt = gt_c2w[:, :3, 3]
    ceiling = float(np.sqrt(((p_gt - p_gt.mean(0)) ** 2).sum(1).mean()))
    print(f"scene {args.scene}   {n} frames   ATE ceiling {ceiling:.5f} m")
    print("the target is GT transformed by g_i; a constant g_i must score ~0\n")

    for mag in args.mags:
        print(f"=== magnitude {mag:g} ===")
        hdr = (f"{'g_i family':<26}{'ATE':>10}{'%ceil':>8}{'absorbed':>10}"
               f"{'jump':>8}{'disp/radius':>13}")
        print(hdr); print("-" * len(hdr))
        for name, sc, yw, tr in families(n, mag):
            tgt = apply_per_frame(gt_c2w, sc, yw, tr)
            ate = rec.ate(rec.evo_utils.get_tum_poses(tgt), gt_traj, True, True)
            q = tgt[:, :3, 3]
            c, R, t = umeyama(q.T, p_gt.T)
            aligned = (c * (R @ q.T) + t).T
            raw = float(np.sqrt(((q - p_gt) ** 2).sum(1).mean()))
            resid = float(np.sqrt(((aligned - p_gt) ** 2).sum(1).mean()))
            absorbed = 1.0 - resid / raw if raw > 1e-12 else float("nan")
            print(f"{name:<26}{ate:>10.5f}{ate/ceiling*100:>7.1f}%{absorbed:>10.4f}"
                  f"{jumpiness(sc, yw, tr):>8.3f}{raw/ceiling:>13.2f}")
        print()


if __name__ == "__main__":
    main()
