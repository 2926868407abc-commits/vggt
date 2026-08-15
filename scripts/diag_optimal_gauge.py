"""The smoothest trajectory displacement a single Sim(3) cannot absorb.

The hand-built gauge families top out: raising their magnitude raises the raw
displacement but the absorbed fraction rises with it, so what survives the
alignment stops growing (trans_ramp saturates at 76%, scale_sine does not grow at
all). Picking families by hand is the wrong move, because the problem has a closed
form.

ATE depends only on camera positions, so a target is just a displacement field
delta in R^{3N}. The evaluator removes one Sim(3), whose action at the identity
spans exactly 7 directions in that space:

    3 translation   delta_i = e_k
    3 rotation      delta_i = omega_k x p_i
    1 scale         delta_i = p_i

Anything inside that span is erased no matter how large. Anything orthogonal to it
survives in full. So the question "which g_i is efficient" is really "which
displacement fields are orthogonal to those 7 vectors", and the smooth ones can be
enumerated: take a low-frequency basis in the frame index, project the Sim(3) span
out of it, and read off what is left, lowest frequency first.

Reports, for each surviving mode, the ATE it produces through the real evaluator
and its frame-to-frame jump, so the result is comparable with the hand-built
families measured in diag_piecewise_gauge.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/data/wangqq/vggt/scripts")
from diag_gauge_invariance import ReconsEval, gt_traj_for, load_run  # noqa: E402

RECONS = Path("/mnt/data/wangqq/recons_eval")
CLEAN = Path("/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3")


def sim3_basis(p: np.ndarray) -> np.ndarray:
    """The 7 displacement directions a single Sim(3) can produce, as (7, 3N)."""
    n = p.shape[0]
    cols = []
    for k in range(3):                      # translation
        d = np.zeros_like(p); d[:, k] = 1.0
        cols.append(d.reshape(-1))
    centred = p - p.mean(0)
    for k in range(3):                      # rotation about each axis
        axis = np.zeros(3); axis[k] = 1.0
        cols.append(np.cross(axis[None, :], centred).reshape(-1))
    cols.append(centred.reshape(-1))        # scale
    return np.stack(cols)


def smooth_basis(n: int, max_order: int) -> np.ndarray:
    """Low-frequency profiles over the frame index, as (max_order+1, n)."""
    idx = np.arange(n, dtype=np.float64)
    return np.stack([np.cos(np.pi * m * idx / max(n - 1, 1))
                     for m in range(max_order + 1)])


def jumpiness(delta: np.ndarray) -> float:
    span = float(np.linalg.norm(delta.max(0) - delta.min(0)))
    if span < 1e-12:
        return 0.0
    return float(np.max(np.linalg.norm(np.diff(delta, axis=0), axis=1))) / span


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=[
        "rgbd_dataset_freiburg3_sitting_xyz",
        "rgbd_dataset_freiburg3_sitting_halfsphere",
        "rgbd_dataset_freiburg3_sitting_static"])
    ap.add_argument("--max_order", type=int, default=5)
    ap.add_argument("--amplitudes", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--work_dir", default="/tmp/optimal_gauge_work")
    args = ap.parse_args()

    rec = ReconsEval(RECONS)
    for scene in args.scenes:
        clean = load_run(CLEAN, scene)
        gt_traj, gt_c2w = gt_traj_for(rec, RECONS / "data/tum", scene,
                                      clean["frame_indices"], "groundtruth_90.txt",
                                      Path(args.work_dir))
        p = gt_c2w[:, :3, 3]
        n = p.shape[0]
        radius = float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))
        B = sim3_basis(p)
        Q, _ = np.linalg.qr(B.T)                       # (3N, 7) orthonormal Sim(3) span

        print(f"\n=== {scene.replace('rgbd_dataset_freiburg3_', '')} "
              f"({n} frames, ceiling {radius:.5f} m) ===")
        print(f"{'mode':<22}{'kept':>8}{'jump':>8}" +
              "".join(f"{'A=' + format(a, 'g'):>11}" for a in args.amplitudes))
        print("-" * (38 + 11 * len(args.amplitudes)))

        profiles = smooth_basis(n, args.max_order)
        for m, prof in enumerate(profiles):
            for k, axis_name in enumerate("xyz"):
                d = np.zeros((n, 3)); d[:, k] = prof
                v = d.reshape(-1)
                v_perp = v - Q @ (Q.T @ v)             # project the Sim(3) span out
                kept = np.linalg.norm(v_perp) / max(np.linalg.norm(v), 1e-12)
                if kept < 0.05:
                    continue                            # this mode is essentially absorbed
                delta = v_perp.reshape(n, 3)
                delta = delta / np.linalg.norm(delta) * math_sqrt_n(n)
                cells = []
                for amp in args.amplitudes:
                    tgt = np.array(gt_c2w, copy=True)
                    tgt[:, :3, 3] = p + amp * radius * delta
                    ate = rec.ate(rec.evo_utils.get_tum_poses(tgt), gt_traj, True, True)
                    cells.append(f"{ate / radius * 100:>10.1f}%")
                print(f"cos order {m}, {axis_name:<10}{kept:>8.3f}{jumpiness(delta):>8.3f}"
                      + "".join(cells))


def math_sqrt_n(n: int) -> float:
    return float(np.sqrt(n))


if __name__ == "__main__":
    main()
