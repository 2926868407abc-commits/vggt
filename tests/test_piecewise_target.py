"""The target the attack aims at must be the one the feasibility study measured.

scripts/diag_piecewise_gauge.py established which per-frame gauges survive the
evaluator's Sim(3) fit. attack_vggt_geometry_tum10.apply_piecewise_gauge builds the
target the attack actually optimises toward. If the two constructions drift apart,
training aims at something whose ATE was never validated -- so this pins them
together on the real TUM trajectories.

    /mnt/data/wangqq/conda_envs/recons_eval/bin/python3 tests/test_piecewise_target.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

VGGT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VGGT_ROOT))
sys.path.insert(0, str(VGGT_ROOT / "scripts"))

from attack_vggt_geometry_tum10 import (  # noqa: E402
    apply_piecewise_gauge, piecewise_gauge_schedule, quat_xyzw_to_rot,
    read_tum_rows, tum_rows_to_c2w,
)
from diag_gauge_invariance import ReconsEval, gt_traj_for, load_run  # noqa: E402

RECONS = Path("/mnt/data/wangqq/recons_eval")
CLEAN = VGGT_ROOT / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"
SCENES = ["rgbd_dataset_freiburg3_sitting_xyz",
          "rgbd_dataset_freiburg3_sitting_halfsphere",
          "rgbd_dataset_freiburg3_sitting_static"]
FAMILIES = ["scale_ramp", "scale_sine", "scale_split", "yaw_ramp", "trans_ramp"]


def yaw_reference(deg: float) -> np.ndarray:
    """Independent yaw construction, matching diag_piecewise_gauge.yaw()."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def reference_apply(c2w, family, mag):
    scales, yaws, trans = piecewise_gauge_schedule(family, c2w.shape[0], mag)
    out = np.array(c2w, dtype=np.float64, copy=True)
    for i in range(c2w.shape[0]):
        R = yaw_reference(yaws[i])
        out[i, :3, :3] = R @ c2w[i, :3, :3]
        out[i, :3, 3] = scales[i] * (R @ c2w[i, :3, 3]) + trans[i]
    return out


def test_quaternion_yaw_matches_the_rotation_matrix_form():
    """apply_piecewise_gauge builds the yaw from a quaternion; the study used a
    matrix. They must be the same rotation about the same axis."""
    for deg in (-45.0, -7.5, 0.0, 12.0, 30.0, 90.0):
        q = quat_xyzw_to_rot(0.0, math.sin(math.radians(deg) / 2.0), 0.0,
                             math.cos(math.radians(deg) / 2.0))
        assert np.abs(q - yaw_reference(deg)).max() < 1e-12, (
            f"yaw {deg} deg: quaternion form differs from the matrix form by "
            f"{np.abs(q - yaw_reference(deg)).max():.3e}")


def test_target_matches_the_feasibility_construction():
    for scene in SCENES:
        rows = read_tum_rows(RECONS / "data/tum" / scene / "groundtruth_90.txt")
        idx = load_run(CLEAN, scene)["frame_indices"]
        gt = tum_rows_to_c2w(rows, idx)
        for fam in FAMILIES:
            for mag in (0.3, 1.0):
                a = apply_piecewise_gauge(gt, fam, mag)
                b = reference_apply(gt, fam, mag)
                d = float(np.abs(a - b).max())
                assert d < 1e-12, f"{scene} {fam} mag={mag}: differs by {d:.3e}"


def test_constant_gauge_is_still_removed_entirely():
    """Sanity on the metric side: magnitude 0 must leave the trajectory untouched."""
    rows = read_tum_rows(RECONS / "data/tum" / SCENES[0] / "groundtruth_90.txt")
    gt = tum_rows_to_c2w(rows, load_run(CLEAN, SCENES[0])["frame_indices"])
    for fam in FAMILIES:
        same = apply_piecewise_gauge(gt, fam, 0.0)
        assert np.abs(same - gt).max() < 1e-12, f"{fam} at magnitude 0 moved the trajectory"


def test_targets_reach_the_ate_the_study_predicted():
    """End to end: the built target, scored by the real evaluator, must land where
    the feasibility study said it would."""
    rec = ReconsEval(RECONS)
    work = Path("/tmp/piecewise_target_test")
    work.mkdir(parents=True, exist_ok=True)
    print(f"\n{'scene':<40}{'family':<13}{'mag':>5}{'ATE':>10}{'%ceiling':>10}")
    print("-" * 78)
    for scene in SCENES:
        clean = load_run(CLEAN, scene)
        gt_traj, gt_c2w = gt_traj_for(rec, RECONS / "data/tum", scene,
                                      clean["frame_indices"], "groundtruth_90.txt", work)
        p = gt_c2w[:, :3, 3]
        ceiling = float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))
        base = rec.ate(rec.evo_utils.get_tum_poses(gt_c2w), gt_traj, True, True)
        assert base < 1e-9, f"{scene}: untransformed GT should score ~0, got {base:.3e}"
        for fam in ("scale_sine", "trans_ramp"):
            tgt = apply_piecewise_gauge(gt_c2w, fam, 1.0)
            ate = rec.ate(rec.evo_utils.get_tum_poses(tgt), gt_traj, True, True)
            print(f"{scene:<40}{fam:<13}{1.0:>5.1f}{ate:>10.5f}{ate/ceiling*100:>9.1f}%")
            assert ate / ceiling > 0.4, (
                f"{scene} {fam}: only {ate/ceiling*100:.1f}% of ceiling survives; the "
                f"feasibility study measured >=56% for these families")


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
