"""The orthogonal-mode target must be what diag_optimal_gauge.py measured.

That script found displacement fields the evaluator's Sim(3) cannot absorb, and
reported the ATE they reach. attack_vggt_geometry_tum10 now builds the same fields
as an attack target. If the two drift apart, training aims at something whose worth
was never established.

    /mnt/data/wangqq/conda_envs/recons_eval/bin/python3 tests/test_orthogonal_mode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

VG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VG))
sys.path.insert(0, str(VG / "scripts"))

from attack_vggt_geometry_tum10 import (  # noqa: E402
    apply_piecewise_gauge, orthogonal_mode_displacement, read_tum_rows,
    sim3_action_basis, tum_rows_to_c2w,
)
from diag_gauge_invariance import ReconsEval, gt_traj_for, load_run  # noqa: E402

RECONS = Path("/mnt/data/wangqq/recons_eval")
CLEAN = VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"
SCENES = ["rgbd_dataset_freiburg3_sitting_xyz",
          "rgbd_dataset_freiburg3_sitting_halfsphere",
          "rgbd_dataset_freiburg3_sitting_static"]


def positions(scene):
    rows = read_tum_rows(RECONS / "data/tum" / scene / "groundtruth_90.txt")
    return tum_rows_to_c2w(rows, load_run(CLEAN, scene)["frame_indices"])


def test_displacement_is_orthogonal_to_the_sim3_span():
    """The whole point: no component of the target lies in the absorbable subspace."""
    for scene in SCENES:
        c2w = positions(scene)
        p = c2w[:, :3, 3]
        q, _ = np.linalg.qr(sim3_action_basis(p).T)
        for order in (2, 3, 4):
            for axis in (0, 1, 2):
                d = orthogonal_mode_displacement(p, order, axis, 1.0).reshape(-1)
                leak = float(np.linalg.norm(q.T @ d)) / max(float(np.linalg.norm(d)), 1e-12)
                assert leak < 1e-10, (
                    f"{scene} order={order} axis={axis}: {leak:.3e} of the displacement "
                    f"still lies inside the Sim(3) span and would be absorbed")


def test_magnitude_scales_the_field_linearly():
    p = positions(SCENES[0])[:, :3, 3]
    base = orthogonal_mode_displacement(p, 2, 0, 1.0)
    for mag in (0.5, 3.0, 10.0):
        got = orthogonal_mode_displacement(p, 2, 0, mag)
        assert np.abs(got - mag * base).max() < 1e-9, f"magnitude {mag} is not linear"


def test_zero_magnitude_leaves_the_trajectory_alone():
    for scene in SCENES:
        c2w = positions(scene)
        same = apply_piecewise_gauge(c2w, "orthogonal_mode", 0.0, order=2, axis=0)
        assert np.abs(same - c2w).max() < 1e-12


def test_reaches_the_ate_the_search_predicted():
    """Order 2 was measured at 89-93% of ceiling at magnitude 3."""
    rec = ReconsEval(RECONS)
    work = Path("/tmp/orthogonal_mode_test"); work.mkdir(parents=True, exist_ok=True)
    best_axis = {"rgbd_dataset_freiburg3_sitting_xyz": 0,
                 "rgbd_dataset_freiburg3_sitting_halfsphere": 1,
                 "rgbd_dataset_freiburg3_sitting_static": 1}
    print(f"\n{'scene':<24}{'order':>6}{'axis':>5}{'ATE':>10}{'%ceiling':>10}")
    print("-" * 55)
    for scene in SCENES:
        clean = load_run(CLEAN, scene)
        gt_traj, gt_c2w = gt_traj_for(rec, RECONS / "data/tum", scene,
                                      clean["frame_indices"], "groundtruth_90.txt", work)
        p = gt_c2w[:, :3, 3]
        ceiling = float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))
        axis = best_axis[scene]
        tgt = apply_piecewise_gauge(gt_c2w, "orthogonal_mode", 3.0, order=2, axis=axis)
        ate = rec.ate(rec.evo_utils.get_tum_poses(tgt), gt_traj, True, True)
        frac = ate / ceiling
        short = scene.replace("rgbd_dataset_freiburg3_", "")
        print(f"{short:<24}{2:>6}{axis:>5}{ate:>10.5f}{frac*100:>9.1f}%")
        assert frac > 0.85, (
            f"{scene}: order 2 axis {axis} only reached {frac*100:.1f}% of ceiling; "
            f"diag_optimal_gauge.py measured 89-93%")


def test_beats_the_best_hand_built_family():
    """The reason this exists: trans_ramp saturates around 76%."""
    rec = ReconsEval(RECONS)
    work = Path("/tmp/orthogonal_mode_test"); work.mkdir(parents=True, exist_ok=True)
    scene = SCENES[0]
    clean = load_run(CLEAN, scene)
    gt_traj, gt_c2w = gt_traj_for(rec, RECONS / "data/tum", scene,
                                  clean["frame_indices"], "groundtruth_90.txt", work)
    p = gt_c2w[:, :3, 3]
    ceiling = float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))
    ramp = rec.ate(rec.evo_utils.get_tum_poses(
        apply_piecewise_gauge(gt_c2w, "trans_ramp", 3.0)), gt_traj, True, True) / ceiling
    orth = rec.ate(rec.evo_utils.get_tum_poses(
        apply_piecewise_gauge(gt_c2w, "orthogonal_mode", 3.0, order=2, axis=0)),
        gt_traj, True, True) / ceiling
    print(f"\n  sitting_xyz at magnitude 3: trans_ramp {ramp*100:.1f}%  "
          f"orthogonal_mode {orth*100:.1f}%")
    assert orth > ramp + 0.10, (
        f"orthogonal mode {orth*100:.1f}% should clearly beat trans_ramp {ramp*100:.1f}%")


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
