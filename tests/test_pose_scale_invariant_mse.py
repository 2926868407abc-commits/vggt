"""Unit tests for pose_scale_invariant_mse (the scale-leak fix in the pose loss).

Run it directly (the attack env has no pytest installed):

    /mnt/data/wangqq/conda_envs/vggt/bin/python3 tests/test_pose_scale_invariant_mse.py

The test functions follow the pytest naming convention, so `python -m pytest` also
works in any env that does have pytest.

The core property under test: for a prediction that differs from GT only by a
global similarity g = (s*R, t), the new loss must read ~0, because the evaluator
(`relpose/evo_utils.py::eval_metrics`, align=True + correct_scale=True) discards
exactly that g. The old `pose_relative_mse` is kept as the contrast row.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack_vggt_geometry_tum10 import (  # noqa: E402
    normalize_c2w_to_first,
    pose_relative_mse,
    pose_scale_invariant_mse,
)

DTYPE = torch.float64
SCALES = (0.5, 1.0, 2.0, 100.0)
ZERO_TOL = 1e-16       # float64 round-off ceiling for a loss that should vanish
LEAK_TOL = 1e-2        # below this the old loss would not count as "significantly > 0"


def make_args(rot_w: float = 1.0, trans_w: float = 1.0, eps: float = 1e-6) -> argparse.Namespace:
    return argparse.Namespace(
        pose_rotation_weight=rot_w,
        pose_translation_weight=trans_w,
        pose_scale_invariant_eps=eps,
    )


def random_rotation(gen: torch.Generator) -> torch.Tensor:
    """Uniformly-ish random rotation via Rodrigues on a random axis/angle."""
    axis = torch.randn(3, generator=gen, dtype=DTYPE)
    axis = axis / axis.norm()
    theta = torch.rand(1, generator=gen, dtype=DTYPE) * 2.0 * math.pi
    K = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=DTYPE,
    )
    return torch.eye(3, dtype=DTYPE) + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def make_gt_trajectory(n_frames: int = 10, seed: int = 0, span: float = 0.03) -> torch.Tensor:
    """A stand-in for a TUM GT trajectory: (1, N, 4, 4) camera-to-world."""
    gen = torch.Generator().manual_seed(seed)
    c2w = torch.eye(4, dtype=DTYPE).repeat(n_frames, 1, 1)
    for i in range(n_frames):
        c2w[i, :3, :3] = random_rotation(gen)
        c2w[i, :3, 3] = torch.randn(3, generator=gen, dtype=DTYPE) * span
    return c2w.unsqueeze(0)


def apply_sim3(c2w: torch.Tensor, scale: float, rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """g = (scale * rot, trans) acting on a camera-to-world trajectory."""
    out = c2w.clone()
    out[..., :3, :3] = rot @ c2w[..., :3, :3]
    out[..., :3, 3] = scale * (c2w[..., :3, 3] @ rot.T) + trans
    return out


def _losses_under_sim3(scale: float, seed: int) -> tuple[float, float]:
    """Returns (old_loss, new_loss) for pred = g * GT with a random (R, t)."""
    args = make_args()
    gen = torch.Generator().manual_seed(seed)
    gt = make_gt_trajectory()
    rot = random_rotation(gen)
    trans = torch.randn(3, generator=gen, dtype=DTYPE) * 0.7
    pred = apply_sim3(gt, scale, rot, trans)
    gt_rel = normalize_c2w_to_first(gt)
    pred_rel = normalize_c2w_to_first(pred)
    old, _ = pose_relative_mse(pred_rel, gt_rel, args)
    new, _ = pose_scale_invariant_mse(pred_rel, gt_rel, args)
    return float(old), float(new)


# --------------------------------------------------------------------------- #
# the property the fix is for
# --------------------------------------------------------------------------- #
def test_new_loss_vanishes_under_random_sim3():
    """pred = (s*R, t) . GT must give ~0 for every s, for several random (R, t)."""
    for scale in SCALES:
        for draw in range(5):
            _, new = _losses_under_sim3(scale, seed=1000 * draw + int(scale * 10))
            assert new < ZERO_TOL, f"new loss not ~0 at s={scale}, draw={draw}: {new:.3e}"


def test_old_loss_leaks_scale():
    """The old loss must be significantly > 0 whenever s != 1 -- that is the leak."""
    for scale in SCALES:
        if scale == 1.0:
            continue
        for draw in range(5):
            old, _ = _losses_under_sim3(scale, seed=1000 * draw + int(scale * 10))
            assert old > LEAK_TOL, f"old loss unexpectedly small at s={scale}: {old:.3e}"


def test_old_loss_already_ignores_rotation_and_translation():
    """At s == 1 the OLD loss is ~0 too.

    normalize_c2w_to_first makes the relative poses invariant to a global
    rotation and translation on its own, so the only gauge the old loss actually
    leaks is the scale DOF. This test pins that down so the fix is not
    mis-attributed to the R/t part.
    """
    for draw in range(5):
        old, new = _losses_under_sim3(1.0, seed=1000 * draw + 10)
        assert old < ZERO_TOL, f"old loss not ~0 at s=1: {old:.3e}"
        assert new < ZERO_TOL, f"new loss not ~0 at s=1: {new:.3e}"


def test_scale_invariance_on_an_unrelated_prediction():
    """Invariance must hold for any prediction, not only for pred = g . GT."""
    args = make_args()
    gt_rel = normalize_c2w_to_first(make_gt_trajectory(seed=0))
    pred = make_gt_trajectory(seed=7)

    base, _ = pose_scale_invariant_mse(normalize_c2w_to_first(pred), gt_rel, args)
    old_base, _ = pose_relative_mse(normalize_c2w_to_first(pred), gt_rel, args)
    assert float(base) > 0.1, "degenerate fixture: unrelated trajectories should not match"

    for scale in SCALES:
        scaled = pred.clone()
        scaled[..., :3, 3] *= scale
        rel = normalize_c2w_to_first(scaled)
        new, _ = pose_scale_invariant_mse(rel, gt_rel, args)
        old, _ = pose_relative_mse(rel, gt_rel, args)
        rel_change = abs(float(new) - float(base)) / float(base)
        assert rel_change < 1e-12, f"new loss moved at s={scale}: relative {rel_change:.3e}"
        if scale != 1.0:
            assert abs(float(old) - float(old_base)) > 1e-3, (
                f"old loss should have moved at s={scale}"
            )


def test_rotation_term_is_untouched():
    """The fix must change the translation term only."""
    args = make_args()
    gt_rel = normalize_c2w_to_first(make_gt_trajectory(seed=0))
    pred_rel = normalize_c2w_to_first(make_gt_trajectory(seed=7))
    _, old_terms = pose_relative_mse(pred_rel, gt_rel, args)
    _, new_terms = pose_scale_invariant_mse(pred_rel, gt_rel, args)
    assert old_terms["pose_rot_mse"] == new_terms["pose_rot_mse"]


def test_translation_term_is_bounded():
    """Both fields have unit RMS after normalization, so the term is <= 4/3."""
    args = make_args()
    worst = 0.0
    for k in range(200):
        a = normalize_c2w_to_first(make_gt_trajectory(seed=100 + k))
        b = normalize_c2w_to_first(make_gt_trajectory(seed=900 + k))
        _, terms = pose_scale_invariant_mse(a, b, args)
        worst = max(worst, terms["pose_trans_mse"])
    assert worst <= 4.0 / 3.0 + 1e-9, f"translation term exceeded 4/3: {worst}"


def test_degenerate_trajectory_does_not_explode():
    """A static prediction (all relative translations zero) must stay finite."""
    args = make_args()
    gt_rel = normalize_c2w_to_first(make_gt_trajectory(seed=0))
    static = torch.eye(4, dtype=DTYPE).repeat(10, 1, 1).unsqueeze(0).requires_grad_(True)

    loss, terms = pose_scale_invariant_mse(normalize_c2w_to_first(static), gt_rel, args)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert math.isfinite(terms["pose_trans_mse"])
    loss.backward()
    assert torch.isfinite(static.grad).all(), "gradient is not finite for a static trajectory"

    # both sides degenerate
    loss, _ = pose_scale_invariant_mse(
        normalize_c2w_to_first(static.detach()),
        normalize_c2w_to_first(torch.eye(4, dtype=DTYPE).repeat(10, 1, 1).unsqueeze(0)),
        args,
    )
    assert torch.isfinite(loss) and float(loss) == 0.0


def test_gradient_is_finite_on_a_normal_trajectory():
    args = make_args()
    gt_rel = normalize_c2w_to_first(make_gt_trajectory(seed=0))
    pred = make_gt_trajectory(seed=3).clone().requires_grad_(True)
    loss, _ = pose_scale_invariant_mse(normalize_c2w_to_first(pred), gt_rel, args)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.norm()) > 0.0


# --------------------------------------------------------------------------- #
def _main() -> int:
    print(f"{'s':>8} | {'old_loss':>14} | {'new_loss':>14} | verdict")
    print("-" * 62)
    for scale in SCALES:
        old, new = _losses_under_sim3(scale, seed=10 + int(scale * 10))
        verdict = "old leaks" if old > LEAK_TOL else "both ~0 (R/t cancel already)"
        print(f"{scale:8.1f} | {old:14.6e} | {new:14.6e} | {verdict}")
    print()

    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
