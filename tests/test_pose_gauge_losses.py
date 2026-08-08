"""Unit tests for the two gauge-aware pose losses.

    pose_pairwise_relative_mse   -- all N(N-1)/2 pairs, no privileged frame
    pose_aligned_residual_mse    -- residual after the evaluator's own Sim(3) alignment

Run directly (the attack env has no pytest):

    /mnt/data/wangqq/conda_envs/vggt/bin/python3 tests/test_pose_gauge_losses.py

The test functions follow the pytest naming convention, so `python -m pytest`
also works in any env that has it.
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
    pose_aligned_residual_mse,
    pose_pairwise_relative_mse,
    pose_scale_invariant_mse,
    umeyama_sim3_torch,
)

DTYPE = torch.float64
SCALES = (0.5, 1.0, 2.0, 100.0)

# How small "should be zero" can actually get differs by construction:
#   pairwise  is a plain MSE of differences, so it bottoms out near eps^2 ~ 1e-32
#   aligned   takes sqrt of a mean-squared residual, and sqrt(1e-30) is 1e-15, so
#             its floor is the square root of the MSE floor. Loosening it to a
#             single shared constant would silently weaken the pairwise check.
ZERO_TOL = {"pairwise": 1e-16, "aligned": 1e-12}


def make_args(rot_w=1.0, trans_w=1.0, eps=1e-6):
    return argparse.Namespace(pose_rotation_weight=rot_w,
                              pose_translation_weight=trans_w,
                              pose_scale_invariant_eps=eps)


def random_rotation(gen):
    axis = torch.randn(3, generator=gen, dtype=DTYPE)
    axis = axis / axis.norm()
    theta = torch.rand(1, generator=gen, dtype=DTYPE) * 2.0 * math.pi
    K = torch.tensor([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
                     dtype=DTYPE)
    return torch.eye(3, dtype=DTYPE) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def make_traj(n_frames=10, seed=0, span=0.03):
    gen = torch.Generator().manual_seed(seed)
    c2w = torch.eye(4, dtype=DTYPE).repeat(n_frames, 1, 1)
    for i in range(n_frames):
        c2w[i, :3, :3] = random_rotation(gen)
        c2w[i, :3, 3] = torch.randn(3, generator=gen, dtype=DTYPE) * span
    return c2w.unsqueeze(0)


def apply_sim3(c2w, scale, rot, trans):
    out = c2w.clone()
    out[..., :3, :3] = rot @ c2w[..., :3, :3]
    out[..., :3, 3] = scale * (c2w[..., :3, 3] @ rot.T) + trans
    return out


LOSSES = {
    "pairwise": pose_pairwise_relative_mse,
    "aligned": pose_aligned_residual_mse,
}


# --------------------------------------------------------------------------- #
# gauge invariance
# --------------------------------------------------------------------------- #
def test_both_losses_vanish_under_random_sim3():
    """pred = (s*R, t) . GT must score ~0 for every s, for both losses."""
    args = make_args()
    gt = make_traj()
    gt_rel = normalize_c2w_to_first(gt)
    for name, fn in LOSSES.items():
        for scale in SCALES:
            for draw in range(5):
                gen = torch.Generator().manual_seed(1000 * draw + int(scale * 10))
                rot = random_rotation(gen)
                trans = torch.randn(3, generator=gen, dtype=DTYPE) * 0.7
                pred_rel = normalize_c2w_to_first(apply_sim3(gt, scale, rot, trans))
                val, _ = fn(pred_rel, gt_rel, args)
                assert float(val) < ZERO_TOL[name], (
                    f"{name} not ~0 at s={scale}: {float(val):.3e} "
                    f"(tolerance {ZERO_TOL[name]:.0e})")


def test_losses_are_invariant_when_pred_is_transformed():
    """For an ARBITRARY pred, applying a Sim(3) to it must not move either loss."""
    args = make_args()
    gt_rel = normalize_c2w_to_first(make_traj(seed=0))
    pred = make_traj(seed=7)
    for name, fn in LOSSES.items():
        base, _ = fn(normalize_c2w_to_first(pred), gt_rel, args)
        assert float(base) > 1e-3, f"{name}: degenerate fixture"
        for scale in SCALES:
            gen = torch.Generator().manual_seed(int(scale * 100))
            rot = random_rotation(gen)
            trans = torch.randn(3, generator=gen, dtype=DTYPE) * 0.5
            moved = normalize_c2w_to_first(apply_sim3(pred, scale, rot, trans))
            val, _ = fn(moved, gt_rel, args)
            rel = abs(float(val) - float(base)) / float(base)
            assert rel < 1e-10, f"{name} moved under g (s={scale}): relative {rel:.3e}"


# --------------------------------------------------------------------------- #
# the frame-0 asymmetry the pairwise loss is meant to remove
# --------------------------------------------------------------------------- #
def test_pairwise_reduces_frame0_leverage():
    """Perturbing frame 0 alone must hurt the pairwise loss far less.

    With frame-0 normalisation every one of the N-1 relative poses is corrupted;
    pairwise only loses the N-1 pairs that contain frame 0, i.e. 9 of 45 here.
    """
    args = make_args()
    gt = make_traj(seed=0)
    gt_rel = normalize_c2w_to_first(gt)
    n = gt.shape[-3]

    gen = torch.Generator().manual_seed(5)
    delta = torch.eye(4, dtype=DTYPE)
    delta[:3, :3] = random_rotation(gen)
    delta[:3, 3] = torch.randn(3, generator=gen, dtype=DTYPE) * 0.02

    frame0 = gt.clone()
    frame0[:, 0] = gt[:, 0] @ delta
    frame0_rel = normalize_c2w_to_first(frame0)

    si, _ = pose_scale_invariant_mse(frame0_rel, gt_rel, args)
    pw, _ = pose_pairwise_relative_mse(frame0_rel, gt_rel, args)

    n_pairs = n * (n - 1) // 2
    affected = (n - 1) / n_pairs  # 9/45 = 0.2
    assert float(pw) < float(si), (
        f"pairwise ({float(pw):.6f}) should be below frame-0-normalised ({float(si):.6f})")
    assert affected < 0.5


def test_pairwise_pair_count():
    args = make_args()
    gt_rel = normalize_c2w_to_first(make_traj(n_frames=10, seed=0))
    pred_rel = normalize_c2w_to_first(make_traj(n_frames=10, seed=3))
    _, terms = pose_pairwise_relative_mse(pred_rel, gt_rel, args)
    assert terms["pose_n_pairs"] == 45


# --------------------------------------------------------------------------- #
# the aligned-residual loss: is it really a differentiable ATE?
# --------------------------------------------------------------------------- #
def test_umeyama_matches_the_numpy_reference():
    """umeyama_sim3_torch must agree with mv_recon/utils.py::umeyama."""
    import numpy as np
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(12, 3, generator=gen, dtype=DTYPE)
    rot = random_rotation(gen)
    y = 2.5 * (x @ rot.T) + torch.tensor([0.3, -1.2, 0.7], dtype=DTYPE)
    y = y + torch.randn(12, 3, generator=gen, dtype=DTYPE) * 0.01

    c, r, t = umeyama_sim3_torch(x, y)

    # closed form from mv_recon/utils.py, which works on (3, n) arrays
    X, Y = x.numpy().T, y.numpy().T
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c_ref = np.trace(np.diag(D) @ S) / var_x
    R_ref = U @ S @ VH
    t_ref = mu_y - c_ref * R_ref @ mu_x

    assert abs(float(c) - c_ref) < 1e-10, f"scale {float(c)} vs {c_ref}"
    assert float((r - torch.from_numpy(R_ref)).abs().max()) < 1e-10
    assert float((t.reshape(3) - torch.from_numpy(t_ref).reshape(3)).abs().max()) < 1e-10


def test_aligned_residual_is_bounded_by_one():
    """The translation term is ATE / ate_ceiling, so it cannot exceed 1."""
    args = make_args(rot_w=0.0, trans_w=1.0)
    gt_rel = normalize_c2w_to_first(make_traj(seed=0))
    worst = 0.0
    for k in range(200):
        pred_rel = normalize_c2w_to_first(make_traj(seed=500 + k, span=0.03 * (1 + k % 7)))
        val, terms = pose_aligned_residual_mse(pred_rel, gt_rel, args)
        worst = max(worst, terms["pose_trans_mse"])
    assert worst <= 1.0 + 1e-9, f"translation term exceeded 1: {worst}"


def test_aligned_residual_reaches_one_when_prediction_is_collapsed():
    """A prediction with no trajectory information scores the ceiling."""
    args = make_args(rot_w=0.0, trans_w=1.0)
    gt_rel = normalize_c2w_to_first(make_traj(seed=0))
    collapsed = torch.eye(4, dtype=DTYPE).repeat(10, 1, 1).unsqueeze(0)
    collapsed[..., :3, 3] += torch.full((10, 3), 1e-9, dtype=DTYPE)
    val, terms = pose_aligned_residual_mse(normalize_c2w_to_first(collapsed), gt_rel, args)
    assert terms["pose_trans_mse"] > 0.99, f"collapsed prediction scored {terms['pose_trans_mse']}"


def test_detached_alignment_gradient_matches_full_autograd():
    """The envelope-theorem shortcut must reproduce the true gradient.

    pose_aligned_residual_mse solves the alignment under no_grad. That is only
    legitimate because (c, R, t) minimise the residual being differentiated, so
    their contribution to dL/d(pred) vanishes. Check it against autograd taken
    straight through the SVD.
    """
    args = make_args(rot_w=0.0, trans_w=1.0)
    gt_rel = normalize_c2w_to_first(make_traj(seed=0))

    for seed in (3, 17, 42):
        pred = make_traj(seed=seed).clone().requires_grad_(True)
        loss, _ = pose_aligned_residual_mse(normalize_c2w_to_first(pred), gt_rel, args)
        loss.backward()
        g_detached = pred.grad.clone()

        pred2 = make_traj(seed=seed).clone().requires_grad_(True)
        rel = normalize_c2w_to_first(pred2)
        p = rel[..., :3, 3]
        q = gt_rel[..., :3, 3]
        scale, rot, trans = umeyama_sim3_torch(p, q, 1e-6)          # gradients flow through SVD
        aligned = scale[..., None, None] * torch.matmul(p, rot.transpose(-1, -2)) + trans
        rmse = (aligned - q).pow(2).sum(-1).mean(-1).clamp_min(0.0).sqrt()
        qc = q - q.mean(dim=-2, keepdim=True)
        ceiling = qc.pow(2).sum(-1).mean(-1).clamp_min(1e-12).sqrt()
        (rmse / ceiling).mean().backward()
        g_full = pred2.grad.clone()

        denom = max(float(g_full.abs().max()), 1e-12)
        rel_err = float((g_detached - g_full).abs().max()) / denom
        assert rel_err < 1e-6, (
            f"seed={seed}: detached gradient differs from full autograd by "
            f"relative {rel_err:.3e}")


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
def test_degenerate_inputs_stay_finite():
    args = make_args()
    gt_rel = normalize_c2w_to_first(make_traj(seed=0))
    static = torch.eye(4, dtype=DTYPE).repeat(10, 1, 1).unsqueeze(0).requires_grad_(True)
    for name, fn in LOSSES.items():
        loss, terms = fn(normalize_c2w_to_first(static), gt_rel, args)
        assert torch.isfinite(loss), f"{name}: non-finite loss on a static trajectory"
        if static.grad is not None:
            static.grad = None
        loss.backward()
        assert torch.isfinite(static.grad).all(), f"{name}: non-finite gradient"


def test_gradients_are_finite_and_nonzero():
    args = make_args()
    gt_rel = normalize_c2w_to_first(make_traj(seed=0))
    for name, fn in LOSSES.items():
        pred = make_traj(seed=3).clone().requires_grad_(True)
        loss, _ = fn(normalize_c2w_to_first(pred), gt_rel, args)
        loss.backward()
        assert torch.isfinite(pred.grad).all(), f"{name}: non-finite gradient"
        assert float(pred.grad.norm()) > 0.0, f"{name}: zero gradient"


# --------------------------------------------------------------------------- #
def _main() -> int:
    args = make_args()
    gt = make_traj()
    gt_rel = normalize_c2w_to_first(gt)
    pred_rel = normalize_c2w_to_first(make_traj(seed=7))
    print("loss values on an unrelated prediction:")
    for name, fn in LOSSES.items():
        val, terms = fn(pred_rel, gt_rel, args)
        print(f"  {name:<10} total={float(val):.6f}  rot={terms['pose_rot_mse']:.6f}  "
              f"trans={terms['pose_trans_mse']:.6f}")

    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    print()
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
