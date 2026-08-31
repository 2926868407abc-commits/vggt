"""CPU tests for the budgeted four-head constraint machinery."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import attack_vggt_geometry_tum10 as A  # noqa: E402


def test_even_median_matches_numpy_convention():
    x = torch.tensor([1.0, 4.0, 2.0, 3.0])
    assert float(A.differentiable_median(x)) == 2.5


def test_budget_hinge_is_zero_inside_and_normalized_outside():
    high = {"clean": 1.0, "threshold": 3.0, "worse": "high"}
    safe, violation = A.budget_safe_value_and_violation(torch.tensor(2.0), high, 0.5)
    assert float(safe) == 2.0 and float(violation) == 0.0
    _, violation = A.budget_safe_value_and_violation(torch.tensor(3.0), high, 0.5)
    assert abs(float(violation) - 0.5) < 1e-7

    low = {"clean": 5.0, "threshold": 3.0, "worse": "low"}
    safe, violation = A.budget_safe_value_and_violation(torch.tensor(4.0), low, 0.5)
    assert float(safe) == 4.0 and float(violation) == 0.0
    _, violation = A.budget_safe_value_and_violation(torch.tensor(3.0), low, 0.5)
    assert abs(float(violation) - 0.5) < 1e-7


def test_static_identical_depth_has_zero_reprojection_error():
    depth = torch.full((1, 3, 6, 8), 2.0, requires_grad=True)
    ext = torch.zeros(1, 3, 3, 4)
    ext[..., 0, 0] = ext[..., 1, 1] = ext[..., 2, 2] = 1.0
    k = torch.zeros(1, 3, 3, 3)
    k[..., 0, 0] = k[..., 1, 1] = 4.0
    k[..., 0, 2] = 3.5
    k[..., 1, 2] = 2.5
    k[..., 2, 2] = 1.0
    metric = A.differentiable_reprojection_metric(depth, ext, k, stride=1)
    assert float(metric) < 1e-6
    metric.backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()


def test_torch_unprojection_has_expected_shape_and_depth():
    depth = torch.full((1, 2, 4, 5), 3.0)
    ext = torch.zeros(1, 2, 3, 4)
    ext[..., 0, 0] = ext[..., 1, 1] = ext[..., 2, 2] = 1.0
    k = torch.eye(3).view(1, 1, 3, 3).expand(1, 2, -1, -1).clone()
    world = A.torch_unproject_depth_world(depth, ext, k)
    assert world.shape == (1, 2, 4, 5, 3)
    assert torch.allclose(world[..., 2], depth)


def test_dual_update_only_uses_reported_violations():
    args = argparse.Namespace(
        attack_loss="geometry_joint_gauge_budgeted", budget_dual_init=0.0,
        budget_dual_lr=2.0, budget_dual_max=10.0,
        _budget_duals={"conf_std": 1.0, "track": 0.0},
    )
    A.update_budget_duals(args, {
        "budget_violation_conf_std": [0.25, 0.75],
        "budget_metric_conf_std": [123.0],
    })
    assert args._budget_duals["conf_std"] == 2.0
    assert args._budget_duals["track"] == 0.0


def test_patch_optimizer_cannot_contain_model_parameters():
    model = torch.nn.Linear(2, 2)
    texture = torch.zeros(1, requires_grad=True)
    good = torch.optim.AdamW([texture], lr=1e-3)
    A.assert_texture_only_optimizer(good, texture, model)
    bad = torch.optim.AdamW([texture, *model.parameters()], lr=1e-3)
    try:
        A.assert_texture_only_optimizer(bad, texture, model)
    except RuntimeError:
        pass
    else:
        raise AssertionError("model parameter in patch optimiser was not rejected")


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test(); print(f"PASS  {test.__name__}")
        except Exception as exc:
            failed += 1; print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(_main())
