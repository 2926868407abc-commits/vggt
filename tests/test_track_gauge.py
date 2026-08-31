"""The fourth head is gauge invariant and must stay at its clean tracks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack_vggt_geometry_tum10 import (  # noqa: E402
    confidence_preservation_loss, dense_scalar_head, make_track_query_points,
    track_consistency_loss,
)


def project(points, c2w, K):
    w2c = np.linalg.inv(c2w)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    return np.stack([K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2],
                     K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]], axis=-1)


def test_per_frame_sim3_cancels_in_tracks():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(32, 3)); points[:, 2] += 5.0
    K = np.array([[500.0, 0, 256.0], [0, 500.0, 256.0], [0, 0, 1]])
    for i, scale in enumerate((0.5, 1.0, 2.0, 4.0)):
        c2w = np.eye(4); c2w[:3, 3] = [0.1 * i, -0.03 * i, 0.0]
        angle = 0.2 * i
        rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        trans = np.array([0.2 * i, -0.1, 0.05])
        transformed_points = scale * (points @ rot.T) + trans
        transformed_c2w = c2w.copy()
        transformed_c2w[:3, :3] = rot @ c2w[:3, :3]
        transformed_c2w[:3, 3] = scale * (rot @ c2w[:3, 3]) + trans
        assert np.allclose(project(points, c2w, K),
                           project(transformed_points, transformed_c2w, K), atol=1e-10)


def test_identical_tracks_have_zero_loss():
    track = torch.rand(1, 4, 6, 2) * 100
    vis = torch.rand(1, 4, 6)
    conf = torch.rand(1, 4, 6)
    loss, terms = track_consistency_loss(track, track, vis, vis, conf, conf,
                                         (100, 100), 0.2, 0.1, 0.1)
    assert float(loss) == 0.0
    assert all(float(v) == 0.0 for v in terms.values())


def test_visibility_cannot_hide_coordinate_error():
    target = torch.zeros(1, 3, 4, 2)
    pred = target.clone(); pred[:, 1:, :, 0] = 10.0
    target_vis = torch.ones(1, 3, 4)
    pred_vis = torch.zeros_like(target_vis)
    loss, terms = track_consistency_loss(pred, target, pred_vis, target_vis, None, None,
                                         (100, 100), 0.2, 0.0, 0.0)
    assert float(terms["track_coord"]) > 0.0
    assert float(loss) == float(terms["track_coord"])


def test_query_grid_is_deterministic_and_inside_margin():
    a = make_track_query_points((100, 200), 3, 4, 0.1, torch.device("cpu"))
    b = make_track_query_points((100, 200), 3, 4, 0.1, torch.device("cpu"))
    assert torch.equal(a, b) and a.shape == (12, 2)
    tol = 1e-4
    assert float(a[:, 0].min()) >= 19.9 - tol and float(a[:, 0].max()) <= 179.1 + tol
    assert float(a[:, 1].min()) >= 9.9 - tol and float(a[:, 1].max()) <= 89.1 + tol


def test_confidence_preservation_loss_is_zero_only_at_clean_target():
    target = torch.tensor([[[1.0, 2.0], [4.0, 8.0]]])
    same = target.clone().requires_grad_(True)
    changed = (target * 0.5).requires_grad_(True)
    assert float(confidence_preservation_loss(same, target)) == 0.0
    loss = confidence_preservation_loss(changed, target)
    assert float(loss) > 0.05
    loss.backward()
    assert torch.isfinite(changed.grad).all()


def test_dense_scalar_head_handles_depth_and_confidence_layouts():
    assert dense_scalar_head(torch.zeros(1, 10, 4, 5, 1)).shape == (10, 4, 5)
    assert dense_scalar_head(torch.zeros(1, 10, 4, 5)).shape == (10, 4, 5)
    assert dense_scalar_head(torch.zeros(10, 4, 5)).shape == (10, 4, 5)


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
