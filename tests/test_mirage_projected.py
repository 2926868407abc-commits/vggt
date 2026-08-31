import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack_vggt_geometry_tum10 import (
    maximin_gradient_weights,
    mirage_projected_depth_score,
    mirage_projected_track_score,
)


def test_maximin_orthogonal_gradients_are_uniform():
    gradients = torch.eye(4, dtype=torch.float32)
    weights, gains = maximin_gradient_weights(gradients)
    np.testing.assert_allclose(weights, np.full(4, 0.25), atol=1e-8)
    np.testing.assert_allclose(gains, np.full(4, 0.25), atol=1e-8)


def test_maximin_five_orthogonal_objectives_are_uniform():
    gradients = torch.eye(5, dtype=torch.float32)
    weights, gains = maximin_gradient_weights(gradients)
    np.testing.assert_allclose(weights, np.full(5, 0.2), atol=1e-8)
    np.testing.assert_allclose(gains, np.full(5, 0.2), atol=1e-8)


def test_maximin_finds_positive_common_direction_with_track_conflict():
    gram = np.asarray([
        [1.000, 0.913, 0.220, -0.465],
        [0.913, 1.000, 0.247, -0.341],
        [0.220, 0.247, 1.000, -0.616],
        [-0.465, -0.341, -0.616, 1.000],
    ])
    gradients = torch.from_numpy(np.linalg.cholesky(gram + 1e-6 * np.eye(4))).float()
    _, gains = maximin_gradient_weights(gradients)
    assert gains.min() > 0.12


def test_projected_depth_score_ignores_global_scale():
    target = torch.linspace(0.5, 5.0, 200).reshape(10, 20)
    valid = torch.ones_like(target, dtype=torch.bool)
    score = mirage_projected_depth_score(2.0 * target, target, valid)
    assert float(score) < 1e-6


def test_projected_depth_score_sees_nonuniform_damage():
    target = torch.linspace(0.5, 5.0, 200).reshape(10, 20)
    pred = target.clone()
    pred[:, :10] *= 1.5
    valid = torch.ones_like(target, dtype=torch.bool)
    score = mirage_projected_depth_score(pred, target, valid)
    assert float(score) > 0.1


def test_track_score_caps_single_outlier():
    target = torch.zeros(3, 4, 2)
    visible = torch.ones(3, 4, dtype=torch.bool)
    known = torch.ones(3, 4, dtype=torch.bool)
    pred = target.clone()
    pred[1, 0, 0] = 10000.0
    score = mirage_projected_track_score(
        pred.unsqueeze(0), target, visible, known, cap_px=20.0
    )
    assert 0.0 < float(score) < 3.0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
