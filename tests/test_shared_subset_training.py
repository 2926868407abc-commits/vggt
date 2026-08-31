"""CPU tests for shared deformation scaling and balanced subset sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack_vggt_geometry_tum10 import (  # noqa: E402
    draw_subset_plan_entries,
    scale_shared_deformation,
    sim3_action_basis,
)
from scripts.solve_shared_deform import (  # noqa: E402
    build_subset_projectors,
    subset_survival_scores,
)


def sample_positions(n: int = 20) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.stack([t, 0.2 * t * t, 0.1 * np.sin(2 * np.pi * t)], axis=1)


def test_shared_deformation_uses_orthogonal_mode_scale() -> None:
    positions = sample_positions()
    rng = np.random.default_rng(4)
    delta_unit = rng.normal(size=positions.shape)
    delta_unit /= np.linalg.norm(delta_unit)
    magnitude = 0.3

    scaled = scale_shared_deformation(delta_unit, positions, magnitude)
    radius = np.sqrt(((positions - positions.mean(0)) ** 2).sum(1).mean())
    row_rms = np.sqrt((scaled * scaled).sum(1).mean())
    np.testing.assert_allclose(row_rms, magnitude * radius, rtol=1e-12, atol=1e-12)


def test_shared_deformation_rejects_wrong_shape_and_normalisation() -> None:
    positions = sample_positions()
    unit = np.zeros_like(positions)
    unit[0, 0] = 1.0

    try:
        scale_shared_deformation(unit[:-1], positions, 0.3)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("shape mismatch should fail")

    try:
        scale_shared_deformation(unit * 2.0, positions, 0.3)
    except ValueError as exc:
        assert "unit Frobenius norm" in str(exc)
    else:
        raise AssertionError("non-unit deformation should fail")


def test_overlapping_subsets_get_identical_global_frame_targets() -> None:
    positions = sample_positions()
    delta_unit = np.arange(positions.size, dtype=np.float64).reshape(positions.shape)
    delta_unit /= np.linalg.norm(delta_unit)
    scaled = scale_shared_deformation(delta_unit, positions, 0.3)

    first = np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    second = np.asarray([1, 2, 5, 6, 9, 10, 13, 14, 17, 18])
    first_target = scaled[first]
    second_target = scaled[second]
    overlap = sorted(set(first.tolist()) & set(second.tolist()))
    for frame in overlap:
        first_row = first_target[np.flatnonzero(first == frame)[0]]
        second_row = second_target[np.flatnonzero(second == frame)[0]]
        np.testing.assert_array_equal(first_row, second_row)


def test_subset_sampler_covers_one_round_without_replacement() -> None:
    plan = [{"name": f"sub{i:02d}"} for i in range(12)]
    rng = np.random.default_rng(9)
    queue: list[int] = []
    names: list[str] = []
    for _ in range(3):
        picks, queue = draw_subset_plan_entries(plan, queue, 4, rng)
        names.extend(entry["name"] for entry in picks)

    assert len(names) == 12
    assert len(set(names)) == 12
    assert queue == []


def test_sim3_projector_removes_gauge_and_preserves_orthogonal_field() -> None:
    positions = sample_positions(10)
    subsets = {"all": list(range(10))}
    projectors, indices = build_subset_projectors(positions, subsets)

    gauge = sim3_action_basis(positions)[0].reshape(10, 3)
    gauge_score = subset_survival_scores(
        torch.tensor(gauge, dtype=torch.float64),
        projectors,
        indices,
    )
    assert float(gauge_score[0]) < 1e-20

    seed = np.zeros((10, 3), dtype=np.float64)
    seed[:, 0] = np.cos(4 * np.pi * np.arange(10) / 9)
    flat = seed.reshape(-1)
    residual = projectors["all"].numpy() @ flat
    orthogonal = torch.tensor(residual.reshape(10, 3), dtype=torch.float64)
    orthogonal_score = subset_survival_scores(
        orthogonal,
        projectors,
        indices,
    )
    assert float(orthogonal_score[0]) > 1.0 - 1e-12


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"{len(tests)}/{len(tests)} shared/subset tests passed")
