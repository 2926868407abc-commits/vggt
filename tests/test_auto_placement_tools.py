"""CPU tests for reusable auto-placement selection and scoring helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pick_probe_positions import pick_spread_positions  # noqa: E402
from score_probe import average_ranks, spearman  # noqa: E402
from score_scan import trajectory_ceiling, w2c_to_c2w  # noqa: E402


def test_trajectory_ceiling_is_rms_radius() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(trajectory_ceiling(positions), 1.0)


def test_w2c_to_c2w_inverts_rigid_transform() -> None:
    angle = np.radians(25.0)
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = np.asarray([0.4, -0.2, 1.3])
    extrinsic = np.concatenate([rotation, translation[:, None]], axis=1)[None]
    c2w = w2c_to_c2w(extrinsic)[0]
    homogeneous = np.eye(4)
    homogeneous[:3] = extrinsic[0]
    np.testing.assert_allclose(c2w @ homogeneous, np.eye(4), atol=1e-12)


def test_average_ranks_handles_ties() -> None:
    ranks = average_ranks(np.asarray([10.0, 20.0, 20.0, 40.0]))
    np.testing.assert_array_equal(ranks, np.asarray([0.0, 1.5, 1.5, 3.0]))
    np.testing.assert_allclose(spearman(np.arange(5.0), np.arange(5.0)), 1.0)
    np.testing.assert_allclose(spearman(np.arange(5.0), np.arange(4.0, -1.0, -1.0)), -1.0)


def test_probe_picker_represents_multiple_planes() -> None:
    candidates = []
    for plane in (0, 1):
        for iu in range(3):
            for iv in range(3):
                candidates.append({
                    "plane": plane,
                    "iu": iu,
                    "iv": iv,
                    "cu": float(iu),
                    "cv": float(iv),
                    "fill": 1.0,
                    "quad": [0.0] * 12,
                })
    selected = pick_spread_positions(candidates, max_positions=8, max_per_plane=4)
    counts = {plane: sum(int(item["plane"]) == plane for item in selected)
              for plane in (0, 1)}
    assert len(selected) == 8
    assert counts == {0: 4, 1: 4}
    assert len({item["tag"] for item in selected}) == 8


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"{len(tests)}/{len(tests)} auto-placement tests passed")
