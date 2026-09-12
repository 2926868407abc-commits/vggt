from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_paper_baseline_placements import plane_quad, quad_dimensions  # noqa: E402


def _plane() -> dict:
    return {
        "centre": [0.0, 0.0, 0.0],
        "u": [1.0, 0.0, 0.0],
        "v": [0.0, 1.0, 0.0],
        "uv_min": [-1.0, -0.5],
        "uv_max": [1.0, 0.5],
    }


def test_plane_quad_full_extent_and_centre_scale() -> None:
    full = plane_quad(_plane())
    small = plane_quad(_plane(), scale_u=0.2, scale_v=0.2)
    assert np.allclose(full.mean(axis=0), small.mean(axis=0))
    assert np.allclose(quad_dimensions(full), (2.0, 1.0, 2.0))
    assert np.allclose(quad_dimensions(small), (0.4, 0.2, 0.08))
