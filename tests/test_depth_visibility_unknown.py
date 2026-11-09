import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack_vggt_geometry_tum10 import build_geometry_arrays_for_plane


def _args(scene_depth: np.ndarray):
    return SimpleNamespace(
        _active_depth_maps=[scene_depth],
        _intrinsics_in_tensor_space=True,
        use_depth_visibility=True,
        depth_min_m=0.1,
        depth_max_m=10.0,
        visibility_depth_margin=0.05,
        surface_support_check=False,
    )


def _render(scene_depth: np.ndarray) -> np.ndarray:
    # Front-facing 2x2 m square at z=2, projected into a 5x5 image.
    quad = np.asarray([
        [-1.0, -1.0, 2.0],
        [1.0, -1.0, 2.0],
        [1.0, 1.0, 2.0],
        [-1.0, 1.0, 2.0],
    ])
    c2w = np.eye(4, dtype=np.float64)[None]
    intrinsics = np.asarray([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]])
    _, masks, _ = build_geometry_arrays_for_plane(
        quad, c2w, ["unused.png"], (5, 5), 5, intrinsics, _args(scene_depth)
    )
    return masks[0, 0]


def test_unknown_scene_depth_does_not_punch_a_hole():
    scene = np.full((5, 5), 2.0, dtype=np.float32)
    scene[2, 2] = 0.0
    mask = _render(scene)
    assert mask[2, 2] == 1.0


def test_valid_nearer_depth_still_occludes():
    scene = np.full((5, 5), 2.0, dtype=np.float32)
    scene[2, 2] = 1.0
    mask = _render(scene)
    assert mask[2, 2] == 0.0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
