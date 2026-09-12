from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tum_gt_pool import (  # noqa: E402
    BASE_ARRAY_FILES,
    POOL_ARRAY_FILES,
    gather_pool,
    open_pool,
    query_count,
)


def make_pool(tmp_path: Path):
    scene = "scene"
    out = tmp_path / scene
    out.mkdir()
    frames = [10, 20, 30, 40]
    n, h, w, p = 4, 2, 3, 3
    arrays = {
        "gt_c2w": np.tile(np.eye(4), (n, 1, 1)),
        "depth": np.arange(n * h * w, dtype=np.float32).reshape(n, h, w),
        "point_map": np.arange(n * h * w * 3, dtype=np.float32).reshape(n, h, w, 3),
        "point_valid": np.ones((n, h, w), dtype=bool),
        "query_points": np.tile(
            np.array([[0, 0], [1, 1], [2, 1]], dtype=np.float32), (n, 1, 1)
        ),
        "query_valid": np.array(
            [[True, True, True], [True, False, True], [False, True, False],
             [True, True, True]], dtype=bool
        ),
        "tracks": np.arange(n * n * p * 2, dtype=np.float32).reshape(n, n, p, 2),
        "track_known": np.ones((n, n, p), dtype=bool),
        "track_visible": np.ones((n, n, p), dtype=bool),
    }
    for name, array in arrays.items():
        np.save(out / POOL_ARRAY_FILES[name], array, allow_pickle=False)
    meta = {
        "format_version": 1,
        "scene": scene,
        "frame_indices": frames,
        "image_names": [f"{frame}.png" for frame in frames],
        "tensor_hw": [h, w],
        "array_files": dict(POOL_ARRAY_FILES),
    }
    (out / "pool_meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    return open_pool(tmp_path, scene), arrays


def make_lazy_pool(tmp_path: Path):
    scene = "lazy_scene"
    out = tmp_path / scene
    out.mkdir()
    frames = [10, 20]
    n, h, w, p = 2, 4, 4, 2
    query = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    arrays = {
        "gt_c2w": np.tile(np.eye(4, dtype=np.float64), (n, 1, 1)),
        "depth": np.full((n, h, w), 2.0, dtype=np.float32),
        "point_map": np.zeros((n, h, w, 3), dtype=np.float32),
        "point_valid": np.ones((n, h, w), dtype=bool),
        "query_points": np.tile(query, (n, 1, 1)),
        "query_valid": np.ones((n, p), dtype=bool),
    }
    for name, array in arrays.items():
        np.save(out / BASE_ARRAY_FILES[name], array, allow_pickle=False)
    meta = {
        "format_version": 2,
        "scene": scene,
        "track_storage": "lazy_reprojection",
        "frame_indices": frames,
        "image_names": [f"{frame}.png" for frame in frames],
        "tensor_hw": [h, w],
        "array_files": dict(BASE_ARRAY_FILES),
        "projection": {"scale_x": 1.0, "scale_y": 1.0, "crop_y": 0.0},
        "camera_intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        "occl_tol": 0.05,
    }
    (out / "pool_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return open_pool(tmp_path, scene), arrays, query


class TumGtPoolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pool, self.arrays = make_pool(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_gather_uses_first_frame_as_query(self):
        result = gather_pool(
            self.pool,
            [20, 40],
            (2, 3),
            image_names=["20.png", "40.png"],
            gt_c2w=self.arrays["gt_c2w"][[1, 3]],
            min_queries=2,
        )
        keep = self.arrays["query_valid"][1]
        np.testing.assert_array_equal(
            result["query_points"], self.arrays["query_points"][1, keep]
        )
        np.testing.assert_array_equal(
            result["tracks"], self.arrays["tracks"][1, [1, 3]][:, keep]
        )
        self.assertEqual(result["tracks"].shape, (2, 2, 2))
        self.assertEqual(result["depth"].shape, (2, 2, 3))

    def test_query_count_and_minimum(self):
        self.assertEqual(query_count(self.pool, 30), 1)
        with self.assertRaisesRegex(ValueError, "only 1 valid GT points"):
            gather_pool(self.pool, [30, 40], (2, 3), min_queries=2)

    def test_grid_and_identity_guards(self):
        with self.assertRaisesRegex(ValueError, "grid"):
            gather_pool(self.pool, [10, 20], (3, 3))
        with self.assertRaisesRegex(ValueError, "image identities"):
            gather_pool(
                self.pool, [10, 20], (2, 3), image_names=["bad.png", "20.png"]
            )
        with self.assertRaisesRegex(KeyError, "absent"):
            gather_pool(self.pool, [10, 99], (2, 3))

    def test_lazy_tracks_are_generated_for_only_the_requested_frames(self):
        pool, arrays, query = make_lazy_pool(self.root)
        self.assertEqual(pool["track_storage"], "lazy_reprojection")
        self.assertNotIn("tracks", pool["arrays"])
        result = gather_pool(
            pool,
            [10, 20],
            (4, 4),
            image_names=["10.png", "20.png"],
            gt_c2w=arrays["gt_c2w"],
            min_queries=2,
        )
        np.testing.assert_allclose(result["tracks"][0], query, atol=1e-6)
        np.testing.assert_allclose(result["tracks"][1], query, atol=1e-6)
        self.assertTrue(np.all(result["track_known"]))
        self.assertTrue(np.all(result["track_visible"]))


if __name__ == "__main__":
    unittest.main()
