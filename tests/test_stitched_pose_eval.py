from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "eval_stitched_pose.py"
SPEC = importlib.util.spec_from_file_location("eval_stitched_pose", MODULE_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def z_rotation(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def trajectory(n: int = 84) -> np.ndarray:
    poses = np.tile(np.eye(4), (n, 1, 1))
    t = np.linspace(0.0, 1.0, n)
    poses[:, 0, 3] = t
    poses[:, 1, 3] = 0.2 * np.sin(2.0 * np.pi * t)
    poses[:, 2, 3] = 0.1 * t * t
    for idx, value in enumerate(t):
        poses[idx, :3, :3] = z_rotation(15.0 * value)
    return poses


def test_windows_cover_once_after_stitch():
    frames = list(range(84))
    windows = MOD.make_windows(frames, window=30, stride=20)
    assert windows[0] == list(range(30))
    assert windows[1] == list(range(20, 50))
    assert len(set(windows[0]) & set(windows[1])) == 10
    assert windows[-1] == list(range(54, 84))
    assert all(len(set(left) & set(right)) >= 10
               for left, right in zip(windows, windows[1:]))
    assert sorted(set(sum(windows, []))) == frames


def test_frozen_irregular_plan_preserves_heldout_order(tmp_path):
    frames = list(range(1, 170, 2))
    windows = MOD.make_windows(frames, window=30, stride=20)
    plan = [
        {
            "name": f"odd_w{index:03d}",
            "scene": "rgbd_dataset_freiburg3_sitting_static",
            "frames": window,
            "contains_training_frames": False,
        }
        for index, window in enumerate(windows)
    ]
    path = tmp_path / "odd_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    selected, loaded = MOD.load_planned_windows(
        path, "rgbd_dataset_freiburg3_sitting_static")
    assert selected == frames
    assert loaded == windows
    assert all(frame % 2 == 1 for frame in selected)


def test_random_window_gauges_are_removed_without_gt():
    rng = np.random.default_rng(7)
    truth = trajectory()
    records = []
    for frames in MOD.make_windows(range(len(truth)), window=30, stride=20):
        scale = float(rng.uniform(0.5, 2.0))
        rotation = z_rotation(float(rng.uniform(-40.0, 40.0)))
        translation = rng.normal(0.0, 1.0, 3)
        local = MOD.apply_sim3_to_c2w(truth[frames], scale, rotation, translation)
        records.append({"frames": frames, "pred": local})
    frames, stitched, diagnostics = MOD.stitch_windows(records, "pred")
    # The first window defines the arbitrary global gauge.  Compare against GT
    # after fitting that one known synthetic gauge, never during stitching.
    _, _, _, fit = MOD.estimate_prediction_sim3(stitched, truth[frames])
    scale, rot, trans, _ = MOD.estimate_prediction_sim3(stitched, truth[frames])
    recovered = MOD.apply_sim3_to_c2w(stitched, scale, rot, trans)
    assert np.max(np.abs(recovered[:, :3, 3] - truth[frames, :3, 3])) < 1e-8
    assert max(row.get("position_residual_rmse", 0.0) for row in diagnostics) < 1e-8
    assert fit["position_residual_rmse"] < 1e-8


def test_non_sim3_deformation_survives_stitching():
    truth = trajectory()
    records = []
    windows = MOD.make_windows(range(len(truth)), window=30, stride=20)
    for window_idx, frames in enumerate(windows):
        local = truth[frames].copy()
        if window_idx == 2:
            local[10:, 1, 3] += np.linspace(0.05, 0.25, 20)
        records.append({"frames": frames, "pred": local})
    frames, stitched, _ = MOD.stitch_windows(records, "pred")
    error = np.linalg.norm(stitched[:, :3, 3] - truth[frames, :3, 3], axis=1)
    assert error.max() > 0.20


if __name__ == "__main__":
    test_windows_cover_once_after_stitch()
    with tempfile.TemporaryDirectory() as directory:
        test_frozen_irregular_plan_preserves_heldout_order(Path(directory))
    test_random_window_gauges_are_removed_without_gt()
    test_non_sim3_deformation_survives_stitching()
    print("4 standalone stitched-pose tests passed")
