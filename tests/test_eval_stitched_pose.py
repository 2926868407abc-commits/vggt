from __future__ import annotations

import importlib.util
from pathlib import Path
import json
import tempfile

import numpy as np


SCRIPT = Path(__file__).with_name("eval_stitched_pose.py")
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_stitched_pose.py"
SPEC = importlib.util.spec_from_file_location("eval_stitched_pose", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def z_rotation(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_poses(points: np.ndarray) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(points), axis=0)
    poses[:, :3, 3] = points
    return poses


def test_apply_sim3_includes_translation() -> None:
    poses = make_poses(np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
    rotation = z_rotation(30.0)
    translation = np.asarray([4.0, -2.0, 0.5])
    mapped = MODULE.apply_sim3_to_c2w(poses, 1.7, rotation, translation)
    expected = 1.7 * (rotation @ poses[:, :3, 3].T).T + translation
    np.testing.assert_allclose(mapped[:, :3, 3], expected, atol=1e-12)


def test_estimated_sim3_maps_overlap_exactly() -> None:
    local = make_poses(
        np.asarray(
            [[0.0, 0.0, 0.0], [0.5, -0.2, 0.1], [1.1, 0.3, -0.4], [1.8, 0.7, 0.2]]
        )
    )
    rotation = z_rotation(-17.0)
    translation = np.asarray([3.0, 1.5, -0.7])
    global_ = MODULE.apply_sim3_to_c2w(local, 2.3, rotation, translation)
    scale, recovered_rotation, recovered_translation, diagnostic = (
        MODULE.estimate_prediction_sim3(local, global_)
    )
    recovered = MODULE.apply_sim3_to_c2w(
        local, scale, recovered_rotation, recovered_translation
    )
    np.testing.assert_allclose(recovered, global_, atol=1e-10)
    assert diagnostic["position_residual_rmse"] < 1e-10


def test_two_window_stitch_recovers_global_trajectory() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.3, 0.1, 0.0], [0.7, 0.2, 0.1],
         [1.2, 0.4, 0.1], [1.8, 0.7, 0.2], [2.5, 1.1, 0.3]]
    )
    global_poses = make_poses(points)
    local_second = MODULE.apply_sim3_to_c2w(
        global_poses[2:], 0.4, z_rotation(23.0), np.asarray([-2.0, 0.8, 1.1])
    )
    records = [
        {"frames": [0, 1, 2, 3], "clean_c2w": global_poses[:4]},
        {"frames": [2, 3, 4, 5], "clean_c2w": local_second},
    ]
    frames, stitched, diagnostics = MODULE.stitch_windows(records, "clean_c2w")
    np.testing.assert_array_equal(frames, np.arange(6))
    np.testing.assert_allclose(stitched, global_poses, atol=1e-10)
    assert diagnostics[1]["position_residual_rmse"] < 1e-10


def test_make_windows_cover_tail_with_overlap() -> None:
    windows = MODULE.make_windows(range(23), window=10, stride=5)
    assert windows == [
        list(range(0, 10)),
        list(range(5, 15)),
        list(range(10, 20)),
        list(range(13, 23)),
    ]
    assert set().union(*(set(window) for window in windows)) == set(range(23))
    assert all(set(a) & set(b) for a, b in zip(windows, windows[1:]))


def test_planned_windows_reject_training_leak_flag(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps([
            {
                "name": "bad",
                "scene": "scene",
                "frames": list(range(10)),
                "contains_training_frames": True,
            }
        ]),
        encoding="utf-8",
    )
    try:
        MODULE.load_planned_windows(plan, "scene")
    except ValueError as exc:
        assert "contains training frames" in str(exc)
    else:
        raise AssertionError("training leakage flag was not rejected")


def test_planned_windows_allow_explicit_training_diagnostic(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps([{
            "name": "training_diagnostic",
            "scene": "scene",
            "frames": list(range(10)),
            "contains_training_frames": True,
        }]),
        encoding="utf-8",
    )
    selected, windows = MODULE.load_planned_windows(
        plan, "scene", allow_training_frames=True
    )
    assert selected == list(range(10))
    assert windows == [list(range(10))]


if __name__ == "__main__":
    test_apply_sim3_includes_translation()
    test_estimated_sim3_maps_overlap_exactly()
    test_two_window_stitch_recovers_global_trajectory()
    test_make_windows_cover_tail_with_overlap()
    with tempfile.TemporaryDirectory() as tmp:
        test_planned_windows_reject_training_leak_flag(Path(tmp))
        test_planned_windows_allow_explicit_training_diagnostic(Path(tmp))
    print("6 non-pytest tests passed")
