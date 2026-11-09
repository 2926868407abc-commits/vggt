import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack_vggt_geometry_tum10 import (
    cap_mirage_projected_track_queries,
    draw_dynamic_frame_indices,
    maximin_gradient_weights,
    mirage_projected_depth_score,
    mirage_projected_task_scores,
    mirage_projected_track_score,
    normalized_objective_gain,
    requested_mirage_projected_tasks,
)


def test_track_query_cap_slices_every_query_axis_consistently():
    raw = {
        "query_points": np.arange(20, dtype=np.float32).reshape(10, 2),
        "tracks": np.arange(3 * 10 * 2, dtype=np.float32).reshape(3, 10, 2),
        "track_known": np.ones((3, 10), dtype=bool),
        "track_visible": np.ones((3, 10), dtype=bool),
        "depth": np.zeros((3, 2, 2), dtype=np.float32),
    }
    capped = cap_mirage_projected_track_queries(raw, 4)
    expected = np.asarray([0, 3, 6, 9])
    np.testing.assert_array_equal(capped["query_points"], raw["query_points"][expected])
    np.testing.assert_array_equal(capped["tracks"], raw["tracks"][:, expected])
    np.testing.assert_array_equal(capped["track_known"], raw["track_known"][:, expected])
    np.testing.assert_array_equal(capped["track_visible"], raw["track_visible"][:, expected])
    assert capped["depth"] is raw["depth"]
    assert int(capped["query_count_uncapped"]) == 10
    assert int(capped["query_count"]) == 4


def test_track_query_cap_zero_preserves_input_object():
    raw = {
        "query_points": np.zeros((5, 2), dtype=np.float32),
        "tracks": np.zeros((2, 5, 2), dtype=np.float32),
        "track_known": np.ones((2, 5), dtype=bool),
        "track_visible": np.ones((2, 5), dtype=bool),
    }
    assert cap_mirage_projected_track_queries(raw, 0) is raw


class _ChoiceSequence:
    """Small deterministic stand-in for numpy's Generator.choice."""

    def __init__(self, values):
        self.values = iter(values)

    def choice(self, *_args, **_kwargs):
        return next(self.values)


def test_dynamic_earliest_query_preserves_historical_sorted_behavior():
    rng = _ChoiceSequence([np.asarray([7, 1, 9, 3])])
    assert draw_dynamic_frame_indices(
        list(range(10)), 4, rng, "earliest"
    ) == [1, 3, 7, 9]


def test_dynamic_uniform_query_stays_at_input_zero_and_targets_are_sorted():
    rng = _ChoiceSequence([7, np.asarray([1, 9, 3])])
    assert draw_dynamic_frame_indices(
        list(range(10)), 4, rng, "uniform_query"
    ) == [7, 1, 3, 9]


def test_dynamic_uniform_query_does_not_duplicate_query():
    class _CheckingChoice:
        def __init__(self):
            self.calls = 0

        def choice(self, values, size=None, replace=False):
            self.calls += 1
            if self.calls == 1:
                return 5
            assert 5 not in values
            assert size == 2
            assert replace is False
            return np.asarray([2, 8])

    assert draw_dynamic_frame_indices(
        list(range(10)), 3, _CheckingChoice(), "uniform_query"
    ) == [5, 2, 8]


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


def test_adaptive_audit_expected_tasks_follow_explicit_subset():
    args = SimpleNamespace(
        attack_loss="mirage_projected_split_pose",
        mirage_projected_tasks="track",
    )
    assert requested_mirage_projected_tasks(args) == ("track",)


def test_default_split_pose_audit_still_requires_all_five_tasks():
    args = SimpleNamespace(
        attack_loss="mirage_projected_split_pose",
        mirage_projected_tasks="",
    )
    assert requested_mirage_projected_tasks(args) == (
        "pose_trans", "pose_rot", "depth", "point", "track"
    )


def test_track_only_scores_do_not_require_other_head_outputs():
    args = SimpleNamespace(
        attack_loss="mirage_projected_split_pose",
        mirage_projected_tasks="track",
        mirage_projected_track_cap_px=20.0,
        mirage_projected_track_mode="eval_epe",
        mirage_projected_track_min_points_per_frame=1,
        mirage_projected_track_target_softmin_tau_px=0.25,
    )
    target = torch.zeros(3, 4, 2)
    pred = torch.ones(1, 3, 4, 2)
    valid = torch.ones(3, 4, dtype=torch.bool)
    item = {
        "mirage_projected_gt": {
            "tracks": target,
            "track_visible": valid,
            "track_known": valid,
        }
    }

    scores = mirage_projected_task_scores({"track": pred}, item, args)

    assert tuple(scores) == ("track",)
    assert torch.allclose(scores["track"], torch.tensor(2.0).sqrt())


def test_normalized_objective_gain_for_maximized_damage():
    assert normalized_objective_gain(2.0, 3.0) == 0.5


def test_normalized_objective_gain_for_minimized_loss_damage():
    # A minimized loss changing 4 -> 3 is represented as damage -4 -> -3.
    assert normalized_objective_gain(-4.0, -3.0) == 0.25


def test_normalized_objective_gain_preserves_failure_sign():
    assert normalized_objective_gain(2.0, 1.5) == -0.25


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


def test_track_score_returns_none_when_dynamic_batch_has_no_visible_gt():
    target = torch.zeros(3, 4, 2)
    visible = torch.zeros(3, 4, dtype=torch.bool)
    known = torch.ones(3, 4, dtype=torch.bool)
    pred = target.clone()
    score = mirage_projected_track_score(
        pred.unsqueeze(0), target, visible, known, cap_px=20.0
    )
    assert score is None


def test_eval_aligned_track_score_drops_frames_below_minimum_support():
    target = torch.zeros(3, 4, 2)
    pred = target.clone()
    # Frame 1 has one catastrophic point but is below the formal evaluation's
    # minimum support and therefore must not affect the result.
    pred[1, 0, 0] = 1000.0
    # Frame 2 has three usable points, each with exactly 2 px EPE.
    pred[2, :3, 0] = 2.0
    known = torch.ones(3, 4, dtype=torch.bool)
    visible = torch.zeros(3, 4, dtype=torch.bool)
    visible[1, 0] = True
    visible[2, :3] = True

    score = mirage_projected_track_score(
        pred.unsqueeze(0), target, visible, known, cap_px=20.0,
        mode="eval_epe", min_points_per_frame=2,
    )
    torch.testing.assert_close(score, torch.tensor(2.0))


def test_eval_aligned_track_score_returns_none_without_usable_target_frame():
    target = torch.zeros(3, 4, 2)
    pred = target.clone()
    known = torch.ones(3, 4, dtype=torch.bool)
    visible = torch.zeros(3, 4, dtype=torch.bool)
    visible[1:, 0] = True

    score = mirage_projected_track_score(
        pred, target, visible, known, cap_px=20.0,
        mode="eval_epe", min_points_per_frame=2,
    )
    assert score is None


def test_eval_aligned_track_score_has_uncapped_metric_scale():
    target = torch.zeros(3, 4, 2)
    pred = target.clone()
    pred[1:, :, 0] = 100.0
    known = torch.ones(3, 4, dtype=torch.bool)
    visible = torch.ones(3, 4, dtype=torch.bool)

    score = mirage_projected_track_score(
        pred, target, visible, known, cap_px=20.0,
        mode="eval_epe", min_points_per_frame=2,
    )
    torch.testing.assert_close(score, torch.tensor(100.0))


def test_track_target_softmin_preserves_equal_frame_epe():
    target = torch.zeros(4, 4, 2)
    pred = target.clone()
    pred[1:, :, 0] = 3.0
    known = torch.ones(4, 4, dtype=torch.bool)
    visible = torch.ones(4, 4, dtype=torch.bool)

    score = mirage_projected_track_score(
        pred, target, visible, known, cap_px=20.0,
        mode="eval_epe_softmin", min_points_per_frame=2,
        target_softmin_tau_px=0.25,
    )
    torch.testing.assert_close(score, torch.tensor(3.0))


def test_track_target_softmin_focuses_on_least_damaged_frame():
    target = torch.zeros(4, 4, 2)
    pred = target.clone()
    pred[1, :, 0] = 0.5
    pred[2, :, 0] = 4.0
    pred[3, :, 0] = 8.0
    known = torch.ones(4, 4, dtype=torch.bool)
    visible = torch.ones(4, 4, dtype=torch.bool)

    mean_score = mirage_projected_track_score(
        pred, target, visible, known, cap_px=20.0,
        mode="eval_epe", min_points_per_frame=2,
    )
    robust_score = mirage_projected_track_score(
        pred, target, visible, known, cap_px=20.0,
        mode="eval_epe_softmin", min_points_per_frame=2,
        target_softmin_tau_px=0.10,
    )
    assert 0.5 <= float(robust_score) < 0.7
    assert float(robust_score) < float(mean_score)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
