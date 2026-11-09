import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack_vggt_geometry_tum10_vggt_loss_debug import (
    normalize_vggt_training_gt,
    vggt_tracking_training_loss,
)


def test_vggt_gt_normalization_centers_first_camera_and_sets_unit_scale():
    c2w = torch.eye(4).repeat(1, 2, 1, 1)
    c2w[:, 0, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
    c2w[:, 1, :3, 3] = torch.tensor([2.0, 0.0, 0.0])
    points = torch.tensor(
        [[[[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]],
          [[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]]]
    )
    valid = torch.ones((1, 2, 1, 2), dtype=torch.bool)
    depths = torch.ones((1, 2, 1, 2)) * 2.0

    extrinsics, normalized_points, normalized_depths, scale = (
        normalize_vggt_training_gt(c2w, points, depths, valid)
    )

    torch.testing.assert_close(extrinsics[:, 0, :3, :3], torch.eye(3)[None])
    torch.testing.assert_close(extrinsics[:, 0, :3, 3], torch.zeros((1, 3)))
    average_radius = normalized_points.norm(dim=-1)[valid].mean()
    torch.testing.assert_close(average_radius, torch.tensor(1.0), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(normalized_depths * scale[:, None, None, None], depths)


def test_vggt_track_loss_is_near_zero_for_exact_confident_predictions():
    tracks = torch.tensor(
        [[[10.0, 10.0], [20.0, 20.0]], [[11.0, 10.0], [21.0, 20.0]]]
    )
    visible = torch.ones((2, 2), dtype=torch.bool)
    known = torch.ones((2, 2), dtype=torch.bool)
    predictions = {
        "track_list": [tracks[None].clone(), tracks[None].clone()],
        "track_vis": torch.full((1, 2, 2), 12.0),
        "track_conf": torch.full((1, 2, 2), 12.0),
    }
    loss = vggt_tracking_training_loss(
        predictions,
        {
            "tracks": tracks,
            "track_visible": visible,
            "track_known": known,
        },
    )
    assert float(loss["coord"]) == 0.0
    assert float(loss["visibility"]) < 1e-4
    assert float(loss["confidence"]) < 1e-4


def test_vggt_track_loss_increases_for_wrong_coordinates():
    tracks = torch.zeros((2, 3, 2))
    visible = torch.ones((2, 3), dtype=torch.bool)
    known = torch.ones((2, 3), dtype=torch.bool)
    exact = {
        "track_list": [tracks[None].clone()],
        "track_vis": torch.full((1, 2, 3), 12.0),
        "track_conf": torch.full((1, 2, 3), 12.0),
    }
    wrong = {
        "track_list": [tracks[None].clone() + 8.0],
        "track_vis": torch.full((1, 2, 3), 12.0),
        "track_conf": torch.full((1, 2, 3), 12.0),
    }
    gt = {"tracks": tracks, "track_visible": visible, "track_known": known}
    exact_loss = vggt_tracking_training_loss(exact, gt)["total"]
    wrong_loss = vggt_tracking_training_loss(wrong, gt)["total"]
    assert float(wrong_loss) > float(exact_loss) + 5.0


if __name__ == "__main__":
    test_vggt_gt_normalization_centers_first_camera_and_sets_unit_scale()
    test_vggt_track_loss_is_near_zero_for_exact_confident_predictions()
    test_vggt_track_loss_increases_for_wrong_coordinates()
    print("3 direct tests passed")
