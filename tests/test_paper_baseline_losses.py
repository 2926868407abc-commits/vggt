from __future__ import annotations

import math
from pathlib import Path
import sys
from unittest import mock

import torch
from torchvision.transforms import functional as TVF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.paper_losses import (  # noqa: E402
    adaptive_mask_regularization,
    aparate_depth_loss,
    fool3d_depth_loss,
    fool3d_non_printability_score,
    fool3d_printable_palette,
    fool3d_smoothness_loss,
    keep_largest_absolute_gradient,
    passive_vo_box_blur,
    paper_color_transform,
    paper_patch_noise_like,
    paper_random_texture,
    project_spoo_borders_,
    rectangular_focus_mask,
    pgd_sign_step,
    spoo_depth_loss,
    spoo_edge_based_adam_step_,
    spoo_initial_borders,
    spoo_matting_laplacian,
    spoo_mask_size_loss,
    spoo_polynomial_lr,
    spoo_rectangle_area_fraction,
    spoo_soft_rectangle,
    spoo_realism_loss,
    spoo_target_area_schedule,
    spoo_updated_mask_weight,
    SpooVgg19Naturalness,
    sample_texture_conversion,
    texture_conversion,
    vo_mprms_loss,
    vo_rms_loss,
)


def _trajectory(xs: list[float]) -> torch.Tensor:
    poses = torch.eye(4).repeat(1, len(xs), 1, 1)
    poses[0, :, 0, 3] = torch.tensor(xs)
    return poses


def test_spoo_is_smaller_for_farther_depth() -> None:
    mask = torch.ones(2, 1, 4, 5)
    near, _ = spoo_depth_loss(torch.ones(2, 4, 5), mask)
    far, _ = spoo_depth_loss(torch.full((2, 4, 5), 4.0), mask)
    assert far < near


def test_aparate_penalises_inside_and_outside_focus() -> None:
    patch = torch.zeros(1, 1, 8, 8)
    patch[:, :, 3:5, 3:5] = 1
    focus = rectangular_focus_mask(patch, scale=2.0)
    assert focus.sum() > patch.sum()
    loss, terms = aparate_depth_loss(torch.ones(1, 8, 8), patch, focus)
    assert loss > 0
    assert terms["baseline_aparate_inside_l2"] > 0
    assert terms["baseline_aparate_outside_l1"] > 0


def test_aparate_uses_paper_sum_reductions() -> None:
    patch = torch.zeros(1, 1, 4, 4)
    patch[:, :, 1:3, 1:3] = 1.0
    focus = torch.ones_like(patch)
    loss, terms = aparate_depth_loss(torch.ones(1, 4, 4), patch, focus)
    assert terms["baseline_aparate_inside_l2"] == 4.0
    assert terms["baseline_aparate_outside_l1"] == 12.0
    assert loss.item() == 16.0


def test_3d2fool_public_losses_match_literal_formulas() -> None:
    texture = torch.tensor(
        [[[[0.0, 1.0], [0.5, 0.25]],
          [[0.0, 0.0], [0.0, 0.0]],
          [[0.0, 0.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    vertical = (texture[:, :, 1:, :-1] - texture[:, :, :-1, :-1]).square()
    horizontal = (texture[:, :, :-1, 1:] - texture[:, :, :-1, :-1]).square()
    expected_smooth = (vertical + horizontal).sum().square() / (1 * 3 * 2 * 2)
    assert torch.allclose(fool3d_smoothness_loss(texture), expected_smooth)

    palette = fool3d_printable_palette(texture.device, texture.dtype)
    image = texture.permute(0, 2, 3, 1)
    expected_nps = (
        (image[:, None] - palette[None, :, None, None]).abs().sum(-1) / 255.0
    ).min(1).values.sum() / 2 / 2
    assert torch.allclose(fool3d_non_printability_score(texture), expected_nps)

    loss, _ = fool3d_depth_loss(torch.ones(1, 2, 2), torch.ones(1, 1, 2, 2))
    assert loss.item() == 10.0


def test_vo_rms_and_mprms_zero_on_identical_trajectory() -> None:
    poses = _trajectory([0.0, 1.0, 2.0, 4.0])
    rms, _ = vo_rms_loss(poses, poses)
    mprms, _ = vo_mprms_loss(poses, poses)
    assert rms.item() == 0.0
    assert mprms.item() == 0.0


def test_mprms_sees_local_drift_not_only_last_frame() -> None:
    target = _trajectory([0.0, 1.0, 2.0, 3.0])
    pred = _trajectory([0.0, 2.0, 2.0, 3.0])
    mprms, terms = vo_mprms_loss(pred, target)
    assert mprms > 0
    assert terms["baseline_vo_short_gap"] > 0


def test_adaptive_mask_area_has_gradient() -> None:
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    loss, terms = adaptive_mask_regularization(
        logits, area_fraction=0.25, area_weight=1.0, binary_weight=0.1,
        tv_weight=0.1, temperature=1.0,
    )
    loss.backward()
    assert logits.grad is not None
    assert math.isclose(float(terms["adaptive_mask_area"]), 0.5, rel_tol=1e-6)


def test_texture_conversion_preserves_shape_and_gradient() -> None:
    texture = torch.rand(1, 3, 8, 8, requires_grad=True)
    converted = texture_conversion(texture, training=True, repeats=3, output_size=16)
    assert converted.shape == (1, 3, 16, 16)
    converted.mean().backward()
    assert texture.grad is not None


def test_texture_conversion_matches_paper_dimensions() -> None:
    texture = torch.rand(1, 3, 128, 128)
    converted = texture_conversion(
        texture, training=False, repeats=6, output_size=4096, upsample_factor=8
    )
    assert converted.shape == (1, 3, 4096, 4096)


def test_texture_conversion_matches_explicit_nearest_expand_and_crop() -> None:
    texture = torch.arange(3 * 3 * 3, dtype=torch.float32).reshape(1, 3, 3, 3)
    actual = texture_conversion(
        texture, training=False, repeats=3, output_size=12, upsample_factor=2
    )
    tiled = texture.repeat(1, 1, 3, 3)
    expanded = torch.nn.functional.interpolate(tiled, scale_factor=2, mode="nearest")
    y0 = (expanded.shape[-2] - 12) // 2
    x0 = (expanded.shape[-1] - 12) // 2
    expected = expanded[..., y0:y0 + 12, x0:x0 + 12]
    assert torch.equal(actual, expected)


def test_fused_texture_sampler_matches_explicit_4096_path() -> None:
    texture = torch.rand(1, 3, 5, 5, requires_grad=True)
    grid = torch.empty(2, 7, 9, 2).uniform_(-1.0, 1.0)
    explicit = texture_conversion(
        texture, training=False, repeats=3, output_size=24, upsample_factor=2
    )
    expected = torch.nn.functional.grid_sample(
        explicit.expand(2, -1, -1, -1), grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    actual = sample_texture_conversion(
        texture, grid, training=False, repeats=3,
        output_size=24, upsample_factor=2,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    actual.mean().backward()
    assert texture.grad is not None and torch.isfinite(texture.grad).all()


def test_texture_conversion_transforms_tiles_independently() -> None:
    texture = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
    torch.manual_seed(123)
    transformed = texture_conversion(texture, training=True, repeats=3, output_size=8)
    plain = texture_conversion(texture, training=False, repeats=3, output_size=8)
    assert not torch.equal(transformed, plain)


def test_pgd_sign_step_descends_accumulated_objective() -> None:
    parameter = torch.tensor([0.2, 0.8], requires_grad=True)
    objective = parameter[0] - parameter[1]
    objective.backward()
    pgd_sign_step([parameter], step_size=0.1)
    assert torch.allclose(parameter.detach(), torch.tensor([0.1, 0.9]))


def test_spoo_soft_rectangle_and_size_loss() -> None:
    full = torch.tensor([0.0, 1.0, 0.0, 1.0], requires_grad=True)
    small = torch.tensor([0.25, 0.75, 0.25, 0.75], requires_grad=True)
    full_mask = spoo_soft_rectangle(full, 64)
    small_mask = spoo_soft_rectangle(small, 64)
    assert full_mask.shape == (1, 1, 64, 64)
    assert full_mask.mean() > small_mask.mean()
    assert spoo_mask_size_loss(full) > spoo_mask_size_loss(small)
    small_mask.mean().backward()
    assert small.grad is not None and torch.isfinite(small.grad).all()

    # Literal public expression at source pixel (y=2,x=3).
    borders = torch.tensor([0.25, 0.75, 0.25, 0.75])
    mask = spoo_soft_rectangle(borders, (8, 8), sharpness=1.0)
    y, x = torch.tensor(2.0), torch.tensor(3.0)
    expected = 0.25 * (1 - torch.tanh(y - 2) * torch.tanh(y - 6)) * (
        1 - torch.tanh(x - 2) * torch.tanh(x - 6)
    )
    assert torch.allclose(mask[0, 0, 2, 3], expected)


def test_spoo_source_initial_rectangle_and_non_square_size_loss() -> None:
    borders = spoo_initial_borders((450, 600))
    assert torch.allclose(borders, torch.tensor([0.0, 1.0, 0.0, 410.0 / 450.0]))
    loss = spoo_mask_size_loss(borders, (450, 600))
    assert torch.allclose(loss, torch.tensor((600.0 + 410.0) / 1050.0))


def test_spoo_edge_step_uses_pixel_units_and_restores_other_edges() -> None:
    borders = torch.tensor([0.1, 0.8, 0.1, 0.8], requires_grad=True)
    optimizer = torch.optim.Adam([borders], lr=1.0, betas=(0.5, 0.9))
    # These normalized-coordinate gradients correspond to source pixel-space
    # gradients [1,-4,2,3] on a 600x450 source image.
    borders.grad = torch.tensor([600.0, -2400.0, 900.0, 1350.0])
    old = borders.detach().clone()
    selected = spoo_edge_based_adam_step_(borders, optimizer, (450, 600))
    assert selected == 1
    assert torch.equal(borders.detach()[[0, 2, 3]], old[[0, 2, 3]])
    assert torch.allclose(
        borders.detach()[1] - old[1], torch.tensor(1.0 / 600.0), atol=1e-7
    )


def test_spoo_equal_budget_edge_step_moves_ten_source_pixels() -> None:
    borders = torch.tensor([0.1, 0.8, 0.1, 0.8], requires_grad=True)
    optimizer = torch.optim.Adam([borders], lr=10.0, betas=(0.5, 0.9))
    borders.grad = torch.tensor([600.0, -2400.0, 900.0, 1350.0])
    old = borders.detach().clone()
    selected = spoo_edge_based_adam_step_(borders, optimizer, (450, 600))
    assert selected == 1
    assert torch.allclose(
        borders.detach()[1] - old[1], torch.tensor(10.0 / 600.0), atol=1e-7
    )


def test_spoo_keeps_one_border_gradient_and_stops_at_target() -> None:
    borders = torch.tensor([0.2, 0.8, 0.2, 0.8], requires_grad=True)
    borders.grad = torch.tensor([1.0, -4.0, 2.0, 3.0])
    assert keep_largest_absolute_gradient(borders) == 1
    assert torch.equal(borders.grad, torch.tensor([0.0, -4.0, 0.0, 0.0]))
    with torch.no_grad():
        borders.copy_(torch.tensor([0.25, 0.55, 0.20, 0.50]))
    reached = project_spoo_borders_(borders, target_area_fraction=1.0 / 9.0)
    assert reached
    # Public code stops after an overshoot; it does not snap to exactly 1/9.
    assert abs(float(spoo_rectangle_area_fraction(borders)) - 0.09) < 1e-6


def test_spoo_source_schedules() -> None:
    assert math.isclose(spoo_target_area_schedule(0, 100, 1.0, 1 / 9), 1.0)
    assert math.isclose(spoo_target_area_schedule(60, 100, 1.0, 1 / 9), 1 / 9)
    assert math.isclose(spoo_target_area_schedule(100, 100, 1.0, 1 / 9), 1 / 9)
    weight = spoo_updated_mask_weight(1.0, 0.5, 0.4, 19)
    assert weight == 1.0
    weight = spoo_updated_mask_weight(1.0, 0.5, 0.4, 20)
    assert weight == 1.5
    weight = spoo_updated_mask_weight(1.0, 0.3, 0.4, 20)
    assert weight == 0.7
    assert 0.49 < spoo_polynomial_lr(1.0, 99, 200) <= 0.5
    assert spoo_polynomial_lr(1.0, 150, 200) == 0.5


def test_spoo_matting_realism_is_near_zero_for_constant_colour() -> None:
    reference = torch.rand(1, 3, 4, 5)
    laplacian = spoo_matting_laplacian(reference)
    constant = torch.full_like(reference, 0.4)
    value = spoo_realism_loss(constant, laplacian)
    assert abs(value.item()) < 1e-4


def test_spoo_vgg19_layer_selection_has_texture_gradient_without_download() -> None:
    class TinyVgg(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            channels = 3
            for block in range(1, 6):
                layers.extend((torch.nn.Conv2d(channels, 4, 3, padding=1), torch.nn.ReLU()))
                channels = 4
                if block == 4:
                    layers.extend((torch.nn.Conv2d(4, 4, 3, padding=1), torch.nn.ReLU()))
                if block < 5:
                    layers.append(torch.nn.MaxPool2d(2))
            self.features = torch.nn.Sequential(*layers)

    reference = torch.rand(1, 3, 32, 32)
    with mock.patch("torchvision.models.vgg19", side_effect=lambda **_: TinyVgg()):
        naturalness = SpooVgg19Naturalness(reference)
    texture = reference.clone().requires_grad_(True)
    style, content = naturalness(texture)
    (style + content).backward()
    assert texture.grad is not None and torch.isfinite(texture.grad).all()


def test_passive_vo_source_box_blur() -> None:
    texture = torch.zeros(1, 3, 11, 11)
    texture[:, :, 5, 5] = 1.0
    blurred = passive_vo_box_blur(texture, 9)
    assert blurred.shape == texture.shape
    assert 0.0 < blurred[0, 0, 5, 5] < 1.0
    assert blurred.sum() > 0


def test_3d2fool_random_initializer_preserves_public_nhwc_rng_layout() -> None:
    torch.manual_seed(7)
    actual = paper_random_texture(
        "baseline_3d2fool_depth", 3, 4, device="cpu"
    )
    torch.manual_seed(7)
    expected = torch.rand(1, 3, 4, 3).permute(0, 3, 1, 2).contiguous()
    assert torch.equal(actual, expected)


def test_aparate_noise_is_bounded_uniform() -> None:
    torch.manual_seed(11)
    noise = paper_patch_noise_like(
        torch.zeros(2, 3, 32, 32), "baseline_aparate_depth", 0.1
    )
    assert float(noise.min()) >= -0.1
    assert float(noise.max()) <= 0.1
    assert torch.unique(noise).numel() > 100


def test_paper_color_transform_matches_torchvision_operator_order() -> None:
    image = torch.rand(1, 3, 8, 9)
    values = torch.tensor([1.1])
    actual = paper_color_transform(
        image,
        brightness=values,
        contrast=torch.tensor([0.9]),
        saturation=torch.tensor([1.05]),
        hue=torch.tensor([0.02]),
    )
    expected = TVF.adjust_brightness(image[0], 1.1)
    expected = TVF.adjust_contrast(expected, 0.9)
    expected = TVF.adjust_saturation(expected, 1.05)
    expected = TVF.adjust_hue(expected, 0.02)
    assert torch.allclose(actual[0], expected)
