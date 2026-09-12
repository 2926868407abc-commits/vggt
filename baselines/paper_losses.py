"""Paper-baseline losses used by the VGGT physical-patch pipeline.

Implementation policy is deliberately strict:

* copy the public reference implementation's mathematics when code exists;
* otherwise implement the published equations and reported hyperparameters;
* isolate unavoidable VGGT/TUM substitutions in explicitly named functions.

The victim model and renderer necessarily differ from the source papers, but a
change of victim is not a reason to silently replace TV, NPS, mask schedules or
texture conversion with convenient generic versions.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF


PAPER_BASELINE_LOSSES = (
    "baseline_spoo_depth",
    "baseline_aparate_depth",
    "baseline_3d2fool_depth",
    "baseline_vo_rms",
    "baseline_vo_mprms",
)

APARATE_SHAPE_EXTENSION = "baseline_aparate_shape"
DEPTH_BASELINE_LOSSES = frozenset((*PAPER_BASELINE_LOSSES[:3], APARATE_SHAPE_EXTENSION))
VO_BASELINE_LOSSES = frozenset(PAPER_BASELINE_LOSSES[3:])


def paper_random_texture(
    attack_loss: str,
    height: int,
    width: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw a random initializer using the public method's tensor layout.

    3D2Fool's released code creates ``[N,H,W,C]`` and only later permutes to
    channel-first form. Drawing directly as NCHW changes which seeded random
    value lands in each pixel/channel.
    """
    if attack_loss == "baseline_3d2fool_depth":
        return torch.rand(
            (1, int(height), int(width), 3), device=device, dtype=dtype
        ).permute(0, 3, 1, 2).contiguous()
    return torch.rand(
        (1, 3, int(height), int(width)), device=device, dtype=dtype
    )


def paper_patch_noise_like(
    tensor: torch.Tensor,
    attack_loss: str,
    amplitude: float,
) -> torch.Tensor:
    """Sample the baseline's stated patch-noise distribution."""
    amplitude = float(amplitude)
    if amplitude <= 0:
        return torch.zeros_like(tensor)
    if attack_loss in ("baseline_aparate_depth", APARATE_SHAPE_EXTENSION):
        # APARATE reports bounded noise in [-0.1, 0.1], not Gaussian noise.
        return torch.empty_like(tensor).uniform_(-amplitude, amplitude)
    return torch.randn_like(tensor) * amplitude


def paper_color_transform(
    images: torch.Tensor,
    *,
    brightness: torch.Tensor | None = None,
    contrast: torch.Tensor | None = None,
    saturation: torch.Tensor | None = None,
    hue: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply torchvision colour operators frame-by-frame, as source code does."""
    rows = []
    for index, image in enumerate(images):
        if brightness is not None:
            image = TVF.adjust_brightness(image, float(brightness[index]))
        if contrast is not None:
            image = TVF.adjust_contrast(image, float(contrast[index]))
        if saturation is not None:
            image = TVF.adjust_saturation(image, float(saturation[index]))
        if hue is not None:
            image = TVF.adjust_hue(image, float(hue[index]))
        rows.append(image)
    return torch.stack(rows, dim=0)


# Exact ten-colour set in Gandolfi et al.'s public 3D2Fool ``attack_base.py``.
# The public implementation normalises these values to [0,1], then (somewhat
# unusually) divides RGB distances by 255 once more in its NPS function.  We
# preserve that behaviour instead of "correcting" it.
FOOL3D_PRINTABLE_COLORS_255 = (
    (0, 0, 0),
    (255, 255, 255),
    (0, 18, 79),
    (5, 80, 214),
    (71, 178, 243),
    (178, 159, 211),
    (77, 58, 0),
    (211, 191, 167),
    (247, 110, 26),
    (110, 76, 16),
)


def pgd_sign_step(parameters: list[torch.Tensor], step_size: float) -> None:
    """In-place descent step on the sign of the accumulated objective gradient.

    The caller negates maximisation losses before backward, so ordinary descent
    here is equivalent to the paper's ``P += alpha * sign(grad attack_loss)``.
    """
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.add_(-float(step_size) * parameter.grad.sign())


def spoo_soft_rectangle(
    borders: torch.Tensor,
    output_size: int | tuple[int, int],
    sharpness: float = 1.0,
) -> torch.Tensor:
    """SPOO's public soft rectangle from normalized ``[l,r,t,b]`` borders.

    The reference code evaluates ``tanh(pixel-border)`` on pixel coordinates.
    Borders are stored here in [0,1] only so the same state works at arbitrary
    texture resolutions; converting both coordinates and borders back to pixel
    units makes the expression identical.  ``sharpness=1`` is the source value.
    """
    if borders.numel() != 4:
        raise ValueError(f"SPOO needs four borders [l,r,t,b], got {tuple(borders.shape)}")
    if isinstance(output_size, int):
        height = width = int(output_size)
    else:
        height, width = (int(value) for value in output_size)
    l, r, t, b = borders.reshape(-1)
    ys = torch.arange(height, device=borders.device, dtype=borders.dtype)
    xs = torch.arange(width, device=borders.device, dtype=borders.dtype)
    gain = max(float(sharpness), 1.0)
    # The source uses right=width and bottom=height for a full mask.
    l_px, r_px = l * width, r * width
    t_px, b_px = t * height, b * height
    inside_y = 1.0 - torch.tanh(gain * (ys - t_px)) * torch.tanh(gain * (ys - b_px))
    inside_x = 1.0 - torch.tanh(gain * (xs - l_px)) * torch.tanh(gain * (xs - r_px))
    return (0.25 * inside_y[:, None] * inside_x[None, :]).clamp(0.0, 1.0)[None, None]


def spoo_initial_borders(
    output_size: int | tuple[int, int],
    bottom_margin_pixels: int = 40,
) -> torch.Tensor:
    """Return SPOO ``paint_mask=-2`` borders in normalized coordinates.

    The public implementation starts from ``[0, W, 0, H-40]`` in pixel
    coordinates.  We store the same rectangle normalized so checkpoints remain
    resolution independent.
    """
    if isinstance(output_size, int):
        height = width = int(output_size)
    else:
        height, width = (int(value) for value in output_size)
    margin = min(max(int(bottom_margin_pixels), 0), max(height - 1, 0))
    return torch.tensor([0.0, 1.0, 0.0, (height - margin) / max(height, 1)])


def spoo_mask_size_loss(
    borders: torch.Tensor,
    output_size: int | tuple[int, int] | None = None,
) -> torch.Tensor:
    """SPOO Eq. (6), expressed for normalized stored borders.

    The source divides the sum of the two edge lengths in pixels by ``H+W``.
    The old square-texture shortcut is retained when no size is supplied.
    """
    l, r, t, b = borders.reshape(-1)
    width_span = (r - l).clamp_min(0.0)
    height_span = (b - t).clamp_min(0.0)
    if output_size is None:
        return 0.5 * (width_span + height_span)
    if isinstance(output_size, int):
        height = width = int(output_size)
    else:
        height, width = (int(value) for value in output_size)
    return (width_span * width + height_span * height) / max(height + width, 1)


def spoo_rectangle_area_fraction(borders: torch.Tensor) -> torch.Tensor:
    l, r, t, b = borders.reshape(-1)
    return (r - l).clamp_min(0.0) * (b - t).clamp_min(0.0)


def keep_largest_absolute_gradient(parameter: torch.Tensor) -> int | None:
    """Keep only SPOO's single largest-magnitude border gradient."""
    if parameter.grad is None or parameter.grad.numel() == 0:
        return None
    flat = parameter.grad.reshape(-1)
    index = int(flat.abs().argmax().item())
    value = flat[index].clone()
    flat.zero_()
    flat[index] = value
    return index


def spoo_edge_based_adam_step_(
    borders: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    output_size: int | tuple[int, int],
) -> int | None:
    """Source-equivalent single-edge Adam update for normalized borders.

    SPOO stores borders in pixels and applies Adam with lr=1 to all four
    gradients, then restores the three non-selected edges.  This repository
    stores borders in [0,1], so gradients and the selected delta are converted
    to and from pixel units around the optimizer step.  Adam's state is still
    updated for all four edges, exactly as in the public ``edge_based_update``.
    """
    if borders.grad is None or borders.grad.numel() == 0:
        return None
    if isinstance(output_size, int):
        height = width = int(output_size)
    else:
        height, width = (int(value) for value in output_size)
    scales = borders.new_tensor([width, width, height, height]).reshape_as(borders)
    # dL/d(pixel) = dL/d(normalized) / pixels-per-normalized-unit.
    borders.grad.div_(scales)
    index = int(borders.grad.reshape(-1).abs().argmax().item())
    old = borders.detach().clone()
    optimizer.step()
    with torch.no_grad():
        delta = borders - old
        borders.copy_(old)
        flat = borders.reshape(-1)
        flat[index].add_(delta.reshape(-1)[index] / scales.reshape(-1)[index])
    return index


def project_spoo_borders_(
    borders: torch.Tensor,
    target_area_fraction: float,
    min_span: float = 1e-3,
    *,
    snap_to_target: bool = False,
) -> bool:
    """Keep a SPOO border state valid and report its rectangle stop condition.

    The public implementation stops region updates after the soft mask crosses
    its target; it does not rescale an overshooting rectangle back to exactly
    the target.  ``snap_to_target`` exists only to read old checkpoints and is
    disabled by every source-faithful profile.
    """
    target = min(max(float(target_area_fraction), min_span**2), 1.0)
    with torch.no_grad():
        borders.clamp_(0.0, 1.0)
        l, r, t, b = [float(value) for value in borders.reshape(-1)]
        if r - l < min_span:
            centre = 0.5 * (l + r)
            l, r = centre - min_span / 2, centre + min_span / 2
        if b - t < min_span:
            centre = 0.5 * (t + b)
            t, b = centre - min_span / 2, centre + min_span / 2
        l, r = max(0.0, l), min(1.0, r)
        t, b = max(0.0, t), min(1.0, b)
        area = max(r - l, min_span) * max(b - t, min_span)
        reached = area <= target
        if reached and snap_to_target:
            scale = math.sqrt(target / max(area, min_span**2))
            width = (r - l) * scale
            height = (b - t) * scale
            cx, cy = 0.5 * (l + r), 0.5 * (t + b)
            l, r = cx - width / 2, cx + width / 2
            t, b = cy - height / 2, cy + height / 2
            if l < 0:
                r -= l
                l = 0.0
            if r > 1:
                l -= r - 1
                r = 1.0
            if t < 0:
                b -= t
                t = 0.0
            if b > 1:
                t -= b - 1
                b = 1.0
        borders.copy_(borders.new_tensor([l, r, t, b]).reshape_as(borders))
    return reached


def spoo_target_area_schedule(
    mask_update: int,
    mask_updates_total: int,
    initial_area_fraction: float,
    final_area_fraction: float,
    schedule_fraction: float = 0.6,
) -> float:
    """Quadratic target schedule from the public ``MaskWeightUpdater``."""
    total = max(int(mask_updates_total), 1)
    end_step = max(float(schedule_fraction) * total, 1.0)
    x = min(max(float(mask_update), 0.0), end_step)
    initial = float(initial_area_fraction)
    final = float(final_area_fraction)
    return (initial - final) / (end_step * end_step) * x * x + (
        2.0 * (final - initial) / end_step
    ) * x + initial


def spoo_updated_mask_weight(
    current_weight: float,
    current_area_fraction: float,
    target_area_fraction: float,
    mask_update: int,
    *,
    interval: int = 20,
    upscaler: float = 1.5,
    downscaler: float = 0.7,
) -> float:
    """One source-faithful adaptive mask-weight update."""
    if int(mask_update) <= 0 or int(mask_update) % max(int(interval), 1) != 0:
        return float(current_weight)
    scale = float(upscaler) if current_area_fraction >= target_area_fraction else float(downscaler)
    return float(current_weight) * scale


def spoo_polynomial_lr(
    base_lr: float,
    update: int,
    total_updates: int,
    *,
    end_fraction: float = 0.5,
    power: float = 0.9,
) -> float:
    """SPOO public polynomial LR: decay to half LR over the first half."""
    max_decay = max(int(total_updates) // 2, 2)
    step = max(1, int(update) + 1)
    end_lr = float(base_lr) * float(end_fraction)
    if step > max_decay:
        return end_lr
    return (float(base_lr) - end_lr) * ((1.0 - step / max_decay) ** float(power)) + end_lr


def _depth_thw(depth: torch.Tensor) -> torch.Tensor:
    """Canonicalise VGGT depth output to ``[T,H,W]``."""
    while depth.ndim > 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth reducible to [T,H,W], got {tuple(depth.shape)}")
    return depth


def _mask_thw(mask: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Canonicalise a projected mask and resize it without inventing gradients."""
    while mask.ndim > 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(f"Expected mask reducible to [T,H,W], got {tuple(mask.shape)}")
    if tuple(mask.shape) != tuple(shape):
        mask = F.interpolate(mask[:, None].float(), size=shape[-2:], mode="bilinear",
                             align_corners=False)[:, 0]
    return mask.to(dtype=torch.float32).clamp(0.0, 1.0)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def rectangular_focus_mask(patch_mask: torch.Tensor, scale: float = 2.0,
                           threshold: float = 1e-3) -> torch.Tensor:
    """Expand each visible patch bounding box into an object-like focus region.

    The depth papers use a target-object segmentation mask.  TUM does not label
    the monitor/table object, so the adapted protocol uses a fixed rectangle
    around the physically projected patch.  ``scale=1`` is patch-oriented;
    values above one test whether the effect spreads beyond the pasted pixels.
    """
    if patch_mask.ndim != 4 or patch_mask.shape[1] != 1:
        raise ValueError(f"Expected patch mask [T,1,H,W], got {tuple(patch_mask.shape)}")
    scale = max(float(scale), 1.0)
    out = torch.zeros_like(patch_mask)
    _, _, height, width = patch_mask.shape
    for frame in range(patch_mask.shape[0]):
        active = torch.nonzero(patch_mask[frame, 0] > threshold, as_tuple=False)
        if active.numel() == 0:
            continue
        y0, x0 = active.min(dim=0).values.tolist()
        y1, x1 = active.max(dim=0).values.tolist()
        cy, cx = 0.5 * (y0 + y1), 0.5 * (x0 + x1)
        half_h = 0.5 * (y1 - y0 + 1) * scale
        half_w = 0.5 * (x1 - x0 + 1) * scale
        yy0, yy1 = max(0, int(math.floor(cy - half_h))), min(height, int(math.ceil(cy + half_h)))
        xx0, xx1 = max(0, int(math.floor(cx - half_w))), min(width, int(math.ceil(cx + half_w)))
        out[frame, 0, yy0:yy1, xx0:xx1] = 1.0
    return torch.maximum(out, patch_mask)


def inverse_depth(depth: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return _depth_thw(depth).float().clamp_min(float(eps)).reciprocal()


def spoo_depth_loss(depth: torch.Tensor, focus_mask: torch.Tensor,
                    eps: float = 1e-3) -> tuple[torch.Tensor, dict[str, float]]:
    """SPOO/Cheng et al. Eq. (3): minimise inverse depth on the target region."""
    disparity = inverse_depth(depth, eps)
    focus = _mask_thw(focus_mask, tuple(disparity.shape)).to(disparity.device)
    loss = masked_mean(disparity.square(), focus)
    return loss, {
        "total": float(loss.detach().cpu()),
        "baseline_depth_inverse_mse": float(loss.detach().cpu()),
        "baseline_focus_fraction": float(focus.mean().detach().cpu()),
    }


def aparate_depth_loss(depth: torch.Tensor, patch_mask: torch.Tensor,
                       focus_mask: torch.Tensor, eps: float = 1e-3,
                       outside_weight: float = 1.0,
                       ) -> tuple[torch.Tensor, dict[str, float]]:
    """APARATE Eq. (7-10), with zero inverse-depth as the far target.

    APARATE applies a quadratic penalty inside the patch and an L1 penalty over
    the remainder of the target object.  VGGT predicts metric depth rather than
    normalised disparity, so the direct equivalent of the paper's far target is
    zero reciprocal depth.
    """
    disparity = inverse_depth(depth, eps)
    patch = _mask_thw(patch_mask, tuple(disparity.shape)).to(disparity.device)
    focus = _mask_thw(focus_mask, tuple(disparity.shape)).to(disparity.device)
    outside = (focus - patch).clamp_min(0.0)
    # Algorithm 1 / Eq. (7-10) use sums, not region-wise means.  Keeping the
    # sums matters because the inside/outside balance changes with mask size.
    inside_loss = (disparity * patch).square().sum()
    outside_loss = (disparity.abs() * outside).sum()
    loss = inside_loss + float(outside_weight) * outside_loss
    return loss, {
        "total": float(loss.detach().cpu()),
        "baseline_aparate_inside_l2": float(inside_loss.detach().cpu()),
        "baseline_aparate_outside_l1": float(outside_loss.detach().cpu()),
        "baseline_focus_fraction": float(focus.mean().detach().cpu()),
    }


def fool3d_depth_loss(depth: torch.Tensor, surface_mask: torch.Tensor,
                      eps: float = 1e-3) -> tuple[torch.Tensor, dict[str, float]]:
    """3D2Fool public vanishing loss, adapted from disparity to 1/depth.

    Its code computes ``sum(10 * disparity**2 * mask) / sum(mask)``.  VGGT
    emits metric depth, so reciprocal depth is the only victim-interface
    substitution; the factor ten and masked mean are preserved exactly.
    """
    disparity = inverse_depth(depth, eps)
    surface = _mask_thw(surface_mask, tuple(disparity.shape)).to(disparity.device)
    loss = 10.0 * masked_mean(disparity.square(), surface)
    terms = {
        "total": float(loss.detach().cpu()),
        "baseline_3d2fool_vanishing": float(loss.detach().cpu()),
        "baseline_focus_fraction": float(surface.mean().detach().cpu()),
    }
    return loss, terms


def _check_pose_pair(pred_rel: torch.Tensor,
                     target_rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_rel = target_rel.detach().to(device=pred_rel.device, dtype=pred_rel.dtype)
    if pred_rel.shape != target_rel.shape or pred_rel.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected matching [...,T,4,4] poses, got {tuple(pred_rel.shape)} and "
            f"{tuple(target_rel.shape)}"
        )
    if pred_rel.shape[-3] < 2:
        raise ValueError("VO trajectory loss needs at least two frames")
    return pred_rel, target_rel


def vo_rms_loss(pred_rel: torch.Tensor,
                target_rel: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Passive-VO RMS criterion: accumulated translation error from frame zero."""
    pred_rel, target_rel = _check_pose_pair(pred_rel, target_rel)
    error = torch.linalg.vector_norm(
        pred_rel[..., :3, 3] - target_rel[..., :3, 3], dim=-1
    )
    loss = error.mean()
    return loss, {
        "total": float(loss.detach().cpu()),
        "pose_loss": float(loss.detach().cpu()),
        "baseline_vo_rms": float(loss.detach().cpu()),
    }


def vo_mprms_loss(pred_rel: torch.Tensor,
                  target_rel: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Passive-VO MPRMS: equal-weight average over every subtrajectory length.

    For every gap ``l`` and start frame ``i``, compare the translation of
    ``T_i^-1 T_(i+l)``.  Averaging within each gap and then across gaps matches
    the paper's mean-partial-trajectory construction instead of letting the many
    short pairs dominate.
    """
    pred_rel, target_rel = _check_pose_pair(pred_rel, target_rel)
    n_frames = pred_rel.shape[-3]
    by_length = []
    for gap in range(1, n_frames):
        pred_pair = torch.linalg.inv(pred_rel[..., :-gap, :, :]) @ pred_rel[..., gap:, :, :]
        target_pair = torch.linalg.inv(target_rel[..., :-gap, :, :]) @ target_rel[..., gap:, :, :]
        error = torch.linalg.vector_norm(
            pred_pair[..., :3, 3] - target_pair[..., :3, 3], dim=-1
        )
        by_length.append(error.mean())
    values = torch.stack(by_length)
    loss = values.mean()
    return loss, {
        "total": float(loss.detach().cpu()),
        "pose_loss": float(loss.detach().cpu()),
        "baseline_vo_mprms": float(loss.detach().cpu()),
        "baseline_vo_short_gap": float(values[0].detach().cpu()),
        "baseline_vo_long_gap": float(values[-1].detach().cpu()),
        "baseline_vo_gap_count": int(n_frames - 1),
    }


def isotropic_total_variation(
    texture: torch.Tensor,
    eps: float = 1e-12,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """APARATE Eq. (12) isotropic TV.

    The paper's objective uses a sum. ``mean`` remains available for the
    repository's non-paper adaptive-mask extension, whose coefficients were
    calibrated under mean reduction.
    """
    dx = texture[..., :-1, :-1] - texture[..., :-1, 1:]
    dy = texture[..., :-1, :-1] - texture[..., 1:, :-1]
    values = torch.sqrt(dx.square() + dy.square() + eps)
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"Unknown TV reduction: {reduction}")


def default_printable_palette(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Small fixed RGB palette used when no measured printer gamut is supplied."""
    values = (
        (0.05, 0.05, 0.05), (0.95, 0.95, 0.95),
        (0.85, 0.10, 0.10), (0.10, 0.65, 0.15), (0.10, 0.25, 0.85),
        (0.90, 0.75, 0.10), (0.75, 0.15, 0.75), (0.10, 0.75, 0.75),
        (0.45, 0.30, 0.15), (0.50, 0.50, 0.50),
    )
    return torch.tensor(values, device=device, dtype=dtype)


def non_printability_score(texture: torch.Tensor,
                           palette: torch.Tensor | None = None,
                           *, reduction: str = "mean") -> torch.Tensor:
    """APARATE Eq. (13), distance to the nearest printable RGB colour."""
    if palette is None:
        palette = default_printable_palette(texture.device, texture.dtype)
    pixels = texture.movedim(1, -1).reshape(-1, 3)
    distances = torch.linalg.vector_norm(pixels[:, None] - palette[None], dim=-1)
    values = distances.min(dim=1).values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"Unknown NPS reduction: {reduction}")


def fool3d_smoothness_loss(texture: torch.Tensor) -> torch.Tensor:
    """Exact public 3D2Fool ``loss_smooth`` for NCHW coarse textures."""
    if texture.ndim != 4:
        raise ValueError(f"Expected NCHW texture, got {tuple(texture.shape)}")
    batch, channels, height, width = texture.shape
    vertical = (texture[:, :, 1:, :-1] - texture[:, :, :-1, :-1]).square()
    horizontal = (texture[:, :, :-1, 1:] - texture[:, :, :-1, :-1]).square()
    return (vertical + horizontal).sum().square() / (batch * channels * height * width)


def fool3d_printable_palette(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(FOOL3D_PRINTABLE_COLORS_255, device=device, dtype=dtype) / 255.0


def fool3d_non_printability_score(
    texture: torch.Tensor,
    palette: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact public 3D2Fool ``loss_nps``, including its extra ``/255``."""
    if texture.ndim != 4 or texture.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W] texture, got {tuple(texture.shape)}")
    if palette is None:
        palette = fool3d_printable_palette(texture.device, texture.dtype)
    image = texture.permute(0, 2, 3, 1)
    distances = (image[:, None] - palette[None, :, None, None]).abs().sum(dim=-1) / 255.0
    nearest = distances.min(dim=1).values
    return nearest.sum() / image.shape[1] / image.shape[2]


def spoo_source_tv_loss(texture: torch.Tensor) -> torch.Tensor:
    """Exact RGB-summed finite-difference TV module in the SPOO repository."""
    if texture.ndim != 4 or texture.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W] texture, got {tuple(texture.shape)}")
    # Source Conv2d has one output channel and sums the same finite difference
    # over all three RGB channels. Zero padding is part of the public code.
    gray_sum = texture.sum(dim=1, keepdim=True)
    kernel_x = texture.new_zeros((1, 1, 3, 3))
    kernel_y = texture.new_zeros((1, 1, 3, 3))
    kernel_x[0, 0, 1, 1] = 1.0
    kernel_x[0, 0, 1, 2] = -1.0
    kernel_y[0, 0, 1, 1] = 1.0
    kernel_y[0, 0, 2, 1] = -1.0
    gx = F.conv2d(gray_sum, kernel_x, padding=1)
    gy = F.conv2d(gray_sum, kernel_y, padding=1)
    return (gx.square() + gy.square()).sum() / 2.0


def spoo_gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """SPOO/Neural-Style Gram matrix with the source normalization."""
    batch, channels, height, width = features.shape
    flat = features.reshape(batch * channels, height * width)
    return flat @ flat.t() / (batch * channels * height * width)


class SpooVgg19Naturalness(torch.nn.Module):
    """SPOO content/style losses using the public VGG19 layer selection.

    The original code uses ImageNet-pretrained torchvision VGG19 features,
    style layers conv1_1..conv5_1 and content layer conv4_2.  Full style/content
    masks are used for the warning-image baseline, so ordinary full-image Gram
    and content losses are source-equivalent.
    """

    STYLE_LAYERS = frozenset(("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"))
    CONTENT_LAYER = "conv4_2"

    def __init__(self, reference: torch.Tensor, *, weights_path: str | None = None):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        if weights_path:
            network = vgg19(weights=None)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            network.load_state_dict(state)
        else:
            network = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = network.features.eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        del network
        self.register_buffer("normalization_mean", reference.new_tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("normalization_std", reference.new_tensor((0.229, 0.224, 0.225))[None, :, None, None])
        self.features.to(device=reference.device, dtype=reference.dtype)

        with torch.no_grad():
            style, content = self._extract(reference)
        for name, value in style.items():
            self.register_buffer(f"style_target_{name}", spoo_gram_matrix(value).detach())
        self.register_buffer("content_target", content.detach())

    def _extract(self, image: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        value = (image - self.normalization_mean) / self.normalization_std
        style: dict[str, torch.Tensor] = {}
        content = None
        block = 1
        convolution = 0
        for layer in self.features:
            if isinstance(layer, torch.nn.Conv2d):
                convolution += 1
                name = f"conv{block}_{convolution}"
            elif isinstance(layer, torch.nn.ReLU):
                value = F.relu(value, inplace=False)
                name = f"relu{block}_{convolution}"
                if name == f"relu{block}_{convolution}":
                    if self.CONTENT_LAYER == f"conv{block}_{convolution}" and content is None:
                        # The source inserts ContentLoss immediately after the
                        # convolution, before this ReLU; content is captured in
                        # the Conv2d branch below instead.
                        pass
                if name == "relu5_1":
                    break
                continue
            elif isinstance(layer, torch.nn.MaxPool2d):
                name = f"pool_{block}"
            else:
                name = layer.__class__.__name__.lower()

            value = layer(value)
            if name in self.STYLE_LAYERS:
                style[name] = value
            if name == self.CONTENT_LAYER:
                content = value
            if name == "conv5_1":
                break
            if isinstance(layer, torch.nn.MaxPool2d):
                block += 1
                convolution = 0
        if content is None or set(style) != set(self.STYLE_LAYERS):
            raise RuntimeError("Could not collect SPOO VGG19 content/style layers")
        return style, content

    def forward(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        style_features, content_features = self._extract(texture)
        style_loss = texture.new_zeros(())
        for name, features in style_features.items():
            target = getattr(self, f"style_target_{name}")
            style_loss = style_loss + F.mse_loss(spoo_gram_matrix(features), target)
        content_loss = F.mse_loss(content_features, self.content_target)
        return style_loss, content_loss


def spoo_matting_laplacian(
    reference: torch.Tensor,
    *,
    eps: float = 1e-7,
    window_radius: int = 1,
) -> torch.Tensor:
    """Closed-form matting Laplacian used by SPOO's second stage."""
    try:
        import scipy.sparse
    except ImportError as exc:  # pragma: no cover - server environment has SciPy
        raise RuntimeError("SPOO realism loss requires scipy") from exc
    image = reference.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    height, width, channels = image.shape
    radius = int(window_radius)
    diameter = radius * 2 + 1
    window_size = diameter * diameter
    if height < diameter or width < diameter:
        raise ValueError("Reference is too small for the SPOO matting Laplacian")
    indices = np.arange(height * width).reshape(height, width)
    windows = np.lib.stride_tricks.sliding_window_view(indices, (diameter, diameter))
    windows = windows.reshape(-1, window_size)
    pixels = image.reshape(-1, channels)[windows]
    means = pixels.mean(axis=1, keepdims=True)
    covariance = np.einsum("...ji,...jk->...ik", pixels, pixels) / window_size
    covariance -= np.einsum("...ji,...jk->...ik", means, means)
    inverse = np.linalg.inv(covariance + (float(eps) / window_size) * np.eye(channels))
    centered = pixels - means
    projected = np.einsum("...ij,...jk->...ik", centered, inverse)
    values = np.eye(window_size) - (1.0 / window_size) * (
        1.0 + np.einsum("...ij,...kj->...ik", projected, centered)
    )
    cols = np.tile(windows, window_size).reshape(-1)
    rows = np.repeat(windows, window_size).reshape(-1)
    matrix = scipy.sparse.coo_matrix(
        (values.reshape(-1), (rows, cols)), shape=(height * width, height * width)
    ).astype(np.float32).tocoo()
    sparse_indices = torch.from_numpy(np.vstack((matrix.row, matrix.col))).long()
    sparse_values = torch.from_numpy(matrix.data)
    return torch.sparse_coo_tensor(
        sparse_indices,
        sparse_values,
        matrix.shape,
        device=reference.device,
        dtype=reference.dtype,
    ).coalesce()


def spoo_realism_loss(texture: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    """SPOO ``sum_c x_c^T L x_c`` realism loss."""
    flat = texture.squeeze(0).reshape(texture.shape[1], -1).t()
    transformed = torch.sparse.mm(laplacian, flat)
    return (flat * transformed).sum()


def passive_vo_box_blur(texture: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    """Passive-VO's public 9x9 box blur before perspective warping."""
    kernel = max(1, int(kernel_size))
    if kernel == 1:
        return texture
    if kernel % 2 == 0:
        raise ValueError("Passive-VO blur kernel must be odd")
    padding = kernel // 2
    padded = F.pad(texture, (padding, padding, padding, padding), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1)


def adaptive_alpha(mask_logits: torch.Tensor, temperature: float = 1.0,
                   output_size: int | tuple[int, int] | None = None) -> torch.Tensor:
    alpha = torch.sigmoid(mask_logits / max(float(temperature), 1e-3))
    if output_size is not None and tuple(alpha.shape[-2:]) != (
            (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)):
        size = (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
        alpha = F.interpolate(alpha, size=size, mode="bilinear", align_corners=False)
    return alpha


def adaptive_mask_regularization(mask_logits: torch.Tensor, area_fraction: float,
                                 area_weight: float, binary_weight: float,
                                 tv_weight: float, temperature: float,
                                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    alpha = adaptive_alpha(mask_logits, temperature)
    area = alpha.mean()
    area_loss = (area - float(area_fraction)).square()
    binary = (alpha * (1.0 - alpha)).mean()
    tv = isotropic_total_variation(alpha)
    total = float(area_weight) * area_loss + float(binary_weight) * binary + float(tv_weight) * tv
    return total, {
        "adaptive_mask_area": area,
        "adaptive_mask_area_loss": area_loss,
        "adaptive_mask_binary": binary,
        "adaptive_mask_tv": tv,
    }


def _texture_conversion_components(
    texture: torch.Tensor,
    training: bool,
    repeats: int = 6,
    output_size: int = 4096,
    upsample_factor: int = 8,
) -> tuple[torch.Tensor, int, int]:
    repeats = max(1, int(repeats))
    upsample_factor = max(1, int(upsample_factor))
    if training and texture.shape[-2] != texture.shape[-1]:
        raise ValueError("3D2Fool's 90-degree tile conversion requires a square seed")
    rows = []
    for _ in range(repeats):
        columns = []
        for _ in range(repeats):
            tile = texture
            if training:
                if bool(torch.rand((), device=texture.device) < 0.5):
                    tile = torch.flip(tile, dims=(-1,))
                if bool(torch.rand((), device=texture.device) < 0.5):
                    tile = torch.flip(tile, dims=(-2,))
                # Public code chooses only 0 or +90 degrees, not four angles.
                if bool(torch.rand((), device=texture.device) > 0.5):
                    tile = torch.rot90(tile, 1, dims=(-2, -1))
            columns.append(tile)
        rows.append(torch.cat(columns, dim=-1))
    tiled_coarse = torch.cat(rows, dim=-2)
    size = max(1, int(output_size))
    full_h = tiled_coarse.shape[-2] * upsample_factor
    full_w = tiled_coarse.shape[-1] * upsample_factor
    if full_h < size or full_w < size:
        extra_y = math.ceil(size / full_h)
        extra_x = math.ceil(size / full_w)
        tiled_coarse = tiled_coarse.repeat(1, 1, extra_y, extra_x)
        full_h = tiled_coarse.shape[-2] * upsample_factor
        full_w = tiled_coarse.shape[-1] * upsample_factor
    max_y, max_x = full_h - size, full_w - size
    if training:
        y0 = int(torch.randint(0, max_y + 1, (), device=texture.device).item()) if max_y else 0
        x0 = int(torch.randint(0, max_x + 1, (), device=texture.device).item()) if max_x else 0
    else:
        y0, x0 = max_y // 2, max_x // 2
    return tiled_coarse, y0, x0


def texture_conversion(
    texture: torch.Tensor,
    training: bool,
    repeats: int = 6,
    output_size: int = 4096,
    upsample_factor: int = 8,
) -> torch.Tensor:
    """Memory-efficient, pixel-exact form of public 3D2Fool ``tex_trans``.

    The released code nearest-expands the learnable 128-square parameter by
    8, independently flips and optionally rotates each tile in a 6x6 layout,
    then takes a 4096-square crop from the resulting 6144-square texture.  We
    perform the tile transforms before expansion (the operations commute with
    nearest expansion) and use integer index selection for the crop.  This
    produces the same pixels without first allocating the 6144-square tensor.

    ``training=False`` disables the random tile transforms and uses a centre
    crop.  With the paper dimensions this is the deterministic 4x4 tiling used
    by the released ``tex_trans0`` deployment path.
    """
    upsample_factor = max(1, int(upsample_factor))
    tiled_coarse, y0, x0 = _texture_conversion_components(
        texture, training, repeats, output_size, upsample_factor
    )
    size = max(1, int(output_size))
    y_index = torch.div(
        torch.arange(size, device=texture.device) + y0,
        upsample_factor,
        rounding_mode="floor",
    )
    x_index = torch.div(
        torch.arange(size, device=texture.device) + x0,
        upsample_factor,
        rounding_mode="floor",
    )
    return tiled_coarse.index_select(-2, y_index).index_select(-1, x_index)


def sample_texture_conversion(
    texture: torch.Tensor,
    grid: torch.Tensor,
    training: bool,
    repeats: int = 6,
    output_size: int = 4096,
    upsample_factor: int = 8,
) -> torch.Tensor:
    """Sample official 3D2Fool conversion without a 4096-square autograd map.

    This is exactly ``grid_sample(texture_conversion(...), grid, mode='bilinear',
    align_corners=True)``.  The four bilinear neighbours are looked up directly
    in the transformed coarse map, so backward stores only image-sized tensors.
    """
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError(f"Expected grid [N,H,W,2], got {tuple(grid.shape)}")
    size = max(1, int(output_size))
    upsample_factor = max(1, int(upsample_factor))
    tiled_coarse, y0, x0 = _texture_conversion_components(
        texture, training, repeats, size, upsample_factor
    )
    # align_corners=True maps -1/+1 to the first/last cropped texture pixels.
    crop_x = ((grid[..., 0] + 1.0) * 0.5 * (size - 1)).clamp(0.0, size - 1.0)
    crop_y = ((grid[..., 1] + 1.0) * 0.5 * (size - 1)).clamp(0.0, size - 1.0)
    x_low = crop_x.floor().to(torch.long)
    y_low = crop_y.floor().to(torch.long)
    x_high = (x_low + 1).clamp_max(size - 1)
    y_high = (y_low + 1).clamp_max(size - 1)
    wx = (crop_x - x_low).unsqueeze(1)
    wy = (crop_y - y_low).unsqueeze(1)

    coarse_w = tiled_coarse.shape[-1]
    flat = tiled_coarse.reshape(1, tiled_coarse.shape[1], -1)
    batch = grid.shape[0]
    flat = flat.expand(batch, -1, -1)

    def lookup(yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
        coarse_y = torch.div(yy + y0, upsample_factor, rounding_mode="floor")
        coarse_x = torch.div(xx + x0, upsample_factor, rounding_mode="floor")
        linear = (coarse_y * coarse_w + coarse_x).reshape(batch, 1, -1)
        linear = linear.expand(-1, tiled_coarse.shape[1], -1)
        values = torch.gather(flat, 2, linear)
        return values.reshape(batch, tiled_coarse.shape[1], *grid.shape[1:3])

    v00 = lookup(y_low, x_low)
    v01 = lookup(y_low, x_high)
    v10 = lookup(y_high, x_low)
    v11 = lookup(y_high, x_high)
    top = v00 * (1.0 - wx) + v01 * wx
    bottom = v10 * (1.0 - wx) + v11 * wx
    return top * (1.0 - wy) + bottom * wy
