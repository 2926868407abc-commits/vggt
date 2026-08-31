"""EOT must vary across frames and stay near identity.

The previous implementation drew one scalar per photometric factor and applied it
before the texture was expanded across frames, so all ten frames of a sequence saw
exactly the same perturbation, and the sampling grid was never perturbed at all.
That is not an expectation over transformations, and on a large patch it let the
attack lean on the augmentation: sitting_halfsphere trained to a translation term
of 0.7366 and scored 0.2671 on the clean render, a 63.7% gap that closed to 0.0%
with PHYSICAL_EOT=0.

    /mnt/data/wangqq/conda_envs/vggt/bin/python3 tests/test_eot.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack_vggt_geometry_tum10 import (  # noqa: E402
    apply_geometry_patch, jitter_sampling_grids, prepare_texture_for_render,
    scheduled_eot_strength,
)

N, C, H, W = 10, 3, 32, 32


def make_args(**over) -> argparse.Namespace:
    base = dict(print_min=0.0, print_max=1.0, physical_eot=True,
                eot_brightness=0.15, eot_contrast=0.15, eot_gamma=0.10,
                eot_noise_std=0.01, eot_geo_translate=0.02, eot_geo_scale=0.03,
                eot_geo_rotate_degrees=2.0, eot_geo_perspective=0.01,
                _eot_strength=1.0)
    base.update(over)
    return argparse.Namespace(**base)


def test_photometric_factors_differ_across_frames():
    torch.manual_seed(0)
    tex = torch.rand(1, C, H, W)
    out = prepare_texture_for_render(tex, make_args(), training=True, n_frames=N)
    assert out.shape[0] == N, f"expected {N} frames, got {out.shape[0]}"
    means = out.mean(dim=(1, 2, 3))
    spread = float(means.max() - means.min())
    assert spread > 1e-3, (
        f"all {N} frames got the same photometric perturbation (spread {spread:.2e}); "
        f"this is the bug the per-frame sampling was meant to fix")


def test_eval_render_is_deterministic_and_unperturbed():
    tex = torch.rand(1, C, H, W)
    a = prepare_texture_for_render(tex, make_args(), training=False, n_frames=N)
    b = prepare_texture_for_render(tex, make_args(), training=False, n_frames=N)
    assert torch.equal(a, b), "the evaluation render must not be stochastic"
    assert torch.allclose(a[0], tex.clamp(0, 1)[0]), "evaluation must not perturb"
    for i in range(1, N):
        assert torch.equal(a[0], a[i]), "frames must be identical without EOT"


def test_grid_jitter_varies_per_frame_and_stays_small():
    torch.manual_seed(0)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing="ij")
    grids = torch.stack([xs, ys], dim=-1)[None].expand(N, -1, -1, -1).contiguous()
    out = jitter_sampling_grids(grids, make_args())
    assert out.shape == grids.shape

    shifts = (out - grids).flatten(1).norm(dim=1) / grids.flatten(1).norm(dim=1)
    assert float(shifts.max() - shifts.min()) > 1e-4, (
        "every frame received the same geometric jitter")
    assert float(shifts.max()) < 0.25, (
        f"jitter is too large ({shifts.max():.3f} relative); it should be a small "
        f"placement error, not a different patch location")


def test_grid_jitter_is_identity_when_disabled():
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing="ij")
    grids = torch.stack([xs, ys], dim=-1)[None].expand(N, -1, -1, -1).contiguous()
    args = make_args(eot_geo_translate=0.0, eot_geo_scale=0.0,
                     eot_geo_rotate_degrees=0.0, eot_geo_perspective=0.0)
    out = jitter_sampling_grids(grids, args)
    assert torch.allclose(out, grids, atol=1e-6), "zero spread must leave the grid alone"


def test_patch_application_still_matches_without_eot():
    """Turning EOT off must reproduce the plain composite exactly."""
    torch.manual_seed(0)
    images = torch.rand(N, C, H, W)
    tex = torch.rand(1, C, H, W)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing="ij")
    grids = torch.stack([xs, ys], dim=-1)[None].expand(N, -1, -1, -1).contiguous()
    masks = (torch.rand(N, 1, H, W) > 0.5).float()
    args = make_args(physical_eot=False)
    a = apply_geometry_patch(images, tex, grids, masks, args, training=True)
    b = apply_geometry_patch(images, tex, grids, masks, args, training=False)
    assert torch.allclose(a, b), "with EOT off, training and eval renders must agree"


def test_gradients_survive_the_jitter():
    torch.manual_seed(0)
    images = torch.rand(N, C, H, W)
    tex = torch.rand(1, C, H, W, requires_grad=True)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing="ij")
    grids = torch.stack([xs, ys], dim=-1)[None].expand(N, -1, -1, -1).contiguous()
    masks = torch.ones(N, 1, H, W)
    out = apply_geometry_patch(images, tex, grids, masks, make_args(), training=True)
    out.sum().backward()
    assert tex.grad is not None and torch.isfinite(tex.grad).all()
    assert float(tex.grad.norm()) > 0


def test_eot_does_not_modify_pixels_outside_patch():
    torch.manual_seed(0)
    images = torch.rand(N, C, H, W)
    tex = torch.rand(1, C, H, W)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing="ij")
    grids = torch.stack([xs, ys], dim=-1)[None].expand(N, -1, -1, -1).contiguous()
    masks = torch.zeros(N, 1, H, W)
    masks[:, :, 8:24, 8:24] = 1.0
    out = apply_geometry_patch(images, tex, grids, masks, make_args(), training=True)
    outside = (masks == 0).expand_as(images)
    assert torch.equal(out[outside], images[outside]), (
        "optimisation EOT must not inject uncontrollable noise outside the patch")


def test_zero_eot_strength_matches_plain_render():
    torch.manual_seed(0)
    images = torch.rand(N, C, H, W)
    tex = torch.rand(1, C, H, W)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing="ij")
    grids = torch.stack([xs, ys], dim=-1)[None].expand(N, -1, -1, -1).contiguous()
    masks = (torch.rand(N, 1, H, W) > 0.5).float()
    warm = apply_geometry_patch(images, tex, grids, masks,
                                make_args(_eot_strength=0.0), training=True)
    plain = apply_geometry_patch(images, tex, grids, masks,
                                 make_args(physical_eot=False), training=True)
    assert torch.allclose(warm, plain), "zero EOT strength must be the clean objective"


def test_eot_schedule_has_clean_warmup_and_reaches_one():
    values = [scheduled_eot_strength(i, 101, 0.25) for i in range(101)]
    assert values[0] == 0.0 and values[25] == 0.0
    assert 0.49 < values[62] < 0.51
    assert values[-1] == 1.0
    assert all(a <= b for a, b in zip(values, values[1:])), "schedule must be monotone"


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
