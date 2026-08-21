"""Is the EOT gradient signal buried in its own noise?

With EOT on, every step renders a different draw, so the gradient at the texture is
a random variable. The optimiser only makes progress on its expectation. If the
per-draw scatter dwarfs that expectation, 1000 steps average to almost nothing --
which matches what was observed: the attack did not move at all (translation term
0.93, i.e. still at the clean distance from the target) rather than converging
slowly.

Draws the gradient repeatedly under independent EOT samples and reports
||mean|| against the per-draw scatter, plus the same quantity with EOT off and
with the photometric and geometric parts enabled separately, so the cause is
attributable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG))
import attack_vggt_geometry_tum10 as A  # noqa: E402


def build(scene: str, axis: int) -> argparse.Namespace:
    argv = [
        "attack.py", "--tum_root", "/mnt/data/wangqq/recons_eval/data/tum",
        "--scene_pattern", scene, "--output_dir", "/tmp/eot_snr",
        "--frame_manifest", str(VG / "data/tum_dynamics_10frame_individual_scenes"
                                    "/tum10_frame_manifest.json"),
        "--ckpt", str(VG / "checkpoints/VGGT-1B"),
        "--texture_size", "128", "--texture_init", "image",
        "--texture_init_image", str(VG / "assets/hazard_textures/mde_attack_warnning.png"),
        "--iterations", "1", "--attack_loss", "pose_piecewise_gauge_targeted",
        "--piecewise_gauge_family", "orthogonal_mode",
        "--piecewise_gauge_magnitude", "3.0",
        "--orthogonal_mode_order", "2", "--orthogonal_mode_axis", str(axis),
        "--pose_rotation_weight", "5.0", "--pose_translation_weight", "1.0",
        "--plane_mode", "fused_depth_surface",
        "--plane_width", "0.30", "--plane_height", "0.20",
        "--clean_vggt_output_root",
        str(VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"),
        "--surface_score_mode", "coverage",
        "--surface_coverage_min", "0.005", "--surface_coverage_max", "0.06",
        "--surface_min_visible_frames", "8", "--surface_min_visibility_ratio", "0.80",
        "--surface_min_support_ratio", "0.50",
        "--surface_support_abs_tolerance", "0.10",
        "--surface_support_rel_tolerance", "0.06",
        "--fused_max_plane_residual", "0.12", "--visibility_depth_margin", "0.08",
        "--seed", "0", "--use_depth_visibility", "--optimize_geometry",
        "--surface_support_check", "--physical_eot",
    ]
    saved, sys.argv = sys.argv, argv
    try:
        return A.parse_args()
    finally:
        sys.argv = saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_xyz")
    ap.add_argument("--axis", type=int, default=0)
    ap.add_argument("--draws", type=int, default=6)
    cli = ap.parse_args()

    args = build(cli.scene, cli.axis)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = A.load_model(args, device)
    for p in model.parameters():
        p.requires_grad_(False)

    dirs = A.list_scene_dirs(Path(args.tum_root), args.scene_pattern)
    manifest = A.load_frame_manifest(args.frame_manifest)
    intr = np.asarray([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]],
                      dtype=np.float64)
    seq = dirs[0]
    with torch.no_grad():
        item = A.load_tum_sequence(seq, manifest[seq.name], args.gt_name, intr,
                                   args.texture_size, args, device, model=model, dtype=dtype)
    target = item["pose_piecewise_rel"]

    def grad_once(a: argparse.Namespace) -> torch.Tensor:
        tex = A.initialize_texture(a, device, requires_grad=True)
        adv = A.apply_geometry_patch(item["images"], tex, item["grids"], item["masks"],
                                     a, training=True)
        preds = A.forward_camera_pose_only(model, adv, dtype, a.activation_checkpoint)
        rel = A.pose_predictions_to_relative_c2w(preds, item["tensor_hw"])
        loss, _ = A.pose_aligned_residual_mse(rel, target, a)
        return torch.autograd.grad(loss, tex)[0].detach().flatten()

    def variant(name: str, clean_mean: torch.Tensor | None = None, **over):
        a = argparse.Namespace(**vars(args))
        for k, v in over.items():
            setattr(a, k, v)
        gs = torch.stack([grad_once(a) for _ in range(cli.draws)])
        mean = gs.mean(0)
        scatter = (gs - mean).norm(dim=1).mean()
        snr = float(mean.norm()) / max(float(scatter), 1e-30)
        cosine = 1.0
        if clean_mean is not None:
            cosine = float(torch.nn.functional.cosine_similarity(
                mean[None], clean_mean[None]).item())
        print(f"{name:<34}{float(mean.norm()):>13.4e}{float(scatter):>13.4e}"
              f"{snr:>9.3f}{cosine:>12.3f}")
        return mean

    print(f"scene {seq.name}   {cli.draws} draws per variant\n")
    print(f"{'EOT variant':<34}{'||mean grad||':>13}{'scatter':>13}{'SNR':>9}"
          f"{'cos(clean)':>12}")
    print("-" * 81)
    clean = variant("off", physical_eot=False, _eot_strength=0.0)
    variant("photometric only", clean, _eot_strength=1.0,
            eot_geo_translate=0.0, eot_geo_scale=0.0,
            eot_geo_rotate_degrees=0.0, eot_geo_perspective=0.0, eot_noise_std=0.0)
    variant("patch noise only", clean, _eot_strength=1.0,
            eot_brightness=0.0, eot_contrast=0.0, eot_gamma=0.0,
            eot_geo_translate=0.0, eot_geo_scale=0.0, eot_geo_rotate_degrees=0.0,
            eot_geo_perspective=0.0)
    variant("geometric only", clean, _eot_strength=1.0,
            eot_brightness=0.0, eot_contrast=0.0, eot_gamma=0.0,
            eot_noise_std=0.0)
    variant("full, ramp strength 0.25", clean, _eot_strength=0.25)
    variant("full, ramp strength 0.50", clean, _eot_strength=0.50)
    variant("full, ramp strength 1.00", clean, _eot_strength=1.00)

    print("\nSNR = ||E[grad]|| / E[||grad - E[grad]||]; below ~0.1 the optimiser is")
    print("averaging mostly noise and 1000 steps will not accumulate much signal.")
    print("cos(clean) shows whether the expected robust gradient still points along the")
    print("solvable clean objective; the warm-up/ramp is justified only if early strengths do.")


if __name__ == "__main__":
    main()
