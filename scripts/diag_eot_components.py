"""Which part of the EOT jitter destroys the gradient, and does averaging fix it?

diag_eot_snr.py established that geometric jitter collapses the SNR (119 -> 0.52)
but ran all four geometric components together, so it could not say which one.
This turns them on one at a time.

It also measures the k-sample average empirically rather than trusting the sqrt(k)
argument: averaging k iid draws should shrink the scatter by sqrt(k), but that
holds only if the draws really are independent, which is exactly what a per-frame
homography on a shared texture might violate.

Configured for the current setup -- monitor placement, sitting_halfsphere,
pose_aligned_residual_mse -- not the automatic placement the earlier diagnostic used.
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

QUAD = "0.3603,0.2939,0.5682,0.3063,0.5644,0.4943,0.3603,0.4795"


def build(scene: str) -> argparse.Namespace:
    argv = [
        "attack.py", "--tum_root", "/mnt/data/wangqq/recons_eval/data/tum",
        "--scene_pattern", scene, "--output_dir", "/tmp/eot_comp",
        "--frame_manifest", str(VG / "data/tum_dynamics_10frame_individual_scenes"
                                    "/tum10_frame_manifest.json"),
        "--ckpt", str(VG / "checkpoints/VGGT-1B"),
        "--texture_size", "128", "--texture_init", "image",
        "--texture_init_image", str(VG / "assets/hazard_textures/mde_attack_warnning.png"),
        "--iterations", "1", "--attack_loss", "pose_aligned_residual_mse",
        "--pose_rotation_weight", "5.0", "--pose_translation_weight", "1.0",
        "--plane_mode", "depth_manual_quad_surface",
        "--manual_quad_xy", QUAD, "--manual_quad_coordinates", "normalized",
        "--manual_quad_depth_sample_stride", "1", "--manual_quad_fit_shrink", "0.75",
        "--manual_quad_plane_inlier_tolerance", "0.06",
        "--manual_quad_min_inlier_ratio", "0.60",
        "--clean_vggt_output_root",
        str(VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"),
        "--surface_score_mode", "coverage",
        "--surface_coverage_min", "0.002", "--surface_coverage_max", "0.08",
        "--surface_min_visible_frames", "4", "--surface_min_visibility_ratio", "0.30",
        "--surface_min_support_ratio", "0.50",
        "--surface_support_abs_tolerance", "0.10",
        "--surface_support_rel_tolerance", "0.01",
        "--fused_max_plane_residual", "0.02", "--visibility_depth_margin", "0.08",
        "--seed", "0", "--use_depth_visibility", "--optimize_geometry",
        "--surface_support_check", "--physical_eot",
    ]
    saved, sys.argv = sys.argv, argv
    try:
        return A.parse_args()
    finally:
        sys.argv = saved


GEO_OFF = dict(eot_geo_translate=0.0, eot_geo_scale=0.0,
               eot_geo_rotate_degrees=0.0, eot_geo_perspective=0.0)
PHOTO_OFF = dict(eot_brightness=0.0, eot_contrast=0.0, eot_gamma=0.0, eot_noise_std=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--pool_draws", type=int, default=24,
                    help="independent draws used for the k-average sweep")
    cli = ap.parse_args()

    args = build(cli.scene)
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
    def grad_once(a: argparse.Namespace) -> torch.Tensor:
        """Exactly one training step's texture gradient, via the training entry point.

        Going through attack_objective_loss rather than calling the loss directly
        keeps this measuring whatever the run would actually optimise, including
        the sign convention, instead of a re-derivation that can drift from it.
        """
        tex = A.initialize_texture(a, device, requires_grad=True)
        adv = A.apply_geometry_patch(item["images"], tex, item["grids"], item["masks"],
                                     a, training=True)
        loss, _, _ = A.attack_objective_loss(model, adv, item, a, dtype)
        return torch.autograd.grad(loss, tex)[0].detach().flatten().float()

    def make(**over) -> argparse.Namespace:
        a = argparse.Namespace(**vars(args))
        for k, v in over.items():
            setattr(a, k, v)
        return a

    def stats(gs: torch.Tensor):
        mean = gs.mean(0)
        scatter = float((gs - mean).norm(dim=1).mean())
        return mean, scatter, float(mean.norm()) / max(scatter, 1e-30)

    def variant(name, clean_mean=None, **over):
        gs = torch.stack([grad_once(make(**over)) for _ in range(cli.draws)])
        mean, scatter, snr = stats(gs)
        cos = 1.0 if clean_mean is None else float(
            torch.nn.functional.cosine_similarity(mean[None], clean_mean[None]).item())
        print(f"{name:<30}{float(mean.norm()):>13.4e}{scatter:>13.4e}{snr:>9.3f}{cos:>12.3f}")
        return mean

    print(f"scene {seq.name}   loss pose_aligned_residual_mse   {cli.draws} draws\n")
    print(f"{'EOT 配置':<30}{'||E[grad]||':>13}{'scatter':>13}{'SNR':>9}{'cos(无EOT)':>12}")
    print("-" * 77)

    clean = variant("关闭 EOT", physical_eot=False, _eot_strength=0.0)
    variant("仅光度", clean, _eot_strength=1.0, **GEO_OFF)

    # one geometric knob at a time, everything else off
    for label, key in (("仅几何-平移", "eot_geo_translate"),
                       ("仅几何-缩放", "eot_geo_scale"),
                       ("仅几何-旋转", "eot_geo_rotate_degrees"),
                       ("仅几何-透视", "eot_geo_perspective")):
        over = dict(PHOTO_OFF)
        over.update(GEO_OFF)
        over[key] = getattr(args, key)          # restore just this one
        variant(label, clean, _eot_strength=1.0, **over)

    variant("几何全开", clean, _eot_strength=1.0, **PHOTO_OFF)
    full = variant("全开（当前配置）", clean, _eot_strength=1.0)

    # does averaging k draws recover the signal, empirically?
    print(f"\n\n=== k 次平均能否救回信噪比（全开配置，{cli.pool_draws} 次独立采样）===")
    print("预期：独立采样时 scatter 按 1/sqrt(k) 缩，SNR 按 sqrt(k) 涨")
    pool = torch.stack([grad_once(make(_eot_strength=1.0)) for _ in range(cli.pool_draws)])
    base_mean, base_scatter, _ = stats(pool)
    print(f"{'k':<6}{'块数':>6}{'||E[grad]||':>14}{'scatter':>13}{'SNR':>9}"
          f"{'相对k=1':>10}{'sqrt(k)':>9}")
    print("-" * 67)
    snr1 = None
    for k in (1, 2, 4, 8):
        m = cli.pool_draws // k
        if m < 2:
            continue
        blocks = torch.stack([pool[i * k:(i + 1) * k].mean(0) for i in range(m)])
        _, sc, snr = stats(blocks)
        if snr1 is None:
            snr1 = snr
        print(f"{k:<6}{m:>6}{float(blocks.mean(0).norm()):>14.4e}{sc:>13.4e}"
              f"{snr:>9.3f}{snr / snr1:>10.2f}x{float(np.sqrt(k)):>9.2f}")

    print("\n注：SNR = ||E[grad]|| / E[||grad - E[grad]||]。低于 ~0.1 时优化器基本原地踏步。")
    print("cos(无EOT) 是该配置的平均梯度与无 EOT 梯度的夹角余弦：接近 1 说明方向没变，")
    print("只是被噪声淹没；明显小于 1 说明 EOT 把攻击方向本身改了。")


if __name__ == "__main__":
    main()
