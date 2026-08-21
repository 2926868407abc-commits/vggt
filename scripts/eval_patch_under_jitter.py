"""How much of the trained attack survives realistic placement error?

The 79.45 deg result was trained with EOT off, i.e. at one exact alignment. The
component diagnostic showed the EOT-robust gradient direction is nearly orthogonal
to the non-robust one (cos 0.097), which predicts the trained patch is tuned to
that exact alignment and should degrade under jitter. This measures it directly:
same texture, same placement, jitter applied at test time only.

If it collapses, "physical patch attack" is not currently supported by the result,
and EOT training is required rather than optional.
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

# This half runs in the vggt env, which has the model but not evo. It writes the
# trajectories; score_patch_under_jitter.py reads them in the recons_eval env.
# Same split the training pipeline already uses.

QUAD = "0.3603,0.2939,0.5682,0.3063,0.5644,0.4943,0.3603,0.4795"
RECONS = Path("/mnt/data/wangqq/recons_eval")
TUM = RECONS / "data/tum"


def build(scene: str) -> argparse.Namespace:
    argv = [
        "attack.py", "--tum_root", str(TUM),
        "--scene_pattern", scene, "--output_dir", "/tmp/jitter_eval",
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    ap.add_argument("--run", default="mon_monhalf_al_1000")
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--out", default="/tmp/jitter_eval/trajectories.npz")
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

    tex_path = (VG / "outputs_attack_geometry_aware_tum10" / cli.run
                / "geometry_patch/geometry_patch_texture.npz")
    d = np.load(tex_path)
    key = [k for k in d.files if "tex" in k.lower()] or list(d.files)
    tex = torch.as_tensor(np.squeeze(d[key[0]]), device=device, dtype=torch.float32)
    if tex.ndim == 3 and tex.shape[-1] in (3, 4):
        tex = tex.permute(2, 0, 1)
    if tex.max() > 1.5:
        tex = tex / 255.0
    tex = tex[:3].unsqueeze(0)
    print(f"texture {tuple(tex.shape)} from {cli.run}")

    def measure(a: argparse.Namespace, training: bool) -> np.ndarray:
        with torch.no_grad():
            adv = A.apply_geometry_patch(item["images"], tex, item["grids"],
                                         item["masks"], a, training=training)
            preds = A.forward_camera_pose_only(model, adv, dtype, a.activation_checkpoint)
        # same path the saved npz takes: pose_enc -> w2c 3x4 -> c2w, unnormalised,
        # so this is the trajectory the production evaluator would have scored
        extrinsic, _ = A.pose_encoding_to_extri_intri(preds["pose_enc"], item["tensor_hw"])
        c2w = A.w2c_3x4_to_c2w(extrinsic).detach().float().cpu().numpy()
        if c2w.ndim == 4:
            c2w = c2w[0]
        return np.asarray(c2w, dtype=np.float64)

    def no_eot(**over):
        a = argparse.Namespace(**vars(args))
        a.physical_eot = False
        a._eot_strength = 0.0
        for k, v in over.items():
            setattr(a, k, v)
        return a

    def with_eot(**over):
        a = argparse.Namespace(**vars(args))
        a.physical_eot = True
        a._eot_strength = 1.0
        for k, v in over.items():
            setattr(a, k, v)
        return a

    PHOTO_OFF = dict(eot_brightness=0.0, eot_contrast=0.0, eot_gamma=0.0, eot_noise_std=0.0)
    GEO_OFF = dict(eot_geo_translate=0.0, eot_geo_scale=0.0,
                   eot_geo_rotate_degrees=0.0, eot_geo_perspective=0.0)

    out = {"nojitter": measure(no_eot(), training=False)[None],
           "frame_indices": np.asarray(item["frame_indices"], dtype=np.int64)}
    for label, over in (("geo", PHOTO_OFF), ("photo", GEO_OFF), ("full", {})):
        out[label] = np.stack([measure(with_eot(**over), training=True)
                               for _ in range(cli.draws)])
        print(f"  drew {cli.draws} for {label}")

    dst = Path(cli.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, scene=seq.name, run=cli.run, **out)
    print(f"\nwrote {dst}\n现在用 recons_eval 环境跑 score_patch_under_jitter.py 打分")


if __name__ == "__main__":
    main()
