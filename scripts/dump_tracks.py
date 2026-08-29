"""GPU half of the four-task evaluation: re-render each patch and run the track head.

Split from the scoring half because the two live in different environments -- the
vggt env can load the model, the recons_eval env has evo/open3d. Pose, depth and
point map are already in each run's stored outputs, so only tracking needs a forward
pass here.

The attacked images are not saved by the training runs, so they are re-rendered
through the attack script's own pipeline (not the visualiser's PIL compositing,
which uses a different resampler). The re-rendered pose is dumped alongside so the
scorer can verify it reproduces the run's stored extrinsic; if it does not, the
render差了, and the tracking numbers would describe a different patch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG))
import attack_vggt_geometry_tum10 as A  # noqa: E402

R = VG / "outputs_attack_geometry_aware_tum10"
TUM = Path("/mnt/data/wangqq/recons_eval/data/tum")
GT_DIR = VG / "outputs/tum_gt_point_track"
CLEAN = "tum10_clean_uniform_l3"


def build_args(scene: str, texture_size: int, quad: str,
               manifest: str, clean: str,
               explicit_quad: str = "") -> argparse.Namespace:
    argv = [
        "attack.py", "--tum_root", str(TUM), "--scene_pattern", scene,
        "--output_dir", "/tmp/four_task", "--ckpt", str(VG / "checkpoints/VGGT-1B"),
        "--frame_manifest", manifest,
        "--texture_size", str(texture_size), "--texture_init", "image",
        "--texture_init_image", str(VG / "assets/hazard_textures/mde_attack_warnning.png"),
        "--clean_vggt_output_root", clean,
        "--plane_mode",
        "explicit_world_quad" if explicit_quad else "depth_manual_quad_surface",
        "--manual_quad_xy", quad, "--manual_quad_coordinates", "normalized",
        "--manual_quad_depth_sample_stride", "1", "--manual_quad_fit_shrink", "0.75",
        "--manual_quad_plane_inlier_tolerance", "0.06",
        "--manual_quad_min_inlier_ratio", "0.60",
        "--surface_score_mode", "coverage",
        "--surface_coverage_min", "0.002", "--surface_coverage_max", "0.08",
        "--surface_min_visible_frames", "4", "--surface_min_visibility_ratio", "0.30",
        "--surface_support_abs_tolerance", "0.10",
        "--surface_support_rel_tolerance", "0.01",
        "--surface_min_support_ratio", "0.50",
        "--fused_max_plane_residual", "0.02", "--visibility_depth_margin", "0.08",
        "--seed", "0", "--use_depth_visibility", "--optimize_geometry",
        "--surface_support_check",
    ]
    if explicit_quad:
        argv.append(f"--explicit_quad_world={explicit_quad}")
    saved, sys.argv = sys.argv, argv
    try:
        return A.parse_args()
    finally:
        sys.argv = saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    ap.add_argument("--quad", default="0.3603,0.2939,0.5682,0.3063,0.5644,0.4943,"
                                      "0.3603,0.4795")
    ap.add_argument("--track_iters", type=int, default=4)
    ap.add_argument("--out_dir", default="/tmp/four_task_tracks")
    ap.add_argument("--gt_dir", default=str(GT_DIR),
                    help="帧子集需指向该子集自己的 GT")
    ap.add_argument("--clean", default=str(R / CLEAN),
                    help="帧子集需指向该子集自己的 clean run")
    ap.add_argument("--manifest",
                    default=str(VG / "data/tum_dynamics_10frame_individual_scenes"
                                     "/tum10_frame_manifest.json"))
    ap.add_argument("--explicit_quad_world", default="",
                    help="给定则把平面钉死在该世界四角，而不是从图像 quad 重新拟合")
    cli = ap.parse_args()

    scene = cli.scene
    gt = np.load(Path(cli.gt_dir) / f"{scene}_gt.npz", allow_pickle=True)
    query_np = gt["query_points"]

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None

    for run in [r.strip() for r in cli.runs.split(",") if r.strip()]:
        if not (R / run / scene / "attack_summary.json").exists():
            print(f"{run}: 缺 attack_summary，跳过")
            continue
        tex_npz = R / run / "geometry_patch/geometry_patch_texture.npz"
        has_patch = tex_npz.exists()
        if has_patch:
            tex_arr = np.squeeze(np.load(tex_npz)["texture"])
            tsize = tex_arr.shape[-1] if tex_arr.shape[0] in (3, 4) else tex_arr.shape[0]
        else:
            tex_arr, tsize = None, 64

        args = build_args(scene, int(tsize), cli.quad,
                          cli.manifest, cli.clean,
                          cli.explicit_quad_world)
        if model is None:
            model = A.load_model(args, device)
            for p in model.parameters():
                p.requires_grad_(False)
        dirs = A.list_scene_dirs(Path(args.tum_root), args.scene_pattern)
        manifest = A.load_frame_manifest(args.frame_manifest)
        intr = np.asarray([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]],
                          dtype=np.float64)
        with torch.no_grad():
            item = A.load_tum_sequence(dirs[0], manifest[dirs[0].name], args.gt_name,
                                       intr, args.texture_size, args, device,
                                       model=model, dtype=dtype)
            if has_patch:
                tex = torch.as_tensor(
                    tex_arr if tex_arr.ndim == 3 and tex_arr.shape[0] in (3, 4)
                    else tex_arr.transpose(2, 0, 1),
                    device=device, dtype=torch.float32)
                if float(tex.max()) > 1.5:
                    tex = tex / 255.0
                adv = A.apply_geometry_patch(item["images"], tex[None, :3],
                                             item["grids"], item["masks"],
                                             args, training=False)
            else:
                adv = item["images"]
            preds = A.forward_camera_pose_only(model, adv, dtype, False)
            extr, _ = A.pose_encoding_to_extri_intri(preds["pose_enc"], item["tensor_hw"])
            query = torch.as_tensor(query_np, device=device, dtype=torch.float32)
            tpred = A.forward_tracking_only(model, adv, query, dtype, cli.track_iters)

        np.savez_compressed(
            out_dir / f"{run}__{scene}.npz",
            run=run, scene=scene,
            track=tpred["track"][0].float().cpu().numpy(),
            track_vis=tpred["track_vis"][0].float().cpu().numpy(),
            rerendered_extrinsic=extr[0].float().cpu().numpy(),
            has_patch=has_patch,
        )
        print(f"{run:<24} 已导出 track {tuple(tpred['track'].shape[1:])}  "
              f"贴片={'有' if has_patch else '无'}")

    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
