"""Are the four joint-loss terms on comparable scales, or is one drowning the others?

Q4 produced near-zero ATE (3.7% and 5.3%, against clean baselines of 3.9% and
3.2%) while the consistency filters read cleaner than clean. That pattern says the
patch barely moved, not that the heads fought each other. The suspicion is the
same failure that made the pairwise loss look useless: three terms added at weight
1/1/1 whose gradients differ by orders of magnitude, so the optimiser only ever
served the loudest one -- and here the depth target for a pure-translation gauge
is "do not change", which would actively hold the prediction still.

Measures each term's value and its gradient at the texture, separately, in one
replicated training iteration.
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


def build_args(scene: str, axis: int, texture_size: int = 128,
               quad: str | None = None) -> argparse.Namespace:
    """Build one training-step config.

    Defaults reproduce the original 128x128 + automatic-surface run this script was
    written for. Pass texture_size / quad to point it at the current setup instead:
    the head weights it derives only apply to the configuration it measured, and
    the 128-era numbers do not transfer (the placement and the texture resolution
    both changed).
    """
    argv = [
        "attack.py",
        "--tum_root", "/mnt/data/wangqq/recons_eval/data/tum",
        "--scene_pattern", scene,
        "--output_dir", "/tmp/joint_diag_out",
        "--frame_manifest", str(VG / "data/tum_dynamics_10frame_individual_scenes"
                                    "/tum10_frame_manifest.json"),
        "--ckpt", str(VG / "checkpoints/VGGT-1B"),
        "--texture_size", str(texture_size), "--texture_init", "image",
        "--texture_init_image", str(VG / "assets/hazard_textures/mde_attack_warnning.png"),
        "--iterations", "1", "--inner_loop", "1", "--scenes_per_iteration", "1",
        "--patch_lr", "0.002", "--feature_layer", "aggregator_final",
        "--attack_loss", "geometry_joint_gauge_targeted",
        "--piecewise_gauge_family", "orthogonal_mode",
        "--piecewise_gauge_magnitude", "3.0",
        "--orthogonal_mode_order", "2", "--orthogonal_mode_axis", str(axis),
        "--pose_rotation_weight", "5.0", "--pose_translation_weight", "1.0",
        "--joint_depth_conf_weight", "1.0", "--joint_point_conf_weight", "1.0",
        "--joint_track_weight", "1.0", "--joint_track_grid_rows", "4",
        "--joint_track_grid_cols", "6", "--joint_track_iters", "2",
        "--clean_vggt_output_root",
        str(VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"),
        "--surface_min_support_ratio", "0.50",
        "--surface_support_abs_tolerance", "0.10",
        "--visibility_depth_margin", "0.08",
        "--tv_weight", "0.001", "--printability_weight", "0.001",
        "--printable_color_levels", "2", "--low_frequency_weight", "0.0",
        "--natural_reference_weight", "0.05",
        "--natural_reference_image", str(VG / "assets/hazard_textures/mde_attack_warnning.png"),
        "--seed", "0",
        "--use_depth_visibility", "--optimize_geometry", "--surface_support_check",
    ]
    if quad:
        argv += [
            "--plane_mode", "depth_manual_quad_surface",
            "--manual_quad_xy", quad, "--manual_quad_coordinates", "normalized",
            "--manual_quad_depth_sample_stride", "1", "--manual_quad_fit_shrink", "0.75",
            "--manual_quad_plane_inlier_tolerance", "0.06",
            "--manual_quad_min_inlier_ratio", "0.60",
            "--surface_support_rel_tolerance", "0.01",
            "--fused_max_plane_residual", "0.02",
            "--surface_score_mode", "coverage",
            "--surface_coverage_min", "0.002", "--surface_coverage_max", "0.08",
            "--surface_min_visible_frames", "4", "--surface_min_visibility_ratio", "0.30",
        ]
    else:
        argv += [
            "--plane_mode", "fused_depth_surface",
            "--plane_width", "0.30", "--plane_height", "0.20",
            "--surface_score_mode", "coverage",
            "--surface_coverage_min", "0.005", "--surface_coverage_max", "0.06",
            "--surface_min_visible_frames", "8", "--surface_min_visibility_ratio", "0.80",
            "--surface_support_rel_tolerance", "0.06",
            "--fused_max_plane_residual", "0.12",
        ]
    saved, sys.argv = sys.argv, argv
    try:
        return A.parse_args()
    finally:
        sys.argv = saved


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_xyz")
    p.add_argument("--axis", type=int, default=0)
    p.add_argument("--texture_size", type=int, default=128)
    p.add_argument("--quad", default=None,
                   help="normalised MANUAL_QUAD_XY; switches placement to the "
                        "hand-marked monitor quad. Omit to keep the original "
                        "automatic-surface configuration.")
    cli = p.parse_args()

    args = build_args(cli.scene, cli.axis, cli.texture_size, cli.quad)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = A.load_model(args, device)
    for prm in model.parameters():
        prm.requires_grad_(False)

    scene_dirs = A.list_scene_dirs(Path(args.tum_root), args.scene_pattern)
    manifest = A.load_frame_manifest(args.frame_manifest)
    intr = np.asarray([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]],
                      dtype=np.float64)
    seq = scene_dirs[0]
    with torch.no_grad():
        item = A.load_tum_sequence(seq, manifest[seq.name], args.gt_name, intr,
                                   args.texture_size, args, device, model=model, dtype=dtype)
    targets = item["joint_gauge_targets"]
    print(f"scene {seq.name}  coverage {item['geometry']['mask_coverage_mean']:.5f}\n")

    # how far the target actually asks each head to move, relative to clean
    with np.load(Path(args.clean_vggt_output_root) / seq.name / "vggt_outputs.npz") as d:
        cd = torch.from_numpy(d["depth"].astype(np.float32))
        cp = torch.from_numpy(d["point_map"].astype(np.float32))
    if cd.ndim == 4 and cd.shape[-1] == 1:
        cd = cd[..., 0]
    dd = (targets["depth"].cpu() - cd).abs().mean() / cd.abs().mean()
    dp = (targets["points"].cpu() - cp).norm(dim=-1).mean() / cp.norm(dim=-1).mean()
    print(f"target asks depth to move {dd*100:.2f}% and points to move {dp*100:.2f}% "
          f"relative to clean\n")

    texture = A.initialize_texture(args, device, requires_grad=True)
    adv = A.apply_geometry_patch(item["images"], texture, item["grids"], item["masks"],
                                 args, training=True)
    track_target = item["joint_track_targets"]
    preds = A.forward_all_geometry_heads(
        model, adv, dtype, query_points=track_target["query_points"],
        track_iters=args.joint_track_iters)

    pred_rel = A.pose_predictions_to_relative_c2w(preds, item["tensor_hw"])
    pose_term, _ = A.pose_aligned_residual_mse(pred_rel, targets["pose_rel"], args)

    dpred = preds["depth"]
    if dpred.ndim == 5:
        dpred = dpred[0]
    if dpred.shape[-1] == 1:
        dpred = dpred[..., 0]
    depth_term = A.scale_invariant_depth_loss(dpred.float(), targets["depth"])

    ppred = preds["world_points"]
    if ppred.ndim == 5:
        ppred = ppred[0]
    point_term = A.aligned_point_residual(ppred.float(), targets["points"],
                                          args.joint_point_stride)

    depth_conf_term = A.confidence_preservation_loss(
        A.dense_scalar_head(preds["depth_conf"]),
        A.dense_scalar_head(targets["depth_conf"]))
    point_conf_term = A.confidence_preservation_loss(
        A.dense_scalar_head(preds["world_points_conf"]),
        A.dense_scalar_head(targets["point_conf"]))

    track_term, track_parts = A.track_consistency_loss(
        preds["track"], track_target["track"], preds["track_vis"], track_target["vis"],
        preds.get("track_conf"), track_target.get("conf"), item["tensor_hw"],
        args.joint_track_min_visibility, args.joint_track_visibility_weight,
        args.joint_track_confidence_weight)

    print(f"{'term':<10}{'value':>12}{'|grad|':>14}{'share of total grad':>22}")
    print("-" * 58)
    grads = {}
    terms = (("pose", pose_term), ("depth", depth_term), ("point", point_term),
             ("depth_conf", depth_conf_term), ("point_conf", point_conf_term),
             ("track", track_term))
    for name, term in terms:
        g = torch.autograd.grad(term, texture, retain_graph=True)[0]
        grads[name] = float(g.norm())
    tot = sum(grads.values())
    for name, term in terms:
        print(f"{name:<10}{float(term):>12.6f}{grads[name]:>14.4e}"
              f"{grads[name]/tot*100:>21.1f}%")
    print(f"  track pieces: coord={float(track_parts['track_coord']):.6g} "
          f"vis={float(track_parts['track_vis']):.6g} "
          f"conf={float(track_parts['track_conf']):.6g}")

    print(f"\nratios: pose/depth {grads['pose']/max(grads['depth'],1e-30):.3g}   "
          f"pose/point {grads['pose']/max(grads['point'],1e-30):.3g}")
    print("\nWeights that would equalise all gradients, relative to pose=1:")
    for name in ("depth", "point", "depth_conf", "point_conf", "track"):
        print(f"  JOINT_{name.upper()}_WEIGHT = "
              f"{grads['pose']/max(grads[name],1e-30):.4g}")


if __name__ == "__main__":
    main()
