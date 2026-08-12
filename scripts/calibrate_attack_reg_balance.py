"""Calibrate the regularizer weights so different attack losses face the same pull.

Why this exists
---------------
The optimiser sees  objective = -attack_loss + regularization_total.  The
regulariser's gradient w.r.t. the texture is a constant of the initial texture,
while the attack gradient scales with whatever magnitude that particular loss
happens to have.  Measured on sitting_xyz at the shared init texture, the ratio
|g_attack| / |g_reg| spanned 1.40 (pose_pairwise_relative_mse) to 93.3
(pose_gt_untargeted) -- a factor of 67.  With one fixed TV/printability weight
that means every loss is compared under a different effective constraint, and a
loss whose ratio approaches 1 converges to a genuine stationary point of
(-attack + reg) instead of to its own optimum.  That is why the pairwise loss sat
at the same place for lr 0.002, 0.02 and 0.10: lr changes how fast you reach a
stationary point, not where it is.

What it does
------------
Replicates exactly one training iteration of `train_geometry_patch` at the
initial texture -- same `load_tum_sequence`, same `apply_geometry_patch(
training=True)`, same `attack_objective_loss` -- and measures
`d(attack)/d(texture)` and `d(reg)/d(texture)` separately.  It then reports the
regulariser scale factor that puts every loss at a common target ratio.

Because objective = -attack + reg, scaling the attack loss by k is identical to
dividing every regulariser weight by k, and those weights are already env vars.
So no change to the attack code is needed: apply the printed weights.

Caveat worth stating in any writeup: this matches the *optimisation dynamics* at
initialisation, not the physical realism of the final patch.  Losses calibrated
this way end up under different absolute regularisation, so always report the
final TV / printability of the resulting texture alongside the attack numbers so
a reader can see whether the realism constraint actually bound comparably.
Matching the raw weight instead (same physical constraint, unequal dynamics) is
the other defensible protocol; it is what produced the pairwise result above.

Usage
-----
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0 /mnt/data/wangqq/conda_envs/vggt/bin/python3 \
        scripts/calibrate_attack_reg_balance.py \
        --scene rgbd_dataset_freiburg3_sitting_xyz \
        --losses pose_gt_untargeted,pose_aligned_residual_mse \
        --target_ratio 40

The geometry flags below must match the ablation being calibrated, otherwise the
measured gradients belong to a different patch.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

VGGT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VGGT_ROOT))

import attack_vggt_geometry_tum10 as A  # noqa: E402

DEFAULT_LOSSES = (
    "pose_gt_untargeted",
    "pose_scale_invariant_mse",
    "pose_pairwise_relative_mse",
    "pose_aligned_residual_mse",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_xyz")
    p.add_argument("--losses", default=",".join(DEFAULT_LOSSES))
    p.add_argument("--target_ratio", type=float, default=None,
                   help="common |g_attack|/|g_reg| to solve for; default = the max "
                        "ratio observed, so no loss is regularised harder than it is now")
    p.add_argument("--tum_root", default="/mnt/data/wangqq/recons_eval/data/tum")
    p.add_argument("--frame_manifest",
                   default=str(VGGT_ROOT / "data/tum_dynamics_10frame_individual_scenes"
                                           "/tum10_frame_manifest.json"))
    p.add_argument("--ckpt", default=str(VGGT_ROOT / "checkpoints/VGGT-1B"))
    p.add_argument("--clean_vggt_output_root",
                   default=str(VGGT_ROOT / "outputs_attack_geometry_aware_tum10"
                                           "/tum10_clean_uniform_l3"))
    p.add_argument("--texture_init_image",
                   default=str(VGGT_ROOT / "assets/hazard_textures/mde_attack_warnning.png"))
    # geometry / regulariser config -- must mirror the ablation
    p.add_argument("--plane_mode", default="fused_depth_surface")
    p.add_argument("--plane_width", type=float, default=0.30)
    p.add_argument("--plane_height", type=float, default=0.20)
    p.add_argument("--tv_weight", type=float, default=0.001)
    p.add_argument("--printability_weight", type=float, default=0.001)
    p.add_argument("--low_frequency_weight", type=float, default=0.0)
    p.add_argument("--natural_reference_weight", type=float, default=0.05)
    p.add_argument("--pose_rotation_weight", type=float, default=5.0)
    p.add_argument("--pose_translation_weight", type=float, default=1.0)
    p.add_argument("--physical_eot", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra", default="",
                   help="raw flags appended to the attack parser, for configs this script "
                        "does not expose. Whatever ablation is being calibrated must be "
                        "mirrored exactly, e.g. "
                        "--extra '--plane_mode depth_manual_anchor_surface "
                        "--manual_anchor_x 0.58 --manual_anchor_y 0.43'")
    p.add_argument("--activation_checkpoint", action="store_true", default=True,
                   help="on by default: gradient checkpointing gives numerically identical "
                        "gradients and the un-checkpointed forward needs >39 GiB of "
                        "activations for 10 frames")
    p.add_argument("--no_activation_checkpoint", dest="activation_checkpoint",
                   action="store_false")
    return p.parse_args()


def build_attack_args(cli: argparse.Namespace) -> argparse.Namespace:
    argv = [
        "attack.py",
        "--tum_root", cli.tum_root,
        "--scene_pattern", cli.scene,
        "--output_dir", "/tmp/calibrate_reg_out",
        "--frame_manifest", cli.frame_manifest,
        "--ckpt", cli.ckpt,
        "--texture_size", "128",
        "--texture_init", "image",
        "--texture_init_image", cli.texture_init_image,
        "--iterations", "1", "--inner_loop", "1", "--scenes_per_iteration", "1",
        "--patch_lr", "0.002", "--feature_layer", "aggregator_final",
        "--attack_loss", "pose_gt_untargeted",
        "--pose_rotation_weight", str(cli.pose_rotation_weight),
        "--pose_translation_weight", str(cli.pose_translation_weight),
        "--plane_mode", cli.plane_mode,
        "--plane_width", str(cli.plane_width),
        "--plane_height", str(cli.plane_height),
        "--clean_vggt_output_root", cli.clean_vggt_output_root,
        "--surface_score_mode", "coverage",
        "--surface_coverage_min", "0.005", "--surface_coverage_max", "0.06",
        "--surface_min_visible_frames", "8", "--surface_min_visibility_ratio", "0.80",
        "--surface_min_support_ratio", "0.50",
        "--surface_support_abs_tolerance", "0.10",
        "--surface_support_rel_tolerance", "0.06",
        "--fused_max_plane_residual", "0.12",
        "--visibility_depth_margin", "0.08",
        "--tv_weight", str(cli.tv_weight),
        "--printability_weight", str(cli.printability_weight),
        "--printable_color_levels", "2",
        "--low_frequency_weight", str(cli.low_frequency_weight),
        "--natural_reference_weight", str(cli.natural_reference_weight),
        "--natural_reference_image", cli.texture_init_image,
        "--seed", str(cli.seed),
        "--use_depth_visibility", "--optimize_geometry", "--surface_support_check",
    ]
    if cli.physical_eot:
        argv.append("--physical_eot")
    if cli.activation_checkpoint:
        argv.append("--activation_checkpoint")
    if cli.extra.strip():
        import shlex

        argv += shlex.split(cli.extra)
    saved, sys.argv = sys.argv, argv
    try:
        return A.parse_args()
    finally:
        sys.argv = saved


def main() -> None:
    cli = parse_args()
    args = build_attack_args(cli)
    device = torch.device("cuda")
    dtype = (torch.bfloat16 if torch.cuda.is_available()
             and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)

    model = A.load_model(args, device)
    # Only d/d(texture) is wanted. Leaving the 1B model parameters trainable makes
    # autograd keep everything needed for parameter gradients too, which is both
    # wasted work and the reason a plain forward here needs >39 GiB.
    # (Note: train_geometry_patch does not freeze them either, so the real training
    # loop computes and stores full parameter gradients on every step for nothing.)
    for p in model.parameters():
        p.requires_grad_(False)
    scene_dirs = A.list_scene_dirs(Path(args.tum_root), args.scene_pattern)
    manifest = A.load_frame_manifest(args.frame_manifest)
    intrinsics = np.asarray([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]],
                            dtype=np.float64)
    def memnote(tag: str) -> None:
        free, _ = torch.cuda.mem_get_info()
        print(f"    [mem] {tag:<26} allocated={torch.cuda.memory_allocated()/2**30:6.2f} "
              f"reserved={torch.cuda.memory_reserved()/2**30:6.2f} "
              f"free={free/2**30:6.2f} GiB", flush=True)

    memnote("after model load")
    seq_dir = scene_dirs[0]
    # The setup forward inside load_tum_sequence needs no gradients; without this the
    # graph it builds stays alive for as long as `item` does and costs ~10 GiB.
    with torch.no_grad():
        item = A.load_tum_sequence(seq_dir, manifest[seq_dir.name], args.gt_name, intrinsics,
                                   args.texture_size, args, device, model=model, dtype=dtype)
    memnote("after load_tum_sequence")
    print(f"scene {seq_dir.name}   patch coverage {item['geometry']['mask_coverage_mean']:.6f}   "
          f"amp {dtype}   physical_eot {bool(cli.physical_eot)}\n")

    # the regulariser gradient does not depend on the attack loss
    tex_reg = A.initialize_texture(args, device, requires_grad=True)
    reg = A.patch_regularization_terms(tex_reg, args)["regularization_total"]
    g_reg = float(torch.autograd.grad(reg, tex_reg)[0].norm())
    print(f"|g_reg| at the init texture = {g_reg:.6e}   "
          f"(tv={cli.tv_weight}, printability={cli.printability_weight}, "
          f"natural_reference={cli.natural_reference_weight})\n")
    del tex_reg, reg
    torch.cuda.empty_cache()

    rows = []
    print(f"{'attack_loss':<30}{'loss':>13}{'|g_attack|':>14}{'ratio':>10}")
    print("-" * 67)
    for loss_name in [s.strip() for s in cli.losses.split(",") if s.strip()]:
        args.attack_loss = loss_name
        memnote(f"before {loss_name}")
        texture = A.initialize_texture(args, device, requires_grad=True)
        adv = A.apply_geometry_patch(item["images"], texture, item["grids"], item["masks"],
                                     args, training=True)
        loss, _, maximize = A.attack_objective_loss(model, adv, item, args, dtype)
        g = torch.autograd.grad(-loss if maximize else loss, texture)[0]
        ga = float(g.norm())
        rows.append((loss_name, float(loss), ga, ga / g_reg if g_reg else float("nan")))
        print(f"{loss_name:<30}{float(loss):>13.6f}{ga:>14.4e}{rows[-1][3]:>10.3f}",
              flush=True)
        del adv, loss, g, texture
        gc.collect()
        torch.cuda.empty_cache()

    target = cli.target_ratio if cli.target_ratio is not None else max(r[3] for r in rows)
    print(f"\ntarget ratio = {target:.3f}"
          f"{'' if cli.target_ratio is not None else '  (max observed; no loss gets regularised harder than it is today)'}")
    print("\nregulariser weights that put every loss at that ratio:\n")
    print(f"{'attack_loss':<30}{'scale':>10}{'TV_WEIGHT':>14}{'PRINTABILITY_WEIGHT':>21}"
          f"{'NATURAL_REFERENCE_WEIGHT':>26}")
    print("-" * 101)
    for name, _, ga, ratio in rows:
        scale = ratio / target if target else float("nan")
        print(f"{name:<30}{scale:>10.4f}{cli.tv_weight*scale:>14.6g}"
              f"{cli.printability_weight*scale:>21.6g}"
              f"{cli.natural_reference_weight*scale:>26.6g}")
    print("\nApply per run, e.g.:")
    name, _, ga, ratio = rows[-1]
    scale = ratio / target if target else 1.0
    print(f"  ATTACK_LOSS={name} TV_WEIGHT={cli.tv_weight*scale:.6g} "
          f"PRINTABILITY_WEIGHT={cli.printability_weight*scale:.6g} "
          f"NATURAL_REFERENCE_WEIGHT={cli.natural_reference_weight*scale:.6g}")
    print("\nReport the final texture's TV / printability next to the attack numbers: "
          "matched dynamics means unmatched absolute regularisation.")


if __name__ == "__main__":
    main()
