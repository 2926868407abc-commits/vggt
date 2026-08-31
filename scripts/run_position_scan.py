"""Stage 2: sweep patch position over every qualified plane with a frozen texture.

Calls the training pipeline's own geometry builder once per candidate rather than
re-deriving projection and z-buffering here. Anything reimplemented would drift from
the renderer that produced the manual-monitor baseline, and the whole comparison
rests on those being identical.

The model and the clean sequence load once; only the four world corners change
between candidates, which is the single variable the plan wants swept.

Fast pass only -- writes each candidate's predicted trajectory so the full four-task
evaluation can run afterwards on the top few, per section 5.4.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

VG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VG))
import attack_vggt_geometry_tum10 as A  # noqa: E402

TUM = Path("/mnt/data/wangqq/recons_eval/data/tum")
PL = VG / "outputs/candidate_planes"
CLEAN_ROOT = VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"


def build_args(
    scene: str,
    texture_size: int,
    tex_path: str,
    *,
    tum_root: Path,
    clean_root: Path,
    checkpoint: Path,
    frame_manifest: Path,
    output_dir: Path,
) -> argparse.Namespace:
    argv = [
        "attack.py", "--tum_root", str(tum_root), "--scene_pattern", scene,
        "--output_dir", str(output_dir), "--ckpt", str(checkpoint),
        "--frame_manifest", str(frame_manifest),
        "--texture_size", str(texture_size),
        "--texture_init", "image", "--texture_init_image", tex_path,
        "--natural_reference_image", str(VG / "assets/hazard_textures/mde_attack_warnning.png"),
        "--clean_vggt_output_root", str(clean_root),
        "--plane_mode", "explicit_world_quad",
        "--explicit_quad_world", "0,0,0,0,0,0,0,0,0,0,0,0",   # replaced per candidate
        "--use_depth_visibility", "--visibility_depth_margin", "0.08",
        "--seed", "0",
    ]
    saved, sys.argv = sys.argv, argv
    try:
        return A.parse_args()
    finally:
        sys.argv = saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    ap.add_argument("--area", type=float, default=0.05)
    historical_texture = (
        VG / "outputs_attack_geometry_aware_tum10/l64_al_s0/geometry_patch/"
        "geometry_patch_texture.png"
    )
    fallback_texture = VG / "assets/hazard_textures/mde_attack_warnning.png"
    ap.add_argument(
        "--texture",
        default=str(historical_texture if historical_texture.exists() else fallback_texture),
    )
    ap.add_argument("--texture_size", type=int, default=64)
    ap.add_argument("--tum_root", type=Path, default=TUM)
    ap.add_argument("--planes_dir", type=Path, default=PL)
    ap.add_argument("--clean_root", type=Path, default=CLEAN_ROOT)
    ap.add_argument("--checkpoint", type=Path, default=VG / "checkpoints/VGGT-1B")
    ap.add_argument(
        "--frame_manifest",
        type=Path,
        default=(VG / "data/tum_dynamics_10frame_individual_scenes/"
                 "tum10_frame_manifest.json"),
    )
    ap.add_argument("--scratch_dir", type=Path, default=Path("/tmp/pos_scan"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    scene = cli.scene
    spec_path = cli.planes_dir / f"{scene}_scan_a{cli.area}.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cands = spec["candidates"][:cli.limit] if cli.limit else spec["candidates"]
    print(f"{scene}  面积 {cli.area} m²  候选 {len(cands)}")

    args = build_args(
        scene,
        cli.texture_size,
        cli.texture,
        tum_root=cli.tum_root,
        clean_root=cli.clean_root,
        checkpoint=cli.checkpoint,
        frame_manifest=cli.frame_manifest,
        output_dir=cli.scratch_dir,
    )
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

    # one load to get images, poses and depth; geometry is rebuilt per candidate
    with torch.no_grad():
        item = A.load_tum_sequence(seq, manifest[seq.name], args.gt_name, intr,
                                   cli.texture_size, args, device, model=model, dtype=dtype)
    images = item["images"]
    tensor_hw = item["tensor_hw"]
    # the geometry trajectory, not the raw GT one: build_geometry_grids places
    # against the same poses the training pipeline uses
    c2w = item["c2w_geometry"]
    image_paths = item["image_paths"]
    # load_tum_sequence does not return the depth maps, and without them every
    # candidate fails its z-buffer visibility test
    sys.path.insert(0, str(VG / "scripts"))
    from diag_gauge_invariance import (match_depth_paths,  # noqa: E402
                                       preprocess_tum_depth_to_vggt_grid)
    dpaths = match_depth_paths(seq, image_paths)
    depth_maps = [None if dp is None else
                  preprocess_tum_depth_to_vggt_grid(dp, Path(ip), tensor_hw)
                  for dp, ip in zip(dpaths, image_paths)]
    print(f"深度关联 {sum(d is not None for d in depth_maps)}/{len(depth_maps)} 帧")

    tex = A.initialize_texture(args, device, requires_grad=False)

    recs = []
    t0 = time.time()
    for i, c in enumerate(cands):
        args.explicit_quad_world = ",".join(f"{v:.6f}" for v in c["quad"])
        try:
            with torch.no_grad():
                grids, masks, meta = A.build_geometry_grids(
                    c2w, image_paths, tensor_hw, cli.texture_size, intr, args,
                    device, depth_maps)
                adv = A.apply_geometry_patch(images, tex, grids, masks, args,
                                             training=False)
                preds = A.forward_camera_pose_only(model, adv, dtype, False)
                extr, _ = A.pose_encoding_to_extri_intri(preds["pose_enc"], tensor_hw)
        except Exception as e:                       # a candidate can be unplaceable
            recs.append({**{k: c[k] for k in ("plane", "iu", "iv", "cu", "cv", "fill")},
                         "error": str(e)[:120]})
            continue
        cov = float(masks.float().mean().item())
        recs.append({
            **{k: c[k] for k in ("plane", "iu", "iv", "cu", "cv", "fill")},
            "coverage": cov,
            "width_m": float(meta.get("width_m", np.nan)),
            "height_m": float(meta.get("height_m", np.nan)),
            "extrinsic": extr[0].float().cpu().numpy().tolist(),
        })
        if (i + 1) % 20 == 0 or i + 1 == len(cands):
            el = time.time() - t0
            print(f"  {i + 1}/{len(cands)}  已用 {el / 60:.1f} 分  "
                  f"预计剩余 {(el / (i + 1)) * (len(cands) - i - 1) / 60:.1f} 分")

    dst = Path(cli.out or (cli.planes_dir / f"{scene}_scanres_a{cli.area}.json"))
    dst.write_text(json.dumps({"scene": scene, "area_m2": cli.area,
                               "texture": cli.texture, "grid": spec.get("grid", 8),
                               "source_spec": str(spec_path), "records": recs}, indent=1),
                   encoding="utf-8")
    bad = sum(1 for r in recs if "error" in r)
    print(f"\n完成 {len(recs)} 个，失败 {bad} 个 -> {dst}")


if __name__ == "__main__":
    main()
