#!/usr/bin/env python3
"""Render a corrected visibility preview without running VGGT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image


REPO = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import attack_vggt_geometry_tum10 as attack  # noqa: E402
from eval_stitched_pose import build_attack_args  # noqa: E402


def contact_sheet(images: list[Image.Image], columns: int = 5) -> Image.Image:
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texture", required=True)
    parser.add_argument("--quad_file", required=True)
    parser.add_argument(
        "--manual_quad_xy",
        default="",
        help="Optional normalized TL,TR,BR,BL image quad. When set, refit the "
             "carrier plane from the anchor frame depth instead of reusing the "
             "explicit world quad.",
    )
    parser.add_argument("--save_quad_file", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_xyz")
    parser.add_argument("--frames", default="0,67,135,202,270,337,405,472,540,607")
    parser.add_argument("--visibility_depth_margin", type=float, default=0.08)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame_indices = [int(value) for value in args.frames.split(",")]
    helper_args = argparse.Namespace(
        tum_root="/mnt/data/wangqq/recons_eval/data/tum_all_sync",
        output_root=str(output),
        protocol=str(REPO / "data/tum10_split/eight_seq_pool.json"),
        ckpt=str(REPO / "checkpoints/VGGT-1B"),
        gt_name="groundtruth.txt",
        quad_file=args.quad_file,
        visibility_depth_margin=args.visibility_depth_margin,
    )
    render_args = build_attack_args(attack, helper_args)
    if args.manual_quad_xy:
        render_args.plane_mode = "depth_manual_quad_surface"
        render_args.explicit_quad_world = None
        render_args.manual_quad_xy = args.manual_quad_xy
        render_args.manual_quad_coordinates = "normalized"
        render_args.manual_anchor_frame = 0
        render_args.manual_quad_fit_shrink = 0.75
        render_args.manual_quad_depth_sample_stride = 1
        render_args.manual_quad_plane_inlier_tolerance = 0.06
        render_args.manual_quad_min_inlier_ratio = 0.25
    device = torch.device("cpu")
    texture = attack.load_texture(args.texture, device, render_args)
    intrinsics = np.asarray([
        [render_args.fx, 0.0, render_args.cx],
        [0.0, render_args.fy, render_args.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    item = attack.load_tum_sequence(
        Path(helper_args.tum_root) / args.scene,
        frame_indices,
        helper_args.gt_name,
        intrinsics,
        int(texture.shape[-1]),
        render_args,
        device,
    )
    patched = attack.apply_geometry_patch(
        item["images"], texture, item["grids"], item["masks"],
        render_args, training=False,
    )

    clean_images: list[Image.Image] = []
    patched_images: list[Image.Image] = []
    debug_images: list[Image.Image] = []
    corners = item["geometry"]["projected_corners"]
    np.savetxt(output / "projected_corners.txt", np.asarray(corners).reshape(-1, 8))
    if args.save_quad_file:
        quad = np.asarray(item["geometry"]["plane_corners_world"], dtype=np.float64)
        Path(args.save_quad_file).write_text(
            ",".join(f"{value:.9f}" for value in quad.reshape(-1)) + "\n",
            encoding="utf-8",
        )
    for position, frame_index in enumerate(frame_indices):
        clean = to_pil_image(item["images"][position].clamp(0, 1))
        adv = to_pil_image(patched[position].clamp(0, 1))
        debug = adv.copy()
        draw = ImageDraw.Draw(debug)
        polygon = [(float(x), float(y)) for x, y in corners[position]]
        draw.line(polygon + [polygon[0]], fill=(255, 0, 0), width=3)
        clean.save(output / f"{position:02d}_frame_{frame_index:06d}_clean.png")
        adv.save(output / f"{position:02d}_frame_{frame_index:06d}_patch.png")
        debug.save(output / f"{position:02d}_frame_{frame_index:06d}_debug.png")
        clean_images.append(clean)
        patched_images.append(adv)
        debug_images.append(debug)

    contact_sheet(patched_images).save(output / "contact_sheet_patch_fixed.png")
    contact_sheet(debug_images).save(output / "contact_sheet_debug_fixed.png")
    np.savez_compressed(
        output / "visibility_fixed.npz",
        frames=np.asarray(frame_indices),
        masks=item["masks"].numpy().astype(np.float16),
        coverage=np.asarray(item["geometry"]["mask_coverage_per_frame"]),
        raw_coverage=np.asarray(item["geometry"]["raw_projected_coverage_per_frame"]),
        visibility=np.asarray(item["geometry"]["visibility_ratio_per_frame"]),
    )
    print("coverage", item["geometry"]["mask_coverage_per_frame"])
    print("raw", item["geometry"]["raw_projected_coverage_per_frame"])
    print("wrote", output)


if __name__ == "__main__":
    main()
