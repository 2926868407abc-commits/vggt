"""Render validated automatic-location candidates with the production geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


VG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VG))
sys.path.insert(0, str(VG / "scripts"))

import attack_vggt_geometry_tum10 as A  # noqa: E402
from diag_gauge_invariance import (  # noqa: E402
    match_depth_paths,
    preprocess_tum_depth_to_vggt_grid,
)
from run_position_scan import build_args  # noqa: E402
from visualize_tum10_geometry_patch import composite_patch, preprocess_like_vggt  # noqa: E402


TUM = Path("/mnt/data/wangqq/recons_eval/data/tum")
PLANES = VG / "outputs/candidate_planes"
CLEAN = VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    parser.add_argument("--area", type=float, default=0.2)
    parser.add_argument("--frame", type=int, default=5)
    parser.add_argument("--top_per_plane", type=int, default=1)
    parser.add_argument("--tum_root", type=Path, default=TUM)
    parser.add_argument("--planes_dir", type=Path, default=PLANES)
    parser.add_argument("--clean_root", type=Path, default=CLEAN)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--texture",
        type=Path,
        default=VG / "assets/hazard_textures/mde_attack_warnning.png",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    source = cli.input or cli.planes_dir / f"{cli.scene}_validated_a{cli.area}.json"
    spec = json.loads(source.read_text(encoding="utf-8"))
    plane_npz = np.load(cli.planes_dir / f"{cli.scene}_planes.npz", allow_pickle=True)
    c2w = plane_npz["gt_c2w"]
    tensor_hw = tuple(int(value) for value in plane_npz["tensor_hw"])
    summary = json.loads(
        (cli.clean_root / cli.scene / "attack_summary.json").read_text(encoding="utf-8")
    )
    image_paths = [str(path) for path in summary["image_paths"]]
    depth_paths = match_depth_paths(cli.tum_root / cli.scene, image_paths)
    depth_maps = [
        preprocess_tum_depth_to_vggt_grid(path, Path(image_path), tensor_hw)
        for path, image_path in zip(depth_paths, image_paths)
    ]
    args = build_args(
        cli.scene,
        64,
        str(cli.texture),
        tum_root=cli.tum_root,
        clean_root=cli.clean_root,
        checkpoint=VG / "checkpoints/VGGT-1B",
        frame_manifest=(VG / "data/tum_dynamics_10frame_individual_scenes/"
                        "tum10_frame_manifest.json"),
        output_dir=Path("/tmp/candidate_viz"),
        surface_support_check=True,
        support_abs_tolerance=0.06,
        support_rel_tolerance=0.02,
    )
    intrinsics = np.asarray(
        [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    by_plane: dict[int, list[dict]] = {}
    for candidate in spec["candidates"]:
        by_plane.setdefault(int(candidate["plane"]), []).append(candidate)
    selected: list[dict] = []
    for plane in sorted(by_plane):
        ranked = sorted(
            by_plane[plane],
            key=lambda item: item["validation"]["mask_coverage_mean"],
            reverse=True,
        )
        selected.extend(ranked[: cli.top_per_plane])

    names = {0: "floor", 1: "table", 2: "monitor", 7: "partition"}
    base = preprocess_like_vggt(image_paths[cli.frame])
    texture = Image.open(cli.texture).convert("RGB")
    tiles: list[Image.Image] = []
    old_depth_maps = getattr(args, "_active_depth_maps", None)
    args._active_depth_maps = depth_maps
    try:
        for candidate in selected:
            quad = np.asarray(candidate["quad"], dtype=np.float64).reshape(4, 3)
            _, masks, meta = A.build_geometry_arrays_for_plane(
                quad,
                c2w,
                image_paths,
                tensor_hw,
                64,
                intrinsics,
                args,
            )
            plane = int(candidate["plane"])
            coverage = float(meta["mask_coverage_per_frame"][cli.frame])
            label = (
                f"plane {plane} ({names.get(plane, 'surface')})  "
                f"grid=({candidate['iu']},{candidate['iv']})  "
                f"visible={coverage:.1%}"
            )
            tile = composite_patch(
                base,
                texture,
                np.asarray(meta["projected_corners"][cli.frame]),
                1.0,
                label,
                visibility_mask=masks[cli.frame],
                draw_outline=True,
                draw_text_label=True,
            )
            tiles.append(tile)
    finally:
        args._active_depth_maps = old_depth_maps

    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    width, height = tiles[0].size
    footer = 42
    sheet = Image.new("RGB", (columns * width, rows * height + footer), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (10, rows * height + 12),
        f"validated candidates; same frame={cli.frame}; same texture; same area={cli.area} m2",
        fill="black",
    )
    output = cli.out or cli.planes_dir / f"{cli.scene}_validated_top_frame{cli.frame}.png"
    sheet.save(output)
    print(f"selected {len(selected)} candidates -> {output}")


if __name__ == "__main__":
    main()
