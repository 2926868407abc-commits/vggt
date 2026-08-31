"""Stage 1 acceptance visuals (plan section 4.3).

Section 4.4 requires a human to confirm each candidate lands on a real, continuous
surface and does not straddle an object boundary. Numbers cannot show that, so this
paints each candidate's inlier points onto the frames they are visible in.

Occluded points are drawn dimmer than visible ones, because a candidate that looks
large but is mostly behind a person is not usable and that has to be visible here
rather than only in the occlusion column of the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

VG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import (  # noqa: E402
    TUM_CX, TUM_CY, TUM_FX, TUM_FY,
    match_depth_paths, preprocess_tum_depth_to_vggt_grid, projection_params,
)
from visualize_tum10_geometry_patch import preprocess_like_vggt  # noqa: E402

TUM = Path("/mnt/data/wangqq/recons_eval/data/tum")
GT_DIR = VG / "outputs/tum_gt_point_track"
CLEAN = VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"
PLANES = VG / "outputs/candidate_planes"

PALETTE = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0),
           (232, 123, 164), (0, 131, 0), (74, 58, 167), (227, 73, 72),
           (120, 200, 230), (180, 120, 60), (100, 180, 100), (200, 80, 160)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    ap.add_argument("--tum_root", type=Path, default=TUM)
    ap.add_argument("--gt_dir", type=Path, default=GT_DIR)
    ap.add_argument("--clean_root", type=Path, default=CLEAN)
    ap.add_argument("--planes_dir", type=Path, default=PLANES)
    ap.add_argument("--frames", default="0,3,6,9")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--occl_tol", type=float, default=0.05)
    ap.add_argument("--voxel", type=float, default=0.02)
    cli = ap.parse_args()

    scene = cli.scene
    meta = json.loads(
        (cli.planes_dir / f"{scene}_planes.json").read_text(encoding="utf-8")
    )
    cands = meta["candidates"][:cli.top]
    gt = np.load(cli.gt_dir / f"{scene}_gt.npz", allow_pickle=True)
    gt_c2w = gt["gt_c2w"]
    tensor_hw = tuple(int(x) for x in gt["tensor_hw"])
    world_all = gt["point_map"][gt["point_valid"]]

    summary = json.loads(
        (cli.clean_root / scene / "attack_summary.json").read_text(encoding="utf-8")
    )
    image_paths = [str(p) for p in summary["image_paths"]]
    proj = projection_params(Path(image_paths[0]), tensor_hw)
    dpaths = match_depth_paths(cli.tum_root / scene, image_paths)
    depths = [preprocess_tum_depth_to_vggt_grid(p, Path(ip), tensor_hw)
              for p, ip in zip(dpaths, image_paths)]

    # rebuild each candidate's own points from its plane + local box, which is what
    # the search will actually place patches on
    keys = np.floor(world_all / cli.voxel).astype(np.int64)
    _, uniq = np.unique(keys, axis=0, return_index=True)
    cloud = world_all[np.sort(uniq)]

    frames = [int(f) for f in cli.frames.split(",")]
    tiles = []
    for fi in frames:
        base = preprocess_like_vggt(image_paths[fi]).convert("RGB")
        img = base.resize((base.width * 2, base.height * 2), Image.Resampling.LANCZOS)
        d = ImageDraw.Draw(img, "RGBA")
        w2c = np.linalg.inv(gt_c2w[fi])
        for ci, c in enumerate(cands):
            centre = np.asarray(c["centre"])
            nrm = np.asarray(c["normal"])
            u, v = np.asarray(c["u"]), np.asarray(c["v"])
            lo, hi = np.asarray(c["uv_min"]), np.asarray(c["uv_max"])
            rel = cloud - centre
            on = np.abs(rel @ nrm) < 0.02
            uu, vv = rel @ u, rel @ v
            box = on & (uu >= lo[0]) & (uu <= hi[0]) & (vv >= lo[1]) & (vv <= hi[1])
            pts = cloud[box]
            if len(pts) == 0:
                continue
            cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
            z = cam[:, 2]
            ok = z > 1e-6
            if not ok.any():
                continue
            uu2 = TUM_FX * cam[ok, 0] / z[ok] + TUM_CX
            vv2 = TUM_FY * cam[ok, 1] / z[ok] + TUM_CY
            xs = uu2 * proj["scale_x"] * 2
            ys = (vv2 * proj["scale_y"] - proj["crop_y"]) * 2
            zz = z[ok]
            H, W = tensor_hw
            inb = (xs >= 0) & (xs < W * 2) & (ys >= 0) & (ys < H * 2)
            col = PALETTE[ci % len(PALETTE)]
            xi = np.clip((xs / 2).astype(int), 0, W - 1)
            yi = np.clip((ys / 2).astype(int), 0, H - 1)
            sd = depths[fi][yi, xi]
            occl = (sd > 1e-6) & (zz > sd + cli.occl_tol)
            for sel, alpha in ((inb & ~occl, 200), (inb & occl, 55)):
                for x, y in zip(xs[sel][::3], ys[sel][::3]):
                    d.rectangle([x - 1, y - 1, x + 1, y + 1], fill=col + (alpha,))
        d.rectangle([0, 0, 190, 22], fill=(20, 20, 20, 210))
        d.text((6, 5), f"frame {fi}", fill=(255, 255, 255, 255))
        tiles.append(img)

    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (tw * cols, th * rows + 120), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    d = ImageDraw.Draw(sheet)
    y0 = th * rows + 8
    d.text((10, y0), f"{scene}  候选平面（暗色=该帧被遮挡）", fill=(0, 0, 0))
    for ci, c in enumerate(cands):
        col = PALETTE[ci % len(PALETTE)]
        yy = y0 + 22 + (ci // 2) * 18
        xx = 10 + (ci % 2) * 640
        d.rectangle([xx, yy + 3, xx + 22, yy + 13], fill=col)
        d.text((xx + 30, yy),
               f"#{ci}  {c['area_m2']:.2f}m²  "
               f"{c['extent_m'][0]:.2f}x{c['extent_m'][1]:.2f}m  "
               f"可见{c['visible_frames']}/10  遮挡{c['occlusion_ratio_mean']:.0%}",
               fill=(0, 0, 0))
    outp = cli.planes_dir / f"{scene}_contact_sheet.png"
    sheet.save(outp)
    print(f"wrote {outp}  {sheet.size}")


if __name__ == "__main__":
    main()
