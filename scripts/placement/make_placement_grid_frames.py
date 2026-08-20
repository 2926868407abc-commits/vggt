"""Reference frames with a normalised coordinate grid, for picking patch corners by hand.

The manual placement modes take normalised image coordinates in the first selected
frame, so this renders that frame at the same preprocessing the attack uses and
overlays a labelled 0..1 grid. Read the four corners of whatever surface you want
off this image and pass them as MANUAL_QUAD_XY.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG))
sys.path.insert(0, str(VG / "scripts"))
from visualize_tum10_geometry_patch import preprocess_like_vggt  # noqa: E402

TUM = Path("/mnt/data/wangqq/recons_eval/data/tum")
ROOT = VG / "outputs_attack_geometry_aware_tum10"
OUT = VG / "outputs/grid_frames"
OUT.mkdir(parents=True, exist_ok=True)

SCENES = ["rgbd_dataset_freiburg3_sitting_xyz",
          "rgbd_dataset_freiburg3_sitting_halfsphere",
          "rgbd_dataset_freiburg3_sitting_static"]
FRAMES = [0, 4, 9]
SCALE = 2  # upscale so the grid labels stay readable


def grid_frame(scene: str, frame_pos: int) -> Image.Image:
    idx = json.loads((ROOT / "tum10_clean_uniform_l3" / scene / "attack_summary.json")
                     .read_text(encoding="utf-8"))["frame_indices"]
    imgs = sorted((TUM / scene / "rgb_90").glob("*.png"))
    base = preprocess_like_vggt(str(imgs[idx[frame_pos]])).convert("RGB")
    w, h = base.size
    img = base.resize((w * SCALE, h * SCALE), Image.Resampling.LANCZOS)
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")

    for k in range(1, 10):
        x = int(W * k / 10)
        y = int(H * k / 10)
        col = (255, 255, 0, 190) if k % 5 == 0 else (255, 255, 255, 110)
        wdt = 2 if k % 5 == 0 else 1
        d.line([(x, 0), (x, H)], fill=col, width=wdt)
        d.line([(0, y), (W, y)], fill=col, width=wdt)
        d.text((x + 3, 3), f"{k/10:.1f}", fill=(255, 255, 0, 255))
        d.text((3, y + 3), f"{k/10:.1f}", fill=(255, 255, 0, 255))
    for k in range(1, 20):
        if k % 2 == 0:
            continue
        x, y = int(W * k / 20), int(H * k / 20)
        d.line([(x, 0), (x, H)], fill=(255, 255, 255, 45), width=1)
        d.line([(0, y), (W, y)], fill=(255, 255, 255, 45), width=1)

    d.rectangle([0, 0, 250, 20], fill=(0, 0, 0, 170))
    d.text((5, 4), f"{scene.replace('rgbd_dataset_freiburg3_', '')}  frame {frame_pos}",
           fill=(255, 255, 255, 255))
    return img


for scene in SCENES:
    for f in FRAMES:
        out = OUT / f"{scene.replace('rgbd_dataset_freiburg3_', '')}_f{f}_grid.png"
        grid_frame(scene, f).save(out)
        print(f"wrote {out.name}")
