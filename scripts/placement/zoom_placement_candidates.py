"""Zoomed crops with a fine grid, so the quad corners can be read accurately."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG / "scripts"))
from visualize_tum10_geometry_patch import preprocess_like_vggt  # noqa: E402

TUM = Path("/mnt/data/wangqq/recons_eval/data/tum")
ROOT = VG / "outputs_attack_geometry_aware_tum10"
OUT = VG / "outputs/grid_frames"

# name, scene, frame, (x0, y0, x1, y1) in normalised coords
CANDIDATES = [
    ("xyz_black_monitor", "rgbd_dataset_freiburg3_sitting_xyz", 0, (0.46, 0.40, 0.80, 0.70)),
    ("half_black_monitor", "rgbd_dataset_freiburg3_sitting_halfsphere", 0,
     (0.30, 0.22, 0.60, 0.55)),
    ("half_poster", "rgbd_dataset_freiburg3_sitting_halfsphere", 0, (0.14, 0.14, 0.38, 0.60)),
]
ZOOM = 5


def crop_grid(scene, frame_pos, box, name):
    idx = json.loads((ROOT / "tum10_clean_uniform_l3" / scene / "attack_summary.json")
                     .read_text(encoding="utf-8"))["frame_indices"]
    imgs = sorted((TUM / scene / "rgb_90").glob("*.png"))
    base = preprocess_like_vggt(str(imgs[idx[frame_pos]])).convert("RGB")
    W, H = base.size
    x0, y0, x1, y1 = box
    px = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
    crop = base.crop(px)
    cw, ch = crop.size
    img = crop.resize((cw * ZOOM, ch * ZOOM), Image.Resampling.LANCZOS)
    CW, CH = img.size
    d = ImageDraw.Draw(img, "RGBA")

    # grid lines every 0.01 in original normalised coords
    step = 0.01
    n = int(round((x1 - x0) / step))
    for k in range(n + 1):
        gx = x0 + k * step
        X = int((gx - x0) / (x1 - x0) * CW)
        major = abs(gx * 100 - round(gx * 100 / 5) * 5) < 1e-6
        d.line([(X, 0), (X, CH)], fill=(255, 255, 0, 200) if major else (255, 255, 255, 70),
               width=2 if major else 1)
        if major:
            d.text((X + 2, 2), f"{gx:.2f}", fill=(255, 255, 0, 255))
    m = int(round((y1 - y0) / step))
    for k in range(m + 1):
        gy = y0 + k * step
        Y = int((gy - y0) / (y1 - y0) * CH)
        major = abs(gy * 100 - round(gy * 100 / 5) * 5) < 1e-6
        d.line([(0, Y), (CW, Y)], fill=(255, 255, 0, 200) if major else (255, 255, 255, 70),
               width=2 if major else 1)
        if major:
            d.text((2, Y + 2), f"{gy:.2f}", fill=(255, 255, 0, 255))

    out = OUT / f"zoom_{name}.png"
    img.save(out)
    print(f"wrote {out.name}  ({img.size[0]}x{img.size[1]})")


for name, scene, f, box in CANDIDATES:
    crop_grid(scene, f, box, name)
