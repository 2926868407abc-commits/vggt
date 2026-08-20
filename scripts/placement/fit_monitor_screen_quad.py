"""Fit the monitor screen as a quadrilateral, not a bounding box.

The monitors are viewed obliquely, so an axis-aligned box either clips the screen or
spills onto the wall. Take the dark component, find its four extreme corners via the
standard sum/difference trick, then shrink slightly toward the centroid so the patch
sits inside the bezel rather than on it.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG / "scripts"))
from visualize_tum10_geometry_patch import preprocess_like_vggt  # noqa: E402

TUM = Path("/mnt/data/wangqq/recons_eval/data/tum")
ROOT = VG / "outputs_attack_geometry_aware_tum10"
OUT = VG / "outputs/grid_frames"

# name, scene, frame, seed point (normalised) inside the screen we want
TARGETS = [
    ("xyz_monitor", "rgbd_dataset_freiburg3_sitting_xyz", 0, (0.64, 0.55)),
    ("half_monitor", "rgbd_dataset_freiburg3_sitting_halfsphere", 0, (0.46, 0.40)),
]
INSET = 0.03  # fraction of the quad pulled in toward the centre, to clear the bezel


def corners_of(mask):
    ys, xs = np.where(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.stack([tl, tr, br, bl])


for name, scene, frame_pos, seed in TARGETS:
    idx = json.loads((ROOT / "tum10_clean_uniform_l3" / scene / "attack_summary.json")
                     .read_text(encoding="utf-8"))["frame_indices"]
    imgs = sorted((TUM / scene / "rgb_90").glob("*.png"))
    base = preprocess_like_vggt(str(imgs[idx[frame_pos]])).convert("RGB")
    a = np.asarray(base, dtype=np.float64)
    W, H = base.size
    dark = ndimage.binary_opening(a.mean(2) < 70, np.ones((3, 3)))
    lbl, _ = ndimage.label(dark)
    want = lbl[int(seed[1] * H), int(seed[0] * W)]
    if want == 0:
        print(f"{name}: seed point is not dark, adjust it")
        continue
    m = lbl == want
    q = corners_of(m)
    c = q.mean(0)
    q_in = c + (q - c) * (1.0 - INSET)

    norm = q_in / np.array([W, H])
    txt = ",".join(f"{v:.4f}" for v in norm.reshape(-1))
    print(f"\n=== {name}  (面积 {m.sum()/(W*H):.2%})")
    print(f"    TL {norm[0]}  TR {norm[1]}  BR {norm[2]}  BL {norm[3]}")
    print(f"    MANUAL_QUAD_XY={txt}")

    vis = base.copy().resize((W * 3, H * 3), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(vis)
    d.polygon([tuple(p * 3) for p in q_in], outline=(255, 0, 0), width=3)
    d.polygon([tuple(p * 3) for p in q], outline=(0, 255, 255), width=1)
    vis.save(OUT / f"quadfit_{name}.png")
    print(f"    wrote quadfit_{name}.png")
