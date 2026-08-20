"""Why did the xyz patch shrink to the lower half of the screen?

The quad only selects *candidate* pixels; the plane is then fit to whatever pixels
inside it carry valid depth, and the rendered rectangle spans the inliers. TUM's
structured-light sensor drops out on dark glossy surfaces, so a monitor screen can
easily have valid depth on only part of its area. Measure that directly.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG))
sys.path.insert(0, str(VG / "scripts"))
import attack_vggt_geometry_tum10 as A  # noqa: E402
from visualize_tum10_geometry_patch import preprocess_like_vggt  # noqa: E402

TUM = Path("/mnt/data/wangqq/recons_eval/data/tum")
ROOT = VG / "outputs_attack_geometry_aware_tum10"

CASES = [
    ("xyz 显示器", "rgbd_dataset_freiburg3_sitting_xyz",
     [(0.5376, 0.4697), (0.7305, 0.4870), (0.7249, 0.6603), (0.5283, 0.6355)]),
    ("half 显示器", "rgbd_dataset_freiburg3_sitting_halfsphere",
     [(0.3603, 0.2939), (0.5682, 0.3063), (0.5644, 0.4943), (0.3603, 0.4795)]),
]


def depth_map_for(scene, frame_idx):
    """TUM depth aligned to the RGB frame, in the same crop the attack sees."""
    imgs = sorted((TUM / scene / "rgb_90").glob("*.png"))
    rows = A.read_depth_rows(TUM / scene / "depth.txt")
    ts = np.asarray([r[0] for r in rows], dtype=np.float64)
    stem = float(imgs[frame_idx].stem)
    d = np.asarray(Image.open(TUM / scene / rows[int(np.argmin(np.abs(ts - stem)))][1]),
                   dtype=np.float64) / 5000.0
    return d


for label, scene, quad in CASES:
    idx = json.loads((ROOT / "tum10_clean_uniform_l3" / scene / "attack_summary.json")
                     .read_text(encoding="utf-8"))["frame_indices"]
    rgb = preprocess_like_vggt(str(sorted((TUM / scene / "rgb_90").glob("*.png"))[idx[0]]))
    W, H = rgb.size
    d = depth_map_for(scene, idx[0])
    dh, dw = d.shape

    q = np.asarray(quad)
    x0, x1 = q[:, 0].min(), q[:, 0].max()
    y0, y1 = q[:, 1].min(), q[:, 1].max()
    sub = d[int(y0 * dh):int(y1 * dh), int(x0 * dw):int(x1 * dw)]
    valid = sub > 0
    print(f"\n=== {label}  四边形内深度有效率 {valid.mean():.1%}  "
          f"({valid.sum()}/{valid.size} 像素)")
    if valid.any():
        v = sub[valid]
        print(f"    深度范围 {v.min():.3f}~{v.max():.3f} m  中位数 {np.median(v):.3f} m")
    # where inside the quad is depth actually available, row by row
    rows_ok = valid.mean(1)
    n = len(rows_ok)
    print("    自上而下分 8 段的深度有效率: " +
          "  ".join(f"{rows_ok[i*n//8:(i+1)*n//8].mean():.0%}" for i in range(8)))
