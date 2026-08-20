"""Zoom the corrected placements to check the patch sits inside the bezel."""
from pathlib import Path

from PIL import Image

VIS = Path("/mnt/data/wangqq/vggt/outputs/vis")
OUT = Path("/mnt/data/wangqq/vggt/outputs/grid_frames")
ZOOM = 4

CASES = [
    ("s2_mon_xyz", "rgbd_dataset_freiburg3_sitting_xyz", 0, (0.46, 0.38, 0.82, 0.72)),
    ("s2_mon_half", "rgbd_dataset_freiburg3_sitting_halfsphere", 0, (0.28, 0.20, 0.66, 0.58)),
    ("s2_post_half", "rgbd_dataset_freiburg3_sitting_halfsphere", 0, (0.14, 0.10, 0.42, 0.58)),
]

for run, scene, frame, box in CASES:
    matches = sorted((VIS / run / scene / "overlay").glob(f"{frame:02d}_*_patch.png"))
    if not matches:
        print(f"missing overlay for {run} frame {frame}")
        continue
    img = Image.open(matches[0]).convert("RGB")
    W, H = img.size
    x0, y0, x1, y1 = box
    crop = img.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
    crop = crop.resize((crop.width * ZOOM, crop.height * ZOOM), Image.Resampling.LANCZOS)
    out = OUT / f"fixed_{run}.png"
    crop.save(out)
    print(f"wrote {out.name}")
