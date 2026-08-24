"""Does VGGT's clean tracking agree with the reprojection GT?

The round-trip check only proves the GT is self-consistent; it would pass just as
happily with wrong intrinsics. This is the check that can actually fail: run the
track head on clean images with the GT query points and compare. Clean VGGT is a
strong tracker, so if the GT is right the two should agree to a few pixels on
points the GT calls visible. A large error means the GT is wrong -- wrong
intrinsics, a bad depth/RGB association, or an occlusion test that lets occluded
points through -- and nothing built on it would mean anything.

Also reports the error on points the GT calls *invisible*, which should be clearly
worse; if visible and invisible score the same, the visibility test is doing nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG))
import attack_vggt_geometry_tum10 as A  # noqa: E402

GT_DIR = VG / "outputs/tum_gt_point_track"
CLEAN_ROOT = VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="")
    ap.add_argument("--iters", type=int, default=4)
    cli = ap.parse_args()

    files = sorted(GT_DIR.glob("*_gt.npz"))
    if cli.scenes:
        want = {s.strip() for s in cli.scenes.split(",") if s.strip()}
        files = [f for f in files if f.stem[:-3] in want]

    argv = [
        "attack.py", "--tum_root", "/mnt/data/wangqq/recons_eval/data/tum",
        "--scene_pattern", "rgbd_dataset_freiburg3_sitting_halfsphere",
        "--output_dir", "/tmp/gt_val", "--ckpt", str(VG / "checkpoints/VGGT-1B"),
        "--frame_manifest", str(VG / "data/tum_dynamics_10frame_individual_scenes"
                                    "/tum10_frame_manifest.json"),
    ]
    saved, sys.argv = sys.argv, argv
    try:
        args = A.parse_args()
    finally:
        sys.argv = saved

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = A.load_model(args, device)
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"{'序列':<44}{'可见点误差(px)':>16}{'不可见点':>12}{'点数':>8}")
    print("-" * 82)
    all_vis = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        scene = str(d["scene"])
        import json
        summary = json.loads((CLEAN_ROOT / scene / "attack_summary.json")
                             .read_text(encoding="utf-8"))
        paths = [str(p) for p in summary["image_paths"]]
        images = A.load_images_as_tensor(paths).to(device) \
            if hasattr(A, "load_images_as_tensor") else None
        if images is None:
            from vggt.utils.load_fn import load_and_preprocess_images
            images = load_and_preprocess_images(paths).to(device)

        query = torch.as_tensor(d["query_points"], device=device, dtype=torch.float32)
        with torch.no_grad():
            pred = A.forward_tracking_only(model, images, query, dtype, cli.iters)
        tr = pred["track"][0].float().cpu().numpy()          # (N, M, 2)

        gt = d["tracks"]
        vis = d["track_visible"]
        known = d["track_known"]
        err = np.linalg.norm(tr - gt, axis=-1)               # (N, M)

        # frame 0 is the query frame itself; exclude it from the agreement statistic
        sel_v = vis[1:] & known[1:]
        sel_i = known[1:] & ~vis[1:]
        med_v = float(np.median(err[1:][sel_v])) if sel_v.any() else float("nan")
        med_i = float(np.median(err[1:][sel_i])) if sel_i.any() else float("nan")
        all_vis.append(med_v)
        print(f"{scene:<44}{med_v:>16.2f}{med_i:>12.2f}{int(sel_v.sum()):>8}")

    print(f"\n可见点中位误差的跨序列中位数: {np.nanmedian(all_vis):.2f} px")
    print("判据：干净 VGGT 是强跟踪器，可见点上应当只差几个像素。")
    print("      若可见点与不可见点误差接近，说明遮挡判定没起作用。")


if __name__ == "__main__":
    main()
