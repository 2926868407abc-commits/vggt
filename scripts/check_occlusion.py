"""Is the GT occlusion flag real, or is it flagging nothing?

validate_gt_tracks showed visible and occluded points have nearly the same
reprojection error. Two readings:
  (a) the occlusion test is broken and "occluded" points are actually visible
  (b) the test is fine and VGGT simply extrapolates occluded points well

These are separable: VGGT's track head predicts its own visibility score. If the GT
flag agrees with that score above chance, the flag is measuring something real.
Reports agreement and, as a sanity anchor, how the score itself splits by GT flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG))
import attack_vggt_geometry_tum10 as A  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402

GT_DIR = VG / "outputs/tum_gt_point_track"
CLEAN_ROOT = VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4)
    cli = ap.parse_args()

    argv = ["attack.py", "--tum_root", "/mnt/data/wangqq/recons_eval/data/tum",
            "--scene_pattern", "x", "--output_dir", "/tmp/occ_val",
            "--ckpt", str(VG / "checkpoints/VGGT-1B"),
            "--frame_manifest", str(VG / "data/tum_dynamics_10frame_individual_scenes"
                                        "/tum10_frame_manifest.json")]
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

    print(f"{'序列':<42}{'VGGT可见分:GT可见':>18}{'GT遮挡':>10}{'AUC':>8}{'遮挡占比':>10}")
    print("-" * 90)
    aucs = []
    for f in sorted(GT_DIR.glob("*_gt.npz")):
        d = np.load(f, allow_pickle=True)
        scene = str(d["scene"])
        summary = json.loads((CLEAN_ROOT / scene / "attack_summary.json")
                             .read_text(encoding="utf-8"))
        images = load_and_preprocess_images(
            [str(p) for p in summary["image_paths"]]).to(device)
        query = torch.as_tensor(d["query_points"], device=device, dtype=torch.float32)
        with torch.no_grad():
            pred = A.forward_tracking_only(model, images, query, dtype, cli.iters)
        score = pred["track_vis"][0].float().cpu().numpy()   # (N, M)

        known = d["track_known"][1:]
        vis = d["track_visible"][1:] & known
        occ = known & ~d["track_visible"][1:]
        s = score[1:]
        if not vis.any() or not occ.any():
            print(f"{scene:<42}  样本不足")
            continue
        sv, so = s[vis], s[occ]
        # AUC = P(random visible scores above random occluded); 0.5 = no information
        allv = np.concatenate([sv, so])
        ranks = allv.argsort().argsort().astype(np.float64)
        auc = float((ranks[:len(sv)].mean() - (len(sv) - 1) / 2) / len(so))
        aucs.append(auc)
        print(f"{scene:<42}{sv.mean():>9.3f}:{so.mean():<8.3f}"
              f"{'':>0}{auc:>8.3f}{occ.mean():>10.1%}")

    print(f"\n跨序列 AUC 中位数 {np.median(aucs):.3f}")
    print("AUC 0.5 = GT 的遮挡标记与 VGGT 的可见性判断完全无关（标记无效）")
    print("AUC > 0.65 = 标记确实抓到了真实的遮挡")


if __name__ == "__main__":
    main()
