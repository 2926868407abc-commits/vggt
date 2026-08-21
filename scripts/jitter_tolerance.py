"""At what placement error does the attack die?

The binary test (jitter on / off) says the non-robust patch retains 1% at 2%
translation. That is a number without a scale attached. Sweeping the magnitude
turns it into a physical tolerance: the displacement, in mm on the printed sheet,
beyond which the attack stops working. That is the number a physical-attack claim
actually needs.

Writes trajectories only; score with score_jitter_tolerance.py in the recons_eval env.
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
sys.path.insert(0, str(VG / "scripts"))
from eval_patch_under_jitter import build  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    ap.add_argument("--runs", default="mon_monhalf_al_1000,eot_k8")
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--out", default="/tmp/jitter_eval/tolerance.npz")
    cli = ap.parse_args()

    args = build(cli.scene)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = A.load_model(args, device)
    for p in model.parameters():
        p.requires_grad_(False)

    dirs = A.list_scene_dirs(Path(args.tum_root), args.scene_pattern)
    manifest = A.load_frame_manifest(args.frame_manifest)
    intr = np.asarray([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]],
                      dtype=np.float64)
    seq = dirs[0]
    with torch.no_grad():
        item = A.load_tum_sequence(seq, manifest[seq.name], args.gt_name, intr,
                                   args.texture_size, args, device, model=model, dtype=dtype)
    patch_w = float(item["geometry"]["width_m"])

    def load_tex(run):
        d = np.load(VG / "outputs_attack_geometry_aware_tum10" / run
                    / "geometry_patch/geometry_patch_texture.npz")
        k = [x for x in d.files if "tex" in x.lower()] or list(d.files)
        t = torch.as_tensor(np.squeeze(d[k[0]]), device=device, dtype=torch.float32)
        if t.ndim == 3 and t.shape[-1] in (3, 4):
            t = t.permute(2, 0, 1)
        if t.max() > 1.5:
            t = t / 255.0
        return t[:3].unsqueeze(0)

    def traj(tex, a, training):
        with torch.no_grad():
            adv = A.apply_geometry_patch(item["images"], tex, item["grids"],
                                         item["masks"], a, training=training)
            preds = A.forward_camera_pose_only(model, adv, dtype, a.activation_checkpoint)
        extrinsic, _ = A.pose_encoding_to_extri_intri(preds["pose_enc"], item["tensor_hw"])
        c2w = A.w2c_3x4_to_c2w(extrinsic).detach().float().cpu().numpy()
        return np.asarray(c2w[0] if c2w.ndim == 4 else c2w, dtype=np.float64)

    # translation only: it is the one component with a direct physical reading
    MAGS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04]
    out = {"frame_indices": np.asarray(item["frame_indices"], dtype=np.int64),
           "scene": seq.name, "mags": np.asarray(MAGS), "patch_width_m": patch_w}
    print(f"patch width {patch_w:.3f} m; translation magnitudes in mm: "
          + ", ".join(f"{m * patch_w * 1000:.1f}" for m in MAGS))

    for run in cli.runs.split(","):
        tex = load_tex(run)
        for mi, mag in enumerate(MAGS):
            a = argparse.Namespace(**vars(args))
            a.physical_eot = mag > 0
            a._eot_strength = 1.0
            a.eot_brightness = a.eot_contrast = a.eot_gamma = a.eot_noise_std = 0.0
            a.eot_geo_scale = a.eot_geo_rotate_degrees = a.eot_geo_perspective = 0.0
            a.eot_geo_translate = mag
            n = 1 if mag == 0 else cli.draws
            out[f"{run}|{mi}"] = np.stack([traj(tex, a, training=mag > 0)
                                           for _ in range(n)])
        print(f"  done {run}")

    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cli.out, runs=cli.runs, **out)
    print(f"wrote {cli.out}")


if __name__ == "__main__":
    main()
