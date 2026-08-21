"""The two remaining attack-effect figures. Both read saved artefacts only -- no GPU.

fig3  the same patch rendered at three placement errors, with the ATE each one
      produces. The point is the mismatch between the two: the renders are
      indistinguishable while the attack goes from working to dead, so "it looks
      fine" is not evidence the patch is still doing anything.

fig4  the predicted point cloud, clean vs attacked, both mapped into the GT frame
      by the trajectory's own Sim(3). Same reason as the trajectory figure: without
      that mapping the picture would show a global transform the evaluation cancels.

Labels are English -- the server has no CJK font.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import ReconsEval, gt_traj_for, load_run  # noqa: E402
from visualize_tum10_geometry_patch import (  # noqa: E402
    composite_patch, preprocess_like_vggt,
)

R = VG / "outputs_attack_geometry_aware_tum10"
OUT = VG / "outputs/figures/current"
OUT.mkdir(parents=True, exist_ok=True)
SCENE = "rgbd_dataset_freiburg3_sitting_halfsphere"
PATCH_MM = 638.87
FRAME = 0

# ---------------------------------------------------------------- fig 3
RUN = "res128_noeot"
summary = json.loads((R / RUN / SCENE / "attack_summary.json").read_text(encoding="utf-8"))
corners = np.asarray(summary["geometry"]["projected_corners"], dtype=np.float64)
masks = np.load(R / RUN / SCENE / "geometry_visibility_masks.npz")["masks"]
tex = np.squeeze(np.load(R / RUN / "geometry_patch/geometry_patch_texture.npz")["texture"])
if tex.shape[0] in (3, 4):
    tex = tex.transpose(1, 2, 0)
tex = tex[..., :3].astype(np.float64)
tex = tex / 255.0 if tex.max() > 1.5 else tex
mm_per_texel = PATCH_MM / tex.shape[0]

base = preprocess_like_vggt(summary["image_paths"][FRAME])
quad = corners[FRAME]
x0, y0 = quad[:, 0].min(), quad[:, 1].min()
x1, y1 = quad[:, 0].max(), quad[:, 1].max()
pad = 0.18 * max(x1 - x0, y1 - y0)
crop_box = (int(max(x0 - pad, 0)), int(max(y0 - pad, 0)),
            int(min(x1 + pad, base.width)), int(min(y1 + pad, base.height)))

# ATE measured by the jitter sweep at these magnitudes, for this run
tol = json.loads(Path("/tmp/jitter_eval/tolerance.json").read_text(encoding="utf-8"))
mm_axis, ate_axis = tol["mm"], tol["series"][RUN]["ate"]
SHOW_MM = [0.0, 1.60, 6.39]


def ate_at(mm: float) -> float:
    i = int(np.argmin([abs(m - mm) for m in mm_axis]))
    return ate_axis[i]


fig, ax = plt.subplots(2, 3, figsize=(12.5, 8.2))
ref = None
for k, mm in enumerate(SHOW_MM):
    shifted = tex if mm == 0 else ndimage.shift(
        tex, (mm / mm_per_texel, mm / mm_per_texel, 0), order=1, mode="nearest")
    img = Image.fromarray((np.clip(shifted, 0, 1) * 255).astype(np.uint8))
    over = composite_patch(base, img, quad, 1.0, "", masks[FRAME],
                           draw_outline=False, draw_text_label=False)
    crop = np.asarray(over.crop(crop_box), dtype=np.float64) / 255.0
    if ref is None:
        ref = crop
    ax[0, k].imshow(crop)
    a = ate_at(mm)
    # spell out the verdict; colour alone would read backwards to anyone who
    # takes green as "safe" rather than "the attack is working"
    verdict = "ATTACK WORKS" if a > 0.10 else ("weakened" if a > 0.02 else "ATTACK DEAD")
    ax[0, k].set_title(f"displacement {mm:.2f} mm", fontsize=11.5)
    ax[0, k].set_xlabel(f"ATE {a:.4f} m  ({a / 0.00816:.0f}x clean)\n{verdict}",
                        fontsize=11, color="#1b6b3a" if a > 0.10 else "#b3400f")
    d = np.abs(crop - ref).mean(2)
    ax[1, k].imshow(d, cmap="magma", vmin=0, vmax=max(d.max(), 1e-3))
    ax[1, k].set_title("pixel difference vs 0 mm", fontsize=10)
    ax[1, k].set_xlabel(f"max {d.max()*255:.1f}/255,  mean {d.mean()*255:.2f}/255",
                        fontsize=9.5)
for a in ax.ravel():
    a.set_xticks([]), a.set_yticks([])
fig.suptitle(f"{RUN}: a 1.6 mm slip changes ~3% of the pixels and costs 6x the ATE "
             f"({mm_per_texel:.1f} mm/texel)", fontsize=13)
fig.tight_layout(h_pad=2.6, rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "mon64_3_displacement.png", dpi=170, facecolor="white")
print(f"wrote {OUT/'mon64_3_displacement.png'}")
for mm in SHOW_MM:
    print(f"    位移 {mm:5.2f} mm -> ATE {ate_at(mm):.4f}")

# ---------------------------------------------------------------- fig 4
rec = ReconsEval(Path("/mnt/data/wangqq/recons_eval"))
umeyama, _, _ = rec.pointmap_fns()
work = Path("/tmp/fig_work")
work.mkdir(parents=True, exist_ok=True)
clean_run = load_run(R / "tum10_clean_uniform_l3", SCENE)
gt_traj, gt_c2w = gt_traj_for(rec, Path("/mnt/data/wangqq/recons_eval/data/tum"),
                              SCENE, clean_run["frame_indices"], "groundtruth_90.txt", work)


def cloud_in_gt_frame(run: str, conf_q: float = 0.6):
    """Point map mapped to the GT frame by the Sim(3) fitted on the cameras.

    Using the camera-fitted transform, not one fitted on the points, keeps this
    consistent with how the trajectory figure and the ATE are aligned.
    """
    d = np.load(R / run / SCENE / "vggt_outputs.npz")
    pts = d["point_map"].reshape(-1, 3)
    conf = d["point_conf"].reshape(-1)
    keep = conf >= np.quantile(conf, conf_q)
    pts = pts[keep]
    c2w = load_run(R / run, SCENE)["c2w"]
    c, Rm, t = umeyama(np.asarray(c2w)[:, :3, 3].T, gt_c2w[:, :3, 3].T)
    out = (c * Rm @ pts.T + t).T
    idx = np.random.default_rng(0).choice(len(out), size=min(60000, len(out)),
                                          replace=False)
    return out[idx]


PANELS = [("clean", "tum10_clean_uniform_l3"),
          ("attacked, 64x64", "res64_noeot"),
          ("attacked, 128x128", "res128_noeot")]
clouds = [(name, cloud_in_gt_frame(run)) for name, run in PANELS]
ref_cloud = clouds[0][1]
lo = np.percentile(ref_cloud, 1, axis=0)
hi = np.percentile(ref_cloud, 99, axis=0)
pad = 0.35 * (hi - lo)

fig4, ax4 = plt.subplots(1, 3, figsize=(15, 5.2))
for a, (name, pc) in zip(ax4, clouds):
    a.scatter(pc[:, 0], pc[:, 2], s=0.6, c=pc[:, 1], cmap="viridis", alpha=0.45,
              linewidths=0, rasterized=True)
    a.plot(gt_c2w[:, 0, 3], gt_c2w[:, 2, 3], "-o", color="#e34948", lw=2, ms=4,
           label="GT camera path")
    a.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    a.set_ylim(lo[2] - pad[2], hi[2] + pad[2])
    a.set_title(name, fontsize=11.5)
    a.set_xlabel("X (m)", fontsize=9.5)
    a.grid(alpha=0.15, lw=0.7)
    inside = ((pc[:, 0] > lo[0]) & (pc[:, 0] < hi[0]) &
              (pc[:, 2] > lo[2]) & (pc[:, 2] < hi[2])).mean()
    a.text(0.02, 0.02, f"{inside:.0%} of points inside the clean extent",
           transform=a.transAxes, fontsize=9.5, color="#52514e")
ax4[0].set_ylabel("Z (m)", fontsize=9.5)
ax4[0].legend(fontsize=9, loc="upper right")
fig4.suptitle("Predicted point cloud, top view, mapped into the GT frame "
              "(same Sim(3) as the ATE)", fontsize=13)
fig4.tight_layout(rect=(0, 0, 1, 0.95))
fig4.savefig(OUT / "mon64_4_pointcloud.png", dpi=150, facecolor="white")
print(f"wrote {OUT/'mon64_4_pointcloud.png'}")
for name, pc in clouds:
    spread = np.percentile(pc, 99, axis=0) - np.percentile(pc, 1, axis=0)
    print(f"    {name:<20} 点云 1-99% 跨度 X {spread[0]:.2f} Y {spread[1]:.2f} Z {spread[2]:.2f} m")
