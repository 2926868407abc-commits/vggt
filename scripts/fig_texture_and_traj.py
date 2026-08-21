"""Attack-effect figures that need no GPU.

fig1  the optimised textures themselves, annotated with their autocorrelation
      half-width in millimetres on the printed sheet -- the shift at which the
      pattern stops matching itself. That is the same physical quantity the
      tolerance sweep measures from the other end, so the two can be checked
      against each other instead of the resolution story resting on eyeballing.

      (A per-texture "high-frequency share" was tried first and discarded: it
      normalises to each texture's own Nyquist, so 128 and 32 are measured against
      different physical scales and the numbers are not comparable.)

fig2  the trajectories, Sim(3)-aligned to GT exactly as the ATE estimator aligns
      them. Alignment is not optional: ~90% of what the attack does to the
      trajectory is a global similarity the evaluation removes, so an unaligned
      plot would advertise damage the metric never sees.

Labels are English -- the server has no CJK font installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import ReconsEval, gt_traj_for, load_run  # noqa: E402

R = VG / "outputs_attack_geometry_aware_tum10"
# figures/ is split into current/ method/ archive_128era/ -- see its README. The
# mon64_ prefix keeps these from colliding with the older fig1..fig9 set.
OUT = VG / "outputs/figures/current"
OUT.mkdir(parents=True, exist_ok=True)
SCENE = "rgbd_dataset_freiburg3_sitting_halfsphere"
HAZARD = VG / "assets/hazard_textures/mde_attack_warnning.png"
PATCH_MM = 638.87


def load_tex(run: str) -> np.ndarray:
    t = np.squeeze(np.load(R / run / "geometry_patch/geometry_patch_texture.npz")["texture"])
    if t.ndim == 3 and t.shape[0] in (3, 4):
        t = t.transpose(1, 2, 0)
    t = t[..., :3].astype(np.float64)
    return t / 255.0 if t.max() > 1.5 else t


def autocorr_halfwidth_mm(img: np.ndarray) -> float:
    """Shift, in mm on the printed patch, at which the pattern decorrelates to 0.5.

    Computed via the Wiener-Khinchin route and averaged over shift direction, so
    it is one number regardless of the pattern's orientation.
    """
    g = img.mean(2)
    g = g - g.mean()
    F = np.fft.fft2(g)
    ac = np.real(np.fft.ifft2(F * np.conj(F)))
    ac = np.fft.fftshift(ac)
    ac /= ac.max()
    n = g.shape[0]
    mm_per_texel = PATCH_MM / n
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - n // 2, xx - n // 2)
    # radial profile, then first crossing of 0.5 with linear interpolation
    edges = np.arange(0, n // 2 + 1)
    prof = np.array([ac[(r >= e) & (r < e + 1)].mean() if ((r >= e) & (r < e + 1)).any()
                     else np.nan for e in edges])
    for i in range(1, len(prof)):
        if np.isfinite(prof[i]) and prof[i] < 0.5:
            a, b = prof[i - 1], prof[i]
            frac = (a - 0.5) / max(a - b, 1e-9)
            return float((i - 1 + frac) * mm_per_texel)
    return float((n // 2) * mm_per_texel)


# ---------------------------------------------------------------- fig 1
# Half-lives are per-axis maximum displacement. The jitter translation lives in
# grid_sample coordinates, which span [-1, 1] across the texture, so a magnitude m
# is m/2 of the patch width -- half what a naive reading of the flag would suggest.
RUNS = [("res128_noeot", "128x128", 5.0, "ATE half-life 0.64 mm"),
        ("res64_noeot", "64x64", 10.0, "ATE half-life 1.6 mm"),
        ("res32_noeot", "32x32", 20.0, "ATE half-life 3.2 mm")]

init = np.asarray(Image.open(HAZARD).convert("RGB").resize((128, 128),
                  Image.Resampling.LANCZOS), dtype=np.float64) / 255.0

def init_at(n: int) -> np.ndarray:
    return np.asarray(Image.open(HAZARD).convert("RGB").resize((n, n),
                      Image.Resampling.LANCZOS), dtype=np.float64) / 255.0


# Two rows: what the patch looks like, and the adversarial part on its own.
# The full texture's autocorrelation is dominated by the hazard stripes it was
# initialised from and pulled back toward, so it measures the visible pattern
# rather than the perturbation that does the attacking. The difference from the
# init isolates the latter.
fig, ax = plt.subplots(2, 4, figsize=(14, 8.6))
ax[0, 0].imshow(init, interpolation="nearest")
ax[0, 0].set_title("Init texture\n(hazard stripes)", fontsize=11)
ax[0, 0].set_xlabel(f"full pattern: {autocorr_halfwidth_mm(init):.1f} mm", fontsize=9.5)
ax[1, 0].axis("off")
ax[1, 0].text(0.5, 0.5,
              "Bottom row = texture minus init,\n"
              "i.e. only what the optimiser added.\n\n"
              "Its decorrelation length tracks the\n"
              "measured placement tolerance;\n"
              "the full pattern's does not.",
              ha="center", va="center", fontsize=10, color="#333333",
              transform=ax[1, 0].transAxes)
rows = []
for k, (run, name, mm_per_texel, note) in enumerate(RUNS, start=1):
    t = load_tex(run)
    d = t - init_at(t.shape[0])
    hw_full, hw_pert = autocorr_halfwidth_mm(t), autocorr_halfwidth_mm(d)
    rows.append((name, mm_per_texel, hw_full, hw_pert, note))
    ax[0, k].imshow(t, interpolation="nearest")
    ax[0, k].set_title(f"{name}   ({mm_per_texel:.1f} mm/texel)", fontsize=11)
    ax[0, k].set_xlabel(f"full pattern: {hw_full:.1f} mm", fontsize=9.5)
    lim = max(abs(d).max(), 1e-6)
    ax[1, k].imshow(d.mean(2), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax[1, k].set_title("adversarial part only", fontsize=10.5)
    ax[1, k].set_xlabel(f"decorrelates at {hw_pert:.1f} mm\n{note}",
                        fontsize=10, color="#b3400f")
for a in ax.ravel():
    if a.axison:
        a.set_xticks([]), a.set_yticks([])
fig.suptitle("Patch textures (top) and the adversarial perturbation alone (bottom)",
             fontsize=13)
# the default pad lets the top row's xlabel collide with the bottom row's title
fig.tight_layout(h_pad=3.2, rect=(0, 0, 1, 0.97))
fig.savefig(OUT / "mon64_1_textures.png", dpi=170, facecolor="white")
print(f"wrote {OUT/'mon64_1_textures.png'}")
print(f"    {'':<9}{'完整图案':>10}{'仅扰动':>10}   实测容差")
print(f"    {'init':<9}{autocorr_halfwidth_mm(init):>8.2f}mm{'-':>10}")
for name, mpt, hwf, hwp, note in rows:
    print(f"    {name:<9}{hwf:>8.2f}mm{hwp:>8.2f}mm   {note}")

# ---------------------------------------------------------------- fig 2
rec = ReconsEval(Path("/mnt/data/wangqq/recons_eval"))
# the evaluator's own Umeyama, so the plot aligns exactly the way the ATE does
umeyama, _, _ = rec.pointmap_fns()
work = Path("/tmp/fig_work")
work.mkdir(parents=True, exist_ok=True)
clean = load_run(R / "tum10_clean_uniform_l3", SCENE)
gt_traj, gt_c2w = gt_traj_for(rec, Path("/mnt/data/wangqq/recons_eval/data/tum"),
                              SCENE, clean["frame_indices"], "groundtruth_90.txt", work)


def aligned_xyz(c2w: np.ndarray) -> np.ndarray:
    """Camera centres after the Sim(3) the ATE estimator fits.

    umeyama returns t already shaped (3, 1), so it broadcasts against the (3, N)
    point matrix as-is.
    """
    p = np.asarray(c2w)[:, :3, 3].T
    c, Rm, t = umeyama(p, gt_c2w[:, :3, 3].T)
    return (c * Rm @ p + t).T


series = [("GT", gt_c2w[:, :3, 3], "#0b0b0b", "-", 2.6),
          ("clean prediction", aligned_xyz(clean["c2w"]), "#898781", "--", 2.0),
          ("attacked, 64x64", aligned_xyz(load_run(R / "res64_noeot", SCENE)["c2w"]),
           "#eb6834", "-", 2.4),
          ("attacked, 128x128", aligned_xyz(load_run(R / "res128_noeot", SCENE)["c2w"]),
           "#2a78d6", "-", 2.4)]

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5.2))
for j, (i0, i1, lab) in enumerate([(0, 2, "top view (X-Z)"), (0, 1, "front view (X-Y)")]):
    for name, xyz, col, ls, lw in series:
        ax2[j].plot(xyz[:, i0], xyz[:, i1], ls, color=col, lw=lw, label=name,
                    marker="o", ms=4.5, mew=0)
        ax2[j].scatter(xyz[0, i0], xyz[0, i1], s=95, facecolor="none",
                       edgecolor=col, lw=1.8, zorder=5)
    ax2[j].set_title(lab, fontsize=11)
    ax2[j].set_xlabel("metres", fontsize=9.5)
    ax2[j].grid(alpha=0.18, lw=0.8)
    ax2[j].set_aspect("equal", adjustable="datalim")
ax2[0].legend(fontsize=9.5, framealpha=0.92)
fig2.suptitle("Camera trajectories after the evaluator's Sim(3) alignment "
              "(open circle = frame 0)", fontsize=13)
fig2.tight_layout()
fig2.savefig(OUT / "mon64_2_trajectory.png", dpi=170, facecolor="white")
print(f"wrote {OUT/'mon64_2_trajectory.png'}")
for name, xyz, *_ in series[1:]:
    d = np.linalg.norm(xyz - gt_c2w[:, :3, 3], axis=1)
    print(f"    {name:<20} 对齐后逐帧位置误差 均值 {d.mean():.4f} m  最大 {d.max():.4f} m")
