"""Tolerance curve as a PNG, so the set in figures/current is pasteable into a doc.

The interactive HTML version carries the same numbers plus a data table; this is
the static twin of it. Reads the measured JSON, never hard-coded values.

Labels are English -- the server has no CJK font.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VG = Path("/mnt/data/wangqq/vggt")
OUT = VG / "outputs/figures/current"
OUT.mkdir(parents=True, exist_ok=True)
TOL = Path("/tmp/jitter_eval/tolerance.json")

CLEAN_ATE, RANDOM_FRAC = 0.00816, 0.905
SERIES = [("res128_noeot", "128x128  (5.0 mm/texel)", "#2a78d6"),
          ("res64_noeot", "64x64  (10.0 mm/texel)", "#eb6834"),
          ("res32_noeot", "32x32  (20.0 mm/texel)", "#1baf7a")]

d = json.loads(TOL.read_text(encoding="utf-8"))
mm, ceiling = d["mm"], d["ceiling_ate"]
rand = RANDOM_FRAC * ceiling
x = np.arange(len(mm))  # samples are log-spaced; plot ordinally and label the ticks

fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.axhspan(rand, ceiling * 1.05, color="#0b0b0b", alpha=0.05, zorder=0)
ax.axhline(rand, color="#898781", ls="--", lw=1.4, zorder=1)
ax.axhline(CLEAN_ATE, color="#898781", ls="--", lw=1.4, zorder=1)
ax.text(len(mm) - 1, rand, f"  random trajectory level {rand:.3f}  (metric saturated)",
        ha="right", va="bottom", fontsize=9, color="#52514e")
# left-anchored: the curves all converge onto this line at the right-hand end
ax.text(0, CLEAN_ATE, f"clean baseline {CLEAN_ATE:.4f}  ",
        ha="left", va="bottom", fontsize=9, color="#52514e")

for key, label, col in SERIES:
    s = d["series"][key]
    ate = np.asarray(s["ate"])
    sd = np.asarray(s["ate_sd"])
    ax.fill_between(x, np.maximum(ate - sd, 0), ate + sd, color=col, alpha=0.15, lw=0)
    ax.plot(x, ate, "-o", color=col, lw=2.2, ms=5.5, label=label, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels([f"{m:.2f}".rstrip("0").rstrip(".") for m in mm])
ax.set_xlabel("placement error (mm on the printed patch, per-axis maximum; "
              "samples log-spaced)", fontsize=10)
ax.set_ylabel("ATE (m)", fontsize=10)
ax.set_ylim(0, ceiling * 1.05)
ax.grid(alpha=0.18, lw=0.8)
ax.legend(fontsize=10, framealpha=0.93, loc="center right")
ax.set_title("How far the patch can slip before the attack stops working\n"
             f"TUM fr3 sitting_halfsphere · 639 mm patch · 6 draws per point · "
             f"shading = ±1 s.d. · ATE ceiling {ceiling:.4f} m", fontsize=11.5)
fig.tight_layout()
fig.savefig(OUT / "mon64_5_tolerance.png", dpi=170, facecolor="white")
print(f"wrote {OUT/'mon64_5_tolerance.png'}")
for key, label, _ in SERIES:
    a = d["series"][key]["ate"]
    print(f"    {label:<26} 0mm {a[0]:.4f} -> {mm[4]:.2f}mm {a[4]:.4f} "
          f"-> {mm[6]:.2f}mm {a[6]:.4f}")
