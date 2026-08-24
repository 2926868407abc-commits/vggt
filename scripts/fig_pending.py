"""Two figures the doc listed as pending whose data was already on disk.

fig 7-1  the Sim(3) invariance sweep. 18 global similarity transforms applied to a
         clean trajectory; the aligned metrics must not move. Plotted on a log axis
         against the unaligned ATE, because the point is the ~15-orders-of-magnitude
         gap between "what the attack changed" and "what the metric sees".

fig 14-4 the four output filters on the budget runs, each margin shown against its
         own threshold. Raw values are not comparable across filters, so each is
         normalised by its threshold: 1.0 is the firing line.

Labels are English -- the server has no CJK font.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VG = Path("/mnt/data/wangqq/vggt")
METH = VG / "outputs/figures/method"
CUR = VG / "outputs/figures/current"
for d in (METH, CUR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- fig 7-1
# The CSV holds two groups. Rows with kind sim3/identity are the invariance sweep
# and must not move the aligned metric. Rows with kind control_non_sim3 are
# deliberate non-gauge damage (per-frame noise, progressive scale drift) and must
# move it -- without them the sweep only shows the test is quiet, not that it works.
rows = list(csv.DictReader(
    (VG / "outputs/diag_gauge_invariance/A_sim3_invariance_pose.csv").open(encoding="utf-8")))
sweep = [r for r in rows if r["kind"] in ("sim3", "identity")]
ctrl = [r for r in rows if r["kind"] == "control_non_sim3"]
ident = [r for r in rows if r["kind"] == "identity"]
ref = float(ident[0]["ATE_align_scale"]) if ident else float(sweep[0]["ATE_align_scale"])

sw = np.array([abs(float(r["ATE_align_scale"]) - ref) / ref for r in sweep])
ct = np.array([abs(float(r["ATE_align_scale"]) - ref) / ref for r in ctrl])
sw_un = np.array([float(r["ATE_noalign"]) for r in sweep])

fig, ax = plt.subplots(figsize=(12, 5.8))
xs = np.arange(len(sweep))
xc = np.arange(len(sweep), len(sweep) + len(ctrl))
FLOOR = 1e-17
ax.semilogy(xs, np.maximum(sw, FLOOR), "o", color="#2a78d6", ms=8,
            label=f"global Sim(3), n={len(sweep)} — must not move")
ax.semilogy(xc, np.maximum(ct, FLOOR), "s", color="#e34948", ms=10,
            label=f"non-Sim(3) control, n={len(ctrl)} — must move")
ax.axhline(1e-14, color="#898781", ls="--", lw=1.3)
ax.text(0, 1.5e-14, "float64 noise floor", fontsize=9.5, color="#52514e")
ax.axvline(len(sweep) - 0.5, color="#c3c2b7", lw=1.2)
def short_ctrl(label: str) -> str:
    """The control labels carry their parameters inline and get clipped otherwise."""
    if "scale drift" in label:
        return "CONTROL: scale drift 1→1.5"
    if "0.0005" in label:
        return "CONTROL: per-frame noise 0.05×extent"
    if "0.002" in label:
        return "CONTROL: per-frame noise 0.2×extent"
    return "CONTROL: " + label[:28]


ax.set_xticks(list(xs) + list(xc))
ax.set_xticklabels([r["label"] for r in sweep] + [short_ctrl(r["label"]) for r in ctrl],
                   rotation=58, ha="right", fontsize=7.5)
ax.set_ylabel("relative change in aligned ATE", fontsize=10)
ax.set_ylim(FLOOR, 10)
ax.set_title("The aligned metric is blind to a global Sim(3), and not blind in general\n"
             f"scene sitting_static · unaligned ATE over the sweep spans "
             f"{sw_un.min():.2f}–{sw_un.max():.2f} m while aligned ATE moves "
             f"{sw.max():.1e}", fontsize=11.5)
ax.legend(fontsize=10, loc="center left")
ax.grid(alpha=0.18, lw=0.8, which="both")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(METH / "method_sim3_invariance.png", dpi=170, facecolor="white")
print(f"wrote {METH/'method_sim3_invariance.png'}")
print(f"    Sim(3) 组 相对变化最大 {sw.max():.3e}   "
      f"未对齐 ATE {sw_un.min():.3f}~{sw_un.max():.3f} m")
print(f"    非 Sim(3) 对照组 相对变化 {ct.min():.3e}~{ct.max():.3e}  "
      f"（比 Sim(3) 组大 {ct.max()/max(sw.max(),1e-30):.1e} 倍）")

# ---------------------------------------------------------------- fig 14-4
frows = list(csv.DictReader(
    (VG / "reports/budget_xyz_eval_20260817/budget_xyz_pareto_filters.csv")
    .open(encoding="utf-8")))
FILTERS = [("conf_std", "conf std"), ("conf_frac_floor", "conf frac floor"),
           ("head_disagree_rel", "head disagreement"), ("reproj_rel_err", "reproj error")]
att = [r for r in frows if r.get("attacked") == "1"]
cln = [r for r in frows if r.get("attacked") == "0"]
groups = [("clean", cln, "#898781"), ("attacked (budget runs)", att, "#2a78d6")]

fig2, ax2 = plt.subplots(figsize=(10, 5.4))
w = 0.35
for gi, (name, rs, col) in enumerate(groups):
    for fi, (key, label) in enumerate(FILTERS):
        vals = []
        for r in rs:
            try:
                v, t = float(r[key]), float(r["threshold_" + key])
                if t:
                    vals.append(v / t)
            except (ValueError, KeyError, TypeError):
                pass
        if not vals:
            continue
        pos = fi + (gi - 0.5) * w
        ax2.bar(pos, float(np.mean(vals)), w * 0.9, color=col,
                label=name if fi == 0 else None, zorder=3)
        ax2.scatter([pos] * len(vals), vals, s=14, color="#0b0b0b", alpha=0.55, zorder=5)
ax2.axhline(1.0, color="#e34948", lw=2, ls="--", zorder=4)
ax2.text(len(FILTERS) - 0.55, 1.03, "firing threshold", color="#e34948", fontsize=10,
         fontweight="bold", ha="right")
ax2.set_xticks(range(len(FILTERS)))
ax2.set_xticklabels([l for _, l in FILTERS], fontsize=10)
ax2.set_ylabel("margin / its own threshold", fontsize=10)
ax2.set_title("Output filters on the budget runs: none fires\n"
              "each filter normalised by its own threshold; dots are individual "
              "sequences", fontsize=11.5)
ax2.legend(fontsize=9.5)
ax2.grid(axis="y", alpha=0.18, lw=0.8)
ax2.set_axisbelow(True)
fig2.tight_layout()
fig2.savefig(CUR / "budget_filters.png", dpi=170, facecolor="white")
print(f"wrote {CUR/'budget_filters.png'}")
print(f"    attacked 行 {len(att)}  clean 行 {len(cln)}")
fired = [r for r in frows if str(r.get("fired", "")).strip() not in ("", "0", "False")]
print(f"    触发过的行: {len(fired)}")
