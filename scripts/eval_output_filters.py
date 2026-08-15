"""How well would a downstream filter catch these attacks, and what does it cost?

Nothing in the current pipeline inspects VGGT's own outputs for self-consistency:
the TUM pose eval reads `extrinsic` and nothing else. So an attack is free to
produce a trajectory error while leaving obvious traces elsewhere, and we would
never see it. This measures three filters that a consumer of VGGT can apply
directly to its outputs, with no ground truth and no extra model:

  conf        -- VGGT's own depth_conf. Note it has no pose confidence at all, so
                 this cannot gate the trajectory directly; it gates the geometry a
                 consumer would use alongside it.
  head        -- point_map vs the depth head unprojected into world coordinates.
                 Two heads that came from one geometry agree; a large gap means
                 the attacked outputs are mutually contradictory.
  reproj      -- couples pose and depth: unproject frame i, move it into frame j
                 with the predicted poses, and compare against frame j's own depth.
                 This is the check that a tracking front end effectively performs.

Thresholds are calibrated on the clean run's own scenes rather than picked by
hand, so "detected" means "outside the range this model produces when unattacked".
The clean sequences also give the false-positive rate, which is the cost side of
the question.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10")
CLEAN_RUN = "tum10_clean_uniform_l3"


def load(run: str, scene: str):
    f = ROOT / run / scene / "vggt_outputs.npz"
    if not f.exists():
        return None
    with np.load(f) as d:
        out = {k: d[k].astype(np.float64) for k in
               ("extrinsic", "intrinsic", "depth", "depth_conf", "point_map",
                "point_conf", "point_cloud_unproj")}
    w2c = np.tile(np.eye(4), (out["extrinsic"].shape[0], 1, 1))
    w2c[:, :3, :4] = out["extrinsic"]
    out["w2c"] = w2c
    out["c2w"] = np.linalg.inv(w2c)
    return out


def feat_conf(d) -> dict:
    c = d["depth_conf"]
    lo = c.min()
    return {
        "conf_mean": float(c.mean()),
        "conf_std": float(c.std()),
        "conf_frac_floor": float((c <= lo + 1e-6).mean()),
    }


def feat_head(d) -> dict:
    diff = np.linalg.norm(d["point_map"] - d["point_cloud_unproj"], axis=-1)
    flat = d["point_map"].reshape(-1, 3)
    radius = float(np.sqrt(((flat - flat.mean(0)) ** 2).sum(1).mean()))
    return {"head_disagree_rel": float(np.median(diff)) / radius if radius > 0 else float("nan")}


def feat_reproj(d, stride: int = 4) -> dict:
    """Unproject frame i, move into frame j with the predicted poses, compare depth."""
    depth = d["depth"][..., 0] if d["depth"].ndim == 4 else d["depth"]
    n, h, w = depth.shape
    K, c2w, w2c = d["intrinsic"], d["c2w"], d["w2c"]
    ys, xs = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride), indexing="ij")
    errs = []
    for i in range(n - 1):
        j = i + 1
        z = depth[i][::stride, ::stride]
        ok = z > 1e-6
        if not ok.any():
            continue
        x = (xs - K[i][0, 2]) / K[i][0, 0] * z
        y = (ys - K[i][1, 2]) / K[i][1, 1] * z
        cam_i = np.stack([x, y, z], axis=-1)
        world = cam_i @ c2w[i][:3, :3].T + c2w[i][:3, 3]
        cam_j = world @ w2c[j][:3, :3].T + w2c[j][:3, 3]
        zj = cam_j[..., 2]
        good = ok & (zj > 1e-6)
        if not good.any():
            continue
        u = K[j][0, 0] * cam_j[..., 0] / np.where(zj > 1e-6, zj, 1) + K[j][0, 2]
        v = K[j][1, 1] * cam_j[..., 1] / np.where(zj > 1e-6, zj, 1) + K[j][1, 2]
        good &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if good.sum() < 50:
            continue
        du = np.clip(np.round(u[good]).astype(int), 0, w - 1)
        dv = np.clip(np.round(v[good]).astype(int), 0, h - 1)
        z_expect = zj[good]
        z_actual = depth[j][dv, du]
        m = z_actual > 1e-6
        if m.sum() < 50:
            continue
        errs.append(np.median(np.abs(z_expect[m] - z_actual[m]) / z_actual[m]))
    return {"reproj_rel_err": float(np.median(errs)) if errs else float("nan")}


FEATURES = ["conf_std", "conf_frac_floor", "head_disagree_rel", "reproj_rel_err"]
# direction each feature moves when the output degrades
WORSE_IS = {"conf_std": "low", "conf_frac_floor": "high",
            "head_disagree_rel": "high", "reproj_rel_err": "high"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="", help="comma list; default = all cal_* runs")
    p.add_argument("--out_csv", default="/tmp/output_filters.csv")
    p.add_argument("--n_sigma", type=float, default=3.0)
    return p.parse_args()


def main():
    args = parse_args()
    clean_scenes = sorted(p.name for p in (ROOT / CLEAN_RUN).iterdir()
                          if p.is_dir() and p.name.startswith("rgbd_"))

    print("=== calibrating on the clean run's own scenes ===")
    clean_rows = []
    for scene in clean_scenes:
        d = load(CLEAN_RUN, scene)
        if d is None:
            continue
        row = {"run": CLEAN_RUN, "seq": scene, "attacked": 0}
        row.update(feat_conf(d)); row.update(feat_head(d)); row.update(feat_reproj(d))
        clean_rows.append(row)
        print(f"  {scene:<44} " + "  ".join(f"{k}={row[k]:.4g}" for k in FEATURES))

    thr = {}
    for f in FEATURES:
        v = np.array([r[f] for r in clean_rows if np.isfinite(r[f])])
        mu, sd = v.mean(), v.std()
        thr[f] = (mu - args.n_sigma * sd) if WORSE_IS[f] == "low" else (mu + args.n_sigma * sd)
        print(f"\n  {f:<20} clean {mu:.5g} +- {sd:.5g}  -> threshold "
              f"{'<' if WORSE_IS[f]=='low' else '>'} {thr[f]:.5g}")

    runs = ([s.strip() for s in args.runs.split(",") if s.strip()] or
            sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("cal_")))

    print(f"\n=== attacked runs vs those thresholds ===")
    hdr = f"{'run':<34}" + "".join(f"{f.split('_')[0][:8]:>11}" for f in FEATURES) + "  fired"
    print(hdr); print("-" * len(hdr))
    rows = list(clean_rows)
    for run in runs:
        scenes = [p.name for p in (ROOT / run).iterdir()
                  if p.is_dir() and p.name.startswith("rgbd_")]
        for scene in scenes:
            d = load(run, scene)
            if d is None:
                continue
            row = {"run": run, "seq": scene, "attacked": 1}
            row.update(feat_conf(d)); row.update(feat_head(d)); row.update(feat_reproj(d))
            fired = [f for f in FEATURES if np.isfinite(row[f]) and
                     (row[f] < thr[f] if WORSE_IS[f] == "low" else row[f] > thr[f])]
            row["fired"] = ";".join(fired)
            rows.append(row)
            print(f"{run:<34}" + "".join(f"{row[f]:>11.4g}" for f in FEATURES)
                  + f"  {len(fired)}/{len(FEATURES)} {','.join(x.split('_')[0] for x in fired)}")

    att = [r for r in rows if r["attacked"]]
    print(f"\n=== detection rate over {len(att)} attacked runs ===")
    for f in FEATURES:
        hit = sum(1 for r in att if np.isfinite(r[f]) and
                  (r[f] < thr[f] if WORSE_IS[f] == "low" else r[f] > thr[f]))
        fp = sum(1 for r in clean_rows if np.isfinite(r[f]) and
                 (r[f] < thr[f] if WORSE_IS[f] == "low" else r[f] > thr[f]))
        print(f"  {f:<20} detects {hit}/{len(att)} ({hit/len(att)*100:5.1f}%)   "
              f"false positives on clean {fp}/{len(clean_rows)}")
    anyhit = sum(1 for r in att if r["fired"])
    print(f"  {'ANY of them':<20} detects {anyhit}/{len(att)} ({anyhit/len(att)*100:5.1f}%)")

    out = Path(args.out_csv)
    with out.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["run", "seq", "attacked", *FEATURES,
                                            "conf_mean", "fired"], extrasaction="ignore")
        wr.writeheader(); wr.writerows(rows)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
