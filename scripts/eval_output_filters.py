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

Thresholds are calibrated against the *matching clean sequence*.  The old version
pooled one scalar from eight very different TUM motions and applied mean +/- 3
standard deviations globally.  That made ``conf_std``'s lower threshold negative,
so that check could never fire, and let sequence identity dominate the threshold.
Here each feature is measured per frame (or per adjacent pair for reprojection),
and an attacked sequence is compared with the empirical clean distribution from
the same sequence.  No ground truth or attacked sample is used for calibration.
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


def samples_conf(d, reference=None) -> dict[str, np.ndarray]:
    c = d["depth_conf"]
    ref = c if reference is None else reference["depth_conf"]
    # VGGT confidence has an exact activation floor.  Use the clean sequence's
    # floor for both clean and attacked outputs; using each attack's own minimum
    # would silently change the statistic being compared.
    floor = float(ref.min()) + 1e-6
    flat = c.reshape(c.shape[0], -1)
    return {
        "conf_std": flat.std(axis=1),
        "conf_frac_floor": (flat <= floor).mean(axis=1),
    }


def samples_head(d) -> dict[str, np.ndarray]:
    diff = np.linalg.norm(d["point_map"] - d["point_cloud_unproj"], axis=-1)
    points = d["point_map"].reshape(d["point_map"].shape[0], -1, 3)
    center = points.mean(axis=1, keepdims=True)
    radius = np.sqrt(((points - center) ** 2).sum(axis=-1).mean(axis=1))
    med = np.median(diff.reshape(diff.shape[0], -1), axis=1)
    values = np.divide(med, radius, out=np.full_like(med, np.nan), where=radius > 0)
    return {"head_disagree_rel": values}


def samples_reproj(d, stride: int = 4) -> dict[str, np.ndarray]:
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
    return {"reproj_rel_err": np.asarray(errs, dtype=np.float64)}


def feature_samples(d, reference=None) -> dict[str, np.ndarray]:
    out = {}
    out.update(samples_conf(d, reference))
    out.update(samples_head(d))
    out.update(samples_reproj(d))
    return out


def summarize_samples(samples: dict[str, np.ndarray]) -> dict[str, float]:
    row = {}
    for name in FEATURES:
        values = np.asarray(samples.get(name, []), dtype=np.float64)
        values = values[np.isfinite(values)]
        row[name] = float(np.median(values)) if values.size else float("nan")
    return row


def matched_thresholds(samples: dict[str, np.ndarray], quantile: float) -> dict[str, float]:
    thresholds = {}
    for name in FEATURES:
        values = np.asarray(samples.get(name, []), dtype=np.float64)
        values = values[np.isfinite(values)]
        if not values.size:
            thresholds[name] = float("nan")
            continue
        q = 1.0 - quantile if WORSE_IS[name] == "low" else quantile
        thresholds[name] = float(np.quantile(values, q))
    return thresholds


def fires(value: float, threshold: float, feature: str) -> bool:
    if not np.isfinite(value) or not np.isfinite(threshold):
        return False
    return value < threshold if WORSE_IS[feature] == "low" else value > threshold


FEATURES = ["conf_std", "conf_frac_floor", "head_disagree_rel", "reproj_rel_err"]
# direction each feature moves when the output degrades
WORSE_IS = {"conf_std": "low", "conf_frac_floor": "high",
            "head_disagree_rel": "high", "reproj_rel_err": "high"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="", help="comma list; default = all cal_* runs")
    p.add_argument("--out_csv", default="/tmp/output_filters.csv")
    p.add_argument("--out_budget_json", default=None,
                   help="optional machine-readable matched-clean thresholds for the "
                        "budgeted attack objective")
    p.add_argument("--calibration", choices=("matched", "global"), default="matched",
                   help="matched uses per-frame clean statistics from the same sequence; "
                        "global preserves the old pooled-scene 3-sigma protocol")
    p.add_argument("--matched_quantile", type=float, default=0.95,
                   help="one-sided empirical clean quantile for matched calibration")
    p.add_argument("--n_sigma", type=float, default=3.0)
    return p.parse_args()


def main():
    args = parse_args()
    clean_scenes = sorted(p.name for p in (ROOT / CLEAN_RUN).iterdir()
                          if p.is_dir() and p.name.startswith("rgbd_"))

    if not 0.5 < args.matched_quantile < 1.0:
        raise ValueError("--matched_quantile must be between 0.5 and 1")

    print(f"=== calibration: {args.calibration} ===")
    clean_rows = []
    clean_data = {}
    clean_samples = {}
    matched_thr = {}
    for scene in clean_scenes:
        d = load(CLEAN_RUN, scene)
        if d is None:
            continue
        clean_data[scene] = d
        sample = feature_samples(d, d)
        clean_samples[scene] = sample
        matched_thr[scene] = matched_thresholds(sample, args.matched_quantile)
        row = {"run": CLEAN_RUN, "seq": scene, "attacked": 0}
        row.update(summarize_samples(sample))
        row["conf_mean"] = float(d["depth_conf"].mean())
        clean_rows.append(row)
        detail = []
        for f in FEATURES:
            direction = "<" if WORSE_IS[f] == "low" else ">"
            detail.append(f"{f}={row[f]:.4g} [{direction}{matched_thr[scene][f]:.4g}]")
        print(f"  {scene:<44} " + "  ".join(detail))

    global_thr = {}
    for f in FEATURES:
        v = np.array([r[f] for r in clean_rows if np.isfinite(r[f])])
        mu, sd = v.mean(), v.std()
        global_thr[f] = ((mu - args.n_sigma * sd) if WORSE_IS[f] == "low"
                         else (mu + args.n_sigma * sd))
        if args.calibration == "global":
            print(f"\n  {f:<20} clean {mu:.5g} +- {sd:.5g}  -> threshold "
                  f"{'<' if WORSE_IS[f]=='low' else '>'} {global_thr[f]:.5g}")

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
            reference = clean_data.get(scene)
            if reference is None:
                print(f"  SKIP {run}/{scene}: no matching clean sequence")
                continue
            row = {"run": run, "seq": scene, "attacked": 1}
            row.update(summarize_samples(feature_samples(d, reference)))
            row["conf_mean"] = float(d["depth_conf"].mean())
            thresholds = matched_thr[scene] if args.calibration == "matched" else global_thr
            fired = [f for f in FEATURES if fires(row[f], thresholds[f], f)]
            for f in FEATURES:
                row[f"threshold_{f}"] = thresholds[f]
            row["calibration"] = args.calibration
            row["fired"] = ";".join(fired)
            rows.append(row)
            print(f"{run:<34}" + "".join(f"{row[f]:>11.4g}" for f in FEATURES)
                  + f"  {len(fired)}/{len(FEATURES)} {','.join(x.split('_')[0] for x in fired)}")

    att = [r for r in rows if r["attacked"]]
    print(f"\n=== detection rate over {len(att)} attacked runs ===")
    for f in FEATURES:
        hit = 0
        for r in att:
            thresholds = matched_thr[r["seq"]] if args.calibration == "matched" else global_thr
            hit += int(fires(r[f], thresholds[f], f))
        # Cost estimate at frame/pair level.  The sequence median used for detection
        # is intentionally more conservative than this empirical tail rate.
        tail_hits = tail_total = 0
        for scene, sample in clean_samples.items():
            threshold = matched_thr[scene][f] if args.calibration == "matched" else global_thr[f]
            values = np.asarray(sample[f], dtype=np.float64)
            values = values[np.isfinite(values)]
            tail_hits += sum(fires(float(v), threshold, f) for v in values)
            tail_total += len(values)
        rate = hit / len(att) * 100 if att else 0.0
        tail_rate = tail_hits / tail_total * 100 if tail_total else 0.0
        print(f"  {f:<20} detects {hit}/{len(att)} ({rate:5.1f}%)   "
              f"clean frame/pair tail {tail_hits}/{tail_total} ({tail_rate:4.1f}%)")
    anyhit = sum(1 for r in att if r["fired"])
    print(f"  {'ANY of them':<20} detects {anyhit}/{len(att)} ({anyhit/len(att)*100:5.1f}%)")

    out = Path(args.out_csv)
    with out.open("w", newline="", encoding="utf-8") as fh:
        threshold_fields = [f"threshold_{f}" for f in FEATURES]
        wr = csv.DictWriter(fh, fieldnames=["run", "seq", "attacked", "calibration",
                                            *FEATURES, *threshold_fields, "conf_mean", "fired"],
                            extrasaction="ignore")
        wr.writeheader(); wr.writerows(rows)
    print(f"\nSaved -> {out}")

    if args.out_budget_json:
        if args.calibration != "matched":
            raise ValueError("--out_budget_json requires --calibration matched")
        payload = {
            "version": 1,
            "clean_run": CLEAN_RUN,
            "matched_quantile": args.matched_quantile,
            "features": FEATURES,
            "worse_is": WORSE_IS,
            "sequences": {},
        }
        for row in clean_rows:
            scene = row["seq"]
            payload["sequences"][scene] = {
                "depth_conf_floor": float(clean_data[scene]["depth_conf"].min()) + 1e-6,
                "metrics": {
                    name: {
                        "clean": float(row[name]),
                        "threshold": float(matched_thr[scene][name]),
                        "worse": WORSE_IS[name],
                    }
                    for name in FEATURES
                },
            }
        budget_out = Path(args.out_budget_json)
        budget_out.parent.mkdir(parents=True, exist_ok=True)
        budget_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved matched budgets -> {budget_out}")


if __name__ == "__main__":
    main()
