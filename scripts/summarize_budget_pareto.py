"""Build the ATE/ceiling-versus-filter-trigger table for budgeted attacks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_one_csv(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, got {len(rows)}")
    return rows[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="comma-separated run names")
    p.add_argument("--filter_csv", required=True)
    p.add_argument("--output_root", default=(
        "/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10"))
    p.add_argument("--gauge_root", default=(
        "/mnt/data/wangqq/recons_eval/outputs/relpose-distance"))
    p.add_argument("--out_md", required=True)
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    runs = [s.strip() for s in args.runs.split(",") if s.strip()]
    with Path(args.filter_csv).open(newline="", encoding="utf-8") as f:
        filter_rows = {row["run"]: row for row in csv.DictReader(f) if row["attacked"] == "1"}

    rows = []
    for run in runs:
        root = Path(args.output_root) / run
        meta = json.loads((root / "geometry_patch/geometry_patch_meta.json").read_text(
            encoding="utf-8"))
        history = [json.loads(line) for line in (
            root / "geometry_patch/training_history.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
        gauge = read_one_csv(Path(args.gauge_root) / f"tum10-gauge-vggt_{run}.csv")
        filt = filter_rows[run]
        budget = meta.get("filter_budget") or {}
        joint = meta.get("joint_gauge") or {}
        violation_keys = sorted({key for row in history for key in row
                                 if key.startswith("budget_violation_")})
        max_violation = max((float(row.get(key, 0.0)) for row in history
                             for key in violation_keys), default=0.0)
        rows.append({
            "run": run,
            "objective": budget.get("pose_objective"),
            "budget_fraction": budget.get("fraction"),
            "order": joint.get("orthogonal_mode_order"),
            "ATE": float(gauge["ATE_align_scale"]),
            "ATE_ceiling_pct": 100.0 * float(gauge["ate_frac_of_ceiling"]),
            "clean_ceiling_pct": 100.0 * float(gauge["ate_clean_frac_of_ceiling"]),
            "devRel": float(gauge["dev_from_clean_rel"]),
            "gauge_absorbed_pct": 100.0 * float(gauge["gauge_absorbed_frac"]),
            "fired_count": len([x for x in filt["fired"].split(";") if x]),
            "fired": filt["fired"] or "none",
            "conf_std": float(filt["conf_std"]),
            "conf_frac_floor": float(filt["conf_frac_floor"]),
            "head_disagree_rel": float(filt["head_disagree_rel"]),
            "reproj_rel_err": float(filt["reproj_rel_err"]),
            "max_train_violation": max_violation,
            "final_primary": float(history[-1]["budget_primary_pose"]),
        })

    columns = ["run", "objective", "budget_fraction", "order", "ATE",
               "ATE_ceiling_pct", "clean_ceiling_pct", "devRel",
               "gauge_absorbed_pct", "fired_count", "fired", "conf_std",
               "conf_frac_floor", "head_disagree_rel", "reproj_rel_err",
               "max_train_violation", "final_primary"]
    if args.out_csv:
        with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(rows)

    lines = [
        "# Budgeted attack Pareto table", "",
        "| run | objective | budget | order | ATE/ceiling | devRel | gauge absorbed | filters | max train violation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['objective']} | {row['budget_fraction']} | "
            f"{row['order']} | {row['ATE_ceiling_pct']:.2f}% | {row['devRel']:.3f} | "
            f"{row['gauge_absorbed_pct']:.1f}% | {row['fired_count']}/4 "
            f"({row['fired']}) | {row['max_train_violation']:.3g} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
