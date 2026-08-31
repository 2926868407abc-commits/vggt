"""Measure whether short training preserves the long-run ranking of positions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from diag_gauge_invariance import ReconsEval, gt_traj_for, load_run  # noqa: E402
from score_scan import trajectory_ceiling  # noqa: E402


DEFAULT_PLANES = REPO_ROOT / "outputs/candidate_planes"
DEFAULT_RUNS = REPO_ROOT / "outputs_attack_geometry_aware_tum10"
DEFAULT_RECONS = Path("/mnt/data/wangqq/recons_eval")


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values with average ranks for ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    rank_a = average_ranks(np.asarray(values_a, dtype=np.float64))
    rank_b = average_ranks(np.asarray(values_b, dtype=np.float64))
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = float(np.sqrt((rank_a * rank_a).sum() * (rank_b * rank_b).sum()))
    return float((rank_a * rank_b).sum() / denominator) if denominator > 1e-12 else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    parser.add_argument("--area", type=float, default=0.2)
    parser.add_argument("--short", type=int, default=300)
    parser.add_argument("--long", type=int, default=1000)
    parser.add_argument("--planes-dir", type=Path, default=DEFAULT_PLANES)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--recons-root", type=Path, default=DEFAULT_RECONS)
    parser.add_argument("--tum-root", type=Path, default=DEFAULT_RECONS / "data/tum")
    parser.add_argument("--clean-run", default="tum10_clean_uniform_l3")
    parser.add_argument("--short-template", default="pb{steps}_{tag}")
    parser.add_argument("--long-template", default="pb{steps}_{tag}")
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--rho-threshold", type=float, default=0.8)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/probe_work"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_path = args.planes_dir / f"{args.scene}_probe_a{args.area}.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    tags = [candidate["tag"] for candidate in probe["candidates"]]

    evaluator = ReconsEval(args.recons_root)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    clean = load_run(args.runs_root / args.clean_run, args.scene)
    gt_traj, _ = gt_traj_for(
        evaluator,
        args.tum_root,
        args.scene,
        clean["frame_indices"],
        "groundtruth_90.txt",
        args.work_dir,
    )
    ceiling = trajectory_ceiling(np.asarray(gt_traj.positions_xyz))
    clean_traj = evaluator.evo_utils.get_tum_poses(np.asarray(clean["c2w"]))
    clean_fraction = float(evaluator.ate(clean_traj, gt_traj, True, True)) / ceiling

    def score(run_name: str) -> tuple[float, float] | None:
        if not (args.runs_root / run_name / args.scene / "attack_summary.json").exists():
            return None
        trajectory = evaluator.evo_utils.get_tum_poses(
            np.asarray(load_run(args.runs_root / run_name, args.scene)["c2w"])
        )
        _, rotation = evaluator.rpe(trajectory, gt_traj, True, True)
        ate = float(evaluator.ate(trajectory, gt_traj, True, True)) / ceiling
        return ate, float(rotation)

    rows = []
    for tag in tags:
        short_name = args.short_template.format(steps=args.short, tag=tag)
        long_name = args.long_template.format(steps=args.long, tag=tag)
        rows.append((tag, score(short_name), score(long_name)))

    print(f"ceiling={ceiling:.6f} m  clean={clean_fraction:.1%}")
    print("position                  short ATE    long ATE   ratio   short RPE   long RPE")
    short_values, long_values, missing = [], [], []
    for tag, short_score, long_score in rows:
        if short_score is None or long_score is None:
            missing.append(tag)
            print(f"{tag:<25} missing")
            continue
        short_values.append(short_score[0])
        long_values.append(long_score[0])
        print(
            f"{tag:<25}{short_score[0]:>9.1%}{long_score[0]:>12.1%}"
            f"{long_score[0] / max(short_score[0], 1e-9):>8.1f}x"
            f"{short_score[1]:>11.2f}°{long_score[1]:>11.2f}°"
        )

    if len(short_values) < args.min_samples:
        raise SystemExit(
            f"Only {len(short_values)} complete pairs; need {args.min_samples}. "
            f"Missing: {missing}"
        )
    short_array = np.asarray(short_values)
    long_array = np.asarray(long_values)
    rho = spearman(short_array, long_array)
    pearson = float(np.corrcoef(short_array, long_array)[0, 1])
    print(
        f"Spearman={rho:.3f}  Pearson={pearson:.3f}\n"
        f"short spread={short_array.max() - short_array.min():.1%}  "
        f"long spread={long_array.max() - long_array.min():.1%}"
    )
    verdict = "proxy accepted" if rho > args.rho_threshold else "proxy rejected"
    print(f"{verdict}: threshold={args.rho_threshold:.2f}")


if __name__ == "__main__":
    main()
