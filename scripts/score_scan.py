"""Score a frozen-texture position scan and draw ATE and RPE heat maps.

ATE is reported as a fraction of the sequence-specific RMS-radius ceiling.  Both
the ceiling and clean baseline are recomputed from the selected sequence instead
of using constants from sitting_halfsphere.  Position spread is only called
distinguishable when the caller supplies a measured same-configuration seed SD.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import ReconsEval, gt_traj_for, load_run  # noqa: E402


DEFAULT_PLANES = VG / "outputs/candidate_planes"
DEFAULT_RUNS = VG / "outputs_attack_geometry_aware_tum10"
DEFAULT_RECONS = Path("/mnt/data/wangqq/recons_eval")


def w2c_to_c2w(extrinsic: np.ndarray) -> np.ndarray:
    """Convert ``(N, 3, 4)`` world-to-camera matrices to camera-to-world."""
    n = extrinsic.shape[0]
    output = np.tile(np.eye(4), (n, 1, 1))
    output[:, :3, :3] = np.transpose(extrinsic[:, :3, :3], (0, 2, 1))
    output[:, :3, 3] = -np.einsum(
        "nij,nj->ni",
        output[:, :3, :3],
        extrinsic[:, :3, 3],
    )
    return output


def trajectory_ceiling(positions: np.ndarray) -> float:
    """RMS radius: the collapse ceiling of aligned ATE for this GT trajectory."""
    positions = np.asarray(positions, dtype=np.float64)
    centred = positions - positions.mean(axis=0, keepdims=True)
    return float(np.sqrt((centred * centred).sum(axis=1).mean()))


def draw_heatmaps(
    groups: list[tuple[int, list[dict]]],
    *,
    metric: str,
    grid_size: int,
    title: str,
    output: Path,
    baseline: float | None = None,
    percent: bool = False,
) -> None:
    if not groups:
        raise ValueError("No valid planes to draw")
    values = np.asarray(
        [record[metric] for _, records in groups for record in records],
        dtype=np.float64,
    )
    vmin = float(baseline) if baseline is not None else float(values.min())
    vmax = max(float(values.max()), vmin + 1e-9)
    scale = 100.0 if percent else 1.0

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(3.4 * len(groups), 3.8),
        squeeze=False,
        layout="constrained",
    )
    for ax, (plane, records) in zip(axes[0], groups):
        grid = np.full((grid_size, grid_size), np.nan)
        for record in records:
            grid[int(record["iv"]), int(record["iu"])] = float(record[metric])
        spread = float(np.nanmax(grid) - np.nanmin(grid))
        image = ax.imshow(
            grid * scale,
            origin="lower",
            cmap="magma",
            vmin=vmin * scale,
            vmax=vmax * scale,
        )
        spread_text = f"{spread:.1%}" if percent else f"{spread:.2f}"
        ax.set_title(f"plane {plane}\nspread {spread_text}", fontsize=10)
        ax.set_xlabel("c_u")
        ax.set_ylabel("c_v")
        plt.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=12)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    parser.add_argument("--area", type=float, default=0.05)
    parser.add_argument("--planes-dir", type=Path, default=DEFAULT_PLANES)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--recons-root", type=Path, default=DEFAULT_RECONS)
    parser.add_argument("--tum-root", type=Path, default=DEFAULT_RECONS / "data/tum")
    parser.add_argument("--clean-run", default="tum10_clean_uniform_l3")
    parser.add_argument(
        "--seed-sd",
        type=float,
        default=None,
        help="measured same-configuration SD of ATE/ceiling; omit when unavailable",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/scan_work"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = args.planes_dir / f"{args.scene}_scanres_a{args.area}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    records = [record for record in result["records"] if "extrinsic" in record]
    if not records:
        raise ValueError(f"No successful candidates in {result_path}")

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
    if ceiling <= 1e-12:
        raise ValueError("GT trajectory has a zero ATE ceiling")
    clean_traj = evaluator.evo_utils.get_tum_poses(np.asarray(clean["c2w"]))
    clean_fraction = float(evaluator.ate(clean_traj, gt_traj, True, True)) / ceiling

    for record in records:
        c2w = w2c_to_c2w(np.asarray(record["extrinsic"], dtype=np.float64))
        trajectory = evaluator.evo_utils.get_tum_poses(c2w)
        _, rotation = evaluator.rpe(trajectory, gt_traj, True, True)
        record["ate_frac"] = (
            float(evaluator.ate(trajectory, gt_traj, True, True)) / ceiling
        )
        record["rpe_rot"] = float(rotation)

    groups = [
        (plane, [record for record in records if int(record["plane"]) == plane])
        for plane in sorted({int(record["plane"]) for record in records})
    ]
    seed_text = "unknown" if args.seed_sd is None else f"{args.seed_sd:.1%}"
    print(
        f"{args.scene}  area={args.area} m²  candidates={len(records)}\n"
        f"ceiling={ceiling:.6f} m  clean={clean_fraction:.1%}  seed SD={seed_text}"
    )
    print("plane  count  best ATE  worst ATE  ATE spread  best RPE  distinguishable")
    for plane, plane_records in groups:
        ate = np.asarray([record["ate_frac"] for record in plane_records])
        rpe = np.asarray([record["rpe_rot"] for record in plane_records])
        spread = float(ate.max() - ate.min())
        distinguishable = (
            "unknown"
            if args.seed_sd is None
            else ("yes" if spread > 2.0 * args.seed_sd else "no")
        )
        print(
            f"{plane:>5}  {len(plane_records):>5}  {ate.max():>8.1%}  "
            f"{ate.min():>9.1%}  {spread:>10.1%}  {rpe.max():>8.2f}  "
            f"{distinguishable}"
        )

    best = max(records, key=lambda record: record["ate_frac"])
    print(
        f"best ATE: plane={best['plane']} grid=({best['iu']},{best['iv']}) "
        f"ATE={best['ate_frac']:.1%} RPE={best['rpe_rot']:.2f}°"
    )

    grid_size = int(result.get("grid", 8))
    ate_output = args.planes_dir / f"{args.scene}_heatmap_ate_a{args.area}.png"
    rpe_output = args.planes_dir / f"{args.scene}_heatmap_rpe_a{args.area}.png"
    draw_heatmaps(
        groups,
        metric="ate_frac",
        grid_size=grid_size,
        title=f"Position sensitivity · ATE / ceiling · area {args.area} m²",
        output=ate_output,
        baseline=clean_fraction,
        percent=True,
    )
    draw_heatmaps(
        groups,
        metric="rpe_rot",
        grid_size=grid_size,
        title=f"Position sensitivity · RPE rotation (degrees) · area {args.area} m²",
        output=rpe_output,
    )

    scored_path = args.planes_dir / f"{args.scene}_scored_a{args.area}.json"
    scored_path.write_text(
        json.dumps(
            {
                "scene": args.scene,
                "area_m2": args.area,
                "ceiling": ceiling,
                "clean_frac": clean_fraction,
                "seed_sd": args.seed_sd,
                "records": records,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"-> {ate_output}\n-> {rpe_output}\n-> {scored_path}")


if __name__ == "__main__":
    main()
