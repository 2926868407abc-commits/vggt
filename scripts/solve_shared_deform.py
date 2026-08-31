"""Solve one trajectory deformation that survives every subset's Sim(3) fit.

For a global displacement field ``d`` and subset ``j``, ``S_j`` selects the
subset's frames and ``Q_j`` projects out the seven displacement directions that
one Sim(3) alignment can absorb.  We optimise

    max_d min_j ||Q_j S_j d||^2 / ||S_j d||^2

and add a second-difference penalty so the target is a smooth trajectory bend,
not independent per-frame noise.  This is CPU-only and does not run VGGT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack_vggt_geometry_tum10 import (  # noqa: E402
    orthogonal_mode_displacement,
    read_tum_rows,
    sim3_action_basis,
    tum_rows_to_c2w,
)


def load_subset_indices(plan_path: Path, scene: str) -> dict[str, list[int]]:
    """Load subset frame indices, tolerating a repository moved after creation."""
    entries = json.loads(plan_path.read_text(encoding="utf-8"))
    subsets: dict[str, list[int]] = {}
    for entry in entries:
        manifest_path = Path(entry["manifest"])
        if not manifest_path.exists():
            manifest_path = plan_path.parent / manifest_path.name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        subsets[str(entry["name"])] = [
            int(i) for i in manifest[scene]["frame_indices"]
        ]
    if not subsets:
        raise ValueError(f"No subsets found in {plan_path}")
    return subsets


def build_subset_projectors(
    positions: np.ndarray,
    subsets: dict[str, list[int]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build each subset's projector onto the complement of its Sim(3) span."""
    projectors: dict[str, torch.Tensor] = {}
    indices: dict[str, torch.Tensor] = {}
    for name, frame_indices in subsets.items():
        idx = np.asarray(frame_indices, dtype=np.int64)
        if idx.ndim != 1 or idx.size < 3:
            raise ValueError(f"Subset {name!r} must contain at least three frames")
        if idx.min() < 0 or idx.max() >= len(positions):
            raise IndexError(f"Subset {name!r} contains an out-of-range frame")
        basis = sim3_action_basis(positions[idx])
        q, _ = np.linalg.qr(basis.T)
        projectors[name] = torch.tensor(
            np.eye(q.shape[0]) - q @ q.T,
            dtype=torch.float64,
        )
        indices[name] = torch.tensor(idx, dtype=torch.long)
    return projectors, indices


def subset_survival_scores(
    deformation: torch.Tensor,
    projectors: dict[str, torch.Tensor],
    indices: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Fraction of each subset's squared deformation left after Sim(3) removal."""
    scores = []
    for name, idx in indices.items():
        flat = deformation[idx].reshape(-1)
        residual = projectors[name] @ flat
        scores.append((residual @ residual) / (flat @ flat).clamp_min(1e-12))
    return torch.stack(scores)


def second_difference_energy(deformation: torch.Tensor) -> torch.Tensor:
    """Temporal roughness measured by the mean squared second difference."""
    n = deformation.shape[0]
    if n < 3:
        return deformation.new_zeros(())
    second = deformation[2:] - 2 * deformation[1:-1] + deformation[:-2]
    return (second * second).sum() / (n - 2)


def solve_shared_deformation(
    positions: np.ndarray,
    subsets: dict[str, list[int]],
    *,
    smooth_weight: float = 0.05,
    steps: int = 4000,
    restarts: int = 6,
    learning_rate: float = 0.05,
    softmin_temperature: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a unit-Frobenius deformation and its per-subset survival scores."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected positions with shape (N, 3), got {positions.shape}")
    if steps < 1 or restarts < 1:
        raise ValueError("steps and restarts must both be positive")

    projectors, indices = build_subset_projectors(positions, subsets)
    best: tuple[float, torch.Tensor, torch.Tensor] | None = None
    n = len(positions)

    for seed in range(restarts):
        generator = torch.Generator().manual_seed(seed)
        deformation = torch.randn(
            n,
            3,
            generator=generator,
            dtype=torch.float64,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([deformation], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad()
            unit = deformation / deformation.norm().clamp_min(1e-12)
            scores = subset_survival_scores(unit, projectors, indices)
            soft_min = -torch.logsumexp(
                -scores * softmin_temperature,
                dim=0,
            ) / softmin_temperature
            objective = -soft_min + smooth_weight * second_difference_energy(unit)
            objective.backward()
            optimizer.step()

        with torch.no_grad():
            unit = (deformation / deformation.norm().clamp_min(1e-12)).detach()
            scores = subset_survival_scores(unit, projectors, indices)
        candidate = (float(scores.min()), unit.clone(), scores.clone())
        if best is None or candidate[0] > best[0]:
            best = candidate
        print(
            f"  seed {seed}: worst={float(scores.min()):.1%} "
            f"mean={float(scores.mean()):.1%}"
        )

    assert best is not None
    return best[1].numpy(), best[2].numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tum-root",
        type=Path,
        default=Path("/mnt/data/wangqq/recons_eval/data/tum"),
    )
    parser.add_argument(
        "--scene",
        default="rgbd_dataset_freiburg3_sitting_halfsphere_mid20",
    )
    parser.add_argument("--gt-name", default="groundtruth_90.txt")
    parser.add_argument(
        "--subset-plan",
        type=Path,
        default=REPO_ROOT / "data/tum10_split/mid_subset_plan.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/tum10_split/shared_deformation.npz",
    )
    parser.add_argument("--smooth-weight", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--softmin-temperature", type=float, default=40.0)
    parser.add_argument("--min-survival", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_tum_rows(args.tum_root / args.scene / args.gt_name)
    c2w = tum_rows_to_c2w(rows, list(range(len(rows))))
    positions = c2w[:, :3, 3]
    subsets = load_subset_indices(args.subset_plan, args.scene)
    sizes = sorted({len(value) for value in subsets.values()})
    print(f"pool={len(positions)} frames, subsets={len(subsets)}, sizes={sizes}")

    delta_unit, survivals = solve_shared_deformation(
        positions,
        subsets,
        smooth_weight=args.smooth_weight,
        steps=args.steps,
        restarts=args.restarts,
        learning_rate=args.learning_rate,
        softmin_temperature=args.softmin_temperature,
    )
    for name, score in zip(subsets, survivals.tolist()):
        print(f"  {name}: {score:6.1%}")
    print(
        f"solved: worst={survivals.min():.1%}, mean={survivals.mean():.1%}, "
        f"spread={survivals.max() - survivals.min():.1%}"
    )

    print("hand-built references:")
    projectors, indices = build_subset_projectors(positions, subsets)
    for order, axis in ((2, 0), (4, 1)):
        reference = orthogonal_mode_displacement(positions, order, axis, 1.0)
        reference_t = torch.tensor(
            reference / np.linalg.norm(reference),
            dtype=torch.float64,
        )
        scores = subset_survival_scores(reference_t, projectors, indices)
        print(
            f"  order={order} axis={axis}: "
            f"worst={float(scores.min()):.1%}, mean={float(scores.mean()):.1%}"
        )

    if float(survivals.min()) < args.min_survival:
        raise SystemExit(
            f"Rejected: worst survival {survivals.min():.1%} "
            f"is below {args.min_survival:.1%}"
        )

    radius = float(
        np.sqrt(((positions - positions.mean(0)) ** 2).sum(1).mean())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        delta_unit=delta_unit,
        survivals=survivals,
        subsets=np.asarray(list(subsets)),
        radius=radius,
        smooth_weight=float(args.smooth_weight),
    )
    print(f"trajectory radius={radius:.4f} m -> {args.output}")


if __name__ == "__main__":
    main()
