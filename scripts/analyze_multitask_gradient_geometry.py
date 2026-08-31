"""Post-process diag_multitask_gradients.py without another model forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


TASKS = ("pose", "depth", "point", "track")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("details")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    details = json.loads(Path(args.details).read_text(encoding="utf-8"))
    output = []
    for detail in details:
        gram = np.asarray([
            [detail["cosines"][left][right] for right in TASKS]
            for left in TASKS
        ], dtype=np.float64)

        nullspace_remaining = {}
        for index, task in enumerate(TASKS):
            others = [i for i in range(len(TASKS)) if i != index]
            other_gram = gram[np.ix_(others, others)]
            cross = gram[others, index]
            coeff = np.linalg.solve(other_gram + 1e-8 * np.eye(len(others)), cross)
            explained = float(np.clip(cross @ coeff, 0.0, 1.0))
            nullspace_remaining[task] = float(np.sqrt(max(0.0, 1.0 - explained)))

        # Find the convex combination of unit task gradients with the largest
        # worst first-order gain.  A positive margin proves that one combined
        # ascent direction improves all four tasks locally.
        n = len(TASKS)
        objective = np.r_[np.zeros(n), -1.0]
        a_ub = np.c_[-gram, np.ones(n)]
        b_ub = np.zeros(n)
        a_eq = np.asarray([[1.0] * n + [0.0]])
        b_eq = np.asarray([1.0])
        result = linprog(
            objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
            bounds=[(0.0, None)] * n + [(None, None)], method="highs"
        )
        if not result.success:
            raise RuntimeError(result.message)
        weights = result.x[:n]
        gains = gram @ weights
        row = {
            "state": detail["state"],
            "nullspace_remaining": nullspace_remaining,
            "maximin_weights": {task: float(weights[i]) for i, task in enumerate(TASKS)},
            "maximin_raw_gains": {task: float(gains[i]) for i, task in enumerate(TASKS)},
            "maximin_margin": float(result.x[-1]),
        }
        output.append(row)

        print(f"\n=== {detail['state']} ===")
        print("remaining gradient after protecting all other tasks:")
        for task in TASKS:
            print(f"  {task:<6} {nullspace_remaining[task]:.3f}")
        print("max-min unit-gradient mixture:")
        print("  weights " + " ".join(
            f"{task}={weights[i]:.3f}" for i, task in enumerate(TASKS)
        ))
        print("  gains   " + " ".join(
            f"{task}={gains[i]:.3f}" for i, task in enumerate(TASKS)
        ))
        print(f"  worst raw gain={result.x[-1]:.3f}")

    out = Path(args.out) if args.out else Path(args.details).with_name(
        "gradient_geometry.json"
    )
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
