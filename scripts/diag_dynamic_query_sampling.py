#!/usr/bin/env python3
"""Measure Track-query coverage for dynamic frame-set sampling modes."""

import argparse
import json
from pathlib import Path

import numpy as np


def draw_indices(pool, frame_count, rng, mode):
    if mode == "earliest":
        return sorted(
            int(value)
            for value in rng.choice(pool, size=frame_count, replace=False)
        )
    query = int(rng.choice(pool))
    remaining = [value for value in pool if value != query]
    targets = sorted(
        int(value)
        for value in rng.choice(
            remaining, size=frame_count - 1, replace=False
        )
    )
    return [query, *targets]


def summarize(pool_size, frame_count, draws, seed, mode):
    rng = np.random.default_rng(seed)
    pool = list(range(pool_size))
    queries = np.asarray([
        draw_indices(pool, frame_count, rng, mode)[0]
        for _ in range(draws)
    ])
    quarter = pool_size // 4
    quartiles = np.minimum(queries // quarter, 3)
    return {
        "mode": mode,
        "draws": draws,
        "query_mean_index": float(queries.mean()),
        "query_median_index": float(np.median(queries)),
        "query_first_quarter_fraction": float((queries < quarter).mean()),
        "query_quartile_counts": np.bincount(quartiles, minlength=4).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_size", type=int, default=608)
    parser.add_argument("--frame_count", type=int, default=10)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = {
        mode: summarize(
            args.pool_size, args.frame_count, args.draws, args.seed, mode
        )
        for mode in ("earliest", "uniform_query")
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
