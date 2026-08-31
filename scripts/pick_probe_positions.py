"""Pick spatially spread positions for short-vs-long training correlation tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANES = REPO_ROOT / "outputs/candidate_planes"


def pick_spread_positions(
    candidates: list[dict],
    *,
    max_positions: int = 8,
    max_per_plane: int = 4,
) -> list[dict]:
    """Select corners/centre per plane, then interleave planes fairly."""
    planes = sorted({int(candidate["plane"]) for candidate in candidates})
    pool: list[dict] = []
    labels = ("bottom_left", "top_right", "top_left", "bottom_right", "centre")
    for plane in planes:
        subset = [candidate for candidate in candidates if int(candidate["plane"]) == plane]
        iu = np.asarray([candidate["iu"] for candidate in subset], dtype=np.float64)
        iv = np.asarray([candidate["iv"] for candidate in subset], dtype=np.float64)
        u = (iu - iu.mean()) / max(float(iu.std()), 1e-9)
        v = (iv - iv.mean()) / max(float(iv.std()), 1e-9)
        indices = (
            int(np.argmin(u + v)),
            int(np.argmax(u + v)),
            int(np.argmin(u - v)),
            int(np.argmax(u - v)),
            int(np.argmin(np.hypot(u, v))),
        )
        seen: set[tuple[int, int]] = set()
        for label, index in zip(labels, indices):
            key = (int(subset[index]["iu"]), int(subset[index]["iv"]))
            if key in seen:
                continue
            seen.add(key)
            pool.append({**subset[index], "tag": f"p{plane}_{label}"})

    selected: list[dict] = []
    counts = {plane: 0 for plane in planes}
    # Round-robin over the per-plane candidate lists so a large plane cannot
    # consume all probe slots before another plane is represented.
    by_plane = {plane: [item for item in pool if int(item["plane"]) == plane]
                for plane in planes}
    depth = 0
    while len(selected) < max_positions:
        added = False
        for plane in planes:
            if counts[plane] >= max_per_plane or depth >= len(by_plane[plane]):
                continue
            selected.append(by_plane[plane][depth])
            counts[plane] += 1
            added = True
            if len(selected) >= max_positions:
                break
        if not added:
            break
        depth += 1
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    parser.add_argument("--area", type=float, default=0.2)
    parser.add_argument("--planes-dir", type=Path, default=DEFAULT_PLANES)
    parser.add_argument("--max-positions", type=int, default=8)
    parser.add_argument("--max-per-plane", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.planes_dir / f"{args.scene}_scan_a{args.area}.json"
    spec = json.loads(source.read_text(encoding="utf-8"))
    selected = pick_spread_positions(
        spec["candidates"],
        max_positions=args.max_positions,
        max_per_plane=args.max_per_plane,
    )
    print(
        f"area={args.area} m²; selected {len(selected)} of "
        f"{len(spec['candidates'])} candidates"
    )
    print("tag                      plane     grid       c_u       c_v     fill")
    for candidate in selected:
        cell = f"({candidate['iu']},{candidate['iv']})"
        print(
            f"{candidate['tag']:<24}{candidate['plane']:>5}{cell:>10}"
            f"{candidate['cu']:>10.3f}{candidate['cv']:>10.3f}"
            f"{candidate['fill']:>9.0%}"
        )

    output = args.out or args.planes_dir / f"{args.scene}_probe_a{args.area}.json"
    output.write_text(
        json.dumps(
            {
                "scene": args.scene,
                "area_m2": args.area,
                "w": spec["w"],
                "h": spec["h"],
                "source_spec": str(source),
                "candidates": selected,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"-> {output}")


if __name__ == "__main__":
    main()
