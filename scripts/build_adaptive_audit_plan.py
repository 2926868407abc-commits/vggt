#!/usr/bin/env python3
"""Build disjoint, deterministic in-training audit frame sets for one TUM scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tum_gt_pool import open_pool, query_count


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) < count:
        raise ValueError(f"Need {count} frames but only {len(values)} are available")
    positions = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    result = [int(values[index]) for index in positions]
    if len(result) != len(set(result)):
        raise ValueError("Even spacing produced duplicate frame indices")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools-json", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--gt-pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames-per-set", type=int, default=10)
    parser.add_argument("--sets", type=int, default=4)
    parser.add_argument("--min-queries", type=int, default=20)
    args = parser.parse_args()

    pools = json.loads(Path(args.pools_json).read_text(encoding="utf-8"))
    pool_frames = [int(value) for value in pools[args.scene]]
    if len(pool_frames) != len(set(pool_frames)):
        raise ValueError("Training pool contains duplicate frames")
    pool_frames.sort()
    if len(pool_frames) < args.frames_per_set * args.sets:
        raise ValueError("Training pool is too small for disjoint audit sets")

    gt = open_pool(args.gt_pool, args.scene)
    chunks = np.array_split(np.asarray(pool_frames, dtype=np.int64), args.sets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for group, chunk_array in enumerate(chunks):
        chunk = [int(value) for value in chunk_array]
        selected = None
        for start_pos, query_frame in enumerate(chunk):
            suffix = chunk[start_pos:]
            if len(suffix) < args.frames_per_set:
                break
            if query_count(gt, query_frame) < args.min_queries:
                continue
            candidate = evenly_spaced(suffix, args.frames_per_set)
            if candidate[0] == query_frame:
                selected = candidate
                break
        if selected is None:
            raise RuntimeError(
                f"No valid query frame with {args.min_queries} points in chunk {group}"
            )

        manifest_path = output_dir / f"audit_group_{group:02d}_manifest.json"
        manifest_path.write_text(
            json.dumps({args.scene: selected}, indent=2) + "\n",
            encoding="utf-8",
        )
        entries.append({
            "name": f"{args.scene}_audit_{group:02d}",
            "scene": str(Path(args.scene_dir)),
            "manifest": str(manifest_path),
            "gt_dir": str(Path(args.gt_pool)),
        })

    plan_path = output_dir / "audit_plan.json"
    plan_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(plan_path)
    for entry in entries:
        manifest = json.loads(Path(entry["manifest"]).read_text(encoding="utf-8"))
        frames = manifest[args.scene]
        print(entry["name"], frames, "queries", query_count(gt, frames[0]))


if __name__ == "__main__":
    main()
