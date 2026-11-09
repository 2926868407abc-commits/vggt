#!/usr/bin/env python3
"""Build frozen sliding-window plans over the training side of a TUM split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_windows(frames: list[int], window: int, stride: int) -> list[list[int]]:
    if window < 2 or not 0 < stride < window:
        raise ValueError("require window >= 2 and 0 < stride < window")
    if len(frames) < window:
        raise ValueError(f"only {len(frames)} training frames for window={window}")
    starts = list(range(0, len(frames) - window + 1, stride))
    final_start = len(frames) - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return [frames[start : start + window] for start in starts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--train_pools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="firsthalf")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    pools = json.loads(args.train_pools.read_text(encoding="utf-8"))
    entries = []
    audit = {}
    for scene in protocol["scenes"]:
        frames = [int(value) for value in pools[scene]]
        if len(frames) != len(set(frames)):
            raise ValueError(f"duplicate training frame in {scene}")
        windows = make_windows(frames, args.window, args.stride)
        covered = set().union(*(set(window) for window in windows))
        if covered != set(frames):
            raise AssertionError(f"training windows do not cover {scene}")
        for index, window_frames in enumerate(windows):
            entries.append({
                "name": (
                    f"{scene}__{args.split}_train_"
                    f"w{args.window}_s{args.stride}_{index:04d}"
                ),
                "scene": scene,
                "frames": window_frames,
                "window": args.window,
                "stride": args.stride,
                "contains_training_frames": True,
                "training_pool_sha256": sha256(args.train_pools),
            })
        audit[scene] = {
            "training_frames": len(frames),
            "windows": len(windows),
            "coverage": len(covered),
        }

    payload = json.dumps(entries, indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"refusing to overwrite different plan: {args.output}")
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
