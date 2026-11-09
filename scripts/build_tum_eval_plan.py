#!/usr/bin/env python3
"""Build and audit a frozen held-out TUM window plan for one context size."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = REPO_ROOT / "data/tum_per_scene_splits"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_windows(frames: list[int], window: int, stride: int) -> list[list[int]]:
    if window < 2 or not 0 < stride < window:
        raise ValueError("require window >= 2 and 0 < stride < window")
    if len(frames) < window:
        raise ValueError(f"only {len(frames)} held-out frames for window={window}")
    starts = list(range(0, len(frames) - window + 1, stride))
    final_start = len(frames) - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return [frames[start : start + window] for start in starts]


def write_frozen(path: Path, payload: object) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a different frozen plan: {path}")
    path.write_text(text, encoding="utf-8")


def build_split(
    split: str,
    protocol: dict,
    window: int,
    stride: int,
) -> tuple[list[dict], dict]:
    pool_path = SPLIT_ROOT / f"{split}_train_pools.json"
    pools = json.loads(pool_path.read_text(encoding="utf-8"))
    entries: list[dict] = []
    audit: dict[str, dict] = {}
    for scene in protocol["scenes"]:
        count = int(protocol["counts"][scene]["all_synchronized"])
        train = [int(v) for v in pools[scene]]
        train_set = set(train)
        if len(train_set) != len(train):
            raise AssertionError(f"{scene}/{split}: duplicate training frame")
        test = [frame for frame in range(count) if frame not in train_set]
        if train_set & set(test) or train_set | set(test) != set(range(count)):
            raise AssertionError(f"{scene}/{split}: train/test partition is invalid")
        windows = make_windows(test, window, stride)
        covered = set().union(*(set(frames) for frames in windows))
        if covered != set(test):
            raise AssertionError(f"{scene}/{split}: windows do not cover held-out split")
        min_overlap = min(
            len(set(left) & set(right))
            for left, right in zip(windows, windows[1:])
        )
        if min_overlap < 1:
            raise AssertionError(f"{scene}/{split}: adjacent windows do not overlap")
        for index, frames in enumerate(windows):
            if train_set & set(frames):
                raise AssertionError(f"{scene}/{split}/w{index}: training leakage")
            entries.append(
                {
                    "name": f"{scene}__{split}_test_w{window}_s{stride}_{index:04d}",
                    "scene": scene,
                    "frames": frames,
                    "window": window,
                    "stride": stride,
                    "contains_training_frames": False,
                    "training_pool_sha256": sha256(pool_path),
                }
            )
        audit[scene] = {
            "total_frames": count,
            "train_frames": len(train),
            "heldout_frames": len(test),
            "windows": len(windows),
            "min_adjacent_overlap": min_overlap,
            "heldout_coverage": len(covered),
            "train_test_intersection": 0,
        }
    return entries, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    protocol_path = SPLIT_ROOT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    tag = f"w{args.window}_s{args.stride}"
    combined_audit = {
        "window": args.window,
        "stride": args.stride,
        "protocol_sha256": sha256(protocol_path),
        "splits": {},
    }
    for split in ("firsthalf", "evenodd"):
        entries, audit = build_split(split, protocol, args.window, args.stride)
        plan_path = SPLIT_ROOT / f"{split}_test_eval_plan_{tag}.json"
        write_frozen(plan_path, entries)
        combined_audit["splits"][split] = {
            "plan": str(plan_path),
            "plan_sha256": sha256(plan_path),
            "scenes": audit,
        }
    audit_path = SPLIT_ROOT / f"eval_plan_{tag}_audit.json"
    write_frozen(audit_path, combined_audit)
    print(json.dumps(combined_audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
