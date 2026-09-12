#!/usr/bin/env python3
"""Freeze per-scene first-half and even/odd TUM training protocols."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


VGGT_ROOT = Path("/mnt/data/wangqq/vggt")
SOURCE_PROTOCOL = VGGT_ROOT / "data/tum_universal_all/protocol.json"
OUT_ROOT = VGGT_ROOT / "data/tum_per_scene_splits"
WINDOW = 30
STRIDE = 20


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checked(path: Path, payload) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite a different frozen file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def preview(indices: list[int], count: int = 10) -> list[int]:
    if len(indices) < count:
        raise ValueError(f"Only {len(indices)} frames; need {count}")
    positions = [round(i * (len(indices) - 1) / (count - 1)) for i in range(count)]
    result = [indices[pos] for pos in positions]
    if len(set(result)) != count:
        raise AssertionError("Preview sampling produced duplicate frames")
    return result


def windows(indices: list[int]) -> list[list[int]]:
    if len(indices) < WINDOW:
        raise ValueError(f"Only {len(indices)} test frames; need at least {WINDOW}")
    result = []
    start = 0
    while start + WINDOW <= len(indices):
        result.append(indices[start : start + WINDOW])
        start += STRIDE
    final = indices[-WINDOW:]
    if not result or final != result[-1]:
        result.append(final)
    covered = set().union(*(set(item) for item in result))
    if covered != set(indices):
        missing = sorted(set(indices) - covered)
        raise AssertionError(f"Window coverage misses {missing[:10]}")
    return result


def main() -> None:
    source = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    scenes = list(source["scenes"])
    counts = source["counts"]

    pools = {"firsthalf": {}, "evenodd": {}}
    manifests = {"firsthalf": {}, "evenodd": {}}
    eval_plans = {"firsthalf": [], "evenodd": []}
    summary = {}

    for scene in scenes:
        n = int(counts[scene]["all_synchronized"])
        split = n // 2
        split_rows = {
            "firsthalf": (list(range(split)), list(range(split, n))),
            "evenodd": (list(range(0, n, 2)), list(range(1, n, 2))),
        }
        summary[scene] = {"all_synchronized": n}

        for split_name, (train, test) in split_rows.items():
            if set(train) & set(test):
                raise AssertionError(f"{scene}/{split_name}: train/test overlap")
            if set(train) | set(test) != set(range(n)):
                raise AssertionError(f"{scene}/{split_name}: train/test union mismatch")
            pools[split_name][scene] = train
            manifests[split_name][scene] = {
                "sampling": f"per_scene_{split_name}_dynamic_pool_preview",
                "frame_count_requested": 10,
                "frame_indices": preview(train),
                "split": split_name,
                "note": "Preview only; formal training draws ten frames from the frozen pool.",
            }
            scene_windows = windows(test)
            for window_index, frames in enumerate(scene_windows):
                manifest_path = OUT_ROOT / (
                    f"{scene}__{split_name}_test_w{window_index:03d}_manifest.json"
                )
                write_checked(
                    manifest_path,
                    {
                        scene: {
                            "sampling": "heldout_30_frame_window_stride_20",
                            "frame_count_requested": WINDOW,
                            "frame_indices": frames,
                            "split": split_name,
                            "contains_training_frames": False,
                        }
                    },
                )
                eval_plans[split_name].append(
                    {
                        "name": f"{scene}__{split_name}_test_w{window_index:03d}",
                        "scene": scene,
                        "manifest": str(manifest_path),
                        "frames": frames,
                        "contains_training_frames": False,
                    }
                )
            summary[scene][split_name] = {
                "train_frames": len(train),
                "test_frames": len(test),
                "test_windows_w30_s20": len(scene_windows),
                "train_first": train[0],
                "train_last": train[-1],
                "test_first": test[0],
                "test_last": test[-1],
            }

    paths = {}
    for split_name in ("firsthalf", "evenodd"):
        pool_path = OUT_ROOT / f"{split_name}_train_pools.json"
        preview_path = OUT_ROOT / f"{split_name}_train_preview_manifest.json"
        plan_path = OUT_ROOT / f"{split_name}_test_eval_plan_w30_s20.json"
        write_checked(pool_path, pools[split_name])
        write_checked(preview_path, manifests[split_name])
        write_checked(plan_path, eval_plans[split_name])
        paths[split_name] = {
            "train_pools": str(pool_path),
            "train_preview_manifest": str(preview_path),
            "test_eval_plan": str(plan_path),
            "gt_pool_root": str(
                VGGT_ROOT
                / (
                    "outputs/tum_gt_pool_universal_all_firsthalf"
                    if split_name == "firsthalf"
                    else "outputs/tum_gt_pool_per_scene_even"
                )
            ),
            "hashes": {
                "train_pools": digest(pool_path),
                "train_preview_manifest": digest(preview_path),
                "test_eval_plan": digest(plan_path),
            },
        }

    existing_firsthalf = VGGT_ROOT / "data/tum_universal_all/universal_train_firsthalf_pools.json"
    if json.loads(existing_firsthalf.read_text(encoding="utf-8")) != pools["firsthalf"]:
        raise AssertionError("New first-half pools disagree with the existing frozen protocol")

    protocol = {
        "name": "tum8_per_scene_split_comparison",
        "source_protocol": str(SOURCE_PROTOCOL),
        "source_protocol_sha256": digest(SOURCE_PROTOCOL),
        "training_unit": "one independently trained patch per scene and split",
        "shared_settings": {
            "model_input_frames_per_update": 10,
            "updates": 1000,
            "seed": 0,
            "loss": "mirage_projected_split_pose (all five active terms)",
            "texture": "64x64 RGB, warning-image cold start",
            "carrier": "the same frozen monitor world quad",
            "eot": False,
        },
        "splits": {
            "firsthalf": {
                "train": "chronological first half",
                "test": "chronological second half only",
                "claim": "future-view extrapolation within the same TUM sequence",
            },
            "evenodd": {
                "train": "all even synchronized frame indices",
                "test": "all odd synchronized frame indices only",
                "claim": "dense within-trajectory interpolation; not future-view generalization",
            },
        },
        "evaluation": {
            "window": WINDOW,
            "stride": STRIDE,
            "pose": "stitch held-out windows, then one final Sim(3) per complete held-out trajectory",
            "other_tasks": "same held-out windows; no training frames enter reported unseen metrics",
        },
        "scenes": scenes,
        "counts": summary,
        "files": paths,
    }
    protocol_path = OUT_ROOT / "protocol.json"
    write_checked(protocol_path, protocol)
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
