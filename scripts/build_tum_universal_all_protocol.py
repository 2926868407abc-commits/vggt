#!/usr/bin/env python3
"""Build the all-frame TUM universal protocol and synchronized scene views."""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path


SOURCE_ROOT = Path("/mnt/data/wangqq/recons_eval/data/tum")
SYNC_ROOT = Path("/mnt/data/wangqq/recons_eval/data/tum_all_sync")
VGGT_ROOT = Path("/mnt/data/wangqq/vggt")
SPLIT_ROOT = VGGT_ROOT / "data/tum_universal_all"
MAX_DT = 0.02
WINDOW = 10
SCENES = [
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_sitting_rpy",
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
    "rgbd_dataset_freiburg3_walking_rpy",
    "rgbd_dataset_freiburg3_walking_halfsphere",
]


def read_rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(line.split())
    return rows


def nearest_index(sorted_times: list[float], value: float) -> tuple[int, float]:
    pos = bisect.bisect_left(sorted_times, value)
    candidates = [index for index in (pos - 1, pos) if 0 <= index < len(sorted_times)]
    index = min(candidates, key=lambda i: abs(sorted_times[i] - value))
    return index, abs(sorted_times[index] - value)


def associate_unique(
    first_times: list[float], second_times: list[float]
) -> dict[int, tuple[int, float]]:
    """TUM-style greedy one-to-one timestamp association."""
    candidates = []
    for first_index, timestamp in enumerate(first_times):
        pos = bisect.bisect_left(second_times, timestamp)
        for second_index in (pos - 1, pos):
            if 0 <= second_index < len(second_times):
                delta = abs(second_times[second_index] - timestamp)
                if delta <= MAX_DT:
                    candidates.append((delta, first_index, second_index))
    matches = {}
    used_second = set()
    for delta, first_index, second_index in sorted(candidates):
        if first_index in matches or second_index in used_second:
            continue
        matches[first_index] = (second_index, delta)
        used_second.add(second_index)
    return matches


def write_text_checked(path: Path, text: str) -> None:
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current != text:
            raise FileExistsError(f"Refusing to overwrite different protocol file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json_checked(path: Path, payload) -> None:
    write_text_checked(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def symlink_checked(link: Path, target: Path, *, directory: bool = False) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise FileExistsError(f"Wrong symlink target: {link} -> {link.resolve()}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    link.symlink_to(target, target_is_directory=directory)


def associate_scene(scene: str) -> tuple[list[dict], list[list[str]]]:
    source = SOURCE_ROOT / scene
    rgb = read_rows(source / "rgb.txt")
    depth = read_rows(source / "depth.txt")
    poses = read_rows(source / "groundtruth.txt")
    rgb_times = [float(row[0]) for row in rgb]
    depth_times = [float(row[0]) for row in depth]
    pose_times = [float(row[0]) for row in poses]
    rgb_depth = associate_unique(rgb_times, depth_times)
    matches = []
    synced_pose_rows = []
    for rgb_index, rgb_row in enumerate(rgb):
        if rgb_index not in rgb_depth:
            continue
        timestamp = float(rgb_row[0])
        depth_index, depth_dt = rgb_depth[rgb_index]
        pose_index, pose_dt = nearest_index(pose_times, timestamp)
        if pose_dt > MAX_DT:
            continue
        pose = poses[pose_index]
        matches.append({
            "frame_index": len(matches),
            "rgb_index": rgb_index,
            "rgb_timestamp": timestamp,
            "rgb_path": rgb_row[1],
            "depth_index": depth_index,
            "depth_timestamp": depth_times[depth_index],
            "depth_path": depth[depth_index][1],
            "depth_dt": depth_dt,
            "pose_index": pose_index,
            "pose_timestamp": pose_times[pose_index],
            "pose_dt": pose_dt,
        })
        # Pose rows are indexed one-to-one with the sorted synchronized image view.
        synced_pose_rows.append([rgb_row[0], *pose[1:8]])
    if len({item["depth_index"] for item in matches}) != len(matches):
        raise RuntimeError(f"{scene}: duplicate RGB-to-depth associations")
    if len({item["pose_index"] for item in matches}) != len(matches):
        raise RuntimeError(f"{scene}: duplicate RGB-to-pose associations")
    if len(matches) < 2 * WINDOW:
        raise RuntimeError(f"{scene}: only {len(matches)} synchronized frames")
    return matches, synced_pose_rows


def fixed_windows(start: int, stop: int) -> list[list[int]]:
    """Cover every frame with ten-frame inputs; only the final window may overlap."""
    if stop - start < WINDOW:
        raise ValueError(f"Range [{start}, {stop}) is shorter than {WINDOW}")
    windows = []
    for first in range(start, stop, WINDOW):
        if first + WINDOW <= stop:
            window = list(range(first, first + WINDOW))
        else:
            window = list(range(stop - WINDOW, stop))
        if not windows or window != windows[-1]:
            windows.append(window)
    covered = set().union(*(set(window) for window in windows))
    if covered != set(range(start, stop)):
        raise AssertionError(f"Window coverage mismatch for [{start}, {stop})")
    return windows


def spaced_preview(stop: int) -> list[int]:
    return sorted({round(i * (stop - 1) / (WINDOW - 1)) for i in range(WINDOW)})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    SYNC_ROOT.mkdir(parents=True, exist_ok=True)
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    train_manifest = {}
    train_pools = {}
    full_plan = []
    unseen_plan = []
    counts = {}

    for scene in SCENES:
        matches, pose_rows = associate_scene(scene)
        source = SOURCE_ROOT / scene
        synced = SYNC_ROOT / scene
        image_dir = synced / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for item in matches:
            source_image = source / item["rgb_path"]
            symlink_checked(image_dir / source_image.name, source_image)
        symlink_checked(synced / "depth", source / "depth", directory=True)
        symlink_checked(synced / "depth.txt", source / "depth.txt")
        rgb_text = "# timestamp path\n" + "\n".join(
            f"{item['rgb_timestamp']:.6f} images/{Path(item['rgb_path']).name}"
            for item in matches
        ) + "\n"
        gt_text = "# timestamp tx ty tz qx qy qz qw\n" + "\n".join(
            " ".join(row) for row in pose_rows
        ) + "\n"
        write_text_checked(synced / "rgb.txt", rgb_text)
        write_text_checked(synced / "groundtruth.txt", gt_text)
        write_json_checked(synced / "association.json", {
            "source_scene": str(source),
            "max_dt_seconds": MAX_DT,
            "matches": matches,
        })

        n = len(matches)
        split = n // 2
        train_indices = list(range(split))
        preview = spaced_preview(split)
        if len(preview) != WINDOW:
            raise AssertionError(f"{scene}: preview did not contain {WINDOW} frames")
        train_pools[scene] = train_indices
        train_manifest[scene] = {
            "sampling": "universal_all_frames_first_half",
            "frame_count_requested": WINDOW,
            "frame_indices": preview,
            "split": f"first_half_[0,{split})",
            "note": "Preview only; training uses the per-scene dynamic pool JSON.",
        }

        full_windows = fixed_windows(0, n)
        unseen_windows = fixed_windows(split, n)
        for index, frames in enumerate(full_windows):
            manifest = {scene: {
                "sampling": "contiguous_window_with_final_overlap",
                "frame_count_requested": WINDOW,
                "frame_indices": frames,
                "split": "all_synchronized_frames",
            }}
            path = SPLIT_ROOT / f"{scene}__full_w{index:03d}_manifest.json"
            write_json_checked(path, manifest)
            full_plan.append({
                "name": f"{scene}__full_w{index:03d}",
                "scene": scene,
                "manifest": str(path),
                "frames": frames,
                "contains_training_frames": any(frame < split for frame in frames),
            })
        for index, frames in enumerate(unseen_windows):
            manifest = {scene: {
                "sampling": "contiguous_window_with_final_overlap",
                "frame_count_requested": WINDOW,
                "frame_indices": frames,
                "split": "strict_unseen_second_half",
            }}
            path = SPLIT_ROOT / f"{scene}__unseen_w{index:03d}_manifest.json"
            write_json_checked(path, manifest)
            unseen_plan.append({
                "name": f"{scene}__unseen_w{index:03d}",
                "scene": scene,
                "manifest": str(path),
                "frames": frames,
                "contains_training_frames": False,
            })
        counts[scene] = {
            "all_synchronized": n,
            "train_first_half": split,
            "strict_unseen_second_half": n - split,
            "full_windows": len(full_windows),
            "unseen_windows": len(unseen_windows),
            "full_unique_frames_covered": len(set().union(*map(set, full_windows))),
            "unseen_unique_frames_covered": len(set().union(*map(set, unseen_windows))),
        }

    train_manifest_path = SPLIT_ROOT / "universal_train_preview_manifest.json"
    train_pools_path = SPLIT_ROOT / "universal_train_firsthalf_pools.json"
    full_plan_path = SPLIT_ROOT / "universal_full_eval_plan.json"
    unseen_plan_path = SPLIT_ROOT / "universal_unseen_eval_plan.json"
    write_json_checked(train_manifest_path, train_manifest)
    write_json_checked(train_pools_path, train_pools)
    write_json_checked(full_plan_path, full_plan)
    write_json_checked(unseen_plan_path, unseen_plan)
    protocol = {
        "name": "tum8_universal_all_synchronized_firsthalf_to_full",
        "source": "all RGB frames with depth and pose associations within 0.02 s",
        "source_root": str(SOURCE_ROOT),
        "synchronized_root": str(SYNC_ROOT),
        "scenes": SCENES,
        "training": {
            "split": "chronological first half of every synchronized sequence",
            "frames_per_model_input": WINDOW,
            "sequence_sampling": "uniform over eight scenes in expectation",
            "frame_sampling": "ten without replacement from the selected scene first half",
            "preview_manifest": str(train_manifest_path),
            "dynamic_pool_json": str(train_pools_path),
            "gt_pool_root": str(VGGT_ROOT / "outputs/tum_gt_pool_universal_all_firsthalf"),
        },
        "evaluation": {
            "full": "all synchronized frames; ten-frame windows; final window overlaps only when needed",
            "strict_unseen": "chronological second half only; same window rule",
            "full_plan": str(full_plan_path),
            "unseen_plan": str(unseen_plan_path),
            "aggregation": "window metrics -> per-sequence weighted mean -> eight-sequence macro mean",
        },
        "counts": counts,
        "totals": {
            "all_synchronized": sum(row["all_synchronized"] for row in counts.values()),
            "train_first_half": sum(row["train_first_half"] for row in counts.values()),
            "strict_unseen_second_half": sum(
                row["strict_unseen_second_half"] for row in counts.values()
            ),
            "full_windows": len(full_plan),
            "unseen_windows": len(unseen_plan),
        },
    }
    protocol["hashes"] = {
        "train_manifest": sha256(train_manifest_path),
        "train_pools": sha256(train_pools_path),
        "full_plan": sha256(full_plan_path),
        "unseen_plan": sha256(unseen_plan_path),
    }
    protocol_path = SPLIT_ROOT / "protocol.json"
    write_json_checked(protocol_path, protocol)
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
