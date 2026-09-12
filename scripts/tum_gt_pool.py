"""Disk-backed TUM four-task GT pools used by dynamic-frame training.

Large arrays are separate ``.npy`` files so NumPy can genuinely memory-map
them. Metadata is written last by the builder; its presence therefore marks a
complete pool rather than a partially written directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


POOL_META_NAME = "pool_meta.json"
BASE_ARRAY_FILES = {
    "gt_c2w": "gt_c2w.npy",
    "depth": "depth.npy",
    "point_map": "point_map.npy",
    "point_valid": "point_valid.npy",
    "query_points": "query_points.npy",
    "query_valid": "query_valid.npy",
}
TRACK_ARRAY_FILES = {
    "tracks": "tracks.npy",
    "track_known": "track_known.npy",
    "track_visible": "track_visible.npy",
}
POOL_ARRAY_FILES = {**BASE_ARRAY_FILES, **TRACK_ARRAY_FILES}


def pool_scene_dir(root: str | Path, scene: str) -> Path:
    return Path(root) / scene


def pool_exists(root: str | Path, scene: str) -> bool:
    return (pool_scene_dir(root, scene) / POOL_META_NAME).is_file()


def open_pool(root: str | Path, scene: str) -> dict[str, Any]:
    """Open one complete scene pool without paging all arrays into RAM."""
    scene_dir = pool_scene_dir(root, scene)
    meta_path = scene_dir / POOL_META_NAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"TUM GT pool metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("scene") != scene:
        raise ValueError(f"GT pool scene {meta.get('scene')!r} != requested {scene!r}")

    arrays: dict[str, np.ndarray] = {}
    track_storage = str(meta.get("track_storage", "dense"))
    if track_storage not in ("dense", "lazy_reprojection"):
        raise ValueError(f"Unsupported GT pool track_storage={track_storage!r}")
    files = dict(BASE_ARRAY_FILES)
    if track_storage == "dense":
        files.update(TRACK_ARRAY_FILES)
    files.update(meta.get("array_files", {}))
    for name, filename in files.items():
        path = scene_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"GT pool array missing: {path}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)

    frame_indices = [int(value) for value in meta["frame_indices"]]
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError("GT pool frame_indices contain duplicates")
    n = len(frame_indices)
    expected_leading = {
        "gt_c2w": (n,),
        "depth": (n,),
        "point_map": (n,),
        "point_valid": (n,),
        "query_points": (n,),
        "query_valid": (n,),
    }
    if track_storage == "dense":
        expected_leading.update({
            "tracks": (n, n),
            "track_known": (n, n),
            "track_visible": (n, n),
        })
    for name, leading in expected_leading.items():
        if tuple(arrays[name].shape[:len(leading)]) != leading:
            raise ValueError(
                f"GT pool {name} shape {arrays[name].shape} does not start with {leading}"
            )
    return {
        "root": str(Path(root)),
        "scene_dir": scene_dir,
        "meta": meta,
        "arrays": arrays,
        "track_storage": track_storage,
        "frame_to_pos": {frame: pos for pos, frame in enumerate(frame_indices)},
    }


def _sample_depth(depth: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour depth lookup, matching the independent GT builder."""
    h, w = depth.shape
    xi = np.rint(xy[:, 0]).astype(np.int64)
    yi = np.rint(xy[:, 1]).astype(np.int64)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    values = np.zeros(len(xy), dtype=np.float64)
    values[inside] = depth[yi[inside], xi[inside]]
    return values, inside


def _project_world(
    world: np.ndarray,
    c2w: np.ndarray,
    projection: dict[str, float],
    intrinsics: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(c2w)
    camera = world @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = float(intrinsics["fx"]) * camera[:, 0] / safe + float(intrinsics["cx"])
    v = float(intrinsics["fy"]) * camera[:, 1] / safe + float(intrinsics["cy"])
    x = u * float(projection["scale_x"])
    y = v * float(projection["scale_y"]) - float(projection["crop_y"])
    return np.stack([x, y], axis=-1), z


def _gather_lazy_tracks(
    pool: dict[str, Any], positions: list[int], q_pos: int, keep: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate only the Q=first-frame to current-set tracks needed this step."""
    arrays = pool["arrays"]
    meta = pool["meta"]
    projection = {key: float(value) for key, value in meta["projection"].items()}
    intrinsics = meta.get(
        "camera_intrinsics",
        {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6},
    )
    query = np.asarray(arrays["query_points"][q_pos, keep], dtype=np.float64)
    query_depth, query_inside = _sample_depth(
        np.asarray(arrays["depth"][q_pos]), query
    )
    if not bool(np.all(query_inside & np.isfinite(query_depth) & (query_depth > 1e-6))):
        raise ValueError("Lazy-track query_valid disagrees with the stored query depth")

    u = query[:, 0] / projection["scale_x"]
    v = (query[:, 1] + projection["crop_y"]) / projection["scale_y"]
    camera = np.stack(
        [
            (u - float(intrinsics["cx"])) / float(intrinsics["fx"]) * query_depth,
            (v - float(intrinsics["cy"])) / float(intrinsics["fy"]) * query_depth,
            query_depth,
        ],
        axis=-1,
    )
    q_c2w = np.asarray(arrays["gt_c2w"][q_pos], dtype=np.float64)
    world = camera @ q_c2w[:3, :3].T + q_c2w[:3, 3]

    p = int(query.shape[0])
    tracks = np.zeros((len(positions), p, 2), dtype=np.float32)
    known = np.zeros((len(positions), p), dtype=bool)
    visible = np.zeros((len(positions), p), dtype=bool)
    occl_tol = float(meta.get("occl_tol", 0.05))
    for out_pos, target_pos in enumerate(positions):
        xy, z = _project_world(
            world,
            np.asarray(arrays["gt_c2w"][target_pos], dtype=np.float64),
            projection,
            intrinsics,
        )
        scene_depth, inside = _sample_depth(
            np.asarray(arrays["depth"][target_pos]), xy
        )
        has_depth = inside & np.isfinite(scene_depth) & (scene_depth > 1e-6)
        in_front = np.isfinite(z) & (z > 1e-6)
        tracks[out_pos] = xy.astype(np.float32, copy=False)
        known[out_pos] = has_depth & in_front
        visible[out_pos] = has_depth & in_front & (z <= scene_depth + occl_tol)
    return tracks, known, visible


def query_count(pool: dict[str, Any], frame_index: int) -> int:
    try:
        pos = pool["frame_to_pos"][int(frame_index)]
    except KeyError as exc:
        raise KeyError(f"Frame {frame_index} is absent from the GT pool") from exc
    return int(np.count_nonzero(pool["arrays"]["query_valid"][pos]))


def gather_pool(
    pool: dict[str, Any],
    frame_indices: list[int],
    tensor_hw: tuple[int, int],
    *,
    image_names: list[str] | None = None,
    gt_c2w: np.ndarray | None = None,
    min_queries: int = 1,
) -> dict[str, np.ndarray]:
    """Gather one ordered frame set and tracks queried from its first frame."""
    frames = [int(value) for value in frame_indices]
    if not frames:
        raise ValueError("Cannot gather an empty frame set")
    try:
        positions = [pool["frame_to_pos"][frame] for frame in frames]
    except KeyError as exc:
        raise KeyError(
            f"Run frame {int(exc.args[0])} is absent from {pool['scene_dir']}"
        ) from exc

    saved_hw = tuple(int(value) for value in pool["meta"]["tensor_hw"])
    if saved_hw != tuple(int(value) for value in tensor_hw):
        raise ValueError(f"GT pool grid {saved_hw} != run grid {tuple(tensor_hw)}")

    if image_names is not None:
        if len(image_names) != len(frames):
            raise ValueError("image_names and frame_indices have different lengths")
        saved_names = pool["meta"].get("image_names")
        if saved_names is None:
            raise ValueError("GT pool metadata has no image_names for identity validation")
        expected = [str(saved_names[pos]) for pos in positions]
        actual = [Path(name).name for name in image_names]
        if actual != expected:
            raise ValueError(f"GT pool image identities {expected} != run images {actual}")

    arrays = pool["arrays"]
    if gt_c2w is not None:
        saved_pose = np.asarray(arrays["gt_c2w"][positions], dtype=np.float64)
        current_pose = np.asarray(gt_c2w, dtype=np.float64)
        if saved_pose.shape != current_pose.shape or not np.allclose(
            saved_pose, current_pose, rtol=1e-7, atol=1e-8
        ):
            raise ValueError("GT pool poses do not match the run frame identities")

    q_pos = positions[0]
    keep = np.asarray(arrays["query_valid"][q_pos], dtype=bool)
    count = int(np.count_nonzero(keep))
    if count < int(min_queries):
        raise ValueError(
            f"Query frame {frames[0]} has only {count} valid GT points; "
            f"requires {int(min_queries)}"
        )

    # Fancy indexing creates compact arrays containing only this step's frames.
    # Explicit copies make ownership clear before torch.from_numpy is called.
    depth = np.array(arrays["depth"][positions], dtype=np.float32, copy=True)
    point_map = np.array(arrays["point_map"][positions], dtype=np.float32, copy=True)
    point_valid = np.array(arrays["point_valid"][positions], dtype=bool, copy=True)
    query_points = np.array(arrays["query_points"][q_pos, keep], dtype=np.float32, copy=True)
    if pool["track_storage"] == "lazy_reprojection":
        tracks, known, visible = _gather_lazy_tracks(pool, positions, q_pos, keep)
    else:
        tracks = np.array(
            arrays["tracks"][q_pos, positions][:, keep], dtype=np.float32, copy=True
        )
        known = np.array(
            arrays["track_known"][q_pos, positions][:, keep], dtype=bool, copy=True
        )
        visible = np.array(
            arrays["track_visible"][q_pos, positions][:, keep], dtype=bool, copy=True
        )

    return {
        "depth": depth,
        "point_map": point_map,
        "point_valid": point_valid,
        "query_points": query_points,
        "tracks": tracks,
        "track_known": known,
        "track_visible": visible,
        "query_frame": np.asarray(frames[0], dtype=np.int64),
        "query_count": np.asarray(count, dtype=np.int64),
    }
