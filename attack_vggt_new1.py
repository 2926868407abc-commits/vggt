"""
PGD/patch attack baseline for VGGT with paper-aligned evaluation metrics.

The attack is optimized against CO3D ground truth when --gt_root is supplied.
Evaluation uses the metric definitions used in the VGGT paper:
camera pose AUC over pairwise relative pose errors, no-scale-alignment and
Sim(3)-aligned Chamfer Accuracy/Completeness/Overall for depth-derived clouds,
Sim(3)-aligned point-map Chamfer, and an optional ALIKED-query two-view
matching protocol for tracking.

Example:
    python attack_vggt_new1.py ^
        --scene_dir examples/kitchen ^
        --clean_npz clean_outputs/kitchen/vggt_outputs.npz ^
        --output_dir outputs_pgd/kitchen ^
        --gt_root /path/to/co3d ^
        --max_frames 10 ^
        --steps 10 ^
        --eps 0.03137255 ^
        --alpha 0.00392157
"""

import argparse
import csv
import glob
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri


CO3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
EVALUATION_PROTOCOL = {
    "reference": "paper_original_dataset_gt",
    "attack_loss": "maximize prediction error against the available task reference; falls back to clean output if a task has no dense GT tensor.",
    "camera": (
        "Paper Table 1 (CO3D/RE10K) protocol: AUC over pairwise relative pose errors, "
        "implemented as AUC over max(rotation_error, translation_error). Dataset-level "
        "CO3D aggregation follows the official protocol: compute metrics per category, "
        "then average categories."
    ),
    "depth": "Paper Table 2: DTU dense MVS benchmark with GT scan point clouds.",
    "depth_align_none": "DTU-style Chamfer between prediction and DTU GT scan point cloud.",
    "depth_align_sim3": (
        "Paper Table 3 (ETH3D), Ours(Depth+Cam): depth+camera unproject + Umeyama Sim(3) "
        "alignment + Chamfer."
    ),
    "point": (
        "Paper Table 3 (ETH3D), Ours(Point): point-head output + Umeyama Sim(3) alignment + Chamfer."
    ),
    "tracking_image_matching": (
        "Paper Table 4 (ScanNet-1500) protocol: ALIKED query points -> VGGT tracks -> "
        "Essential matrix -> pose AUC@5/10/20."
    ),
    "dataset_note": (
        "Paper protocol uses different datasets per task: camera=CO3Dv2/Re10K, "
        "depth=DTU, point=ETH3D, tracking=ScanNet-1500."
    ),
}
EVAL_TASKS = ("camera", "depth", "point", "tracking")
PAPER_TASK_DATASETS = {
    "camera": {"co3d", "re10k"},
    "depth": {"dtu"},
    "point": {"eth3d"},
    "tracking": {"scannet1500"},
}

try:
    import cv2
except ImportError as exc:
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None


def require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for CO3D depth/mask loading and Essential matrix matching. "
            "Install opencv-python or run in an environment that provides cv2."
        ) from CV2_IMPORT_ERROR
    return cv2


def unproject_depth_map_to_point_map(*args, **kwargs):
    from vggt.utils.geometry import unproject_depth_map_to_point_map as unproject

    return unproject(*args, **kwargs)


def parse_eval_tasks(value: str | list[str] | tuple[str, ...] | None) -> set[str]:
    if value is None:
        return set(EVAL_TASKS)
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip()]
    else:
        parts = [str(part).strip().lower() for part in value if str(part).strip()]
    if not parts or "all" in parts:
        return set(EVAL_TASKS)

    tasks = set(parts)
    unknown = sorted(tasks.difference(EVAL_TASKS))
    if unknown:
        raise ValueError(f"Unknown eval task(s): {unknown}. Choose from: {', '.join(EVAL_TASKS)}")
    return tasks


def infer_paper_eval_dataset(eval_tasks: set[str], requested: str) -> str:
    if requested != "auto":
        return requested
    if len(eval_tasks) != 1:
        raise ValueError(
            "--eval_dataset auto can only infer a dataset for a single --eval_tasks value. "
            "Run paper tasks separately, e.g. --eval_tasks depth --eval_dataset dtu."
        )
    task = next(iter(eval_tasks))
    if task == "camera":
        return "co3d"
    if task == "depth":
        return "dtu"
    if task == "point":
        return "eth3d"
    if task == "tracking":
        return "scannet1500"
    raise ValueError(f"Unsupported eval task: {task}")


def validate_paper_eval_dataset(eval_tasks: set[str], eval_dataset: str) -> None:
    if len(eval_tasks) != 1:
        raise ValueError(
            "Paper-aligned evaluation must be run one task at a time because VGGT reports "
            "camera/depth/point/tracking on different datasets. Use --eval_tasks camera, "
            "then --eval_tasks depth, etc."
        )
    task = next(iter(eval_tasks))
    allowed = PAPER_TASK_DATASETS[task]
    if eval_dataset not in allowed:
        raise ValueError(
            f"Paper task '{task}' must use one of {sorted(allowed)}, got --eval_dataset {eval_dataset!r}. "
            "This prevents CO3D proxy metrics from being reported as paper metrics."
        )


def find_images(scene_dir: Path) -> list[str]:
    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        image_dir = scene_dir

    paths: list[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(glob.glob(str(image_dir / ext)))
    return sorted(paths)


def derive_scene_seed(seed: int, scene_name: str) -> int:
    scene_hash = hashlib.sha256(scene_name.encode("utf-8")).digest()
    scene_offset = int.from_bytes(scene_hash[:4], byteorder="little", signed=False)
    return (int(seed) + scene_offset) % (2**32)


def subsample(paths: list[str], max_frames: int, seed: int) -> tuple[list[str], np.ndarray]:
    indices = subsample_indices(len(paths), max_frames, seed)
    return [paths[int(i)] for i in indices], indices


def subsample_indices(length: int, max_frames: int, seed: int) -> np.ndarray:
    if max_frames <= 0 or length <= max_frames:
        return np.arange(length)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=max_frames, replace=False))


def autocast_context(device: torch.device, dtype: torch.dtype):
    enabled = device.type == "cuda"
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def forward_vggt(
    model: torch.nn.Module,
    images: torch.Tensor,
    dtype: torch.dtype,
    query_points: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    with autocast_context(images.device, dtype):
        preds = model(images, query_points=query_points)
    return {k: v for k, v in preds.items() if torch.is_tensor(v)}


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    from vggt.models.vggt import VGGT

    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_file():
        model = VGGT()
        if ckpt_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(ckpt_path), device="cpu")
        else:
            state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        return model.to(device).eval()

    kwargs = {"local_files_only": True} if args.local_files_only else {}
    return VGGT.from_pretrained(args.ckpt, **kwargs).to(device).eval()


def detach_predictions(preds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = ("pose_enc", "depth", "depth_conf", "world_points", "world_points_conf", "track", "vis", "conf")
    return {k: preds[k].detach() for k in keys if k in preds}


def load_clean_reference(
    npz_path: Path,
    device: torch.device,
    image_size_hw: tuple[int, int],
    frame_indices: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    data = np.load(npz_path)
    refs: dict[str, torch.Tensor] = {}

    def select_frames(array: np.ndarray) -> np.ndarray:
        if frame_indices is None:
            return array
        return array[frame_indices]

    def select_sequence_frames(array: np.ndarray) -> np.ndarray:
        if frame_indices is None:
            return array
        if array.ndim >= 2 and array.shape[0] == 1:
            return array[:, frame_indices]
        return array[frame_indices]

    if "depth" in data:
        refs["depth"] = torch.from_numpy(select_frames(data["depth"]).astype(np.float32)).to(device).unsqueeze(0)
    if "point_map" in data:
        refs["world_points"] = torch.from_numpy(select_frames(data["point_map"]).astype(np.float32)).to(device).unsqueeze(0)
    if "extrinsic" in data and "intrinsic" in data:
        extrinsic = torch.from_numpy(select_frames(data["extrinsic"]).astype(np.float32)).to(device).unsqueeze(0)
        intrinsic = torch.from_numpy(select_frames(data["intrinsic"]).astype(np.float32)).to(device).unsqueeze(0)
        refs["pose_enc"] = extri_intri_to_pose_encoding(extrinsic, intrinsic, image_size_hw)
    if "tracks" in data:
        tracks = select_sequence_frames(data["tracks"]).astype(np.float32)
        if tracks.ndim == 3:
            tracks = tracks[None]
        refs["track"] = torch.from_numpy(tracks).to(device)
    if "track_visibility" in data:
        vis = select_sequence_frames(data["track_visibility"]).astype(np.float32)
        if vis.ndim == 2:
            vis = vis[None]
        refs["vis"] = torch.from_numpy(vis).to(device)

    if not refs:
        raise ValueError(f"No usable clean VGGT outputs found in {npz_path}")
    return refs


def ensure_prediction_batch(array: np.ndarray, unbatched_ndim: int) -> np.ndarray:
    if array.ndim == unbatched_ndim:
        return array[None]
    return array


def load_prefixed_predictions_from_npz(
    npz_path: Path,
    prefix: str,
    device: torch.device,
    image_size_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    data = np.load(npz_path)
    refs: dict[str, torch.Tensor] = {}

    def get(name: str) -> np.ndarray | None:
        key = f"{prefix}{name}"
        return data[key] if key in data else None

    pose_enc = get("pose_enc")
    extrinsic = get("extrinsic")
    intrinsic = get("intrinsic")
    if pose_enc is not None:
        refs["pose_enc"] = torch.from_numpy(ensure_prediction_batch(pose_enc, 2).astype(np.float32)).to(device)
    elif extrinsic is not None and intrinsic is not None:
        extrinsic_t = torch.from_numpy(ensure_prediction_batch(extrinsic, 3).astype(np.float32)).to(device)
        intrinsic_t = torch.from_numpy(ensure_prediction_batch(intrinsic, 3).astype(np.float32)).to(device)
        refs["extrinsic"] = extrinsic_t
        refs["intrinsic"] = intrinsic_t
        refs["pose_enc"] = extri_intri_to_pose_encoding(extrinsic_t, intrinsic_t, image_size_hw)

    depth = get("depth")
    if depth is not None:
        refs["depth"] = torch.from_numpy(ensure_prediction_batch(depth, 4).astype(np.float32)).to(device)
    depth_conf = get("depth_conf")
    if depth_conf is not None:
        refs["depth_conf"] = torch.from_numpy(ensure_prediction_batch(depth_conf, 3).astype(np.float32)).to(device)

    world_points = get("world_points")
    if world_points is not None:
        refs["world_points"] = torch.from_numpy(ensure_prediction_batch(world_points, 4).astype(np.float32)).to(device)
    world_points_conf = get("world_points_conf")
    if world_points_conf is not None:
        refs["world_points_conf"] = torch.from_numpy(ensure_prediction_batch(world_points_conf, 3).astype(np.float32)).to(device)

    track = get("track")
    if track is not None:
        refs["track"] = torch.from_numpy(ensure_prediction_batch(track, 3).astype(np.float32)).to(device)
    vis = get("vis")
    if vis is not None:
        refs["vis"] = torch.from_numpy(ensure_prediction_batch(vis, 2).astype(np.float32)).to(device)
    conf = get("conf")
    if conf is not None:
        refs["conf"] = torch.from_numpy(ensure_prediction_batch(conf, 2).astype(np.float32)).to(device)

    if not refs:
        raise ValueError(f"No {prefix} prediction tensors found in {npz_path}")
    return refs


def load_saved_attack_images(npz_path: Path, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    data = np.load(npz_path)
    clean_images = None
    adv_images = None
    if "clean_images" in data:
        clean_images = torch.from_numpy(data["clean_images"].astype(np.float32)).to(device)
    if "adv_images" in data:
        adv_images = torch.from_numpy(data["adv_images"].astype(np.float32)).to(device)
    return clean_images, adv_images


def read_ply_xyz(ply_path: Path) -> np.ndarray:
    with open(ply_path, "rb") as f:
        header_lines: list[bytes] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header is missing end_header: {ply_path}")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = [line.decode("ascii", errors="replace").strip() for line in header_lines]
        vertex_count = 0
        fmt = None
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header:
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and line.startswith("property "):
                _, dtype_name, name = line.split()[:3]
                properties.append((dtype_name, name))

        if vertex_count <= 0:
            raise ValueError(f"PLY file has no vertices: {ply_path}")
        if not {"x", "y", "z"}.issubset({name for _, name in properties}):
            raise ValueError(f"PLY vertices must contain x/y/z properties: {ply_path}")

        if fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                values = f.readline().decode("ascii", errors="replace").split()
                rows.append([float(values[[name for _, name in properties].index(axis)]) for axis in ("x", "y", "z")])
            return np.asarray(rows, dtype=np.float32)

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format {fmt!r}; use ascii or binary_little_endian: {ply_path}")

        dtype_map = {
            "char": "i1",
            "uchar": "u1",
            "int8": "i1",
            "uint8": "u1",
            "short": "<i2",
            "ushort": "<u2",
            "int16": "<i2",
            "uint16": "<u2",
            "int": "<i4",
            "uint": "<u4",
            "int32": "<i4",
            "uint32": "<u4",
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "float64": "<f8",
        }
        dtype = np.dtype([(name, dtype_map[dtype_name]) for dtype_name, name in properties])
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
        return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)


def resolve_dtu_gt_ply(args: argparse.Namespace, scene_dir: Path) -> Path:
    if args.gt_pointcloud:
        return Path(args.gt_pointcloud)
    if not args.gt_root:
        raise ValueError("--eval_dataset dtu needs --gt_pointcloud or --gt_root")
    scan_id = args.scan_id
    if scan_id is None:
        digits = "".join(ch for ch in scene_dir.name if ch.isdigit())
        if digits:
            scan_id = int(digits)
    candidates = []
    gt_root = Path(args.gt_root)
    if scan_id is not None:
        candidates.extend(
            [
                gt_root / "Points" / "stl" / f"stl{int(scan_id):03d}_total.ply",
                gt_root / "Points" / "stl" / f"stl{int(scan_id)}_total.ply",
                gt_root / f"stl{int(scan_id):03d}_total.ply",
                gt_root / f"scan{int(scan_id)}" / "gt.ply",
            ]
        )
    candidates.extend([gt_root / scene_dir.name / "gt.ply", gt_root / f"{scene_dir.name}.ply"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate DTU GT point cloud. Tried: {[str(p) for p in candidates]}")


def resolve_dtu_scan_id(args: argparse.Namespace, scene_dir: Path) -> int:
    if args.scan_id is not None:
        return int(args.scan_id)
    digits = "".join(ch for ch in scene_dir.name if ch.isdigit())
    if digits:
        return int(digits)
    raise ValueError("--eval_dataset dtu needs --scan_id unless the scene directory name contains the scan id")


def resolve_eth3d_gt_ply(args: argparse.Namespace, scene_dir: Path) -> Path:
    if args.gt_pointcloud:
        return Path(args.gt_pointcloud)
    if not args.gt_root:
        raise ValueError("--eval_dataset eth3d needs --gt_pointcloud or --gt_root")
    gt_root = Path(args.gt_root)
    candidates = [
        gt_root / scene_dir.name / "dslr_scan_eval" / "scan_alignment.mlp.ply",
        gt_root / scene_dir.name / "dslr_scan_eval" / "gt.ply",
        gt_root / scene_dir.name / "gt.ply",
        gt_root / f"{scene_dir.name}.ply",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate ETH3D GT point cloud. Tried: {[str(p) for p in candidates]}")


def load_gt_pointcloud_reference(
    ply_path: Path,
    device: torch.device,
    alignment: str,
) -> dict[str, torch.Tensor | str]:
    points = read_ply_xyz(ply_path)
    return {
        "gt_point_cloud": torch.from_numpy(points.astype(np.float32)),
        "pointcloud_alignment": alignment,
        "gt_point_cloud_path": str(ply_path),
    }


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or len(points) == 0:
        return points
    voxels = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(voxels, axis=0, return_index=True)
    return points[np.sort(indices)]


def dtu_official_chamfer_metrics(
    pred_points: np.ndarray,
    reference: dict,
    prefix: str,
    device: torch.device,
) -> dict[str, float | str]:
    try:
        from scipy.io import loadmat
        from scipy.spatial import cKDTree
    except Exception as exc:
        return {f"{prefix}_status": f"skipped_missing_scipy_for_dtu_official_eval: {exc}"}

    eval_root = Path(str(reference["dtu_eval_root"]))
    scan_id = int(reference["dtu_scan_id"])
    patch = float(reference.get("dtu_patch_size", 60.0))
    max_dist = float(reference.get("dtu_max_dist", 20.0))
    downsample = float(reference.get("dtu_downsample", 0.2))

    obs_path = eval_root / "ObsMask" / f"ObsMask{scan_id}_10.mat"
    plane_path = eval_root / "ObsMask" / f"Plane{scan_id}.mat"
    stl_path = eval_root / "Points" / "stl" / f"stl{scan_id:03d}_total.ply"
    if not obs_path.exists():
        return {f"{prefix}_status": f"missing_dtu_obsmask:{obs_path}"}
    if not stl_path.exists():
        return {f"{prefix}_status": f"missing_dtu_stl:{stl_path}"}

    pred = pred_points.reshape(-1, 3).astype(np.float32)
    pred = pred[np.isfinite(pred).all(axis=1)]
    pred = voxel_downsample(pred, downsample)

    obs = loadmat(obs_path)
    obs_mask = obs["ObsMask"].astype(bool)
    bb = obs["BB"].astype(np.float32)
    res = float(np.asarray(obs["Res"]).reshape(-1)[0])

    inbound = ((pred >= bb[:1] - patch) & (pred < bb[1:] + patch * 2)).sum(axis=-1) == 3
    pred_in = pred[inbound]
    grid = np.rint((pred_in - bb[:1]) / res).astype(np.int32)
    grid_inbound = ((grid >= 0) & (grid < np.asarray(obs_mask.shape)[None])).sum(axis=-1) == 3
    grid_valid = grid[grid_inbound]
    pred_valid = pred_in[grid_inbound]
    pred_valid = pred_valid[obs_mask[grid_valid[:, 0], grid_valid[:, 1], grid_valid[:, 2]]]
    if len(pred_valid) == 0:
        return {f"{prefix}_status": "skipped_no_pred_points_inside_dtu_obsmask"}

    stl = read_ply_xyz(stl_path).astype(np.float32)
    if plane_path.exists():
        plane = loadmat(plane_path)["P"].reshape(-1).astype(np.float32)
        stl_h = np.concatenate([stl, np.ones((len(stl), 1), dtype=np.float32)], axis=1)
        stl = stl[(stl_h @ plane) > 0]

    d2s = cKDTree(stl).query(pred_valid, k=1, workers=-1)[0]
    s2d = cKDTree(pred_valid).query(stl, k=1, workers=-1)[0]
    acc_values = d2s[d2s < max_dist]
    comp_values = s2d[s2d < max_dist]
    if len(acc_values) == 0 or len(comp_values) == 0:
        return {f"{prefix}_status": "skipped_no_dtu_distances_under_max_dist"}
    acc = float(acc_values.mean())
    comp = float(comp_values.mean())
    return {
        f"{prefix}_accuracy": acc,
        f"{prefix}_completeness": comp,
        f"{prefix}_overall": float((acc + comp) * 0.5),
        f"{prefix}_dtu_pred_points": float(len(pred_valid)),
    }


def load_gt_camera_npz_reference(
    npz_path: Path | None,
    device: torch.device,
    image_size_hw: tuple[int, int],
    image_paths: list[str],
) -> dict[str, torch.Tensor]:
    if npz_path is None:
        return {}
    data = np.load(npz_path)
    if "extrinsic" not in data or "intrinsic" not in data:
        raise ValueError(f"{npz_path} must contain extrinsic and intrinsic arrays")
    extrinsic = data["extrinsic"].astype(np.float32)
    intrinsic = data["intrinsic"].astype(np.float32)
    if "image_paths" in data:
        by_name = {Path(str(name)).name: idx for idx, name in enumerate(data["image_paths"].tolist())}
        indices = []
        for image_path in image_paths:
            name = Path(image_path).name
            if name not in by_name:
                raise ValueError(f"GT camera npz {npz_path} has no camera for image {name}")
            indices.append(by_name[name])
        extrinsic = extrinsic[indices]
        intrinsic = intrinsic[indices]

    extrinsic_t = torch.from_numpy(ensure_prediction_batch(extrinsic, 3).astype(np.float32)).to(device)
    intrinsic_t = torch.from_numpy(ensure_prediction_batch(intrinsic, 3).astype(np.float32)).to(device)
    return {
        "extrinsic": extrinsic_t,
        "intrinsic": intrinsic_t,
        "pose_enc": extri_intri_to_pose_encoding(extrinsic_t, intrinsic_t, image_size_hw),
    }


def read_clean_image_names(npz_path: Path | None) -> list[str] | None:
    if npz_path is None or not npz_path.exists():
        return None
    data = np.load(npz_path)
    if "image_paths" not in data:
        return None
    return [str(x) for x in data["image_paths"].tolist()]


def align_image_paths_to_clean(
    scene_dir: Path,
    clean_npz: Path | None,
    max_frames: int,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    clean_names = read_clean_image_names(clean_npz)
    if clean_names is None:
        return subsample(find_images(scene_dir), max_frames, seed)

    by_name = {Path(path).name: path for path in find_images(scene_dir)}
    missing = [name for name in clean_names if Path(name).name not in by_name]
    if missing:
        raise ValueError(f"{len(missing)} clean frames are missing from {scene_dir}; first missing: {missing[0]}")

    frame_indices = subsample_indices(len(clean_names), max_frames, seed)
    paths = [by_name[Path(clean_names[i]).name] for i in frame_indices]
    return paths, frame_indices


def parse_flat_scene_name(scene_dir: Path) -> tuple[str, str]:
    name = scene_dir.name
    if "__" not in name:
        raise ValueError(f"CO3D flat scene name must look like category__sequence, got: {name}")
    category, sequence_name = name.split("__", 1)
    return category, sequence_name


def load_co3d_category_annotations(gt_root: Path, category: str) -> list[dict]:
    anno_path = gt_root / category / "frame_annotations.jgz"
    if not anno_path.exists():
        raise FileNotFoundError(f"Missing CO3D frame annotations: {anno_path}")
    with gzip.open(anno_path, "rt") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list annotations in {anno_path}, got {type(data)}")
    return data


def load_co3d_official_camera_reference(
    anno_root: Path,
    scene_dir: Path,
    image_paths: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    category, sequence_name = parse_flat_scene_name(scene_dir)
    anno_path = anno_root / f"{category}_test.jgz"
    if not anno_path.exists():
        raise FileNotFoundError(f"Missing official CO3D test annotation: {anno_path}")
    with gzip.open(anno_path, "rt") as f:
        data = json.load(f)
    if sequence_name not in data:
        raise ValueError(f"Sequence {sequence_name} is not in {anno_path}")

    by_basename = {Path(item["filepath"]).name: item for item in data[sequence_name]}
    extrinsics = []
    for image_path in image_paths:
        basename = Path(image_path).name
        if basename not in by_basename:
            raise ValueError(f"No official CO3D GT camera for {basename} in {category}/{sequence_name}")
        item = by_basename[basename]
        extrinsics.append(co3d_rt_to_opencv_extrinsic(item["R"], item["T"]))
    extrinsic_t = torch.from_numpy(np.stack(extrinsics).astype(np.float32)).to(device).unsqueeze(0)
    return {"extrinsic": extrinsic_t}


def co3d_ndc_intrinsics_to_pixels(viewpoint: dict, image_size_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_size_hw
    min_size = float(min(height, width))
    focal = np.asarray(viewpoint["focal_length"], dtype=np.float32)
    principal = np.asarray(viewpoint["principal_point"], dtype=np.float32)

    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = focal[0] * min_size / 2.0
    intrinsics[1, 1] = focal[1] * min_size / 2.0
    intrinsics[0, 2] = width / 2.0 - principal[0] * min_size / 2.0
    intrinsics[1, 2] = height / 2.0 - principal[1] * min_size / 2.0
    return intrinsics


def co3d_rt_to_opencv_extrinsic(rotation: list | np.ndarray, translation: list | np.ndarray) -> np.ndarray:
    rot_pt3d = np.asarray(rotation, dtype=np.float32)
    trans_pt3d = np.asarray(translation, dtype=np.float32)
    trans_pt3d = trans_pt3d.copy()
    rot_pt3d = rot_pt3d.copy()
    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    return np.hstack((rot_pt3d, trans_pt3d[:, None])).astype(np.float32)


def co3d_viewpoint_to_opencv_extrinsic(viewpoint: dict) -> np.ndarray:
    """Convert CO3D (PyTorch3D convention) viewpoint to OpenCV-convention extrinsic.

    PyTorch3D uses ROW-vector convention: X_cam = X_world @ R_pt3d + T_pt3d.
    OpenCV uses COLUMN-vector convention: X_cam = R_opencv @ X_world + T_opencv.
    So R_opencv (in orientation terms) is R_pt3d.T.
    PyTorch3D camera frame is (-x, -y, +z), OpenCV is (+x, +y, +z), so we
    additionally left-multiply by diag(-1, -1, 1) to flip the camera axes.
    Reference: pytorch3d.utils.opencv_from_cameras_projection.
    """
    R_pt3d = np.asarray(viewpoint["R"], dtype=np.float32)
    T_pt3d = np.asarray(viewpoint["T"], dtype=np.float32)
    # Row-vector -> column-vector: transpose
    R = R_pt3d.T
    T = T_pt3d
    # Flip camera-frame axes: PyTorch3D (-x, -y, +z) -> OpenCV (+x, +y, +z)
    R = CO3D_TO_OPENCV @ R
    T = CO3D_TO_OPENCV @ T
    return np.concatenate([R, T[:, None]], axis=1).astype(np.float32)


def read_co3d_depth(depth_path: Path, scale_adjustment: float | None) -> np.ndarray:
    cv = require_cv2()
    depth_raw = cv.imread(str(depth_path), cv.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth map: {depth_path}")
    depth = depth_raw.astype(np.float32)
    if depth_raw.dtype == np.uint16:
        depth = depth / 1000.0
    if scale_adjustment is not None:
        depth = depth * float(scale_adjustment)
    return depth


def resolve_co3d_asset(gt_root: Path, category: str, rel_path: str) -> Path:
    candidates = [
        gt_root / category / rel_path,
        gt_root / rel_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def preprocess_gt_like_vggt_input(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_h, target_w = target_hw
    orig_h, orig_w = depth.shape[:2]
    new_w = target_w
    new_h = round(orig_h * (new_w / orig_w) / 14) * 14
    scale = new_w / orig_w

    cv = require_cv2()
    depth = cv.resize(depth, (new_w, new_h), interpolation=cv.INTER_NEAREST)
    mask = cv.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv.INTER_NEAREST) > 0

    intrinsics = intrinsics.copy()
    intrinsics[:2, :] *= scale

    if new_h > target_h:
        start_y = (new_h - target_h) // 2
        depth = depth[start_y : start_y + target_h]
        mask = mask[start_y : start_y + target_h]
        intrinsics[1, 2] -= start_y
    elif new_h < target_h:
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        depth = np.pad(depth, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=False)
        intrinsics[1, 2] += pad_top

    depth[~mask] = 0.0
    return depth.astype(np.float32), mask.astype(bool), intrinsics.astype(np.float32)


def preprocess_intrinsics_like_vggt_input(
    intrinsics: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    target_h, target_w = target_hw
    orig_h, orig_w = source_hw
    new_w = target_w
    new_h = round(orig_h * (new_w / orig_w) / 14) * 14
    scale = new_w / orig_w

    intrinsics = intrinsics.copy()
    intrinsics[:2, :] *= scale
    if new_h > target_h:
        start_y = (new_h - target_h) // 2
        intrinsics[1, 2] -= start_y
    elif new_h < target_h:
        pad_top = (target_h - new_h) // 2
        intrinsics[1, 2] += pad_top
    return intrinsics.astype(np.float32)


def load_co3d_gt_reference(
    gt_root: Path,
    scene_dir: Path,
    image_paths: list[str],
    device: torch.device,
    image_size_hw: tuple[int, int],
    load_geometry: bool = True,
) -> dict[str, torch.Tensor]:
    category, sequence_name = parse_flat_scene_name(scene_dir)
    annotations = load_co3d_category_annotations(gt_root, category)
    sequence_frames = [item for item in annotations if item.get("sequence_name") == sequence_name]
    if not sequence_frames:
        raise ValueError(f"No CO3D GT frames for {category}/{sequence_name}")

    by_basename = {Path(item["image"]["path"]).name: item for item in sequence_frames}
    depths = []
    masks = []
    extrinsics = []
    intrinsics = []
    world_points = []

    for image_path in image_paths:
        basename = Path(image_path).name
        if basename not in by_basename:
            raise ValueError(f"No GT annotation for frame {basename} in {category}/{sequence_name}")
        anno = by_basename[basename]
        gt_image_path = resolve_co3d_asset(gt_root, category, anno["image"]["path"])
        if not gt_image_path.exists():
            raise FileNotFoundError(f"Could not resolve GT image path: {gt_image_path}")
        with Image.open(gt_image_path) as gt_image:
            image_width, image_height = gt_image.size

        extrinsic = co3d_viewpoint_to_opencv_extrinsic(anno["viewpoint"])
        intrinsic = co3d_ndc_intrinsics_to_pixels(anno["viewpoint"], (int(image_height), int(image_width)))

        if load_geometry:
            depth_rel = anno["depth"]["path"]
            mask_rel = anno["depth"].get("mask_path") or anno.get("mask", {}).get("path")
            depth_path = resolve_co3d_asset(gt_root, category, depth_rel)
            mask_path = resolve_co3d_asset(gt_root, category, mask_rel)
            depth = read_co3d_depth(depth_path, anno["depth"].get("scale_adjustment"))
            cv = require_cv2()
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read depth mask: {mask_path}")
            mask = mask > 128

            depth, mask, intrinsic = preprocess_gt_like_vggt_input(depth, mask, intrinsic, image_size_hw)
            world = unproject_depth_map_to_point_map(depth[None, ..., None], extrinsic[None], intrinsic[None])[0]
            depths.append(depth)
            masks.append(mask)
            world_points.append(world)
        else:
            intrinsic = preprocess_intrinsics_like_vggt_input(
                intrinsic,
                (int(image_height), int(image_width)),
                image_size_hw,
            )
        extrinsics.append(extrinsic)
        intrinsics.append(intrinsic)

    extrinsics_t = torch.from_numpy(np.stack(extrinsics).astype(np.float32)).to(device).unsqueeze(0)
    intrinsics_t = torch.from_numpy(np.stack(intrinsics).astype(np.float32)).to(device).unsqueeze(0)
    refs = {
        "pose_enc": extri_intri_to_pose_encoding(extrinsics_t, intrinsics_t, image_size_hw),
        "extrinsic": extrinsics_t,
        "intrinsic": intrinsics_t,
    }
    if load_geometry:
        refs.update(
            {
                "depth": torch.from_numpy(np.stack(depths).astype(np.float32)).to(device).unsqueeze(0)[..., None],
                "world_points": torch.from_numpy(np.stack(world_points).astype(np.float32)).to(device).unsqueeze(0),
                "point_mask": torch.from_numpy(np.stack(masks).astype(bool)).to(device).unsqueeze(0),
            }
        )
    return refs


def normalized_mse(
    pred: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    reference = reference.detach()
    diff = pred - reference
    if mask is not None:
        while mask.ndim < diff.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.to(device=diff.device, dtype=torch.bool).expand_as(diff)
        if mask.sum() == 0:
            return (0.0 * pred).mean()
        scale = reference[mask].abs().mean().clamp_min(eps)
        return ((diff[mask]) / scale).pow(2).mean()

    scale = reference.abs().mean().clamp_min(eps)
    return (diff / scale).pow(2).mean()


def attack_loss(
    adv: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    terms: dict[str, torch.Tensor] = {}
    point_mask = reference.get("point_mask")

    if "depth" in adv and "depth" in reference:
        terms["depth"] = normalized_mse(adv["depth"], reference["depth"], mask=point_mask)
    if "pose_enc" in adv and "pose_enc" in reference:
        terms["pose"] = normalized_mse(adv["pose_enc"], reference["pose_enc"])
    if "world_points" in adv and "world_points" in reference:
        terms["points"] = normalized_mse(adv["world_points"], reference["world_points"], mask=point_mask)

    if not terms:
        raise RuntimeError("No comparable VGGT outputs were produced.")

    total = torch.zeros((), device=next(iter(adv.values())).device)
    term_values: dict[str, float] = {}
    for name, value in terms.items():
        weight = weights.get(name, 1.0)
        total = total + weight * value
        term_values[name] = float(value.detach().cpu())
    term_values["total"] = float(total.detach().cpu())
    return total, term_values


def has_attack_reference_terms(reference: dict) -> bool:
    return any(key in reference for key in ("depth", "pose_enc", "world_points"))


def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    clean_preds: dict[str, torch.Tensor],
    dtype: torch.dtype,
    steps: int,
    eps: float,
    alpha: float,
    random_start: bool,
    weights: dict[str, float],
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    base = images.detach()
    if random_start:
        delta = torch.empty_like(base).uniform_(-eps, eps)
    else:
        delta = torch.zeros_like(base)
    adv_images = (base + delta).clamp(0.0, 1.0).detach()

    history: list[dict[str, float]] = []
    for step in range(steps):
        adv_images.requires_grad_(True)
        preds = forward_vggt(model, adv_images, dtype)
        loss, terms = attack_loss(preds, clean_preds, weights)

        model.zero_grad(set_to_none=True)
        if adv_images.grad is not None:
            adv_images.grad.zero_()
        loss.backward()

        with torch.no_grad():
            grad = adv_images.grad
            if grad is None:
                raise RuntimeError("PGD gradient is None; check the forward graph.")
            adv_images = adv_images + alpha * grad.sign()
            delta = (adv_images - base).clamp(-eps, eps)
            adv_images = (base + delta).clamp(0.0, 1.0).detach()

        terms["step"] = step + 1
        history.append(terms)
        print(
            f"[pgd] step {step + 1:03d}/{steps:03d} "
            f"loss={terms['total']:.6f} "
            f"depth={terms.get('depth', 0.0):.6f} "
            f"pose={terms.get('pose', 0.0):.6f} "
            f"points={terms.get('points', 0.0):.6f}"
        )

    return adv_images, history


def resolve_patch_box(
    image_hw: tuple[int, int],
    patch_size: int,
    patch_x: int,
    patch_y: int,
) -> tuple[int, int, int, int]:
    height, width = image_hw
    patch_h = min(patch_size, height)
    patch_w = min(patch_size, width)
    if patch_x < 0:
        patch_x = (width - patch_w) // 2
    if patch_y < 0:
        patch_y = (height - patch_h) // 2
    patch_x = int(np.clip(patch_x, 0, width - patch_w))
    patch_y = int(np.clip(patch_y, 0, height - patch_h))
    return patch_x, patch_y, patch_h, patch_w


def apply_adversarial_patch(
    images: torch.Tensor,
    patch: torch.Tensor,
    patch_x: int,
    patch_y: int,
) -> torch.Tensor:
    _, _, height, width = images.shape
    _, _, patch_h, patch_w = patch.shape
    mask = torch.zeros((1, 1, height, width), device=images.device, dtype=images.dtype)
    canvas = torch.zeros((1, 3, height, width), device=images.device, dtype=images.dtype)
    mask[:, :, patch_y : patch_y + patch_h, patch_x : patch_x + patch_w] = 1.0
    canvas[:, :, patch_y : patch_y + patch_h, patch_x : patch_x + patch_w] = patch.to(dtype=images.dtype)
    return (images * (1.0 - mask) + canvas * mask).clamp(0.0, 1.0)


def patch_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    reference_preds: dict[str, torch.Tensor],
    dtype: torch.dtype,
    steps: int,
    alpha: float,
    patch_size: int,
    patch_x: int,
    patch_y: int,
    weights: dict[str, float],
) -> tuple[torch.Tensor, list[dict[str, float]], torch.Tensor, dict[str, int]]:
    base = images.detach()
    image_hw = tuple(base.shape[-2:])
    patch_x, patch_y, patch_h, patch_w = resolve_patch_box(image_hw, patch_size, patch_x, patch_y)
    patch = torch.rand((1, 3, patch_h, patch_w), device=base.device, dtype=torch.float32)
    patch.requires_grad_(True)
    patch.retain_grad()

    history: list[dict[str, float]] = []
    for step in range(steps):
        adv_images = apply_adversarial_patch(base, patch, patch_x, patch_y)
        preds = forward_vggt(model, adv_images, dtype)
        loss, terms = attack_loss(preds, reference_preds, weights)

        model.zero_grad(set_to_none=True)
        if patch.grad is not None:
            patch.grad.zero_()
        loss.backward()

        with torch.no_grad():
            grad = patch.grad
            if grad is None:
                raise RuntimeError("Patch gradient is None; check the forward graph.")
            patch = (patch + alpha * grad.sign()).clamp(0.0, 1.0).detach()
            patch.requires_grad_(True)
            patch.retain_grad()

        terms["step"] = step + 1
        history.append(terms)
        print(
            f"[patch] step {step + 1:03d}/{steps:03d} "
            f"loss={terms['total']:.6f} "
            f"depth={terms.get('depth', 0.0):.6f} "
            f"pose={terms.get('pose', 0.0):.6f} "
            f"points={terms.get('points', 0.0):.6f}"
        )

    adv_images = apply_adversarial_patch(base, patch, patch_x, patch_y).detach()
    patch_meta = {
        "patch_x": patch_x,
        "patch_y": patch_y,
        "patch_h": patch_h,
        "patch_w": patch_w,
    }
    return adv_images, history, patch.detach(), patch_meta


def tensor_to_numpy(preds: dict[str, torch.Tensor], image_size_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "extrinsic" in preds:
        out["extrinsic"] = preds["extrinsic"].detach().float().cpu().numpy().astype(np.float32)
    if "intrinsic" in preds:
        out["intrinsic"] = preds["intrinsic"].detach().float().cpu().numpy().astype(np.float32)
    if "pose_enc" in preds:
        out["pose_enc"] = preds["pose_enc"].detach().float().cpu().numpy().astype(np.float32)
        if "extrinsic" not in out or "intrinsic" not in out:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], image_size_hw)
            out.setdefault("extrinsic", extrinsic.detach().float().cpu().numpy().astype(np.float32))
            out.setdefault("intrinsic", intrinsic.detach().float().cpu().numpy().astype(np.float32))
    for key in ("depth", "depth_conf", "world_points", "world_points_conf"):
        if key in preds:
            out[key] = preds[key].detach().float().cpu().numpy().astype(np.float32)
    for key in ("track", "vis", "conf"):
        if key in preds:
            out[key] = preds[key].detach().float().cpu().numpy().astype(np.float32)
    return out


def relative_rmse(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    denom = b.detach().abs().mean().clamp_min(eps)
    return float((((a - b.detach()) / denom).pow(2).mean().sqrt()).detach().cpu())


def normalize_to_first_frame(extrinsic: torch.Tensor) -> torch.Tensor:
    """Re-express extrinsics so that the first camera sits at the world origin.

    extrinsic: [B, N, 3, 4] camera-from-world (OpenCV convention).
    Returns same shape, with extrinsic_new[:, 0] = [I | 0] up to numerical noise.
    This is gauge-fixing: pair-wise relative poses are gauge-invariant in theory
    but doing this explicitly removes any global-frame mismatch between the
    VGGT prediction (first-frame-as-origin) and CO3D GT (scene-centered).
    """
    R0 = extrinsic[..., 0, :3, :3]                              # [B, 3, 3]
    t0 = extrinsic[..., 0, :3, 3]                               # [B, 3]
    R = extrinsic[..., :3, :3]                                  # [B, N, 3, 3]
    t = extrinsic[..., :3, 3]                                   # [B, N, 3]
    # New world frame = old cam0 frame. The pose mapping old-world -> new-world
    # is X_new = R0 @ X_old + t0. So the new extrinsic for cam i is:
    #   R_new_i = R_i @ R0^T
    #   t_new_i = t_i - R_i @ R0^T @ t0 = t_i - R_new_i @ t0
    R_new = torch.matmul(R, R0.unsqueeze(-3).transpose(-1, -2))
    t_new = t - torch.matmul(R_new, t0.unsqueeze(-3).unsqueeze(-1)).squeeze(-1)
    out = extrinsic.clone()
    out[..., :3, :3] = R_new
    out[..., :3, 3] = t_new
    return out


def rotation_angle_deg(rel_a: torch.Tensor, rel_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rel = torch.matmul(rel_a.transpose(-1, -2), rel_b)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    return torch.rad2deg(torch.acos(cos))


def translation_angle_deg_and_valid(
    t_a: torch.Tensor,
    t_b: torch.Tensor,
    eps: float = 1e-6,
    ambiguity: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_a = t_a.norm(dim=-1)
    norm_b = t_b.norm(dim=-1)
    valid = (norm_a > eps) & (norm_b > eps)
    cos = torch.zeros_like(norm_a)
    cos[valid] = (t_a[valid] * t_b[valid]).sum(dim=-1) / (norm_a[valid] * norm_b[valid])
    cos = cos.clamp(-1.0 + eps, 1.0 - eps)
    angle = torch.rad2deg(torch.acos(cos))
    if ambiguity:
        angle = torch.minimum(angle, (180.0 - angle).abs())
    return angle, valid


def camera_auc_from_max_errors(r_errors: np.ndarray, t_errors: np.ndarray, threshold: int) -> float:
    """Compute AUC@threshold over max(rotation_error, translation_error) in degrees.

    Follows the standard PoseDiffusion / VGGSfM / VGGT-style definition:
    integrate the CDF of max-pair-errors over [0, threshold] degrees, then
    normalize by threshold so the score lies in [0, 1].
    """
    if r_errors.size == 0 or t_errors.size == 0:
        return float("nan")
    error_matrix = np.concatenate((r_errors[:, None], t_errors[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    normalized_histogram = histogram.astype(float) / float(len(max_errors))
    # Cumulative-sum gives the empirical CDF evaluated at each integer degree;
    # mean of CDF over [0, threshold) = (1/threshold) * sum_{t=1..threshold} F(t).
    return float(np.mean(np.cumsum(normalized_histogram)))


def pose_metrics_from_pair_errors(
    records: dict[str, list[float]],
    thresholds: tuple[int, ...],
    prefix: str,
) -> dict[str, float]:
    r_err_np = np.asarray(records.get("rotation_deg", []), dtype=np.float64)
    t_err_np = np.asarray(records.get("translation_deg", []), dtype=np.float64)
    if r_err_np.size == 0 or t_err_np.size == 0:
        return {}

    r_err = torch.from_numpy(r_err_np)
    t_err = torch.from_numpy(t_err_np)
    metrics = {
        f"{prefix}_pair_count": float(len(r_err_np)),
    }
    for threshold in thresholds:
        metrics[f"{prefix}_rra@{threshold}"] = float((r_err < threshold).float().mean().detach().cpu())
        metrics[f"{prefix}_rta@{threshold}"] = float((t_err < threshold).float().mean().detach().cpu())
        metrics[f"{prefix}_auc@{threshold}"] = camera_auc_from_max_errors(r_err_np, t_err_np, threshold)
    return metrics


def camera_pair_error_records(
    clean: dict[str, torch.Tensor],
    adv: dict[str, torch.Tensor],
    image_size_hw: tuple[int, int],
) -> dict[str, list[float]]:
    if "extrinsic" in clean:
        clean_extri = clean["extrinsic"].detach().float()
    elif "pose_enc" in clean:
        clean_extri, _ = pose_encoding_to_extri_intri(clean["pose_enc"], image_size_hw)
        clean_extri = clean_extri.detach().float()
    else:
        return {"rotation_deg": [], "translation_deg": []}

    if "extrinsic" in adv:
        adv_extri = adv["extrinsic"].detach().float()
    elif "pose_enc" in adv:
        adv_extri, _ = pose_encoding_to_extri_intri(adv["pose_enc"], image_size_hw)
        adv_extri = adv_extri.detach().float()
    else:
        return {"rotation_deg": [], "translation_deg": []}

    # Gauge-fix: both prediction and GT are expressed relative to their own
    # first-frame, so global-frame mismatch cannot leak into pair errors.
    clean_extri = normalize_to_first_frame(clean_extri)
    adv_extri = normalize_to_first_frame(adv_extri)

    rotations_clean: list[torch.Tensor] = []
    rotations_adv: list[torch.Tensor] = []
    translations_clean: list[torch.Tensor] = []
    translations_adv: list[torch.Tensor] = []

    _, n_frames = clean_extri.shape[:2]
    for i in range(n_frames):
        r_i_clean = clean_extri[:, i, :3, :3]
        t_i_clean = clean_extri[:, i, :3, 3]
        r_i_adv = adv_extri[:, i, :3, :3]
        t_i_adv = adv_extri[:, i, :3, 3]
        for j in range(i + 1, n_frames):
            r_j_clean = clean_extri[:, j, :3, :3]
            t_j_clean = clean_extri[:, j, :3, 3]
            r_j_adv = adv_extri[:, j, :3, :3]
            t_j_adv = adv_extri[:, j, :3, 3]

            rotations_clean.append(torch.matmul(r_j_clean, r_i_clean.transpose(-1, -2)))
            rotations_adv.append(torch.matmul(r_j_adv, r_i_adv.transpose(-1, -2)))
            translations_clean.append(t_j_clean - torch.matmul(rotations_clean[-1], t_i_clean[..., None]).squeeze(-1))
            translations_adv.append(t_j_adv - torch.matmul(rotations_adv[-1], t_i_adv[..., None]).squeeze(-1))

    if not rotations_clean:
        return {"rotation_deg": [], "translation_deg": []}

    rot_clean = torch.cat(rotations_clean, dim=0)
    rot_adv = torch.cat(rotations_adv, dim=0)
    trans_clean = torch.cat(translations_clean, dim=0)
    trans_adv = torch.cat(translations_adv, dim=0)

    r_err = rotation_angle_deg(rot_clean, rot_adv)
    t_err_all, valid_t = translation_angle_deg_and_valid(trans_clean, trans_adv)
    if valid_t.sum() == 0:
        return {"rotation_deg": [], "translation_deg": []}
    r_err = r_err[valid_t]
    t_err = t_err_all[valid_t]
    return {
        "rotation_deg": r_err.detach().cpu().numpy().astype(float).tolist(),
        "translation_deg": t_err.detach().cpu().numpy().astype(float).tolist(),
    }


def camera_metrics_from_pair_errors(records: dict[str, list[float]], thresholds: tuple[int, ...] = (5, 10, 20, 30)) -> dict[str, float]:
    return pose_metrics_from_pair_errors(records, thresholds=thresholds, prefix="camera")


def camera_paper_metrics(clean: dict[str, torch.Tensor], adv: dict[str, torch.Tensor], image_size_hw: tuple[int, int]) -> dict[str, float]:
    return camera_metrics_from_pair_errors(camera_pair_error_records(clean, adv, image_size_hw))


def sample_point_cloud(points: np.ndarray, max_points: int, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        points = points[mask.astype(bool)]
    else:
        points = points.reshape(-1, 3)
    points = points.reshape(-1, 3).astype(np.float32)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[idx]
    return points


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form Sim(3) alignment: find s, R, t minimizing ||s * R @ src_i + t - dst_i||^2.

    src, dst: [N, 3] paired point arrays. Returns (scale, R(3x3), t(3,)).
    Reference: Umeyama 1991.
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / n
    scale = float(np.trace(np.diag(D) @ S) / max(var_src, 1e-12))
    t = mu_dst - scale * R @ mu_src
    return scale, R.astype(np.float32), t.astype(np.float32)


def estimate_sim3_from_paired_pointmaps(
    src_pointmap: np.ndarray,
    dst_pointmap: np.ndarray,
    mask: np.ndarray | None = None,
    max_samples: int = 20000,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Estimate Sim(3) from two pixel-aligned pointmaps (same H, W).

    src_pointmap, dst_pointmap: [N, H, W, 3] OR flattened [K, 3] with point-to-point
    correspondence (i-th pred point corresponds to i-th GT point).
    mask: optional boolean mask of valid GT points, same leading shape.
    """
    src = src_pointmap.reshape(-1, 3).astype(np.float32)
    dst = dst_pointmap.reshape(-1, 3).astype(np.float32)
    if mask is not None:
        m = mask.reshape(-1).astype(bool)
        src = src[m]
        dst = dst[m]
    # Drop non-finite entries
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[finite]
    dst = dst[finite]
    if len(src) < 8:
        return None
    if len(src) > max_samples:
        idx = np.linspace(0, len(src) - 1, max_samples, dtype=np.int64)
        src = src[idx]
        dst = dst[idx]
    return umeyama_sim3(src, dst)


def camera_centers_from_extrinsics(extrinsic: np.ndarray) -> np.ndarray:
    extrinsic = extrinsic.reshape(-1, 3, 4).astype(np.float32)
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    centers = -np.matmul(rotation.transpose(0, 2, 1), translation[..., None]).squeeze(-1)
    return centers.astype(np.float32)


def estimate_sim3_from_camera_centers(
    reference: dict[str, torch.Tensor],
    pred_np: dict[str, np.ndarray],
    image_size_hw: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray] | None:
    if "extrinsic" not in reference:
        return None
    if "extrinsic" in pred_np:
        pred_extrinsic = pred_np["extrinsic"][0]
    elif "pose_enc" in pred_np:
        pose_enc = torch.from_numpy(pred_np["pose_enc"]).float()
        pred_extrinsic_t, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw)
        pred_extrinsic = pred_extrinsic_t.detach().cpu().numpy()[0]
    else:
        return None

    ref_extrinsic = reference["extrinsic"][0].detach().float().cpu().numpy()
    pred_centers = camera_centers_from_extrinsics(pred_extrinsic)
    ref_centers = camera_centers_from_extrinsics(ref_extrinsic)
    if len(pred_centers) != len(ref_centers) or len(pred_centers) < 3:
        return None
    return umeyama_sim3(pred_centers, ref_centers)


def nearest_distances(src: torch.Tensor, dst: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    mins = []
    for start in range(0, src.shape[0], chunk_size):
        chunk = src[start : start + chunk_size]
        mins.append(torch.cdist(chunk, dst).min(dim=1).values)
    return torch.cat(mins, dim=0)


def chamfer_metrics(
    reference_points: np.ndarray,
    pred_points: np.ndarray,
    prefix: str,
    max_points: int,
    device: torch.device,
    reference_mask: np.ndarray | None = None,
    pred_mask: np.ndarray | None = None,
    sim3: tuple[float, np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """Bidirectional Chamfer between two point clouds.

    If `sim3 = (s, R, t)` is given, the pred cloud is first aligned to the
    reference frame via `pred -> s * R @ pred + t` before computing distances.
    This is the standard protocol for evaluating up-to-scale predictions
    (DUSt3R / MASt3R / VGGT report numbers under this alignment).
    """
    reference_points = sample_point_cloud(reference_points, max_points, reference_mask)
    pred_points = sample_point_cloud(pred_points, max_points, pred_mask)
    if len(reference_points) == 0 or len(pred_points) == 0:
        return {}

    if sim3 is not None:
        s, R, t = sim3
        pred_points = (s * (R @ pred_points.T).T + t).astype(np.float32)

    reference_t = torch.from_numpy(reference_points).to(device)
    pred_t = torch.from_numpy(pred_points).to(device)
    acc = nearest_distances(pred_t, reference_t).mean()
    comp = nearest_distances(reference_t, pred_t).mean()
    overall = (acc + comp) * 0.5
    return {
        f"{prefix}_accuracy": float(acc.detach().cpu()),
        f"{prefix}_completeness": float(comp.detach().cpu()),
        f"{prefix}_overall": float(overall.detach().cpu()),
    }


def reference_point_mask(reference: dict[str, torch.Tensor]) -> np.ndarray | None:
    if "point_mask" not in reference:
        return None
    return reference["point_mask"].detach().cpu().numpy()[0].astype(bool)


def depth_paper_metrics(
    reference: dict[str, torch.Tensor],
    pred: dict[str, torch.Tensor],
    image_size_hw: tuple[int, int],
    max_points: int,
    device: torch.device,
) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    reference_np = tensor_to_numpy(reference, image_size_hw)
    pred_np = tensor_to_numpy(pred, image_size_hw)
    reference_mask = reference_point_mask(reference)

    if "gt_point_cloud" in reference and all(k in pred_np for k in ("depth", "extrinsic", "intrinsic")):
        gt_points = reference["gt_point_cloud"].detach().float().cpu().numpy()
        pred_depth_points = unproject_depth_map_to_point_map(
            pred_np["depth"][0], pred_np["extrinsic"][0], pred_np["intrinsic"][0]
        )
        if "dtu_eval_root" in reference:
            metrics.update(dtu_official_chamfer_metrics(pred_depth_points, reference, "depth_align_none", device))
            return metrics
        alignment = str(reference.get("pointcloud_alignment", "none"))
        sim3 = None
        if alignment == "sim3":
            sim3 = estimate_sim3_from_camera_centers(reference, pred_np, image_size_hw)
            if sim3 is not None:
                metrics["depth_align_sim3_scale"] = float(sim3[0])
            else:
                metrics["depth_align_sim3_status"] = "skipped_missing_camera_correspondences"
                return metrics
        prefix = "depth_align_sim3" if sim3 is not None else "depth_align_none"
        metrics.update(
            chamfer_metrics(
                gt_points,
                pred_depth_points,
                prefix,
                max_points,
                device,
                sim3=sim3,
            )
        )
    elif all(k in reference_np for k in ("depth", "extrinsic", "intrinsic")) and all(
        k in pred_np for k in ("depth", "extrinsic", "intrinsic")
    ):
        reference_depth_points = unproject_depth_map_to_point_map(
            reference_np["depth"][0], reference_np["extrinsic"][0], reference_np["intrinsic"][0]
        )
        pred_depth_points = unproject_depth_map_to_point_map(
            pred_np["depth"][0], pred_np["extrinsic"][0], pred_np["intrinsic"][0]
        )
        metrics.update(
            chamfer_metrics(
                reference_depth_points,
                pred_depth_points,
                "depth_align_none",
                max_points,
                device,
                reference_mask=reference_mask,
                pred_mask=reference_mask,
                sim3=None,
            )
        )
        sim3 = estimate_sim3_from_paired_pointmaps(
            src_pointmap=pred_depth_points,
            dst_pointmap=reference_depth_points,
            mask=reference_mask,
            max_samples=max_points,
        )
        if sim3 is not None:
            metrics["depth_align_sim3_scale"] = float(sim3[0])
            metrics.update(
                chamfer_metrics(
                    reference_depth_points,
                    pred_depth_points,
                    "depth_align_sim3",
                    max_points,
                    device,
                    reference_mask=reference_mask,
                    pred_mask=reference_mask,
                    sim3=sim3,
                )
            )
        else:
            metrics["depth_align_sim3_status"] = "skipped_insufficient_correspondences"
    else:
        metrics["depth_status"] = "skipped_missing_depth_or_camera"
    return metrics


def point_paper_metrics(
    reference: dict[str, torch.Tensor],
    pred: dict[str, torch.Tensor],
    image_size_hw: tuple[int, int],
    max_points: int,
    device: torch.device,
) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    reference_np = tensor_to_numpy(reference, image_size_hw)
    pred_np = tensor_to_numpy(pred, image_size_hw)
    reference_mask = reference_point_mask(reference)

    if "gt_point_cloud" in reference and "world_points" in pred_np:
        gt_points = reference["gt_point_cloud"].detach().float().cpu().numpy()
        alignment = str(reference.get("pointcloud_alignment", "sim3"))
        sim3 = None
        if alignment == "sim3":
            sim3 = estimate_sim3_from_camera_centers(reference, pred_np, image_size_hw)
            if sim3 is not None:
                metrics["point_sim3_scale"] = float(sim3[0])
            else:
                metrics["point_status"] = "skipped_missing_camera_correspondences_for_sim3"
                return metrics
        metrics.update(
            chamfer_metrics(
                gt_points,
                pred_np["world_points"][0],
                "point",
                max_points,
                device,
                sim3=sim3,
            )
        )
    elif "world_points" in reference_np and "world_points" in pred_np:
        wp_sim3 = estimate_sim3_from_paired_pointmaps(
            src_pointmap=pred_np["world_points"][0],
            dst_pointmap=reference_np["world_points"][0],
            mask=reference_mask,
            max_samples=max_points,
        )
        if wp_sim3 is not None:
            metrics["point_sim3_scale"] = float(wp_sim3[0])
            metrics.update(
                chamfer_metrics(
                    reference_np["world_points"][0],
                    pred_np["world_points"][0],
                    "point",
                    max_points,
                    device,
                    reference_mask=reference_mask,
                    pred_mask=reference_mask,
                    sim3=wp_sim3,
                )
            )
        else:
            metrics["point_status"] = "skipped_insufficient_correspondences"
    else:
        metrics["point_status"] = "skipped_missing_pointmap"
    return metrics


def vggt_paper_metrics(
    reference: dict[str, torch.Tensor],
    pred: dict[str, torch.Tensor],
    image_size_hw: tuple[int, int],
    max_points: int,
    device: torch.device,
    eval_tasks: set[str] | None = None,
) -> dict[str, float | str]:
    tasks = eval_tasks or {"camera", "depth", "point"}
    metrics: dict[str, float | str] = {}
    if "camera" in tasks:
        metrics.update(camera_paper_metrics(reference, pred, image_size_hw))
    if "depth" in tasks:
        metrics.update(depth_paper_metrics(reference, pred, image_size_hw, max_points, device))
    if "point" in tasks:
        metrics.update(point_paper_metrics(reference, pred, image_size_hw, max_points, device))

    if "tracking" in tasks:
        metrics["tracking_image_matching"] = "not_computed"
    return metrics


def vggt_paper_records(
    reference: dict[str, torch.Tensor],
    pred: dict[str, torch.Tensor],
    image_size_hw: tuple[int, int],
    max_points: int,
    device: torch.device,
    eval_tasks: set[str] | None = None,
) -> dict:
    tasks = eval_tasks or {"camera", "depth", "point"}
    metrics = vggt_paper_metrics(reference, pred, image_size_hw, max_points, device, eval_tasks=tasks)
    depth_keys = [
        f"depth_align_{align}_{name}"
        for align in ("none", "sim3")
        for name in ("accuracy", "completeness", "overall")
    ]
    records: dict = {}
    if "camera" in tasks:
        records["camera_pair_errors"] = camera_pair_error_records(reference, pred, image_size_hw)
    if "depth" in tasks:
        records["depth_scene_metrics"] = {
            key: metrics[key]
            for key in depth_keys
            if key in metrics
        }
    if "point" in tasks:
        records["point_scene_metrics"] = {
            key: metrics[key]
            for key in ("point_accuracy", "point_completeness", "point_overall")
            if key in metrics
        }
    if "tracking" in tasks:
        records["tracking_image_matching_pair_errors"] = {
            "rotation_deg": [],
            "translation_deg": [],
            "status": "not_computed",
        }
    return records


def relative_pose_from_extrinsics_np(extri_i: np.ndarray, extri_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return camera-j-from-camera-i relative pose from world-to-camera extrinsics."""
    r_i = extri_i[:3, :3]
    t_i = extri_i[:3, 3]
    r_j = extri_j[:3, :3]
    t_j = extri_j[:3, 3]
    r_rel = r_j @ r_i.T
    t_rel = t_j - r_rel @ t_i
    return r_rel.astype(np.float64), t_rel.astype(np.float64)


def rotation_error_deg_np(r_gt: np.ndarray, r_pred: np.ndarray, eps: float = 1e-7) -> float:
    rel = r_gt.T @ r_pred
    cos = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0 + eps, 1.0 - eps)
    return float(np.degrees(np.arccos(cos)))


def translation_error_deg_np(
    t_gt: np.ndarray,
    t_pred: np.ndarray,
    eps: float = 1e-12,
    ambiguity: bool = True,
) -> float | None:
    norm_gt = float(np.linalg.norm(t_gt))
    norm_pred = float(np.linalg.norm(t_pred))
    if norm_gt <= eps or norm_pred <= eps:
        return None
    cos = float(np.dot(t_gt, t_pred) / (norm_gt * norm_pred))
    angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    if ambiguity:
        angle = min(angle, abs(180.0 - angle))
    return angle


def normalized_image_points(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    points = points.astype(np.float64).reshape(-1, 1, 2)
    intrinsic = intrinsic.astype(np.float64)
    cv = require_cv2()
    return cv.undistortPoints(points, intrinsic, None).reshape(-1, 2)


def estimate_relative_pose_from_tracks(
    points_i: np.ndarray,
    points_j: np.ndarray,
    intrinsic_i: np.ndarray,
    intrinsic_j: np.ndarray,
    ransac_thresh: float,
    min_inliers: int,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    if len(points_i) < max(8, min_inliers) or len(points_j) < max(8, min_inliers):
        return None

    pts_i = normalized_image_points(points_i, intrinsic_i)
    pts_j = normalized_image_points(points_j, intrinsic_j)
    cv = require_cv2()
    essential, inlier_mask = cv.findEssentialMat(
        pts_i,
        pts_j,
        cameraMatrix=np.eye(3, dtype=np.float64),
        method=cv.RANSAC,
        prob=0.999,
        threshold=ransac_thresh,
    )
    if essential is None or inlier_mask is None:
        return None

    best: tuple[np.ndarray, np.ndarray, int] | None = None
    essential = essential.reshape(-1, 3, 3)
    for e_mat in essential:
        inliers, r_pred, t_pred, _ = cv.recoverPose(
            e_mat,
            pts_i,
            pts_j,
            cameraMatrix=np.eye(3, dtype=np.float64),
            mask=inlier_mask,
        )
        if inliers < min_inliers:
            continue
        if best is None or inliers > best[2]:
            best = (r_pred.astype(np.float64), t_pred.reshape(3).astype(np.float64), int(inliers))
    return best


def build_aliked_extractor(
    max_keypoints: int,
    detection_threshold: float,
    device: torch.device,
):
    try:
        from lightglue import ALIKED
    except Exception as exc:
        return None, f"skipped_missing_lightglue_aliked: {exc}"

    extractor = ALIKED(max_num_keypoints=max_keypoints, detection_threshold=detection_threshold).to(device).eval()
    return extractor, None


def extract_aliked_keypoints(image: torch.Tensor, extractor) -> tuple[torch.Tensor | None, str | None]:
    with torch.no_grad():
        data = extractor.extract(image, invalid_mask=None)
    keypoints = data.get("keypoints")
    if keypoints is None or keypoints.shape[1] == 0:
        return None, "skipped_no_aliked_keypoints"
    return keypoints.float(), None


def build_pair_indices(num_frames: int, max_pairs: int) -> list[tuple[int, int]]:
    pairs = [(i, j) for i in range(num_frames) for j in range(i + 1, num_frames)]
    if max_pairs > 0 and len(pairs) > max_pairs:
        selected = np.linspace(0, len(pairs) - 1, max_pairs, dtype=int)
        pairs = [pairs[int(idx)] for idx in selected]
    return pairs


def read_matching_pairs(pair_file: Path | None, image_paths: list[str]) -> list[tuple[int, int]] | None:
    if pair_file is None:
        return None
    name_to_idx = {Path(path).name: idx for idx, path in enumerate(image_paths)}
    stem_to_idx = {Path(path).stem: idx for idx, path in enumerate(image_paths)}
    pairs: list[tuple[int, int]] = []
    with open(pair_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            if parts[0].isdigit() and parts[1].isdigit():
                i, j = int(parts[0]), int(parts[1])
            else:
                token_i = Path(parts[0]).name
                token_j = Path(parts[1]).name
                i = name_to_idx.get(token_i, stem_to_idx.get(Path(token_i).stem, -1))
                j = name_to_idx.get(token_j, stem_to_idx.get(Path(token_j).stem, -1))
            if i < 0 or j < 0 or i >= len(image_paths) or j >= len(image_paths):
                raise ValueError(f"Pair {parts[:2]} from {pair_file} is not present in image_paths")
            pairs.append((i, j))
    return pairs


def tracking_image_matching_records(
    model: torch.nn.Module,
    images: torch.Tensor,
    reference: dict[str, torch.Tensor],
    dtype: torch.dtype,
    max_keypoints: int,
    detection_threshold: float,
    ransac_thresh: float,
    min_inliers: int,
    min_visibility: float,
    min_confidence: float,
    max_pairs: int,
    pairs: list[tuple[int, int]] | None = None,
) -> dict[str, list[float] | str | int]:
    """VGGT paper-style image matching: ALIKED queries -> VGGT tracks -> Essential matrix."""
    if "extrinsic" not in reference or "intrinsic" not in reference:
        return {"rotation_deg": [], "translation_deg": [], "status": "skipped_missing_gt_cameras"}

    num_frames = images.shape[0]
    pairs = pairs if pairs is not None else build_pair_indices(num_frames, max_pairs)
    if not pairs:
        return {"rotation_deg": [], "translation_deg": [], "status": "skipped_not_enough_frames"}

    records: dict[str, list[float] | str | int] = {
        "rotation_deg": [],
        "translation_deg": [],
        "status": "computed",
        "attempted_pairs": len(pairs),
        "valid_pairs": 0,
    }

    extrinsics = reference["extrinsic"][0].detach().float().cpu().numpy()
    intrinsics = reference["intrinsic"][0].detach().float().cpu().numpy()
    height, width = images.shape[-2:]
    extractor, extractor_error = build_aliked_extractor(max_keypoints, detection_threshold, images.device)
    if extractor is None:
        return {"rotation_deg": [], "translation_deg": [], "status": extractor_error or "skipped_no_aliked"}

    for pair_idx, (i, j) in enumerate(pairs):
        query_points, skip_reason = extract_aliked_keypoints(images[i], extractor)
        if query_points is None:
            records["status"] = skip_reason or "skipped_keypoint_failure"
            break

        image_pair = torch.stack([images[i], images[j]], dim=0)
        with torch.no_grad():
            preds = forward_vggt(model, image_pair, dtype, query_points=query_points)
        if "track" not in preds:
            records["status"] = "skipped_missing_track_head"
            break

        tracks = preds["track"][0].detach().float()
        src = tracks[0]
        dst = tracks[1]
        finite = torch.isfinite(src).all(dim=-1) & torch.isfinite(dst).all(dim=-1)
        in_bounds = (
            (src[:, 0] >= 0)
            & (src[:, 0] < width)
            & (src[:, 1] >= 0)
            & (src[:, 1] < height)
            & (dst[:, 0] >= 0)
            & (dst[:, 0] < width)
            & (dst[:, 1] >= 0)
            & (dst[:, 1] < height)
        )
        valid = finite & in_bounds
        if "vis" in preds:
            valid = valid & (preds["vis"][0, 1].detach().float() >= min_visibility)
        if "conf" in preds:
            valid = valid & (preds["conf"][0, 1].detach().float() >= min_confidence)

        if int(valid.sum().item()) < max(8, min_inliers):
            continue

        src_np = src[valid].detach().cpu().numpy().astype(np.float64)
        dst_np = dst[valid].detach().cpu().numpy().astype(np.float64)
        pose = estimate_relative_pose_from_tracks(
            src_np,
            dst_np,
            intrinsics[i],
            intrinsics[j],
            ransac_thresh=ransac_thresh,
            min_inliers=min_inliers,
        )
        if pose is None:
            continue

        r_pred, t_pred, _ = pose
        r_gt, t_gt = relative_pose_from_extrinsics_np(extrinsics[i], extrinsics[j])
        t_err = translation_error_deg_np(t_gt, t_pred, ambiguity=True)
        if t_err is None:
            continue

        records["rotation_deg"].append(rotation_error_deg_np(r_gt, r_pred))
        records["translation_deg"].append(t_err)
        records["valid_pairs"] = int(records["valid_pairs"]) + 1
        print(
            f"[matching] pair {pair_idx + 1:03d}/{len(pairs):03d} "
            f"frames=({i},{j}) R={records['rotation_deg'][-1]:.3f} T={t_err:.3f}"
        )

    if not records["rotation_deg"]:
        records["status"] = str(records["status"]) if records["status"] != "computed" else "skipped_no_valid_pairs"
    return records


def add_tracking_matching_metrics(metrics: dict, records: dict[str, list[float] | str | int]) -> None:
    if records.get("rotation_deg") and records.get("translation_deg"):
        metrics["tracking_image_matching"] = "computed"
        metrics.update(
            pose_metrics_from_pair_errors(
                records,  # type: ignore[arg-type]
                thresholds=(5, 10, 20),
                prefix="tracking",
            )
        )
    else:
        metrics["tracking_image_matching"] = records.get("status", "not_computed")


def compare_predictions(
    clean: dict[str, torch.Tensor],
    adv: dict[str, torch.Tensor],
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "depth" in clean and "depth" in adv:
        metrics["depth_rel_rmse"] = relative_rmse(adv["depth"], clean["depth"])
    if "world_points" in clean and "world_points" in adv:
        metrics["points_rel_rmse"] = relative_rmse(adv["world_points"], clean["world_points"])
    if "pose_enc" in clean and "pose_enc" in adv:
        metrics["pose_rel_rmse"] = relative_rmse(adv["pose_enc"], clean["pose_enc"])
        metrics["translation_rmse"] = float(
            (adv["pose_enc"][..., :3] - clean["pose_enc"][..., :3]).pow(2).mean().sqrt().detach().cpu()
        )
        metrics["fov_rmse"] = float(
            (adv["pose_enc"][..., 7:9] - clean["pose_enc"][..., 7:9]).pow(2).mean().sqrt().detach().cpu()
        )

    delta = (adv_images - clean_images).detach()
    metrics["linf"] = float(delta.abs().max().cpu())
    metrics["l2_mean"] = float(delta.flatten(1).norm(dim=1).mean().cpu())
    metrics["pixel_mae"] = float(delta.abs().mean().cpu())
    return metrics


def finite_float(value) -> float | None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def metric_delta(after: dict, before: dict, key: str, larger_is_worse: bool) -> float | None:
    after_value = finite_float(after.get(key))
    before_value = finite_float(before.get(key))
    if after_value is None or before_value is None:
        return None
    delta = after_value - before_value if larger_is_worse else before_value - after_value
    return float(delta)


def scene_attack_row(summary: dict) -> dict[str, float | str | None]:
    scene = Path(summary.get("scene", "")).name or str(summary.get("scene", ""))
    metrics = summary.get("metrics", {})
    clean_vs_gt = metrics.get("clean_vs_gt", {})
    pgd_vs_gt = metrics.get("pgd_vs_gt", {})
    clean_vs_pgd = metrics.get("clean_vs_pgd", {})

    row: dict[str, float | str | None] = {
        "scene": scene,
        "attack_type": summary.get("attack_type"),
        "steps": summary.get("steps"),
        "eps": summary.get("eps"),
        "linf": finite_float(clean_vs_pgd.get("linf")),
        "pixel_mae": finite_float(clean_vs_pgd.get("pixel_mae")),
        "clean_adv_depth_rel_rmse": finite_float(clean_vs_pgd.get("depth_rel_rmse")),
        "clean_adv_points_rel_rmse": finite_float(clean_vs_pgd.get("points_rel_rmse")),
        "clean_adv_pose_rel_rmse": finite_float(clean_vs_pgd.get("pose_rel_rmse")),
        "clean_depth_align_none_overall": finite_float(clean_vs_gt.get("depth_align_none_overall")),
        "pgd_depth_align_none_overall": finite_float(pgd_vs_gt.get("depth_align_none_overall")),
        "depth_align_none_overall_delta": metric_delta(
            pgd_vs_gt, clean_vs_gt, "depth_align_none_overall", larger_is_worse=True
        ),
        "clean_depth_align_sim3_overall": finite_float(clean_vs_gt.get("depth_align_sim3_overall")),
        "pgd_depth_align_sim3_overall": finite_float(pgd_vs_gt.get("depth_align_sim3_overall")),
        "depth_align_sim3_overall_delta": metric_delta(
            pgd_vs_gt, clean_vs_gt, "depth_align_sim3_overall", larger_is_worse=True
        ),
        "clean_point_overall": finite_float(clean_vs_gt.get("point_overall")),
        "pgd_point_overall": finite_float(pgd_vs_gt.get("point_overall")),
        "point_overall_delta": metric_delta(pgd_vs_gt, clean_vs_gt, "point_overall", larger_is_worse=True),
        "clean_camera_auc30": finite_float(clean_vs_gt.get("camera_auc@30")),
        "pgd_camera_auc30": finite_float(pgd_vs_gt.get("camera_auc@30")),
        "camera_auc30_drop": metric_delta(pgd_vs_gt, clean_vs_gt, "camera_auc@30", larger_is_worse=False),
        "clean_tracking_auc20": finite_float(clean_vs_gt.get("tracking_auc@20")),
        "pgd_tracking_auc20": finite_float(pgd_vs_gt.get("tracking_auc@20")),
        "tracking_auc20_drop": metric_delta(pgd_vs_gt, clean_vs_gt, "tracking_auc@20", larger_is_worse=False),
    }
    return row


def normalized_attack_ranking(summaries: list[dict]) -> list[dict]:
    rows = [scene_attack_row(summary) for summary in summaries]
    score_weights = {
        "clean_adv_depth_rel_rmse": 0.25,
        "clean_adv_points_rel_rmse": 0.25,
        "clean_adv_pose_rel_rmse": 0.15,
        "depth_align_sim3_overall_delta": 0.15,
        "point_overall_delta": 0.15,
        "camera_auc30_drop": 0.05,
    }

    max_values: dict[str, float] = {}
    for key in score_weights:
        positives = [float(row[key]) for row in rows if finite_float(row.get(key)) is not None and float(row[key]) > 0]
        max_values[key] = max(positives) if positives else 0.0

    for row in rows:
        weighted_sum = 0.0
        used_weight = 0.0
        for key, weight in score_weights.items():
            value = finite_float(row.get(key))
            max_value = max_values[key]
            if value is None or max_value <= 0:
                continue
            weighted_sum += weight * max(0.0, value) / max_value
            used_weight += weight
        row["attack_score"] = float(weighted_sum / used_weight) if used_weight > 0 else 0.0

    rows.sort(key=lambda item: float(item.get("attack_score") or 0.0), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def write_attack_ranking(summaries: list[dict], output_dir: Path) -> list[dict]:
    ranking = normalized_attack_ranking(summaries)
    with open(output_dir / "pgd_attack_ranking.json", "w", encoding="utf-8") as f:
        json.dump(ranking, f, indent=2)

    if ranking:
        fieldnames = list(ranking[0].keys())
        with open(output_dir / "pgd_attack_ranking.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ranking)
    return ranking


def save_adv_images(adv_images: torch.Tensor, image_paths: list[str], out_dir: Path) -> None:
    img_dir = out_dir / "adv_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    frames = adv_images.detach().cpu()
    for i, img in enumerate(frames):
        stem = Path(image_paths[i]).stem
        to_pil_image(img.clamp(0, 1)).save(img_dir / f"{i:03d}_{stem}_adv.png")


def save_delta_preview(clean_images: torch.Tensor, adv_images: torch.Tensor, out_dir: Path) -> None:
    delta = (adv_images - clean_images).detach().cpu()
    vis = (delta / (2 * delta.abs().max().clamp_min(1e-8)) + 0.5).clamp(0, 1)
    grid = torch.cat([clean_images.detach().cpu(), adv_images.detach().cpu(), vis], dim=-1)
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(grid):
        to_pil_image(img).save(preview_dir / f"{i:03d}_clean_adv_delta.png")


def save_patch_image(patch: torch.Tensor | None, out_dir: Path) -> None:
    if patch is None:
        return
    patch_dir = out_dir / "patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    to_pil_image(patch.squeeze(0).detach().cpu().clamp(0, 1)).save(patch_dir / "learned_patch.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PGD attack baseline against VGGT.")
    parser.add_argument("--scene_dir", default=None, help="Single scene directory, with images under scene/images or scene itself.")
    parser.add_argument("--scenes_root", default=None, help="Batch mode: parent directory containing scene folders.")
    parser.add_argument("--scene_pattern", default="*", help="Batch mode glob pattern for scene folders.")
    parser.add_argument("--output_dir", required=True, help="Directory for metrics and adversarial outputs.")
    parser.add_argument("--clean_npz", default=None, help="Existing clean vggt_outputs.npz for this scene.")
    parser.add_argument(
        "--clean_output_root",
        default=None,
        help="Batch mode: root containing scene_name/vggt_outputs.npz clean outputs.",
    )
    parser.add_argument(
        "--gt_root",
        default=None,
        help="Task GT root. CO3D raw root for attacks/proxy loading; DTU/ETH3D roots for paper point-cloud GT.",
    )
    parser.add_argument(
        "--eval_dataset",
        choices=("auto", "co3d", "re10k", "dtu", "eth3d", "scannet1500"),
        default="auto",
        help="Original paper dataset for --eval_tasks. auto infers from a single task.",
    )
    parser.add_argument(
        "--co3d_anno_dir",
        default=None,
        help="Official CO3D annotation dir containing category_test.jgz for paper camera evaluation.",
    )
    parser.add_argument(
        "--gt_pointcloud",
        default=None,
        help="Explicit DTU/ETH3D GT point cloud PLY. Overrides auto lookup under --gt_root.",
    )
    parser.add_argument("--scan_id", type=int, default=None, help="DTU scan id for locating stlXXX_total.ply.")
    parser.add_argument("--dtu_downsample", type=float, default=0.2, help="DTU official eval-style prediction downsample density.")
    parser.add_argument("--dtu_patch_size", type=float, default=60.0, help="DTU official ObsMask patch size.")
    parser.add_argument("--dtu_max_dist", type=float, default=20.0, help="DTU official max distance threshold.")
    parser.add_argument(
        "--gt_camera_npz",
        default=None,
        help="GT cameras npz with extrinsic/intrinsic/image_paths for RE10K/DTU/ETH3D/ScanNet-1500.",
    )
    parser.add_argument(
        "--pairs_file",
        default=None,
        help="Paper tracking pair list for ScanNet-1500 style matching. Lines may be 'idx idx' or 'image image'.",
    )
    parser.add_argument("--ckpt", default="facebook/VGGT-1B", help="Hugging Face model id or local checkpoint path.")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load Hugging Face checkpoint from local cache only; useful on offline servers.",
    )
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum number of frames to attack/evaluate; 0 keeps all frames.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for frame sampling and attack initialization.")
    parser.add_argument("--attack_type", choices=("global", "patch"), default="global", help="Attack perturbation type.")
    parser.add_argument("--steps", type=int, default=10, help="PGD iterations.")
    parser.add_argument("--eps", type=float, default=8 / 255, help="L-infinity perturbation budget in [0, 1] pixels.")
    parser.add_argument("--alpha", type=float, default=1 / 255, help="PGD step size in [0, 1] pixels.")
    parser.add_argument("--patch_size", type=int, default=96, help="Square patch size in preprocessed input pixels.")
    parser.add_argument("--patch_alpha", type=float, default=None, help="Patch PGD step size; defaults to --alpha.")
    parser.add_argument("--patch_x", type=int, default=-1, help="Patch left coordinate; -1 centers the patch.")
    parser.add_argument("--patch_y", type=int, default=-1, help="Patch top coordinate; -1 centers the patch.")
    parser.add_argument("--no_random_start", action="store_true", help="Start PGD from the clean images.")
    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--pose_weight", type=float, default=0.2)
    parser.add_argument("--points_weight", type=float, default=0.5)
    parser.add_argument(
        "--metric_max_points",
        type=int,
        default=20000,
        help="Maximum sampled points per cloud for Sim(3)-aligned Chamfer metrics.",
    )
    parser.add_argument(
        "--skip_matching_eval",
        action="store_true",
        help="Skip the ALIKED-query two-view matching evaluation for the tracking head.",
    )
    parser.add_argument("--matching_max_keypoints", type=int, default=5000, help="ALIKED keypoints per query image.")
    parser.add_argument("--matching_det_thresh", type=float, default=0.005, help="ALIKED detection threshold.")
    parser.add_argument(
        "--matching_ransac_thresh",
        type=float,
        default=1e-3,
        help="RANSAC threshold in normalized camera coordinates for Essential matrix fitting.",
    )
    parser.add_argument("--matching_min_inliers", type=int, default=15, help="Minimum recoverPose inliers per pair.")
    parser.add_argument("--matching_min_vis", type=float, default=0.5, help="Minimum VGGT visibility score.")
    parser.add_argument("--matching_min_conf", type=float, default=0.0, help="Minimum VGGT tracking confidence score.")
    parser.add_argument(
        "--matching_max_pairs",
        type=int,
        default=0,
        help="Maximum frame pairs for matching evaluation; 0 evaluates all pairs.",
    )
    parser.add_argument("--save_adv_images", action="store_true", help="Save adversarial input frames.")
    parser.add_argument(
        "--eval_tasks",
        default="camera",
        help=(
            "Paper task to run. Use exactly one of camera,depth,point,tracking; "
            "the corresponding dataset is enforced by --eval_dataset."
        ),
    )
    parser.add_argument(
        "--attack_only",
        action="store_true",
        help="Run the attack and save outputs, but skip clean/attack evaluation metrics.",
    )
    parser.add_argument(
        "--eval_saved_npz",
        default=None,
        help="Skip the attack and evaluate a saved pgd_vggt_outputs.npz from a previous run.",
    )
    parser.add_argument(
        "--run_clean_forward",
        action="store_true",
        help="Ignore --clean_npz and run a clean VGGT forward as the reference.",
    )
    parser.add_argument(
        "--eval_only_clean",
        action="store_true",
        help="Skip the attack and adversarial forward; only evaluate clean predictions against GT. "
             "Useful for sanity-checking the evaluation pipeline.",
    )
    return parser.parse_args()


def set_tracking_eval_status(records: dict, metrics: dict, status: str) -> None:
    records["tracking_image_matching_pair_errors"] = {
        "rotation_deg": [],
        "translation_deg": [],
        "status": status,
    }
    metrics["tracking_image_matching"] = status


def evaluate_prediction_pair(
    *,
    model: torch.nn.Module | None,
    image_paths: list[str],
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    clean_preds: dict[str, torch.Tensor],
    adv_preds: dict[str, torch.Tensor],
    gt_refs: dict[str, torch.Tensor],
    image_size_hw: tuple[int, int],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    eval_tasks: set[str],
) -> tuple[dict, dict]:
    clean_gt_metrics = vggt_paper_metrics(
        gt_refs,
        clean_preds,
        image_size_hw,
        max_points=args.metric_max_points,
        device=device,
        eval_tasks=eval_tasks,
    )
    adv_gt_metrics = vggt_paper_metrics(
        gt_refs,
        adv_preds,
        image_size_hw,
        max_points=args.metric_max_points,
        device=device,
        eval_tasks=eval_tasks,
    )
    clean_gt_records = vggt_paper_records(
        gt_refs,
        clean_preds,
        image_size_hw,
        max_points=args.metric_max_points,
        device=device,
        eval_tasks=eval_tasks,
    )
    adv_gt_records = vggt_paper_records(
        gt_refs,
        adv_preds,
        image_size_hw,
        max_points=args.metric_max_points,
        device=device,
        eval_tasks=eval_tasks,
    )
    clean_adv_metrics = compare_predictions(clean_preds, adv_preds, clean_images, adv_images)
    clean_adv_records = vggt_paper_records(
        clean_preds,
        adv_preds,
        image_size_hw,
        max_points=args.metric_max_points,
        device=device,
        eval_tasks=eval_tasks,
    )

    if "tracking" in eval_tasks:
        pairs_override = read_matching_pairs(Path(args.pairs_file) if args.pairs_file else None, image_paths)
        if args.skip_matching_eval:
            set_tracking_eval_status(clean_gt_records, clean_gt_metrics, "skipped_by_user")
            set_tracking_eval_status(adv_gt_records, adv_gt_metrics, "skipped_by_user")
        elif model is None:
            set_tracking_eval_status(clean_gt_records, clean_gt_metrics, "skipped_missing_model")
            set_tracking_eval_status(adv_gt_records, adv_gt_metrics, "skipped_missing_model")
        else:
            print("\n[tracking/image matching: clean vs real GT]")
            clean_matching_records = tracking_image_matching_records(
                model=model,
                images=clean_images,
                reference=gt_refs,
                dtype=dtype,
                max_keypoints=args.matching_max_keypoints,
                detection_threshold=args.matching_det_thresh,
                ransac_thresh=args.matching_ransac_thresh,
                min_inliers=args.matching_min_inliers,
                min_visibility=args.matching_min_vis,
                min_confidence=args.matching_min_conf,
                max_pairs=args.matching_max_pairs,
                pairs=pairs_override,
            )
            clean_gt_records["tracking_image_matching_pair_errors"] = clean_matching_records
            add_tracking_matching_metrics(clean_gt_metrics, clean_matching_records)

            print("\n[tracking/image matching: pgd vs real GT]")
            adv_matching_records = tracking_image_matching_records(
                model=model,
                images=adv_images,
                reference=gt_refs,
                dtype=dtype,
                max_keypoints=args.matching_max_keypoints,
                detection_threshold=args.matching_det_thresh,
                ransac_thresh=args.matching_ransac_thresh,
                min_inliers=args.matching_min_inliers,
                min_visibility=args.matching_min_vis,
                min_confidence=args.matching_min_conf,
                max_pairs=args.matching_max_pairs,
                pairs=pairs_override,
            )
            adv_gt_records["tracking_image_matching_pair_errors"] = adv_matching_records
            add_tracking_matching_metrics(adv_gt_metrics, adv_matching_records)

    metrics = {
        "clean_vs_gt": clean_gt_metrics,
        "pgd_vs_gt": adv_gt_metrics,
        "clean_vs_pgd": clean_adv_metrics,
    }
    records = {
        "clean_vs_gt": clean_gt_records,
        "pgd_vs_gt": adv_gt_records,
        "clean_vs_pgd": clean_adv_records,
    }
    return metrics, records


def print_metric_group(title: str, metrics: dict) -> None:
    print(f"\n[{title}]")
    if not metrics:
        print("  skipped")
        return
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def load_saved_scene_predictions(
    scene_dir: Path,
    npz_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[str], torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], tuple[int, int]]:
    scene_seed = derive_scene_seed(args.seed, scene_dir.name)
    image_paths, _ = align_image_paths_to_clean(scene_dir, npz_path, 0, scene_seed)
    clean_images, adv_images = load_saved_attack_images(npz_path, device)
    if clean_images is None or adv_images is None:
        raise ValueError(f"{npz_path} must contain clean_images and adv_images for saved evaluation")
    image_size_hw = tuple(clean_images.shape[-2:])
    clean_preds = load_prefixed_predictions_from_npz(npz_path, "clean_", device, image_size_hw)
    adv_preds = load_prefixed_predictions_from_npz(npz_path, "adv_", device, image_size_hw)
    return image_paths, clean_images, adv_images, clean_preds, adv_preds, image_size_hw


def load_paper_gt_reference(
    args: argparse.Namespace,
    eval_dataset: str,
    scene_dir: Path,
    image_paths: list[str],
    device: torch.device,
    image_size_hw: tuple[int, int],
    needs_attack_geometry: bool,
) -> dict:
    if eval_dataset == "co3d":
        if args.co3d_anno_dir:
            return load_co3d_official_camera_reference(Path(args.co3d_anno_dir), scene_dir, image_paths, device)
        if not args.gt_root:
            raise ValueError("--eval_dataset co3d needs --co3d_anno_dir or --gt_root")
        return load_co3d_gt_reference(
            Path(args.gt_root),
            scene_dir,
            image_paths,
            device,
            image_size_hw,
            load_geometry=needs_attack_geometry,
        )
    if eval_dataset == "dtu":
        scan_id = resolve_dtu_scan_id(args, scene_dir)
        ply_path = resolve_dtu_gt_ply(args, scene_dir)
        refs = load_gt_pointcloud_reference(ply_path, device, alignment="none")
        refs.update(
            {
                "dtu_eval_root": str(Path(args.gt_root)),
                "dtu_scan_id": scan_id,
                "dtu_downsample": args.dtu_downsample,
                "dtu_patch_size": args.dtu_patch_size,
                "dtu_max_dist": args.dtu_max_dist,
            }
        )
        refs.update(
            load_gt_camera_npz_reference(
                Path(args.gt_camera_npz) if args.gt_camera_npz else None,
                device,
                image_size_hw,
                image_paths,
            )
        )
        return refs
    if eval_dataset == "eth3d":
        if not args.gt_camera_npz:
            raise ValueError("--eval_dataset eth3d needs --gt_camera_npz so Sim(3) is estimated from matching cameras")
        ply_path = resolve_eth3d_gt_ply(args, scene_dir)
        refs = load_gt_pointcloud_reference(ply_path, device, alignment="sim3")
        refs.update(
            load_gt_camera_npz_reference(
                Path(args.gt_camera_npz) if args.gt_camera_npz else None,
                device,
                image_size_hw,
                image_paths,
            )
        )
        return refs
    if eval_dataset == "re10k":
        if not args.gt_camera_npz:
            raise ValueError("--eval_dataset re10k needs --gt_camera_npz converted from the official RE10K test cameras")
        return load_gt_camera_npz_reference(Path(args.gt_camera_npz), device, image_size_hw, image_paths)
    if eval_dataset == "scannet1500":
        if not args.gt_camera_npz:
            raise ValueError("--eval_dataset scannet1500 needs --gt_camera_npz with extrinsic/intrinsic/image_paths")
        if not args.pairs_file:
            raise ValueError("--eval_dataset scannet1500 needs --pairs_file with the official pair list")
        return load_gt_camera_npz_reference(Path(args.gt_camera_npz), device, image_size_hw, image_paths)
    raise ValueError(f"Unsupported --eval_dataset {eval_dataset}")


def process_scene(
    model: torch.nn.Module | None,
    scene_dir: Path,
    out_dir: Path,
    clean_npz: Path | None,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_tasks = parse_eval_tasks(args.eval_tasks)
    eval_dataset = infer_paper_eval_dataset(eval_tasks, args.eval_dataset)
    validate_paper_eval_dataset(eval_tasks, eval_dataset)
    scene_seed = derive_scene_seed(args.seed, scene_dir.name)

    patch_tensor = None
    patch_meta = None
    history: list[dict[str, float]] = []
    weights = {
        "depth": args.depth_weight,
        "pose": args.pose_weight,
        "points": args.points_weight,
    }

    if args.eval_saved_npz:
        frame_sampling_method = "saved_npz_image_paths"
        saved_npz = Path(args.eval_saved_npz)
        image_paths, clean_images, adv_images, clean_preds, adv_preds, image_size_hw = load_saved_scene_predictions(
            scene_dir,
            saved_npz,
            args,
            device,
        )
        frame_indices = np.arange(len(image_paths), dtype=int)
        gt_refs = load_paper_gt_reference(
            args,
            eval_dataset,
            scene_dir,
            image_paths,
            device,
            image_size_hw,
            needs_attack_geometry=False,
        )
        print(f"[eval_saved_npz] loaded predictions from {saved_npz}")
        print(f"[gt] loaded {eval_dataset} paper GT")
    else:
        frame_sampling_method = "random_without_replacement"
        if model is None:
            raise ValueError("A model is required unless --eval_saved_npz is used without tracking eval.")

        image_paths, frame_indices = align_image_paths_to_clean(scene_dir, clean_npz, args.max_frames, scene_seed)
        if not image_paths:
            raise ValueError(f"No images found under {scene_dir}")

        clean_images = load_and_preprocess_images(image_paths).to(device)
        image_size_hw = tuple(clean_images.shape[-2:])
        gt_refs = load_paper_gt_reference(
            args,
            eval_dataset,
            scene_dir,
            image_paths,
            device,
            image_size_hw,
            needs_attack_geometry=(not args.eval_only_clean) or bool(eval_tasks & {"depth", "point"}),
        )
        print(f"[gt] loaded {eval_dataset} paper GT")

        if clean_npz is not None and not args.run_clean_forward:
            clean_preds = load_clean_reference(clean_npz, device, image_size_hw, frame_indices=frame_indices)
            print(f"[clean] loaded reference outputs from {clean_npz}")
        else:
            t0 = time.time()
            with torch.no_grad():
                clean_preds_full = forward_vggt(model, clean_images, dtype)
            clean_preds = detach_predictions(clean_preds_full)
            print(f"[clean] forward reference done in {time.time() - t0:.2f}s")

        if args.eval_only_clean:
            adv_images = clean_images.detach().clone()
            adv_preds = {k: v.detach().clone() for k, v in clean_preds.items()}
            print("[eval_only_clean] skipping attack; adv_* = clean_*")
        elif args.attack_type == "patch":
            attack_reference = gt_refs if has_attack_reference_terms(gt_refs) else clean_preds
            adv_images, history, patch_tensor, patch_meta = patch_attack(
                model=model,
                images=clean_images,
                reference_preds=attack_reference,
                dtype=dtype,
                steps=args.steps,
                alpha=args.patch_alpha if args.patch_alpha is not None else args.alpha,
                patch_size=args.patch_size,
                patch_x=args.patch_x,
                patch_y=args.patch_y,
                weights=weights,
            )
        else:
            attack_reference = gt_refs if has_attack_reference_terms(gt_refs) else clean_preds
            adv_images, history = pgd_attack(
                model=model,
                images=clean_images,
                clean_preds=attack_reference,
                dtype=dtype,
                steps=args.steps,
                eps=args.eps,
                alpha=args.alpha,
                random_start=not args.no_random_start,
                weights=weights,
            )

        if not args.eval_only_clean:
            track_query = clean_preds["track"][:, 0] if "track" in clean_preds else None
            with torch.no_grad():
                adv_preds_full = forward_vggt(model, adv_images, dtype, query_points=track_query)
            adv_preds = detach_predictions(adv_preds_full)

        clean_np = tensor_to_numpy(clean_preds, image_size_hw)
        adv_np = tensor_to_numpy(adv_preds, image_size_hw)
        np.savez_compressed(
            out_dir / "pgd_vggt_outputs.npz",
            image_paths=np.array([Path(p).name for p in image_paths]),
            clean_images=clean_images.detach().cpu().numpy().astype(np.float16),
            adv_images=adv_images.detach().cpu().numpy().astype(np.float16),
            **{f"clean_{k}": v for k, v in clean_np.items()},
            **{f"adv_{k}": v for k, v in adv_np.items()},
        )

    if args.attack_only and not args.eval_saved_npz:
        metrics = {"clean_vs_gt": {}, "pgd_vs_gt": {}, "clean_vs_pgd": {}}
        eval_records = {"clean_vs_gt": {}, "pgd_vs_gt": {}, "clean_vs_pgd": {}}
        print("\n[attack_only] saved adversarial outputs; skipped evaluation")
    else:
        metrics, eval_records = evaluate_prediction_pair(
            model=model,
            image_paths=image_paths,
            clean_images=clean_images,
            adv_images=adv_images,
            clean_preds=clean_preds,
            adv_preds=adv_preds,
            gt_refs=gt_refs,
            image_size_hw=image_size_hw,
            args=args,
            device=device,
            dtype=dtype,
            eval_tasks=eval_tasks,
        )
        print_metric_group("clean vs real GT", metrics["clean_vs_gt"])
        print_metric_group("pgd vs real GT", metrics["pgd_vs_gt"])
        print_metric_group("clean vs pgd", metrics["clean_vs_pgd"])

    if "__" in scene_dir.name:
        scene_category, scene_sequence = scene_dir.name.split("__", 1)
    else:
        scene_category, scene_sequence = scene_dir.name, ""

    summary = {
        "scene": str(scene_dir),
        "category": scene_category,
        "sequence_name": scene_sequence,
        "ckpt": args.ckpt,
        "n_frames": len(image_paths),
        "steps": args.steps,
        "attack_type": args.attack_type,
        "eps": args.eps,
        "alpha": args.alpha,
        "patch": patch_meta,
        "random_start": not args.no_random_start,
        "weights": weights,
        "mode": "eval_saved_npz" if args.eval_saved_npz else ("attack_only" if args.attack_only else "attack_and_eval"),
        "eval_tasks": sorted(eval_tasks),
        "eval_dataset": eval_dataset,
        "frame_sampling": {
            "method": frame_sampling_method,
            "seed": int(args.seed),
            "scene_seed": int(scene_seed),
            "frame_indices": frame_indices.astype(int).tolist(),
        },
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "metrics": metrics,
        "eval_records": eval_records,
        "history": history,
        "image_paths": [str(p) for p in image_paths],
    }
    with open(out_dir / "pgd_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.eval_saved_npz:
        save_delta_preview(clean_images, adv_images, out_dir)
        save_patch_image(patch_tensor, out_dir)
        if args.save_adv_images:
            save_adv_images(adv_images, image_paths, out_dir)

    print(f"\n[done] saved -> {out_dir}")
    return summary


def mean_scene_metric(summaries: list[dict], eval_key: str, record_name: str, metric_name: str) -> float | None:
    values = []
    for summary in summaries:
        records = summary.get("eval_records", {}).get(eval_key, {})
        value = records.get(record_name, {}).get(metric_name)
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))


def scene_category_from_summary(summary: dict) -> str:
    category = summary.get("category")
    if isinstance(category, str) and category:
        return category

    scene_name = Path(summary.get("scene", "")).name
    if "__" in scene_name:
        return scene_name.split("__", 1)[0]
    return scene_name or "unknown"


def aggregate_camera_by_category(
    summaries: list[dict],
    eval_key: str,
    thresholds: tuple[int, ...] = (30,),
) -> dict:
    """Official CO3D-style dataset aggregation: metric per category, then category mean."""
    records_by_category: dict[str, dict[str, list[float]]] = {}
    pooled_records = {"rotation_deg": [], "translation_deg": []}

    for summary in summaries:
        records = summary.get("eval_records", {}).get(eval_key, {})
        camera_records = records.get("camera_pair_errors", {})
        r_errors = camera_records.get("rotation_deg", [])
        t_errors = camera_records.get("translation_deg", [])
        if not r_errors or not t_errors:
            continue

        category = scene_category_from_summary(summary)
        category_records = records_by_category.setdefault(category, {"rotation_deg": [], "translation_deg": []})
        category_records["rotation_deg"].extend(r_errors)
        category_records["translation_deg"].extend(t_errors)
        pooled_records["rotation_deg"].extend(r_errors)
        pooled_records["translation_deg"].extend(t_errors)

    if not records_by_category:
        return {}

    category_metrics = {
        category: camera_metrics_from_pair_errors(records, thresholds=thresholds)
        for category, records in sorted(records_by_category.items())
    }

    metrics: dict[str, float | str] = {
        "camera_aggregation": "category_mean",
        "camera_category_count": float(len(category_metrics)),
        "camera_pair_count": float(len(pooled_records["rotation_deg"])),
    }
    for threshold in thresholds:
        for name in ("rra", "rta", "auc"):
            key = f"camera_{name}@{threshold}"
            values = [cat_metrics[key] for cat_metrics in category_metrics.values() if key in cat_metrics]
            if values:
                metrics[key] = float(np.mean(values))

    pooled_metrics = camera_metrics_from_pair_errors(pooled_records, thresholds=thresholds)
    for threshold in thresholds:
        for name in ("rra", "rta", "auc"):
            key = f"camera_{name}@{threshold}"
            if key in pooled_metrics:
                metrics[f"{key}_pooled"] = pooled_metrics[key]

    return metrics


def aggregate_eval_key(summaries: list[dict], eval_key: str) -> dict:
    matching_r_errors: list[float] = []
    matching_t_errors: list[float] = []

    for summary in summaries:
        records = summary.get("eval_records", {}).get(eval_key, {})
        matching_records = records.get("tracking_image_matching_pair_errors", {})
        matching_r_errors.extend(matching_records.get("rotation_deg", []))
        matching_t_errors.extend(matching_records.get("translation_deg", []))

    metrics: dict[str, dict] = {
        "camera": {},
        "depth": {},
        "point": {},
        "tracking_image_matching": {},
    }

    metrics["camera"] = aggregate_camera_by_category(summaries, eval_key, thresholds=(30,))

    for align in ("none", "sim3"):
        for name in ("accuracy", "completeness", "overall"):
            metric_key = f"depth_align_{align}_{name}"
            value = mean_scene_metric(summaries, eval_key, "depth_scene_metrics", metric_key)
            if value is not None:
                metrics["depth"][f"align_{align}_{name}"] = value

    for name in ("accuracy", "completeness", "overall"):
        value = mean_scene_metric(summaries, eval_key, "point_scene_metrics", f"point_{name}")
        if value is not None:
            metrics["point"][name] = value

    if matching_r_errors and matching_t_errors:
        metrics["tracking_image_matching"] = pose_metrics_from_pair_errors(
            {"rotation_deg": matching_r_errors, "translation_deg": matching_t_errors},
            thresholds=(5, 10, 20),
            prefix="tracking",
        )
    else:
        metrics["tracking_image_matching"] = {
            "status": "not_computed_or_no_valid_pairs",
        }

    return metrics


def aggregate_dataset_metrics(summaries: list[dict]) -> dict:
    return {
        "protocol": EVALUATION_PROTOCOL,
        "clean_vs_gt": aggregate_eval_key(summaries, "clean_vs_gt"),
        "pgd_vs_gt": aggregate_eval_key(summaries, "pgd_vs_gt"),
        "clean_vs_pgd": aggregate_eval_key(summaries, "clean_vs_pgd"),
    }


def main() -> None:
    args = parse_args()
    eval_tasks = parse_eval_tasks(args.eval_tasks)
    eval_dataset = infer_paper_eval_dataset(eval_tasks, args.eval_dataset)
    validate_paper_eval_dataset(eval_tasks, eval_dataset)
    set_random_seeds(args.seed)
    if args.scene_dir is None and args.scenes_root is None:
        raise ValueError("Provide either --scene_dir for one scene or --scenes_root for batch mode.")
    if args.scene_dir is not None and args.scenes_root is not None:
        raise ValueError("Use either --scene_dir or --scenes_root, not both.")
    if args.eval_saved_npz and args.scenes_root is not None:
        raise ValueError("--eval_saved_npz is single-scene mode; use --scene_dir and run it once per saved attack output.")
    if args.eval_saved_npz and args.attack_only:
        raise ValueError("--attack_only and --eval_saved_npz are mutually exclusive.")
    if args.attack_only and args.eval_only_clean:
        raise ValueError("--attack_only and --eval_only_clean are mutually exclusive.")
    if (
        args.scenes_root is not None
        and args.clean_output_root is None
        and not args.run_clean_forward
        and not args.eval_saved_npz
    ):
        raise ValueError("Batch mode needs --clean_output_root unless --run_clean_forward is set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"[cfg] device={device} dtype={dtype}")
    print(f"[eval] tasks={','.join(sorted(eval_tasks))} dataset={eval_dataset}")

    needs_model = not args.eval_saved_npz or ("tracking" in eval_tasks and not args.skip_matching_eval)
    model = None
    if needs_model:
        print(f"[model] loading {args.ckpt}")
        model = load_model(args, device)
        for param in model.parameters():
            param.requires_grad_(False)

    if args.scene_dir is not None:
        scene_dir = Path(args.scene_dir)
        clean_npz = Path(args.clean_npz) if args.clean_npz else None
        output_dir = Path(args.output_dir)
        summary = process_scene(model, scene_dir, output_dir, clean_npz, args, device, dtype)
        dataset_metrics = aggregate_dataset_metrics([summary])
        with open(output_dir / "pgd_dataset_metrics.json", "w", encoding="utf-8") as f:
            json.dump(dataset_metrics, f, indent=2)
        write_attack_ranking([summary], output_dir)
        return

    output_root = Path(args.output_dir)
    clean_root = Path(args.clean_output_root) if args.clean_output_root else None
    scene_dirs = sorted(d for d in Path(args.scenes_root).glob(args.scene_pattern) if d.is_dir())
    summaries = []
    for scene_dir in scene_dirs:
        clean_npz = None
        if clean_root is not None:
            clean_npz = clean_root / scene_dir.name / "vggt_outputs.npz"
            if not clean_npz.exists():
                print(f"[skip] {scene_dir.name}: missing clean output {clean_npz}")
                continue
        print(f"\n[scene] {scene_dir.name}")
        try:
            summary = process_scene(
                model,
                scene_dir,
                output_root / scene_dir.name,
                clean_npz,
                args,
                device,
                dtype,
            )
            summaries.append(summary)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            print(f"[error] {scene_dir.name}: CUDA OOM: {exc}")
        except Exception as exc:
            torch.cuda.empty_cache()
            print(f"[error] {scene_dir.name}: {exc}")

    with open(output_root / "pgd_batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    dataset_metrics = aggregate_dataset_metrics(summaries)
    with open(output_root / "pgd_dataset_metrics.json", "w", encoding="utf-8") as f:
        json.dump(dataset_metrics, f, indent=2)
    ranking = write_attack_ranking(summaries, output_root)
    print("\n[dataset metrics: VGGT paper-aligned metrics]")
    for group, values in dataset_metrics.items():
        if group == "protocol":
            continue
        print(f"  {group}: {values}")
    if ranking:
        print("\n[attack ranking: largest clean-to-PGD output change]")
        for row in ranking[:10]:
            print(
                f"  #{row['rank']} {row['scene']}: "
                f"score={float(row['attack_score']):.4f} "
                f"depth_rmse={row.get('clean_adv_depth_rel_rmse')} "
                f"point_rmse={row.get('clean_adv_points_rel_rmse')}"
            )
    print(f"\n[batch done] {len(summaries)}/{len(scene_dirs)} scenes attacked")


if __name__ == "__main__":
    main()
