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

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri


CO3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
EVALUATION_PROTOCOL = {
    "reference": "real_co3d_gt",
    "attack_loss": "maximize prediction error against real CO3D GT",
    "camera": (
        "Paper Table 1 (CO3D/RE10K) protocol: AUC over pairwise relative pose errors, "
        "implemented as AUC over max(rotation_error, translation_error)."
    ),
    "depth": "Reports both depth_align_none_* and depth_align_sim3_* for the depth+camera unprojected cloud.",
    "depth_align_none": (
        "Paper Table 2 (DTU) metric definition: depth+camera unproject + Chamfer, no Sim(3). "
        "When run on CO3D, absolute values are scale-mismatched proxy values; use relative "
        "clean-vs-adversarial changes."
    ),
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
    "dataset_note": "This script loads CO3D GT; use DTU/ETH3D/ScanNet loaders for exact paper dataset splits.",
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
    model: VGGT,
    images: torch.Tensor,
    dtype: torch.dtype,
    query_points: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    with autocast_context(images.device, dtype):
        preds = model(images, query_points=query_points)
    return {k: v for k, v in preds.items() if torch.is_tensor(v)}


def load_model(args: argparse.Namespace, device: torch.device) -> VGGT:
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


def load_co3d_gt_reference(
    gt_root: Path,
    scene_dir: Path,
    image_paths: list[str],
    device: torch.device,
    image_size_hw: tuple[int, int],
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
        extrinsics.append(extrinsic)
        intrinsics.append(intrinsic)
        world_points.append(world)

    extrinsics_t = torch.from_numpy(np.stack(extrinsics).astype(np.float32)).to(device).unsqueeze(0)
    intrinsics_t = torch.from_numpy(np.stack(intrinsics).astype(np.float32)).to(device).unsqueeze(0)
    refs = {
        "pose_enc": extri_intri_to_pose_encoding(extrinsics_t, intrinsics_t, image_size_hw),
        "depth": torch.from_numpy(np.stack(depths).astype(np.float32)).to(device).unsqueeze(0)[..., None],
        "world_points": torch.from_numpy(np.stack(world_points).astype(np.float32)).to(device).unsqueeze(0),
        "point_mask": torch.from_numpy(np.stack(masks).astype(bool)).to(device).unsqueeze(0),
        "extrinsic": extrinsics_t,
        "intrinsic": intrinsics_t,
    }
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


def pgd_attack(
    model: VGGT,
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
    model: VGGT,
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
    if "pose_enc" not in clean or "pose_enc" not in adv:
        return {"rotation_deg": [], "translation_deg": []}

    clean_extri, _ = pose_encoding_to_extri_intri(clean["pose_enc"], image_size_hw)
    adv_extri, _ = pose_encoding_to_extri_intri(adv["pose_enc"], image_size_hw)
    clean_extri = clean_extri.detach().float()
    adv_extri = adv_extri.detach().float()

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


def vggt_paper_metrics(
    reference: dict[str, torch.Tensor],
    pred: dict[str, torch.Tensor],
    image_size_hw: tuple[int, int],
    max_points: int,
    device: torch.device,
) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    metrics.update(camera_paper_metrics(reference, pred, image_size_hw))

    reference_np = tensor_to_numpy(reference, image_size_hw)
    pred_np = tensor_to_numpy(pred, image_size_hw)
    reference_mask = None
    if "point_mask" in reference:
        reference_mask = reference["point_mask"].detach().cpu().numpy()[0].astype(bool)

    if all(k in reference_np for k in ("depth", "extrinsic", "intrinsic")) and all(
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

    if "world_points" in reference_np and "world_points" in pred_np:
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

    metrics["tracking_image_matching"] = "not_computed"
    return metrics


def vggt_paper_records(
    reference: dict[str, torch.Tensor],
    pred: dict[str, torch.Tensor],
    image_size_hw: tuple[int, int],
    max_points: int,
    device: torch.device,
) -> dict:
    metrics = vggt_paper_metrics(reference, pred, image_size_hw, max_points, device)
    depth_keys = [
        f"depth_align_{align}_{name}"
        for align in ("none", "sim3")
        for name in ("accuracy", "completeness", "overall")
    ]
    return {
        "camera_pair_errors": camera_pair_error_records(reference, pred, image_size_hw),
        "depth_scene_metrics": {
            key: metrics[key]
            for key in depth_keys
            if key in metrics
        },
        "point_scene_metrics": {
            key: metrics[key]
            for key in ("point_accuracy", "point_completeness", "point_overall")
            if key in metrics
        },
        "tracking_image_matching_pair_errors": {
            "rotation_deg": [],
            "translation_deg": [],
            "status": "not_computed",
        },
    }


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


def tracking_image_matching_records(
    model: VGGT,
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
) -> dict[str, list[float] | str | int]:
    """VGGT paper-style image matching: ALIKED queries -> VGGT tracks -> Essential matrix."""
    if "extrinsic" not in reference or "intrinsic" not in reference:
        return {"rotation_deg": [], "translation_deg": [], "status": "skipped_missing_gt_cameras"}

    num_frames = images.shape[0]
    pairs = build_pair_indices(num_frames, max_pairs)
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
        required=True,
        help="CO3D raw GT root containing category/frame_annotations.jgz plus images/depths/depth_masks.",
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


def process_scene(
    model: VGGT,
    scene_dir: Path,
    out_dir: Path,
    clean_npz: Path | None,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_seed = derive_scene_seed(args.seed, scene_dir.name)
    image_paths, frame_indices = align_image_paths_to_clean(scene_dir, clean_npz, args.max_frames, scene_seed)
    if not image_paths:
        raise ValueError(f"No images found under {scene_dir}")

    clean_images = load_and_preprocess_images(image_paths).to(device)
    image_size_hw = tuple(clean_images.shape[-2:])
    gt_refs = load_co3d_gt_reference(Path(args.gt_root), scene_dir, image_paths, device, image_size_hw)
    print(f"[gt] loaded CO3D real GT from {args.gt_root}")

    if clean_npz is not None and not args.run_clean_forward:
        clean_preds = load_clean_reference(clean_npz, device, image_size_hw, frame_indices=frame_indices)
        print(f"[clean] loaded reference outputs from {clean_npz}")
    else:
        t0 = time.time()
        with torch.no_grad():
            clean_preds_full = forward_vggt(model, clean_images, dtype)
        clean_preds = detach_predictions(clean_preds_full)
        print(f"[clean] forward reference done in {time.time() - t0:.2f}s")

    weights = {
        "depth": args.depth_weight,
        "pose": args.pose_weight,
        "points": args.points_weight,
    }
    patch_tensor = None
    patch_meta = None
    history: list[dict[str, float]] = []
    if args.eval_only_clean:
        # Skip the attack entirely; reuse clean images/predictions in place of adv ones
        # so that downstream eval/save logic stays identical.
        adv_images = clean_images.detach().clone()
        adv_preds = {k: v.detach().clone() for k, v in clean_preds.items()}
        print("[eval_only_clean] skipping attack; adv_* = clean_*")
    elif args.attack_type == "patch":
        adv_images, history, patch_tensor, patch_meta = patch_attack(
            model=model,
            images=clean_images,
            reference_preds=gt_refs,
            dtype=dtype,
            steps=args.steps,
            alpha=args.patch_alpha if args.patch_alpha is not None else args.alpha,
            patch_size=args.patch_size,
            patch_x=args.patch_x,
            patch_y=args.patch_y,
            weights=weights,
        )
    else:
        adv_images, history = pgd_attack(
            model=model,
            images=clean_images,
            clean_preds=gt_refs,
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
    clean_gt_metrics = vggt_paper_metrics(
        gt_refs,
        clean_preds,
        image_size_hw,
        max_points=args.metric_max_points,
        device=device,
    )
    adv_gt_metrics = vggt_paper_metrics(
        gt_refs,
        adv_preds,
        image_size_hw,
        max_points=args.metric_max_points,
        device=device,
    )
    clean_gt_records = vggt_paper_records(
        gt_refs,
        clean_preds,
        image_size_hw,
        max_points=args.metric_max_points,
        device=device,
    )
    adv_gt_records = vggt_paper_records(
        gt_refs,
        adv_preds,
        image_size_hw,
        max_points=args.metric_max_points,
        device=device,
    )
    clean_adv_metrics = compare_predictions(clean_preds, adv_preds, clean_images, adv_images)
    clean_adv_records = vggt_paper_records(
        clean_preds,
        adv_preds,
        image_size_hw,
        max_points=args.metric_max_points,
        device=device,
    )

    if args.skip_matching_eval:
        clean_gt_records["tracking_image_matching_pair_errors"]["status"] = "skipped_by_user"
        adv_gt_records["tracking_image_matching_pair_errors"]["status"] = "skipped_by_user"
        clean_gt_metrics["tracking_image_matching"] = "skipped_by_user"
        adv_gt_metrics["tracking_image_matching"] = "skipped_by_user"
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
        )
        adv_gt_records["tracking_image_matching_pair_errors"] = adv_matching_records
        add_tracking_matching_metrics(adv_gt_metrics, adv_matching_records)

    print("\n[clean vs real GT]")
    for key, value in clean_gt_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n[pgd vs real GT]")
    for key, value in adv_gt_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n[clean vs pgd]")
    for key, value in clean_adv_metrics.items():
        print(f"  {key}: {value:.6f}")

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

    summary = {
        "scene": str(scene_dir),
        "ckpt": args.ckpt,
        "n_frames": len(image_paths),
        "steps": args.steps,
        "attack_type": args.attack_type,
        "eps": args.eps,
        "alpha": args.alpha,
        "patch": patch_meta,
        "random_start": not args.no_random_start,
        "weights": weights,
        "frame_sampling": {
            "method": "random_without_replacement",
            "seed": int(args.seed),
            "scene_seed": int(scene_seed),
            "frame_indices": frame_indices.astype(int).tolist(),
        },
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "metrics": {
            "clean_vs_gt": clean_gt_metrics,
            "pgd_vs_gt": adv_gt_metrics,
            "clean_vs_pgd": clean_adv_metrics,
        },
        "eval_records": {
            "clean_vs_gt": clean_gt_records,
            "pgd_vs_gt": adv_gt_records,
            "clean_vs_pgd": clean_adv_records,
        },
        "history": history,
        "image_paths": [str(p) for p in image_paths],
    }
    with open(out_dir / "pgd_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

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


def aggregate_eval_key(summaries: list[dict], eval_key: str) -> dict:
    camera_r_errors: list[float] = []
    camera_t_errors: list[float] = []
    matching_r_errors: list[float] = []
    matching_t_errors: list[float] = []

    for summary in summaries:
        records = summary.get("eval_records", {}).get(eval_key, {})
        camera_records = records.get("camera_pair_errors", {})
        camera_r_errors.extend(camera_records.get("rotation_deg", []))
        camera_t_errors.extend(camera_records.get("translation_deg", []))

        matching_records = records.get("tracking_image_matching_pair_errors", {})
        matching_r_errors.extend(matching_records.get("rotation_deg", []))
        matching_t_errors.extend(matching_records.get("translation_deg", []))

    metrics: dict[str, dict] = {
        "camera": {},
        "depth": {},
        "point": {},
        "tracking_image_matching": {},
    }

    metrics["camera"] = camera_metrics_from_pair_errors(
        {"rotation_deg": camera_r_errors, "translation_deg": camera_t_errors},
        thresholds=(30,),
    )

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
    set_random_seeds(args.seed)
    if args.scene_dir is None and args.scenes_root is None:
        raise ValueError("Provide either --scene_dir for one scene or --scenes_root for batch mode.")
    if args.scene_dir is not None and args.scenes_root is not None:
        raise ValueError("Use either --scene_dir or --scenes_root, not both.")
    if args.scenes_root is not None and args.clean_output_root is None and not args.run_clean_forward:
        raise ValueError("Batch mode needs --clean_output_root unless --run_clean_forward is set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"[cfg] device={device} dtype={dtype}")
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
