"""
PGD attack baseline for VGGT.

This script performs an untargeted PGD attack in pixel space. It can either use
existing clean VGGT outputs as the fixed reference, or run one clean forward
when no clean output file is provided.

Example:
    python attack_vggt_pgd.py ^
        --scene_dir examples/kitchen ^
        --clean_npz clean_outputs/kitchen/vggt_outputs.npz ^
        --output_dir outputs_pgd/kitchen ^
        --max_frames 4 ^
        --steps 10 ^
        --eps 0.03137255 ^
        --alpha 0.00392157
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri


def find_images(scene_dir: Path) -> list[str]:
    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        image_dir = scene_dir

    paths: list[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(glob.glob(str(image_dir / ext)))
    return sorted(paths)


def subsample(paths: list[str], max_frames: int) -> list[str]:
    if max_frames <= 0 or len(paths) <= max_frames:
        return paths
    return [paths[i] for i in subsample_indices(len(paths), max_frames)]


def subsample_indices(length: int, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or length <= max_frames:
        return np.arange(length)
    return np.linspace(0, length - 1, max_frames, dtype=int)


def autocast_context(device: torch.device, dtype: torch.dtype):
    enabled = device.type == "cuda"
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def forward_vggt(model: VGGT, images: torch.Tensor, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    with autocast_context(images.device, dtype):
        preds = model(images)
    return {k: v for k, v in preds.items() if torch.is_tensor(v)}


def detach_predictions(preds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = ("pose_enc", "depth", "depth_conf", "world_points", "world_points_conf")
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

    if "depth" in data:
        refs["depth"] = torch.from_numpy(select_frames(data["depth"]).astype(np.float32)).to(device).unsqueeze(0)
    if "point_map" in data:
        refs["world_points"] = torch.from_numpy(select_frames(data["point_map"]).astype(np.float32)).to(device).unsqueeze(0)
    if "extrinsic" in data and "intrinsic" in data:
        extrinsic = torch.from_numpy(select_frames(data["extrinsic"]).astype(np.float32)).to(device).unsqueeze(0)
        intrinsic = torch.from_numpy(select_frames(data["intrinsic"]).astype(np.float32)).to(device).unsqueeze(0)
        refs["pose_enc"] = extri_intri_to_pose_encoding(extrinsic, intrinsic, image_size_hw)

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


def align_image_paths_to_clean(scene_dir: Path, clean_npz: Path | None, max_frames: int) -> tuple[list[str], np.ndarray | None]:
    clean_names = read_clean_image_names(clean_npz)
    if clean_names is None:
        paths = subsample(find_images(scene_dir), max_frames)
        return paths, None

    by_name = {Path(path).name: path for path in find_images(scene_dir)}
    missing = [name for name in clean_names if Path(name).name not in by_name]
    if missing:
        raise ValueError(f"{len(missing)} clean frames are missing from {scene_dir}; first missing: {missing[0]}")

    frame_indices = subsample_indices(len(clean_names), max_frames)
    paths = [by_name[Path(clean_names[i]).name] for i in frame_indices]
    return paths, frame_indices


def normalized_mse(adv: torch.Tensor, clean: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    scale = clean.detach().abs().mean().clamp_min(eps)
    return ((adv - clean.detach()) / scale).pow(2).mean()


def attack_loss(
    adv: dict[str, torch.Tensor],
    clean: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    terms: dict[str, torch.Tensor] = {}

    if "depth" in adv and "depth" in clean:
        terms["depth"] = normalized_mse(adv["depth"], clean["depth"])
    if "pose_enc" in adv and "pose_enc" in clean:
        terms["pose"] = normalized_mse(adv["pose_enc"], clean["pose_enc"])
    if "world_points" in adv and "world_points" in clean:
        terms["points"] = normalized_mse(adv["world_points"], clean["world_points"])

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


def tensor_to_numpy(preds: dict[str, torch.Tensor], image_size_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "pose_enc" in preds:
        extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], image_size_hw)
        out["extrinsic"] = extrinsic.detach().cpu().numpy().astype(np.float32)
        out["intrinsic"] = intrinsic.detach().cpu().numpy().astype(np.float32)
        out["pose_enc"] = preds["pose_enc"].detach().cpu().numpy().astype(np.float32)
    for key in ("depth", "depth_conf", "world_points", "world_points_conf"):
        if key in preds:
            out[key] = preds[key].detach().cpu().numpy().astype(np.float32)
    return out


def relative_rmse(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    denom = b.detach().abs().mean().clamp_min(eps)
    return float((((a - b.detach()) / denom).pow(2).mean().sqrt()).detach().cpu())


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
    parser.add_argument("--ckpt", default="facebook/VGGT-1B", help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--max_frames", type=int, default=4, help="Maximum number of frames to attack; 0 keeps all frames.")
    parser.add_argument("--steps", type=int, default=10, help="PGD iterations.")
    parser.add_argument("--eps", type=float, default=8 / 255, help="L-infinity perturbation budget in [0, 1] pixels.")
    parser.add_argument("--alpha", type=float, default=1 / 255, help="PGD step size in [0, 1] pixels.")
    parser.add_argument("--no_random_start", action="store_true", help="Start PGD from the clean images.")
    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--pose_weight", type=float, default=0.2)
    parser.add_argument("--points_weight", type=float, default=0.5)
    parser.add_argument("--save_adv_images", action="store_true", help="Save adversarial input frames.")
    parser.add_argument(
        "--run_clean_forward",
        action="store_true",
        help="Ignore --clean_npz and run a clean VGGT forward as the reference.",
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

    image_paths, frame_indices = align_image_paths_to_clean(scene_dir, clean_npz, args.max_frames)
    if not image_paths:
        raise ValueError(f"No images found under {scene_dir}")

    clean_images = load_and_preprocess_images(image_paths).to(device)
    image_size_hw = tuple(clean_images.shape[-2:])

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
    adv_images, history = pgd_attack(
        model=model,
        images=clean_images,
        clean_preds=clean_preds,
        dtype=dtype,
        steps=args.steps,
        eps=args.eps,
        alpha=args.alpha,
        random_start=not args.no_random_start,
        weights=weights,
    )

    with torch.no_grad():
        adv_preds_full = forward_vggt(model, adv_images, dtype)
    adv_preds = detach_predictions(adv_preds_full)
    metrics = compare_predictions(clean_preds, adv_preds, clean_images, adv_images)

    print("\n[baseline vs pgd]")
    for key, value in metrics.items():
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
        "eps": args.eps,
        "alpha": args.alpha,
        "random_start": not args.no_random_start,
        "weights": weights,
        "metrics": metrics,
        "history": history,
        "image_paths": [str(p) for p in image_paths],
    }
    with open(out_dir / "pgd_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_delta_preview(clean_images, adv_images, out_dir)
    if args.save_adv_images:
        save_adv_images(adv_images, image_paths, out_dir)

    print(f"\n[done] saved -> {out_dir}")
    return summary


def main() -> None:
    args = parse_args()
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

    model = VGGT.from_pretrained(args.ckpt).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    if args.scene_dir is not None:
        scene_dir = Path(args.scene_dir)
        clean_npz = Path(args.clean_npz) if args.clean_npz else None
        process_scene(model, scene_dir, Path(args.output_dir), clean_npz, args, device, dtype)
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
    print(f"\n[batch done] {len(summaries)}/{len(scene_dirs)} scenes attacked")


if __name__ == "__main__":
    main()
