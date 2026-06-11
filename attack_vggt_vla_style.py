"""VLA-Attacker-style universal visible patch adaptation for VGGT.

This keeps the patch-training mechanism that can transfer to VGGT:

* one visible universal patch shared across scenes
* AdamW optimization with random position, rotation, and shear augmentation
* patch values initialized and constrained to [0, 1]

VLA action-space losses do not exist for VGGT, so the optimization target is
the label-free feature-distance objective used by the local VGGT baseline:

    maximize L1(feature(VGGT, adversarial input), feature(VGGT, clean input))

The trained patch is then applied to every evaluation scene and saved in the
same vggt_outputs.npz layout used by the official VGGT inference bridge.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from attack_vggt_new1 import (
    autocast_context,
    derive_scene_seed,
    detach_predictions,
    extract_features,
    feature_l1_loss,
    find_images,
    forward_vggt,
    load_model,
    resolve_patch_box,
    save_official_style_npz,
    set_random_seeds,
    subsample,
)


DPT_LAYER_INDICES = (4, 11, 17, 23)


def list_scene_dirs(root: str, pattern: str) -> list[Path]:
    scene_dirs = sorted(path for path in Path(root).glob(pattern) if path.is_dir())
    if not scene_dirs:
        raise ValueError(f"No scene folders matched {Path(root) / pattern}")
    return scene_dirs


def load_scene_images(
    scene_dir: Path,
    max_frames: int,
    seed: int,
    device: torch.device,
    frame_indices_override: list[int] | None = None,
) -> tuple[torch.Tensor, list[str], np.ndarray]:
    from vggt.utils.load_fn import load_and_preprocess_images

    all_images = find_images(scene_dir)
    if frame_indices_override is not None:
        frame_indices = np.asarray(frame_indices_override, dtype=int)
        image_paths = [all_images[int(idx)] for idx in frame_indices]
    else:
        scene_seed = derive_scene_seed(seed, scene_dir.name)
        image_paths, frame_indices = subsample(all_images, max_frames, scene_seed)
    if not image_paths:
        raise ValueError(f"No images found under {scene_dir}")
    images = load_and_preprocess_images(image_paths).to(device)
    return images, image_paths, frame_indices


def load_frame_manifest(path: str | None) -> dict[str, list[int]]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    result: dict[str, list[int]] = {}
    for seq, value in raw.items():
        if isinstance(value, dict):
            indices = value.get("frame_indices")
        else:
            indices = value
        if indices is None:
            raise ValueError(f"No frame_indices found for {seq} in {path}")
        result[seq] = [int(idx) for idx in indices]
    return result


def train_max_frames(args: argparse.Namespace) -> int:
    return args.train_max_frames if args.train_max_frames is not None else args.max_frames


def eval_max_frames(args: argparse.Namespace) -> int:
    return args.eval_max_frames if args.eval_max_frames is not None else args.max_frames


def manifest_indices(args: argparse.Namespace, scene_name: str) -> list[int] | None:
    return getattr(args, "frame_manifest_data", {}).get(scene_name)


def resolve_universal_patch_size(images: torch.Tensor, args: argparse.Namespace) -> int:
    height, width = images.shape[-2:]
    if args.patch_size > 0:
        return min(args.patch_size, height, width)
    size = round(math.sqrt(args.patch_area_ratio * height * width))
    return max(1, min(size, height, width))


def sample_geometry(rng: np.random.Generator, args: argparse.Namespace) -> dict[str, float]:
    if args.no_geometry or rng.random() >= args.geometry_prob:
        return {"angle": 0.0, "shear_x": 0.0, "shear_y": 0.0}
    return {
        "angle": float(rng.uniform(-args.rotation_degrees, args.rotation_degrees)),
        "shear_x": float(rng.uniform(-args.shear, args.shear)),
        "shear_y": float(rng.uniform(-args.shear, args.shear)),
    }


def transform_patch(
    patch: torch.Tensor,
    angle: float,
    shear_x: float,
    shear_y: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a differentiable in-plane affine transform to patch and alpha mask."""
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotation = torch.tensor(
        [[cos_a, -sin_a], [sin_a, cos_a]],
        device=patch.device,
        dtype=patch.dtype,
    )
    shear = torch.tensor(
        [[1.0, shear_x], [shear_y, 1.0]],
        device=patch.device,
        dtype=patch.dtype,
    )
    forward_matrix = rotation @ shear
    inverse_matrix = torch.linalg.inv(forward_matrix)
    theta = torch.zeros((1, 2, 3), device=patch.device, dtype=patch.dtype)
    theta[0, :, :2] = inverse_matrix

    grid = F.affine_grid(theta, patch.shape, align_corners=False)
    transformed_patch = F.grid_sample(
        patch,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    alpha = torch.ones(
        (patch.shape[0], 1, patch.shape[2], patch.shape[3]),
        device=patch.device,
        dtype=patch.dtype,
    )
    transformed_alpha = F.grid_sample(
        alpha,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).clamp(0.0, 1.0)
    return transformed_patch, transformed_alpha


def apply_patch(
    images: torch.Tensor,
    patch: torch.Tensor,
    alpha: torch.Tensor,
    patch_x: int,
    patch_y: int,
) -> torch.Tensor:
    _, _, height, width = images.shape
    _, _, patch_h, patch_w = patch.shape
    padding = (patch_x, width - patch_x - patch_w, patch_y, height - patch_y - patch_h)
    canvas = F.pad(patch, padding)
    mask = F.pad(alpha, padding)
    return (images * (1.0 - mask) + canvas * mask).clamp(0.0, 1.0)


def sample_training_position(
    rng: np.random.Generator,
    image_hw: tuple[int, int],
    patch_size: int,
    args: argparse.Namespace,
) -> tuple[int, int]:
    if args.fixed_training_position:
        patch_x, patch_y, _, _ = resolve_patch_box(
            image_hw,
            patch_size,
            args.patch_x,
            args.patch_y,
        )
        return patch_x, patch_y
    height, width = image_hw
    patch_x = int(rng.integers(0, width - patch_size + 1))
    patch_y = int(rng.integers(0, height - patch_size + 1))
    return patch_x, patch_y


def save_patch(patch: torch.Tensor, patch_dir: Path, metadata: dict) -> Path:
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_npz = patch_dir / "universal_patch.npz"
    np.savez_compressed(patch_npz, patch=patch.detach().float().cpu().numpy())
    to_pil_image(patch.squeeze(0).detach().float().cpu().clamp(0, 1)).save(
        patch_dir / "universal_patch.png"
    )
    with (patch_dir / "universal_patch_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return patch_npz


def load_patch(path: str, device: torch.device) -> torch.Tensor:
    with np.load(path) as data:
        if "patch" not in data:
            raise KeyError(f"`patch` not found in {path}")
        patch = torch.from_numpy(np.asarray(data["patch"]).astype(np.float32))
    if patch.ndim != 4 or patch.shape[0] != 1 or patch.shape[1] != 3:
        raise ValueError(f"Expected patch shape (1,3,H,W), got {tuple(patch.shape)}")
    return patch.to(device)


def load_patch_metadata(path: str) -> dict:
    metadata_path = Path(path).with_name("universal_patch_meta.json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def scheduled_lr(
    base_lr: float,
    update_idx_zero_based: int,
    total_updates: int,
    warmup_updates: int,
    scheduler: str,
) -> float:
    if scheduler == "none":
        return base_lr
    if warmup_updates > 0 and update_idx_zero_based < warmup_updates:
        return base_lr * float(update_idx_zero_based + 1) / float(warmup_updates)
    if scheduler == "cosine":
        decay_updates = max(1, total_updates - warmup_updates)
        progress = min(1.0, max(0.0, (update_idx_zero_based - warmup_updates) / decay_updates))
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown scheduler: {scheduler}")


def train_universal_patch(
    model: torch.nn.Module,
    train_scenes: list[Path],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> tuple[torch.Tensor, dict]:
    rng = np.random.default_rng(args.seed)
    first_images, _, _ = load_scene_images(train_scenes[0], train_max_frames(args), args.seed, device)
    patch_size = resolve_universal_patch_size(first_images, args)
    first_hw = tuple(int(value) for value in first_images.shape[-2:])
    del first_images

    patch = torch.rand(
        (1, 3, patch_size, patch_size),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    optimizer = torch.optim.AdamW([patch], lr=args.patch_lr)

    patch_dir = output_dir / "universal_patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    history_path = patch_dir / "training_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    total_updates = args.iterations * args.inner_loop
    warmup_updates = args.warmup_iterations * args.inner_loop
    update_idx = 0
    last_terms: dict[str, float] | None = None
    started = time.time()

    with history_path.open("a", encoding="utf-8") as history_file:
        for iteration in range(args.iterations):
            n_batch = min(args.scenes_per_iteration, len(train_scenes))
            scene_indices = rng.choice(len(train_scenes), size=n_batch, replace=False)
            scene_batch = []
            for scene_idx in scene_indices:
                scene_dir = train_scenes[int(scene_idx)]
                images, _, _ = load_scene_images(scene_dir, train_max_frames(args), args.seed, device)
                image_hw = tuple(int(value) for value in images.shape[-2:])
                _, _, patch_h, patch_w = resolve_patch_box(
                    image_hw,
                    patch_size,
                    args.patch_x,
                    args.patch_y,
                )
                if (patch_h, patch_w) != (patch_size, patch_size):
                    raise ValueError(
                        f"Patch {patch_size}x{patch_size} does not fit scene "
                        f"{scene_dir.name} with image size {image_hw}"
                    )
                with torch.no_grad():
                    clean_features = [
                        feature.detach()
                        for feature in extract_features(model, images, dtype, args.feature_layer)
                    ]
                scene_batch.append(
                    {
                        "scene_dir": scene_dir,
                        "images": images,
                        "clean_features": clean_features,
                        "image_hw": image_hw,
                    }
                )

            for inner_step in range(args.inner_loop):
                optimizer.zero_grad(set_to_none=True)
                current_lr = scheduled_lr(
                    args.patch_lr,
                    update_idx,
                    total_updates,
                    warmup_updates,
                    args.scheduler,
                )
                set_optimizer_lr(optimizer, current_lr)
                feature_l1_values = []
                geometry_records = []
                for scene_item in scene_batch:
                    geometry = sample_geometry(rng, args)
                    patch_x, patch_y = sample_training_position(
                        rng,
                        scene_item["image_hw"],
                        patch_size,
                        args,
                    )
                    transformed_patch, alpha = transform_patch(patch, **geometry)
                    adv_images = apply_patch(
                        scene_item["images"],
                        transformed_patch,
                        alpha,
                        patch_x,
                        patch_y,
                    )
                    adv_features = extract_features(
                        model,
                        adv_images,
                        dtype,
                        args.feature_layer,
                        args.activation_checkpoint,
                    )
                    loss, terms = feature_l1_loss(adv_features, scene_item["clean_features"])
                    (-loss / len(scene_batch)).backward()
                    feature_l1_values.append(terms["feature_l1"])
                    geometry_records.append(
                        {
                            "scene": scene_item["scene_dir"].name,
                            "patch_x": patch_x,
                            "patch_y": patch_y,
                            **geometry,
                        }
                    )

                if patch.grad is None:
                    raise RuntimeError("Universal patch gradient is None; check the forward graph.")
                optimizer.step()
                with torch.no_grad():
                    patch.clamp_(0.0, 1.0)

                update_idx += 1
                mean_feature_l1 = float(np.mean(feature_l1_values))
                terms = {"feature_l1": mean_feature_l1, "total": mean_feature_l1}
                record = {
                    "iteration": iteration + 1,
                    "inner_step": inner_step + 1,
                    "update": update_idx,
                    "lr": current_lr,
                    "scenes": [item["scene_dir"].name for item in scene_batch],
                    "geometries": geometry_records,
                    **terms,
                }
                history_file.write(json.dumps(record) + "\n")
                last_terms = terms
                if update_idx == 1 or update_idx % args.log_every == 0 or update_idx == total_updates:
                    print(
                        f"[train] update {update_idx:06d}/{total_updates:06d} "
                        f"scenes={len(scene_batch)} feature_l1={mean_feature_l1:.6f}"
                    )

            del scene_batch

    elapsed = time.time() - started
    metadata = {
        "mode": "vla_attacker_style_universal_feature_patch",
        "attack_target": "feature_l1_clean_vs_adversarial",
        "feature_layer": args.feature_layer,
        "optimizer": "adamw",
        "patch_lr": args.patch_lr,
        "scheduler": args.scheduler,
        "warmup_iterations": args.warmup_iterations,
        "warmup_updates": warmup_updates,
        "patch_shape": list(patch.shape),
        "patch_area_ratio_requested": args.patch_area_ratio,
        "reference_image_hw": list(first_hw),
        "actual_patch_area_ratio_on_reference": (patch_size * patch_size) / (first_hw[0] * first_hw[1]),
        "max_frames": args.max_frames,
        "train_max_frames": train_max_frames(args),
        "eval_max_frames": eval_max_frames(args),
        "iterations": args.iterations,
        "inner_loop": args.inner_loop,
        "scenes_per_iteration": args.scenes_per_iteration,
        "total_updates": total_updates,
        "total_scene_gradient_evaluations": total_updates * min(args.scenes_per_iteration, len(train_scenes)),
        "evaluation_patch_x": args.patch_x,
        "evaluation_patch_y": args.patch_y,
        "training_position": "fixed" if args.fixed_training_position else "random",
        "training_geometry": {
            "enabled": not args.no_geometry,
            "probability": args.geometry_prob,
            "rotation_degrees": args.rotation_degrees,
            "shear_x_y": args.shear,
        },
        "activation_checkpoint": args.activation_checkpoint,
        "train_scenes_root": args.train_scenes_root,
        "train_scene_pattern": args.train_scene_pattern,
        "n_train_scenes": len(train_scenes),
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "last_logged_feature_l1": None if last_terms is None else last_terms["feature_l1"],
        "paper_alignment": {
            "universal_visible_patch": True,
            "patch_area_ratio": 0.05,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "batch_size": 8,
            "inner_loop": 50,
            "max_iterations": 2000,
            "random_shear_rotation": True,
            "cosine_scheduler": True,
            "warmup_iterations": 20,
        },
        "official_implementation_alignment": {
            "random_training_position": not args.fixed_training_position,
            "rotation_degrees": args.rotation_degrees,
            "shear_x_y": args.shear,
            "geometry_probability": args.geometry_prob,
        },
        "adaptation": "VLA action loss replaced by label-free VGGT feature L1.",
    }
    save_patch(patch, patch_dir, metadata)
    print(f"[train done] saved universal patch -> {patch_dir / 'universal_patch.npz'}")
    return patch.detach(), metadata


def feature_layer_indices(feature_layer: str, n_layers: int) -> list[int]:
    def normalize(idx: int) -> int:
        idx = idx + n_layers if idx < 0 else idx
        if not 0 <= idx < n_layers:
            raise IndexError(f"Feature layer index {idx} is out of range for {n_layers} layers.")
        return idx

    if feature_layer == "aggregator_final":
        return [n_layers - 1]
    if feature_layer == "aggregator_all":
        return list(range(n_layers))
    if feature_layer.startswith("aggregator_"):
        idx_text = feature_layer.removeprefix("aggregator_")
        if idx_text.lstrip("-").isdigit():
            return [normalize(int(idx_text))]
    if feature_layer.lstrip("-").isdigit():
        return [normalize(int(feature_layer))]
    raise ValueError(f"Unknown feature layer: {feature_layer}")


def extract_logging_tokens(
    model: torch.nn.Module,
    images: torch.Tensor,
    dtype: torch.dtype,
    feature_layer: str,
) -> tuple[dict[int, torch.Tensor], list[int], int, int]:
    images_b = images if images.ndim == 5 else images[None]
    with autocast_context(images.device, dtype):
        tokens_list, patch_start_idx = model.aggregator(images_b)
    target_indices = feature_layer_indices(feature_layer, len(tokens_list))
    final_idx = len(tokens_list) - 1
    keep_indices = sorted(set(target_indices + [final_idx, *DPT_LAYER_INDICES]))
    kept = {
        idx: tokens_list[idx].detach().cpu()
        for idx in keep_indices
        if idx < len(tokens_list)
    }
    return kept, target_indices, final_idx, int(patch_start_idx)


def summarize_values(values: torch.Tensor, changed_threshold: float) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0, "changed_ratio": 0.0}
    return {
        "mean": float(flat.mean().cpu()),
        "p95": float(torch.quantile(flat, 0.95).cpu()),
        "max": float(flat.max().cpu()),
        "changed_ratio": float((flat > changed_threshold).float().mean().cpu()),
    }


def token_l1_stats(
    clean_tokens: dict[int, torch.Tensor],
    adv_tokens: dict[int, torch.Tensor],
    final_idx: int,
    patch_start_idx: int,
    changed_threshold: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    if clean_tokens.keys() != adv_tokens.keys():
        raise ValueError("Clean and adversarial logging token layers differ.")

    stats: dict[str, dict] = {}
    maps: dict[str, np.ndarray] = {}

    def layer_stats(layer_idx: int, include_special: bool) -> dict:
        per_token = (adv_tokens[layer_idx].float() - clean_tokens[layer_idx].float()).abs().mean(dim=-1)
        result = {
            "all_tokens": summarize_values(per_token, changed_threshold),
            "patch_tokens": summarize_values(per_token[:, :, patch_start_idx:], changed_threshold),
        }
        if include_special:
            result["camera_token"] = summarize_values(per_token[:, :, 0], changed_threshold)
            result["register_tokens"] = summarize_values(
                per_token[:, :, 1:patch_start_idx],
                changed_threshold,
            )
        maps[f"layer_{layer_idx}_patch_token_l1"] = (
            per_token[:, :, patch_start_idx:].squeeze(0).cpu().numpy().astype(np.float32)
        )
        return result

    stats["final"] = layer_stats(final_idx, include_special=True)
    for layer_idx in DPT_LAYER_INDICES:
        if layer_idx in clean_tokens:
            stats[f"layer_{layer_idx}"] = layer_stats(layer_idx, include_special=False)
    return stats, maps


def evaluate_scene(
    model: torch.nn.Module,
    patch: torch.Tensor,
    scene_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    patch_metadata: dict,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    images, image_paths, frame_indices = load_scene_images(
        scene_dir,
        eval_max_frames(args),
        args.seed,
        device,
        manifest_indices(args, scene_dir.name),
    )
    image_hw = tuple(int(value) for value in images.shape[-2:])
    patch_size = int(patch.shape[-1])
    patch_x, patch_y, patch_h, patch_w = resolve_patch_box(
        image_hw,
        patch_size,
        args.patch_x,
        args.patch_y,
    )
    if (patch_h, patch_w) != tuple(patch.shape[-2:]):
        raise ValueError(
            f"Patch {tuple(patch.shape[-2:])} does not fit scene {scene_dir.name} with image size {image_hw}"
        )

    alpha = torch.ones(
        (1, 1, patch.shape[-2], patch.shape[-1]),
        device=patch.device,
        dtype=patch.dtype,
    )
    adv_images = apply_patch(images, patch, alpha, patch_x, patch_y).detach()

    with torch.no_grad():
        clean_tokens, target_indices, final_idx, patch_start_idx = extract_logging_tokens(
            model,
            images,
            dtype,
            args.feature_layer,
        )
        adv_tokens, adv_target_indices, adv_final_idx, adv_patch_start_idx = extract_logging_tokens(
            model,
            adv_images,
            dtype,
            args.feature_layer,
        )
        if patch_start_idx != adv_patch_start_idx:
            raise ValueError("Clean and adversarial patch_start_idx differ.")
        if target_indices != adv_target_indices or final_idx != adv_final_idx:
            raise ValueError("Clean and adversarial logging layer indices differ.")
        clean_target = [clean_tokens[idx] for idx in target_indices]
        adv_target = [adv_tokens[idx] for idx in target_indices]
        final_loss, _ = feature_l1_loss(adv_target, clean_target)
        adv_preds = detach_predictions(forward_vggt(model, adv_images, dtype))

    image_names = [Path(path).name for path in image_paths]
    save_official_style_npz(out_dir / "vggt_outputs.npz", adv_preds, image_names, image_hw)
    stats, maps = token_l1_stats(
        clean_tokens,
        adv_tokens,
        final_idx,
        patch_start_idx,
        args.changed_token_threshold,
    )
    np.savez_compressed(out_dir / "feature_l1_maps.npz", **maps)

    summary = {
        "scene": str(scene_dir),
        "dataset": args.dataset,
        "n_frames": len(image_paths),
        "max_frames": args.max_frames,
        "train_max_frames": train_max_frames(args),
        "eval_max_frames": eval_max_frames(args),
        "image_paths": [str(path) for path in image_paths],
        "frame_indices": frame_indices.astype(int).tolist(),
        "frame_manifest": args.frame_manifest,
        "ckpt": args.ckpt,
        "mode": "vla_attacker_style_universal_feature_patch",
        "attack_target": "feature_l1_clean_vs_adversarial",
        "feature_layer": args.feature_layer,
        "final_feature_l1": float(final_loss.detach().cpu()),
        "token_l1_stats": stats,
        "changed_token_threshold": args.changed_token_threshold,
        "patch": {
            "patch_x": patch_x,
            "patch_y": patch_y,
            "patch_h": patch_h,
            "patch_w": patch_w,
            "evaluation_geometry": "none",
            "shared_universal_patch": True,
        },
        "universal_patch_metadata": patch_metadata,
        "outputs": {
            "attacked_vggt_outputs": "vggt_outputs.npz",
            "feature_l1_maps": "feature_l1_maps.npz",
        },
    }
    with (out_dir / "attack_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[apply] {scene_dir.name}: final_feature_l1={summary['final_feature_l1']:.6f} "
        f"-> {out_dir / 'vggt_outputs.npz'}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and apply a VLA-Attacker-style universal feature patch against VGGT."
    )
    parser.add_argument("--train_scenes_root", required=True, help="Scene root used to optimize the universal patch.")
    parser.add_argument("--train_scene_pattern", default="*", help="Glob for training scene folders.")
    parser.add_argument("--eval_scenes_root", help="Scene root used to generate attacked outputs; defaults to training root.")
    parser.add_argument("--eval_scene_pattern", help="Glob for evaluation scenes; defaults to training pattern.")
    parser.add_argument("--output_dir", required=True, help="Output root containing universal_patch/ and scene outputs.")
    parser.add_argument("--patch_path", help="Reuse an existing universal_patch.npz instead of training a new patch.")
    parser.add_argument("--dataset", default="generic", help="Dataset name stored in metadata.")
    parser.add_argument("--ckpt", default="facebook/VGGT-1B", help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--local_files_only", action="store_true", help="Load Hugging Face checkpoint from local cache only.")
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum frames per scene; 0 keeps all frames.")
    parser.add_argument("--train_max_frames", type=int, default=None, help="Maximum frames per training scene; defaults to --max_frames.")
    parser.add_argument("--eval_max_frames", type=int, default=None, help="Maximum frames per evaluation scene; defaults to --max_frames.")
    parser.add_argument("--frame_manifest", default=None, help="Optional JSON mapping eval scene names to exact frame indices.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--feature_layer", default="aggregator_final", help="Feature layer used by the attack objective.")

    parser.add_argument("--iterations", type=int, default=2000, help="Outer patch-training iterations.")
    parser.add_argument("--inner_loop", type=int, default=50, help="Patch updates per sampled scene.")
    parser.add_argument(
        "--scenes_per_iteration",
        type=int,
        default=6,
        help="Scenes accumulated per outer iteration; use 1 for expensive multi-frame scenes.",
    )
    parser.add_argument("--patch_lr", type=float, default=0.001, help="AdamW learning rate.")
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--warmup_iterations", type=int, default=20)
    parser.add_argument("--patch_area_ratio", type=float, default=0.05, help="Patch area ratio when --patch_size is not set.")
    parser.add_argument("--patch_size", type=int, default=0, help="Explicit square patch size; 0 derives it from area ratio.")
    parser.add_argument("--patch_x", type=int, default=-1, help="Fixed patch left coordinate; -1 centers the patch.")
    parser.add_argument("--patch_y", type=int, default=-1, help="Fixed patch top coordinate; -1 centers the patch.")
    parser.add_argument(
        "--fixed_training_position",
        action="store_true",
        help="Use --patch_x/--patch_y during training instead of sampling a random position.",
    )
    parser.add_argument("--rotation_degrees", type=float, default=30.0, help="Maximum absolute training rotation.")
    parser.add_argument("--shear", type=float, default=0.2, help="Maximum absolute x/y shear coefficient.")
    parser.add_argument("--geometry_prob", type=float, default=0.8, help="Probability of applying rotation/shear per update.")
    parser.add_argument("--no_geometry", action="store_true", help="Disable random training rotation and shear.")
    parser.add_argument(
        "--activation_checkpoint",
        action="store_true",
        help="Use VGGT aggregator activation checkpointing during patch optimization.",
    )
    parser.add_argument("--changed_token_threshold", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=10, help="Print every N patch updates.")
    parser.add_argument("--skip_existing_outputs", action="store_true", help="Skip scenes with vggt_outputs.npz already present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seeds(args.seed)

    if args.iterations <= 0 or args.inner_loop <= 0 or args.scenes_per_iteration <= 0:
        raise ValueError("--iterations, --inner_loop, and --scenes_per_iteration must be positive.")
    if not 0.0 < args.patch_area_ratio <= 1.0:
        raise ValueError("--patch_area_ratio must be in (0, 1].")
    if not 0.0 <= args.geometry_prob <= 1.0:
        raise ValueError("--geometry_prob must be in [0, 1].")

    eval_scenes_root = args.eval_scenes_root or args.train_scenes_root
    eval_scene_pattern = args.eval_scene_pattern or args.train_scene_pattern
    train_scenes = list_scene_dirs(args.train_scenes_root, args.train_scene_pattern)
    eval_scenes = list_scene_dirs(eval_scenes_root, eval_scene_pattern)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.frame_manifest_data = load_frame_manifest(args.frame_manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"[cfg] device={device} dtype={dtype}")
    print(
        f"[cfg] dataset={args.dataset} feature_layer={args.feature_layer} "
        f"train_scenes={len(train_scenes)} eval_scenes={len(eval_scenes)} "
        f"train_max_frames={train_max_frames(args)} eval_max_frames={eval_max_frames(args)} "
        f"frame_manifest={args.frame_manifest} activation_checkpoint={args.activation_checkpoint}"
    )
    print(f"[model] loading {args.ckpt}")
    model = load_model(args, device)
    for param in model.parameters():
        param.requires_grad_(False)

    if args.patch_path:
        patch = load_patch(args.patch_path, device)
        patch_metadata = {
            **load_patch_metadata(args.patch_path),
            "mode": "vla_attacker_style_universal_feature_patch",
            "loaded_patch_path": str(Path(args.patch_path)),
            "patch_shape": list(patch.shape),
            "adaptation": "VLA action loss replaced by label-free VGGT feature L1.",
        }
        print(f"[patch] loaded -> {args.patch_path}")
    else:
        patch, patch_metadata = train_universal_patch(
            model,
            train_scenes,
            args,
            device,
            dtype,
            output_dir,
        )

    summaries = []
    for scene_dir in eval_scenes:
        out_dir = output_dir / scene_dir.name
        if args.skip_existing_outputs and (out_dir / "vggt_outputs.npz").exists():
            print(f"[skip] {scene_dir.name}: existing vggt_outputs.npz")
            summary_path = out_dir / "attack_summary.json"
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as f:
                    summaries.append(json.load(f))
            continue
        try:
            summaries.append(
                evaluate_scene(
                    model,
                    patch,
                    scene_dir,
                    out_dir,
                    args,
                    device,
                    dtype,
                    patch_metadata,
                )
            )
        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[error] {scene_dir.name}: CUDA OOM: {exc}")
        except Exception as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[error] {scene_dir.name}: {exc}")

    with (output_dir / "attack_batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"[done] generated {len(summaries)}/{len(eval_scenes)} scene outputs")


if __name__ == "__main__":
    main()
