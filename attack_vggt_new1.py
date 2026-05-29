"""Label-free feature-distance attack for VGGT.

Baseline idea:
    maximize L1(feature(VGGT, adversarial input), feature(VGGT, clean input))

The attack does not use dataset labels or GT. It only creates adversarial input
and saves VGGT outputs for later evaluation with recons_eval.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image


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


def subsample_indices(length: int, max_frames: int, seed: int) -> np.ndarray:
    if max_frames <= 0 or length <= max_frames:
        return np.arange(length)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=max_frames, replace=False))


def subsample(paths: list[str], max_frames: int, seed: int) -> tuple[list[str], np.ndarray]:
    indices = subsample_indices(len(paths), max_frames, seed)
    return [paths[int(i)] for i in indices], indices


def set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.cuda.amp.autocast(dtype=dtype)


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


def forward_vggt(model: torch.nn.Module, images: torch.Tensor, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    with autocast_context(images.device, dtype):
        preds = model(images)
    return {key: value for key, value in preds.items() if torch.is_tensor(value)}


def detach_predictions(preds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = ("pose_enc", "depth", "depth_conf", "world_points", "world_points_conf")
    return {key: preds[key].detach() for key in keys if key in preds}


def select_feature_tokens(tokens_list: list[torch.Tensor], feature_layer: str) -> list[torch.Tensor]:
    if feature_layer == "aggregator_final":
        return [tokens_list[-1]]
    if feature_layer == "aggregator_all":
        return list(tokens_list)
    if feature_layer.startswith("aggregator_"):
        idx_text = feature_layer.removeprefix("aggregator_")
        if idx_text.lstrip("-").isdigit():
            return [tokens_list[int(idx_text)]]
    if feature_layer.lstrip("-").isdigit():
        return [tokens_list[int(feature_layer)]]
    raise ValueError(
        f"Unknown --feature_layer {feature_layer!r}. Use aggregator_final, aggregator_all, "
        "aggregator_<index>, or an integer index."
    )


def extract_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    dtype: torch.dtype,
    feature_layer: str,
) -> list[torch.Tensor]:
    images_b = images if images.ndim == 5 else images[None]
    with autocast_context(images.device, dtype):
        tokens_list, _ = model.aggregator(images_b)
    return select_feature_tokens(tokens_list, feature_layer)


def feature_l1_loss(
    adv_features: list[torch.Tensor],
    clean_features: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(adv_features) != len(clean_features):
        raise RuntimeError("Feature count mismatch between clean and adversarial inputs.")
    losses = [
        (adv.float() - clean.detach().float()).abs().mean()
        for adv, clean in zip(adv_features, clean_features)
    ]
    loss = torch.stack(losses).mean()
    return loss, {
        "feature_l1": float(loss.detach().cpu()),
        "total": float(loss.detach().cpu()),
    }


def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    clean_features: list[torch.Tensor],
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    base = images.detach()
    delta = torch.empty_like(base).uniform_(-args.eps, args.eps) if not args.no_random_start else torch.zeros_like(base)
    adv_images = (base + delta).clamp(0.0, 1.0).detach()

    history: list[dict[str, float]] = []
    for step in range(args.steps):
        adv_images.requires_grad_(True)
        adv_features = extract_features(model, adv_images, dtype, args.feature_layer)
        loss, terms = feature_l1_loss(adv_features, clean_features)

        model.zero_grad(set_to_none=True)
        if adv_images.grad is not None:
            adv_images.grad.zero_()
        loss.backward()

        with torch.no_grad():
            grad = adv_images.grad
            if grad is None:
                raise RuntimeError("PGD gradient is None; check the forward graph.")
            adv_images = adv_images + args.alpha * grad.sign()
            delta = (adv_images - base).clamp(-args.eps, args.eps)
            adv_images = (base + delta).clamp(0.0, 1.0).detach()

        terms["step"] = step + 1
        history.append(terms)
        print_attack_step("global", step, args.steps, terms)
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
    clean_features: list[torch.Tensor],
    args: argparse.Namespace,
    dtype: torch.dtype,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, list[dict[str, float]], torch.Tensor, dict[str, int]]:
    base = images.detach()
    patch_x, patch_y, patch_h, patch_w = resolve_patch_box(image_hw, args.patch_size, args.patch_x, args.patch_y)
    patch = torch.rand((1, 3, patch_h, patch_w), device=base.device, dtype=torch.float32, requires_grad=True)
    patch_lr = args.patch_alpha if args.patch_alpha is not None else args.alpha
    optimizer = torch.optim.Adam([patch], lr=patch_lr)

    history: list[dict[str, float]] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        adv_images = apply_adversarial_patch(base, patch, patch_x, patch_y)
        adv_features = extract_features(model, adv_images, dtype, args.feature_layer)
        loss, terms = feature_l1_loss(adv_features, clean_features)

        model.zero_grad(set_to_none=True)
        (-loss).backward()

        with torch.no_grad():
            if patch.grad is None:
                raise RuntimeError("Patch gradient is None; check the forward graph.")
        optimizer.step()
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)

        terms["step"] = step + 1
        history.append(terms)
        print_attack_step("patch", step, args.steps, terms)

    adv_images = apply_adversarial_patch(base, patch, patch_x, patch_y).detach()
    patch_meta = {
        "patch_x": patch_x,
        "patch_y": patch_y,
        "patch_h": patch_h,
        "patch_w": patch_w,
    }
    return adv_images, history, patch.detach(), patch_meta


def print_attack_step(kind: str, step: int, total_steps: int, terms: dict[str, float]) -> None:
    print(
        f"[{kind}] step {step + 1:03d}/{total_steps:03d} "
        f"feature_l1={terms['feature_l1']:.6f}"
    )


def prediction_payload(preds: dict[str, torch.Tensor], image_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    payload: dict[str, np.ndarray] = {}
    if "pose_enc" in preds:
        extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], image_hw)
        payload["extrinsic"] = extrinsic.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        payload["intrinsic"] = intrinsic.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        payload["pose_enc"] = preds["pose_enc"].squeeze(0).detach().float().cpu().numpy().astype(np.float32)
    if "depth" in preds:
        payload["depth"] = preds["depth"].squeeze(0).detach().float().cpu().numpy().astype(np.float32)
    if "depth_conf" in preds:
        payload["depth_conf"] = preds["depth_conf"].squeeze(0).detach().float().cpu().numpy().astype(np.float32)
    if "world_points" in preds:
        payload["point_map"] = preds["world_points"].squeeze(0).detach().float().cpu().numpy().astype(np.float32)
    if "world_points_conf" in preds:
        payload["point_conf"] = preds["world_points_conf"].squeeze(0).detach().float().cpu().numpy().astype(np.float32)
    if {"depth", "extrinsic", "intrinsic"}.issubset(payload):
        payload["point_cloud_unproj"] = unproject_depth_map_to_point_map(
            payload["depth"],
            payload["extrinsic"],
            payload["intrinsic"],
        ).astype(np.float32)
    return payload


def save_official_style_npz(
    out_path: Path,
    preds: dict[str, torch.Tensor],
    image_names: list[str],
    image_hw: tuple[int, int],
) -> None:
    payload = prediction_payload(preds, image_hw)
    np.savez_compressed(out_path, image_paths=np.array(image_names), **payload)


def save_debug_pair_npz(
    out_path: Path,
    clean_preds: dict[str, torch.Tensor],
    adv_preds: dict[str, torch.Tensor],
    adv_images: torch.Tensor,
    image_names: list[str],
    image_hw: tuple[int, int],
) -> None:
    clean_payload = prediction_payload(clean_preds, image_hw)
    adv_payload = prediction_payload(adv_preds, image_hw)
    np.savez_compressed(
        out_path,
        image_paths=np.array(image_names),
        adv_images=adv_images.detach().cpu().numpy().astype(np.float16),
        **{f"clean_{key}": value for key, value in clean_payload.items()},
        **{f"adv_{key}": value for key, value in adv_payload.items()},
    )


def save_adv_images(adv_images: torch.Tensor, image_paths: list[str], out_dir: Path) -> None:
    image_dir = out_dir / "adv_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx, (img, image_path) in enumerate(zip(adv_images.detach().cpu(), image_paths)):
        stem = Path(image_path).stem
        to_pil_image(img.clamp(0, 1)).save(image_dir / f"{idx:03d}_{stem}_adv.png")


def save_delta_preview(clean_images: torch.Tensor, adv_images: torch.Tensor, out_dir: Path) -> None:
    preview_dir = out_dir / "delta_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for idx, (clean, adv) in enumerate(zip(clean_images.detach().cpu(), adv_images.detach().cpu())):
        delta = (adv - clean).abs()
        scale = delta.max().clamp_min(1e-6)
        img = (delta / scale).clamp(0, 1)
        to_pil_image(img).save(preview_dir / f"{idx:03d}_delta.png")


def save_patch_image(patch: torch.Tensor | None, out_dir: Path) -> None:
    if patch is None:
        return
    patch_dir = out_dir / "patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    to_pil_image(patch.squeeze(0).detach().cpu().clamp(0, 1)).save(patch_dir / "learned_patch.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run label-free feature-L1 patch/global attacks against VGGT.")
    parser.add_argument("--scene_dir", default=None, help="Single scene directory, with images under scene/images or scene itself.")
    parser.add_argument("--scenes_root", default=None, help="Batch mode: parent directory containing scene folders.")
    parser.add_argument("--scene_pattern", default="*", help="Batch mode glob pattern for scene folders.")
    parser.add_argument("--output_dir", required=True, help="Directory for attacked VGGT outputs.")
    parser.add_argument("--dataset", default="generic", help="Optional dataset name kept only for metadata.")
    parser.add_argument("--ckpt", default="facebook/VGGT-1B", help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--local_files_only", action="store_true", help="Load Hugging Face checkpoint from local cache only.")
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum number of frames to attack; 0 keeps all frames.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for frame sampling and attack initialization.")
    parser.add_argument("--attack_type", choices=("patch", "global"), default="patch", help="Patch is the main attack; global PGD is retained.")
    parser.add_argument("--feature_layer", default="aggregator_final", help="aggregator_final, aggregator_all, aggregator_<index>, or integer index.")
    parser.add_argument("--steps", type=int, default=10, help="Attack iterations.")
    parser.add_argument("--eps", type=float, default=8 / 255, help="Global L-infinity perturbation budget in [0, 1] pixels.")
    parser.add_argument("--alpha", type=float, default=1 / 255, help="Step size in [0, 1] pixels.")
    parser.add_argument("--patch_size", type=int, default=96, help="Square patch size in preprocessed input pixels.")
    parser.add_argument("--patch_alpha", type=float, default=None, help="Patch optimizer learning rate; defaults to --alpha.")
    parser.add_argument("--patch_x", type=int, default=-1, help="Patch left coordinate; -1 centers the patch.")
    parser.add_argument("--patch_y", type=int, default=-1, help="Patch top coordinate; -1 centers the patch.")
    parser.add_argument("--no_random_start", action="store_true", help="Start global PGD from the clean images.")
    parser.add_argument("--save_adv_images", action="store_true", help="Save adversarial input frames as PNGs.")
    parser.add_argument("--save_debug_pair", action="store_true", help="Save a debug npz with clean and attacked VGGT outputs.")
    return parser.parse_args()


def process_scene(
    model: torch.nn.Module,
    scene_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_seed = derive_scene_seed(args.seed, scene_dir.name)
    image_paths, frame_indices = subsample(find_images(scene_dir), args.max_frames, scene_seed)
    if not image_paths:
        raise ValueError(f"No images found under {scene_dir}")

    from vggt.utils.load_fn import load_and_preprocess_images

    images = load_and_preprocess_images(image_paths).to(device)
    image_hw = tuple(images.shape[-2:])
    image_names = [Path(path).name for path in image_paths]

    t0 = time.time()
    with torch.no_grad():
        clean_features = [feat.detach() for feat in extract_features(model, images, dtype, args.feature_layer)]
    print(f"[feature] clean {args.feature_layer} extracted in {time.time() - t0:.2f}s")

    patch_tensor = None
    patch_meta = None
    if args.attack_type == "patch":
        adv_images, history, patch_tensor, patch_meta = patch_attack(model, images, clean_features, args, dtype, image_hw)
    else:
        adv_images, history = pgd_attack(model, images, clean_features, args, dtype)

    with torch.no_grad():
        adv_preds = detach_predictions(forward_vggt(model, adv_images, dtype))
    save_official_style_npz(out_dir / "vggt_outputs.npz", adv_preds, image_names, image_hw)

    if args.save_debug_pair:
        with torch.no_grad():
            clean_preds = detach_predictions(forward_vggt(model, images, dtype))
        save_debug_pair_npz(out_dir / "attack_debug_outputs.npz", clean_preds, adv_preds, adv_images, image_names, image_hw)

    save_delta_preview(images, adv_images, out_dir)
    save_patch_image(patch_tensor, out_dir)
    if args.save_adv_images:
        save_adv_images(adv_images, image_paths, out_dir)

    summary = {
        "scene": str(scene_dir),
        "dataset": args.dataset,
        "n_frames": len(image_paths),
        "image_paths": [str(path) for path in image_paths],
        "frame_indices": frame_indices.astype(int).tolist(),
        "ckpt": args.ckpt,
        "attack_type": args.attack_type,
        "attack_target": "feature_l1_clean_vs_adversarial",
        "feature_layer": args.feature_layer,
        "steps": args.steps,
        "eps": args.eps,
        "alpha": args.alpha,
        "patch_optimizer": "adam" if args.attack_type == "patch" else None,
        "patch_lr": (args.patch_alpha if args.patch_alpha is not None else args.alpha) if args.attack_type == "patch" else None,
        "patch": patch_meta,
        "random_start": not args.no_random_start,
        "mode": "label_free_feature_attack_saved_for_recons_eval",
        "history": history,
        "outputs": {"attacked_vggt_outputs": "vggt_outputs.npz"},
    }
    with (out_dir / "attack_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] saved attacked VGGT output -> {out_dir / 'vggt_outputs.npz'}")
    return summary


def main() -> None:
    args = parse_args()
    set_random_seeds(args.seed)

    if args.scene_dir is None and args.scenes_root is None:
        raise ValueError("Provide either --scene_dir for one scene or --scenes_root for batch mode.")
    if args.scene_dir is not None and args.scenes_root is not None:
        raise ValueError("Use either --scene_dir or --scenes_root, not both.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"[cfg] device={device} dtype={dtype}")
    print(f"[cfg] attack={args.attack_type} feature_layer={args.feature_layer} dataset={args.dataset}")
    print(f"[model] loading {args.ckpt}")
    model = load_model(args, device)
    for param in model.parameters():
        param.requires_grad_(False)

    if args.scene_dir is not None:
        process_scene(model, Path(args.scene_dir), Path(args.output_dir), args, device, dtype)
        return

    output_root = Path(args.output_dir)
    scene_dirs = sorted(path for path in Path(args.scenes_root).glob(args.scene_pattern) if path.is_dir())
    summaries = []
    for scene_dir in scene_dirs:
        print(f"\n[scene] {scene_dir.name}")
        try:
            summary = process_scene(model, scene_dir, output_root / scene_dir.name, args, device, dtype)
            summaries.append(summary)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            print(f"[error] {scene_dir.name}: CUDA OOM: {exc}")
        except Exception as exc:
            torch.cuda.empty_cache()
            print(f"[error] {scene_dir.name}: {exc}")

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "attack_batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n[batch done] {len(summaries)}/{len(scene_dirs)} scenes attacked")


if __name__ == "__main__":
    main()
