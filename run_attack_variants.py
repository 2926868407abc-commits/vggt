from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one or more VGGT attack variants without running the evaluation suite."
    )
    parser.add_argument("--scene_dir", default=None, help="Single scene directory.")
    parser.add_argument("--scenes_root", default=None, help="Batch mode: parent directory containing scene folders.")
    parser.add_argument("--scene_pattern", default="*", help="Batch mode glob pattern for scene folders.")
    parser.add_argument("--output_dir", required=True, help="Directory for attack outputs.")
    parser.add_argument("--clean_npz", default=None, help="Single-scene clean vggt_outputs.npz.")
    parser.add_argument("--clean_output_root", default=None, help="Batch root containing scene_name/vggt_outputs.npz.")
    parser.add_argument(
        "--gt_root",
        required=True,
        help="CO3D raw GT root containing category/frame_annotations.jgz plus images/depths/depth_masks.",
    )
    parser.add_argument("--ckpt", default="facebook/VGGT-1B", help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--local_files_only", action="store_true", help="Load HF checkpoint from local cache only.")
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum frames; 0 keeps all frames.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--attack_type", choices=("global", "patch"), default="patch")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=1 / 255)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--patch_alpha", type=float, default=None)
    parser.add_argument("--patch_x", type=int, default=-1)
    parser.add_argument("--patch_y", type=int, default=-1)
    parser.add_argument(
        "--patch_positions",
        default=None,
        help="Comma-separated patch positions for patch attacks: center,0:0,128:128.",
    )
    parser.add_argument("--no_random_start", action="store_true")

    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--pose_weight", type=float, default=0.2)
    parser.add_argument("--points_weight", type=float, default=0.5)
    parser.add_argument("--save_adv_images", action="store_true")
    parser.add_argument(
        "--run_clean_forward",
        action="store_true",
        help="Run a clean VGGT forward for the saved clean_* tensors instead of reading --clean_npz.",
    )
    return parser.parse_args()


def scene_dirs_from_args(args: argparse.Namespace) -> list[Path]:
    if args.scene_dir is None and args.scenes_root is None:
        raise ValueError("Provide either --scene_dir or --scenes_root.")
    if args.scene_dir is not None and args.scenes_root is not None:
        raise ValueError("Use either --scene_dir or --scenes_root, not both.")
    if args.scene_dir is not None:
        return [Path(args.scene_dir)]
    return sorted(d for d in Path(args.scenes_root).glob(args.scene_pattern) if d.is_dir())


def clean_npz_for_scene(args: argparse.Namespace, scene_dir: Path) -> Path | None:
    if args.clean_npz:
        return Path(args.clean_npz)
    if args.clean_output_root:
        return Path(args.clean_output_root) / scene_dir.name / "vggt_outputs.npz"
    return None


def output_dir_for_scene(args: argparse.Namespace, scene_dir: Path) -> Path:
    root = Path(args.output_dir)
    if args.scenes_root is not None:
        return root / scene_dir.name
    return root


def process_scene(
    *,
    model,
    scene_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    from vggt.utils.load_fn import load_and_preprocess_images

    from attack_vggt_new1 import (
        align_image_paths_to_clean,
        derive_scene_seed,
        detach_predictions,
        forward_vggt,
        load_co3d_gt_reference,
        set_random_seeds,
    )
    from vggt_attack_eval.attacks import (
        AttackWeights,
        build_attack_specs,
        run_attack_spec,
        save_attack_output,
    )
    from vggt_attack_eval.evaluation import load_or_forward_clean_predictions

    scene_seed = derive_scene_seed(args.seed, scene_dir.name)
    clean_npz = clean_npz_for_scene(args, scene_dir)
    if clean_npz is not None and not clean_npz.exists() and not args.run_clean_forward:
        raise FileNotFoundError(f"Missing clean output: {clean_npz}")

    image_paths, frame_indices = align_image_paths_to_clean(scene_dir, clean_npz, args.max_frames, scene_seed)
    if not image_paths:
        raise ValueError(f"No images found under {scene_dir}")

    clean_images = load_and_preprocess_images(image_paths).to(device)
    image_size_hw = tuple(clean_images.shape[-2:])
    gt_refs = load_co3d_gt_reference(Path(args.gt_root), scene_dir, image_paths, device, image_size_hw)
    clean_preds = load_or_forward_clean_predictions(
        model=model,
        clean_npz=clean_npz,
        clean_images=clean_images,
        device=device,
        dtype=dtype,
        image_size_hw=image_size_hw,
        frame_indices=frame_indices,
        run_clean_forward=args.run_clean_forward,
    )

    weights = AttackWeights(args.depth_weight, args.pose_weight, args.points_weight)
    specs = build_attack_specs(
        attack_type=args.attack_type,
        steps=args.steps,
        eps=args.eps,
        alpha=args.alpha,
        random_start=not args.no_random_start,
        patch_size=args.patch_size,
        patch_alpha=args.patch_alpha,
        patch_x=args.patch_x,
        patch_y=args.patch_y,
        patch_positions=args.patch_positions,
    )

    scene_out_dir = output_dir_for_scene(args, scene_dir)
    scene_out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for spec in specs:
        set_random_seeds(derive_scene_seed(scene_seed, spec.name))
        print(f"[attack] {scene_dir.name} / {spec.name}")
        result = run_attack_spec(
            model=model,
            images=clean_images,
            reference_preds=gt_refs,
            dtype=dtype,
            spec=spec,
            weights=weights,
        )
        track_query = clean_preds["track"][:, 0] if "track" in clean_preds else None
        with torch.no_grad():
            adv_full = forward_vggt(model, result.adv_images, dtype, query_points=track_query)
        adv_preds = detach_predictions(adv_full)

        variant_out_dir = scene_out_dir / spec.name if len(specs) > 1 else scene_out_dir
        npz_path = save_attack_output(
            out_dir=variant_out_dir,
            result=result,
            clean_images=clean_images,
            adv_preds=adv_preds,
            clean_preds=clean_preds,
            image_paths=image_paths,
            image_size_hw=image_size_hw,
            weights=weights,
            save_images=args.save_adv_images,
        )
        summaries.append(
            {
                "scene": str(scene_dir),
                "variant": spec.name,
                "attack_type": spec.attack_type,
                "attack_npz": str(npz_path),
                "output_dir": str(variant_out_dir),
                "n_frames": len(image_paths),
                "attack_spec": {
                    "steps": spec.steps,
                    "eps": spec.eps,
                    "alpha": spec.alpha,
                    "patch_size": spec.patch_size,
                    "patch_x": spec.patch_x,
                    "patch_y": spec.patch_y,
                    "patch_alpha": spec.patch_alpha,
                    "random_start": spec.random_start,
                },
                "patch": result.patch_meta,
                "history": result.history,
            }
        )

    with open(scene_out_dir / "attack_variants_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    return summaries


def main() -> None:
    args = parse_args()
    from attack_vggt_new1 import load_model, set_random_seeds

    set_random_seeds(args.seed)

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

    all_summaries: list[dict] = []
    scene_dirs = scene_dirs_from_args(args)
    for scene_dir in scene_dirs:
        try:
            all_summaries.extend(
                process_scene(model=model, scene_dir=scene_dir, args=args, device=device, dtype=dtype)
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            print(f"[error] {scene_dir.name}: CUDA OOM: {exc}")
        except Exception as exc:
            torch.cuda.empty_cache()
            print(f"[error] {scene_dir.name}: {exc}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "attack_variants_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[done] saved {len(all_summaries)} attack variants -> {out_root}")


if __name__ == "__main__":
    main()
