from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one clean VGGT baseline against one or more saved attack variants."
    )
    parser.add_argument("--scene_dir", default=None, help="Single scene directory.")
    parser.add_argument("--scenes_root", default=None, help="Batch mode: parent directory containing scene folders.")
    parser.add_argument("--scene_pattern", default="*", help="Batch mode glob pattern for scene folders.")
    parser.add_argument("--output_dir", required=True, help="Directory for evaluation outputs.")
    parser.add_argument("--clean_npz", default=None, help="Single-scene clean vggt_outputs.npz.")
    parser.add_argument("--clean_output_root", default=None, help="Batch root containing scene_name/vggt_outputs.npz.")
    parser.add_argument(
        "--attack_npz",
        action="append",
        default=[],
        help="Explicit attack npz for single-scene mode. Can be repeated.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Named explicit variant for single-scene mode: name=/path/to/attack_outputs.npz.",
    )
    parser.add_argument(
        "--attack_output_root",
        action="append",
        default=[],
        help="Root containing scene/variant/attack_outputs.npz or scene/variant/pgd_vggt_outputs.npz. Can be repeated.",
    )
    parser.add_argument(
        "--gt_root",
        required=True,
        help="CO3D raw GT root containing category/frame_annotations.jgz plus images/depths/depth_masks.",
    )
    parser.add_argument("--ckpt", default="facebook/VGGT-1B", help="Needed for matching eval or --run_clean_forward.")
    parser.add_argument("--local_files_only", action="store_true", help="Load HF checkpoint from local cache only.")
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum frames; 0 keeps all frames.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--metric_max_points", type=int, default=20000)
    parser.add_argument("--skip_matching_eval", action="store_true")
    parser.add_argument("--matching_max_keypoints", type=int, default=5000)
    parser.add_argument("--matching_det_thresh", type=float, default=0.005)
    parser.add_argument("--matching_ransac_thresh", type=float, default=1e-3)
    parser.add_argument("--matching_min_inliers", type=int, default=15)
    parser.add_argument("--matching_min_vis", type=float, default=0.5)
    parser.add_argument("--matching_min_conf", type=float, default=0.0)
    parser.add_argument("--matching_max_pairs", type=int, default=0)
    parser.add_argument(
        "--run_clean_forward",
        action="store_true",
        help="Run clean VGGT forward instead of loading --clean_npz/--clean_output_root.",
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


def scene_output_dir(args: argparse.Namespace, scene_dir: Path) -> Path:
    root = Path(args.output_dir)
    if args.scenes_root is not None:
        return root / scene_dir.name
    return root


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def label_for_npz(npz_path: Path, scene_name: str) -> str:
    parent = npz_path.parent
    if parent.name == scene_name:
        return npz_path.stem
    if parent.parent.name == scene_name:
        return parent.name
    return parent.name or npz_path.stem


def discover_variant_npzs(root: Path, scene_name: str, single_scene: bool) -> list[Path]:
    candidates: list[Path] = []
    scene_root = root / scene_name
    for base in (scene_root, root if single_scene else scene_root):
        candidates.extend(
            [
                base / "attack_outputs.npz",
                base / "pgd_vggt_outputs.npz",
            ]
        )
        candidates.extend(base.glob("*/attack_outputs.npz"))
        candidates.extend(base.glob("*/pgd_vggt_outputs.npz"))
    return unique_paths(path for path in candidates if path.exists())


def explicit_variants(args: argparse.Namespace) -> list[tuple[str | None, Path]]:
    variants: list[tuple[str | None, Path]] = []
    for item in args.attack_npz:
        variants.append((None, Path(item)))
    for item in args.variant:
        if "=" not in item:
            raise ValueError(f"Invalid --variant {item!r}; expected name=/path/to/file.npz")
        name, path_text = item.split("=", 1)
        variants.append((name.strip(), Path(path_text)))
    return variants


def collect_variants_for_scene(args: argparse.Namespace, scene_dir: Path) -> list[tuple[str, Path]]:
    collected: list[tuple[str, Path]] = []
    for name, path in explicit_variants(args):
        if not path.exists():
            raise FileNotFoundError(f"Missing attack npz: {path}")
        collected.append((name or label_for_npz(path, scene_dir.name), path))

    for root_text in args.attack_output_root:
        root = Path(root_text)
        for path in discover_variant_npzs(root, scene_dir.name, single_scene=args.scene_dir is not None):
            collected.append((label_for_npz(path, scene_dir.name), path))

    seen: set[str] = set()
    unique: list[tuple[str, Path]] = []
    for name, path in collected:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, path))
    return unique


def load_variant_metadata(npz_path: Path, variant_name: str) -> dict:
    meta_path = npz_path.parent / "attack_meta.json"
    metadata: dict = {"variant": variant_name}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            saved = json.load(f)
        metadata.update(saved)
        attack_spec = saved.get("attack_spec", {})
        if isinstance(attack_spec, dict):
            metadata.update({k: v for k, v in attack_spec.items() if k not in metadata})
    return metadata


def make_eval_config(args: argparse.Namespace) -> EvaluationConfig:
    from vggt_attack_eval.evaluation import EvaluationConfig, MatchingConfig

    return EvaluationConfig(
        metric_max_points=args.metric_max_points,
        skip_matching_eval=args.skip_matching_eval,
        matching=MatchingConfig(
            max_keypoints=args.matching_max_keypoints,
            det_thresh=args.matching_det_thresh,
            ransac_thresh=args.matching_ransac_thresh,
            min_inliers=args.matching_min_inliers,
            min_visibility=args.matching_min_vis,
            min_confidence=args.matching_min_conf,
            max_pairs=args.matching_max_pairs,
        ),
    )


def process_scene(
    *,
    model,
    scene_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    config: EvaluationConfig,
) -> list[dict]:
    from vggt.utils.load_fn import load_and_preprocess_images

    from attack_vggt_new1 import align_image_paths_to_clean, derive_scene_seed, load_co3d_gt_reference
    from vggt_attack_eval.evaluation import (
        PredictionVariant,
        evaluate_variants,
        load_images_from_npz,
        load_or_forward_clean_predictions,
        load_predictions_from_npz,
        read_npz_image_names,
        write_scene_evaluation,
    )

    clean_npz = clean_npz_for_scene(args, scene_dir)
    if clean_npz is not None and not clean_npz.exists() and not args.run_clean_forward:
        raise FileNotFoundError(f"Missing clean output: {clean_npz}")

    variant_paths = collect_variants_for_scene(args, scene_dir)
    if not variant_paths:
        raise ValueError(f"No attack variants found for {scene_dir.name}")

    scene_seed = derive_scene_seed(args.seed, scene_dir.name)
    alignment_npz = clean_npz if clean_npz is not None else variant_paths[0][1]
    alignment_max_frames = args.max_frames if clean_npz is not None else 0
    image_paths, frame_indices = align_image_paths_to_clean(
        scene_dir,
        alignment_npz,
        alignment_max_frames,
        scene_seed,
    )
    if not image_paths:
        raise ValueError(f"No images found under {scene_dir}")

    clean_images = None
    if clean_npz is None:
        clean_images = load_images_from_npz(variant_paths[0][1], device, image_key="clean_images")
    if clean_images is None:
        clean_images = load_and_preprocess_images(image_paths).to(device)
    image_size_hw = tuple(clean_images.shape[-2:])
    gt_refs = load_co3d_gt_reference(Path(args.gt_root), scene_dir, image_paths, device, image_size_hw)
    if clean_npz is None and not args.run_clean_forward:
        try:
            clean_preds = load_predictions_from_npz(
                variant_paths[0][1],
                device,
                image_size_hw,
                prefix="clean_",
            )
        except ValueError as exc:
            raise ValueError(
                "No clean npz was provided and the first attack npz does not contain clean_* tensors. "
                "Pass --clean_npz/--clean_output_root or use --run_clean_forward."
            ) from exc
    else:
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

    variants: list[PredictionVariant] = []
    expected_names = [Path(path).name for path in image_paths]
    for name, npz_path in variant_paths:
        variant_names = read_npz_image_names(npz_path)
        if variant_names is not None and variant_names != expected_names:
            print(
                f"[warn] {name}: image_paths in {npz_path} do not match the evaluation frames; "
                "metrics may compare different frame selections."
            )
        preds = load_predictions_from_npz(npz_path, device, image_size_hw, prefix="adv_")
        images = load_images_from_npz(npz_path, device, image_key="adv_images")
        variants.append(
            PredictionVariant(
                name=name,
                preds=preds,
                images=images,
                npz_path=npz_path,
                metadata=load_variant_metadata(npz_path, name),
            )
        )

    clean_baseline, summaries = evaluate_variants(
        model=model,
        clean_images=clean_images,
        gt_refs=gt_refs,
        clean_preds=clean_preds,
        variants=variants,
        scene_dir=scene_dir,
        image_size_hw=image_size_hw,
        dtype=dtype,
        device=device,
        config=config,
    )

    out_dir = scene_output_dir(args, scene_dir)
    write_scene_evaluation(out_dir=out_dir, clean_baseline=clean_baseline, variant_summaries=summaries)
    print(f"[eval] {scene_dir.name}: {len(summaries)} variants -> {out_dir}")
    return summaries


def main() -> None:
    args = parse_args()
    from attack_vggt_new1 import load_model, set_random_seeds
    from vggt_attack_eval.evaluation import (
        aggregate_dataset_metrics_by_variant,
        write_variant_tables,
    )

    set_random_seeds(args.seed)
    config = make_eval_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    needs_model = (not args.skip_matching_eval) or args.run_clean_forward
    model = None
    if needs_model:
        print(f"[model] loading {args.ckpt}")
        model = load_model(args, device)
        for param in model.parameters():
            param.requires_grad_(False)

    all_summaries: list[dict] = []
    scene_dirs = scene_dirs_from_args(args)
    for scene_dir in scene_dirs:
        try:
            all_summaries.extend(
                process_scene(
                    model=model,
                    scene_dir=scene_dir,
                    args=args,
                    device=device,
                    dtype=dtype,
                    config=config,
                )
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            print(f"[error] {scene_dir.name}: CUDA OOM: {exc}")
        except Exception as exc:
            torch.cuda.empty_cache()
            print(f"[error] {scene_dir.name}: {exc}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    with open(out_root / "dataset_metrics_by_variant.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_dataset_metrics_by_variant(all_summaries), f, indent=2)
    write_variant_tables(all_summaries, out_root, stem="all_variant_comparison")
    print(f"[done] evaluated {len(all_summaries)} variants -> {out_root}")


if __name__ == "__main__":
    main()
