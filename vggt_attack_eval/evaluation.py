from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vggt.utils.pose_enc import extri_intri_to_pose_encoding


@dataclass
class MatchingConfig:
    max_keypoints: int = 5000
    det_thresh: float = 0.005
    ransac_thresh: float = 1e-3
    min_inliers: int = 15
    min_visibility: float = 0.5
    min_confidence: float = 0.0
    max_pairs: int = 0


@dataclass
class EvaluationConfig:
    metric_max_points: int = 20000
    skip_matching_eval: bool = False
    matching: MatchingConfig = field(default_factory=MatchingConfig)


@dataclass
class PredictionVariant:
    name: str
    preds: dict[str, torch.Tensor]
    images: torch.Tensor | None = None
    npz_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanBaseline:
    metrics: dict[str, float | str]
    records: dict[str, Any]


@cache
def _legacy():
    import attack_vggt_new1

    return attack_vggt_new1


def _select_sequence(array: np.ndarray, frame_indices: np.ndarray | None) -> np.ndarray:
    if frame_indices is None:
        return array
    if array.ndim >= 2 and array.shape[0] == 1:
        return array[:, frame_indices]
    return array[frame_indices]


def _to_float_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32)).to(device)


def _ensure_batch(array: np.ndarray, unbatched_ndim: int) -> np.ndarray:
    if array.ndim == unbatched_ndim:
        return array[None]
    return array


def _first_array(
    data: np.lib.npyio.NpzFile,
    names: tuple[str, ...],
    prefix: str = "",
) -> np.ndarray | None:
    for name in names:
        prefixed = f"{prefix}{name}" if prefix else name
        if prefixed in data:
            return data[prefixed]
    if prefix:
        for name in names:
            if name in data:
                return data[name]
    return None


def load_predictions_from_npz(
    npz_path: Path,
    device: torch.device,
    image_size_hw: tuple[int, int],
    *,
    prefix: str = "",
    frame_indices: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Load VGGT predictions from clean outputs or prefixed attack outputs.

    Clean batch outputs normally use keys like depth/point_map/tracks. Attack
    outputs produced by this repo use adv_depth/adv_world_points/adv_track.
    """
    data = np.load(npz_path)
    preds: dict[str, torch.Tensor] = {}

    pose_enc = _first_array(data, ("pose_enc",), prefix)
    extrinsic = _first_array(data, ("extrinsic",), prefix)
    intrinsic = _first_array(data, ("intrinsic",), prefix)
    if pose_enc is not None:
        pose_enc = _ensure_batch(_select_sequence(pose_enc, frame_indices), 2)
        preds["pose_enc"] = _to_float_tensor(pose_enc, device)
    elif extrinsic is not None and intrinsic is not None:
        extrinsic = _ensure_batch(_select_sequence(extrinsic, frame_indices), 3)
        intrinsic = _ensure_batch(_select_sequence(intrinsic, frame_indices), 3)
        extrinsic_t = _to_float_tensor(extrinsic, device)
        intrinsic_t = _to_float_tensor(intrinsic, device)
        preds["extrinsic"] = extrinsic_t
        preds["intrinsic"] = intrinsic_t
        preds["pose_enc"] = extri_intri_to_pose_encoding(extrinsic_t, intrinsic_t, image_size_hw)

    depth = _first_array(data, ("depth",), prefix)
    if depth is not None:
        depth = _ensure_batch(_select_sequence(depth, frame_indices), 4)
        preds["depth"] = _to_float_tensor(depth, device)

    depth_conf = _first_array(data, ("depth_conf",), prefix)
    if depth_conf is not None:
        depth_conf = _ensure_batch(_select_sequence(depth_conf, frame_indices), 3)
        preds["depth_conf"] = _to_float_tensor(depth_conf, device)

    world_points = _first_array(data, ("world_points", "point_map"), prefix)
    if world_points is not None:
        world_points = _ensure_batch(_select_sequence(world_points, frame_indices), 4)
        preds["world_points"] = _to_float_tensor(world_points, device)

    world_points_conf = _first_array(data, ("world_points_conf", "point_conf"), prefix)
    if world_points_conf is not None:
        world_points_conf = _ensure_batch(_select_sequence(world_points_conf, frame_indices), 3)
        preds["world_points_conf"] = _to_float_tensor(world_points_conf, device)

    tracks = _first_array(data, ("track", "tracks"), prefix)
    if tracks is not None:
        tracks = _ensure_batch(_select_sequence(tracks, frame_indices), 3)
        preds["track"] = _to_float_tensor(tracks, device)

    vis = _first_array(data, ("vis", "track_visibility"), prefix)
    if vis is not None:
        vis = _ensure_batch(_select_sequence(vis, frame_indices), 2)
        preds["vis"] = _to_float_tensor(vis, device)

    conf = _first_array(data, ("conf", "track_confidence"), prefix)
    if conf is not None:
        conf = _ensure_batch(_select_sequence(conf, frame_indices), 2)
        preds["conf"] = _to_float_tensor(conf, device)

    if not preds:
        raise ValueError(f"No VGGT prediction tensors found in {npz_path} with prefix={prefix!r}")
    return preds


def load_images_from_npz(
    npz_path: Path,
    device: torch.device,
    *,
    image_key: str = "adv_images",
) -> torch.Tensor | None:
    data = np.load(npz_path)
    if image_key not in data:
        return None
    images = data[image_key].astype(np.float32)
    return torch.from_numpy(images).to(device)


def read_npz_image_names(npz_path: Path) -> list[str] | None:
    data = np.load(npz_path)
    if "image_paths" not in data:
        return None
    return [str(x) for x in data["image_paths"].tolist()]


def compare_prediction_outputs(
    clean: dict[str, torch.Tensor],
    variant: dict[str, torch.Tensor],
    clean_images: torch.Tensor | None,
    variant_images: torch.Tensor | None,
) -> dict[str, float | str]:
    legacy = _legacy()
    if clean_images is not None and variant_images is not None:
        return legacy.compare_predictions(clean, variant, clean_images, variant_images)

    metrics: dict[str, float | str] = {"pixel_metrics": "skipped_missing_images"}
    if "depth" in clean and "depth" in variant:
        metrics["depth_rel_rmse"] = legacy.relative_rmse(variant["depth"], clean["depth"])
    if "world_points" in clean and "world_points" in variant:
        metrics["points_rel_rmse"] = legacy.relative_rmse(variant["world_points"], clean["world_points"])
    if "pose_enc" in clean and "pose_enc" in variant:
        metrics["pose_rel_rmse"] = legacy.relative_rmse(variant["pose_enc"], clean["pose_enc"])
        metrics["translation_rmse"] = float(
            (variant["pose_enc"][..., :3] - clean["pose_enc"][..., :3]).pow(2).mean().sqrt().detach().cpu()
        )
        metrics["fov_rmse"] = float(
            (variant["pose_enc"][..., 7:9] - clean["pose_enc"][..., 7:9]).pow(2).mean().sqrt().detach().cpu()
        )
    return metrics


def evaluate_clean_baseline(
    *,
    model,
    clean_images: torch.Tensor,
    gt_refs: dict[str, torch.Tensor],
    clean_preds: dict[str, torch.Tensor],
    image_size_hw: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    config: EvaluationConfig,
) -> CleanBaseline:
    legacy = _legacy()
    metrics = legacy.vggt_paper_metrics(
        gt_refs,
        clean_preds,
        image_size_hw,
        max_points=config.metric_max_points,
        device=device,
    )
    records = legacy.vggt_paper_records(
        gt_refs,
        clean_preds,
        image_size_hw,
        max_points=config.metric_max_points,
        device=device,
    )

    if config.skip_matching_eval:
        records["tracking_image_matching_pair_errors"]["status"] = "skipped_by_user"
        metrics["tracking_image_matching"] = "skipped_by_user"
    elif model is None:
        records["tracking_image_matching_pair_errors"]["status"] = "skipped_missing_model"
        metrics["tracking_image_matching"] = "skipped_missing_model"
    else:
        matching = legacy.tracking_image_matching_records(
            model=model,
            images=clean_images,
            reference=gt_refs,
            dtype=dtype,
            max_keypoints=config.matching.max_keypoints,
            detection_threshold=config.matching.det_thresh,
            ransac_thresh=config.matching.ransac_thresh,
            min_inliers=config.matching.min_inliers,
            min_visibility=config.matching.min_visibility,
            min_confidence=config.matching.min_confidence,
            max_pairs=config.matching.max_pairs,
        )
        records["tracking_image_matching_pair_errors"] = matching
        legacy.add_tracking_matching_metrics(metrics, matching)

    return CleanBaseline(metrics=metrics, records=records)


def evaluate_variant(
    *,
    scene_dir: Path,
    variant: PredictionVariant,
    clean_baseline: CleanBaseline,
    clean_preds: dict[str, torch.Tensor],
    clean_images: torch.Tensor,
    gt_refs: dict[str, torch.Tensor],
    image_size_hw: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    config: EvaluationConfig,
    model=None,
) -> dict[str, Any]:
    legacy = _legacy()
    variant_gt_metrics = legacy.vggt_paper_metrics(
        gt_refs,
        variant.preds,
        image_size_hw,
        max_points=config.metric_max_points,
        device=device,
    )
    variant_gt_records = legacy.vggt_paper_records(
        gt_refs,
        variant.preds,
        image_size_hw,
        max_points=config.metric_max_points,
        device=device,
    )
    clean_variant_metrics = compare_prediction_outputs(clean_preds, variant.preds, clean_images, variant.images)
    clean_variant_records = legacy.vggt_paper_records(
        clean_preds,
        variant.preds,
        image_size_hw,
        max_points=config.metric_max_points,
        device=device,
    )

    if config.skip_matching_eval:
        variant_gt_records["tracking_image_matching_pair_errors"]["status"] = "skipped_by_user"
        variant_gt_metrics["tracking_image_matching"] = "skipped_by_user"
    elif model is None:
        variant_gt_records["tracking_image_matching_pair_errors"]["status"] = "skipped_missing_model"
        variant_gt_metrics["tracking_image_matching"] = "skipped_missing_model"
    elif variant.images is None:
        variant_gt_records["tracking_image_matching_pair_errors"]["status"] = "skipped_missing_variant_images"
        variant_gt_metrics["tracking_image_matching"] = "skipped_missing_variant_images"
    else:
        matching = legacy.tracking_image_matching_records(
            model=model,
            images=variant.images,
            reference=gt_refs,
            dtype=dtype,
            max_keypoints=config.matching.max_keypoints,
            detection_threshold=config.matching.det_thresh,
            ransac_thresh=config.matching.ransac_thresh,
            min_inliers=config.matching.min_inliers,
            min_visibility=config.matching.min_visibility,
            min_confidence=config.matching.min_confidence,
            max_pairs=config.matching.max_pairs,
        )
        variant_gt_records["tracking_image_matching_pair_errors"] = matching
        legacy.add_tracking_matching_metrics(variant_gt_metrics, matching)

    if "__" in scene_dir.name:
        category, sequence_name = scene_dir.name.split("__", 1)
    else:
        category, sequence_name = scene_dir.name, ""

    summary = {
        "scene": str(scene_dir),
        "category": category,
        "sequence_name": sequence_name,
        "variant": variant.name,
        "variant_npz": str(variant.npz_path) if variant.npz_path is not None else None,
        "attack_type": variant.metadata.get("attack_type", variant.name),
        "steps": variant.metadata.get("steps"),
        "eps": variant.metadata.get("eps"),
        "alpha": variant.metadata.get("alpha"),
        "patch": variant.metadata.get("patch"),
        "evaluation_protocol": legacy.EVALUATION_PROTOCOL,
        "evaluation_config": asdict(config),
        "metrics": {
            "clean_vs_gt": clean_baseline.metrics,
            "variant_vs_gt": variant_gt_metrics,
            "clean_vs_variant": clean_variant_metrics,
            # Compatibility with the original ranking helpers.
            "pgd_vs_gt": variant_gt_metrics,
            "clean_vs_pgd": clean_variant_metrics,
        },
        "eval_records": {
            "clean_vs_gt": clean_baseline.records,
            "variant_vs_gt": variant_gt_records,
            "clean_vs_variant": clean_variant_records,
            "pgd_vs_gt": variant_gt_records,
            "clean_vs_pgd": clean_variant_records,
        },
        "variant_metadata": variant.metadata,
    }
    return summary


def evaluate_variants(
    *,
    model,
    clean_images: torch.Tensor,
    gt_refs: dict[str, torch.Tensor],
    clean_preds: dict[str, torch.Tensor],
    variants: list[PredictionVariant],
    scene_dir: Path,
    image_size_hw: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    config: EvaluationConfig,
) -> tuple[CleanBaseline, list[dict[str, Any]]]:
    clean_baseline = evaluate_clean_baseline(
        model=model,
        clean_images=clean_images,
        gt_refs=gt_refs,
        clean_preds=clean_preds,
        image_size_hw=image_size_hw,
        dtype=dtype,
        device=device,
        config=config,
    )
    summaries = [
        evaluate_variant(
            scene_dir=scene_dir,
            variant=variant,
            clean_baseline=clean_baseline,
            clean_preds=clean_preds,
            clean_images=clean_images,
            gt_refs=gt_refs,
            image_size_hw=image_size_hw,
            dtype=dtype,
            device=device,
            config=config,
            model=model,
        )
        for variant in variants
    ]
    return clean_baseline, summaries


def load_or_forward_clean_predictions(
    *,
    model,
    clean_npz: Path | None,
    clean_images: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    image_size_hw: tuple[int, int],
    frame_indices: np.ndarray | None,
    run_clean_forward: bool,
) -> dict[str, torch.Tensor]:
    if clean_npz is not None and not run_clean_forward:
        return load_predictions_from_npz(clean_npz, device, image_size_hw, frame_indices=frame_indices)
    if model is None:
        raise ValueError("--run_clean_forward needs a loaded model")
    legacy = _legacy()
    with torch.no_grad():
        clean_full = legacy.forward_vggt(model, clean_images, dtype)
    return legacy.detach_predictions(clean_full)


def variant_row(summary: dict[str, Any]) -> dict[str, Any]:
    legacy = _legacy()
    row = legacy.scene_attack_row(summary)
    row["variant"] = summary.get("variant")
    row["variant_npz"] = summary.get("variant_npz")

    metrics = summary.get("metrics", {})
    clean_vs_gt = metrics.get("clean_vs_gt", {})
    variant_vs_gt = metrics.get("variant_vs_gt", metrics.get("pgd_vs_gt", {}))
    clean_vs_variant = metrics.get("clean_vs_variant", metrics.get("clean_vs_pgd", {}))
    row.update(
        {
            "clean_variant_depth_rel_rmse": legacy.finite_float(clean_vs_variant.get("depth_rel_rmse")),
            "clean_variant_points_rel_rmse": legacy.finite_float(clean_vs_variant.get("points_rel_rmse")),
            "clean_variant_pose_rel_rmse": legacy.finite_float(clean_vs_variant.get("pose_rel_rmse")),
            "variant_depth_align_sim3_overall": legacy.finite_float(variant_vs_gt.get("depth_align_sim3_overall")),
            "variant_point_overall": legacy.finite_float(variant_vs_gt.get("point_overall")),
            "variant_camera_auc30": legacy.finite_float(variant_vs_gt.get("camera_auc@30")),
            "depth_align_sim3_overall_delta": legacy.metric_delta(
                variant_vs_gt, clean_vs_gt, "depth_align_sim3_overall", larger_is_worse=True
            ),
            "point_overall_delta": legacy.metric_delta(
                variant_vs_gt, clean_vs_gt, "point_overall", larger_is_worse=True
            ),
            "camera_auc30_drop": legacy.metric_delta(
                variant_vs_gt, clean_vs_gt, "camera_auc@30", larger_is_worse=False
            ),
        }
    )
    return row


def write_variant_tables(summaries: list[dict[str, Any]], out_dir: Path, stem: str = "variant_comparison") -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [variant_row(summary) for summary in summaries]
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(out_dir / f"{stem}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def write_scene_evaluation(
    *,
    out_dir: Path,
    clean_baseline: CleanBaseline,
    variant_summaries: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "clean_baseline.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": clean_baseline.metrics, "eval_records": clean_baseline.records}, f, indent=2)
    with open(out_dir / "variant_summaries.json", "w", encoding="utf-8") as f:
        json.dump(variant_summaries, f, indent=2)
    write_variant_tables(variant_summaries, out_dir)


def aggregate_dataset_metrics_by_variant(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    legacy = _legacy()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        grouped.setdefault(str(summary.get("variant") or "variant"), []).append(summary)
    return {
        variant: legacy.aggregate_dataset_metrics(variant_summaries)
        for variant, variant_summaries in sorted(grouped.items())
    }
