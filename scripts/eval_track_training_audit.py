#!/usr/bin/env python3
"""Evaluate formal reprojection-track metrics on a fixed training audit.

This diagnostic deliberately reports training-frame performance.  It reuses the
same GT construction, VGGT forward path, EPE filtering and visibility F1 used by
``eval_universal_four_tasks.py``; it does not train or update the patch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import attack_vggt_geometry_tum10 as attack  # noqa: E402
from eval_stitched_pose import build_attack_args  # noqa: E402
from eval_universal_four_tasks import (  # noqa: E402
    build_window_gt,
    extract_predictions,
    track_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--audit_plan", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--quad_file", required=True)
    parser.add_argument(
        "--tum_root", default="/mnt/data/wangqq/recons_eval/data/tum_all_sync"
    )
    parser.add_argument("--gt_name", default="groundtruth.txt")
    parser.add_argument("--ckpt", default=str(REPO_ROOT / "checkpoints/VGGT-1B"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visibility_depth_margin", type=float, default=0.03)
    parser.add_argument("--query_rows", type=int, default=10)
    parser.add_argument("--query_cols", type=int, default=20)
    parser.add_argument("--query_margin", type=float, default=0.08)
    parser.add_argument("--occl_tol", type=float, default=0.08)
    parser.add_argument("--min_track_points", type=int, default=20)
    return parser.parse_args()


def aggregate(rows: list[dict], condition: str) -> dict[str, float | int]:
    tracks = [row[condition] for row in rows]
    weighted = [
        (float(track["epe_mean"]), int(track["points"]))
        for track in tracks
        if np.isfinite(float(track["epe_mean"])) and int(track["points"]) > 0
    ]
    points = sum(weight for _, weight in weighted)
    epe = sum(value * weight for value, weight in weighted) / max(points, 1)
    tp = sum(int(track["tp"]) for track in tracks)
    fp = sum(int(track["fp"]) for track in tracks)
    fn = sum(int(track["fn"]) for track in tracks)
    return {
        "track_epe_mean": float(epe),
        "track_vis_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "track_points": int(points),
        "track_frames": int(sum(int(track["frames"]) for track in tracks)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )
    render_args = build_attack_args(attack, args)
    model = attack.load_model(render_args, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    texture = attack.load_texture(args.texture, device, render_args)
    intrinsics = np.asarray(
        [
            [render_args.fx, 0.0, render_args.cx],
            [0.0, render_args.fy, render_args.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    audit_path = Path(args.audit_plan).resolve()
    entries = json.loads(audit_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for index, entry in enumerate(entries):
        scene_dir = (
            Path(entry["scene"])
            if entry.get("scene")
            else Path(args.tum_root) / entry["base_scene"]
        )
        manifest_path = Path(entry["manifest"])
        frames_by_scene = attack.load_frame_manifest(manifest_path)
        frames = [int(frame) for frame in frames_by_scene[scene_dir.name]]
        item = attack.load_tum_sequence(
            scene_dir,
            frames,
            args.gt_name,
            intrinsics,
            int(texture.shape[-1]),
            render_args,
            device,
            model=model,
            dtype=dtype,
        )
        gt = build_window_gt(
            item,
            args.query_rows,
            args.query_cols,
            args.query_margin,
            args.occl_tol,
        )
        adversarial = attack.apply_geometry_patch(
            item["images"],
            texture,
            item["grids"],
            item["masks"],
            render_args,
            training=False,
        ).detach()
        clean_pred = extract_predictions(
            model,
            item["images"],
            gt["query_points"],
            device,
            dtype,
            tuple(item["tensor_hw"]),
        )
        attack_pred = extract_predictions(
            model,
            adversarial,
            gt["query_points"],
            device,
            dtype,
            tuple(item["tensor_hw"]),
        )
        clean = track_stats(
            clean_pred["track"], clean_pred["track_vis"], gt, args.min_track_points
        )
        attacked = track_stats(
            attack_pred["track"],
            attack_pred["track_vis"],
            gt,
            args.min_track_points,
        )
        row = {
            "name": entry["name"],
            "scene": scene_dir.name,
            "frames": frames,
            "contains_training_frames": True,
            "coverage": float(item["masks"].float().mean().cpu()),
            "query_points": int(len(gt["query_points"])),
            "gt_roundtrip_max_px": float(gt["roundtrip_max_px"]),
            "clean": clean,
            "attack": attacked,
        }
        rows.append(row)
        print(
            f"[audit] {index + 1}/{len(entries)} {entry['name']} "
            f"EPE={clean['epe_mean']:.4f}->{attacked['epe_mean']:.4f} "
            f"visF1={clean['vis_f1']:.4f}->{attacked['vis_f1']:.4f}",
            flush=True,
        )
        del item, gt, adversarial, clean_pred, attack_pred
        torch.cuda.empty_cache()

    clean = aggregate(rows, "clean")
    attacked = aggregate(rows, "attack")
    texture_path = Path(args.texture).resolve()
    summary = {
        "protocol": "fixed_training_audit_only_not_heldout",
        "audit_plan": str(audit_path),
        "audit_plan_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "texture": str(texture_path),
        "texture_sha256": hashlib.sha256(texture_path.read_bytes()).hexdigest(),
        "n_groups": len(rows),
        "clean": clean,
        "attack": attacked,
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "track_training_audit.json").write_text(
        json.dumps({"summary": summary, "groups": rows}, indent=2),
        encoding="utf-8",
    )
    with (output_root / "track_training_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "track_epe_mean",
                "track_vis_f1",
                "track_points",
                "track_frames",
            ],
        )
        writer.writeheader()
        for condition, values in (("clean", clean), ("attack", attacked)):
            writer.writerow(
                {"condition": condition, **{key: values[key] for key in writer.fieldnames[1:]}}
            )
    print(
        "[summary] "
        f"EPE={clean['track_epe_mean']:.4f}->{attacked['track_epe_mean']:.4f}; "
        f"visF1={clean['track_vis_f1']:.4f}->{attacked['track_vis_f1']:.4f}; "
        f"{output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
