"""Run clean VGGT inference on the uniform TUM-10 frame manifest.

This keeps the clean protocol aligned with TUM-10 attacked outputs:
the exact frame indices are read from tum10_frame_manifest.json and saved in
attack_summary.json so the TUM-10 evaluator can reuse the same code path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack_vggt_new1 import detach_predictions, find_images, forward_vggt, load_model, save_official_style_npz, set_random_seeds
from attack_vggt_vla_style import load_frame_manifest
from vggt.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum_root", required=True)
    parser.add_argument("--scene_pattern", default="rgbd_dataset_freiburg3_*")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--frame_manifest", required=True)
    parser.add_argument("--ckpt", default="facebook/VGGT-1B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    manifest = load_frame_manifest(args.frame_manifest)
    tum_root = Path(args.tum_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model = load_model(args, device)
    model.eval()

    summaries = []
    for seq_dir in sorted(path for path in tum_root.glob(args.scene_pattern) if path.is_dir()):
        if seq_dir.name not in manifest:
            continue
        out_dir = output_root / seq_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and (out_dir / "vggt_outputs.npz").exists() and (out_dir / "attack_summary.json").exists():
            continue

        all_images = find_images(seq_dir)
        frame_indices = np.asarray(manifest[seq_dir.name], dtype=int)
        image_paths = [all_images[int(idx)] for idx in frame_indices]
        image_names = [Path(path).name for path in image_paths]

        started = time.time()
        images = load_and_preprocess_images(image_paths).to(device)
        with torch.no_grad():
            preds = detach_predictions(forward_vggt(model, images, dtype))
        image_hw = tuple(int(value) for value in images.shape[-2:])
        save_official_style_npz(out_dir / "vggt_outputs.npz", preds, image_names, image_hw)

        summary = {
            "scene": str(seq_dir),
            "dataset": "tum-dynamics-10frame-clean",
            "mode": "clean_vggt_tum10_uniform",
            "n_frames": len(image_paths),
            "image_paths": [str(path) for path in image_paths],
            "frame_indices": frame_indices.astype(int).tolist(),
            "frame_manifest": args.frame_manifest,
            "ckpt": args.ckpt,
            "elapsed_seconds": time.time() - started,
            "outputs": {"clean_vggt_outputs": "vggt_outputs.npz"},
        }
        with (out_dir / "attack_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        summaries.append(summary)
        print(f"[clean] {seq_dir.name}: {len(image_paths)} frames -> {out_dir / 'vggt_outputs.npz'}")

    with (output_root / "clean_batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"[done] generated {len(summaries)} clean TUM-10 outputs in {output_root}")


if __name__ == "__main__":
    main()
