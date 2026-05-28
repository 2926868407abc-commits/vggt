#!/usr/bin/env python3
"""Prepare Neural-RGBD sampled sequences as VGGT batch scenes.

The recons_eval mv_recon benchmark uses pre-sampled frame ids stored in
datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf*.json. This script builds one
VGGT scene per Neural-RGBD sequence and exposes only those sampled frames under
scene/images/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def copy_or_link(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Neural-RGBD sampled VGGT scenes.")
    parser.add_argument("--nrgbd_root", required=True, help="Path to recons_eval/data/nrgbd")
    parser.add_argument("--seq_id_map", required=True, help="Path to NRGBD mv_recon seq-id-map JSON")
    parser.add_argument("--out_root", required=True, help="Output root for VGGT scenes")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nrgbd_root = Path(args.nrgbd_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with Path(args.seq_id_map).open("r", encoding="utf-8") as f:
        seq_id_map = json.load(f)

    n_frames = 0
    for seq_name, ids in seq_id_map.items():
        image_out = out_root / seq_name / "images"
        image_out.mkdir(parents=True, exist_ok=True)
        for order, frame_id in enumerate(ids):
            src = nrgbd_root / seq_name / "images" / f"img{frame_id}.png"
            if not src.exists():
                raise FileNotFoundError(src)
            dst = image_out / f"{order:06d}_img{frame_id}.png"
            copy_or_link(src.resolve(), dst, args.copy)
            n_frames += 1

    print(f"Prepared {len(seq_id_map)} VGGT scenes and {n_frames} frames in {out_root}")


if __name__ == "__main__":
    main()
