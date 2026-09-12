#!/usr/bin/env python3
"""Validate all-frame universal GT pools, including lazy track reprojection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


VGGT = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VGGT / "scripts"))

from tum_gt_pool import gather_pool, open_pool  # noqa: E402


ROOT = VGGT / "outputs/tum_gt_pool_universal_all_firsthalf"
POOL_PATH = VGGT / "data/tum_universal_all/universal_train_firsthalf_pools.json"
MIN_QUERIES = 20


def main() -> None:
    pools = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260911)
    report = []
    for scene, expected_frames in pools.items():
        pool = open_pool(ROOT, scene)
        meta = pool["meta"]
        arrays = pool["arrays"]
        if meta["frame_indices"] != expected_frames:
            raise RuntimeError(f"{scene}: frame indices differ from frozen protocol")
        if pool["track_storage"] != "lazy_reprojection":
            raise RuntimeError(f"{scene}: expected lazy track storage")
        if any(name in arrays for name in ("tracks", "track_known", "track_visible")):
            raise RuntimeError(f"{scene}: dense track arrays were unexpectedly stored")
        n = len(expected_frames)
        if arrays["depth"].shape[0] != n or arrays["gt_c2w"].shape != (n, 4, 4):
            raise RuntimeError(f"{scene}: wrong leading frame dimensions")
        query_counts = np.asarray(arrays["query_valid"]).sum(axis=1).astype(int)
        if int(query_counts.min()) < MIN_QUERIES:
            raise RuntimeError(f"{scene}: query minimum {int(query_counts.min())}")
        if not np.isfinite(np.asarray(arrays["depth"])).all():
            raise RuntimeError(f"{scene}: non-finite depth")
        if not np.isfinite(np.asarray(arrays["gt_c2w"])).all():
            raise RuntimeError(f"{scene}: non-finite pose")

        checks = [list(range(10))]
        checks.extend(
            sorted(int(value) for value in rng.choice(n, size=10, replace=False))
            for _ in range(3)
        )
        roundtrip = 0.0
        for frames in checks:
            raw = gather_pool(
                pool,
                frames,
                tuple(meta["tensor_hw"]),
                image_names=[meta["image_names"][frame] for frame in frames],
                gt_c2w=np.asarray(arrays["gt_c2w"][frames]),
                min_queries=MIN_QUERIES,
            )
            error = float(np.max(np.abs(raw["tracks"][0] - raw["query_points"])))
            roundtrip = max(roundtrip, error)
            if error > 1e-3:
                raise RuntimeError(f"{scene}: lazy roundtrip {error:.6f}px")
        report.append({
            "scene": scene,
            "frames": n,
            "query_min": int(query_counts.min()),
            "query_max": int(query_counts.max()),
            "lazy_roundtrip_max_px": roundtrip,
            "track_storage": pool["track_storage"],
        })
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("PASS: 8/8 all-frame universal first-half GT pools")


if __name__ == "__main__":
    main()
