"""Synthesise point-map and tracking ground truth for TUM from GT depth + GT pose.

Why this exists: the evaluation currently has ground truth only for camera pose. The
point-map head is scored against a GT cloud built the same way (already implemented
in diag_gauge_invariance), and the track head is scored against VGGT's own *clean
prediction* -- which can only say "it moved", never "it is wrong". Reprojection GT
fixes that: unproject a query with GT depth, move it with GT poses, project into
every other frame. That is the standard reprojection-GT track protocol.

Two things this deliberately does NOT paper over:

  occlusion  a reprojected point can land on a surface that is nearer in the target
             frame. Those are marked invisible by comparing the reprojected depth
             against the target frame's own GT depth, rather than assuming every
             in-bounds point is trackable.

  depth holes TUM's structured-light depth is missing on dark, glossy and distant
             surfaces. Queries without valid depth are dropped, and targets whose
             GT depth is missing are marked unknown (not visible, not occluded), so
             they can be excluded from metrics instead of silently counted as errors.

Geometry conventions are imported from the code that builds the point-map GT, so the
tracks land on the same tensor grid as VGGT's predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG))
sys.path.insert(0, str(VG / "scripts"))

from diag_gauge_invariance import (  # noqa: E402
    TUM_CX, TUM_CY, TUM_FX, TUM_FY,
    build_gt_world_points, match_depth_paths,
    preprocess_tum_depth_to_vggt_grid, projection_params, tum_rows_to_c2w,
    read_tum_rows,
)

TUM_ROOT = Path("/mnt/data/wangqq/recons_eval/data/tum")
CLEAN_ROOT = VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"
# build_for_scene reads frame indices from a clean run, so a frame subset needs
# its own clean run rather than the default ten-frame one.


def project_world_to_pixels(world: np.ndarray, c2w: np.ndarray,
                            proj: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """World points -> (pixel xy on the tensor grid, depth along the camera axis).

    Inverse of the unprojection in build_gt_world_points, so a point unprojected
    from pixel p in frame i and projected back into frame i returns to p.
    """
    w2c = np.linalg.inv(c2w)
    cam = world @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = TUM_FX * cam[:, 0] / safe + TUM_CX
    v = TUM_FY * cam[:, 1] / safe + TUM_CY
    xs = u * proj["scale_x"]
    ys = v * proj["scale_y"] - proj["crop_y"]
    return np.stack([xs, ys], axis=-1), z


def sample_depth(depth: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour depth lookup; returns (depth, in-bounds mask).

    Nearest rather than bilinear on purpose: interpolating across a depth
    discontinuity invents a surface that is not there, which would turn a genuine
    occlusion boundary into a soft ramp and let occluded points pass the test.
    """
    h, w = depth.shape
    xi = np.rint(xy[:, 0]).astype(np.int64)
    yi = np.rint(xy[:, 1]).astype(np.int64)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(len(xy), dtype=np.float64)
    out[inside] = depth[yi[inside], xi[inside]]
    return out, inside


def build_for_scene(scene: str, rows: int, cols: int, margin: float,
                    occl_tol: float, out_dir: Path,
                    clean_root: Path | None = None) -> dict:
    summary_path = (clean_root or CLEAN_ROOT) / scene / "attack_summary.json"
    if not summary_path.exists():
        return {"scene": scene, "status": "缺少 clean run"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    image_paths = [str(p) for p in summary["image_paths"]]
    frame_indices = [int(i) for i in summary["frame_indices"]]
    n = len(image_paths)

    seq_dir = TUM_ROOT / scene
    gt_c2w = tum_rows_to_c2w(read_tum_rows(seq_dir / "groundtruth_90.txt"), frame_indices)

    depth_paths = match_depth_paths(seq_dir, image_paths)
    if any(p is None for p in depth_paths):
        missing = [i for i, p in enumerate(depth_paths) if p is None]
        return {"scene": scene, "status": f"深度关联失败于帧 {missing}"}

    # tensor grid comes from the clean prediction, so GT and prediction share it
    pred = np.load(CLEAN_ROOT / scene / "vggt_outputs.npz")
    tensor_hw = (int(pred["depth"].shape[1]), int(pred["depth"].shape[2]))
    proj = projection_params(Path(image_paths[0]), tensor_hw)
    depths = [preprocess_tum_depth_to_vggt_grid(p, Path(ip), tensor_hw)
              for p, ip in zip(depth_paths, image_paths)]

    # ---- point-map GT (same routine the pointmap metrics already use)
    pts, pts_valid = build_gt_world_points(depths, gt_c2w, proj)

    # ---- query points: an interior grid in frame 0, keeping only valid depth
    h, w = tensor_hw
    ys = np.linspace(margin * h, (1 - margin) * h, rows)
    xs = np.linspace(margin * w, (1 - margin) * w, cols)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    query = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    q_depth, q_inside = sample_depth(depths[0], query)
    keep = q_inside & (q_depth > 1e-6)
    query = query[keep]
    if len(query) == 0:
        return {"scene": scene, "status": "第 0 帧无有效深度的 query 点"}

    # unproject the queries with frame 0's GT depth
    qz, _ = sample_depth(depths[0], query)
    u = query[:, 0] / proj["scale_x"]
    v = (query[:, 1] + proj["crop_y"]) / proj["scale_y"]
    cam0 = np.stack([(u - TUM_CX) / TUM_FX * qz,
                     (v - TUM_CY) / TUM_FY * qz, qz], axis=-1)
    world = cam0 @ gt_c2w[0][:3, :3].T + gt_c2w[0][:3, 3]

    m = len(query)
    tracks = np.zeros((n, m, 2), dtype=np.float32)
    visible = np.zeros((n, m), dtype=bool)
    known = np.zeros((n, m), dtype=bool)   # False where the target GT depth is missing
    for j in range(n):
        xy, z = project_world_to_pixels(world, gt_c2w[j], proj)
        tracks[j] = xy
        scene_z, inside = sample_depth(depths[j], xy)
        has_depth = inside & (scene_z > 1e-6)
        in_front = z > 1e-6
        # occluded when the target frame sees something meaningfully nearer
        not_occluded = z <= scene_z + occl_tol
        visible[j] = has_depth & in_front & not_occluded
        known[j] = has_depth & in_front

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{scene}_gt.npz",
        scene=scene, frame_indices=np.asarray(frame_indices, dtype=np.int64),
        tensor_hw=np.asarray(tensor_hw, dtype=np.int64),
        gt_c2w=gt_c2w, point_map=pts.astype(np.float32), point_valid=pts_valid,
        query_points=query.astype(np.float32),
        tracks=tracks, track_visible=visible, track_known=known,
        occl_tol=occl_tol,
    )
    # frame 0 must round-trip to its own query pixels
    round_trip = float(np.abs(tracks[0] - query).max())
    return {
        "scene": scene, "status": "ok", "frames": n,
        "queries": int(m), "grid": f"{rows}x{cols}",
        "point_valid": float(pts_valid.mean()),
        "vis_mean": float(visible.mean()),
        "vis_per_frame": [float(x) for x in visible.mean(axis=1)],
        "known_mean": float(known.mean()),
        "roundtrip_px": round_trip,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="", help="逗号分隔；默认所有有 GT 的序列")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--occl_tol", type=float, default=0.05,
                    help="重投影深度比目标帧自身深度深多少米才算被遮挡")
    ap.add_argument("--out_dir", default=str(VG / "outputs/tum_gt_point_track"))
    ap.add_argument("--clean_root", default=str(CLEAN_ROOT),
                    help="读取帧号的 clean run 根目录；帧子集需要指向该子集自己的 clean run")
    cli = ap.parse_args()

    if cli.scenes:
        scenes = [s.strip() for s in cli.scenes.split(",") if s.strip()]
    else:
        scenes = sorted(d.name for d in TUM_ROOT.iterdir()
                        if (d / "groundtruth_90.txt").exists() and (d / "depth.txt").exists())

    out_dir = Path(cli.out_dir)
    results = []
    for scene in scenes:
        r = build_for_scene(scene, cli.rows, cli.cols, cli.margin,
                            cli.occl_tol, out_dir, Path(cli.clean_root))
        results.append(r)
        if r["status"] != "ok":
            print(f"{scene:<46} {r['status']}")
            continue
        print(f"{scene:<46} query {r['queries']:>4}  "
              f"深度有效 {r['point_valid']:.1%}  "
              f"可见率 {r['vis_mean']:.1%}  "
              f"回投误差 {r['roundtrip_px']:.3f}px")

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{len(ok)}/{len(results)} 个序列完成 -> {out_dir}")
    if ok:
        worst = max(r["roundtrip_px"] for r in ok)
        print(f"最大回投误差 {worst:.4f} px（第 0 帧应当回到 query 像素，"
              f"这是投影约定的自检）")
        print("\n逐帧可见率（第 0 帧必为 100%）")
        for r in ok:
            print(f"  {r['scene'][:44]:<46} "
                  + " ".join(f"{v:.0%}" for v in r["vis_per_frame"]))
    (out_dir / "build_summary.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
