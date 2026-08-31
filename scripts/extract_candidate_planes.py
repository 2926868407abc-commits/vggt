"""Stage 1 of the auto-placement plan: extract candidate carrier planes.

Answers only "which 3-D surfaces could hold a patch" -- position and area come
later. Per section 4.2 of the plan: fuse the ten depth frames into one world cloud,
RANSAC out several planes, merge duplicates, build a local frame per plane, measure
the usable region, and reject candidates that are too small, too thinly supported,
or visible in too few frames.

Two things the plan is explicit about that are easy to get wrong:

  the plane must not straddle an object boundary (section 4.4). RANSAC will happily
  fit one plane through a desk and the wall behind it if they are near-coplanar, so
  inliers are split into connected components in the plane's own local coordinates
  and each component becomes its own candidate.

  visibility is per-frame and includes occlusion (section 4.2 step 8), not just
  whether the region falls inside the image. A plane behind a person is not usable
  even though it projects in bounds.

Reuses the GT world cloud already built by build_tum_gt_point_track.py, so the
geometry convention matches the point-map ground truth exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import (  # noqa: E402
    TUM_CX, TUM_CY, TUM_FX, TUM_FY,
    match_depth_paths, preprocess_tum_depth_to_vggt_grid, projection_params,
)

TUM = Path("/mnt/data/wangqq/recons_eval/data/tum")
GT_DIR = VG / "outputs/tum_gt_point_track"
CLEAN = VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"


def voxel_downsample(pts: np.ndarray, size: float) -> np.ndarray:
    keys = np.floor(pts / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def fit_plane(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = pts.mean(0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    return c, vh[2] / max(np.linalg.norm(vh[2]), 1e-12)


def estimate_normals(pts: np.ndarray, radius: float, max_nb: int, rng) -> np.ndarray:
    """Per-point normal from a local PCA, used to group before fitting.

    Without this, greedy RANSAC ranks candidates by raw inlier count and the floor
    wins every round, taking near-coplanar surfaces with it.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    idx = tree.query_ball_point(pts, radius)
    out = np.zeros_like(pts)
    for i, nb in enumerate(idx):
        if len(nb) < 6:
            out[i] = (0.0, 0.0, 1.0)
            continue
        if len(nb) > max_nb:
            nb = list(rng.choice(nb, max_nb, replace=False))
        q = pts[nb]
        _, _, vh = np.linalg.svd(q - q.mean(0), full_matrices=False)
        out[i] = vh[2]
    # sign is arbitrary; fold to a half-sphere so opposite-facing walls group together
    flip = out[:, 2] < 0
    out[flip] = -out[flip]
    return out


def normal_groups(normals: np.ndarray, ang_deg: float) -> list[np.ndarray]:
    """Greedy clustering of normals; returns index arrays."""
    cos_t = np.cos(np.radians(ang_deg))
    unassigned = np.ones(len(normals), dtype=bool)
    groups = []
    order = np.arange(len(normals))
    while unassigned.any():
        seed = order[unassigned][0]
        sim = np.abs(normals @ normals[seed])
        sel = unassigned & (sim > cos_t)
        groups.append(np.where(sel)[0])
        unassigned &= ~sel
    return groups


def ransac_planes(pts: np.ndarray, n_planes: int, thresh: float,
                  min_inliers: int, iters: int, rng,
                  normals: np.ndarray | None = None,
                  group_angle: float = 20.0,
                  cell: float = 0.05, min_cells: int = 40) -> list[dict]:
    """RANSAC within each normal group, splitting components before consuming points.

    Splitting first matters: a plane through the monitor and the partition behind it
    would otherwise remove both surfaces' points in one round, and the monitor would
    never appear as its own candidate.
    """
    if normals is None:
        groups = [np.arange(len(pts))]
    else:
        groups = [g for g in normal_groups(normals, group_angle)
                  if len(g) >= min_inliers // 4]
    found = []
    for gi in groups:
        remaining = pts[gi]
        for _ in range(n_planes):
            if len(remaining) < min_inliers // 4:
                break
            best_mask, best_n = None, 0
            for _ in range(iters):
                smp = remaining[rng.choice(len(remaining), 3, replace=False)]
                nrm = np.cross(smp[1] - smp[0], smp[2] - smp[0])
                ln = np.linalg.norm(nrm)
                if ln < 1e-9:
                    continue
                nrm = nrm / ln
                mask = np.abs((remaining - smp[0]) @ nrm) < thresh
                if mask.sum() > best_n:
                    best_n, best_mask = int(mask.sum()), mask
            if best_mask is None or best_n < min_inliers // 4:
                break
            c, nrm = fit_plane(remaining[best_mask])
            mask = np.abs((remaining - c) @ nrm) < thresh
            inl = remaining[mask]
            u, v = local_frame(c, nrm, inl)
            rel = inl - c
            uv = np.stack([rel @ u, rel @ v], -1)
            labels, ncomp = connected_components(uv, cell, min_cells)
            for ci in range(ncomp):
                sub = inl[labels == ci]
                if len(sub) < min_inliers // 4:
                    continue
                cc, nn = fit_plane(sub)
                found.append({"centre": cc, "normal": nn, "points": sub})
            remaining = remaining[~mask]
    return found


def merge_planes(planes: list[dict], ang_deg: float, dist: float) -> list[dict]:
    out: list[dict] = []
    for p in planes:
        hit = None
        for q in out:
            cos = abs(float(p["normal"] @ q["normal"]))
            gap = abs(float((p["centre"] - q["centre"]) @ q["normal"]))
            if cos > np.cos(np.radians(ang_deg)) and gap < dist:
                hit = q
                break
        if hit is None:
            out.append(dict(p))
        else:
            pts = np.vstack([hit["points"], p["points"]])
            c, n = fit_plane(pts)
            hit.update({"points": pts, "centre": c, "normal": n})
    return out


def local_frame(centre: np.ndarray, normal: np.ndarray, pts: np.ndarray):
    """u along the inliers' dominant in-plane direction, so the local box is tight."""
    rel = pts - centre
    proj = rel - np.outer(rel @ normal, normal)
    _, _, vh = np.linalg.svd(proj, full_matrices=False)
    u = vh[0] / max(np.linalg.norm(vh[0]), 1e-12)
    v = np.cross(normal, u)
    return u, v / max(np.linalg.norm(v), 1e-12)


def connected_components(uv: np.ndarray, cell: float, min_cells: int):
    """Split inliers into connected blobs on a grid, so one RANSAC plane spanning two
    physical surfaces becomes two candidates instead of one bridged region."""
    keys = np.floor(uv / cell).astype(np.int64)
    occupied = {tuple(k): i for i, k in enumerate(map(tuple, keys))}
    seen, comps = set(), []
    for k in occupied:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    nb = (cur[0] + du, cur[1] + dv)
                    if nb in occupied and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(comp) >= min_cells:
            comps.append(set(comp))
    labels = np.full(len(uv), -1, dtype=np.int64)
    for ci, comp in enumerate(comps):
        for i, k in enumerate(map(tuple, keys)):
            if k in comp:
                labels[i] = ci
    return labels, len(comps)


def project(world: np.ndarray, c2w: np.ndarray, proj: dict):
    w2c = np.linalg.inv(c2w)
    cam = world @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = TUM_FX * cam[:, 0] / safe + TUM_CX
    v = TUM_FY * cam[:, 1] / safe + TUM_CY
    return np.stack([u * proj["scale_x"], v * proj["scale_y"] - proj["crop_y"]], -1), z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    ap.add_argument("--tum_root", type=Path, default=TUM)
    ap.add_argument("--gt_dir", type=Path, default=GT_DIR)
    ap.add_argument("--clean_root", type=Path, default=CLEAN)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--ransac_thresh", type=float, default=0.02)
    ap.add_argument("--ransac_planes", type=int, default=8)
    ap.add_argument("--ransac_iters", type=int, default=400)
    ap.add_argument("--min_inliers", type=int, default=800)
    ap.add_argument("--merge_angle", type=float, default=12.0)
    ap.add_argument("--merge_dist", type=float, default=0.05)
    ap.add_argument("--cell", type=float, default=0.05)
    ap.add_argument("--min_cells", type=int, default=40)
    ap.add_argument("--min_area", type=float, default=0.10)
    ap.add_argument("--min_visible_frames", type=int, default=4)
    ap.add_argument("--occl_tol", type=float, default=0.05)
    ap.add_argument("--group_normals", type=int, default=1,
                    help="1 = RANSAC within normal groups; 0 = the old greedy global fit")
    ap.add_argument("--group_angle", type=float, default=20.0)
    ap.add_argument("--normal_radius", type=float, default=0.06)
    ap.add_argument("--normal_max_nb", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=str(VG / "outputs/candidate_planes"))
    cli = ap.parse_args()

    rng = np.random.default_rng(cli.seed)
    scene = cli.scene
    gt = np.load(cli.gt_dir / f"{scene}_gt.npz", allow_pickle=True)
    world = gt["point_map"][gt["point_valid"]]
    gt_c2w = gt["gt_c2w"]
    tensor_hw = tuple(int(x) for x in gt["tensor_hw"])

    summary = json.loads(
        (cli.clean_root / scene / "attack_summary.json").read_text(encoding="utf-8")
    )
    image_paths = [str(p) for p in summary["image_paths"]]
    proj = projection_params(Path(image_paths[0]), tensor_hw)
    dpaths = match_depth_paths(cli.tum_root / scene, image_paths)
    depths = [preprocess_tum_depth_to_vggt_grid(p, Path(ip), tensor_hw)
              for p, ip in zip(dpaths, image_paths)]

    print(f"{scene}\n  有效三维点 {len(world):,}")
    cloud = voxel_downsample(world, cli.voxel)
    print(f"  体素降采样({cli.voxel} m) -> {len(cloud):,}")

    normals = None
    if cli.group_normals:
        normals = estimate_normals(cloud, cli.normal_radius, cli.normal_max_nb, rng)
        print(f"  法向估计完成，分组阈值 {cli.group_angle}°")
    planes = ransac_planes(cloud, cli.ransac_planes, cli.ransac_thresh,
                           cli.min_inliers, cli.ransac_iters, rng,
                           normals=normals, group_angle=cli.group_angle,
                           cell=cli.cell, min_cells=cli.min_cells)
    print(f"  RANSAC 找到 {len(planes)} 个平面")
    planes = merge_planes(planes, cli.merge_angle, cli.merge_dist)
    print(f"  合并近似重复 -> {len(planes)} 个")

    cands = []
    for pi, p in enumerate(planes):
        u, v = local_frame(p["centre"], p["normal"], p["points"])
        rel = p["points"] - p["centre"]
        uv = np.stack([rel @ u, rel @ v], -1)
        labels, ncomp = connected_components(uv, cli.cell, cli.min_cells)
        for ci in range(ncomp):
            sel = labels == ci
            if sel.sum() < cli.min_inliers // 4:
                continue
            sub_uv, sub_pts = uv[sel], p["points"][sel]
            lo, hi = sub_uv.min(0), sub_uv.max(0)
            extent = hi - lo
            # filled area, not the bounding box: an L-shaped blob should not claim
            # the rectangle around it
            cells = np.unique(np.floor(sub_uv / cli.cell).astype(np.int64), axis=0)
            area = float(len(cells) * cli.cell ** 2)
            if area < cli.min_area:
                continue
            centre = sub_pts.mean(0)

            vis_frac, occl_frac = [], []
            for j in range(len(gt_c2w)):
                xy, z = project(sub_pts, gt_c2w[j], proj)
                h, w = tensor_hw
                xi = np.rint(xy[:, 0]).astype(np.int64)
                yi = np.rint(xy[:, 1]).astype(np.int64)
                inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (z > 1e-6)
                if not inside.any():
                    vis_frac.append(0.0), occl_frac.append(0.0)
                    continue
                sd = np.zeros(len(xy))
                sd[inside] = depths[j][yi[inside], xi[inside]]
                has = inside & (sd > 1e-6)
                occluded = has & (z > sd + cli.occl_tol)
                vis_frac.append(float((has & ~occluded).mean()))
                occl_frac.append(float(occluded.sum() / max(has.sum(), 1)))
            nvis = int(sum(1 for f in vis_frac if f > 0.05))
            if nvis < cli.min_visible_frames:
                continue

            cands.append({
                "plane_id": pi, "component": ci,
                "centre": centre.tolist(), "normal": p["normal"].tolist(),
                "u": u.tolist(), "v": v.tolist(),
                "uv_min": lo.tolist(), "uv_max": hi.tolist(),
                "extent_m": extent.tolist(), "area_m2": area,
                "n_points": int(sel.sum()),
                "visible_frames": nvis,
                "visible_ratio_mean": float(np.mean(vis_frac)),
                "occlusion_ratio_mean": float(np.mean(occl_frac)),
                "visible_per_frame": vis_frac,
            })

    cands.sort(key=lambda c: (-c["visible_frames"], -c["area_m2"]))
    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{scene}_planes.npz",
                        scene=scene, tensor_hw=np.asarray(tensor_hw),
                        gt_c2w=gt_c2w, candidates=json.dumps(cands))
    (out / f"{scene}_planes.json").write_text(
        json.dumps({"scene": scene, "n_candidates": len(cands), "candidates": cands},
                   indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n合格候选 {len(cands)} 个")
    print(f"{'#':>3}{'面积m²':>9}{'尺寸m':>14}{'点数':>8}{'可见帧':>7}"
          f"{'可见率':>8}{'遮挡率':>8}")
    print("-" * 60)
    for i, c in enumerate(cands):
        print(f"{i:>3}{c['area_m2']:>9.3f}"
              f"{c['extent_m'][0]:>7.2f}x{c['extent_m'][1]:<6.2f}"
              f"{c['n_points']:>8}{c['visible_frames']:>7}"
              f"{c['visible_ratio_mean']:>8.1%}{c['occlusion_ratio_mean']:>8.1%}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
