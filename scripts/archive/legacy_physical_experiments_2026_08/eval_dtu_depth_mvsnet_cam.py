from pathlib import Path
import argparse, json, re
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree

from attack_vggt_new1 import read_ply_xyz, voxel_downsample, unproject_depth_map_to_point_map

def centers_from_extrinsics(ext):
    R = ext[:, :3, :3]
    t = ext[:, :3, 3]
    return -np.matmul(R.transpose(0, 2, 1), t[..., None]).squeeze(-1)

def umeyama(src, dst):
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    ms, md = src.mean(0), dst.mean(0)
    X, Y = src - ms, dst - md
    U, D, Vt = np.linalg.svd((Y.T @ X) / len(src))
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / ((X * X).sum() / len(src))
    t = md - scale * (R @ ms)
    return float(scale), R.astype(np.float32), t.astype(np.float32)

def read_mvsnet_cam(path):
    toks = Path(path).read_text().split()
    i = toks.index("extrinsic") + 1
    ext4 = np.asarray(toks[i:i + 16], dtype=np.float32).reshape(4, 4)
    i = toks.index("intrinsic") + 1
    K = np.asarray(toks[i:i + 9], dtype=np.float32).reshape(3, 3)
    return ext4[:3], K

def view_id_from_name(name):
    m = re.search(r"_(\d{3})_", str(name))
    if not m:
        raise ValueError(f"Cannot parse view id from {name}")
    return int(m.group(1)) - 1

def dtu_eval(pred_points, gt_root, scan_id, downsample=0.2, patch=60.0, max_dist=20.0):
    gt_root = Path(gt_root)
    obs_path = gt_root / "ObsMask" / f"ObsMask{scan_id}_10.mat"
    plane_path = gt_root / "ObsMask" / f"Plane{scan_id}.mat"
    stl_path = gt_root / "Points" / "stl" / f"stl{scan_id:03d}_total.ply"

    pred = pred_points.reshape(-1, 3).astype(np.float32)
    pred = pred[np.isfinite(pred).all(axis=1)]
    pred = voxel_downsample(pred, downsample)

    obs = loadmat(obs_path)
    obs_mask = obs["ObsMask"].astype(bool)
    bb = obs["BB"].astype(np.float32)
    res = float(np.asarray(obs["Res"]).reshape(-1)[0])

    inbound = ((pred >= bb[:1] - patch) & (pred < bb[1:] + patch * 2)).sum(axis=-1) == 3
    pred_in = pred[inbound]
    grid = np.rint((pred_in - bb[:1]) / res).astype(np.int32)
    grid_inbound = ((grid >= 0) & (grid < np.asarray(obs_mask.shape)[None])).sum(axis=-1) == 3
    grid_valid = grid[grid_inbound]
    pred_valid = pred_in[grid_inbound]
    pred_valid = pred_valid[obs_mask[grid_valid[:, 0], grid_valid[:, 1], grid_valid[:, 2]]]

    if len(pred_valid) == 0:
        return {"status": "no_pred_points_inside_dtu_obsmask"}

    stl = read_ply_xyz(stl_path).astype(np.float32)
    if plane_path.exists():
        plane = loadmat(plane_path)["P"].reshape(-1).astype(np.float32)
        stl_h = np.concatenate([stl, np.ones((len(stl), 1), dtype=np.float32)], axis=1)
        stl = stl[(stl_h @ plane) > 0]

    d2s = cKDTree(stl).query(pred_valid, k=1)[0]
    s2d = cKDTree(pred_valid).query(stl, k=1)[0]

    acc = d2s[d2s < max_dist]
    comp = s2d[s2d < max_dist]
    if len(acc) == 0 or len(comp) == 0:
        return {"status": "no_dtu_distances_under_max_dist", "pred_points": int(len(pred_valid))}

    return {
        "accuracy": float(acc.mean()),
        "completeness": float(comp.mean()),
        "overall": float((acc.mean() + comp.mean()) * 0.5),
        "pred_points": int(len(pred_valid)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--cam_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred_root = Path(args.pred_root)
    cam_dir = Path(args.cam_dir)
    all_metrics = {}
    overalls = []

    for npz_path in sorted(pred_root.glob("scan*/vggt_outputs.npz")):
        scan_id = int(re.sub(r"\D", "", npz_path.parent.name))
        d = np.load(npz_path)
        names = [str(x) for x in d["image_paths"].tolist()]

        gt_ext = []
        for name in names:
            vid = view_id_from_name(name)
            ext, _ = read_mvsnet_cam(cam_dir / f"{vid:08d}_cam.txt")
            gt_ext.append(ext)
        gt_ext = np.stack(gt_ext).astype(np.float32)

        pred_ext = d["extrinsic"].astype(np.float32)
        scale, R, t = umeyama(
            centers_from_extrinsics(pred_ext),
            centers_from_extrinsics(gt_ext),
        )

        pred_points = unproject_depth_map_to_point_map(
            d["depth"].astype(np.float32),
            pred_ext,
            d["intrinsic"].astype(np.float32),
        )
        flat = pred_points.reshape(-1, 3)
        aligned = (scale * (R @ flat.T).T + t).reshape(pred_points.shape)

        metrics = dtu_eval(aligned, args.gt_root, scan_id)
        metrics["sim3_scale"] = float(scale)
        metrics["aligned_bbox_min"] = aligned.reshape(-1, 3).min(axis=0).tolist()
        metrics["aligned_bbox_max"] = aligned.reshape(-1, 3).max(axis=0).tolist()

        all_metrics[f"scan{scan_id}"] = metrics
        if "overall" in metrics:
            overalls.append(metrics["overall"])
        print(f"scan{scan_id}:", metrics)

    all_metrics["mean_overall"] = float(np.mean(overalls)) if overalls else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print("saved ->", args.out)

if __name__ == "__main__":
    main()
