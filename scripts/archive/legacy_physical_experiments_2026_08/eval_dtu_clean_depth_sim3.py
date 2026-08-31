from pathlib import Path
import re, json, argparse
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from attack_vggt_new1 import read_ply_xyz, voxel_downsample, unproject_depth_map_to_point_map

def umeyama(src, dst):
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    cov = (Y.T @ X) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    Rot = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / ((X * X).sum() / len(src))
    trans = mu_d - scale * (Rot @ mu_s)
    return float(scale), Rot.astype(np.float32), trans.astype(np.float32)

def centers_from_extrinsics(ext):
    Rm = ext[:, :3, :3]
    t = ext[:, :3, 3]
    return -np.matmul(Rm.transpose(0,2,1), t[...,None]).squeeze(-1)

def load_left_gt_for_images(cal_mat, image_names):
    d = loadmat(cal_mat)
    fc = d["fc"].reshape(-1).astype(np.float32)
    cc = d["cc"].reshape(-1).astype(np.float32)

    extrinsics, intrinsics = [], []
    for name in image_names:
        m = re.search(r"_(\d{3})_", str(name))
        if not m:
            raise ValueError(f"cannot parse view id from {name}")
        vid = int(m.group(1))

        omc = d[f"omc_{vid}"].reshape(3)
        Tc = d[f"Tc_{vid}"].reshape(3)
        Rm = R.from_rotvec(omc).as_matrix().astype(np.float32)
        ext = np.concatenate([Rm, Tc[:,None].astype(np.float32)], axis=1)
        extrinsics.append(ext)

        K = np.array([[fc[0], 0, cc[0]], [0, fc[1], cc[1]], [0, 0, 1]], dtype=np.float32)
        intrinsics.append(K)

    return np.stack(extrinsics), np.stack(intrinsics)

def dtu_eval(pred_points, gt_root, scan_id, downsample=0.2, patch=60.0, max_dist=20.0):
    gt_root = Path(gt_root)
    obs_path = gt_root / "ObsMask" / f"ObsMask{scan_id}_10.mat"
    plane_path = gt_root / "ObsMask" / f"Plane{scan_id}.mat"
    stl_path = gt_root / "Points" / "stl" / f"stl{scan_id:03d}_total.ply"

    pred = pred_points.reshape(-1,3).astype(np.float32)
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
    pred_valid = pred_valid[obs_mask[grid_valid[:,0], grid_valid[:,1], grid_valid[:,2]]]
    if len(pred_valid) == 0:
        return {"status": "no_pred_points_inside_dtu_obsmask"}

    stl = read_ply_xyz(stl_path).astype(np.float32)
    if plane_path.exists():
        plane = loadmat(plane_path)["P"].reshape(-1).astype(np.float32)
        stl_h = np.concatenate([stl, np.ones((len(stl),1), np.float32)], axis=1)
        stl = stl[(stl_h @ plane) > 0]

    d2s = cKDTree(stl).query(pred_valid, k=1, workers=-1)[0]
    s2d = cKDTree(pred_valid).query(stl, k=1, workers=-1)[0]
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
    ap.add_argument("--clean_root", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--cal_mat", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    all_metrics = {}
    vals = []

    for scene in sorted(clean_root.glob("scan*/vggt_outputs.npz")):
        scan_id = int(re.sub(r"\D", "", scene.parent.name))
        d = np.load(scene)
        names = [str(x) for x in d["image_paths"].tolist()]

        pred_ext = d["extrinsic"].astype(np.float32)
        gt_ext, _ = load_left_gt_for_images(args.cal_mat, names)

        pred_centers = centers_from_extrinsics(pred_ext)
        gt_centers = centers_from_extrinsics(gt_ext)
        s, Rot, t = umeyama(pred_centers, gt_centers)

        pred_points = unproject_depth_map_to_point_map(
            d["depth"].astype(np.float32),
            pred_ext,
            d["intrinsic"].astype(np.float32),
        )
        pred_points = (s * (Rot @ pred_points.reshape(-1,3).T).T + t).reshape(pred_points.shape)

        metrics = dtu_eval(pred_points, args.gt_root, scan_id)
        metrics["sim3_scale"] = s
        all_metrics[f"scan{scan_id}"] = metrics
        if "overall" in metrics:
            vals.append(metrics["overall"])
        print(f"scan{scan_id}:", metrics)

    all_metrics["mean_overall"] = float(np.mean(vals)) if vals else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print("saved ->", args.out)

if __name__ == "__main__":
    main()
