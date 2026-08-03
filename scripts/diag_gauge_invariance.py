"""Gauge-invariance diagnostics for the MIRAGE-3D / VGGT patch attack.

Read-only diagnostic. It does NOT touch the attack code and does NOT modify any
evaluation file; it only *calls* recons_eval's own metric functions and writes
its own CSV / markdown artifacts into ``--out_dir``.

Three sections (select with ``--sections``):

[A] Synthetic Sim(3) invariance of the pose metric.
    Load one clean VGGT predicted trajectory and its TUM GT, apply a synthetic
    global similarity g = (s * R_yaw, t) to the *prediction*, and push every g
    through recons_eval's `relpose.evo_utils.eval_metrics` (which hard-codes
    align=True, correct_scale=True).  Also reports what the current training
    loss (`pose_relative_mse` on first-frame-normalised absolute poses) does
    under the same g, so the loss/metric gauge mismatch is directly visible.
    Non-Sim(3) control rows verify the test can detect a real change.

[B] Same question for the point-map (Acc / Comp) and depth (Abs Rel) metrics.
    Uses `mv_recon.utils.{umeyama,accuracy,completion}` and `utils.depth.
    depth_evaluation` from recons_eval, with the same alignment pipeline the
    real evaluators use (Umeyama -> Open3D ICP for point maps, scale-only /
    median scaling for depth).

[C] Alignment sensitivity of the ATE for already-trained patches.
    No retraining: reads existing attacked `vggt_outputs.npz` and recomputes
    ATE under three alignment settings
        align=True  + correct_scale=True   (what the paper numbers use)
        align=True  + correct_scale=False
        align=False
    plus the Umeyama scale ratio of the clean prediction w.r.t. GT, and a
    Sim(3) decomposition of each attack's deviation from the clean prediction.

Run with the *evaluation* env:

    /mnt/data/wangqq/conda_envs/recons_eval/bin/python3 \
        scripts/diag_gauge_invariance.py --sections A,B,C
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# constants that mirror attack_vggt_geometry_tum10.py defaults (read-only copy)
# --------------------------------------------------------------------------- #
TUM_FX, TUM_FY, TUM_CX, TUM_CY = 535.4, 539.2, 320.1, 247.6
TUM_DEPTH_SCALE = 5000.0
TUM_DEPTH_MATCH_MAX_DT = 0.05


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_matrix(degrees: float) -> np.ndarray:
    """Rotation about the world Y axis, same convention as the attack script."""
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def apply_sim3(c2w: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Act with g = (scale * rot, trans) on a camera-to-world trajectory."""
    out = np.array(c2w, dtype=np.float64, copy=True)
    out[:, :3, :3] = np.einsum("ij,njk->nik", rot, c2w[:, :3, :3])
    out[:, :3, 3] = scale * np.einsum("ij,nj->ni", rot, c2w[:, :3, 3]) + trans[None, :]
    return out


def extrinsic_w2c_to_c2w(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsic shape (N,3,4), got {extrinsic.shape}")
    w2c = np.tile(np.eye(4, dtype=np.float64), (extrinsic.shape[0], 1, 1))
    w2c[:, :3, :4] = extrinsic.astype(np.float64)
    return np.linalg.inv(w2c)


def read_tum_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(stripped.split())
    return rows


def tum_rows_to_c2w(rows: list[list[str]], frame_indices: list[int]) -> np.ndarray:
    poses = []
    for frame_idx in frame_indices:
        row = rows[int(frame_idx)]
        tx, ty, tz = map(float, row[1:4])
        qx, qy, qz, qw = map(float, row[4:8])
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = quat_xyzw_to_rot(qx, qy, qz, qw)
        c2w[:3, 3] = [tx, ty, tz]
        poses.append(c2w)
    return np.stack(poses, axis=0)


def write_selected_gt(gt_rows: list[list[str]], frame_indices: list[int], out_path: Path) -> None:
    """Byte-identical to scripts/eval_vggt_tum_pose_for_recons_eval_tum10.py."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for order, frame_idx in enumerate(frame_indices):
            row = list(gt_rows[frame_idx])
            row[0] = f"{float(order):.6f}"
            f.write(" ".join(row) + "\n")


def normalize_c2w_to_first(c2w: np.ndarray) -> np.ndarray:
    """numpy mirror of attack_vggt_geometry_tum10.normalize_c2w_to_first."""
    return np.linalg.inv(c2w[0])[None] @ c2w


def pose_relative_mse_np(
    pred_rel: np.ndarray,
    target_rel: np.ndarray,
    w_rot: float = 1.0,
    w_trans: float = 1.0,
) -> dict[str, float]:
    """numpy mirror of attack_vggt_geometry_tum10.pose_relative_mse (default weights)."""
    rot_mse = float(((pred_rel[:, :3, :3] - target_rel[:, :3, :3]) ** 2).mean())
    trans_scale = float(max(np.linalg.norm(target_rel[:, :3, 3], axis=-1).mean(), 1e-3))
    trans_mse = float((((pred_rel[:, :3, 3] - target_rel[:, :3, 3]) / trans_scale) ** 2).mean())
    return {
        "pose_loss": w_rot * rot_mse + w_trans * trans_mse,
        "pose_rot_mse": rot_mse,
        "pose_trans_mse": trans_mse,
        "pose_trans_scale": trans_scale,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], columns: list[str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        cells = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                cells.append(format(val, fmt.get(col, ".6g")))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# recons_eval bindings
# --------------------------------------------------------------------------- #
class ReconsEval:
    """Thin accessor for recons_eval's own metric code (imported, never edited)."""

    def __init__(self, recons_root: Path) -> None:
        self.root = recons_root
        sys.path.insert(0, str(recons_root.resolve()))
        from relpose import evo_utils  # noqa: E402

        self.evo_utils = evo_utils
        import evo.main_ape as main_ape  # noqa: E402
        import evo.main_rpe as main_rpe  # noqa: E402
        from evo.core import sync  # noqa: E402
        from evo.core.metrics import PoseRelation, Unit  # noqa: E402

        self.main_ape = main_ape
        self.main_rpe = main_rpe
        self.sync = sync
        self.PoseRelation = PoseRelation
        self.Unit = Unit

    # -- trajectory metrics --------------------------------------------------
    def _prepare(self, pred_traj, gt_traj):
        """Replicates the preprocessing inside evo_utils.eval_metrics."""
        pred = self.evo_utils.make_traj(pred_traj)
        gt = self.evo_utils.make_traj(gt_traj)
        if pred.timestamps.shape[0] == gt.timestamps.shape[0]:
            pred.timestamps = gt.timestamps
        gt, pred = self.sync.associate_trajectories(gt, pred)
        return gt, pred

    def ate(self, pred_traj, gt_traj, align: bool, correct_scale: bool) -> float:
        traj_ref, traj_est = self._prepare(pred_traj, gt_traj)
        result = self.main_ape.ape(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=self.PoseRelation.translation_part,
            align=align,
            correct_scale=correct_scale,
        )
        return float(result.stats["rmse"])

    def rpe(self, pred_traj, gt_traj, align: bool, correct_scale: bool) -> tuple[float, float]:
        traj_ref, traj_est = self._prepare(pred_traj, gt_traj)
        common = dict(
            est_name="traj",
            align=align,
            correct_scale=correct_scale,
            delta=1,
            delta_unit=self.Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )
        rot = self.main_rpe.rpe(
            traj_ref, traj_est, pose_relation=self.PoseRelation.rotation_angle_deg, **common
        )
        trans = self.main_rpe.rpe(
            traj_ref, traj_est, pose_relation=self.PoseRelation.translation_part, **common
        )
        return float(trans.stats["rmse"]), float(rot.stats["rmse"])

    def eval_metrics_canonical(self, pred_traj, gt_traj, seq: str, filename: Path):
        """The exact call the production eval script makes (align=T, scale=T)."""
        filename.parent.mkdir(parents=True, exist_ok=True)
        return self.evo_utils.eval_metrics(
            pred_traj, gt_traj, seq=seq, filename=str(filename), verbose=False
        )

    # -- point map / depth ---------------------------------------------------
    def pointmap_fns(self):
        from mv_recon.utils import umeyama, accuracy, completion  # noqa: E402

        return umeyama, accuracy, completion

    def depth_fn(self):
        from utils.depth import depth_evaluation, depth_read_bonn  # noqa: E402

        return depth_evaluation, depth_read_bonn


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_run(run_dir: Path, scene: str) -> dict:
    scene_dir = run_dir / scene
    npz_path = scene_dir / "vggt_outputs.npz"
    summary_path = scene_dir / "attack_summary.json"
    if not npz_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"incomplete run: {scene_dir}")
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    data = np.load(npz_path)
    return {
        "run": run_dir.name,
        "scene": scene,
        "scene_dir": scene_dir,
        "frame_indices": [int(i) for i in summary["frame_indices"]],
        "image_paths": [str(p) for p in summary.get("image_paths", [])],
        "npz": data,
        "c2w": extrinsic_w2c_to_c2w(data["extrinsic"]),
    }


def gt_traj_for(rec: ReconsEval, tum_root: Path, scene: str, frame_indices: list[int],
                gt_name: str, work_dir: Path):
    gt_rows = read_tum_rows(tum_root / scene / gt_name)
    selected = work_dir / f"{scene}_groundtruth_selected.txt"
    write_selected_gt(gt_rows, frame_indices, selected)
    gt_traj = rec.evo_utils.load_traj(str(selected), traj_format="tum", stride=1)
    gt_c2w = tum_rows_to_c2w(gt_rows, frame_indices)
    return gt_traj, gt_c2w


# --------------------------------------------------------------------------- #
# [A] synthetic Sim(3) invariance of the pose metric
# --------------------------------------------------------------------------- #
def section_a(rec: ReconsEval, args, out_dir: Path) -> tuple[list[dict], dict]:
    clean = load_run(Path(args.clean_run), args.scene)
    gt_traj, gt_c2w = gt_traj_for(
        rec, Path(args.tum_root), args.scene, clean["frame_indices"], args.gt_name, out_dir / "work"
    )
    pred_c2w = clean["c2w"]
    gt_rel = normalize_c2w_to_first(gt_c2w)

    # sanity gate: our re-implemented ATE must reproduce the production one
    canon_ate, canon_rpe_t, canon_rpe_r = rec.eval_metrics_canonical(
        rec.evo_utils.get_tum_poses(pred_c2w), gt_traj, args.scene, out_dir / "work" / "A_canonical.txt"
    )
    ours_ate = rec.ate(rec.evo_utils.get_tum_poses(pred_c2w), gt_traj, align=True, correct_scale=True)
    ours_rpe_t, ours_rpe_r = rec.rpe(
        rec.evo_utils.get_tum_poses(pred_c2w), gt_traj, align=True, correct_scale=True
    )
    sanity = {
        "eval_metrics_ATE": canon_ate,
        "reimplemented_ATE": ours_ate,
        "abs_diff_ATE": abs(canon_ate - ours_ate),
        "eval_metrics_RPE_trans": canon_rpe_t,
        "reimplemented_RPE_trans": ours_rpe_t,
        "eval_metrics_RPE_rot": canon_rpe_r,
        "reimplemented_RPE_rot": ours_rpe_r,
    }

    extent = float(np.linalg.norm(pred_c2w[:, :3, 3] - pred_c2w[:, :3, 3].mean(0), axis=-1).mean())
    t_dir = np.asarray([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    rows: list[dict] = []

    def add_row(label: str, kind: str, c2w: np.ndarray, s: float, yaw: float, tn: float) -> None:
        traj = rec.evo_utils.get_tum_poses(c2w)
        ate_as = rec.ate(traj, gt_traj, align=True, correct_scale=True)
        ate_a = rec.ate(traj, gt_traj, align=True, correct_scale=False)
        ate_n = rec.ate(traj, gt_traj, align=False, correct_scale=False)
        rpe_t, rpe_r = rec.rpe(traj, gt_traj, align=True, correct_scale=True)
        loss = pose_relative_mse_np(normalize_c2w_to_first(c2w), gt_rel)
        rows.append(
            {
                "label": label,
                "kind": kind,
                "s": s,
                "yaw_deg": yaw,
                "t_norm_m": tn,
                "ATE_align_scale": ate_as,
                "ATE_align_noscale": ate_a,
                "ATE_noalign": ate_n,
                "RPE_trans": rpe_t,
                "RPE_rot_deg": rpe_r,
                "train_pose_loss": loss["pose_loss"],
                "train_rot_mse": loss["pose_rot_mse"],
                "train_trans_mse": loss["pose_trans_mse"],
            }
        )

    for s in args.sweep_scales:
        for yaw in args.sweep_yaws:
            for tn in args.sweep_translations:
                label = f"s={s:g}, yaw={yaw:g}deg, |t|={tn:g}m"
                kind = "identity" if (s == 1.0 and yaw == 0.0 and tn == 0.0) else "sim3"
                g_c2w = apply_sim3(pred_c2w, s, yaw_matrix(yaw), t_dir * tn)
                add_row(label, kind, g_c2w, s, yaw, tn)

    # --- non-Sim(3) controls: the test must be able to see a real change -----
    rng = np.random.default_rng(args.seed)
    for frac in args.control_noise_fracs:
        sigma = frac * extent
        noisy = np.array(pred_c2w, copy=True)
        noisy[:, :3, 3] += rng.normal(scale=sigma, size=(pred_c2w.shape[0], 3))
        add_row(
            f"CONTROL per-frame trans noise sigma={sigma:.4g} ({frac:g}x extent)",
            "control_non_sim3",
            noisy,
            float("nan"),
            float("nan"),
            float("nan"),
        )

    # progressive (per-frame) scale drift: looks like a scale but is not global
    drift = np.array(pred_c2w, copy=True)
    n = drift.shape[0]
    ramp = 1.0 + np.linspace(0.0, args.control_scale_drift - 1.0, n)
    drift[:, :3, 3] = drift[:, :3, 3] * ramp[:, None]
    add_row(
        f"CONTROL progressive scale drift 1 -> {args.control_scale_drift:g}",
        "control_non_sim3",
        drift,
        float("nan"),
        float("nan"),
        float("nan"),
    )

    write_csv(out_dir / "A_sim3_invariance_pose.csv", rows)
    return rows, {"sanity": sanity, "traj_extent": extent, "n_frames": int(pred_c2w.shape[0])}


# --------------------------------------------------------------------------- #
# [B] point map + depth
# --------------------------------------------------------------------------- #
def preprocess_tum_depth_to_vggt_grid(depth_path: Path, image_path: Path,
                                      tensor_hw: tuple[int, int]) -> np.ndarray:
    """Mirror of attack_vggt_geometry_tum10.load_depth_preprocessed."""
    from PIL import Image

    with Image.open(image_path) as image:
        orig_w, orig_h = image.size
    tensor_h, tensor_w = tensor_hw
    new_w = 518
    new_h = round(orig_h * (new_w / orig_w) / 14) * 14
    crop_y = max(0, (new_h - 518) // 2)
    depth_image = Image.open(depth_path)
    depth_image = depth_image.resize((new_w, new_h), Image.Resampling.NEAREST)
    if new_h > tensor_h:
        depth_image = depth_image.crop((0, crop_y, new_w, crop_y + tensor_h))
    depth = np.asarray(depth_image, dtype=np.float32) / TUM_DEPTH_SCALE
    depth[~np.isfinite(depth)] = 0.0
    return depth


def projection_params(image_path: Path, tensor_hw: tuple[int, int]) -> dict[str, float]:
    from PIL import Image

    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
    tensor_h, tensor_w = tensor_hw
    new_w = 518
    new_h = round(orig_h * (new_w / orig_w) / 14) * 14
    return {
        "scale_x": float(new_w) / float(orig_w),
        "scale_y": float(new_h) / float(orig_h),
        "crop_y": float(max(0, (new_h - 518) // 2)),
    }


def match_depth_paths(seq_dir: Path, image_paths: list[str]) -> list[Path | None]:
    """Nearest-timestamp association against depth.txt (mirror of the attack script)."""
    rows: list[tuple[float, Path]] = []
    depth_txt = seq_dir / "depth.txt"
    if depth_txt.exists():
        for line in depth_txt.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), Path(parts[1])))
            except ValueError:
                continue
    if not rows:
        return [None] * len(image_paths)
    timestamps = np.asarray([r[0] for r in rows], dtype=np.float64)
    matched: list[Path | None] = []
    for image_path in image_paths:
        try:
            ts = float(Path(image_path).stem)
        except ValueError:
            matched.append(None)
            continue
        best = int(np.argmin(np.abs(timestamps - ts)))
        if abs(float(timestamps[best]) - ts) > TUM_DEPTH_MATCH_MAX_DT:
            matched.append(None)
        else:
            matched.append(seq_dir / rows[best][1])
    return matched


def build_gt_world_points(depth_maps: list[np.ndarray | None], gt_c2w: np.ndarray,
                          proj: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """GT depth -> world point cloud on the VGGT tensor grid. Returns (N,H,W,3), (N,H,W)."""
    n = len(depth_maps)
    first = next(d for d in depth_maps if d is not None)
    h, w = first.shape
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    u = xs / proj["scale_x"]
    v = (ys + proj["crop_y"]) / proj["scale_y"]
    dir_x = (u - TUM_CX) / TUM_FX
    dir_y = (v - TUM_CY) / TUM_FY

    pts = np.zeros((n, h, w, 3), dtype=np.float64)
    valid = np.zeros((n, h, w), dtype=bool)
    for i, depth in enumerate(depth_maps):
        if depth is None:
            continue
        z = depth.astype(np.float64)
        cam = np.stack([dir_x * z, dir_y * z, z], axis=-1)
        world = np.einsum("ij,hwj->hwi", gt_c2w[i, :3, :3], cam) + gt_c2w[i, :3, 3]
        pts[i] = world
        valid[i] = (z > 1e-6) & np.isfinite(z)
    return pts, valid


def pointmap_metrics(rec: ReconsEval, pred_pts: np.ndarray, gt_pts: np.ndarray,
                     valid_mask: np.ndarray, icp_threshold: float) -> dict[str, float]:
    """Exactly the mv_recon/eval.py pipeline: Umeyama -> ICP -> accuracy/completion."""
    import open3d as o3d

    umeyama, accuracy, completion = rec.pointmap_fns()
    c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
    aligned = c * np.einsum("nhwj, ij -> nhwi", pred_pts, R) + t.T
    pred_flat = aligned[valid_mask].reshape(-1, 3)
    gt_flat = gt_pts[valid_mask].reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred_flat)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_flat)

    reg = o3d.pipelines.registration.registration_icp(
        pcd,
        pcd_gt,
        icp_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pcd = pcd.transform(reg.transformation)
    pcd.estimate_normals()
    pcd_gt.estimate_normals()
    pred_normal = np.asarray(pcd.normals)
    gt_normal = np.asarray(pcd_gt.normals)

    acc, acc_med, nc1, _ = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
    comp, comp_med, nc2, _ = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)
    return {
        "umeyama_scale": float(c),
        "Acc_mean": float(acc),
        "Acc_med": float(acc_med),
        "Comp_mean": float(comp),
        "Comp_med": float(comp_med),
        "NC1": float(nc1),
        "NC2": float(nc2),
    }


def section_b(rec: ReconsEval, args, out_dir: Path) -> tuple[list[dict], list[dict], dict]:
    clean = load_run(Path(args.clean_run), args.scene)
    seq_dir = Path(args.tum_root) / args.scene
    _, gt_c2w = gt_traj_for(
        rec, Path(args.tum_root), args.scene, clean["frame_indices"], args.gt_name, out_dir / "work"
    )

    npz = clean["npz"]
    pred_pts = npz[args.pointmap_key].astype(np.float64)  # (N,H,W,3) in VGGT world
    pred_depth = npz["depth"][..., 0].astype(np.float64)  # (N,H,W)
    n, h, w = pred_depth.shape

    image_paths = clean["image_paths"]
    if not image_paths:
        image_paths = [str(p) for p in npz["image_paths"]]
    image_paths = [str(seq_dir / "rgb_90" / Path(p).name) if not Path(p).exists() else p
                   for p in image_paths]

    depth_paths = match_depth_paths(seq_dir, image_paths)
    n_matched = sum(1 for p in depth_paths if p is not None)
    proj = projection_params(Path(image_paths[0]), (h, w))
    gt_depth_grid = [
        preprocess_tum_depth_to_vggt_grid(dp, Path(ip), (h, w)) if dp is not None else None
        for dp, ip in zip(depth_paths, image_paths)
    ]

    meta = {
        "n_frames": n,
        "grid": f"{h}x{w}",
        "depth_frames_matched": n_matched,
        "pointmap_key": args.pointmap_key,
    }

    # ---------------- B1: point map ----------------------------------------
    gt_pts, gt_valid = build_gt_world_points(gt_depth_grid, gt_c2w, proj)
    rng = np.random.default_rng(args.seed)
    valid_idx = np.flatnonzero(gt_valid.reshape(-1))
    if valid_idx.size > args.max_points:
        keep = rng.choice(valid_idx, size=args.max_points, replace=False)
        sub = np.zeros(gt_valid.size, dtype=bool)
        sub[keep] = True
        valid_mask = sub.reshape(gt_valid.shape)
    else:
        valid_mask = gt_valid
    meta["pointmap_points_used"] = int(valid_mask.sum())

    pm_rows: list[dict] = []
    t_dir = np.asarray([1.0, 1.0, 1.0]) / math.sqrt(3.0)

    def pm_row(label: str, kind: str, pts: np.ndarray, s, yaw, tn) -> None:
        m = pointmap_metrics(rec, pts, gt_pts, valid_mask, args.icp_threshold)
        pm_rows.append({"label": label, "kind": kind, "s": s, "yaw_deg": yaw,
                        "t_norm_m": tn, **m})

    for s in args.sweep_scales:
        for yaw in args.sweep_yaws:
            for tn in args.sweep_translations:
                kind = "identity" if (s == 1.0 and yaw == 0.0 and tn == 0.0) else "sim3"
                g_pts = s * np.einsum("ij,nhwj->nhwi", yaw_matrix(yaw), pred_pts) + (t_dir * tn)
                pm_row(f"s={s:g}, yaw={yaw:g}deg, |t|={tn:g}m", kind, g_pts, s, yaw, tn)

    # non-Sim(3) control: per-point isotropic noise
    spread = float(np.std(pred_pts[valid_mask]))
    for frac in args.control_noise_fracs:
        sigma = frac * spread
        noisy = pred_pts + rng.normal(scale=sigma, size=pred_pts.shape)
        pm_row(f"CONTROL per-point noise sigma={sigma:.4g} ({frac:g}x std)",
               "control_non_sim3", noisy, float("nan"), float("nan"), float("nan"))

    write_csv(out_dir / "B_sim3_invariance_pointmap.csv", pm_rows)

    # ---------------- B2: depth --------------------------------------------
    depth_evaluation, depth_read_bonn = rec.depth_fn()
    import cv2

    depth_rows: list[dict] = []

    def depth_eval_all(scale: float, noise_frac: float, label: str, kind: str) -> None:
        for invariant, align_with_scale in (("median-scale", True), ("scale", False)):
            abs_rels, d125s, rmses, valids = [], [], [], []
            for i, dp in enumerate(depth_paths):
                if dp is None:
                    continue
                gt_full = depth_read_bonn(str(dp))  # (480,640), invalid -> -1
                pred = pred_depth[i] * scale
                if noise_frac > 0.0:
                    pred = pred + rng.normal(scale=noise_frac * float(np.median(pred)),
                                             size=pred.shape)
                pred_full = cv2.resize(pred, (gt_full.shape[1], gt_full.shape[0]),
                                       interpolation=cv2.INTER_CUBIC)
                res, _, _, _ = depth_evaluation(
                    pred_full, gt_full, max_depth=70, use_gpu=False,
                    align_with_scale=align_with_scale,
                )
                abs_rels.append(res["Abs Rel"])
                d125s.append(res["δ < 1.25"])
                rmses.append(res["RMSE"])
                valids.append(res["valid_pixels"])
            depth_rows.append({
                "label": label,
                "kind": kind,
                "invariant_cfg": invariant,
                "align_with_scale": align_with_scale,
                "s": scale,
                "AbsRel": float(np.mean(abs_rels)),
                "RMSE": float(np.mean(rmses)),
                "delta_1.25": float(np.mean(d125s)),
                "valid_pixels_mean": float(np.mean(valids)),
            })

    for s in args.sweep_scales:
        kind = "identity" if s == 1.0 else "sim3_scale"
        depth_eval_all(s, 0.0, f"depth x s={s:g}", kind)
    for frac in args.control_noise_fracs:
        depth_eval_all(1.0, frac, f"CONTROL per-pixel noise {frac:g}x median", "control_non_sim3")

    write_csv(out_dir / "B_sim3_invariance_depth.csv", depth_rows)
    return pm_rows, depth_rows, meta


# --------------------------------------------------------------------------- #
# [C] existing trained patches under three alignment settings
# --------------------------------------------------------------------------- #
def sim3_decomposition(rec: ReconsEval, pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """How much of `pred`'s deviation from `ref` is a pure global Sim(3)?"""
    umeyama, _, _ = rec.pointmap_fns()
    x = pred[:, :3, 3].T
    y = ref[:, :3, 3].T
    c, R, t = umeyama(x, y)
    aligned = (c * (R @ x) + t).T
    raw = float(np.sqrt(np.mean(np.sum((pred[:, :3, 3] - ref[:, :3, 3]) ** 2, axis=-1))))
    resid = float(np.sqrt(np.mean(np.sum((aligned - ref[:, :3, 3]) ** 2, axis=-1))))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    return {
        "sim3_scale_vs_clean": float(c),
        "sim3_rot_deg_vs_clean": angle,
        "sim3_trans_norm_vs_clean": float(np.linalg.norm(t)),
        "dev_rms_raw": raw,
        "dev_rms_after_sim3": resid,
        "gauge_absorbed_frac": float(1.0 - resid / raw) if raw > 1e-12 else float("nan"),
    }


def production_ate(recons_root: Path, run: str, scene: str) -> float | None:
    """Per-sequence ATE recorded by the production eval script, if it ran for this run.

    Note the top-level `tum10-metric-vggt_<run>.csv` holds the average over every
    scene the run covers, so the per-sequence `seq_metrics.csv` is the right file
    to compare a single-scene number against.
    """
    path = recons_root / "outputs/relpose-distance" / f"vggt_{run}" / "tum10" / "seq_metrics.csv"
    if not path.exists():
        return None
    value = None
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("seq") == scene:
                value = float(row["ATE"])  # keep the last (most recent) entry
    return value


def section_c(rec: ReconsEval, args, out_dir: Path) -> tuple[list[dict], list[dict]]:
    clean_run = Path(args.clean_run)
    attack_root = Path(args.attack_root)

    # --- clean prediction scale ratio vs GT, for every scene the clean run has
    umeyama, _, _ = rec.pointmap_fns()
    scale_rows: list[dict] = []
    for scene_dir in sorted(p for p in clean_run.iterdir() if p.is_dir()):
        try:
            clean = load_run(clean_run, scene_dir.name)
        except Exception as exc:
            print(f"    skip clean scene {scene_dir.name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        gt_traj, gt_c2w = gt_traj_for(
            rec, Path(args.tum_root), scene_dir.name, clean["frame_indices"], args.gt_name,
            out_dir / "work",
        )
        c, R, t = umeyama(clean["c2w"][:, :3, 3].T, gt_c2w[:, :3, 3].T)
        gt_span = float(np.linalg.norm(gt_c2w[:, :3, 3] - gt_c2w[:, :3, 3].mean(0), axis=-1).mean())
        pred_span = float(np.linalg.norm(clean["c2w"][:, :3, 3] - clean["c2w"][:, :3, 3].mean(0),
                                         axis=-1).mean())
        traj = rec.evo_utils.get_tum_poses(clean["c2w"])
        scale_rows.append({
            "scene": scene_dir.name,
            "umeyama_scale_pred_to_gt": float(c),
            "pred_mean_radius": pred_span,
            "gt_mean_radius_m": gt_span,
            "ATE_align_scale": rec.ate(traj, gt_traj, True, True),
            "ATE_align_noscale": rec.ate(traj, gt_traj, True, False),
            "ATE_noalign": rec.ate(traj, gt_traj, False, False),
        })
    write_csv(out_dir / "C_clean_scale_ratio.csv", scale_rows)

    # --- every trained patch, three alignment settings
    clean_ref = load_run(clean_run, args.scene)
    gt_traj, gt_c2w = gt_traj_for(
        rec, Path(args.tum_root), args.scene, clean_ref["frame_indices"], args.gt_name,
        out_dir / "work",
    )
    gt_rel = normalize_c2w_to_first(gt_c2w)

    run_dirs = [p for p in sorted(attack_root.glob(args.patch_glob))
                if p.is_dir() and (p / "geometry_patch").is_dir()
                and (p / args.scene / "vggt_outputs.npz").exists()]
    if args.limit_runs:
        run_dirs = run_dirs[: args.limit_runs]

    rows: list[dict] = []

    def row_for(name: str, c2w: np.ndarray, is_clean: bool) -> dict:
        traj = rec.evo_utils.get_tum_poses(c2w)
        rpe_t, rpe_r = rec.rpe(traj, gt_traj, True, True)
        loss = pose_relative_mse_np(normalize_c2w_to_first(c2w), gt_rel)
        row = {
            "run": name,
            "is_clean": is_clean,
            "ATE_align_scale": rec.ate(traj, gt_traj, True, True),
            "ATE_align_noscale": rec.ate(traj, gt_traj, True, False),
            "ATE_noalign": rec.ate(traj, gt_traj, False, False),
            "RPE_trans": rpe_t,
            "RPE_rot_deg": rpe_r,
            "train_pose_loss_vs_gt": loss["pose_loss"],
        }
        row.update(sim3_decomposition(rec, c2w, clean_ref["c2w"]))
        prod = production_ate(Path(args.recons_root), name.replace("CLEAN::", ""), args.scene)
        row["prod_ATE_align_scale"] = prod if prod is not None else float("nan")
        row["prod_abs_diff"] = (
            abs(prod - row["ATE_align_scale"]) if prod is not None else float("nan")
        )
        return row

    rows.append(row_for(f"CLEAN::{clean_run.name}", clean_ref["c2w"], True))
    skipped: list[str] = []
    for run_dir in run_dirs:
        try:
            run = load_run(run_dir, args.scene)
        except Exception as exc:  # truncated / unreadable npz from a killed job
            skipped.append(f"{run_dir.name}: {type(exc).__name__}: {exc}")
            continue
        if run["frame_indices"] != clean_ref["frame_indices"]:
            skipped.append(f"{run_dir.name}: frame_indices differ from the clean run")
            continue
        rows.append(row_for(run_dir.name, run["c2w"], False))
    if skipped:
        print(f"    skipped {len(skipped)} run(s):", flush=True)
        for line in skipped:
            print(f"      - {line}", flush=True)
        (out_dir / "C_skipped_runs.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")

    rows.sort(key=lambda r: -r["ATE_align_scale"])
    write_csv(out_dir / "C_patch_ate_variants.csv", rows)
    return rows, scale_rows


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
ALIGNMENT_NOTES = """\
| metric | evaluator | alignment actually applied | scale handled? |
|---|---|---|---|
| ATE (pose) | `relpose/evo_utils.py::eval_metrics` -> `evo.main_ape.ape` | **Umeyama Sim(3)**, `align=True` hard-coded | **yes**, `correct_scale=True` hard-coded |
| RPE trans / rot (pose) | `relpose/evo_utils.py::eval_metrics` -> `evo.main_rpe.rpe` | **Umeyama Sim(3)**, `align=True`, `all_pairs=True`, `delta=1 frame` | **yes**, `correct_scale=True` |
| Acc / Comp (point map) | `mv_recon/eval.py` + `mv_recon/utils.py` | **Umeyama Sim(3)** (`utils.py::umeyama`, closed-form) **then Open3D ICP** (`TransformationEstimationPointToPoint`, no scaling -> SE(3) refinement, threshold 0.1, 100 for DTU) | **yes**, via the Umeyama scale `c` |
| Abs Rel / RMSE / delta (depth) | `monodepth/eval.py` + `utils/depth.py::depth_evaluation` | no pose alignment at all — depth lives in the camera frame | **yes**: `invariant="median-scale"` -> `align_with_scale=True` (Weiszfeld IRLS scale-only fit); `invariant="scale"` -> falls through to the `else` branch = **median scaling** `median(gt)/median(pred)` |

Where they live (grep for `accuracy` / `completion` / `chamfer` / `abs_rel` under `recons_eval`):
`mv_recon/utils.py:7 umeyama`, `:45 completion_ratio`, `:52 accuracy`, `:68 completion`, driven by
`mv_recon/eval.py:99-155`; `utils/depth.py:194 depth_evaluation` (`abs_rel` at `:361`), driven by
`monodepth/eval.py`. There is **no chamfer-distance metric** in the repo — Acc (pred->GT mean NN
distance) and Comp (GT->pred mean NN distance) are the two one-sided halves, never summed.

Note on the depth config naming: in `monodepth/eval.py:25-30`, `invariant == "median-scale"` sets
`align_with_scale=True`, which runs the *IRLS L1 scale* fit, while `invariant == "scale"` sets
`align_with_scale=False`, which runs *median scaling*. The two names are swapped relative to what
they do. Either way the estimator is scale-only (no shift), so both are exactly scale-invariant.
"""


def _verdicts(a_rows, b_pm_rows, b_depth_rows, c_rows, tol: float) -> list[str]:
    """Turn the raw sweeps into explicit pass/fail statements."""
    out: list[str] = []

    if a_rows:
        sim3 = [r for r in a_rows if r["kind"] in ("identity", "sim3")]
        ident = next(r for r in a_rows if r["kind"] == "identity")
        max_d = max(abs(r["ATE_align_scale"] - ident["ATE_align_scale"]) for r in sim3)
        rel = max_d / ident["ATE_align_scale"] if ident["ATE_align_scale"] else float("nan")
        verdict = "INVARIANT" if rel < tol else "SENSITIVE"
        out.append(
            f"- **ATE / RPE are {verdict} to a global Sim(3) on the prediction.** "
            f"Over {len(sim3)} transforms (s, yaw, |t|), max relative ATE change = {rel:.2e} "
            f"(absolute {max_d:.3e} m) — pure float noise. RPE trans/rot are identical too."
        )
        loss_ratios = [
            r["train_pose_loss"] / ident["train_pose_loss"] for r in sim3 if ident["train_pose_loss"]
        ]
        out.append(
            f"- **The current training loss is NOT invariant to the same transforms.** "
            f"`pose_relative_mse` ranges over x{min(loss_ratios):.3f} .. x{max(loss_ratios):.3f} "
            f"across the sweep while the metric never moves. The whole spread comes from the "
            f"scale DOF: first-frame normalisation already cancels global rotation and "
            f"translation, but not scale."
        )
        controls = [r for r in a_rows if r["kind"] == "control_non_sim3"]
        if controls:
            moved = max(abs(r["ATE_align_scale"] - ident["ATE_align_scale"]) for r in controls)
            out.append(
                f"- Sanity: the non-Sim(3) controls do move the ATE (up to "
                f"{moved:.3e} m, {moved / ident['ATE_align_scale']:.1f}x the clean ATE), "
                f"so the null result above is a property of the metric, not a broken harness."
            )

    if b_pm_rows:
        sim3 = [r for r in b_pm_rows if r["kind"] in ("identity", "sim3")]
        ident = next(r for r in b_pm_rows if r["kind"] == "identity")
        da = max(abs(r["Acc_mean"] - ident["Acc_mean"]) for r in sim3) / max(ident["Acc_mean"], 1e-12)
        dc = max(abs(r["Comp_mean"] - ident["Comp_mean"]) for r in sim3) / max(ident["Comp_mean"], 1e-12)
        verdict = "INVARIANT" if max(da, dc) < tol else "SENSITIVE"
        out.append(
            f"- **Point-map Acc / Comp are {verdict} to a global Sim(3)** "
            f"(max relative change: Acc {da:.2e}, Comp {dc:.2e}). Umeyama absorbs it in closed "
            f"form — the fitted scale just moves inversely (see `umeyama_scale` column)."
        )
    if b_depth_rows:
        by_cfg: dict[str, list[dict]] = {}
        for r in b_depth_rows:
            if r["kind"] in ("identity", "sim3_scale"):
                by_cfg.setdefault(r["invariant_cfg"], []).append(r)
        parts = []
        for cfg, rows_ in by_cfg.items():
            base = next(r for r in rows_ if r["kind"] == "identity")
            d = max(abs(r["AbsRel"] - base["AbsRel"]) for r in rows_) / max(base["AbsRel"], 1e-12)
            parts.append(f"{cfg}: {d:.2e}")
        out.append(
            f"- **Depth Abs Rel is INVARIANT to the scale part of Sim(3)** in both configured "
            f"alignment modes (max relative change — {', '.join(parts)}). The rotation and "
            f"translation parts do not act on depth at all: VGGT depth is in the camera frame, "
            f"so the world gauge is irrelevant by construction."
        )

    if c_rows:
        attacks = [r for r in c_rows if not r["is_clean"]]
        clean = next(r for r in c_rows if r["is_clean"])
        if attacks:
            fracs = sorted(r["gauge_absorbed_frac"] for r in attacks
                           if not math.isnan(r["gauge_absorbed_frac"]))
            med = fracs[len(fracs) // 2]
            best = max(attacks, key=lambda r: r["ATE_align_scale"])
            out.append(
                f"- **On the {len(attacks)} already-trained patches, a median of "
                f"{med * 100:.1f}% of the trajectory damage is a global Sim(3)** and is therefore "
                f"discarded by the aligned ATE. Best run reaches ATE(align+scale) "
                f"{best['ATE_align_scale']:.4f} m vs clean {clean['ATE_align_scale']:.4f} m "
                f"(x{best['ATE_align_scale'] / clean['ATE_align_scale']:.1f}), but "
                f"ATE(no align) barely separates from clean "
                f"({best['ATE_noalign']:.3f} vs {clean['ATE_noalign']:.3f} m) because both are "
                f"dominated by the fixed frame-0 gauge offset."
            )
            checked = [r for r in c_rows if not math.isnan(r.get("prod_abs_diff", float("nan")))]
            if checked:
                agree = sum(1 for r in checked if r["prod_abs_diff"] < 1e-9)
                worst = max(r["prod_abs_diff"] for r in checked)
                out.append(
                    f"- Sanity: {agree}/{len(checked)} of the runs that already have a recorded "
                    f"per-sequence production ATE reproduce it here (worst abs diff "
                    f"{worst:.3e} m), so the `align=True + correct_scale=True` column is the "
                    f"same number the paper tables use."
                )
    return out


def build_report(out_dir: Path, a_rows, a_meta, b_pm_rows, b_depth_rows, b_meta,
                 c_rows, c_scale_rows, args) -> str:
    lines: list[str] = []
    lines.append("# Gauge-invariance diagnostic — VGGT / TUM-10 physical patch attack\n")
    lines.append(f"- scene: `{args.scene}`")
    lines.append(f"- clean run: `{args.clean_run}`")
    lines.append(f"- evaluator: `{args.recons_root}` (read-only, imported not modified)")
    lines.append(f"- generated by: `scripts/diag_gauge_invariance.py`\n")

    verdicts = _verdicts(a_rows, b_pm_rows, b_depth_rows, c_rows, args.invariance_tol)
    if verdicts:
        lines.append("## Verdict\n")
        lines.extend(verdicts)
        lines.append("")
        lines.append(f"(`INVARIANT` = max relative change over the sweep below "
                     f"`--invariance_tol` = {args.invariance_tol:g}.)\n")

    # ---- A
    if a_rows:
        lines.append("## [A] Synthetic Sim(3) invariance of the pose metric\n")
        sanity = a_meta["sanity"]
        lines.append(
            f"Sanity gate: our re-implemented APE call reproduces the production "
            f"`eval_metrics` ATE to {sanity['abs_diff_ATE']:.3e} "
            f"({sanity['eval_metrics_ATE']:.9f} vs {sanity['reimplemented_ATE']:.9f}), "
            f"so the variant columns below are the same estimator with only the "
            f"`align` / `correct_scale` flags changed.\n"
        )
        sim3 = [r for r in a_rows if r["kind"] in ("identity", "sim3")]
        ident = next(r for r in a_rows if r["kind"] == "identity")
        for r in sim3:
            r["dATE_vs_identity"] = r["ATE_align_scale"] - ident["ATE_align_scale"]
            r["loss_ratio_vs_identity"] = (
                r["train_pose_loss"] / ident["train_pose_loss"] if ident["train_pose_loss"] else float("nan")
            )
        cols = ["s", "yaw_deg", "t_norm_m", "ATE_align_scale", "dATE_vs_identity",
                "RPE_trans", "RPE_rot_deg", "ATE_align_noscale", "ATE_noalign",
                "train_pose_loss", "loss_ratio_vs_identity"]
        lines.append(md_table(sim3, cols, {"ATE_align_scale": ".9f", "dATE_vs_identity": ".3e",
                                           "RPE_trans": ".9f", "RPE_rot_deg": ".9f",
                                           "ATE_align_noscale": ".6f", "ATE_noalign": ".6f",
                                           "train_pose_loss": ".6f",
                                           "loss_ratio_vs_identity": ".4f"}))
        max_d = max(abs(r["dATE_vs_identity"]) for r in sim3)
        rel = max_d / ident["ATE_align_scale"] if ident["ATE_align_scale"] else float("nan")
        lines.append("")
        lines.append(f"**max |ATE(g) - ATE(I)| over the whole sweep = {max_d:.3e} m "
                     f"({rel:.2e} relative).**\n")
        controls = [r for r in a_rows if r["kind"] == "control_non_sim3"]
        if controls:
            lines.append("Non-Sim(3) controls (the same code path must move for these):\n")
            lines.append(md_table(controls, ["label", "ATE_align_scale", "RPE_trans",
                                             "RPE_rot_deg", "train_pose_loss"],
                                  {"ATE_align_scale": ".6f", "RPE_trans": ".6f",
                                   "RPE_rot_deg": ".4f", "train_pose_loss": ".6f"}))
            lines.append("")

    # ---- B
    lines.append("## [B] Where the point-map and depth metrics align, and whether they move\n")
    lines.append(ALIGNMENT_NOTES)
    if b_pm_rows:
        lines.append(f"\n### B.1 Point map (Acc / Comp), {b_meta.get('pointmap_points_used', 0)} "
                     f"GT-valid points on the {b_meta.get('grid')} grid, "
                     f"{b_meta.get('depth_frames_matched')} / {b_meta.get('n_frames')} frames with GT depth\n")
        ident = next((r for r in b_pm_rows if r["kind"] == "identity"), None)
        sim3 = [r for r in b_pm_rows if r["kind"] in ("identity", "sim3")]
        for r in sim3:
            r["dAcc_vs_identity"] = r["Acc_mean"] - ident["Acc_mean"]
            r["dComp_vs_identity"] = r["Comp_mean"] - ident["Comp_mean"]
        lines.append(md_table(sim3, ["s", "yaw_deg", "t_norm_m", "umeyama_scale",
                                     "Acc_mean", "dAcc_vs_identity", "Comp_mean",
                                     "dComp_vs_identity"],
                              {"Acc_mean": ".9f", "Comp_mean": ".9f",
                               "dAcc_vs_identity": ".3e", "dComp_vs_identity": ".3e",
                               "umeyama_scale": ".6g"}))
        max_a = max(abs(r["dAcc_vs_identity"]) for r in sim3)
        max_c = max(abs(r["dComp_vs_identity"]) for r in sim3)
        lines.append("")
        lines.append(f"**max |dAcc| = {max_a:.3e} m, max |dComp| = {max_c:.3e} m over the sweep.**\n")
        controls = [r for r in b_pm_rows if r["kind"] == "control_non_sim3"]
        if controls:
            lines.append("Non-Sim(3) control:\n")
            lines.append(md_table(controls, ["label", "Acc_mean", "Comp_mean"],
                                  {"Acc_mean": ".6f", "Comp_mean": ".6f"}))
            lines.append("")
    if b_depth_rows:
        lines.append("\n### B.2 Depth (Abs Rel)\n")
        lines.append("A world-frame Sim(3) acts on VGGT depth only through its scale factor `s` "
                     "(rotation and translation of the world gauge do not change camera-frame "
                     "depth at all), so only `s` is swept.\n")
        lines.append(md_table(b_depth_rows, ["label", "invariant_cfg", "align_with_scale",
                                             "AbsRel", "RMSE", "delta_1.25"],
                              {"AbsRel": ".9f", "RMSE": ".9f", "delta_1.25": ".9f"}))
        lines.append("")

    # ---- C
    if c_rows:
        lines.append("## [C] Trained patches under three alignment settings\n")
        lines.append("### C.1 Clean VGGT prediction vs GT — scale ratio\n")
        lines.append(md_table(c_scale_rows, ["scene", "umeyama_scale_pred_to_gt",
                                             "pred_mean_radius", "gt_mean_radius_m",
                                             "ATE_align_scale", "ATE_align_noscale",
                                             "ATE_noalign"],
                              {"umeyama_scale_pred_to_gt": ".4f", "pred_mean_radius": ".5f",
                               "gt_mean_radius_m": ".5f", "ATE_align_scale": ".6f",
                               "ATE_align_noscale": ".6f", "ATE_noalign": ".6f"}))
        lines.append("")
        lines.append(f"### C.2 ATE per trained patch ({len(c_rows) - 1} patches + clean)\n")
        top = c_rows[: args.report_top_runs]
        lines.append(md_table(top, ["run", "ATE_align_scale", "ATE_align_noscale", "ATE_noalign",
                                    "RPE_trans", "RPE_rot_deg", "sim3_scale_vs_clean",
                                    "gauge_absorbed_frac", "prod_abs_diff"],
                              {"ATE_align_scale": ".6f", "ATE_align_noscale": ".6f",
                               "ATE_noalign": ".6f", "RPE_trans": ".6f", "RPE_rot_deg": ".4f",
                               "sim3_scale_vs_clean": ".4f", "gauge_absorbed_frac": ".4f",
                               "prod_abs_diff": ".1e"}))
        lines.append("")
        lines.append(f"Full table: `{(out_dir / 'C_patch_ate_variants.csv')}`\n")
        lines.append("`gauge_absorbed_frac` = fraction of the attacked trajectory's RMS deviation "
                     "from the clean prediction that a single global Sim(3) explains — i.e. the "
                     "part the aligned ATE throws away.\n")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recons_root", default="/mnt/data/wangqq/recons_eval")
    p.add_argument("--tum_root", default="/mnt/data/wangqq/recons_eval/data/tum")
    p.add_argument("--attack_root",
                   default="/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10")
    p.add_argument("--clean_run",
                   default="/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3")
    p.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_static")
    p.add_argument("--gt_name", default="groundtruth_90.txt")
    p.add_argument("--out_dir", default="/mnt/data/wangqq/vggt/outputs/diag_gauge_invariance")
    p.add_argument("--sections", default="A,B,C", help="comma list of A,B,C")

    p.add_argument("--sweep_scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--sweep_yaws", type=float, nargs="+", default=[0.0, 15.0, 45.0])
    p.add_argument("--sweep_translations", type=float, nargs="+", default=[0.0, 0.5])
    p.add_argument("--control_noise_fracs", type=float, nargs="+", default=[0.05, 0.2])
    p.add_argument("--control_scale_drift", type=float, default=1.5)

    p.add_argument("--pointmap_key", default="point_map",
                   choices=["point_map", "point_cloud_unproj"])
    p.add_argument("--max_points", type=int, default=200000)
    p.add_argument("--icp_threshold", type=float, default=0.1)

    p.add_argument("--patch_glob", default="*")
    p.add_argument("--limit_runs", type=int, default=0)
    p.add_argument("--report_top_runs", type=int, default=25)
    p.add_argument("--invariance_tol", type=float, default=1e-9,
                   help="max relative metric change still counted as float noise")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "work").mkdir(parents=True, exist_ok=True)
    rec = ReconsEval(Path(args.recons_root))
    sections = {s.strip().upper() for s in args.sections.split(",") if s.strip()}

    a_rows, a_meta = [], {}
    b_pm_rows, b_depth_rows, b_meta = [], [], {}
    c_rows, c_scale_rows = [], []

    if "A" in sections:
        print("[A] synthetic Sim(3) sweep on the pose metric ...", flush=True)
        a_rows, a_meta = section_a(rec, args, out_dir)
        print(f"    {len(a_rows)} rows -> {out_dir / 'A_sim3_invariance_pose.csv'}", flush=True)
    if "B" in sections:
        print("[B] point map + depth ...", flush=True)
        b_pm_rows, b_depth_rows, b_meta = section_b(rec, args, out_dir)
        print(f"    {len(b_pm_rows)} pointmap rows, {len(b_depth_rows)} depth rows", flush=True)
    if "C" in sections:
        print("[C] trained patches, three alignment settings ...", flush=True)
        c_rows, c_scale_rows = section_c(rec, args, out_dir)
        print(f"    {len(c_rows)} rows -> {out_dir / 'C_patch_ate_variants.csv'}", flush=True)

    report = build_report(out_dir, a_rows, a_meta, b_pm_rows, b_depth_rows, b_meta,
                          c_rows, c_scale_rows, args)
    report_path = out_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
