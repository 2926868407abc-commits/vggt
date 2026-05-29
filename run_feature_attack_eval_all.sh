#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
RECONS_ROOT="${RECONS_ROOT:-/mnt/data/wangqq/recons_eval}"
VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"
RECONS_PY="${RECONS_PY:-/mnt/data/wangqq/conda_envs/recons_eval/bin/python3}"
CKPT="${CKPT:-$VGGT_ROOT/checkpoints/VGGT-1B}"

STEPS="${STEPS:-10}"
FEATURE_LAYER="${FEATURE_LAYER:-aggregator_final}"
GLOBAL_EPS="${GLOBAL_EPS:-0.03137255}"
GLOBAL_ALPHA="${GLOBAL_ALPHA:-0.00392157}"
PATCH_LR="${PATCH_LR:-0.00392157}"
PATCH_SIZE="${PATCH_SIZE:-96}"
FORCE_ATTACK="${FORCE_ATTACK:-0}"

NYU_SCENES="${NYU_SCENES:-$VGGT_ROOT/data/nyu_v2_recons_eval_scenes}"
BONN_SCENES="${BONN_SCENES:-$VGGT_ROOT/data/bonn_monodepth_scenes}"
TUM_ROOT="${TUM_ROOT:-$RECONS_ROOT/data/tum}"
NRGBD_SCENES="${NRGBD_SCENES:-$VGGT_ROOT/data/nrgbd_sparse_mv_recon_scenes}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing directory: $1" >&2
    exit 1
  fi
}

run_attack() {
  local out_dir="$1"
  shift
  if [[ "$FORCE_ATTACK" != "1" && -f "$out_dir/attack_batch_summary.json" ]]; then
    log "skip existing attack: $out_dir"
    return
  fi
  log "run attack -> $out_dir"
  mkdir -p "$out_dir"
  (
    cd "$VGGT_ROOT"
    "$VGGT_PY" attack_vggt_new1.py "$@" \
      --output_dir "$out_dir" \
      --ckpt "$CKPT" \
      --steps "$STEPS" \
      --feature_layer "$FEATURE_LAYER"
  )
}

prepare_tum_images_link() {
  log "prepare TUM images links"
  if ! find "$TUM_ROOT" -maxdepth 2 -type d -name rgb_90 | grep -q .; then
    (
      cd "$RECONS_ROOT"
      "$RECONS_PY" datasets/preprocess/prepare_tum.py
    )
  fi
  for seq_dir in "$TUM_ROOT"/rgbd_dataset_freiburg3_*; do
    if [[ -d "$seq_dir/rgb_90" ]]; then
      ln -sfn rgb_90 "$seq_dir/images"
    fi
  done
}

prepare_bonn_scenes_if_needed() {
  if compgen -G "$BONN_SCENES/rgbd_bonn_*__*" >/dev/null; then
    return
  fi
  log "prepare Bonn one-frame VGGT scenes -> $BONN_SCENES"
  BONN_ROOT="$RECONS_ROOT/data/bonn" BONN_SCENES="$BONN_SCENES" "$RECONS_PY" - <<'PY'
import json
import os
from pathlib import Path

bonn_root = Path(os.environ["BONN_ROOT"])
out_root = Path(os.environ["BONN_SCENES"])
out_root.mkdir(parents=True, exist_ok=True)
if not bonn_root.exists():
    raise SystemExit(f"Missing Bonn root: {bonn_root}")

n = 0
for seq_dir in sorted(bonn_root.glob("rgbd_bonn_*")):
    image_dir = seq_dir / "rgb_110"
    if not image_dir.is_dir():
        continue
    for frame_path in sorted(image_dir.glob("*.png")):
        scene_name = f"{seq_dir.name}__{frame_path.stem}"
        scene_dir = out_root / scene_name
        dst = scene_dir / "images" / frame_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(frame_path.resolve())
        with (scene_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump({
                "dataset": "bonn",
                "mode": "monodepth",
                "sequence": seq_dir.name,
                "frame": frame_path.name,
                "source_image": str(frame_path),
            }, f, indent=2)
        n += 1
if n == 0:
    raise SystemExit(f"No Bonn rgb_110 frames found under {bonn_root}")
print(f"Prepared {n} Bonn one-frame scenes in {out_root}")
PY
}

prepare_nrgbd_scenes_if_needed() {
  if compgen -G "$NRGBD_SCENES/*" >/dev/null; then
    return
  fi
  log "prepare Neural-RGBD sparse VGGT scenes -> $NRGBD_SCENES"
  NRGBD_ROOT="$RECONS_ROOT/data/nrgbd" \
  NRGBD_SCENES="$NRGBD_SCENES" \
  SEQ_MAP="$RECONS_ROOT/datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf500.json" \
  "$RECONS_PY" - <<'PY'
import json
import os
from pathlib import Path

nrgbd_root = Path(os.environ["NRGBD_ROOT"])
out_root = Path(os.environ["NRGBD_SCENES"])
seq_map_path = Path(os.environ["SEQ_MAP"])
out_root.mkdir(parents=True, exist_ok=True)

with seq_map_path.open("r", encoding="utf-8") as f:
    seq_id_map = json.load(f)

n = 0
for seq_name, ids in seq_id_map.items():
    image_out = out_root / seq_name / "images"
    image_out.mkdir(parents=True, exist_ok=True)
    for order, frame_id in enumerate(ids):
        src = nrgbd_root / seq_name / "images" / f"img{frame_id}.png"
        if not src.exists():
            raise FileNotFoundError(src)
        dst = image_out / f"{order:06d}_img{frame_id}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        n += 1
print(f"Prepared {len(seq_id_map)} Neural-RGBD scenes and {n} frames in {out_root}")
PY
}

convert_nyu_predictions() {
  log "convert NYU attack depth predictions"
  RECONS_ROOT="$RECONS_ROOT" VGGT_ROOT="$VGGT_ROOT" "$RECONS_PY" - <<'PY'
import os
from pathlib import Path
import numpy as np

recons = Path(os.environ["RECONS_ROOT"])
vggt = Path(os.environ["VGGT_ROOT"])
jobs = [
    (vggt / "outputs_attack/nyu_v2_feature_global_l3",
     recons / "outputs/monodepth/vggt_nyu_v2_feature_global_l3/nyu-v2"),
    (vggt / "outputs_attack/nyu_v2_feature_patch_adam_l3",
     recons / "outputs/monodepth/vggt_nyu_v2_feature_patch_adam_l3/nyu-v2"),
]
for src_root, dst_root in jobs:
    dst_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for scene_dir in sorted(src_root.glob("nyu_*")):
        npz_path = scene_dir / "vggt_outputs.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path) as data:
            depth = np.squeeze(data["depth"]).astype(np.float32)
        np.save(dst_root / f"{scene_dir.name}.npy", depth)
        n += 1
    print(f"{src_root.name}: converted {n} NYU predictions -> {dst_root}")
PY
}

convert_bonn_predictions() {
  log "convert Bonn attack depth predictions"
  RECONS_ROOT="$RECONS_ROOT" VGGT_ROOT="$VGGT_ROOT" "$RECONS_PY" - <<'PY'
import os
from pathlib import Path
import numpy as np

recons = Path(os.environ["RECONS_ROOT"])
vggt = Path(os.environ["VGGT_ROOT"])
jobs = [
    (vggt / "outputs_attack/bonn_feature_global_l3",
     recons / "outputs/monodepth/vggt_bonn_feature_global_l3/bonn"),
    (vggt / "outputs_attack/bonn_feature_patch_adam_l3",
     recons / "outputs/monodepth/vggt_bonn_feature_patch_adam_l3/bonn"),
]
for src_root, dst_root in jobs:
    dst_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for scene_dir in sorted(src_root.glob("rgbd_bonn_*__*")):
        npz_path = scene_dir / "vggt_outputs.npz"
        if not npz_path.exists():
            continue
        seq, frame = scene_dir.name.rsplit("__", 1)
        out_dir = dst_root / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        with np.load(npz_path) as data:
            depth = np.squeeze(data["depth"]).astype(np.float32)
        np.save(out_dir / f"{frame}.npy", depth)
        n += 1
    print(f"{src_root.name}: converted {n} Bonn predictions -> {dst_root}")
PY
}

eval_tum_pose() {
  log "evaluate TUM camera pose attacks"
  RECONS_ROOT="$RECONS_ROOT" VGGT_ROOT="$VGGT_ROOT" "$RECONS_PY" - <<'PY'
import csv
import os
import sys
from pathlib import Path

import numpy as np

recons = Path(os.environ["RECONS_ROOT"])
vggt = Path(os.environ["VGGT_ROOT"])
sys.path.insert(0, str(recons.resolve()))

from relpose.evo_utils import calculate_averages, eval_metrics, get_tum_poses, load_traj, save_tum_poses

def extrinsic_w2c_to_c2w(extrinsic):
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsic shape (N,3,4), got {extrinsic.shape}")
    w2c = np.tile(np.eye(4, dtype=np.float64), (extrinsic.shape[0], 1, 1))
    w2c[:, :3, :4] = extrinsic.astype(np.float64)
    return np.linalg.inv(w2c)

jobs = [
    (vggt / "outputs_attack/tum_feature_global_l3", "vggt_tum_feature_global_l3"),
    (vggt / "outputs_attack/tum_feature_patch_adam_l3", "vggt_tum_feature_patch_adam_l3"),
]
data_root = recons / "data/tum"
for vggt_root, model_name in jobs:
    output_root = recons / "outputs/relpose-distance" / model_name / "tum"
    metric_csv = recons / "outputs/relpose-distance" / f"tum-metric-{model_name}.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    metric_csv.parent.mkdir(parents=True, exist_ok=True)
    seq_metrics_csv = output_root / "seq_metrics.csv"
    if seq_metrics_csv.exists():
        seq_metrics_csv.unlink()

    results = []
    scene_dirs = sorted(p for p in vggt_root.glob("rgbd_dataset_freiburg3_*") if p.is_dir())
    if not scene_dirs:
        raise SystemExit(f"No TUM outputs found under {vggt_root}")
    for seq_dir in scene_dirs:
        seq = seq_dir.name
        npz_path = seq_dir / "vggt_outputs.npz"
        gt_path = data_root / seq / "groundtruth_90.txt"
        with np.load(npz_path) as data:
            c2w = extrinsic_w2c_to_c2w(data["extrinsic"])
        seq_out = output_root / seq
        seq_out.mkdir(parents=True, exist_ok=True)
        np.save(seq_out / "pred_poses.npy", c2w[:, :3, :4].astype(np.float32))
        pred_traj = get_tum_poses(c2w)
        save_tum_poses(pred_traj, seq_out / "pred_traj.txt")
        gt_traj = load_traj(str(gt_path), traj_format="tum", stride=1)
        ate, rpe_trans, rpe_rot = eval_metrics(
            pred_traj,
            gt_traj,
            seq=seq,
            filename=str(seq_out / "eval_metric.txt"),
            verbose=False,
        )
        results.append((seq, ate, rpe_trans, rpe_rot))
        row = {
            "model": model_name,
            "dataset": "tum",
            "seq": seq,
            "ATE": ate,
            "RPE trans": rpe_trans,
            "RPE rot": rpe_rot,
        }
        with seq_metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(f"{model_name} {seq}: ATE={ate:.6f}, RPE trans={rpe_trans:.6f}, RPE rot={rpe_rot:.6f}")

    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)
    metrics = {
        "model": model_name,
        "ATE": avg_ate,
        "RPE trans": avg_rpe_trans,
        "RPE rot": avg_rpe_rot,
    }
    with metric_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Saved {metric_csv}")
PY
}

eval_nrgbd_pointmap() {
  log "evaluate Neural-RGBD sparse point-map attacks"
  RECONS_ROOT="$RECONS_ROOT" VGGT_ROOT="$VGGT_ROOT" "$RECONS_PY" - <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

recons = Path(os.environ["RECONS_ROOT"])
vggt = Path(os.environ["VGGT_ROOT"])
sys.path.insert(0, str(recons.resolve()))

from datasets.nrgbd import NRGBD
from models.vggt.utils.geometry import unproject_depth_map_to_point_map
from mv_recon.utils import accuracy, completion, umeyama

def write_csv_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

def resize_depth(depth, target_hw):
    depth = np.squeeze(depth)
    if depth.ndim == 2:
        depth = depth[None]
    if depth.shape[-2:] == target_hw:
        return depth.astype(np.float32)
    depth_t = torch.from_numpy(depth.astype(np.float32))[:, None]
    depth_t = F.interpolate(depth_t, target_hw, mode="bilinear", align_corners=False, antialias=True)
    return depth_t[:, 0].cpu().numpy().astype(np.float32)

def scale_intrinsics(intrinsic, source_hw, target_hw):
    if source_hw == target_hw:
        return intrinsic.astype(np.float32)
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    scaled = intrinsic.astype(np.float32).copy()
    scaled[:, 0, :] *= target_w / source_w
    scaled[:, 1, :] *= target_h / source_h
    return scaled

def load_depth_unproject_points(npz_path, target_hw):
    with np.load(npz_path) as data:
        raw_depth = np.asarray(data["depth"])
        extrinsic = np.asarray(data["extrinsic"]).astype(np.float32)
        intrinsic = np.asarray(data["intrinsic"]).astype(np.float32)
    squeezed = np.squeeze(raw_depth)
    if squeezed.ndim == 2:
        source_hw = squeezed.shape
    elif squeezed.ndim == 3:
        source_hw = squeezed.shape[-2:]
    elif squeezed.ndim == 4:
        source_hw = squeezed.shape[1:3]
    else:
        raise ValueError(f"Unexpected depth shape {raw_depth.shape} in {npz_path}")
    depth = resize_depth(raw_depth, target_hw)
    intrinsic = scale_intrinsics(intrinsic, source_hw, target_hw)
    return unproject_depth_map_to_point_map(depth[..., None], extrinsic, intrinsic).astype(np.float32)

dataset_name = "NRGBD-sparse"
dataset = NRGBD(
    NRGBD_DIR=str(recons / "data/nrgbd"),
    load_img_size=518,
    cache_file=str(recons / "data/dataset_cache/nrgbd_mv_recon_cache.npy"),
)
with (recons / "datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf500.json").open("r", encoding="utf-8") as f:
    seq_id_map = json.load(f)

metric_names = [
    "Acc-mean", "Acc-med", "Comp-mean", "Comp-med",
    "NC-mean", "NC-med", "NC1-mean", "NC1-med", "NC2-mean", "NC2-med",
]
jobs = [
    (vggt / "outputs_attack/nrgbd_sparse_feature_global_l3", "vggt_nrgbd_sparse_feature_global_l3"),
    (vggt / "outputs_attack/nrgbd_sparse_feature_patch_adam_l3", "vggt_nrgbd_sparse_feature_patch_adam_l3"),
]

for vggt_root, model_name in jobs:
    out_root = recons / "outputs/mv_recon" / model_name / dataset_name
    metric_csv = recons / "outputs/mv_recon" / f"{dataset_name}-metric-{model_name}.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    metric_csv.parent.mkdir(parents=True, exist_ok=True)
    all_samples_csv = out_root / "_all_samples.csv"
    if all_samples_csv.exists():
        all_samples_csv.unlink()
    sums = {name: 0.0 for name in metric_names}

    for seq_name, ids in seq_id_map.items():
        data = dataset.get_data(sequence_name=seq_name, ids=ids)
        gt_pts = data["pointclouds"]
        valid_mask = data["valid_mask"]
        images = data["images"]
        npz_path = vggt_root / seq_name / "vggt_outputs.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        pred_pts = load_depth_unproject_points(npz_path, target_hw=gt_pts.shape[1:3])
        if pred_pts.shape != gt_pts.shape:
            raise ValueError(f"{seq_name}: pred {pred_pts.shape} != gt {gt_pts.shape}")

        colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)
        c, r, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
        pred_pts = c * np.einsum("nhwj,ij->nhwi", pred_pts, r) + t.T
        pred_flat = pred_pts[valid_mask].reshape(-1, 3)
        gt_flat = gt_pts[valid_mask].reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred_flat)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_flat)
        pcd_gt.colors = o3d.utility.Vector3dVector(colors)

        reg = o3d.pipelines.registration.registration_icp(
            pcd,
            pcd_gt,
            0.1,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        pcd.transform(reg.transformation)
        pcd.estimate_normals()
        pcd_gt.estimate_normals()
        pred_normal = np.asarray(pcd.normals)
        gt_normal = np.asarray(pcd_gt.normals)

        acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
        comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)
        row = {
            "seq": seq_name,
            "Acc-mean": acc,
            "Acc-med": acc_med,
            "Comp-mean": comp,
            "Comp-med": comp_med,
            "NC1-mean": nc1,
            "NC1-med": nc1_med,
            "NC2-mean": nc2,
            "NC2-med": nc2_med,
        }
        write_csv_row(all_samples_csv, row)
        sums["Acc-mean"] += acc
        sums["Acc-med"] += acc_med
        sums["Comp-mean"] += comp
        sums["Comp-med"] += comp_med
        sums["NC1-mean"] += nc1
        sums["NC1-med"] += nc1_med
        sums["NC2-mean"] += nc2
        sums["NC2-med"] += nc2_med
        sums["NC-mean"] += (nc1 + nc2) / 2
        sums["NC-med"] += (nc1_med + nc2_med) / 2
        print(f"{model_name} {seq_name}: Acc={acc:.6f}, Comp={comp:.6f}, NC={(nc1 + nc2) / 2:.6f}")

    metrics = {"model": model_name, **{key: value / len(dataset) for key, value in sums.items()}}
    with metric_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Saved {metric_csv}")
PY
}

log "check paths"
require_file "$VGGT_PY"
require_file "$RECONS_PY"
require_file "$VGGT_ROOT/attack_vggt_new1.py"
require_dir "$RECONS_ROOT"
require_dir "$NYU_SCENES"
grep -q "torch.optim.Adam" "$VGGT_ROOT/attack_vggt_new1.py" || {
  echo "attack_vggt_new1.py does not look like the Adam-patch version." >&2
  exit 1
}

prepare_bonn_scenes_if_needed
prepare_tum_images_link
prepare_nrgbd_scenes_if_needed

log "NYU-v2 attacks"
run_attack "$VGGT_ROOT/outputs_attack/nyu_v2_feature_global_l3" \
  --dataset nyu-v2 \
  --attack_type global \
  --scenes_root "$NYU_SCENES" \
  --scene_pattern "nyu_*" \
  --max_frames 1 \
  --eps "$GLOBAL_EPS" \
  --alpha "$GLOBAL_ALPHA"
run_attack "$VGGT_ROOT/outputs_attack/nyu_v2_feature_patch_adam_l3" \
  --dataset nyu-v2 \
  --attack_type patch \
  --scenes_root "$NYU_SCENES" \
  --scene_pattern "nyu_*" \
  --max_frames 1 \
  --alpha "$GLOBAL_ALPHA" \
  --patch_alpha "$PATCH_LR" \
  --patch_size "$PATCH_SIZE" \
  --patch_x -1 \
  --patch_y -1

log "Bonn attacks"
run_attack "$VGGT_ROOT/outputs_attack/bonn_feature_global_l3" \
  --dataset bonn \
  --attack_type global \
  --scenes_root "$BONN_SCENES" \
  --scene_pattern "rgbd_bonn_*__*" \
  --max_frames 1 \
  --eps "$GLOBAL_EPS" \
  --alpha "$GLOBAL_ALPHA"
run_attack "$VGGT_ROOT/outputs_attack/bonn_feature_patch_adam_l3" \
  --dataset bonn \
  --attack_type patch \
  --scenes_root "$BONN_SCENES" \
  --scene_pattern "rgbd_bonn_*__*" \
  --max_frames 1 \
  --alpha "$GLOBAL_ALPHA" \
  --patch_alpha "$PATCH_LR" \
  --patch_size "$PATCH_SIZE" \
  --patch_x -1 \
  --patch_y -1

log "TUM-dynamics attacks"
run_attack "$VGGT_ROOT/outputs_attack/tum_feature_global_l3" \
  --dataset tum-dynamics \
  --attack_type global \
  --scenes_root "$TUM_ROOT" \
  --scene_pattern "rgbd_dataset_freiburg3_*" \
  --max_frames 90 \
  --eps "$GLOBAL_EPS" \
  --alpha "$GLOBAL_ALPHA"
run_attack "$VGGT_ROOT/outputs_attack/tum_feature_patch_adam_l3" \
  --dataset tum-dynamics \
  --attack_type patch \
  --scenes_root "$TUM_ROOT" \
  --scene_pattern "rgbd_dataset_freiburg3_*" \
  --max_frames 90 \
  --alpha "$GLOBAL_ALPHA" \
  --patch_alpha "$PATCH_LR" \
  --patch_size "$PATCH_SIZE" \
  --patch_x -1 \
  --patch_y -1

log "Neural-RGBD sparse attacks"
run_attack "$VGGT_ROOT/outputs_attack/nrgbd_sparse_feature_global_l3" \
  --dataset nrgbd-sparse \
  --attack_type global \
  --scenes_root "$NRGBD_SCENES" \
  --scene_pattern "*" \
  --max_frames 999 \
  --eps "$GLOBAL_EPS" \
  --alpha "$GLOBAL_ALPHA"
run_attack "$VGGT_ROOT/outputs_attack/nrgbd_sparse_feature_patch_adam_l3" \
  --dataset nrgbd-sparse \
  --attack_type patch \
  --scenes_root "$NRGBD_SCENES" \
  --scene_pattern "*" \
  --max_frames 999 \
  --alpha "$GLOBAL_ALPHA" \
  --patch_alpha "$PATCH_LR" \
  --patch_size "$PATCH_SIZE" \
  --patch_x -1 \
  --patch_y -1

convert_nyu_predictions
convert_bonn_predictions

log "evaluate NYU-v2 monodepth"
(
  cd "$RECONS_ROOT"
  "$RECONS_PY" monodepth/eval.py \
    'eval_models=[vggt_nyu_v2_feature_global_l3,vggt_nyu_v2_feature_patch_adam_l3]' \
    'eval_datasets=[nyu-v2]' \
    output_dir="$RECONS_ROOT/outputs/monodepth" \
    save_suffix=nyu_v2_feature_attacks_adam_l3
)

log "evaluate Bonn monodepth"
(
  cd "$RECONS_ROOT"
  "$RECONS_PY" monodepth/eval.py \
    'eval_models=[vggt_bonn_feature_global_l3,vggt_bonn_feature_patch_adam_l3]' \
    'eval_datasets=[bonn]' \
    output_dir="$RECONS_ROOT/outputs/monodepth" \
    save_suffix=bonn_feature_attacks_adam_l3
)

eval_tum_pose
eval_nrgbd_pointmap

log "all done"
echo "NYU:   $RECONS_ROOT/outputs/monodepth/nyu-v2-metric-nyu_v2_feature_attacks_adam_l3.csv"
echo "Bonn:  $RECONS_ROOT/outputs/monodepth/bonn-metric-bonn_feature_attacks_adam_l3.csv"
echo "TUM:   $RECONS_ROOT/outputs/relpose-distance/tum-metric-vggt_tum_feature_global_l3.csv"
echo "TUM:   $RECONS_ROOT/outputs/relpose-distance/tum-metric-vggt_tum_feature_patch_adam_l3.csv"
echo "NRGBD: $RECONS_ROOT/outputs/mv_recon/NRGBD-sparse-metric-vggt_nrgbd_sparse_feature_global_l3.csv"
echo "NRGBD: $RECONS_ROOT/outputs/mv_recon/NRGBD-sparse-metric-vggt_nrgbd_sparse_feature_patch_adam_l3.csv"
