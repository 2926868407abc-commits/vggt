#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
RUN="$V/outputs_attack_geometry_aware_tum10/universal_all8_joint_cold_s0_u1000"
POSE="$RUN/stitched_pose_unseen_w30_s20"
OUT="$RUN/four_task_unseen_w30_s20"
TEXTURE="$RUN/geometry_patch/geometry_patch_texture.npz"
PY="$V/../conda_envs/vggt/bin/python3"

mkdir -p "$OUT/logs"

run_group() {
  local gpu="$1"
  shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONUNBUFFERED=1
  "$PY" "$V/scripts/eval_universal_four_tasks.py" infer \
    --output_root "$OUT" \
    --pose_root "$POSE" \
    --texture "$TEXTURE" \
    --max_point_samples 200000 \
    --resume \
    --scenes "$@"
}

case "${1:?usage: $0 GROUP}" in
  0) run_group 0 \
       rgbd_dataset_freiburg3_sitting_static \
       rgbd_dataset_freiburg3_sitting_xyz ;;
  1) run_group 1 \
       rgbd_dataset_freiburg3_sitting_rpy \
       rgbd_dataset_freiburg3_sitting_halfsphere ;;
  2) run_group 2 \
       rgbd_dataset_freiburg3_walking_static \
       rgbd_dataset_freiburg3_walking_xyz ;;
  3) run_group 3 \
       rgbd_dataset_freiburg3_walking_rpy \
       rgbd_dataset_freiburg3_walking_halfsphere ;;
  *) echo "unknown group: $1" >&2; exit 2 ;;
esac
