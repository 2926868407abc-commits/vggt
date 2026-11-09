#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
PY=/mnt/data/wangqq/conda_envs/vggt/bin/python3
POOLS="$V/data/tum_per_scene_splits/firsthalf_train_pools.json"
GT_POOL="$V/outputs/tum_gt_pool_universal_all_firsthalf"
TUM_ROOT=/mnt/data/wangqq/recons_eval/data/tum_all_sync

SCENES=(
  rgbd_dataset_freiburg3_sitting_static
  rgbd_dataset_freiburg3_sitting_xyz
  rgbd_dataset_freiburg3_sitting_rpy
  rgbd_dataset_freiburg3_sitting_halfsphere
  rgbd_dataset_freiburg3_walking_static
  rgbd_dataset_freiburg3_walking_xyz
  rgbd_dataset_freiburg3_walking_rpy
  rgbd_dataset_freiburg3_walking_halfsphere
)

for context in 10 20 30; do
  for scene in "${SCENES[@]}"; do
    short="${scene#rgbd_dataset_freiburg3_}"
    PYTHONPATH="$V/scripts" "$PY" "$V/scripts/build_adaptive_audit_plan.py" \
      --pools-json "$POOLS" \
      --scene "$scene" \
      --scene-dir "$TUM_ROOT/$scene" \
      --gt-pool "$GT_POOL" \
      --output-dir "$V/data/adaptive_audits/firsthalf_${short}_k${context}" \
      --frames-per-set "$context"
  done
done
