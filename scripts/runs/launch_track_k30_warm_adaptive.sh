#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
GPU="${1:-3}"
MAX_STEPS="${2:-1500}"
SETS_PER_UPDATE="${3:-2}"
RUN_NAME="${4:-trackonly_v5_k30_firsthalf_sitting_xyz_multiquery${SETS_PER_UPDATE}_warm_adaptive_s0_umax${MAX_STEPS}}"

export PER_SCENE_SCHEDULER=adaptive_plateau
export PER_SCENE_AUDIT_PLAN="$V/data/adaptive_audits/firsthalf_sitting_xyz_k30/audit_plan.json"
export PER_SCENE_AUDIT_EVERY="${ADAPTIVE_AUDIT_EVERY:-100}"
export PER_SCENE_CHECKPOINT_EVERY="${ADAPTIVE_CHECKPOINT_EVERY:-100}"
export PER_SCENE_AUTO_RESUME="${ADAPTIVE_AUTO_RESUME:-1}"
export PER_SCENE_ADAPTIVE_LR_FACTOR="${ADAPTIVE_LR_FACTOR:-0.5}"
export PER_SCENE_ADAPTIVE_LR_PATIENCE="${ADAPTIVE_LR_PATIENCE:-3}"
export PER_SCENE_ADAPTIVE_STOP_PATIENCE="${ADAPTIVE_STOP_PATIENCE:-5}"
export PER_SCENE_ADAPTIVE_MIN_LR="${ADAPTIVE_MIN_LR:-0.00001}"
export PER_SCENE_ADAPTIVE_IMPROVEMENT_THRESHOLD="${ADAPTIVE_IMPROVEMENT_THRESHOLD:-0.005}"

export PER_SCENE_MIRAGE_PROJECTED_TASKS=track
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MODE=eval_epe
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MIN_POINTS_PER_FRAME=20
export PER_SCENE_MIRAGE_PROJECTED_MAX_QUERIES="${TRACK_MAX_QUERIES:-64}"
export PER_SCENE_PROJECTED_SUBSET_AGG=mean
export PER_SCENE_DYNAMIC_QUERY_SAMPLING=uniform_query
export PER_SCENE_DYNAMIC_SETS_PER_STEP="$SETS_PER_UPDATE"
export PER_SCENE_ACTIVATION_CHECKPOINT=1
export PER_SCENE_ACTIVATION_CHECKPOINT_MODE=all
export PER_SCENE_TEXTURE_INIT_IMAGE="$V/outputs_attack_geometry_aware_tum10/formal_xyz_v3_mirage_joint_warm_adaptive_s0_umax6000/geometry_patch/optimized_patch.png"

exec "$V/scripts/runs/launch_per_scene_split_joint_v5.sh" \
  firsthalf \
  rgbd_dataset_freiburg3_sitting_xyz \
  "$GPU" \
  30 \
  "$MAX_STEPS" \
  "$RUN_NAME"
