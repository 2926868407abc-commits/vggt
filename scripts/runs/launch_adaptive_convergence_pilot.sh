#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
SCENE="${1:?usage: $0 SCENE GPU [MAX_STEPS] [RUN_NAME]}"
GPU="${2:?usage: $0 SCENE GPU [MAX_STEPS] [RUN_NAME]}"
MAX_STEPS="${3:-3000}"
SHORT="${SCENE#rgbd_dataset_freiburg3_}"
RUN_NAME="${4:-perseq_v5_k10_firsthalf_${SHORT}_joint_adaptive_s0_umax${MAX_STEPS}}"

case "$SCENE" in
  rgbd_dataset_freiburg3_sitting_static)
    AUDIT_PLAN="$V/data/adaptive_audits/firsthalf_sitting_static_k10/audit_plan.json"
    ;;
  rgbd_dataset_freiburg3_sitting_xyz)
    AUDIT_PLAN="$V/data/adaptive_audits/firsthalf_sitting_xyz_k10/audit_plan.json"
    ;;
  *)
    echo "Adaptive convergence pilot is limited to sitting_static and sitting_xyz: $SCENE" >&2
    exit 2
    ;;
esac

test -f "$AUDIT_PLAN"
export PER_SCENE_SCHEDULER=adaptive_plateau
export PER_SCENE_AUDIT_PLAN="$AUDIT_PLAN"
export PER_SCENE_AUDIT_EVERY="${ADAPTIVE_AUDIT_EVERY:-100}"
export PER_SCENE_CHECKPOINT_EVERY="${ADAPTIVE_CHECKPOINT_EVERY:-100}"
export PER_SCENE_AUTO_RESUME="${ADAPTIVE_AUTO_RESUME:-1}"
export PER_SCENE_ADAPTIVE_LR_FACTOR="${ADAPTIVE_LR_FACTOR:-0.5}"
export PER_SCENE_ADAPTIVE_LR_PATIENCE="${ADAPTIVE_LR_PATIENCE:-3}"
export PER_SCENE_ADAPTIVE_STOP_PATIENCE="${ADAPTIVE_STOP_PATIENCE:-5}"
export PER_SCENE_ADAPTIVE_MIN_LR="${ADAPTIVE_MIN_LR:-0.00001}"
export PER_SCENE_ADAPTIVE_IMPROVEMENT_THRESHOLD="${ADAPTIVE_IMPROVEMENT_THRESHOLD:-0.005}"

exec "$V/scripts/runs/launch_per_scene_split_joint_v5.sh" \
  firsthalf "$SCENE" "$GPU" 10 "$MAX_STEPS" "$RUN_NAME"
