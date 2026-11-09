#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
SCENE="${1:?usage: $0 SCENE GPU CONTEXT [MAX_STEPS] [RUN_NAME]}"
GPU="${2:?usage: $0 SCENE GPU CONTEXT [MAX_STEPS] [RUN_NAME]}"
CONTEXT="${3:?usage: $0 SCENE GPU CONTEXT [MAX_STEPS] [RUN_NAME]}"
MAX_STEPS="${4:-3000}"
SHORT="${SCENE#rgbd_dataset_freiburg3_}"
RUN_NAME="${5:-perseq_v5_k${CONTEXT}_firsthalf_${SHORT}_joint_adaptive_s0_umax${MAX_STEPS}}"
AUDIT_PLAN="$V/data/adaptive_audits/firsthalf_${SHORT}_k${CONTEXT}/audit_plan.json"

case "$CONTEXT" in
  10|20|30) ;;
  *) echo "context must be 10, 20, or 30: $CONTEXT" >&2; exit 2 ;;
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
  firsthalf "$SCENE" "$GPU" "$CONTEXT" "$MAX_STEPS" "$RUN_NAME"
