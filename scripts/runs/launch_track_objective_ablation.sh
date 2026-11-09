#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
MODE="${1:?usage: $0 capped_all|eval_epe GPU [STEPS]}"
GPU="${2:?usage: $0 capped_all|eval_epe GPU [STEPS]}"
STEPS="${3:-300}"

case "$MODE" in
  capped_all|eval_epe) ;;
  *) echo "unknown Track objective mode: $MODE" >&2; exit 2 ;;
esac

export PER_SCENE_MIRAGE_PROJECTED_TASKS=track
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MODE="$MODE"
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MIN_POINTS_PER_FRAME=20
export PER_SCENE_DYNAMIC_QUERY_SAMPLING="${TRACK_QUERY_SAMPLING:-earliest}"
export PER_SCENE_CHECKPOINT_EVERY=50
export PER_SCENE_AUTO_RESUME=1

SCENE=rgbd_dataset_freiburg3_sitting_xyz
RUN="trackonly_v5_k10_firsthalf_sitting_xyz_${MODE}_s0_u${STEPS}"

exec "$V/scripts/runs/launch_per_scene_split_joint_v5.sh" \
  firsthalf "$SCENE" "$GPU" 10 "$STEPS" "$RUN"
