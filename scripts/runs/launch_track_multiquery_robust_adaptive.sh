#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
GPU="${1:-1}"
MAX_STEPS="${2:-3000}"
RUN_NAME="${3:-trackonly_v5_k10_firsthalf_sitting_xyz_multiquery6_robust_adaptive_s0_umax${MAX_STEPS}}"

export PER_SCENE_MIRAGE_PROJECTED_TASKS=track
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MODE=eval_epe_softmin
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MIN_POINTS_PER_FRAME=20
export PER_SCENE_MIRAGE_PROJECTED_TRACK_TARGET_SOFTMIN_TAU_PX=0.25
export PER_SCENE_PROJECTED_SUBSET_AGG=softmin
export PER_SCENE_PROJECTED_SUBSET_SOFTMIN_TAU_PX=0.25
export PER_SCENE_DYNAMIC_QUERY_SAMPLING=uniform_query
export PER_SCENE_DYNAMIC_SETS_PER_STEP=6
export PER_SCENE_CHECKPOINT_EVERY=100
export PER_SCENE_AUTO_RESUME=1

exec "$V/scripts/runs/launch_adaptive_convergence_pilot.sh" \
  rgbd_dataset_freiburg3_sitting_xyz \
  "$GPU" \
  "$MAX_STEPS" \
  "$RUN_NAME"
