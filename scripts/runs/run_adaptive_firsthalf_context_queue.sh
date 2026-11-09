#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
GPU="${1:?usage: $0 GPU CONTEXT [MAX_STEPS]}"
CONTEXT="${2:?usage: $0 GPU CONTEXT [MAX_STEPS]}"
MAX_STEPS="${3:-3000}"
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
LOG_ROOT="$V/logs/adaptive_firsthalf_k${CONTEXT}_umax${MAX_STEPS}"
mkdir -p "$LOG_ROOT"

for scene in "${SCENES[@]}"; do
  short="${scene#rgbd_dataset_freiburg3_}"
  run="perseq_v5_k${CONTEXT}_firsthalf_${short}_joint_adaptive_s0_umax${MAX_STEPS}"
  root="$V/outputs_attack_geometry_aware_tum10/$run"
  final="$root/geometry_patch/geometry_patch_texture.npz"
  log="$LOG_ROOT/${run}.log"
  if [[ -f "$final" ]]; then
    echo "[skip-complete] $(date -Is) $run"
    continue
  fi
  echo "[start-or-resume] $(date -Is) gpu=$GPU $run"
  "$V/scripts/runs/launch_adaptive_firsthalf_context.sh" \
    "$scene" "$GPU" "$CONTEXT" "$MAX_STEPS" "$run" >"$log" 2>&1
  test -f "$final"
  echo "[done] $(date -Is) $run"
done

echo "[queue-complete] $(date -Is) context=$CONTEXT"
