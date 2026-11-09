#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
PID_SOURCE="${1:?usage: $0 PID_OR_PID_FILE GPU RUN_NAME}"
GPU="${2:?usage: $0 PID_OR_PID_FILE GPU RUN_NAME}"
RUN_NAME="${3:?usage: $0 PID_OR_PID_FILE GPU RUN_NAME}"
SCENE=rgbd_dataset_freiburg3_sitting_xyz

if [[ "$PID_SOURCE" =~ ^[0-9]+$ ]]; then
  pid="$PID_SOURCE"
else
  pid="$(cat "$PID_SOURCE")"
fi
while kill -0 "$pid" 2>/dev/null; do
  sleep 30
done

texture="$V/outputs_attack_geometry_aware_tum10/$RUN_NAME/geometry_patch/geometry_patch_texture.npz"
if [[ ! -f "$texture" ]]; then
  echo "training process $pid ended without $texture" >&2
  exit 3
fi

exec bash "$V/scripts/runs/run_controlled_baseline_eval.sh" \
  "$GPU" "$SCENE" "$RUN_NAME"
