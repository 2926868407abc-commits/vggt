#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
STEPS="${1:-1000}"
GPU=3
LOG_ROOT="$V/logs/per_scene_split_matrix_u${STEPS}"

for scene in \
  rgbd_dataset_freiburg3_walking_rpy \
  rgbd_dataset_freiburg3_walking_halfsphere; do
  short="${scene#rgbd_dataset_freiburg3_}"
  for split in firsthalf evenodd; do
    run="perseq_${split}_${short}_joint_cold_s0_u${STEPS}"
    output="$V/outputs_attack_geometry_aware_tum10/$run/geometry_patch/geometry_patch_texture.npz"
    log="$LOG_ROOT/$run.log"
    if [[ -f "$output" ]]; then
      echo "[skip] $(date -Is) $run already complete"
      continue
    fi
    echo "[repair-start] $(date -Is) gpu=$GPU run=$run"
    bash "$V/scripts/runs/launch_per_scene_split_joint.sh" \
      "$split" "$scene" "$GPU" "$STEPS" "$run" >"$log" 2>&1
    test -f "$output"
    echo "[repair-done] $(date -Is) gpu=$GPU run=$run"
  done
done

