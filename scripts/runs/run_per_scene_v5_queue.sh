#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
GPU="${1:?usage: $0 GPU CONTEXT SPLITS [STEPS]}"
CONTEXT="${2:?usage: $0 GPU CONTEXT SPLITS [STEPS]}"
SPLITS_CSV="${3:?usage: $0 GPU CONTEXT SPLITS [STEPS]}"
STEPS="${4:-1000}"
IFS=',' read -r -a SPLITS <<< "$SPLITS_CSV"

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

LOG_ROOT="$V/logs/per_scene_v5_k${CONTEXT}_u${STEPS}"
mkdir -p "$LOG_ROOT"
QUEUE_LOG="$LOG_ROOT/gpu${GPU}_${SPLITS_CSV//,/_}.queue.log"

for scene in "${SCENES[@]}"; do
  short="${scene#rgbd_dataset_freiburg3_}"
  for split in "${SPLITS[@]}"; do
    run="perseq_v5_k${CONTEXT}_${split}_${short}_joint_cold_s0_u${STEPS}"
    final="$V/outputs_attack_geometry_aware_tum10/$run/geometry_patch/geometry_patch_texture.npz"
    checkpoint="$V/outputs_attack_geometry_aware_tum10/$run/geometry_patch/training_checkpoint.pt"
    log="$LOG_ROOT/$run.log"
    if [[ -f "$final" ]]; then
      echo "[skip-complete] $(date -Is) $run" | tee -a "$QUEUE_LOG"
      continue
    fi
    if [[ -f "$checkpoint" ]]; then
      echo "[resume] $(date -Is) gpu=$GPU run=$run" | tee -a "$QUEUE_LOG"
    else
      echo "[start] $(date -Is) gpu=$GPU run=$run" | tee -a "$QUEUE_LOG"
    fi
    if ! bash "$V/scripts/runs/launch_per_scene_split_joint_v5.sh" \
        "$split" "$scene" "$GPU" "$CONTEXT" "$STEPS" "$run" \
        >"$log" 2>&1; then
      echo "[failed] $(date -Is) gpu=$GPU run=$run log=$log" | tee -a "$QUEUE_LOG"
      exit 1
    fi
    test -f "$final"
    echo "[done] $(date -Is) gpu=$GPU run=$run" | tee -a "$QUEUE_LOG"
  done
done

echo "[queue-complete] $(date -Is) gpu=$GPU context=$CONTEXT splits=$SPLITS_CSV" \
  | tee -a "$QUEUE_LOG"
