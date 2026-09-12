#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
STEPS="${1:-1000}"
LOG_ROOT="$V/logs/per_scene_split_matrix_u${STEPS}"
mkdir -p "$LOG_ROOT"

groups=(
  "rgbd_dataset_freiburg3_sitting_static rgbd_dataset_freiburg3_sitting_xyz"
  "rgbd_dataset_freiburg3_sitting_rpy rgbd_dataset_freiburg3_sitting_halfsphere"
  "rgbd_dataset_freiburg3_walking_static rgbd_dataset_freiburg3_walking_xyz"
  "rgbd_dataset_freiburg3_walking_rpy rgbd_dataset_freiburg3_walking_halfsphere"
)

worker() {
  local gpu="$1"
  local scene split short run log output
  for scene in ${groups[$gpu]}; do
    short="${scene#rgbd_dataset_freiburg3_}"
    for split in firsthalf evenodd; do
      run="perseq_${split}_${short}_joint_cold_s0_u${STEPS}"
      output="$V/outputs_attack_geometry_aware_tum10/$run/geometry_patch/geometry_patch_texture.npz"
      log="$LOG_ROOT/$run.log"
      if [[ -f "$output" ]]; then
        echo "[skip] $run already complete" | tee -a "$LOG_ROOT/worker${gpu}.log"
        continue
      fi
      echo "[start] $(date -Is) gpu=$gpu run=$run" | tee -a "$LOG_ROOT/worker${gpu}.log"
      if ! bash "$V/scripts/runs/launch_per_scene_split_joint.sh" \
          "$split" "$scene" "$gpu" "$STEPS" "$run" >"$log" 2>&1; then
        echo "[failed] $(date -Is) gpu=$gpu run=$run log=$log" | tee -a "$LOG_ROOT/worker${gpu}.log"
        return 1
      fi
      test -f "$output"
      echo "[done] $(date -Is) gpu=$gpu run=$run" | tee -a "$LOG_ROOT/worker${gpu}.log"
    done
  done
}

pids=()
for gpu in 0 1 2 3; do
  worker "$gpu" >"$LOG_ROOT/worker${gpu}.stdout.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "one or more workers failed" >&2
  exit "$status"
fi
echo "all 16 per-scene split runs completed"
