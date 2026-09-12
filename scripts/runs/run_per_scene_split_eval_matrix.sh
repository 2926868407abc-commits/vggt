#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
VGGT_PY=/mnt/data/wangqq/conda_envs/vggt/bin/python3
EVAL_PY=/mnt/data/wangqq/conda_envs/recons_eval/bin/python3
STEPS="${1:-1000}"
LOG_ROOT="$V/logs/per_scene_split_eval_u${STEPS}"
mkdir -p "$LOG_ROOT"

groups=(
  "rgbd_dataset_freiburg3_sitting_static rgbd_dataset_freiburg3_sitting_xyz"
  "rgbd_dataset_freiburg3_sitting_rpy rgbd_dataset_freiburg3_sitting_halfsphere"
  "rgbd_dataset_freiburg3_walking_static rgbd_dataset_freiburg3_walking_xyz"
  "rgbd_dataset_freiburg3_walking_rpy rgbd_dataset_freiburg3_walking_halfsphere"
)

eval_one() {
  local gpu="$1" split="$2" scene="$3"
  local short run root texture plan dense
  short="${scene#rgbd_dataset_freiburg3_}"
  run="perseq_${split}_${short}_joint_cold_s0_u${STEPS}"
  root="$V/outputs_attack_geometry_aware_tum10/$run"
  texture="$root/geometry_patch/geometry_patch_texture.npz"
  plan="$V/data/tum_per_scene_splits/${split}_test_eval_plan_w30_s20.json"
  dense="$root/heldout_w30_s20/four_tasks"
  test -f "$texture"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPYCACHEPREFIX=/mnt/data/wangqq/tmp_pyc \
    "$VGGT_PY" "$V/scripts/eval_universal_four_tasks.py" infer \
      --output_root "$dense" \
      --texture "$texture" \
      --protocol "$V/data/tum_per_scene_splits/protocol.json" \
      --frame_plan "$plan" \
      --max_point_samples 200000 \
      --resume \
      --scenes "$scene"

  PYTHONPYCACHEPREFIX=/mnt/data/wangqq/tmp_pyc \
    "$EVAL_PY" "$V/scripts/eval_stitched_pose.py" metrics \
      --output_root "$dense" \
      --scenes "$scene"

  PYTHONPYCACHEPREFIX=/mnt/data/wangqq/tmp_pyc \
    "$EVAL_PY" "$V/scripts/eval_universal_four_tasks.py" summarize \
      --output_root "$dense" \
      --pose_csv "$dense/stitched_pose_metrics.csv" \
      --scenes "$scene"
}

worker() {
  local gpu="$1" scene split log
  for scene in ${groups[$gpu]}; do
    for split in firsthalf evenodd; do
      log="$LOG_ROOT/${split}_${scene#rgbd_dataset_freiburg3_}.log"
      echo "[start] $(date -Is) gpu=$gpu split=$split scene=$scene"
      eval_one "$gpu" "$split" "$scene" >"$log" 2>&1
      echo "[done] $(date -Is) gpu=$gpu split=$split scene=$scene"
    done
  done
}

pids=()
for gpu in 0 1 2 3; do
  worker "$gpu" >"$LOG_ROOT/worker${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "one or more evaluation workers failed" >&2
  exit "$status"
fi

PYTHONPYCACHEPREFIX=/mnt/data/wangqq/tmp_pyc \
  "$EVAL_PY" "$V/scripts/summarize_per_scene_split_results.py" \
    --steps "$STEPS" \
    --output_dir "$V/reports/per_scene_split_comparison"
echo "all 16 per-scene split evaluations completed"
