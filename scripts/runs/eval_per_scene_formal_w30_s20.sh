#!/usr/bin/env bash
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
GPU="${1:?usage: $0 GPU SCENE RUN_NAME}"
SCENE="${2:?usage: $0 GPU SCENE RUN_NAME}"
RUN_NAME="${3:?usage: $0 GPU SCENE RUN_NAME}"
VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"
EVAL_PY="${EVAL_PY:-/mnt/data/wangqq/conda_envs/recons_eval/bin/python3}"

PLAN="$V/data/tum_per_scene_splits/firsthalf_test_eval_plan_w30_s20.json"
PROTOCOL="$V/data/tum_per_scene_splits/protocol.json"
QUAD="$V/assets/visibility_fix/sitting_xyz_monitor_on_stable_plane_v5.txt"
ROOT="$V/outputs_attack_geometry_aware_tum10/$RUN_NAME"
TEXTURE="$ROOT/geometry_patch/geometry_patch_texture.npz"
OUT="$ROOT/heldout_w30_s20_v5_controlled/four_tasks"
CAMERA_OUT="$ROOT/heldout_w30_s20_v5_controlled/camera_only_pose"
LOG_ROOT="$V/logs/joint_context_10_20_30/formal_eval"
LOG="$LOG_ROOT/${RUN_NAME}.log"
CACHE_ROOT=/mnt/data/wangqq/cache/vggt_eval_pycache

test -f "$PLAN"
test -f "$PROTOCOL"
test -f "$QUAD"
test -f "$TEXTURE"
test -f "$V/scripts/restitch_camera_only.py"
mkdir -p "$OUT" "$CAMERA_OUT" "$LOG_ROOT" "$CACHE_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="$CACHE_ROOT"
export PYTHONPATH="/mnt/data/wangqq/recons_eval${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$OUT/four_task_by_scene.csv" \
      && -f "$CAMERA_OUT/stitched_pose_metrics.csv" ]]; then
  echo "[eval-skip-complete] $(date -Is) scene=$SCENE run=$RUN_NAME" | tee -a "$LOG"
  exit 0
fi

{
  echo "[eval-start] $(date -Is) gpu=$GPU scene=$SCENE run=$RUN_NAME protocol=w30_s20"
  "$VGGT_PY" "$V/scripts/eval_universal_four_tasks.py" infer \
    --output_root "$OUT" \
    --texture "$TEXTURE" \
    --protocol "$PROTOCOL" \
    --frame_plan "$PLAN" \
    --quad_file "$QUAD" \
    --save_window_points 2048 \
    --max_point_samples 200000 \
    --resume \
    --device cuda:0 \
    --scenes "$SCENE"

  # Dense points help diagnose scale stitching, but this is not the paper's
  # primary camera-pose number.
  "$EVAL_PY" "$V/scripts/eval_stitched_pose.py" metrics \
    --output_root "$OUT" --scenes "$SCENE"
  cp "$OUT/stitched_pose_metrics.csv" \
    "$OUT/dense_point_assisted_pose_diagnostic.csv"

  # Main pose protocol: CameraHead poses only, one complete held-out
  # trajectory, and exactly one final Sim(3) alignment to GT.
  "$VGGT_PY" "$V/scripts/restitch_camera_only.py" \
    --input_root "$OUT" --output_root "$CAMERA_OUT" --scene "$SCENE"
  "$EVAL_PY" "$V/scripts/eval_stitched_pose.py" metrics \
    --output_root "$CAMERA_OUT" --scenes "$SCENE"

  "$EVAL_PY" "$V/scripts/eval_universal_four_tasks.py" summarize \
    --output_root "$OUT" \
    --pose_csv "$CAMERA_OUT/stitched_pose_metrics.csv" \
    --scenes "$SCENE"

  test -f "$OUT/four_task_by_scene.csv"
  test -f "$OUT/$SCENE/four_task_summary.json"
  test -f "$CAMERA_OUT/stitched_pose_metrics.csv"
  echo "[eval-done] $(date -Is) gpu=$GPU scene=$SCENE run=$RUN_NAME protocol=w30_s20"
} >"$LOG" 2>&1
