#!/usr/bin/env bash
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
GPU="${1:?usage: $0 GPU [MAX_STEPS]}"
MAX_STEPS="${2:-8000}"
CONTEXT=10
SCENE=rgbd_dataset_freiburg3_sitting_xyz
RUN="pointab_v5_k10_firsthalf_xyz_pointonly_cold_adaptive_s0_umax${MAX_STEPS}"
OUT="$V/outputs_attack_geometry_aware_tum10/$RUN/geometry_patch"
AUDIT="$V/data/adaptive_audits/firsthalf_sitting_xyz_k10/audit_plan.json"
LOG="$V/logs/joint_context_10_20_30/point_single_target/${RUN}.log"
PY=/mnt/data/wangqq/conda_envs/vggt/bin/python3

test -f "$AUDIT"
mkdir -p "$(dirname "$LOG")"
cd "$V"

export PER_SCENE_SCHEDULER=adaptive_plateau
export PER_SCENE_AUDIT_PLAN="$AUDIT"
export PER_SCENE_AUDIT_EVERY=100
export PER_SCENE_CHECKPOINT_EVERY=100
export PER_SCENE_AUTO_RESUME=1
export PER_SCENE_ADAPTIVE_LR_FACTOR=0.5
export PER_SCENE_ADAPTIVE_LR_PATIENCE=3
export PER_SCENE_ADAPTIVE_STOP_PATIENCE=5
export PER_SCENE_ADAPTIVE_MIN_LR=0.00001
export PER_SCENE_ADAPTIVE_IMPROVEMENT_THRESHOLD=0.005
export PER_SCENE_DYNAMIC_SETS_PER_STEP=1
export PER_SCENE_DYNAMIC_QUERY_SAMPLING=uniform_query
export PER_SCENE_PROJECTED_SUBSET_AGG=mean
export PER_SCENE_MIRAGE_PROJECTED_TASKS=point
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MODE=eval_epe
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MIN_POINTS_PER_FRAME=20
export PER_SCENE_MIRAGE_PROJECTED_MAX_QUERIES=0
export PER_SCENE_ACTIVATION_CHECKPOINT=1
export PER_SCENE_ACTIVATION_CHECKPOINT_MODE=all
export PER_SCENE_SEED=0
unset PER_SCENE_TEXTURE_INIT_IMAGE

is_converged() {
  local meta="$1"
  [[ -f "$meta" ]] || return 1
  "$PY" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["adaptive_scheduler"]["state"]["stopped"]' \
    "$meta" >/dev/null 2>&1
}

{
  echo "[point-only-start] $(date -Is) gpu=$GPU max_steps=$MAX_STEPS run=$RUN"
  if ! is_converged "$OUT/geometry_patch_meta.json"; then
    "$V/scripts/runs/launch_per_scene_split_joint_v5_autodl_resume_20261004.sh" \
      firsthalf "$SCENE" "$GPU" "$CONTEXT" "$MAX_STEPS" "$RUN"
  fi
  is_converged "$OUT/geometry_patch_meta.json"
  test -f "$OUT/optimized_patch.png"
  echo "[point-only-converged] $(date -Is)"

  bash "$V/scripts/runs/eval_per_scene_formal_w30_s20.sh" \
    "$GPU" "$SCENE" "$RUN"
  test -f "$V/outputs_attack_geometry_aware_tum10/$RUN/heldout_w30_s20_v5_controlled/four_tasks/four_task_by_scene.csv"
  echo "[point-only-evaluated] $(date -Is)"
} >>"$LOG" 2>&1

echo "$RUN"
