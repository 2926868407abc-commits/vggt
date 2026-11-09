#!/usr/bin/env bash
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
GPU="${1:?usage: $0 GPU CONTEXT SCENE [MAX_STEPS]}"
CONTEXT="${2:?usage: $0 GPU CONTEXT SCENE [MAX_STEPS]}"
SCENE="${3:?usage: $0 GPU CONTEXT SCENE [MAX_STEPS]}"
MAX_STEPS="${4:-8000}"
SHORT="${SCENE#rgbd_dataset_freiburg3_}"

case "$CONTEXT" in
  10|20|30) ;;
  *) echo "CONTEXT must be 10, 20, or 30 (got: $CONTEXT)" >&2; exit 2 ;;
esac

AUDIT="$V/data/adaptive_audits/firsthalf_${SHORT}_k${CONTEXT}/audit_plan.json"
POSE_RUN="jointctx_v5_k${CONTEXT}_firsthalf_${SHORT}_posewarm_adaptive_s0_umax${MAX_STEPS}"
POSE_TRACK_RUN="jointctx_v5_k${CONTEXT}_firsthalf_${SHORT}_posetrack_adaptive_s0_umax${MAX_STEPS}"
JOINT_RUN="jointctx_v5_k${CONTEXT}_firsthalf_${SHORT}_fivetarget_uncapped_adaptive_s0_umax${MAX_STEPS}"
POSE_DIR="$V/outputs_attack_geometry_aware_tum10/$POSE_RUN/geometry_patch"
POSE_TRACK_DIR="$V/outputs_attack_geometry_aware_tum10/$POSE_TRACK_RUN/geometry_patch"
JOINT_DIR="$V/outputs_attack_geometry_aware_tum10/$JOINT_RUN/geometry_patch"
LOG_DIR="$V/logs/joint_context_10_20_30/k${CONTEXT}_per_scene"
LOG="$LOG_DIR/${SHORT}.three_stage.log"

test -f "$AUDIT"
mkdir -p "$LOG_DIR"
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
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MODE=eval_epe
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MIN_POINTS_PER_FRAME=20
export PER_SCENE_MIRAGE_PROJECTED_MAX_QUERIES=0
export PER_SCENE_ACTIVATION_CHECKPOINT=1
export PER_SCENE_ACTIVATION_CHECKPOINT_MODE=all
export PER_SCENE_SEED=0

is_converged() {
  local meta="$1"
  [[ -f "$meta" ]] || return 1
  "$V/../conda_envs/vggt/bin/python3" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["adaptive_scheduler"]["state"]["stopped"]' \
    "$meta" >/dev/null 2>&1
}

require_converged() {
  local meta="$1"
  is_converged "$meta" || {
    echo "stage hit the safety cap without adaptive convergence: $meta" >&2
    return 1
  }
}

run_stage() {
  local tasks="$1"
  local init_image="$2"
  local run_name="$3"
  local stage_dir="$4"
  if is_converged "$stage_dir/geometry_patch_meta.json"; then
    echo "[stage-skip-converged] $(date -Is) run=$run_name"
    return 0
  fi
  export PER_SCENE_MIRAGE_PROJECTED_TASKS="$tasks"
  if [[ -n "$init_image" ]]; then
    test -f "$init_image"
    export PER_SCENE_TEXTURE_INIT_IMAGE="$init_image"
  else
    unset PER_SCENE_TEXTURE_INIT_IMAGE
  fi
  "$V/scripts/runs/launch_per_scene_split_joint_v5_autodl_resume_20261004.sh" \
    firsthalf "$SCENE" "$GPU" "$CONTEXT" "$MAX_STEPS" "$run_name"
  require_converged "$stage_dir/geometry_patch_meta.json"
  test -f "$stage_dir/optimized_patch.png"
  echo "[stage-converged] $(date -Is) run=$run_name"
}

{
  echo "[three-stage-start] $(date -Is) gpu=$GPU context=$CONTEXT scene=$SCENE max_steps=$MAX_STEPS"

  # A completed final stage is authoritative.  This also makes the wrapper
  # safe for the already-running sitting_xyz jobs.
  if is_converged "$JOINT_DIR/geometry_patch_meta.json"; then
    echo "[three-stage-skip-final-converged] $(date -Is) run=$JOINT_RUN"
    exit 0
  fi

  run_stage "pose_trans,pose_rot" "" "$POSE_RUN" "$POSE_DIR"
  run_stage "pose_trans,pose_rot,track" \
    "$POSE_DIR/optimized_patch.png" "$POSE_TRACK_RUN" "$POSE_TRACK_DIR"
  run_stage "pose_trans,pose_rot,depth,point,track" \
    "$POSE_TRACK_DIR/optimized_patch.png" "$JOINT_RUN" "$JOINT_DIR"

  echo "[three-stage-complete] $(date -Is) run=$JOINT_RUN"
} >>"$LOG" 2>&1
