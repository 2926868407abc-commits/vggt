#!/usr/bin/env bash
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
GPU="${1:?usage: $0 GPU CONTEXT SCENE RUN [MAX_STEPS]}"
CONTEXT="${2:?usage: $0 GPU CONTEXT SCENE RUN [MAX_STEPS]}"
SCENE="${3:?usage: $0 GPU CONTEXT SCENE RUN [MAX_STEPS]}"
RUN="${4:?usage: $0 GPU CONTEXT SCENE RUN [MAX_STEPS]}"
MAX_STEPS="${5:-8000}"
SHORT="${SCENE#rgbd_dataset_freiburg3_}"
OUT="$V/outputs_attack_geometry_aware_tum10/$RUN/geometry_patch"
AUDIT="$V/data/adaptive_audits/firsthalf_${SHORT}_k${CONTEXT}/audit_plan.json"
LOG_DIR="$V/logs/autodl_resume_20261004"
LOG="$LOG_DIR/${RUN}.log"

test -f "$OUT/training_checkpoint.pt"
test -f "$OUT/initial_patch.png"
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
export PER_SCENE_MIRAGE_PROJECTED_TASKS=pose_trans,pose_rot,depth,point,track
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MODE=eval_epe
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MIN_POINTS_PER_FRAME=20
export PER_SCENE_MIRAGE_PROJECTED_MAX_QUERIES=0
export PER_SCENE_ACTIVATION_CHECKPOINT=1
export PER_SCENE_ACTIVATION_CHECKPOINT_MODE=all
export PER_SCENE_SEED=0
export PER_SCENE_TEXTURE_INIT_IMAGE="$OUT/initial_patch.png"

{
  echo "[resume-final-start] $(date -Is) gpu=$GPU context=$CONTEXT run=$RUN"
  bash "$V/scripts/runs/launch_per_scene_split_joint_v5_autodl_resume_20261004.sh" \
    firsthalf "$SCENE" "$GPU" "$CONTEXT" "$MAX_STEPS" "$RUN"
  echo "[resume-final-train-finished] $(date -Is) run=$RUN"
  bash "$V/scripts/runs/eval_per_scene_formal_w30_s20.sh" \
    "$GPU" "$SCENE" "$RUN"
  echo "[resume-final-eval-finished] $(date -Is) run=$RUN"
} >>"$LOG" 2>&1
