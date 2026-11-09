#!/usr/bin/env bash
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
GPU="${1:?usage: $0 GPU [MAX_STEPS]}"
MAX_STEPS="${2:-8000}"
SCENE="rgbd_dataset_freiburg3_sitting_static"
SHORT="sitting_static"
CONTEXT=20

AUDIT="$V/data/adaptive_audits/firsthalf_${SHORT}_k${CONTEXT}/audit_plan.json"
INIT_RUN="jointctx_v5_k20_firsthalf_sitting_static_posetrack_adaptive_s0_umax8000"
INIT_PATCH="$V/outputs_attack_geometry_aware_tum10/$INIT_RUN/geometry_patch/optimized_patch.png"
RUN="jointctx_v5_k20_firsthalf_sitting_static_fivetarget_gt_epe_softmin_adaptive_s0_umax${MAX_STEPS}"
OUT="$V/outputs_attack_geometry_aware_tum10/$RUN"
LOG_DIR="$V/logs/joint_context_10_20_30/k20_per_scene"
LOG="$LOG_DIR/${SHORT}.fivetarget_gt_epe_softmin.log"

test -f "$AUDIT"
test -f "$INIT_PATCH"
mkdir -p "$LOG_DIR"

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

# Keep every setting from the successful K=20 run.  The only objective change
# is the target-frame aggregation: maximise the least-damaged eligible frame's
# GT EPE instead of allowing easy target frames to dominate the mean.
export PER_SCENE_MIRAGE_PROJECTED_TASKS=pose_trans,pose_rot,depth,point,track
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MODE=eval_epe_softmin
export PER_SCENE_MIRAGE_PROJECTED_TRACK_MIN_POINTS_PER_FRAME=20
export PER_SCENE_MIRAGE_PROJECTED_TRACK_TARGET_SOFTMIN_TAU_PX=0.25
export PER_SCENE_PROJECTED_SUBSET_AGG=mean
export PER_SCENE_PROJECTED_SUBSET_SOFTMIN_TAU_PX=0.25
export PER_SCENE_MIRAGE_PROJECTED_MAX_QUERIES=0
export PER_SCENE_DYNAMIC_SETS_PER_STEP=1
export PER_SCENE_DYNAMIC_QUERY_SAMPLING=uniform_query
export PER_SCENE_ACTIVATION_CHECKPOINT=1
export PER_SCENE_ACTIVATION_CHECKPOINT_MODE=all
export PER_SCENE_TEXTURE_INIT_IMAGE="$INIT_PATCH"
export PER_SCENE_SEED=0

is_converged() {
  local meta="$1"
  [[ -f "$meta" ]] || return 1
  "$V/../conda_envs/vggt/bin/python3" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["adaptive_scheduler"]["state"]["stopped"]' \
    "$meta" >/dev/null 2>&1
}

{
  echo "[gt-epe-softmin-start] $(date -Is) gpu=$GPU run=$RUN"
  if ! is_converged "$OUT/geometry_patch/geometry_patch_meta.json"; then
    "$V/scripts/runs/launch_per_scene_split_joint_v5_autodl_resume_20261004.sh" \
      firsthalf "$SCENE" "$GPU" "$CONTEXT" "$MAX_STEPS" "$RUN"
  fi
  is_converged "$OUT/geometry_patch/geometry_patch_meta.json"
  test -f "$OUT/geometry_patch/optimized_patch.png"
  echo "[gt-epe-softmin-train-done] $(date -Is) run=$RUN"

  "$V/scripts/runs/eval_per_scene_formal_w30_s20.sh" "$GPU" "$SCENE" "$RUN"
  test -f "$OUT/heldout_w30_s20_v5_controlled/four_tasks/four_task_by_scene.csv"
  echo "[gt-epe-softmin-eval-done] $(date -Is) run=$RUN"
} >>"$LOG" 2>&1
