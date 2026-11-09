#!/usr/bin/env bash
# One-sequence, convergence-controlled comparison used for the main-method
# sanity check.  Training uses sitting_xyz first-half frames.  Evaluation is
# deliberately separate and must use the fixed w30/s20 stitch protocol.
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
MIRAGE_GPU="${1:-0}"
SPOO_GPU="${2:-1}"
MAX_STEPS="${3:-6000}"
SCENE=rgbd_dataset_freiburg3_sitting_xyz
SHORT=sitting_xyz
CONTEXT=10
AUDIT="$V/data/adaptive_audits/firsthalf_${SHORT}_k${CONTEXT}/audit_plan.json"

POSE_RUN="formal_xyz_v3_mirage_pose_warm_adaptive_s0_umax${MAX_STEPS}"
JOINT_RUN="formal_xyz_v3_mirage_joint_warm_adaptive_s0_umax${MAX_STEPS}"
SPOO_STAGE1_RUN="formal_xyz_v3_spoo_stage1_adaptive_s17_umax${MAX_STEPS}"
SPOO_RUN="formal_xyz_v3_spoo_stage2_adaptive_s17_umax${MAX_STEPS}"
POSE_TEXTURE="$V/outputs_attack_geometry_aware_tum10/$POSE_RUN/geometry_patch/optimized_patch.png"
SPOO_STAGE1_TEXTURE="$V/outputs_attack_geometry_aware_tum10/$SPOO_STAGE1_RUN/geometry_patch/optimized_patch.png"

require_early_stop() {
  local run="$1"
  "$V/../conda_envs/vggt/bin/python3" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["adaptive_scheduler"]["state"]["stopped"], "stage hit cap without convergence"' \
    "$V/outputs_attack_geometry_aware_tum10/$run/geometry_patch/geometry_patch_meta.json"
}

test -f "$AUDIT"
mkdir -p "$V/logs/formal_xyz_comparison"

run_mirage() {
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
  export PER_SCENE_SEED=0

  export PER_SCENE_MIRAGE_PROJECTED_TASKS=pose_trans,pose_rot
  unset PER_SCENE_TEXTURE_INIT_IMAGE
  "$V/scripts/runs/launch_per_scene_split_joint_v5.sh" \
    firsthalf "$SCENE" "$MIRAGE_GPU" "$CONTEXT" "$MAX_STEPS" "$POSE_RUN"

  test -f "$POSE_TEXTURE"
  require_early_stop "$POSE_RUN"
  export PER_SCENE_MIRAGE_PROJECTED_TASKS=pose_trans,pose_rot,depth,point,track
  export PER_SCENE_TEXTURE_INIT_IMAGE="$POSE_TEXTURE"
  "$V/scripts/runs/launch_per_scene_split_joint_v5.sh" \
    firsthalf "$SCENE" "$MIRAGE_GPU" "$CONTEXT" "$MAX_STEPS" "$JOINT_RUN"
}

run_spoo() {
  export CONTROLLED_SCHEDULER=adaptive_plateau
  export CONTROLLED_AUDIT_PLAN="$AUDIT"
  export CONTROLLED_AUDIT_EVERY=100
  export CONTROLLED_CHECKPOINT_EVERY=100
  export CONTROLLED_AUTO_RESUME=1
  export CONTROLLED_ADAPTIVE_LR_FACTOR=0.5
  export CONTROLLED_ADAPTIVE_LR_PATIENCE=3
  export CONTROLLED_ADAPTIVE_STOP_PATIENCE=5
  # The source polynomial schedule bottoms out at half the initial LR.
  export CONTROLLED_ADAPTIVE_MIN_LR=0.18445
  export CONTROLLED_ADAPTIVE_IMPROVEMENT_THRESHOLD=0.005

  export CONTROLLED_SPOO_FORCE_SECOND_STAGE=0
  export CONTROLLED_SPOO_LATE_START_FRACTION=2.0
  unset CONTROLLED_TEXTURE_INIT_IMAGE
  "$V/scripts/runs/launch_controlled_baseline_v5.sh" \
    spoo "$SCENE" "$SPOO_GPU" "$MAX_STEPS" "$CONTEXT" "$SPOO_STAGE1_RUN"

  test -f "$SPOO_STAGE1_TEXTURE"
  require_early_stop "$SPOO_STAGE1_RUN"
  export CONTROLLED_SPOO_FORCE_SECOND_STAGE=1
  export CONTROLLED_SPOO_LATE_START_FRACTION=0.0
  export CONTROLLED_TEXTURE_INIT_IMAGE="$SPOO_STAGE1_TEXTURE"
  "$V/scripts/runs/launch_controlled_baseline_v5.sh" \
    spoo "$SCENE" "$SPOO_GPU" "$MAX_STEPS" "$CONTEXT" "$SPOO_RUN"
}

run_mirage >"$V/logs/formal_xyz_comparison/mirage_train.log" 2>&1 &
MIRAGE_PID=$!
run_spoo >"$V/logs/formal_xyz_comparison/spoo_train.log" 2>&1 &
SPOO_PID=$!
echo "MIRAGE_PID=$MIRAGE_PID SPOO_PID=$SPOO_PID"
wait "$MIRAGE_PID"
wait "$SPOO_PID"
echo "TRAINING_COMPLETE pose=$POSE_RUN joint=$JOINT_RUN spoo_stage1=$SPOO_STAGE1_RUN spoo=$SPOO_RUN"
