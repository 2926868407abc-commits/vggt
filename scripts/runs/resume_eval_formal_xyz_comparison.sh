#!/usr/bin/env bash
# Resume only the capped MIRAGE joint stage, require adaptive convergence,
# then evaluate MIRAGE and SPOO with the sole formal w30/s20 protocol.
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
GPU_MIRAGE="${1:-0}"
GPU_SPOO="${2:-1}"
RESUME_CAP="${3:-8000}"
SCENE=rgbd_dataset_freiburg3_sitting_xyz
CONTEXT=10
AUDIT="$V/data/adaptive_audits/firsthalf_sitting_xyz_k10/audit_plan.json"
POSE_RUN=formal_xyz_v3_mirage_pose_warm_adaptive_s0_umax6000
MIRAGE_RUN=formal_xyz_v3_mirage_joint_warm_adaptive_s0_umax6000
SPOO_RUN=formal_xyz_v3_spoo_stage2_adaptive_s17_umax6000
POSE_TEXTURE="$V/outputs_attack_geometry_aware_tum10/$POSE_RUN/geometry_patch/optimized_patch.png"
MIRAGE_META="$V/outputs_attack_geometry_aware_tum10/$MIRAGE_RUN/geometry_patch/geometry_patch_meta.json"
LOG_DIR="$V/logs/formal_xyz_comparison"

test -f "$AUDIT"
test -f "$POSE_TEXTURE"
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
export PER_SCENE_DYNAMIC_SETS_PER_STEP=1
export PER_SCENE_DYNAMIC_QUERY_SAMPLING=uniform_query
export PER_SCENE_PROJECTED_SUBSET_AGG=mean
export PER_SCENE_SEED=0
export PER_SCENE_MIRAGE_PROJECTED_TASKS=pose_trans,pose_rot,depth,point,track
export PER_SCENE_TEXTURE_INIT_IMAGE="$POSE_TEXTURE"

"$V/scripts/runs/launch_per_scene_split_joint_v5.sh" \
  firsthalf "$SCENE" "$GPU_MIRAGE" "$CONTEXT" "$RESUME_CAP" "$MIRAGE_RUN" \
  >"$LOG_DIR/mirage_joint_resume.log" 2>&1

"$V/../conda_envs/vggt/bin/python3" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["adaptive_scheduler"]["state"]["stopped"], "MIRAGE still hit cap without convergence"' \
  "$MIRAGE_META"

{
  echo "[comparison-eval-start] $(date -Is) protocol=w30_s20"
  "$V/scripts/runs/run_controlled_baseline_eval.sh" "$GPU_MIRAGE" "$SCENE" "$MIRAGE_RUN" &
  MIRAGE_PID=$!
  "$V/scripts/runs/run_controlled_baseline_eval.sh" "$GPU_SPOO" "$SCENE" "$SPOO_RUN" &
  SPOO_PID=$!
  wait "$MIRAGE_PID"
  wait "$SPOO_PID"
  "$V/../conda_envs/vggt/bin/python3" \
    "$V/scripts/summarize_formal_xyz_comparison.py" report \
    --root "$V" --max_steps 6000
  echo "[comparison-eval-done] $(date -Is) protocol=w30_s20"
} >"$LOG_DIR/eval_w30_s20.log" 2>&1
