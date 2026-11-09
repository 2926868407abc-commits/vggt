#!/usr/bin/env bash
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
GPU="${1:-2}"
MAX_STEPS="${2:-6000}"
SCENE=rgbd_dataset_freiburg3_sitting_xyz
SHORT=sitting_xyz
CONTEXT=10
AUDIT="$V/data/adaptive_audits/firsthalf_${SHORT}_k${CONTEXT}/audit_plan.json"

STAGE1_RUN="curvegrad_xyz_v3_spoo_stage1_adaptive_s17_umax${MAX_STEPS}"
STAGE2_RUN="curvegrad_xyz_v3_spoo_stage2_adaptive_s17_umax${MAX_STEPS}"
STAGE1_TEXTURE="$V/outputs_attack_geometry_aware_tum10/$STAGE1_RUN/geometry_patch/optimized_patch.png"

require_early_stop() {
  local run="$1"
  "$V/../conda_envs/vggt/bin/python3" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["adaptive_scheduler"]["state"]["stopped"], "stage hit cap without convergence"' \
    "$V/outputs_attack_geometry_aware_tum10/$run/geometry_patch/geometry_patch_meta.json"
}

test -f "$AUDIT"
mkdir -p "$V/logs/training_curve_replay"

export CONTROLLED_SCHEDULER=adaptive_plateau
export CONTROLLED_AUDIT_PLAN="$AUDIT"
export CONTROLLED_AUDIT_EVERY=100
export CONTROLLED_CHECKPOINT_EVERY=100
export CONTROLLED_AUTO_RESUME=1
export CONTROLLED_ADAPTIVE_LR_FACTOR=0.5
export CONTROLLED_ADAPTIVE_LR_PATIENCE=3
export CONTROLLED_ADAPTIVE_STOP_PATIENCE=5
export CONTROLLED_ADAPTIVE_MIN_LR=0.18445
export CONTROLLED_ADAPTIVE_IMPROVEMENT_THRESHOLD=0.005

export CONTROLLED_SPOO_FORCE_SECOND_STAGE=0
export CONTROLLED_SPOO_LATE_START_FRACTION=2.0
unset CONTROLLED_TEXTURE_INIT_IMAGE
"$V/scripts/runs/launch_controlled_baseline_v5.sh" \
  spoo "$SCENE" "$GPU" "$MAX_STEPS" "$CONTEXT" "$STAGE1_RUN"

test -f "$STAGE1_TEXTURE"
require_early_stop "$STAGE1_RUN"

export CONTROLLED_SPOO_FORCE_SECOND_STAGE=1
export CONTROLLED_SPOO_LATE_START_FRACTION=0.0
export CONTROLLED_TEXTURE_INIT_IMAGE="$STAGE1_TEXTURE"
"$V/scripts/runs/launch_controlled_baseline_v5.sh" \
  spoo "$SCENE" "$GPU" "$MAX_STEPS" "$CONTEXT" "$STAGE2_RUN"

require_early_stop "$STAGE2_RUN"
echo "SPOO_GRAD_REPLAY_COMPLETE stage1=$STAGE1_RUN stage2=$STAGE2_RUN"
