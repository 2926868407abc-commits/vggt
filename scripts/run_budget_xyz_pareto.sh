#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data/wangqq/vggt"
TEXTURE="$ROOT/assets/hazard_textures/mde_attack_warnning.png"

launch() {
  local gpu="$1" run="$2" objective="$3" fraction="$4" order="$5"
  echo "launch gpu=$gpu run=$run objective=$objective fraction=$fraction order=$order"
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    TUM10_FRAME_SCENES="/tmp/g2vlm_budget/$run/frame_scenes" \
    TUM_GEOM_RUN_NAME="$run" TUM_GEOM_MODEL="vggt_${run}" \
    SCENE_PATTERN=rgbd_dataset_freiburg3_sitting_xyz \
    ITERATIONS=250 INNER_LOOP=1 SCENES_PER_ITERATION=1 PATCH_LR=0.002 \
    TEXTURE_SIZE=128 TEXTURE_INIT=image TEXTURE_INIT_IMAGE="$TEXTURE" \
    ATTACK_LOSS=geometry_joint_gauge_budgeted \
    PIECEWISE_GAUGE_FAMILY=orthogonal_mode PIECEWISE_GAUGE_MAGNITUDE=3 \
    ORTHOGONAL_MODE_ORDER="$order" ORTHOGONAL_MODE_AXIS=0 \
    POSE_ROTATION_WEIGHT=0 POSE_TRANSLATION_WEIGHT=1 \
    FILTER_BUDGET_JSON="$ROOT/configs/tum10_filter_budgets.json" \
    FILTER_BUDGET_FRACTION="$fraction" BUDGET_POSE_OBJECTIVE="$objective" \
    BUDGET_DUAL_INIT=0 BUDGET_DUAL_LR=1 BUDGET_DUAL_MAX=100 BUDGET_RHO=1 \
    BUDGET_REPROJ_STRIDE=4 BUDGET_TRACK_PIXELS=2 \
    JOINT_TRACK_GRID_ROWS=4 JOINT_TRACK_GRID_COLS=6 JOINT_TRACK_ITERS=2 \
    PLANE_MODE=fused_depth_surface PLANE_WIDTH=0.30 PLANE_HEIGHT=0.20 \
    SURFACE_SCORE_MODE=coverage SURFACE_COVERAGE_MIN=0.005 SURFACE_COVERAGE_MAX=0.06 \
    SURFACE_MIN_VISIBLE_FRAMES=8 SURFACE_MIN_VISIBILITY_RATIO=0.80 \
    SURFACE_SUPPORT_CHECK=1 SURFACE_MIN_SUPPORT_RATIO=0.50 \
    SURFACE_SUPPORT_ABS_TOLERANCE=0.10 SURFACE_SUPPORT_REL_TOLERANCE=0.06 \
    FUSED_MAX_PLANE_RESIDUAL=0.12 VISIBILITY_DEPTH_MARGIN=0.08 \
    TV_WEIGHT=0.001 PRINTABILITY_WEIGHT=0.001 PRINTABLE_COLOR_LEVELS=2 \
    NATURAL_REFERENCE_IMAGE="$TEXTURE" NATURAL_REFERENCE_WEIGHT=0.05 \
    PHYSICAL_EOT=0 FORCE_TRAIN=1 FORCE_APPLY=1 FORCE_CLEAN=0 \
    FREEZE_MODEL_PARAMETERS=1 RUN_CONSISTENCY_CHECK=0 RUN_GAUGE_DIAG=1 RUN_EVAL=0 \
    bash "$ROOT/run_geometry_aware_tum10.sh" >"/tmp/${run}.log" 2>&1 &
  echo "$!" >"/tmp/${run}.pid"
}

launch 0 budget_xyz_u_f05_o2_250 untargeted_gt 0.50 2
launch 1 budget_xyz_u_f08_o2_250 untargeted_gt 0.80 2
launch 2 budget_xyz_t_f08_o1_250 targeted_gauge 0.80 1
launch 3 budget_xyz_t_f08_o2_250 targeted_gauge 0.80 2
