#!/usr/bin/env bash
set -Eeuo pipefail

# One-scene TUM-10 pose-output losses for the fixed natural wall/table carriers.
# It runs two label/output objectives at each carrier:
#   1) pose_gt_untargeted: maximize distance from GT relative camera trajectory
#   2) pose_reverse_targeted: minimize distance to the inverse relative trajectory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENE_PATTERN="${SCENE_PATTERN:-rgbd_dataset_freiburg3_sitting_static}"
UPDATES="${UPDATES:-1000}"

COMMON_ENV=(
  SCENE_PATTERN="$SCENE_PATTERN"
  TUM_CLEAN_MODEL="${TUM_CLEAN_MODEL:-vggt_tum10_sitting_static_clean_uniform_l3}"
  PLANE_MODE=vggt_manual_anchor_surface
  MANUAL_ANCHOR_COORDINATES=normalized
  MANUAL_ANCHOR_FRAME=0
  MANUAL_ANCHOR_SEARCH_RADIUS="${MANUAL_ANCHOR_SEARCH_RADIUS:-16}"
  ITERATIONS="$UPDATES"
  INNER_LOOP=1
  SCENES_PER_ITERATION=1
  PATCH_LR="${PATCH_LR:-0.001}"
  POSE_ROTATION_WEIGHT="${POSE_ROTATION_WEIGHT:-1.0}"
  POSE_TRANSLATION_WEIGHT="${POSE_TRANSLATION_WEIGHT:-1.0}"
  POSE_REVERSE_REFERENCE="${POSE_REVERSE_REFERENCE:-gt}"
  FEATURE_LAYER="${FEATURE_LAYER:-aggregator_final}"
  OPTIMIZE_GEOMETRY=0
  SURFACE_STRENGTH_SEARCH=0
  NATURAL_AUTO_RELAX=0
  USE_DEPTH_VISIBILITY=1
  SURFACE_SUPPORT_CHECK=1
  SURFACE_SUPPORT_ABS_TOLERANCE="${SURFACE_SUPPORT_ABS_TOLERANCE:-0.08}"
  SURFACE_SUPPORT_REL_TOLERANCE="${SURFACE_SUPPORT_REL_TOLERANCE:-0.05}"
  SURFACE_MIN_SUPPORT_RATIO="${SURFACE_MIN_SUPPORT_RATIO:-0.4}"
  PHYSICAL_EOT="${PHYSICAL_EOT:-1}"
  TV_WEIGHT="${TV_WEIGHT:-0.01}"
  PRINTABILITY_WEIGHT="${PRINTABILITY_WEIGHT:-0.002}"
  PRINTABLE_COLOR_LEVELS="${PRINTABLE_COLOR_LEVELS:-8}"
  LOW_FREQUENCY_WEIGHT="${LOW_FREQUENCY_WEIGHT:-0.005}"
  LOW_FREQUENCY_KERNEL="${LOW_FREQUENCY_KERNEL:-11}"
  FORCE_PREPARE_TUM10="${FORCE_PREPARE_TUM10:-0}"
  FORCE_CLEAN="${FORCE_CLEAN:-0}"
  FORCE_TRAIN=1
  FORCE_APPLY=1
  RUN_EVAL="${RUN_EVAL:-1}"
)

run_one() {
  local label="$1"
  local attack_loss="$2"
  local run_name="$3"
  local model_name="$4"
  local anchor_x="$5"
  local anchor_y="$6"
  local roll="$7"
  local width="$8"
  local height="$9"
  shift 9

  echo "===== ${label}: ${attack_loss} ====="
  env "${COMMON_ENV[@]}" \
    ATTACK_LOSS="$attack_loss" \
    TUM_GEOM_RUN_NAME="$run_name" \
    TUM_GEOM_MODEL="$model_name" \
    MANUAL_ANCHOR_X="$anchor_x" \
    MANUAL_ANCHOR_Y="$anchor_y" \
    MANUAL_ANCHOR_ROLL_DEGREES="$roll" \
    PLANE_WIDTH="$width" \
    PLANE_HEIGHT="$height" \
    "$@" \
    bash "$SCRIPT_DIR/run_geometry_aware_tum10.sh"
}

run_one \
  "fixed wall poster" \
  "pose_gt_untargeted" \
  "${WALL_GT_RUN_NAME:-tum10_sitting_static_fixed_wall_pose_gt_untargeted_${UPDATES}}" \
  "${WALL_GT_MODEL:-vggt_tum10_sitting_static_fixed_wall_pose_gt_untargeted_${UPDATES}}" \
  "${WALL_ANCHOR_X:-0.62}" \
  "${WALL_ANCHOR_Y:-0.28}" \
  "${WALL_ROLL_DEGREES:-0}" \
  "${WALL_WIDTH:-0.14}" \
  "${WALL_HEIGHT:-0.20}" \
  SURFACE_SUPPORT_ABS_TOLERANCE="${WALL_SURFACE_SUPPORT_ABS_TOLERANCE:-0.04}" \
  SURFACE_SUPPORT_REL_TOLERANCE="${WALL_SURFACE_SUPPORT_REL_TOLERANCE:-0.03}" \
  SURFACE_MIN_SUPPORT_RATIO="${WALL_SURFACE_MIN_SUPPORT_RATIO:-0.75}"

run_one \
  "fixed wall poster" \
  "pose_reverse_targeted" \
  "${WALL_REVERSE_RUN_NAME:-tum10_sitting_static_fixed_wall_pose_reverse_targeted_${UPDATES}}" \
  "${WALL_REVERSE_MODEL:-vggt_tum10_sitting_static_fixed_wall_pose_reverse_targeted_${UPDATES}}" \
  "${WALL_ANCHOR_X:-0.62}" \
  "${WALL_ANCHOR_Y:-0.28}" \
  "${WALL_ROLL_DEGREES:-0}" \
  "${WALL_WIDTH:-0.14}" \
  "${WALL_HEIGHT:-0.20}" \
  SURFACE_SUPPORT_ABS_TOLERANCE="${WALL_SURFACE_SUPPORT_ABS_TOLERANCE:-0.04}" \
  SURFACE_SUPPORT_REL_TOLERANCE="${WALL_SURFACE_SUPPORT_REL_TOLERANCE:-0.03}" \
  SURFACE_MIN_SUPPORT_RATIO="${WALL_SURFACE_MIN_SUPPORT_RATIO:-0.75}"

run_one \
  "fixed table sticker" \
  "pose_gt_untargeted" \
  "${TABLE_GT_RUN_NAME:-tum10_sitting_static_fixed_table_pose_gt_untargeted_${UPDATES}}" \
  "${TABLE_GT_MODEL:-vggt_tum10_sitting_static_fixed_table_pose_gt_untargeted_${UPDATES}}" \
  "${TABLE_ANCHOR_X:-0.52}" \
  "${TABLE_ANCHOR_Y:-0.58}" \
  "${TABLE_ROLL_DEGREES:-0}" \
  "${TABLE_WIDTH:-0.18}" \
  "${TABLE_HEIGHT:-0.12}"

run_one \
  "fixed table sticker" \
  "pose_reverse_targeted" \
  "${TABLE_REVERSE_RUN_NAME:-tum10_sitting_static_fixed_table_pose_reverse_targeted_${UPDATES}}" \
  "${TABLE_REVERSE_MODEL:-vggt_tum10_sitting_static_fixed_table_pose_reverse_targeted_${UPDATES}}" \
  "${TABLE_ANCHOR_X:-0.52}" \
  "${TABLE_ANCHOR_Y:-0.58}" \
  "${TABLE_ROLL_DEGREES:-0}" \
  "${TABLE_WIDTH:-0.18}" \
  "${TABLE_HEIGHT:-0.12}"

echo "===== fixed wall/table pose-output loss experiments done ====="
