#!/usr/bin/env bash
set -Eeuo pipefail

# Targeted bad-label pose attacks for fixed natural wall/table carriers.
# Bad targets:
#   drift: gradual translation drift, default final-frame x drift = 0.5m
#   scale: scale the relative trajectory translation, default scale = 2.0
#   yaw: gradual yaw drift, default final-frame yaw = 30 deg

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENE_PATTERN="${SCENE_PATTERN:-rgbd_dataset_freiburg3_sitting_static}"
UPDATES="${UPDATES:-1000}"
RUN_WALL="${RUN_WALL:-1}"
RUN_TABLE="${RUN_TABLE:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

COMMON_ENV=(
  SCENE_PATTERN="$SCENE_PATTERN"
  TUM_CLEAN_MODEL="${TUM_CLEAN_MODEL:-vggt_tum10_sitting_static_clean_uniform_l3}"
  PLANE_MODE=depth_manual_anchor_surface
  MANUAL_ANCHOR_COORDINATES=normalized
  MANUAL_ANCHOR_FRAME=0
  MANUAL_ANCHOR_SEARCH_RADIUS="${MANUAL_ANCHOR_SEARCH_RADIUS:-16}"
  ITERATIONS="$UPDATES"
  INNER_LOOP=1
  SCENES_PER_ITERATION=1
  PATCH_LR="${PATCH_LR:-0.001}"
  POSE_BAD_REFERENCE="${POSE_BAD_REFERENCE:-gt}"
  POSE_ROTATION_WEIGHT="${POSE_ROTATION_WEIGHT:-1.0}"
  POSE_TRANSLATION_WEIGHT="${POSE_TRANSLATION_WEIGHT:-1.0}"
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
  local carrier="$1"
  local target_name="$2"
  local attack_loss="$3"
  local anchor_x="$4"
  local anchor_y="$5"
  local roll="$6"
  local width="$7"
  local height="$8"
  shift 8

  local run_name="tum10_sitting_static_fixed_${carrier}_pose_${target_name}_targeted_${UPDATES}${RUN_SUFFIX}"
  local model_name="vggt_tum10_sitting_static_fixed_${carrier}_pose_${target_name}_targeted_${UPDATES}${RUN_SUFFIX}"

  echo "===== ${carrier}: ${target_name} (${attack_loss}) ====="
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

run_carrier() {
  local carrier="$1"
  local anchor_x="$2"
  local anchor_y="$3"
  local roll="$4"
  local width="$5"
  local height="$6"
  shift 6

  run_one "$carrier" "drift" "pose_drift_targeted" "$anchor_x" "$anchor_y" "$roll" "$width" "$height" \
    POSE_DRIFT_X_M="${POSE_DRIFT_X_M:-0.5}" \
    POSE_DRIFT_Y_M="${POSE_DRIFT_Y_M:-0.0}" \
    POSE_DRIFT_Z_M="${POSE_DRIFT_Z_M:-0.0}" \
    POSE_DRIFT_YAW_DEGREES="${POSE_DRIFT_YAW_DEGREES:-0.0}" \
    "$@"

  run_one "$carrier" "scale" "pose_scale_targeted" "$anchor_x" "$anchor_y" "$roll" "$width" "$height" \
    POSE_TRANSLATION_SCALE="${POSE_TRANSLATION_SCALE:-2.0}" \
    "$@"

  run_one "$carrier" "yaw" "pose_yaw_targeted" "$anchor_x" "$anchor_y" "$roll" "$width" "$height" \
    POSE_YAW_DEGREES="${POSE_YAW_DEGREES:-30.0}" \
    "$@"
}

if [[ "$RUN_WALL" == "1" ]]; then
  run_carrier \
    "wall" \
    "${WALL_ANCHOR_X:-0.62}" \
    "${WALL_ANCHOR_Y:-0.28}" \
    "${WALL_ROLL_DEGREES:-0}" \
    "${WALL_WIDTH:-0.14}" \
    "${WALL_HEIGHT:-0.20}" \
    SURFACE_SUPPORT_ABS_TOLERANCE="${WALL_SURFACE_SUPPORT_ABS_TOLERANCE:-0.04}" \
    SURFACE_SUPPORT_REL_TOLERANCE="${WALL_SURFACE_SUPPORT_REL_TOLERANCE:-0.03}" \
    SURFACE_MIN_SUPPORT_RATIO="${WALL_SURFACE_MIN_SUPPORT_RATIO:-0.75}"
fi

if [[ "$RUN_TABLE" == "1" ]]; then
  run_carrier \
    "table" \
    "${TABLE_ANCHOR_X:-0.52}" \
    "${TABLE_ANCHOR_Y:-0.58}" \
    "${TABLE_ROLL_DEGREES:-0}" \
    "${TABLE_WIDTH:-0.18}" \
    "${TABLE_HEIGHT:-0.12}"
fi

echo "===== fixed wall/table pose bad-target experiments done ====="
