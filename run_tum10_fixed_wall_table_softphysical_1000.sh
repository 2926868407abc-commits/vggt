#!/usr/bin/env bash
set -Eeuo pipefail

# One-scene ablation: fixed 3D carrier positions with feature-L1 plus soft
# physical texture losses. Anchors are normalized coordinates in the first
# selected VGGT-preprocessed frame and can be overridden from the environment.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENE_PATTERN="${SCENE_PATTERN:-rgbd_dataset_freiburg3_sitting_static}"

COMMON_ENV=(
  SCENE_PATTERN="$SCENE_PATTERN"
  TUM_CLEAN_MODEL="${TUM_CLEAN_MODEL:-vggt_tum10_sitting_static_clean_uniform_l3}"
  PLANE_MODE=vggt_manual_anchor_surface
  MANUAL_ANCHOR_COORDINATES=normalized
  MANUAL_ANCHOR_FRAME=0
  MANUAL_ANCHOR_SEARCH_RADIUS="${MANUAL_ANCHOR_SEARCH_RADIUS:-16}"
  ITERATIONS="${ITERATIONS:-1000}"
  INNER_LOOP=1
  SCENES_PER_ITERATION=1
  PATCH_LR="${PATCH_LR:-0.001}"
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

echo "===== fixed wall poster: 1000 updates ====="
env "${COMMON_ENV[@]}" \
  TUM_GEOM_RUN_NAME="${WALL_RUN_NAME:-tum10_sitting_static_fixed_wall_poster_softphysical_1000}" \
  TUM_GEOM_MODEL="${WALL_MODEL:-vggt_tum10_sitting_static_fixed_wall_poster_softphysical_1000}" \
  MANUAL_ANCHOR_X="${WALL_ANCHOR_X:-0.50}" \
  MANUAL_ANCHOR_Y="${WALL_ANCHOR_Y:-0.22}" \
  MANUAL_ANCHOR_ROLL_DEGREES="${WALL_ROLL_DEGREES:-0}" \
  PLANE_WIDTH="${WALL_WIDTH:-0.20}" \
  PLANE_HEIGHT="${WALL_HEIGHT:-0.28}" \
  bash "$SCRIPT_DIR/run_geometry_aware_tum10.sh"

echo "===== fixed table sticker: 1000 updates ====="
env "${COMMON_ENV[@]}" \
  TUM_GEOM_RUN_NAME="${TABLE_RUN_NAME:-tum10_sitting_static_fixed_table_sticker_softphysical_1000}" \
  TUM_GEOM_MODEL="${TABLE_MODEL:-vggt_tum10_sitting_static_fixed_table_sticker_softphysical_1000}" \
  MANUAL_ANCHOR_X="${TABLE_ANCHOR_X:-0.52}" \
  MANUAL_ANCHOR_Y="${TABLE_ANCHOR_Y:-0.58}" \
  MANUAL_ANCHOR_ROLL_DEGREES="${TABLE_ROLL_DEGREES:-0}" \
  PLANE_WIDTH="${TABLE_WIDTH:-0.18}" \
  PLANE_HEIGHT="${TABLE_HEIGHT:-0.12}" \
  bash "$SCRIPT_DIR/run_geometry_aware_tum10.sh"

echo "===== fixed wall/table experiments done ====="
