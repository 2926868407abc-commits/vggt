#!/usr/bin/env bash
set -Eeuo pipefail

# Physical-location comparison for hazard patches on TUM sitting_static.
# The candidates are deliberately anchored to visually plausible carriers:
#   - wall poster / yellow partition region
#   - left/right computer monitor surfaces
#
# Each candidate writes to an independent run name so results never overwrite.

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
TEXTURE_ROOT="${TEXTURE_ROOT:-$VGGT_ROOT/assets/hazard_textures}"
HAZARD_TEXTURE="${HAZARD_TEXTURE:-$TEXTURE_ROOT/mde_attack_warnning.png}"
MDE_WARNING_TEXTURE="${MDE_WARNING_TEXTURE:-$VGGT_ROOT/_external_MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/style/Warnning.png}"
UPDATES="${UPDATES:-1000}"

mkdir -p "$TEXTURE_ROOT"
if [[ ! -f "$HAZARD_TEXTURE" && -f "$MDE_WARNING_TEXTURE" ]]; then
  cp "$MDE_WARNING_TEXTURE" "$HAZARD_TEXTURE"
fi
if [[ ! -f "$HAZARD_TEXTURE" ]]; then
  "${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}" \
    "$VGGT_ROOT/scripts/make_hazard_stripe_texture.py" \
    --out "$HAZARD_TEXTURE"
fi

COMMON_ENV=(
  FORCE_PREPARE_TUM10="${FORCE_PREPARE_TUM10:-0}"
  FORCE_CLEAN="${FORCE_CLEAN:-0}"
  FORCE_TRAIN=1
  FORCE_APPLY=1
  RUN_EVAL="${RUN_EVAL:-1}"
  SCENE_PATTERN="${SCENE_PATTERN:-rgbd_dataset_freiburg3_sitting_static}"
  TUM_CLEAN_MODEL="${TUM_CLEAN_MODEL:-vggt_tum10_sitting_static_clean_uniform_l3}"
  ATTACK_LOSS="${ATTACK_LOSS:-pose_gt_untargeted}"
  POSE_REVERSE_REFERENCE="${POSE_REVERSE_REFERENCE:-gt}"
  POSE_BAD_REFERENCE="${POSE_BAD_REFERENCE:-gt}"
  POSE_ROTATION_WEIGHT="${POSE_ROTATION_WEIGHT:-5.0}"
  POSE_TRANSLATION_WEIGHT="${POSE_TRANSLATION_WEIGHT:-1.0}"
  PLANE_MODE=depth_manual_anchor_surface
  MANUAL_ANCHOR_COORDINATES=normalized
  MANUAL_ANCHOR_FRAME="${MANUAL_ANCHOR_FRAME:-0}"
  OPTIMIZE_GEOMETRY=0
  SURFACE_STRENGTH_SEARCH=0
  NATURAL_AUTO_RELAX=0
  USE_DEPTH_VISIBILITY=1
  VISIBILITY_DEPTH_MARGIN="${VISIBILITY_DEPTH_MARGIN:-0.08}"
  SURFACE_SUPPORT_CHECK=1
  FUSED_MAX_PLANE_RESIDUAL="${FUSED_MAX_PLANE_RESIDUAL:-0.12}"
  TEXTURE_INIT=image
  TEXTURE_INIT_IMAGE="$HAZARD_TEXTURE"
  NATURAL_REFERENCE_IMAGE="$HAZARD_TEXTURE"
  NATURAL_REFERENCE_WEIGHT="${NATURAL_REFERENCE_WEIGHT:-0.05}"
  ITERATIONS="$UPDATES"
  INNER_LOOP="${INNER_LOOP:-1}"
  SCENES_PER_ITERATION="${SCENES_PER_ITERATION:-1}"
  PATCH_LR="${PATCH_LR:-0.002}"
  FEATURE_LAYER="${FEATURE_LAYER:-aggregator_final}"
  PHYSICAL_EOT="${PHYSICAL_EOT:-1}"
  TV_WEIGHT="${TV_WEIGHT:-0.001}"
  PRINTABILITY_WEIGHT="${PRINTABILITY_WEIGHT:-0.001}"
  PRINTABLE_COLOR_LEVELS="${PRINTABLE_COLOR_LEVELS:-2}"
  LOW_FREQUENCY_WEIGHT="${LOW_FREQUENCY_WEIGHT:-0.0}"
)

run_candidate() {
  local tag="$1"
  local anchor_x="$2"
  local anchor_y="$3"
  local roll="$4"
  local width="$5"
  local height="$6"
  local support="$7"
  local abs_tol="$8"
  local rel_tol="$9"
  local search_radius="${10}"

  local run_name="tum10_sitting_static_hazard_physcmp_${tag}_${UPDATES}"
  local model_name="vggt_tum10_sitting_static_hazard_physcmp_${tag}_${UPDATES}"

  echo "===== physical location candidate ${tag} ====="
  env "${COMMON_ENV[@]}" \
    TUM_GEOM_RUN_NAME="$run_name" \
    TUM_GEOM_MODEL="$model_name" \
    MANUAL_ANCHOR_X="$anchor_x" \
    MANUAL_ANCHOR_Y="$anchor_y" \
    MANUAL_ANCHOR_ROLL_DEGREES="$roll" \
    MANUAL_ANCHOR_SEARCH_RADIUS="$search_radius" \
    PLANE_WIDTH="$width" \
    PLANE_HEIGHT="$height" \
    SURFACE_MIN_SUPPORT_RATIO="$support" \
    SURFACE_SUPPORT_ABS_TOLERANCE="$abs_tol" \
    SURFACE_SUPPORT_REL_TOLERANCE="$rel_tol" \
    bash "$VGGT_ROOT/run_geometry_aware_tum10.sh"
}

if [[ "${RUN_WALL:-1}" == "1" ]]; then
  # Move left/down from the previous bad wall anchors to stay on the yellow
  # partition/poster area and avoid the black monitor edge.
  run_candidate wall_poster_center 0.42 0.31 0 0.20 0.16 0.45 0.09 0.06 12
  run_candidate wall_poster_large  0.44 0.31 0 0.24 0.18 0.38 0.11 0.07 14
fi

if [[ "${RUN_MONITOR:-1}" == "1" ]]; then
  # Right monitor: anchor lower than the previous invalid fullwide run so the
  # patch is centered on the black screen rather than the screen/wall boundary.
  run_candidate monitor_right_fit  0.58 0.43 0 0.26 0.17 0.28 0.10 0.06 8
  run_candidate monitor_right_full 0.58 0.43 0 0.32 0.21 0.22 0.12 0.08 8

  # Left monitor is smaller but visually cleaner. It may be weaker, but gives a
  # useful physical control if the right monitor depth is noisy.
  run_candidate monitor_left_fit   0.30 0.42 0 0.17 0.12 0.25 0.10 0.06 8
fi

echo "===== done physical location comparison ====="
