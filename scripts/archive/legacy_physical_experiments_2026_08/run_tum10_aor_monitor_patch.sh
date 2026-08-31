#!/usr/bin/env bash
set -Eeuo pipefail

# AoR-style monitor patch on TUM sitting_static.
#
# This keeps the placement physically conservative: the patch is a visible
# planar marker/sticker fully anchored on the right computer monitor screen.
# Unlike the surface-search experiments, geometry and position are not
# optimized, so the patch cannot drift to an easier but visually invalid area.

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
TEXTURE_ROOT="${TEXTURE_ROOT:-$VGGT_ROOT/assets/hazard_textures}"
HAZARD_TEXTURE="${HAZARD_TEXTURE:-$TEXTURE_ROOT/mde_attack_warnning.png}"
MDE_WARNING_TEXTURE="${MDE_WARNING_TEXTURE:-$VGGT_ROOT/_external_MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/style/Warnning.png}"
UPDATES="${UPDATES:-1000}"
# Normalized screen quadrilateral on the right monitor in the first selected
# frame, ordered top-left, top-right, bottom-right, bottom-left. This is the
# physically stricter AoR-style setting: the marker is bounded by the carrier
# screen ROI instead of by an arbitrary metric width/height around an anchor.
SCREEN_QUAD_XY="${SCREEN_QUAD_XY:-0.525,0.355,0.675,0.385,0.660,0.575,0.505,0.550}"
SCREEN_INNER_QUAD_XY="${SCREEN_INNER_QUAD_XY:-0.486306,0.338312,0.668433,0.338312,0.668433,0.536368,0.486306,0.536368}"
SCREEN_DISPLAY_QUAD_XY="${SCREEN_DISPLAY_QUAD_XY:-$SCREEN_INNER_QUAD_XY}"
SCREEN_DISPLAY_TAG="${SCREEN_DISPLAY_TAG:-screen_display_clean}"

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
  MANUAL_ANCHOR_X="${MANUAL_ANCHOR_X:-0.58}"
  MANUAL_ANCHOR_Y="${MANUAL_ANCHOR_Y:-0.43}"
  # This radius is only for estimating the local screen plane from depth
  # neighbors. It is not a position/strength search.
  MANUAL_ANCHOR_SEARCH_RADIUS="${MANUAL_ANCHOR_SEARCH_RADIUS:-8}"
  MANUAL_ANCHOR_ROLL_DEGREES="${MANUAL_ANCHOR_ROLL_DEGREES:-0}"
  OPTIMIZE_GEOMETRY=0
  SURFACE_STRENGTH_SEARCH=0
  NATURAL_AUTO_RELAX=0
  USE_DEPTH_VISIBILITY=1
  VISIBILITY_DEPTH_MARGIN="${VISIBILITY_DEPTH_MARGIN:-0.08}"
  SURFACE_SUPPORT_CHECK=1
  SURFACE_SCORE_MODE=natural
  SURFACE_COVERAGE_MIN="${SURFACE_COVERAGE_MIN:-0.002}"
  SURFACE_COVERAGE_MAX="${SURFACE_COVERAGE_MAX:-0.012}"
  SURFACE_MIN_VISIBLE_FRAMES="${SURFACE_MIN_VISIBLE_FRAMES:-8}"
  SURFACE_MIN_VISIBILITY_RATIO="${SURFACE_MIN_VISIBILITY_RATIO:-0.90}"
  SURFACE_MIN_SUPPORT_RATIO="${SURFACE_MIN_SUPPORT_RATIO:-0.95}"
  SURFACE_SUPPORT_ABS_TOLERANCE="${SURFACE_SUPPORT_ABS_TOLERANCE:-0.08}"
  SURFACE_SUPPORT_REL_TOLERANCE="${SURFACE_SUPPORT_REL_TOLERANCE:-0.05}"
  FUSED_MAX_PLANE_RESIDUAL="${FUSED_MAX_PLANE_RESIDUAL:-0.08}"
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

run_monitor_patch() {
  local tag="$1"
  local width="$2"
  local height="$3"
  local coverage_max="${4:-0.012}"
  local support="${5:-0.95}"
  local abs_tol="${6:-0.08}"
  local rel_tol="${7:-0.05}"

  local run_name="tum10_sitting_static_aor_monitor_${tag}_${UPDATES}"
  local model_name="vggt_tum10_sitting_static_aor_monitor_${tag}_${UPDATES}"

  echo "===== AoR-style monitor patch ${tag} ====="
  env "${COMMON_ENV[@]}" \
    TUM_GEOM_RUN_NAME="$run_name" \
    TUM_GEOM_MODEL="$model_name" \
    PLANE_WIDTH="$width" \
    PLANE_HEIGHT="$height" \
    SURFACE_COVERAGE_MAX="$coverage_max" \
    SURFACE_MIN_SUPPORT_RATIO="$support" \
    SURFACE_SUPPORT_ABS_TOLERANCE="$abs_tol" \
    SURFACE_SUPPORT_REL_TOLERANCE="$rel_tol" \
    bash "$VGGT_ROOT/run_geometry_aware_tum10.sh"
}

if [[ "${RUN_SMALL:-1}" == "1" ]]; then
  run_monitor_patch small 0.22 0.14
fi
if [[ "${RUN_FIT:-1}" == "1" ]]; then
  run_monitor_patch fit 0.26 0.17
fi
if [[ "${RUN_FULL:-1}" == "1" ]]; then
  run_monitor_patch full 0.32 0.21
fi
if [[ "${RUN_SCREEN_COVER:-1}" == "1" ]]; then
  # A larger "display content" patch: intended to cover most of the right
  # monitor screen, not merely a small sticker. The support thresholds are
  # slightly relaxed because the TUM depth around monitor borders is noisy.
  run_monitor_patch screen_cover 0.62 0.40 0.050 0.80 0.10 0.06
fi
if [[ "${RUN_SCREEN_QUAD:-0}" == "1" ]]; then
  echo "===== AoR-style monitor patch screen_quad_cover ====="
  env "${COMMON_ENV[@]}" \
    TUM_GEOM_RUN_NAME="tum10_sitting_static_aor_monitor_screen_quad_cover_${UPDATES}" \
    TUM_GEOM_MODEL="vggt_tum10_sitting_static_aor_monitor_screen_quad_cover_${UPDATES}" \
    PLANE_MODE=depth_manual_quad_surface \
    MANUAL_QUAD_COORDINATES=normalized \
    MANUAL_QUAD_XY="$SCREEN_QUAD_XY" \
    MANUAL_QUAD_DEPTH_SAMPLE_STRIDE=1 \
    MANUAL_QUAD_FIT_SHRINK="${MANUAL_QUAD_FIT_SHRINK:-0.70}" \
    MANUAL_QUAD_PLANE_INLIER_TOLERANCE="${MANUAL_QUAD_PLANE_INLIER_TOLERANCE:-0.08}" \
    MANUAL_QUAD_MIN_INLIER_RATIO="${MANUAL_QUAD_MIN_INLIER_RATIO:-0.20}" \
    SURFACE_COVERAGE_MAX=0.050 \
    SURFACE_MIN_VISIBLE_FRAMES=8 \
    SURFACE_MIN_VISIBILITY_RATIO=0.80 \
    SURFACE_MIN_SUPPORT_RATIO=0.60 \
    SURFACE_SUPPORT_ABS_TOLERANCE=0.12 \
    SURFACE_SUPPORT_REL_TOLERANCE=0.08 \
    FUSED_MAX_PLANE_RESIDUAL=0.12 \
    bash "$VGGT_ROOT/run_geometry_aware_tum10.sh"
fi
if [[ "${RUN_SCREEN_INNER:-0}" == "1" ]]; then
  echo "===== AoR-style monitor patch screen_inner_clean ====="
  env "${COMMON_ENV[@]}" \
    TUM_GEOM_RUN_NAME="tum10_sitting_static_aor_monitor_screen_inner_clean_${UPDATES}" \
    TUM_GEOM_MODEL="vggt_tum10_sitting_static_aor_monitor_screen_inner_clean_${UPDATES}" \
    PLANE_MODE=depth_manual_quad_surface \
    MANUAL_QUAD_COORDINATES=normalized \
    MANUAL_QUAD_XY="$SCREEN_INNER_QUAD_XY" \
    MANUAL_QUAD_DEPTH_SAMPLE_STRIDE=1 \
    MANUAL_QUAD_FIT_SHRINK="${MANUAL_QUAD_FIT_SHRINK:-0.75}" \
    MANUAL_QUAD_PLANE_INLIER_TOLERANCE="${MANUAL_QUAD_PLANE_INLIER_TOLERANCE:-0.06}" \
    MANUAL_QUAD_MIN_INLIER_RATIO="${MANUAL_QUAD_MIN_INLIER_RATIO:-0.25}" \
    SURFACE_COVERAGE_MAX="${SCREEN_DISPLAY_COVERAGE_MAX:-0.04}" \
    SURFACE_MIN_VISIBLE_FRAMES=8 \
    SURFACE_MIN_VISIBILITY_RATIO=0.85 \
    SURFACE_MIN_SUPPORT_RATIO=0.70 \
    SURFACE_SUPPORT_ABS_TOLERANCE=0.08 \
    SURFACE_SUPPORT_REL_TOLERANCE=0.05 \
    FUSED_MAX_PLANE_RESIDUAL=0.10 \
    bash "$VGGT_ROOT/run_geometry_aware_tum10.sh"
fi
if [[ "${RUN_SCREEN_DISPLAY:-0}" == "1" ]]; then
  echo "===== AoR-style monitor patch ${SCREEN_DISPLAY_TAG} ====="
  env "${COMMON_ENV[@]}" \
    TUM_GEOM_RUN_NAME="tum10_sitting_static_aor_monitor_${SCREEN_DISPLAY_TAG}_${UPDATES}" \
    TUM_GEOM_MODEL="vggt_tum10_sitting_static_aor_monitor_${SCREEN_DISPLAY_TAG}_${UPDATES}" \
    PLANE_MODE=depth_manual_quad_surface \
    MANUAL_QUAD_COORDINATES=normalized \
    MANUAL_QUAD_XY="$SCREEN_DISPLAY_QUAD_XY" \
    MANUAL_QUAD_DEPTH_SAMPLE_STRIDE=1 \
    MANUAL_QUAD_FIT_SHRINK="${MANUAL_QUAD_FIT_SHRINK:-0.75}" \
    MANUAL_QUAD_PLANE_INLIER_TOLERANCE="${MANUAL_QUAD_PLANE_INLIER_TOLERANCE:-0.06}" \
    MANUAL_QUAD_MIN_INLIER_RATIO="${MANUAL_QUAD_MIN_INLIER_RATIO:-0.25}" \
    USE_DEPTH_VISIBILITY=0 \
    SURFACE_SUPPORT_CHECK=0 \
    SURFACE_COVERAGE_MAX="${SCREEN_DISPLAY_COVERAGE_MAX:-0.04}" \
    SURFACE_MIN_VISIBLE_FRAMES=8 \
    SURFACE_MIN_VISIBILITY_RATIO=0.95 \
    FUSED_MAX_PLANE_RESIDUAL=0.10 \
    bash "$VGGT_ROOT/run_geometry_aware_tum10.sh"
fi

echo "===== done AoR-style monitor patch ====="
