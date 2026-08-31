#!/usr/bin/env bash
set -Eeuo pipefail

# Hazard patch candidates placed manually on the computer monitor.
# This keeps the MDE_Attack warning-texture idea, but restricts the carrier to
# the monitor plane instead of free auto-position over the full scene.

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
  MANUAL_ANCHOR_SEARCH_RADIUS="${MANUAL_ANCHOR_SEARCH_RADIUS:-10}"
  OPTIMIZE_GEOMETRY=0
  SURFACE_STRENGTH_SEARCH=0
  NATURAL_AUTO_RELAX=0
  USE_DEPTH_VISIBILITY=1
  VISIBILITY_DEPTH_MARGIN="${VISIBILITY_DEPTH_MARGIN:-0.08}"
  SURFACE_SUPPORT_CHECK=1
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

  local run_name="tum10_sitting_static_hazard_monitor_${tag}_${UPDATES}"
  local model_name="vggt_tum10_sitting_static_hazard_monitor_${tag}_${UPDATES}"

  echo "===== monitor candidate ${tag} ====="
  env "${COMMON_ENV[@]}" \
    TUM_GEOM_RUN_NAME="$run_name" \
    TUM_GEOM_MODEL="$model_name" \
    MANUAL_ANCHOR_X="$anchor_x" \
    MANUAL_ANCHOR_Y="$anchor_y" \
    MANUAL_ANCHOR_ROLL_DEGREES="$roll" \
    PLANE_WIDTH="$width" \
    PLANE_HEIGHT="$height" \
    SURFACE_MIN_SUPPORT_RATIO="$support" \
    SURFACE_SUPPORT_ABS_TOLERANCE="$abs_tol" \
    SURFACE_SUPPORT_REL_TOLERANCE="$rel_tol" \
    bash "$VGGT_ROOT/run_geometry_aware_tum10.sh"
}

run_candidate screen_small  0.57 0.36 0 0.16 0.10 0.35 0.08 0.05
run_candidate screen_medium 0.57 0.36 0 0.22 0.13 0.30 0.10 0.06

echo "===== done monitor candidates ====="
