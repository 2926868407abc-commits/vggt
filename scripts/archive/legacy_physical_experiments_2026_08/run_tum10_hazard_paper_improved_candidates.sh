#!/usr/bin/env bash
set -Eeuo pipefail

# Paper-inspired, geometry-constrained hazard patch candidates.
# Borrowed from MDE_Attack at the mechanism level:
#   - visible warning texture initialization/reference
#   - printable/color-limited patch regularization
#   - search over effective patch carriers
# Improved for VGGT:
#   - candidates are restricted to manual wall/table physical carriers
#   - GT depth visibility and surface support checks keep the patch on one plane
#   - every candidate gets a separate run name and recons_eval metric

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
    --out "$HAZARD_TEXTURE" \
    --size "${HAZARD_SIZE:-256}" \
    --stripe_width "${HAZARD_STRIPE_WIDTH:-24}"
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
  MANUAL_ANCHOR_SEARCH_RADIUS="${MANUAL_ANCHOR_SEARCH_RADIUS:-14}"
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

  local run_name="tum10_sitting_static_hazard_paper_improved_${tag}_${UPDATES}"
  local model_name="vggt_tum10_sitting_static_hazard_paper_improved_${tag}_${UPDATES}"

  echo "===== candidate ${tag} ====="
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

if [[ "${RUN_WALL:-1}" == "1" ]]; then
  run_candidate wall_poster_medium 0.50 0.25 0 0.22 0.22 0.60 0.05 0.04
  run_candidate wall_poster_large  0.50 0.25 0 0.30 0.24 0.45 0.08 0.05
fi

if [[ "${RUN_TABLE:-1}" == "1" ]]; then
  run_candidate table_sticker_medium 0.52 0.58 -8 0.20 0.12 0.35 0.08 0.05
  run_candidate table_sticker_large  0.53 0.56 -8 0.26 0.16 0.30 0.10 0.06
fi

echo "===== done paper-improved candidates ====="
