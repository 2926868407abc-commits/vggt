#!/usr/bin/env bash
set -Eeuo pipefail

# Refinement sweep for the best-looking wall-poster hazard patch.
# Goal: keep the patch physically on the wall/poster carrier, then test which
# knob improves attack strength most: area, anchor, optimization budget, natural
# reference strength, or learning rate.

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
TEXTURE_ROOT="${TEXTURE_ROOT:-$VGGT_ROOT/assets/hazard_textures}"
HAZARD_TEXTURE="${HAZARD_TEXTURE:-$TEXTURE_ROOT/mde_attack_warnning.png}"
MDE_WARNING_TEXTURE="${MDE_WARNING_TEXTURE:-$VGGT_ROOT/_external_MDE_Attack/DeepPhotoStyle_pytorch/asset/src_img/style/Warnning.png}"

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
  INNER_LOOP="${INNER_LOOP:-1}"
  SCENES_PER_ITERATION="${SCENES_PER_ITERATION:-1}"
  FEATURE_LAYER="${FEATURE_LAYER:-aggregator_final}"
  PHYSICAL_EOT="${PHYSICAL_EOT:-1}"
  TV_WEIGHT="${TV_WEIGHT:-0.001}"
  PRINTABILITY_WEIGHT="${PRINTABILITY_WEIGHT:-0.001}"
  PRINTABLE_COLOR_LEVELS="${PRINTABLE_COLOR_LEVELS:-2}"
  LOW_FREQUENCY_WEIGHT="${LOW_FREQUENCY_WEIGHT:-0.0}"
)

run_candidate() {
  local tag="$1"
  local updates="$2"
  local anchor_x="$3"
  local anchor_y="$4"
  local width="$5"
  local height="$6"
  local ref_weight="$7"
  local lr="$8"
  local support="$9"
  local abs_tol="${10}"
  local rel_tol="${11}"

  local run_name="tum10_sitting_static_hazard_wall_refine_${tag}"
  local model_name="vggt_tum10_sitting_static_hazard_wall_refine_${tag}"

  echo "===== wall refine ${tag} ====="
  env "${COMMON_ENV[@]}" \
    TUM_GEOM_RUN_NAME="$run_name" \
    TUM_GEOM_MODEL="$model_name" \
    ITERATIONS="$updates" \
    PATCH_LR="$lr" \
    NATURAL_REFERENCE_WEIGHT="$ref_weight" \
    MANUAL_ANCHOR_X="$anchor_x" \
    MANUAL_ANCHOR_Y="$anchor_y" \
    MANUAL_ANCHOR_ROLL_DEGREES="${WALL_ROLL_DEGREES:-0}" \
    PLANE_WIDTH="$width" \
    PLANE_HEIGHT="$height" \
    SURFACE_MIN_SUPPORT_RATIO="$support" \
    SURFACE_SUPPORT_ABS_TOLERANCE="$abs_tol" \
    SURFACE_SUPPORT_REL_TOLERANCE="$rel_tol" \
    bash "$VGGT_ROOT/run_geometry_aware_tum10.sh"
}

# Reference neighborhood: old wall_poster_large was about 0.30m x 0.24m at (0.50, 0.25).
# These are ordered from safer/cheaper to more aggressive.
run_candidate area034_pos050_1000_ref005_lr002 1000 0.50 0.25 0.34 0.26 0.05 0.002 0.40 0.08 0.05
run_candidate area034_pos052_1000_ref005_lr002 1000 0.52 0.26 0.34 0.26 0.05 0.002 0.40 0.08 0.05
run_candidate area034_pos054_1000_ref005_lr002 1000 0.54 0.26 0.34 0.26 0.05 0.002 0.38 0.09 0.06
run_candidate area038_pos052_1000_ref005_lr002 1000 0.52 0.26 0.38 0.28 0.05 0.002 0.35 0.10 0.07
run_candidate area034_pos052_1000_ref002_lr002 1000 0.52 0.26 0.34 0.26 0.02 0.002 0.40 0.08 0.05
run_candidate area034_pos052_1000_ref001_lr002 1000 0.52 0.26 0.34 0.26 0.01 0.002 0.40 0.08 0.05
run_candidate area034_pos052_1000_ref002_lr003 1000 0.52 0.26 0.34 0.26 0.02 0.003 0.40 0.08 0.05
run_candidate area034_pos052_2000_ref002_lr002 2000 0.52 0.26 0.34 0.26 0.02 0.002 0.40 0.08 0.05

echo "===== done wall refine candidates ====="
