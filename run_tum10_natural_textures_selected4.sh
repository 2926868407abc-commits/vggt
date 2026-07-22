#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
TEXTURE_ROOT="${TEXTURE_ROOT:-$VGGT_ROOT/assets/natural_textures}"
DTD_ROOT="${DTD_ROOT:-$VGGT_ROOT/data/textures/dtd/images}"

mkdir -p "$TEXTURE_ROOT"

copy_texture() {
  local src="$1"
  local dst="$2"
  [[ -f "$src" ]] || { echo "Missing DTD texture: $src" >&2; exit 1; }
  if [[ ! -f "$dst" ]]; then
    cp "$src" "$dst"
  fi
}

copy_texture "$DTD_ROOT/woven/woven_0003.jpg" "$TEXTURE_ROOT/dtd_woven_0003_texture.jpg"
copy_texture "$DTD_ROOT/banded/banded_0009.jpg" "$TEXTURE_ROOT/dtd_banded_0009_texture.jpg"
copy_texture "$DTD_ROOT/blotchy/blotchy_0006.jpg" "$TEXTURE_ROOT/dtd_blotchy_0006_texture.jpg"
copy_texture "$DTD_ROOT/woven/woven_0001.jpg" "$TEXTURE_ROOT/dtd_woven_0001_texture.jpg"

run_one() {
  local name="$1"
  local natural_img="$2"

  echo "===== run ${name} ====="
  FORCE_PREPARE_TUM10=0 \
  FORCE_CLEAN=0 \
  FORCE_TRAIN=1 \
  FORCE_APPLY=1 \
  RUN_EVAL=1 \
  SCENE_PATTERN=rgbd_dataset_freiburg3_sitting_static \
  TUM_GEOM_RUN_NAME="tum10_sitting_static_wall_pose_gt_${name}_natural_lr002_ref005" \
  TUM_GEOM_MODEL="vggt_tum10_sitting_static_wall_pose_gt_${name}_natural_lr002_ref005" \
  TUM_CLEAN_MODEL=vggt_tum10_sitting_static_clean_uniform_l3 \
  ATTACK_LOSS=pose_gt_untargeted \
  ACTIVATION_CHECKPOINT="${ACTIVATION_CHECKPOINT:-0}" \
  PLANE_MODE=depth_manual_anchor_surface \
  SURFACE_SCORE_MODE=natural \
  SURFACE_COVERAGE_MIN="${SURFACE_COVERAGE_MIN:-0.003}" \
  SURFACE_COVERAGE_MAX="${SURFACE_COVERAGE_MAX:-0.05}" \
  SURFACE_MIN_VISIBLE_FRAMES="${SURFACE_MIN_VISIBLE_FRAMES:-4}" \
  SURFACE_MIN_VISIBILITY_RATIO="${SURFACE_MIN_VISIBILITY_RATIO:-0.5}" \
  MANUAL_ANCHOR_COORDINATES=normalized \
  MANUAL_ANCHOR_FRAME=0 \
  MANUAL_ANCHOR_X=0.50 \
  MANUAL_ANCHOR_Y=0.25 \
  MANUAL_ANCHOR_SEARCH_RADIUS=16 \
  MANUAL_ANCHOR_ROLL_DEGREES=0 \
  PLANE_WIDTH=0.18 \
  PLANE_HEIGHT=0.25 \
  TEXTURE_INIT=image \
  TEXTURE_INIT_IMAGE="$natural_img" \
  NATURAL_REFERENCE_IMAGE="$natural_img" \
  NATURAL_REFERENCE_WEIGHT="${NATURAL_REFERENCE_WEIGHT:-0.05}" \
  ITERATIONS="${ITERATIONS:-1000}" \
  INNER_LOOP=1 \
  SCENES_PER_ITERATION=1 \
  PATCH_LR="${PATCH_LR:-0.002}" \
  POSE_ROTATION_WEIGHT="${POSE_ROTATION_WEIGHT:-5.0}" \
  POSE_TRANSLATION_WEIGHT=1.0 \
  FEATURE_LAYER=aggregator_final \
  OPTIMIZE_GEOMETRY=0 \
  SURFACE_STRENGTH_SEARCH=0 \
  NATURAL_AUTO_RELAX=0 \
  USE_DEPTH_VISIBILITY=1 \
  SURFACE_SUPPORT_CHECK=1 \
  SURFACE_SUPPORT_ABS_TOLERANCE="${SURFACE_SUPPORT_ABS_TOLERANCE:-0.08}" \
  SURFACE_SUPPORT_REL_TOLERANCE="${SURFACE_SUPPORT_REL_TOLERANCE:-0.05}" \
  SURFACE_MIN_SUPPORT_RATIO="${SURFACE_MIN_SUPPORT_RATIO:-0.55}" \
  PHYSICAL_EOT=1 \
  TV_WEIGHT=0.005 \
  PRINTABILITY_WEIGHT=0.001 \
  PRINTABLE_COLOR_LEVELS=8 \
  LOW_FREQUENCY_WEIGHT=0.002 \
  LOW_FREQUENCY_KERNEL=11 \
  bash "$VGGT_ROOT/run_geometry_aware_tum10.sh"
}

run_one dtd_woven_0003 "$TEXTURE_ROOT/dtd_woven_0003_texture.jpg"
run_one dtd_banded_0009 "$TEXTURE_ROOT/dtd_banded_0009_texture.jpg"
run_one dtd_blotchy_0006 "$TEXTURE_ROOT/dtd_blotchy_0006_texture.jpg"
run_one dtd_woven_0001 "$TEXTURE_ROOT/dtd_woven_0001_texture.jpg"

echo "===== done selected natural textures ====="
