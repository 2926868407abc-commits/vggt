#!/usr/bin/env bash
set -Eeuo pipefail

# Stronger-but-still-physical monitor-screen patch sweep.
# All variants keep the patch constrained to the right monitor display.

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# Earlier detected screen quad looked a bit more natural than the stricter
# refined quad, so use it as the main high-strength placement.
OLD_SCREEN_QUAD_XY="${OLD_SCREEN_QUAD_XY:-0.486306,0.338312,0.668433,0.338312,0.668433,0.536368,0.486306,0.536368}"
REFINED_SCREEN_QUAD_XY="${REFINED_SCREEN_QUAD_XY:-0.4930,0.3470,0.6620,0.3470,0.6620,0.5270,0.4930,0.5270}"

run_one() {
  local tag="$1"
  local quad_xy="$2"
  local updates="$3"
  local lr="$4"
  local ref_weight="$5"
  local rot_weight="$6"
  local trans_weight="$7"
  local eot_brightness="$8"
  local eot_contrast="$9"
  local eot_gamma="${10}"
  local eot_noise="${11}"

  echo "===== attack boost: ${tag} ====="
  RUN_SMALL=0 \
  RUN_FIT=0 \
  RUN_FULL=0 \
  RUN_SCREEN_COVER=0 \
  RUN_SCREEN_QUAD=0 \
  RUN_SCREEN_INNER=0 \
  RUN_SCREEN_DISPLAY=1 \
  SCREEN_DISPLAY_TAG="$tag" \
  SCREEN_DISPLAY_QUAD_XY="$quad_xy" \
  SCREEN_DISPLAY_COVERAGE_MAX=0.04 \
  UPDATES="$updates" \
  PATCH_LR="$lr" \
  NATURAL_REFERENCE_WEIGHT="$ref_weight" \
  POSE_ROTATION_WEIGHT="$rot_weight" \
  POSE_TRANSLATION_WEIGHT="$trans_weight" \
  PHYSICAL_EOT=1 \
  EOT_BRIGHTNESS="$eot_brightness" \
  EOT_CONTRAST="$eot_contrast" \
  EOT_GAMMA="$eot_gamma" \
  EOT_NOISE_STD="$eot_noise" \
  FORCE_PREPARE_TUM10=0 \
  FORCE_CLEAN=0 \
  FORCE_TRAIN=1 \
  FORCE_APPLY=1 \
  RUN_EVAL=1 \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  bash "$VGGT_ROOT/run_tum10_aor_monitor_patch.sh"
}

# 1. Old, visually nicer screen quad + stronger optimizer.
run_one screen_display_boost_oldquad_lr003_ref002 \
  "$OLD_SCREEN_QUAD_XY" 2000 0.003 0.02 5.0 1.0 0.15 0.15 0.10 0.01

# 2. Same old screen quad, more steps, conservative lr.
run_one screen_display_boost_oldquad_3000_ref002 \
  "$OLD_SCREEN_QUAD_XY" 3000 0.002 0.02 5.0 1.0 0.15 0.15 0.10 0.01

# 3. Keep refined quad, but make EOT less noisy to recover attack strength.
run_one screen_display_boost_refined_weakeot_lr003 \
  "$REFINED_SCREEN_QUAD_XY" 2000 0.003 0.02 5.0 1.0 0.05 0.05 0.03 0.0

# 4. Push translation harder, since ATE/RPE-trans are the main pose metrics.
run_one screen_display_boost_oldquad_trans3_lr003 \
  "$OLD_SCREEN_QUAD_XY" 2000 0.003 0.02 5.0 3.0 0.15 0.15 0.10 0.01

echo "===== done attack boost sweep ====="

