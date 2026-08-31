#!/usr/bin/env bash
set -Eeuo pipefail

# Same physically plausible monitor-screen placement, then vary only attack
# strength knobs. This keeps the comparison clean: no automatic position search.

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# Slightly shrink the detected right-monitor screen quad so the patch sits
# inside the black display rather than touching the bezel.
REFINED_SCREEN_QUAD_XY="${REFINED_SCREEN_QUAD_XY:-0.4930,0.3470,0.6620,0.3470,0.6620,0.5270,0.4930,0.5270}"

run_one() {
  local tag="$1"
  local updates="$2"
  local lr="$3"
  local ref_weight="$4"
  local physical_eot="$5"

  echo "===== screen-display strengthen: ${tag} ====="
  RUN_SMALL=0 \
  RUN_FIT=0 \
  RUN_FULL=0 \
  RUN_SCREEN_COVER=0 \
  RUN_SCREEN_QUAD=0 \
  RUN_SCREEN_INNER=0 \
  RUN_SCREEN_DISPLAY=1 \
  SCREEN_DISPLAY_TAG="$tag" \
  SCREEN_DISPLAY_QUAD_XY="$REFINED_SCREEN_QUAD_XY" \
  SCREEN_DISPLAY_COVERAGE_MAX=0.04 \
  UPDATES="$updates" \
  PATCH_LR="$lr" \
  NATURAL_REFERENCE_WEIGHT="$ref_weight" \
  PHYSICAL_EOT="$physical_eot" \
  FORCE_PREPARE_TUM10=0 \
  FORCE_CLEAN=0 \
  FORCE_TRAIN=1 \
  FORCE_APPLY=1 \
  RUN_EVAL=1 \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  bash "$VGGT_ROOT/run_tum10_aor_monitor_patch.sh"
}

run_one screen_display_refined_base 1000 0.002 0.05 1
run_one screen_display_refined_ref002 1000 0.002 0.02 1
run_one screen_display_refined_lr003 1000 0.003 0.05 1
run_one screen_display_refined_2000 2000 0.002 0.05 1
run_one screen_display_refined_noeot 1000 0.002 0.05 0

echo "===== done screen-display strengthen sweep ====="

