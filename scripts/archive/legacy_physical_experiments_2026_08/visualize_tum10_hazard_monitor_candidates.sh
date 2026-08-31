#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
BASE="$VGGT_ROOT/outputs_attack_geometry_aware_tum10"
UPDATES="${UPDATES:-1000}"
OUT_BASE="${OUT_BASE:-$BASE/hazard_monitor_candidate_visualizations}"

visualize_one() {
  local tag="$1"
  local name="tum10_sitting_static_hazard_monitor_${tag}_${UPDATES}"
  local root="$BASE/$name"

  echo "===== visualize monitor $tag before ====="
  GEOMETRY_OUTPUT_ROOT="$root" \
  VIS_TEXTURE_PATH="$root/geometry_patch/initial_texture.png" \
  VIS_OUT_DIR="$OUT_BASE/${name}_before" \
  VIS_SCENE_PATTERN="rgbd_dataset_freiburg3_sitting_static" \
  VIS_FRAMES="all" \
  VIS_CONTACT_SHEET_ONLY=1 \
  bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"

  echo "===== visualize monitor $tag after ====="
  GEOMETRY_OUTPUT_ROOT="$root" \
  VIS_TEXTURE_PATH="$root/geometry_patch/geometry_patch_texture.png" \
  VIS_OUT_DIR="$OUT_BASE/${name}_after" \
  VIS_SCENE_PATTERN="rgbd_dataset_freiburg3_sitting_static" \
  VIS_FRAMES="all" \
  VIS_CONTACT_SHEET_ONLY=1 \
  bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"
}

visualize_one screen_small
visualize_one screen_medium

echo "===== done monitor visualizations ====="
echo "$OUT_BASE"
