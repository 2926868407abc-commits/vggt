#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
BASE="$VGGT_ROOT/outputs_attack_geometry_aware_tum10"
UPDATES="${UPDATES:-1000}"
OUT_BASE="${OUT_BASE:-$BASE/hazard_monitor_cover_visualizations}"

visualize_one() {
  local tag="$1"
  local name="tum10_sitting_static_hazard_monitor_cover_${tag}_${UPDATES}"
  local root="$BASE/$name"

  echo "===== visualize monitor cover $tag before ====="
  GEOMETRY_OUTPUT_ROOT="$root" \
  VIS_TEXTURE_PATH="$root/geometry_patch/initial_texture.png" \
  VIS_OUT_DIR="$OUT_BASE/${name}_before" \
  VIS_SCENE_PATTERN="rgbd_dataset_freiburg3_sitting_static" \
  VIS_FRAMES="all" \
  VIS_CONTACT_SHEET_ONLY=1 \
  bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"

  echo "===== visualize monitor cover $tag after ====="
  GEOMETRY_OUTPUT_ROOT="$root" \
  VIS_TEXTURE_PATH="$root/geometry_patch/geometry_patch_texture.png" \
  VIS_OUT_DIR="$OUT_BASE/${name}_after" \
  VIS_SCENE_PATTERN="rgbd_dataset_freiburg3_sitting_static" \
  VIS_FRAMES="all" \
  VIS_CONTACT_SHEET_ONLY=1 \
  bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"
}

if [[ "${RUN_CONSERVATIVE:-1}" == "1" ]]; then
  visualize_one conservative
fi
if [[ "${RUN_FULL:-1}" == "1" ]]; then
  visualize_one full
fi
if [[ "${RUN_FULLWIDE:-0}" == "1" ]]; then
  visualize_one fullwide
fi

echo "===== done monitor cover visualizations ====="
echo "$OUT_BASE"
