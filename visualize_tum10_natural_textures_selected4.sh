#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
BASE="${BASE:-$VGGT_ROOT/outputs_attack_geometry_aware_tum10}"
OUT_BASE="${OUT_BASE:-$BASE/natural_texture_selected4_visualizations}"
SCENE="${SCENE:-rgbd_dataset_freiburg3_sitting_static}"

visualize_one() {
  local name="$1"
  local root="$BASE/$name"

  GEOMETRY_OUTPUT_ROOT="$root" \
  VIS_TEXTURE_PATH="$root/geometry_patch/initial_texture.png" \
  VIS_OUT_DIR="$OUT_BASE/before/$name" \
  VIS_SCENE_PATTERN="$SCENE" \
  VIS_FRAMES=all \
  VIS_CONTACT_SHEET_ONLY=1 \
  bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"

  GEOMETRY_OUTPUT_ROOT="$root" \
  VIS_TEXTURE_PATH="$root/geometry_patch/geometry_patch_texture.png" \
  VIS_OUT_DIR="$OUT_BASE/after/$name" \
  VIS_SCENE_PATTERN="$SCENE" \
  VIS_FRAMES=all \
  VIS_CONTACT_SHEET_ONLY=1 \
  bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"
}

visualize_one tum10_sitting_static_wall_pose_gt_dtd_woven_0003_natural_lr002_ref005
visualize_one tum10_sitting_static_wall_pose_gt_dtd_banded_0009_natural_lr002_ref005
visualize_one tum10_sitting_static_wall_pose_gt_dtd_blotchy_0006_natural_lr002_ref005
visualize_one tum10_sitting_static_wall_pose_gt_dtd_woven_0001_natural_lr002_ref005

echo "Saved visualizations under: $OUT_BASE"
