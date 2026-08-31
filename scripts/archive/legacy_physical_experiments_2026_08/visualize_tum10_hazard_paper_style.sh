#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
BASE="${BASE:-$VGGT_ROOT/outputs_attack_geometry_aware_tum10}"
NAME="${NAME:-tum10_sitting_static_hazard_paperstyle_autopos_pose_gt_1000}"
ROOT="$BASE/$NAME"
OUT_BASE="${OUT_BASE:-$BASE/hazard_paperstyle_visualizations}"
SCENE="${SCENE:-rgbd_dataset_freiburg3_sitting_static}"

GEOMETRY_OUTPUT_ROOT="$ROOT" \
VIS_TEXTURE_PATH="$ROOT/geometry_patch/initial_texture.png" \
VIS_OUT_DIR="$OUT_BASE/before/$NAME" \
VIS_SCENE_PATTERN="$SCENE" \
VIS_FRAMES=all \
VIS_CONTACT_SHEET_ONLY=1 \
bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"

GEOMETRY_OUTPUT_ROOT="$ROOT" \
VIS_TEXTURE_PATH="$ROOT/geometry_patch/geometry_patch_texture.png" \
VIS_OUT_DIR="$OUT_BASE/after/$NAME" \
VIS_SCENE_PATTERN="$SCENE" \
VIS_FRAMES=all \
VIS_CONTACT_SHEET_ONLY=1 \
bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"

echo "Saved visualizations under: $OUT_BASE"
