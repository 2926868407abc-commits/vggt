#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
BASE="$VGGT_ROOT/outputs_attack_geometry_aware_tum10"
OUT_BASE="${OUT_BASE:-$BASE/hazard_wall_refine_visualizations}"

TAGS=(
  area034_pos050_1000_ref005_lr002
  area034_pos052_1000_ref005_lr002
  area034_pos054_1000_ref005_lr002
  area038_pos052_1000_ref005_lr002
  area034_pos052_1000_ref002_lr002
  area034_pos052_1000_ref001_lr002
  area034_pos052_1000_ref002_lr003
  area034_pos052_2000_ref002_lr002
)

visualize_one() {
  local tag="$1"
  local name="tum10_sitting_static_hazard_wall_refine_${tag}"
  local root="$BASE/$name"

  echo "===== visualize wall refine $tag before ====="
  GEOMETRY_OUTPUT_ROOT="$root" \
  VIS_TEXTURE_PATH="$root/geometry_patch/initial_texture.png" \
  VIS_OUT_DIR="$OUT_BASE/${name}_before" \
  VIS_SCENE_PATTERN="rgbd_dataset_freiburg3_sitting_static" \
  VIS_FRAMES="all" \
  VIS_CONTACT_SHEET_ONLY=1 \
  bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"

  echo "===== visualize wall refine $tag after ====="
  GEOMETRY_OUTPUT_ROOT="$root" \
  VIS_TEXTURE_PATH="$root/geometry_patch/geometry_patch_texture.png" \
  VIS_OUT_DIR="$OUT_BASE/${name}_after" \
  VIS_SCENE_PATTERN="rgbd_dataset_freiburg3_sitting_static" \
  VIS_FRAMES="all" \
  VIS_CONTACT_SHEET_ONLY=1 \
  bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"
}

for tag in "${TAGS[@]}"; do
  visualize_one "$tag"
done

echo "===== done wall refine visualizations ====="
echo "$OUT_BASE"
