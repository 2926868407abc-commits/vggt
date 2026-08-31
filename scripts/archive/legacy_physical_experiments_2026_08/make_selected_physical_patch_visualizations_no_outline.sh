#!/usr/bin/env bash
set -Eeuo pipefail

# Produce final, presentation-friendly visualizations for selected reasonable
# physical patch runs. These are actual composited inputs: no red debug outline
# and no black frame label. Output layout:
#   final_physical_patch_visualizations_no_outline/
#     wall/
#       patch_texture.png
#       after/contact_sheet_patch.png
#       after/overlay/*.png
#     monitor/
#       patch_texture.png
#       after/contact_sheet_patch.png
#       after/overlay/*.png

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
BASE="$VGGT_ROOT/outputs_attack_geometry_aware_tum10"
OUT_BASE="${OUT_BASE:-$BASE/final_physical_patch_visualizations_no_outline}"

make_one() {
  local label="$1"
  local run_name="$2"
  local out_dir="$OUT_BASE/$label"

  echo "===== render $label: $run_name ====="
  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  cp "$BASE/$run_name/geometry_patch/geometry_patch_texture.png" "$out_dir/patch_texture.png"
  cp "$BASE/$run_name/geometry_patch/initial_texture.png" "$out_dir/initial_texture.png"

  GEOMETRY_OUTPUT_ROOT="$BASE/$run_name" \
  VIS_OUT_DIR="$out_dir/after" \
  VIS_SCENE_PATTERN="${VIS_SCENE_PATTERN:-rgbd_dataset_freiburg3_sitting_static}" \
  VIS_FRAMES="${VIS_FRAMES:-all}" \
  VIS_ALPHA="${VIS_ALPHA:-1.0}" \
  VIS_CONTACT_COLUMNS="${VIS_CONTACT_COLUMNS:-5}" \
  VIS_THUMB_WIDTH="${VIS_THUMB_WIDTH:-260}" \
  VIS_CONTACT_SHEET_ONLY=0 \
  VIS_NO_OUTLINE=1 \
  VIS_NO_LABEL=1 \
  VIS_NO_OUTLINE_FRAMES=1 \
  bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"
}

make_one wall tum10_sitting_static_hazard_physcmp_wall_poster_center_1000
make_one monitor tum10_sitting_static_aor_monitor_screen_display_refined_2000_2000

echo "===== done final no-outline visualizations ====="
echo "$OUT_BASE"
