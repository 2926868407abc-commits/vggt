#!/usr/bin/env bash
set -Eeuo pipefail

# Grouped before/after contact sheets for the refined monitor-screen sweep.

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
BASE="$VGGT_ROOT/outputs_attack_geometry_aware_tum10"
OUT_BASE="${OUT_BASE:-$BASE/aor_monitor_screen_display_strengthen_visualizations}"
SCRATCH="$OUT_BASE/_scratch"

NAMES=(
  tum10_sitting_static_aor_monitor_screen_display_refined_base_1000
  tum10_sitting_static_aor_monitor_screen_display_refined_ref002_1000
  tum10_sitting_static_aor_monitor_screen_display_refined_lr003_1000
  tum10_sitting_static_aor_monitor_screen_display_refined_2000_2000
  tum10_sitting_static_aor_monitor_screen_display_refined_noeot_1000
)

render_phase() {
  local name="$1"
  local phase="$2"
  local texture="$3"
  local tmp="$SCRATCH/${name}_${phase}"

  rm -rf "$tmp"
  mkdir -p "$tmp"

  GEOMETRY_OUTPUT_ROOT="$BASE/$name" \
  VIS_TEXTURE_PATH="$texture" \
  VIS_OUT_DIR="$tmp" \
  VIS_SCENE_PATTERN="${VIS_SCENE_PATTERN:-rgbd_dataset_freiburg3_sitting_static}" \
  VIS_FRAMES="${VIS_FRAMES:-all}" \
  VIS_CONTACT_SHEET_ONLY=1 \
  bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"

  while IFS= read -r sheet; do
    local scene
    scene="$(basename "$(dirname "$sheet")")"
    local dest="$OUT_BASE/$name/$scene/$phase"
    mkdir -p "$dest"
    cp "$sheet" "$dest/contact_sheet_patch.png"
    cp "$texture" "$dest/patch_texture.png"
  done < <(find "$tmp" -name contact_sheet_patch.png -type f)
}

mkdir -p "$OUT_BASE"
for name in "${NAMES[@]}"; do
  root="$BASE/$name"
  echo "===== grouped visualize $name ====="
  render_phase "$name" before "$root/geometry_patch/initial_texture.png"
  render_phase "$name" after "$root/geometry_patch/geometry_patch_texture.png"
done

echo "===== done screen-display strengthen visualizations ====="
echo "$OUT_BASE"
