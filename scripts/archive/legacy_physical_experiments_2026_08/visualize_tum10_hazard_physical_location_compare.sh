#!/usr/bin/env bash
set -Eeuo pipefail

# Grouped visualizations for the physical-location comparison.
# Output layout:
#   OUT_BASE/<run_name>/<scene>/before/contact_sheet_patch.png
#   OUT_BASE/<run_name>/<scene>/after/contact_sheet_patch.png

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
BASE="$VGGT_ROOT/outputs_attack_geometry_aware_tum10"
UPDATES="${UPDATES:-1000}"
OUT_BASE="${OUT_BASE:-$BASE/physical_location_compare_visualizations}"
SCRATCH="$OUT_BASE/_scratch"

TAGS=()
if [[ "${RUN_WALL:-1}" == "1" ]]; then
  TAGS+=(wall_poster_center wall_poster_large)
fi
if [[ "${RUN_MONITOR:-1}" == "1" ]]; then
  TAGS+=(monitor_right_fit monitor_right_full monitor_left_fit)
fi

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

visualize_one() {
  local tag="$1"
  local name="tum10_sitting_static_hazard_physcmp_${tag}_${UPDATES}"
  local root="$BASE/$name"

  echo "===== grouped visualize $name ====="
  render_phase "$name" before "$root/geometry_patch/initial_texture.png"
  render_phase "$name" after "$root/geometry_patch/geometry_patch_texture.png"
}

mkdir -p "$OUT_BASE"
for tag in "${TAGS[@]}"; do
  visualize_one "$tag"
done

echo "===== done grouped physical-location visualizations ====="
echo "$OUT_BASE"
