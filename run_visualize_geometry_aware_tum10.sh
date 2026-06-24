#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"

GEOMETRY_OUTPUT_ROOT="${GEOMETRY_OUTPUT_ROOT:-$VGGT_ROOT/outputs_attack_geometry_aware_tum10/tum10_vggt_pointmap_geometry_feature_l3}"
VIS_OUT_DIR="${VIS_OUT_DIR:-$VGGT_ROOT/outputs_attack_geometry_aware_tum10/visualizations_vggt_pointmap}"
VIS_SCENE_PATTERN="${VIS_SCENE_PATTERN:-rgbd_dataset_freiburg3_*}"
VIS_FRAMES="${VIS_FRAMES:-all}"
VIS_ALPHA="${VIS_ALPHA:-0.9}"
VIS_CONTACT_COLUMNS="${VIS_CONTACT_COLUMNS:-5}"
VIS_THUMB_WIDTH="${VIS_THUMB_WIDTH:-260}"
VIS_CONTACT_SHEET_ONLY="${VIS_CONTACT_SHEET_ONLY:-0}"
VIS_TEXTURE_PATH="${VIS_TEXTURE_PATH:-}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }
}

log "check paths"
require_file "$VGGT_PY"
require_file "$VGGT_ROOT/scripts/visualize_tum10_geometry_patch.py"
require_dir "$GEOMETRY_OUTPUT_ROOT"
if [[ -n "$VIS_TEXTURE_PATH" ]]; then
  require_file "$VIS_TEXTURE_PATH"
else
  require_file "$GEOMETRY_OUTPUT_ROOT/geometry_patch/geometry_patch_texture.png"
fi

log "visualize geometry-aware patch"
echo "geometry_output_root=$GEOMETRY_OUTPUT_ROOT"
echo "out_dir=$VIS_OUT_DIR"
echo "scene_pattern=$VIS_SCENE_PATTERN"
echo "frames=$VIS_FRAMES"
echo "contact_sheet_only=$VIS_CONTACT_SHEET_ONLY"
echo "texture_path=${VIS_TEXTURE_PATH:-$GEOMETRY_OUTPUT_ROOT/geometry_patch/geometry_patch_texture.png}"

visualization_args=()
if [[ "$VIS_CONTACT_SHEET_ONLY" == "1" ]]; then
  visualization_args+=(--contact_sheet_only)
fi
if [[ -n "$VIS_TEXTURE_PATH" ]]; then
  visualization_args+=(--texture_path "$VIS_TEXTURE_PATH")
fi

"$VGGT_PY" "$VGGT_ROOT/scripts/visualize_tum10_geometry_patch.py" \
  --geometry_output_root "$GEOMETRY_OUTPUT_ROOT" \
  --out_dir "$VIS_OUT_DIR" \
  --scene_pattern "$VIS_SCENE_PATTERN" \
  --frames "$VIS_FRAMES" \
  --alpha "$VIS_ALPHA" \
  --contact_columns "$VIS_CONTACT_COLUMNS" \
  --thumb_width "$VIS_THUMB_WIDTH" \
  "${visualization_args[@]}"

log "all done"
echo "Patch texture: $VIS_OUT_DIR/geometry_patch_texture.png"
echo "Contact sheets: $VIS_OUT_DIR/<sequence>/contact_sheet_patch.png"
if [[ "$VIS_CONTACT_SHEET_ONLY" != "1" ]]; then
  echo "Overlay frames: $VIS_OUT_DIR/<sequence>/overlay"
  echo "Outline frames: $VIS_OUT_DIR/<sequence>/outline"
fi
