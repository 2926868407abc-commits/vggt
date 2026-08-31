#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
TEXTURE_ROOT="${TEXTURE_ROOT:-$VGGT_ROOT/assets/hazard_textures}"
HAZARD_TEXTURE="${HAZARD_TEXTURE:-$TEXTURE_ROOT/tum_monitor_initial_texture.png}"

RUN_NAME="${RUN_NAME:-nrgbd_sparse_gt_depth_geometry_tum_hazard_init_feature_l3}"
MODEL_NAME="${MODEL_NAME:-vggt_nrgbd_sparse_gt_depth_geometry_tum_hazard_init_feature_l3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$VGGT_ROOT/outputs_attack_geometry_aware_nrgbd/$RUN_NAME}"
VIS_ROOT="${VIS_ROOT:-$VGGT_ROOT/outputs_attack_geometry_aware_nrgbd/visualizations_$RUN_NAME}"

mkdir -p "$TEXTURE_ROOT"
if [[ ! -f "$HAZARD_TEXTURE" ]]; then
  echo "Missing the exact TUM hazard initialization texture: $HAZARD_TEXTURE" >&2
  exit 1
fi

echo "===== Neural-RGBD TUM-hazard-initialized geometry-aware patch ====="
echo "texture=$HAZARD_TEXTURE"
echo "texture_md5=$(md5sum "$HAZARD_TEXTURE" | awk '{print $1}')"
echo "run=$RUN_NAME"

NRGBD_GEOM_RUN_NAME="$RUN_NAME" \
NRGBD_GEOM_MODEL="$MODEL_NAME" \
TEXTURE_INIT=image \
TEXTURE_INIT_IMAGE="$HAZARD_TEXTURE" \
NATURAL_REFERENCE_IMAGE="$HAZARD_TEXTURE" \
NATURAL_REFERENCE_WEIGHT="${NATURAL_REFERENCE_WEIGHT:-0.05}" \
ITERATIONS="${ITERATIONS:-200}" \
INNER_LOOP="${INNER_LOOP:-10}" \
SCENES_PER_ITERATION="${SCENES_PER_ITERATION:-1}" \
PATCH_LR="${PATCH_LR:-0.001}" \
FEATURE_LAYER="${FEATURE_LAYER:-aggregator_final}" \
ATTACK_LOSS="${ATTACK_LOSS:-feature_l1}" \
PHYSICAL_EOT="${PHYSICAL_EOT:-1}" \
FORCE_PREPARE="${FORCE_PREPARE:-0}" \
FORCE_TRAIN="${FORCE_TRAIN:-1}" \
FORCE_APPLY="${FORCE_APPLY:-1}" \
RUN_EVAL="${RUN_EVAL:-1}" \
bash "$VGGT_ROOT/run_geometry_aware_nrgbd.sh"

echo "===== visualize learned patch and green_room views ====="
GEOMETRY_OUTPUT_ROOT="$OUTPUT_ROOT" \
VIS_OUT_DIR="$VIS_ROOT" \
VIS_SCENE_PATTERN=green_room \
VIS_FRAMES=all \
VIS_ALPHA=1.0 \
VIS_CONTACT_COLUMNS=4 \
VIS_CONTACT_SHEET_ONLY=1 \
VIS_NO_OUTLINE=1 \
VIS_NO_LABEL=1 \
bash "$VGGT_ROOT/run_visualize_geometry_aware_tum10.sh"

echo "===== all done ====="
echo "metric=/mnt/data/wangqq/recons_eval/outputs/mv_recon/NRGBD-sparse-metric-$MODEL_NAME.csv"
echo "patch=$VIS_ROOT/geometry_patch_texture.png"
echo "views=$VIS_ROOT/green_room/contact_sheet_patch.png"
