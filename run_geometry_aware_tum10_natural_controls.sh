#!/usr/bin/env bash
set -Eeuo pipefail

# Natural physical-sticker controls. These use the same geometry constraints as
# run_geometry_aware_tum10_natural.sh, but freeze the texture so there is no
# adversarial optimization. They are occlusion/control baselines.

echo "===== random natural sticker control ====="
TUM_GEOM_RUN_NAME="${RANDOM_RUN_NAME:-tum10_vggt_pointmap_natural_random_control_l3}" \
TUM_GEOM_MODEL="${RANDOM_MODEL:-vggt_tum10_vggt_pointmap_natural_random_control_l3}" \
TEXTURE_INIT=random \
FREEZE_TEXTURE=1 \
FORCE_TRAIN=1 \
FORCE_APPLY=1 \
SURFACE_STRENGTH_SEARCH="${CONTROL_SURFACE_STRENGTH_SEARCH:-0}" \
bash "$(dirname "$0")/run_geometry_aware_tum10_natural.sh"

echo "===== gray natural sticker control ====="
TUM_GEOM_RUN_NAME="${GRAY_RUN_NAME:-tum10_vggt_pointmap_natural_gray_control_l3}" \
TUM_GEOM_MODEL="${GRAY_MODEL:-vggt_tum10_vggt_pointmap_natural_gray_control_l3}" \
TEXTURE_INIT=gray \
FREEZE_TEXTURE=1 \
FORCE_TRAIN=1 \
FORCE_APPLY=1 \
SURFACE_STRENGTH_SEARCH="${CONTROL_SURFACE_STRENGTH_SEARCH:-0}" \
bash "$(dirname "$0")/run_geometry_aware_tum10_natural.sh"

echo "===== natural sticker controls done ====="
