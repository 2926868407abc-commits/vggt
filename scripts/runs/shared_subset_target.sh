#!/usr/bin/env bash
# Reproduce the 20-frame medium-viewpoint shared-target experiment. Each update
# draws balanced 10-frame subsets; every physical frame keeps the same solved
# deformation target wherever it appears.
set -Eeuo pipefail

VGGT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$VGGT_ROOT"

QUAD_FILE="${QUAD_FILE:-$VGGT_ROOT/data/tum10_split/plane_quad_world.txt}"
SUBSET_PLAN_FILE="${SUBSET_PLAN_FILE:-$VGGT_ROOT/data/tum10_split/mid_subset_plan.json}"
DEFORMATION_FILE="${DEFORMATION_FILE:-$VGGT_ROOT/data/tum10_split/shared_deformation.npz}"

for required in "$QUAD_FILE" "$SUBSET_PLAN_FILE" "$DEFORMATION_FILE"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done

export TUM_GEOM_RUN_NAME="${RUN_NAME:-shared_pose_s0_1000}"
export SCENE_PATTERN="${SCENE_PATTERN:-rgbd_dataset_freiburg3_sitting_halfsphere_mid20}"
export SUBSET_PLAN="$SUBSET_PLAN_FILE"
export SUBSET_ACCUM="${SUBSET_ACCUM:-4}"
export TUM10_FRAME_MANIFEST="${TUM10_FRAME_MANIFEST:-$VGGT_ROOT/data/tum10_split/mid00_manifest.json}"
export TUM_CLEAN_OUT="${TUM_CLEAN_OUT:-$VGGT_ROOT/outputs_attack_geometry_aware_tum10/clean_mid00}"
export TUM_CLEAN_MODEL="${TUM_CLEAN_MODEL:-vggt_mid00_clean}"
export MIRAGE_PROJECTED_GT_DIR="${MIRAGE_PROJECTED_GT_DIR:-$VGGT_ROOT/outputs/tum_gt_point_track_mid00}"

export ATTACK_LOSS="${STAGE_LOSS:-pose_piecewise_gauge_targeted}"
export SHARED_DEFORMATION=1
export SHARED_DEFORMATION_FILE="$DEFORMATION_FILE"
export PIECEWISE_GAUGE_FAMILY=orthogonal_mode
export PIECEWISE_GAUGE_MAGNITUDE="${PIECEWISE_GAUGE_MAGNITUDE:-0.30}"
export ITERATIONS="${STEPS:-1000}"
export INNER_LOOP=1
export SCENES_PER_ITERATION=1

export TEXTURE_SIZE="${TEXTURE_SIZE:-64}"
export TEXTURE_INIT=image
export TEXTURE_INIT_IMAGE="${TEXTURE_INIT_IMAGE:-$VGGT_ROOT/assets/hazard_textures/mde_attack_warnning.png}"
export PATCH_LR="${PATCH_LR:-0.002}"
export POSE_ROTATION_WEIGHT="${POSE_ROTATION_WEIGHT:-5.0}"
export POSE_TRANSLATION_WEIGHT="${POSE_TRANSLATION_WEIGHT:-1.0}"
export MIRAGE_PROJECTED_DIRECTION_SCALE="${MIRAGE_PROJECTED_DIRECTION_SCALE:-0.004719185642898083}"
export JOINT_POINT_STRIDE="${JOINT_POINT_STRIDE:-8}"
export JOINT_TRACK_ITERS="${JOINT_TRACK_ITERS:-4}"

export PLANE_MODE=explicit_world_quad
export EXPLICIT_QUAD_WORLD="$(<"$QUAD_FILE")"
export USE_DEPTH_VISIBILITY=1
export OPTIMIZE_GEOMETRY=1
export SURFACE_SUPPORT_CHECK=0
export VISIBILITY_DEPTH_MARGIN="${VISIBILITY_DEPTH_MARGIN:-0.08}"

export PHYSICAL_EOT=0
export TV_WEIGHT="${TV_WEIGHT:-0.001}"
export PRINTABILITY_WEIGHT="${PRINTABILITY_WEIGHT:-0.001}"
export PRINTABLE_COLOR_LEVELS="${PRINTABLE_COLOR_LEVELS:-2}"
export LOW_FREQUENCY_WEIGHT="${LOW_FREQUENCY_WEIGHT:-0.0}"
export NATURAL_REFERENCE_WEIGHT="${NATURAL_REFERENCE_WEIGHT:-0.05}"
export NATURAL_REFERENCE_IMAGE="${NATURAL_REFERENCE_IMAGE:-$TEXTURE_INIT_IMAGE}"

export FORCE_TRAIN=1
export FORCE_APPLY=1
export RUN_EVAL=0
export RUN_GAUGE_DIAG=0
export RUN_CONSISTENCY_CHECK=0
export FREEZE_MODEL_PARAMETERS=1
export SEED="${SEED:-0}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec bash "$VGGT_ROOT/run_geometry_aware_tum10.sh"
