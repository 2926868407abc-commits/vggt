#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
STEPS="${1:-20}"
GPU="${2:-0}"
RUN="${3:-universal_all8_joint_cold_s0_u${STEPS}}"
WARMUP="${UNIVERSAL_WARMUP_ITERATIONS:-2}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep the human-readable "[train-geometry] update ..." lines visible in
# redirected log files while training is still running.
export PYTHONUNBUFFERED=1
export TUM_ROOT=/mnt/data/wangqq/recons_eval/data/tum_all_sync
export TUM10_FRAME_MANIFEST="$V/data/tum_universal_all/universal_train_preview_manifest.json"
export GT_NAME=groundtruth.txt
export SCENE_PATTERN='rgbd_dataset_freiburg3_*'
export TUM_GEOM_RUN_NAME="$RUN"

export ITERATIONS="$STEPS"
export INNER_LOOP=1
export SCENES_PER_ITERATION=1
export PATCH_LR=0.002
export SCHEDULER=cosine
export WARMUP_ITERATIONS="$WARMUP"
export TEXTURE_SIZE=64
export TEXTURE_INIT=image
export TEXTURE_INIT_IMAGE="$V/assets/hazard_textures/mde_attack_warnning.png"

export ATTACK_LOSS=mirage_projected_split_pose
export MIRAGE_PROJECTED_GT_DIR="$V/outputs/tum_gt_pool_universal_all_firsthalf"
export MIRAGE_PROJECTED_MIN_QUERIES=20
export MIRAGE_PROJECTED_TASKS=''
export DYNAMIC_FRAME_POOL=''
export DYNAMIC_FRAME_POOL_JSON="$V/data/tum_universal_all/universal_train_firsthalf_pools.json"
export DYNAMIC_SETS_PER_STEP=1
export DYNAMIC_FRAMES_PER_SET=10
export SUBSET_AGG=mean

export PLANE_MODE=explicit_world_quad
export EXPLICIT_QUAD_WORLD="$(tr -d '\n\r' < "$V/data/tum10_split/plane_quad_world.txt")"
export USE_DEPTH_VISIBILITY=1
export VISIBILITY_DEPTH_MARGIN=0.08

export TV_WEIGHT=0.001
export PRINTABILITY_WEIGHT=0.001
export PRINTABLE_COLOR_LEVELS=2
export NATURAL_REFERENCE_IMAGE="$V/assets/hazard_textures/mde_attack_warnning.png"
export NATURAL_REFERENCE_WEIGHT=0.05
export PHYSICAL_EOT=0
export SEED=0

export SKIP_TUM_PREPARE=1
export SKIP_CLEAN_VGGT=1
export RUN_EVAL=0
export RUN_GAUGE_DIAG=0
export RUN_CONSISTENCY_CHECK=0

cd "$V"
exec bash "$V/run_geometry_aware_tum10.sh"
