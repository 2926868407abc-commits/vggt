#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
SPLIT="${1:?usage: $0 SPLIT SCENE GPU [STEPS] [RUN_NAME]}"
SCENE="${2:?usage: $0 SPLIT SCENE GPU [STEPS] [RUN_NAME]}"
GPU="${3:?usage: $0 SPLIT SCENE GPU [STEPS] [RUN_NAME]}"
STEPS="${4:-1000}"
SHORT="${SCENE#rgbd_dataset_freiburg3_}"
RUN="${5:-perseq_${SPLIT}_${SHORT}_joint_cold_s0_u${STEPS}}"

case "$SPLIT" in
  firsthalf)
    POOLS="$V/data/tum_per_scene_splits/firsthalf_train_pools.json"
    MANIFEST="$V/data/tum_per_scene_splits/firsthalf_train_preview_manifest.json"
    GT_POOL="$V/outputs/tum_gt_pool_universal_all_firsthalf"
    ;;
  evenodd)
    POOLS="$V/data/tum_per_scene_splits/evenodd_train_pools.json"
    MANIFEST="$V/data/tum_per_scene_splits/evenodd_train_preview_manifest.json"
    GT_POOL="$V/outputs/tum_gt_pool_per_scene_even"
    ;;
  *)
    echo "unknown split: $SPLIT" >&2
    exit 2
    ;;
esac

test -f "$POOLS"
test -f "$MANIFEST"
test -f "$GT_POOL/$SCENE/pool_meta.json"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX=/mnt/data/wangqq/tmp_pyc
export TUM_ROOT=/mnt/data/wangqq/recons_eval/data/tum_all_sync
export TUM10_FRAME_MANIFEST="$MANIFEST"
export GT_NAME=groundtruth.txt
export SCENE_PATTERN="$SCENE"
export TUM_GEOM_RUN_NAME="$RUN"

export ITERATIONS="$STEPS"
export INNER_LOOP=1
export SCENES_PER_ITERATION=1
export PATCH_LR=0.002
export SCHEDULER=cosine
export WARMUP_ITERATIONS="${PER_SCENE_WARMUP_ITERATIONS:-20}"
export TEXTURE_SIZE=64
export TEXTURE_INIT=image
export TEXTURE_INIT_IMAGE="$V/assets/hazard_textures/mde_attack_warnning.png"

export ATTACK_LOSS=mirage_projected_split_pose
export MIRAGE_PROJECTED_GT_DIR="$GT_POOL"
export MIRAGE_PROJECTED_MIN_QUERIES=20
export MIRAGE_PROJECTED_TASKS=''
export JOINT_TRACK_ITERS=2
export DYNAMIC_FRAME_POOL=''
export DYNAMIC_FRAME_POOL_JSON="$POOLS"
export DYNAMIC_SETS_PER_STEP=1
export DYNAMIC_FRAMES_PER_SET=10
export SUBSET_PLAN=''
export SUBSET_ACCUM=1
export SUBSET_AGG=mean
export SOFTMIN_TRANSFORMS=identity

export PLANE_MODE=explicit_world_quad
export EXPLICIT_QUAD_WORLD="$(tr -d '\n\r' < "$V/data/tum10_split/plane_quad_world.txt")"
export OPTIMIZE_GEOMETRY=0
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
export FORCE_TRAIN=1
export RUN_EVAL=0
export RUN_GAUGE_DIAG=0
export RUN_CONSISTENCY_CHECK=0
export AUDIT_PLAN=''

cd "$V"
exec bash "$V/run_geometry_aware_tum10.sh"
