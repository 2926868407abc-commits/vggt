#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
PY=/mnt/data/wangqq/conda_envs/recons_eval/bin/python3
STEPS="${1:-200}"
GPU="${2:-0}"
RUN_NAME="${3:-dynfive_pool_smoke_s0_u${STEPS}}"

cd "$V"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX=/mnt/data/wangqq/tmp_pyc
export CUDA_VISIBLE_DEVICES="$GPU"

POOL="$($PY -c 'import json; p=json.load(open("data/tum10_split/eight_seq_pool.json")); print(",".join(map(str,p["dynamic_pool"])))')"

export SCENE_PATTERN=rgbd_dataset_freiburg3_sitting_halfsphere
export TUM10_FRAME_MANIFEST="$V/data/tum10_split/train10_manifest.json"
export TUM_GEOM_RUN_NAME="$RUN_NAME"
export ITERATIONS="$STEPS"
export INNER_LOOP=1
export SCENES_PER_ITERATION=1
export TEXTURE_SIZE=64
export TEXTURE_INIT=image
export TEXTURE_INIT_IMAGE="$V/outputs_attack_geometry_aware_tum10/dynpose_2000/geometry_patch/optimized_patch.png"
export PATCH_LR=0.002
export SCHEDULER=cosine
export WARMUP_ITERATIONS=20

export ATTACK_LOSS=mirage_projected_split_pose
export MIRAGE_PROJECTED_GT_DIR="$V/outputs/tum_gt_pool"
export MIRAGE_PROJECTED_TASKS=
export MIRAGE_PROJECTED_DIRECTION_SCALE=0.0
export MIRAGE_PROJECTED_MIN_QUERIES=20
export JOINT_TRACK_ITERS=2

export DYNAMIC_FRAME_POOL="$POOL"
export DYNAMIC_SETS_PER_STEP=6
export DYNAMIC_FRAMES_PER_SET=10
export SUBSET_PLAN=
export SUBSET_ACCUM=1
export SUBSET_AGG=mean
export SOFTMIN_TRANSFORMS=identity

export PLANE_MODE=explicit_world_quad
export EXPLICIT_QUAD_WORLD="$(cat "$V/data/tum10_split/plane_quad_world.txt")"
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
export AUDIT_PLAN="$V/data/tum10_split/audit_plan.json"
export AUDIT_EVERY=100

exec bash "$V/run_geometry_aware_tum10.sh"
