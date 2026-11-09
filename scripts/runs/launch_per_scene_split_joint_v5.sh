#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
SPLIT="${1:?usage: $0 SPLIT SCENE GPU CONTEXT [STEPS] [RUN_NAME]}"
SCENE="${2:?usage: $0 SPLIT SCENE GPU CONTEXT [STEPS] [RUN_NAME]}"
GPU="${3:?usage: $0 SPLIT SCENE GPU CONTEXT [STEPS] [RUN_NAME]}"
CONTEXT="${4:?usage: $0 SPLIT SCENE GPU CONTEXT [STEPS] [RUN_NAME]}"
STEPS="${5:-1000}"
SHORT="${SCENE#rgbd_dataset_freiburg3_}"
RUN="${6:-perseq_v5_k${CONTEXT}_${SPLIT}_${SHORT}_joint_cold_s0_u${STEPS}}"

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
export VGGT_PY=/mnt/data/wangqq/conda_envs/vggt/bin/python3
# Training-only runs do not invoke recons_eval.  Reuse the VGGT interpreter on
# compact AutoDL images where a separate recons_eval environment is absent.
export RECONS_PY="$VGGT_PY"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
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
export SCHEDULER="${PER_SCENE_SCHEDULER:-cosine}"
export WARMUP_ITERATIONS="${PER_SCENE_WARMUP_ITERATIONS:-20}"
export ADAPTIVE_LR_FACTOR="${PER_SCENE_ADAPTIVE_LR_FACTOR:-0.5}"
export ADAPTIVE_LR_PATIENCE="${PER_SCENE_ADAPTIVE_LR_PATIENCE:-3}"
export ADAPTIVE_STOP_PATIENCE="${PER_SCENE_ADAPTIVE_STOP_PATIENCE:-5}"
export ADAPTIVE_MIN_LR="${PER_SCENE_ADAPTIVE_MIN_LR:-0.00001}"
export ADAPTIVE_IMPROVEMENT_THRESHOLD="${PER_SCENE_ADAPTIVE_IMPROVEMENT_THRESHOLD:-0.005}"
export TEXTURE_SIZE=64
export TEXTURE_INIT=image
export TEXTURE_INIT_IMAGE="${PER_SCENE_TEXTURE_INIT_IMAGE:-$V/assets/hazard_textures/mde_attack_warnning.png}"

export ATTACK_LOSS=mirage_projected_split_pose
export MIRAGE_PROJECTED_GT_DIR="$GT_POOL"
export MIRAGE_PROJECTED_MIN_QUERIES=20
export MIRAGE_PROJECTED_MAX_QUERIES="${PER_SCENE_MIRAGE_PROJECTED_MAX_QUERIES:-0}"
export MIRAGE_PROJECTED_TASKS="${PER_SCENE_MIRAGE_PROJECTED_TASKS:-}"
export MIRAGE_PROJECTED_TRACK_MODE="${PER_SCENE_MIRAGE_PROJECTED_TRACK_MODE:-capped_all}"
export MIRAGE_PROJECTED_TRACK_MIN_POINTS_PER_FRAME="${PER_SCENE_MIRAGE_PROJECTED_TRACK_MIN_POINTS_PER_FRAME:-20}"
export MIRAGE_PROJECTED_TRACK_TARGET_SOFTMIN_TAU_PX="${PER_SCENE_MIRAGE_PROJECTED_TRACK_TARGET_SOFTMIN_TAU_PX:-0.25}"
export PROJECTED_SUBSET_AGG="${PER_SCENE_PROJECTED_SUBSET_AGG:-mean}"
export PROJECTED_SUBSET_SOFTMIN_TAU_PX="${PER_SCENE_PROJECTED_SUBSET_SOFTMIN_TAU_PX:-0.25}"
export JOINT_TRACK_ITERS=2
export DYNAMIC_FRAME_POOL=''
export DYNAMIC_FRAME_POOL_JSON="$POOLS"
export DYNAMIC_SETS_PER_STEP="${PER_SCENE_DYNAMIC_SETS_PER_STEP:-1}"
export DYNAMIC_FRAMES_PER_SET="$CONTEXT"
export DYNAMIC_QUERY_SAMPLING="${PER_SCENE_DYNAMIC_QUERY_SAMPLING:-earliest}"
export SUBSET_PLAN=''
export SUBSET_ACCUM=1
export SUBSET_AGG=mean
export SOFTMIN_TRANSFORMS=identity

export PLANE_MODE=explicit_world_quad
export EXPLICIT_QUAD_WORLD="$(tr -d '\n\r' < "$V/assets/visibility_fix/sitting_xyz_monitor_on_stable_plane_v5.txt")"
export OPTIMIZE_GEOMETRY=0
export USE_DEPTH_VISIBILITY=1
export VISIBILITY_DEPTH_MARGIN=0.08

export TV_WEIGHT=0.001
export PRINTABILITY_WEIGHT=0.001
export PRINTABLE_COLOR_LEVELS=2
export NATURAL_REFERENCE_IMAGE="$V/assets/hazard_textures/mde_attack_warnning.png"
export NATURAL_REFERENCE_WEIGHT=0.05
export PHYSICAL_EOT=0
export SEED="${PER_SCENE_SEED:-0}"

export SKIP_TUM_PREPARE=1
export SKIP_CLEAN_VGGT=1
export FORCE_TRAIN=1
export CHECKPOINT_EVERY="${PER_SCENE_CHECKPOINT_EVERY:-10}"
export AUTO_RESUME="${PER_SCENE_AUTO_RESUME:-1}"
if (( CONTEXT >= 30 )); then
  activation_checkpoint_default=1
else
  activation_checkpoint_default=0
fi
export ACTIVATION_CHECKPOINT="${PER_SCENE_ACTIVATION_CHECKPOINT:-$activation_checkpoint_default}"
export ACTIVATION_CHECKPOINT_MODE="${PER_SCENE_ACTIVATION_CHECKPOINT_MODE:-global}"
export RUN_EVAL=0
export RUN_GAUGE_DIAG=0
export RUN_CONSISTENCY_CHECK=0
export AUDIT_PLAN="${PER_SCENE_AUDIT_PLAN:-}"
export AUDIT_EVERY="${PER_SCENE_AUDIT_EVERY:-0}"

cd "$V"
exec bash "$V/run_geometry_aware_tum10.sh"
