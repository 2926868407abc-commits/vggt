#!/usr/bin/env bash
# Controlled main-table adaptation of the four paper baselines.
#
# All methods use the same frozen v5 monitor carrier and the same first-half
# dynamic training pool.  Paper-specific texture initialisation, objective,
# optimiser, and published EOT choices are retained.  Source-faithful
# placement/area experiments remain in scripts/runs/paper_baselines.sh.
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
METHOD="${1:?usage: $0 METHOD SCENE GPU [STEPS] [CONTEXT] [RUN_NAME]}"
SCENE="${2:?usage: $0 METHOD SCENE GPU [STEPS] [CONTEXT] [RUN_NAME]}"
GPU="${3:?usage: $0 METHOD SCENE GPU [STEPS] [CONTEXT] [RUN_NAME]}"
STEPS="${4:-1000}"
CONTEXT="${5:-10}"
SHORT="${SCENE#rgbd_dataset_freiburg3_}"

case "$METHOD" in
  spoo)
    ATTACK_LOSS=baseline_spoo_depth
    SEED=17
    ;;
  aparate)
    ATTACK_LOSS=baseline_aparate_depth
    SEED=0
    ;;
  3d2fool)
    ATTACK_LOSS=baseline_3d2fool_depth
    SEED=0
    ;;
  passive_vo)
    ATTACK_LOSS=baseline_vo_mprms
    SEED=0
    ;;
  *)
    echo "unknown controlled baseline: $METHOD" >&2
    exit 2
    ;;
esac

RUN="${6:-controlled_v5_k${CONTEXT}_firsthalf_${SHORT}_${METHOD}_s${SEED}_u${STEPS}}"
POOLS="$V/data/tum_per_scene_splits/firsthalf_train_pools.json"
MANIFEST="$V/data/tum_per_scene_splits/firsthalf_train_preview_manifest.json"
QUAD_FILE="$V/assets/visibility_fix/sitting_xyz_monitor_on_stable_plane_v5.txt"

test -f "$POOLS"
test -f "$MANIFEST"
test -f "$QUAD_FILE"
QUAD="$(tr -d '\n\r' < "$QUAD_FILE")"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"
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
export DYNAMIC_FRAME_POOL=''
export DYNAMIC_FRAME_POOL_JSON="$POOLS"
export DYNAMIC_SETS_PER_STEP=1
export DYNAMIC_FRAMES_PER_SET="$CONTEXT"
export SUBSET_PLAN=''
export SUBSET_ACCUM=1
export SUBSET_AGG=mean

export ATTACK_LOSS
export SEED
export TEXTURE_SIZE=64
export TEXTURE_INIT=random
export TEXTURE_INIT_IMAGE=''
export TEXTURE_INIT_PRESERVE_ASPECT=0
export PATCH_LR=0.01
export SCHEDULER="${CONTROLLED_SCHEDULER:-none}"
export WARMUP_ITERATIONS=0
export AUDIT_PLAN="${CONTROLLED_AUDIT_PLAN:-}"
export AUDIT_EVERY="${CONTROLLED_AUDIT_EVERY:-0}"
export ADAPTIVE_LR_FACTOR="${CONTROLLED_ADAPTIVE_LR_FACTOR:-0.5}"
export ADAPTIVE_LR_PATIENCE="${CONTROLLED_ADAPTIVE_LR_PATIENCE:-3}"
export ADAPTIVE_STOP_PATIENCE="${CONTROLLED_ADAPTIVE_STOP_PATIENCE:-5}"
export ADAPTIVE_MIN_LR="${CONTROLLED_ADAPTIVE_MIN_LR:-0.00001}"
export ADAPTIVE_IMPROVEMENT_THRESHOLD="${CONTROLLED_ADAPTIVE_IMPROVEMENT_THRESHOLD:-0.005}"
export BASELINE_SPOO_FORCE_SECOND_STAGE="${CONTROLLED_SPOO_FORCE_SECOND_STAGE:-0}"
export BASELINE_SPOO_LATE_START_FRACTION="${CONTROLLED_SPOO_LATE_START_FRACTION:-0.5}"

# The controlled comparison freezes the physical support: every baseline fills
# exactly the same monitor quad.  SPOO's source region search is therefore
# disabled only in this controlled profile, while its texture loss/optimiser is
# preserved.  The source 1/9 region-search result remains an appendix baseline.
export PLANE_MODE=explicit_world_quad
export EXPLICIT_QUAD_WORLD="$QUAD"
export BASELINE_FOCUS_QUAD_WORLD="$QUAD"
export BASELINE_FOCUS_SCALE=1.0
export OPTIMIZE_GEOMETRY=0
export USE_DEPTH_VISIBILITY=1
export VISIBILITY_DEPTH_MARGIN=0.08
export SURFACE_SUPPORT_CHECK=1
export SURFACE_SUPPORT_ABS_TOLERANCE=0.06
export SURFACE_SUPPORT_REL_TOLERANCE=0.02
export SURFACE_MIN_SUPPORT_RATIO=0.50

# Disable repository-specific regularisers. Each method adds only its own
# published regularisation below/in the attack implementation.
export TV_WEIGHT=0
export PRINTABILITY_WEIGHT=0
export LOW_FREQUENCY_WEIGHT=0
export NATURAL_REFERENCE_WEIGHT=0

# Shared paper-baseline defaults.
export PHYSICAL_EOT=1
export EOT_SAMPLES=1
export EOT_BRIGHTNESS=0.10
export EOT_CONTRAST=0.20
export EOT_SATURATION=0.0
export EOT_HUE=0.0
export EOT_GAMMA=0.0
export EOT_NOISE_STD=0.10
export EOT_WARMUP_FRACTION=0.0
export EOT_GEO_ROTATE_DEGREES=20
export EOT_GEO_SCALE=0.03
export EOT_GEO_TRANSLATE=0.0
export EOT_GEO_PERSPECTIVE=0.0
export BASELINE_WEATHER_STRENGTH=0.0
export BASELINE_TEXTURE_REPEATS=6
export BASELINE_CONVERTED_TEXTURE_SIZE=4096
export BASELINE_TEXTURE_UPSAMPLE_FACTOR=8

case "$METHOD" in
  spoo)
    export TEXTURE_INIT=image
    export TEXTURE_INIT_IMAGE="$V/assets/hazard_textures/mde_attack_warnning.png"
    export TEXTURE_SIZE=600
    export TEXTURE_INIT_PRESERVE_ASPECT=1
    export PATCH_LR=0.3689
    export BASELINE_SPOO_NATURALNESS=1
    export BASELINE_SPOO_NATURALNESS_REFERENCE_IMAGE="$V/assets/hazard_textures/mde_attack_warnning.png"
    export BASELINE_SPOO_TARGET_AREA_FRACTION=1.0
    export BASELINE_SPOO_BOTTOM_MARGIN_PIXELS=0
    export BASELINE_SPOO_MASK_WEIGHT=0.0
    export BASELINE_SPOO_MASK_SHARPNESS=40.0
    export BASELINE_SPOO_BORDER_LR=0.0
    export BASELINE_SPOO_MASK_UPDATE_INTERVAL=2
    export PHYSICAL_EOT=0
    ;;
  aparate)
    # Paper: random RGB patch, Adam 0.01, colour/noise EOT and +/-20 degrees.
    export PATCH_LR=0.01
    ;;
  3d2fool)
    # Paper: 128x128 random seed, repeated/upsampled texture conversion.
    export TEXTURE_SIZE=128
    export PATCH_LR=0.01
    export EOT_BRIGHTNESS=0.20
    export EOT_CONTRAST=0.10
    export EOT_SATURATION=0.10
    export EOT_HUE=0.05
    export EOT_NOISE_STD=0.0
    export EOT_GEO_ROTATE_DEGREES=0
    export EOT_GEO_SCALE=0.0
    ;;
  passive_vo)
    # Paper: random RGB, 9x9 pre-warp blur, Linf PGD-sign, no generic EOT.
    export PATCH_LR=0.00392156862745098
    export BASELINE_VO_BLUR_KERNEL=9
    export PHYSICAL_EOT=0
    ;;
esac

# A formal two-stage SPOO run starts phase two from phase one's restored best
# texture.  Keep the paper's warning-image initializer when no override exists.
if [[ -n "${CONTROLLED_TEXTURE_INIT_IMAGE:-}" ]]; then
  export TEXTURE_INIT=image
  export TEXTURE_INIT_IMAGE="$CONTROLLED_TEXTURE_INIT_IMAGE"
  export TEXTURE_INIT_PRESERVE_ASPECT=1
fi

export SKIP_TUM_PREPARE=1
export SKIP_CLEAN_VGGT=1
export FORCE_TRAIN=1
export FORCE_APPLY=1
export CHECKPOINT_EVERY="${CONTROLLED_CHECKPOINT_EVERY:-10}"
export AUTO_RESUME="${CONTROLLED_AUTO_RESUME:-1}"
export RUN_EVAL=0
export RUN_GAUGE_DIAG=0
export RUN_CONSISTENCY_CHECK=0

mkdir -p "$V/logs/controlled_baselines"
echo "[controlled-baseline] run=$RUN method=$METHOD scene=$SCENE context=$CONTEXT updates=$STEPS gpu=$GPU"
cd "$V"
exec bash "$V/run_geometry_aware_tum10.sh"
