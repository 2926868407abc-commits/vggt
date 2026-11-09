#!/usr/bin/env bash
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
GPU="${1:?usage: $0 GPU [MAX_STEPS] [formal|smoke]}"
MAX_STEPS="${2:-8000}"
MODE="${3:-formal}"
SCENE="rgbd_dataset_freiburg3_sitting_static"
SHORT="sitting_static"
CONTEXT=20
AUDIT="$V/data/adaptive_audits/firsthalf_${SHORT}_k${CONTEXT}/audit_plan.json"
INIT_RUN="jointctx_v5_k20_firsthalf_sitting_static_posetrack_adaptive_s0_umax8000"
INIT_PATCH="$V/outputs_attack_geometry_aware_tum10/$INIT_RUN/geometry_patch/optimized_patch.png"
GT_POOL="$V/outputs/tum_gt_pool_universal_all_firsthalf"
POOLS="$V/data/tum_per_scene_splits/firsthalf_train_pools.json"
MANIFEST="$V/data/tum_per_scene_splits/firsthalf_train_preview_manifest.json"
if [[ "$MODE" == "smoke" ]]; then
  MAX_STEPS=1
  RUN="jointctx_v5_k20_firsthalf_sitting_static_vggt_training_loss_smoke_s0_u1"
else
  RUN="jointctx_v5_k20_firsthalf_sitting_static_vggt_training_loss_adaptive_s0_umax${MAX_STEPS}"
fi
OUT="$V/outputs_attack_geometry_aware_tum10/$RUN"
LOG="$V/logs/joint_context_10_20_30/k20_per_scene/${SHORT}.vggt_training_loss.log"

for required in "$AUDIT" "$INIT_PATCH" "$POOLS" "$MANIFEST" \
  "$GT_POOL/$SCENE/pool_meta.json"; do
  test -e "$required"
done
mkdir -p "$(dirname "$LOG")"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export VGGT_PY=/mnt/data/wangqq/conda_envs/vggt/bin/python3
export RECONS_PY="$VGGT_PY"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPYCACHEPREFIX=/mnt/data/wangqq/tmp_pyc
export TUM_ROOT=/mnt/data/wangqq/recons_eval/data/tum_all_sync
export TUM10_FRAME_MANIFEST="$MANIFEST"
export GT_NAME=groundtruth.txt
export SCENE_PATTERN="$SCENE"
export TUM_GEOM_RUN_NAME="$RUN"

export ITERATIONS="$MAX_STEPS" INNER_LOOP=1 SCENES_PER_ITERATION=1
export PATCH_LR=0.002 SCHEDULER=adaptive_plateau WARMUP_ITERATIONS=20
export ADAPTIVE_LR_FACTOR=0.5 ADAPTIVE_LR_PATIENCE=3
export ADAPTIVE_STOP_PATIENCE=5 ADAPTIVE_MIN_LR=0.00001
export ADAPTIVE_IMPROVEMENT_THRESHOLD=0.005
export AUDIT_PLAN="$AUDIT" AUDIT_EVERY=100
export CHECKPOINT_EVERY=100 AUTO_RESUME=1 FORCE_TRAIN=1
if [[ "$MODE" == "smoke" ]]; then
  export SCHEDULER=none AUDIT_PLAN='' AUDIT_EVERY=0
  export CHECKPOINT_EVERY=1 AUTO_RESUME=0
fi

export TEXTURE_SIZE=64 TEXTURE_INIT=image TEXTURE_INIT_IMAGE="$INIT_PATCH"
export ATTACK_LOSS=vggt_training_loss
export VGGT_CAMERA_WEIGHT=1.0 VGGT_DEPTH_WEIGHT=1.0
export VGGT_POINT_WEIGHT=1.0 VGGT_TRACK_WEIGHT=0.05
export VGGT_TASK_GRAD_EVERY=10
export MIRAGE_PROJECTED_GT_DIR="$GT_POOL"
export MIRAGE_PROJECTED_MIN_QUERIES=20 MIRAGE_PROJECTED_MAX_QUERIES=0
export JOINT_TRACK_ITERS=2
export DYNAMIC_FRAME_POOL='' DYNAMIC_FRAME_POOL_JSON="$POOLS"
export DYNAMIC_SETS_PER_STEP=1 DYNAMIC_FRAMES_PER_SET="$CONTEXT"
export DYNAMIC_QUERY_SAMPLING=uniform_query
export SUBSET_PLAN='' SUBSET_ACCUM=1 SUBSET_AGG=mean
export SOFTMIN_TRANSFORMS=identity

export PLANE_MODE=explicit_world_quad
export EXPLICIT_QUAD_WORLD="$(tr -d '\n\r' < "$V/assets/visibility_fix/sitting_xyz_monitor_on_stable_plane_v5.txt")"
export OPTIMIZE_GEOMETRY=0 USE_DEPTH_VISIBILITY=1 VISIBILITY_DEPTH_MARGIN=0.08
export TV_WEIGHT=0.001 PRINTABILITY_WEIGHT=0.001 PRINTABLE_COLOR_LEVELS=2
export NATURAL_REFERENCE_IMAGE="$V/assets/hazard_textures/mde_attack_warnning.png"
export NATURAL_REFERENCE_WEIGHT=0.05 PHYSICAL_EOT=0 SEED=0
export ACTIVATION_CHECKPOINT=1 ACTIVATION_CHECKPOINT_MODE=all
export SKIP_TUM_PREPARE=1 SKIP_CLEAN_VGGT=1 RUN_EVAL=0
export RUN_GAUGE_DIAG=0 RUN_CONSISTENCY_CHECK=0

is_converged() {
  local meta="$1"
  [[ -f "$meta" ]] || return 1
  "$VGGT_PY" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["adaptive_scheduler"]["state"]["stopped"]' \
    "$meta" >/dev/null 2>&1
}

{
  echo "[vggt-loss-start] $(date -Is) gpu=$GPU run=$RUN"
  if [[ "$MODE" == "smoke" ]]; then
    cd "$V"
    bash "$V/scripts/runs/run_vggt_loss_debug.sh"
    test -f "$OUT/geometry_patch/optimized_patch.png"
    test -s "$OUT/geometry_patch/training_history.jsonl"
    echo "[vggt-loss-smoke-done] $(date -Is) run=$RUN"
    exit 0
  fi
  if ! is_converged "$OUT/geometry_patch/geometry_patch_meta.json"; then
    cd "$V"
    bash "$V/scripts/runs/run_vggt_loss_debug.sh"
  fi
  is_converged "$OUT/geometry_patch/geometry_patch_meta.json"
  test -f "$OUT/geometry_patch/optimized_patch.png"
  echo "[vggt-loss-train-done] $(date -Is) run=$RUN"

  "$V/scripts/runs/eval_per_scene_formal_w30_s20.sh" "$GPU" "$SCENE" "$RUN"
  test -f "$OUT/heldout_w30_s20_v5_controlled/four_tasks/four_task_by_scene.csv"
  echo "[vggt-loss-eval-done] $(date -Is) run=$RUN"
} >>"$LOG" 2>&1
