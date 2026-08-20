#!/usr/bin/env bash
# Repeats on the monitor placement, sitting_halfsphere, to test whether the RPE
# rotation ranking is stable. Only one run per cell exists so far, and the measured
# noise floor on a same-config repeat was ~7%, which is nowhere near the 8x-vs-180x
# spread the ranking rests on -- but that floor was measured over 20 steps on the
# ATE fraction, not over 1000 steps on RPE rotation, so it does not transfer.
#
# Two extra seeds per loss, giving three runs per cell with the existing seed 0.
set -Eeuo pipefail
VGGT_ROOT=/mnt/data/wangqq/vggt
HAZARD="$VGGT_ROOT/assets/hazard_textures/mde_attack_warnning.png"
LOG_ROOT=/tmp/g2vlm_rep/logs
mkdir -p "$LOG_ROOT"
HALF=rgbd_dataset_freiburg3_sitting_halfsphere
Q=0.3603,0.2939,0.5682,0.3063,0.5644,0.4943,0.3603,0.4795

BASE=(
  FREEZE_MODEL_PARAMETERS=1
  FORCE_PREPARE_TUM10=0 FORCE_CLEAN=0 FORCE_TRAIN=1 FORCE_APPLY=1
  RUN_EVAL=0 RUN_GAUGE_DIAG=1 RUN_CONSISTENCY_CHECK=1
  POSE_REVERSE_REFERENCE=gt POSE_BAD_REFERENCE=gt
  POSE_ROTATION_WEIGHT=5.0 POSE_TRANSLATION_WEIGHT=1.0
  PLANE_MODE=depth_manual_quad_surface
  MANUAL_QUAD_COORDINATES=normalized MANUAL_QUAD_XY="$Q"
  MANUAL_QUAD_DEPTH_SAMPLE_STRIDE=1 MANUAL_QUAD_FIT_SHRINK=0.75
  MANUAL_QUAD_PLANE_INLIER_TOLERANCE=0.06 MANUAL_QUAD_MIN_INLIER_RATIO=0.60
  SURFACE_SUPPORT_REL_TOLERANCE=0.01 SURFACE_SUPPORT_ABS_TOLERANCE=0.10
  FUSED_MAX_PLANE_RESIDUAL=0.02
  SURFACE_SCORE_MODE=coverage
  SURFACE_COVERAGE_MIN=0.002 SURFACE_COVERAGE_MAX=0.08
  SURFACE_MIN_VISIBLE_FRAMES=4 SURFACE_MIN_VISIBILITY_RATIO=0.30
  SURFACE_MIN_SUPPORT_RATIO=0.50
  OPTIMIZE_GEOMETRY=1 SURFACE_STRENGTH_SEARCH=0 NATURAL_AUTO_RELAX=0
  USE_DEPTH_VISIBILITY=1 VISIBILITY_DEPTH_MARGIN=0.08 SURFACE_SUPPORT_CHECK=1
  TEXTURE_INIT=image TEXTURE_INIT_IMAGE="$HAZARD"
  NATURAL_REFERENCE_IMAGE="$HAZARD"
  ITERATIONS=1000 INNER_LOOP=1 SCENES_PER_ITERATION=1
  PATCH_LR=0.002 FEATURE_LAYER=aggregator_final
  PHYSICAL_EOT=0 PRINTABLE_COLOR_LEVELS=2 LOW_FREQUENCY_WEIGHT=0.0
  SCENE_PATTERN="$HALF"
)

# loss|tag|TV|PRINT|NATREF   (same calibrated weights as the seed-0 runs)
CELLS=(
  "pose_gt_untargeted|old|0.000851005|0.000851005|0.0425503"
  "pose_scale_invariant_mse|si|3.04012e-05|3.04012e-05|0.00152006"
  "pose_pairwise_relative_mse|pw|4.23716e-05|4.23716e-05|0.00211858"
  "pose_aligned_residual_mse|al|0.001|0.001|0.05"
)

i=0
for seed in 1 2; do
  gpu=0
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r loss tag tv pr nr <<< "$cell"
    name="rep_monhalf_${tag}_s${seed}"
    echo "[launch] gpu=$gpu $tag seed=$seed"
    nohup env "${BASE[@]}" \
      ATTACK_LOSS="$loss" SEED="$seed" \
      TV_WEIGHT="$tv" PRINTABILITY_WEIGHT="$pr" NATURAL_REFERENCE_WEIGHT="$nr" \
      TUM_GEOM_RUN_NAME="$name" TUM_GEOM_MODEL="vggt_$name" \
      TUM10_FRAME_SCENES="/tmp/g2vlm_rep/$name/frame_scenes" \
      TUM10_FRAME_MANIFEST="/tmp/g2vlm_rep/$name/frame_scenes/tum10_frame_manifest.json" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      bash "$VGGT_ROOT/run_geometry_aware_tum10.sh" > "$LOG_ROOT/$name.log" 2>&1 &
    gpu=$((gpu + 1))
    sleep 60
  done
  echo "[wait] seed $seed"
  wait
done
echo REPEATS_DONE
grep -l Traceback "$LOG_ROOT"/*.log 2>/dev/null || echo "no tracebacks"
