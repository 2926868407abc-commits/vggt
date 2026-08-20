#!/usr/bin/env bash
# The calibrated loss ablation, re-run on the hand-placed monitor / poster surfaces.
#
# Why re-run: sections 6 onward used the automatic fused_depth_surface, which put a
# 0.36x0.24 m patch on whatever plane scored best -- often a desk or partition, not
# the monitor the earlier sitting_static work used. These runs put the patch on the
# screen, filling it, at a physically sensible 0.63x0.41 m.
#
# Placement was verified over three smoke passes (see placement notes): the quads
# come from fitting the dark screen component's four extreme corners, and the plane
# guards are tightened (surface_support_rel_tolerance 0.01 -> 0.06 m tolerance floor,
# fused_max_plane_residual 0.02) so a tilted fit raises instead of silently
# rendering a 1.36x2.69 m plane, which is what the loose defaults did once.
#
# Regulariser weights are per-(surface, loss) from calibrate_attack_reg_balance.py
# re-measured on THIS placement -- the published weights were measured on the
# automatic patch and do not transfer. Unlike the earlier calibration, gradients
# were measured with --physical_eot 0, matching what these runs actually train with.
set -Eeuo pipefail
VGGT_ROOT=/mnt/data/wangqq/vggt
HAZARD="$VGGT_ROOT/assets/hazard_textures/mde_attack_warnning.png"
LOG_ROOT=/tmp/g2vlm_mon/logs
MAX_PARALLEL="${MAX_PARALLEL:-4}"
GPUS="${GPUS:-0 1 2 3}"
mkdir -p "$LOG_ROOT"

BASE=(
  FREEZE_MODEL_PARAMETERS=1
  FORCE_PREPARE_TUM10=0 FORCE_CLEAN=0 FORCE_TRAIN=1 FORCE_APPLY=1
  RUN_EVAL=0 RUN_GAUGE_DIAG=1 RUN_CONSISTENCY_CHECK=1
  POSE_REVERSE_REFERENCE=gt POSE_BAD_REFERENCE=gt
  POSE_ROTATION_WEIGHT=5.0 POSE_TRANSLATION_WEIGHT=1.0
  PLANE_MODE=depth_manual_quad_surface
  MANUAL_QUAD_COORDINATES=normalized
  MANUAL_QUAD_DEPTH_SAMPLE_STRIDE=1
  MANUAL_QUAD_FIT_SHRINK=0.75
  MANUAL_QUAD_PLANE_INLIER_TOLERANCE=0.06
  MANUAL_QUAD_MIN_INLIER_RATIO=0.60
  SURFACE_SUPPORT_REL_TOLERANCE=0.01
  SURFACE_SUPPORT_ABS_TOLERANCE=0.10
  FUSED_MAX_PLANE_RESIDUAL=0.02
  SURFACE_SCORE_MODE=coverage
  SURFACE_COVERAGE_MIN=0.002 SURFACE_COVERAGE_MAX=0.08
  SURFACE_MIN_VISIBLE_FRAMES=4 SURFACE_MIN_VISIBILITY_RATIO=0.30
  OPTIMIZE_GEOMETRY=1 SURFACE_STRENGTH_SEARCH=0 NATURAL_AUTO_RELAX=0
  USE_DEPTH_VISIBILITY=1 VISIBILITY_DEPTH_MARGIN=0.08 SURFACE_SUPPORT_CHECK=1
  TEXTURE_INIT=image TEXTURE_INIT_IMAGE="$HAZARD"
  NATURAL_REFERENCE_IMAGE="$HAZARD"
  ITERATIONS=1000 INNER_LOOP=1 SCENES_PER_ITERATION=1
  PATCH_LR=0.002 FEATURE_LAYER=aggregator_final
  PHYSICAL_EOT=0
  PRINTABLE_COLOR_LEVELS=2 LOW_FREQUENCY_WEIGHT=0.0
  SEED=0
)

XYZ=rgbd_dataset_freiburg3_sitting_xyz
HALF=rgbd_dataset_freiburg3_sitting_halfsphere
Q_MON_XYZ=0.5330,0.4520,0.7380,0.4670,0.7320,0.6340,0.5450,0.6220
Q_MON_HALF=0.3603,0.2939,0.5682,0.3063,0.5644,0.4943,0.3603,0.4795
Q_POST_HALF=0.2220,0.1680,0.3450,0.1720,0.3430,0.4870,0.2240,0.4830

# surface|scene|quad|support|loss|tag|TV|PRINT|NATREF
JOBS=(
  "monxyz|$XYZ|$Q_MON_XYZ|0.50|pose_gt_untargeted|old|0.001|0.001|0.05"
  "monxyz|$XYZ|$Q_MON_XYZ|0.50|pose_scale_invariant_mse|si|4.14859e-05|4.14859e-05|0.00207429"
  "monxyz|$XYZ|$Q_MON_XYZ|0.50|pose_pairwise_relative_mse|pw|2.9321e-05|2.9321e-05|0.00146605"
  "monxyz|$XYZ|$Q_MON_XYZ|0.50|pose_aligned_residual_mse|al|0.000959746|0.000959746|0.0479873"
  "monhalf|$HALF|$Q_MON_HALF|0.50|pose_gt_untargeted|old|0.000851005|0.000851005|0.0425503"
  "monhalf|$HALF|$Q_MON_HALF|0.50|pose_scale_invariant_mse|si|3.04012e-05|3.04012e-05|0.00152006"
  "monhalf|$HALF|$Q_MON_HALF|0.50|pose_pairwise_relative_mse|pw|4.23716e-05|4.23716e-05|0.00211858"
  "monhalf|$HALF|$Q_MON_HALF|0.50|pose_aligned_residual_mse|al|0.001|0.001|0.05"
  "posthalf|$HALF|$Q_POST_HALF|0.30|pose_gt_untargeted|old|0.001|0.001|0.05"
  "posthalf|$HALF|$Q_POST_HALF|0.30|pose_scale_invariant_mse|si|4.57198e-05|4.57198e-05|0.00228599"
  "posthalf|$HALF|$Q_POST_HALF|0.30|pose_pairwise_relative_mse|pw|7.96462e-05|7.96462e-05|0.00398231"
  "posthalf|$HALF|$Q_POST_HALF|0.30|pose_aligned_residual_mse|al|0.000878964|0.000878964|0.0439482"
)

launch() {
  local spec="$1" gpu="$2"
  IFS='|' read -r surf scene quad support loss tag tv pr nr <<< "$spec"
  local name="mon_${surf}_${tag}_1000"
  echo "[launch] gpu=$gpu $surf $loss  tv=$tv"
  nohup env "${BASE[@]}" \
    SCENE_PATTERN="$scene" ATTACK_LOSS="$loss" \
    MANUAL_QUAD_XY="$quad" SURFACE_MIN_SUPPORT_RATIO="$support" \
    TV_WEIGHT="$tv" PRINTABILITY_WEIGHT="$pr" NATURAL_REFERENCE_WEIGHT="$nr" \
    TUM_GEOM_RUN_NAME="$name" TUM_GEOM_MODEL="vggt_$name" \
    TUM10_FRAME_SCENES="/tmp/g2vlm_mon/$name/frame_scenes" \
    TUM10_FRAME_MANIFEST="/tmp/g2vlm_mon/$name/frame_scenes/tum10_frame_manifest.json" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    bash "$VGGT_ROOT/run_geometry_aware_tum10.sh" \
    > "$LOG_ROOT/$name.log" 2>&1 &
}

read -ra GPU_ARR <<< "$GPUS"
i=0
for spec in "${JOBS[@]}"; do
  launch "$spec" "${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  i=$((i + 1))
  if (( i % MAX_PARALLEL == 0 )); then
    echo "[wait] batch of $MAX_PARALLEL"
    wait
  else
    sleep 90
  fi
done
wait
echo MONITOR_ABLATION_DONE
grep -l "Traceback" "$LOG_ROOT"/*.log 2>/dev/null || echo "no tracebacks"
