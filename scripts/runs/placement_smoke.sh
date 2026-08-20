#!/usr/bin/env bash
# Third placement pass, xyz only. Round 2 produced a 1.36x2.69 m plane at residual
# 0.110 -- a failed fit that passed the guard silently, because
#   tolerance = max(0.06, median_z * surface_support_rel_tolerance) = 2.7 * 0.05 = 0.134 m
# admitted the noisy depth off the dark screen, and fused_max_plane_residual=0.12
# was above the residual it produced.
#
# Two changes, so a bad fit is loud instead of silent:
#   SURFACE_SUPPORT_REL_TOLERANCE=0.01  -> tolerance falls back to the 0.06 m floor
#   FUSED_MAX_PLANE_RESIDUAL=0.02       -> a tilted plane raises instead of rendering
#
# Two candidate quads: A is round 1's, already verified clean (0.63x0.41, residual
# 0.0074) but overhanging the top bezel; B is A nudged down so it sits inside the screen.
set -Eeuo pipefail
VGGT_ROOT=/mnt/data/wangqq/vggt
HAZARD=$VGGT_ROOT/assets/hazard_textures/mde_attack_warnning.png
LOG=/tmp/g2vlm_place4/logs
mkdir -p "$LOG"

COMMON=(
  FORCE_PREPARE_TUM10=0 FORCE_CLEAN=0 FORCE_TRAIN=1 FORCE_APPLY=1
  RUN_EVAL=0 RUN_GAUGE_DIAG=0 RUN_CONSISTENCY_CHECK=0
  ATTACK_LOSS=pose_aligned_residual_mse
  POSE_ROTATION_WEIGHT=5.0 POSE_TRANSLATION_WEIGHT=1.0
  PLANE_MODE=depth_manual_quad_surface
  MANUAL_QUAD_COORDINATES=normalized
  MANUAL_QUAD_DEPTH_SAMPLE_STRIDE=1
  MANUAL_QUAD_FIT_SHRINK=0.75
  MANUAL_QUAD_PLANE_INLIER_TOLERANCE=0.06
  MANUAL_QUAD_MIN_INLIER_RATIO=0.60
  SURFACE_SUPPORT_REL_TOLERANCE=0.01
  FUSED_MAX_PLANE_RESIDUAL=0.02
  OPTIMIZE_GEOMETRY=0 SURFACE_STRENGTH_SEARCH=0 NATURAL_AUTO_RELAX=0
  USE_DEPTH_VISIBILITY=1 VISIBILITY_DEPTH_MARGIN=0.08
  SURFACE_SUPPORT_CHECK=0
  SURFACE_SCORE_MODE=coverage
  SURFACE_COVERAGE_MIN=0.002 SURFACE_COVERAGE_MAX=0.08
  SURFACE_MIN_VISIBLE_FRAMES=4 SURFACE_MIN_VISIBILITY_RATIO=0.30
  TEXTURE_INIT=image TEXTURE_INIT_IMAGE="$HAZARD"
  NATURAL_REFERENCE_IMAGE="$HAZARD" NATURAL_REFERENCE_WEIGHT=0.05
  ITERATIONS=20 INNER_LOOP=1 SCENES_PER_ITERATION=1 WARMUP_ITERATIONS=5
  PATCH_LR=0.002 FEATURE_LAYER=aggregator_final PHYSICAL_EOT=0
  TV_WEIGHT=0.001 PRINTABILITY_WEIGHT=0.001 PRINTABLE_COLOR_LEVELS=2
  LOW_FREQUENCY_WEIGHT=0.0 SEED=0 FREEZE_MODEL_PARAMETERS=1
)

SCENE=rgbd_dataset_freiburg3_sitting_xyz
JOBS=(
  "xyzA 0.5330,0.4520,0.7380,0.4670,0.7320,0.6340,0.5450,0.6220"
  "xyzB 0.5350,0.4640,0.7360,0.4790,0.7300,0.6460,0.5430,0.6340"
)
# also re-verify the halfsphere pair under the tightened guards
JOBS+=(
  "halfM 0.3603,0.2939,0.5682,0.3063,0.5644,0.4943,0.3603,0.4795"
  "halfP 0.2220,0.1680,0.3450,0.1720,0.3430,0.4870,0.2240,0.4830"
)

gpu=0
for job in "${JOBS[@]}"; do
  read -r name quad <<< "$job"
  sc=$SCENE
  case "$name" in half*) sc=rgbd_dataset_freiburg3_sitting_halfsphere ;; esac
  echo "[launch] gpu=$gpu $name  $sc"
  nohup env "${COMMON[@]}" \
    SCENE_PATTERN="$sc" MANUAL_QUAD_XY="$quad" \
    TUM_GEOM_RUN_NAME="s3_$name" TUM_GEOM_MODEL="vggt_s3_$name" \
    TUM10_FRAME_SCENES="/tmp/g2vlm_place4/$name/frame_scenes" \
    TUM10_FRAME_MANIFEST="/tmp/g2vlm_place4/$name/frame_scenes/tum10_frame_manifest.json" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    bash "$VGGT_ROOT/run_geometry_aware_tum10.sh" > "$LOG/$name.log" 2>&1 &
  gpu=$(( (gpu + 1) % 4 ))
  sleep 45
done
wait
echo PLACE_SMOKE3_DONE
for f in "$LOG"/*.log; do
  echo "--- $(basename "$f")"
  grep -E "not planar enough|RuntimeError|Traceback" "$f" | head -3 || echo "    ok"
done
