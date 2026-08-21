#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data/wangqq/vggt"
PY="/mnt/data/wangqq/conda_envs/vggt/bin/python3"
TEXTURE="$ROOT/assets/hazard_textures/mde_attack_warnning.png"

launch() {
  local gpu="$1" run="$2" scene="$3" axis="$4"
  local depth_w="$5" point_w="$6" depth_conf_w="$7" point_conf_w="$8" track_w="$9"
  local log="/tmp/${run}.log"
  echo "launch gpu=$gpu run=$run scene=$scene -> $log"
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    VGGT_ROOT="$ROOT" VGGT_PY="$PY" \
    TUM10_FRAME_SCENES="/tmp/g2vlm_q4cal/$run/frame_scenes" \
    TUM_GEOM_RUN_NAME="$run" TUM_GEOM_MODEL="vggt_${run}" \
    SCENE_PATTERN="$scene" \
    ITERATIONS=250 INNER_LOOP=1 SCENES_PER_ITERATION=1 PATCH_LR=0.002 \
    TEXTURE_SIZE=128 TEXTURE_INIT=image TEXTURE_INIT_IMAGE="$TEXTURE" \
    ATTACK_LOSS=geometry_joint_gauge_targeted \
    PIECEWISE_GAUGE_FAMILY=orthogonal_mode PIECEWISE_GAUGE_MAGNITUDE=3.0 \
    ORTHOGONAL_MODE_ORDER=2 ORTHOGONAL_MODE_AXIS="$axis" \
    POSE_ROTATION_WEIGHT=5.0 POSE_TRANSLATION_WEIGHT=1.0 \
    JOINT_POSE_WEIGHT=1.0 JOINT_DEPTH_WEIGHT="$depth_w" \
    JOINT_POINT_WEIGHT="$point_w" JOINT_DEPTH_CONF_WEIGHT="$depth_conf_w" \
    JOINT_POINT_CONF_WEIGHT="$point_conf_w" JOINT_TRACK_WEIGHT="$track_w" \
    JOINT_TRACK_GRID_ROWS=4 JOINT_TRACK_GRID_COLS=6 JOINT_TRACK_ITERS=2 \
    PLANE_MODE=fused_depth_surface PLANE_WIDTH=0.30 PLANE_HEIGHT=0.20 \
    SURFACE_SCORE_MODE=coverage SURFACE_COVERAGE_MIN=0.005 SURFACE_COVERAGE_MAX=0.06 \
    SURFACE_MIN_VISIBLE_FRAMES=8 SURFACE_MIN_VISIBILITY_RATIO=0.80 \
    SURFACE_SUPPORT_CHECK=1 SURFACE_MIN_SUPPORT_RATIO=0.50 \
    SURFACE_SUPPORT_ABS_TOLERANCE=0.10 SURFACE_SUPPORT_REL_TOLERANCE=0.06 \
    FUSED_MAX_PLANE_RESIDUAL=0.12 VISIBILITY_DEPTH_MARGIN=0.08 \
    TV_WEIGHT=0.001 PRINTABILITY_WEIGHT=0.001 PRINTABLE_COLOR_LEVELS=2 \
    NATURAL_REFERENCE_IMAGE="$TEXTURE" NATURAL_REFERENCE_WEIGHT=0.05 \
    PHYSICAL_EOT=0 FORCE_TRAIN=1 FORCE_APPLY=1 FORCE_CLEAN=0 \
    FREEZE_MODEL_PARAMETERS=1 RUN_CONSISTENCY_CHECK=1 RUN_GAUGE_DIAG=1 RUN_EVAL=0 \
    bash "$ROOT/run_geometry_aware_tum10.sh" >"$log" 2>&1 &
  echo "$!" >"/tmp/${run}.pid"
}

# Equal-gradient constraints and a 3x-stronger consistency setting per sequence.
launch 0 q4conf_xyz_k1_250 rgbd_dataset_freiburg3_sitting_xyz 0 \
  0.03587 0.1679 0.08939 0.1208 4.841
launch 1 q4conf_xyz_k3_250 rgbd_dataset_freiburg3_sitting_xyz 0 \
  0.10761 0.5037 0.26817 0.3624 14.523
launch 2 q4conf_half_k1_250 rgbd_dataset_freiburg3_sitting_halfsphere 1 \
  0.1434 0.3139 0.02537 0.1212 0.2402
launch 3 q4conf_half_k3_250 rgbd_dataset_freiburg3_sitting_halfsphere 1 \
  0.4302 0.9417 0.07611 0.3636 0.7206
