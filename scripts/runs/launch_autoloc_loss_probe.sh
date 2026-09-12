#!/usr/bin/env bash
# Controlled probe: same validated monitor location and initialization, two losses.
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"
RECONS_PY="${RECONS_PY:-/mnt/data/wangqq/conda_envs/recons_eval/bin/python3}"
SCENE="${SCENE:-rgbd_dataset_freiburg3_sitting_halfsphere}"
AREA="${AREA:-0.2}"
PLANE="${PLANE:-2}"
UPDATES="${UPDATES:-300}"
SPEC="${SPEC:-$VGGT_ROOT/outputs/candidate_planes/${SCENE}_validated_a${AREA}.json}"
POSE_RUN="${POSE_RUN:-autoloc_v2_p2_best_pose_s0_${UPDATES}}"
JOINT_RUN="${JOINT_RUN:-autoloc_v2_p2_best_joint_s0_${UPDATES}}"
TRACK_DIR="${TRACK_DIR:-$VGGT_ROOT/outputs/autoloc_v2_tracks}"
OUT_CSV="${OUT_CSV:-$VGGT_ROOT/outputs/autoloc_v2_loss_probe_${UPDATES}.csv}"

cd "$VGGT_ROOT"
mkdir -p logs "$TRACK_DIR"

QUAD="$($VGGT_PY -c 'import json,sys; p=json.load(open(sys.argv[1])); c=max((x for x in p["candidates"] if int(x["plane"])==int(sys.argv[2])), key=lambda x:x["validation"]["mask_coverage_mean"]); print(",".join(map(str,c["quad"])))' "$SPEC" "$PLANE")"
echo "spec=$SPEC"
echo "plane=$PLANE updates=$UPDATES"
echo "quad=$QUAD"

common=(
  SCENE_PATTERN="$SCENE"
  ITERATIONS="$UPDATES"
  INNER_LOOP=1
  SCENES_PER_ITERATION=1
  PATCH_LR=0.002
  TEXTURE_SIZE=64
  TEXTURE_INIT=image
  TEXTURE_INIT_IMAGE="$VGGT_ROOT/assets/hazard_textures/mde_attack_warnning.png"
  NATURAL_REFERENCE_IMAGE="$VGGT_ROOT/assets/hazard_textures/mde_attack_warnning.png"
  NATURAL_REFERENCE_WEIGHT=0.05
  TV_WEIGHT=0.001
  PRINTABILITY_WEIGHT=0.001
  PRINTABLE_COLOR_LEVELS=2
  LOW_FREQUENCY_WEIGHT=0.0
  PLANE_MODE=explicit_world_quad
  EXPLICIT_QUAD_WORLD="$QUAD"
  OPTIMIZE_GEOMETRY=0
  USE_DEPTH_VISIBILITY=1
  VISIBILITY_DEPTH_MARGIN=0.08
  SURFACE_SUPPORT_CHECK=1
  SURFACE_SUPPORT_ABS_TOLERANCE=0.06
  SURFACE_SUPPORT_REL_TOLERANCE=0.02
  SURFACE_MIN_SUPPORT_RATIO=0.60
  PHYSICAL_EOT=0
  POSE_ROTATION_WEIGHT=5.0
  POSE_TRANSLATION_WEIGHT=1.0
  SEED=0
  FORCE_TRAIN=1
  FORCE_APPLY=1
  RUN_EVAL=0
  RUN_GAUGE_DIAG=0
  RUN_CONSISTENCY_CHECK=0
)

run_pose() {
  env CUDA_VISIBLE_DEVICES=0 "${common[@]}" \
    TUM_GEOM_RUN_NAME="$POSE_RUN" \
    ATTACK_LOSS=pose_aligned_residual_mse \
    bash run_geometry_aware_tum10.sh >"logs/${POSE_RUN}.log" 2>&1
}

run_joint() {
  env CUDA_VISIBLE_DEVICES=1 "${common[@]}" \
    TUM_GEOM_RUN_NAME="$JOINT_RUN" \
    ATTACK_LOSS=mirage_projected_split_pose \
    MIRAGE_PROJECTED_DIRECTION_SCALE=0.004719185642898083 \
    bash run_geometry_aware_tum10.sh >"logs/${JOINT_RUN}.log" 2>&1
}

run_pose & pose_pid=$!
run_joint & joint_pid=$!
wait "$pose_pid"
wait "$joint_pid"

CUDA_VISIBLE_DEVICES=0 "$VGGT_PY" -B scripts/dump_tracks.py \
  --runs "$POSE_RUN,$JOINT_RUN" --scene "$SCENE" \
  --out_dir "$TRACK_DIR" \
  --gt_dir "$VGGT_ROOT/outputs/tum_gt_point_track" \
  --clean tum10_clean_uniform_l3 \
  --explicit_quad_world "$QUAD"

"$RECONS_PY" -B scripts/eval_four_tasks.py \
  --runs "$POSE_RUN,$JOINT_RUN" --scene "$SCENE" \
  --track_dir "$TRACK_DIR" \
  --gt_dir "$VGGT_ROOT/outputs/tum_gt_point_track" \
  --clean tum10_clean_uniform_l3 \
  --out_csv "$OUT_CSV"

for run in "$POSE_RUN" "$JOINT_RUN"; do
  "$VGGT_PY" -B scripts/visualize_tum10_geometry_patch.py \
    --geometry_output_root "$VGGT_ROOT/outputs_attack_geometry_aware_tum10/$run" \
    --out_dir "$VGGT_ROOT/outputs/report_position_visuals/$run" \
    --frames 0,5,9 --contact_columns 3 --thumb_width 640 \
    --contact_sheet_only --no_label
done

echo "CSV=$OUT_CSV"
