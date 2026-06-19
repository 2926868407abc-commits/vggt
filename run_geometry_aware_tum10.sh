#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
RECONS_ROOT="${RECONS_ROOT:-/mnt/data/wangqq/recons_eval}"
VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"
RECONS_PY="${RECONS_PY:-/mnt/data/wangqq/conda_envs/recons_eval/bin/python3}"
CKPT="${CKPT:-$VGGT_ROOT/checkpoints/VGGT-1B}"

TUM_ROOT="${TUM_ROOT:-$RECONS_ROOT/data/tum}"
TUM10_FRAME_SCENES="${TUM10_FRAME_SCENES:-$VGGT_ROOT/data/tum_dynamics_10frame_individual_scenes}"
TUM10_FRAME_MANIFEST="${TUM10_FRAME_MANIFEST:-$TUM10_FRAME_SCENES/tum10_frame_manifest.json}"
TUM_FRAME_COUNT="${TUM_FRAME_COUNT:-10}"

OUT_BASE="${OUT_BASE:-$VGGT_ROOT/outputs_attack_geometry_aware_tum10}"
TUM_CLEAN_OUT="$OUT_BASE/tum10_clean_uniform_l3"
TUM_GEOM_RUN_NAME="${TUM_GEOM_RUN_NAME:-tum10_vggt_pointmap_geometry_feature_l3}"
TUM_GEOM_OUT="$OUT_BASE/$TUM_GEOM_RUN_NAME"

TUM_CLEAN_MODEL="vggt_tum10_clean_uniform_l3_geomrun"
TUM_GEOM_MODEL="${TUM_GEOM_MODEL:-vggt_tum10_vggt_pointmap_geometry_feature_l3}"

ITERATIONS="${ITERATIONS:-200}"
INNER_LOOP="${INNER_LOOP:-10}"
SCENES_PER_ITERATION="${SCENES_PER_ITERATION:-1}"
PATCH_LR="${PATCH_LR:-0.001}"
TEXTURE_SIZE="${TEXTURE_SIZE:-128}"
FEATURE_LAYER="${FEATURE_LAYER:-aggregator_final}"
SEED="${SEED:-0}"

PLANE_WIDTH="${PLANE_WIDTH:-0.6}"
PLANE_HEIGHT="${PLANE_HEIGHT:-0.6}"
PLANE_DISTANCE="${PLANE_DISTANCE:-2.0}"
PLANE_CENTER_X="${PLANE_CENTER_X:-0.0}"
PLANE_CENTER_Y="${PLANE_CENTER_Y:-0.0}"
PLANE_MODE="${PLANE_MODE:-vggt_pointmap_surface}"
VGGT_POINT_CONF_PERCENTILE="${VGGT_POINT_CONF_PERCENTILE:-40}"
USE_DEPTH_VISIBILITY="${USE_DEPTH_VISIBILITY:-1}"
OPTIMIZE_GEOMETRY="${OPTIMIZE_GEOMETRY:-1}"
SURFACE_CANDIDATE_GRID="${SURFACE_CANDIDATE_GRID:-4}"
SURFACE_SEARCH_MARGIN="${SURFACE_SEARCH_MARGIN:-0.18}"
GEOMETRY_SIZE_SCALES="${GEOMETRY_SIZE_SCALES:-0.8,1.0,1.2}"
GEOMETRY_ROLL_DEGREES="${GEOMETRY_ROLL_DEGREES:--15,0,15}"
VISIBILITY_DEPTH_MARGIN="${VISIBILITY_DEPTH_MARGIN:-0.05}"
FUSED_POINT_STRIDE="${FUSED_POINT_STRIDE:-28}"
FUSED_MAX_POINTS="${FUSED_MAX_POINTS:-6000}"
FUSED_SURFACE_CANDIDATES="${FUSED_SURFACE_CANDIDATES:-64}"
FUSED_NORMAL_RADIUS="${FUSED_NORMAL_RADIUS:-0.25}"
FUSED_MIN_NEIGHBORS="${FUSED_MIN_NEIGHBORS:-24}"
FUSED_MAX_NEIGHBORS="${FUSED_MAX_NEIGHBORS:-256}"
FUSED_MAX_PLANE_RESIDUAL="${FUSED_MAX_PLANE_RESIDUAL:-0.08}"
SURFACE_SCORE_MODE="${SURFACE_SCORE_MODE:-coverage}"
SURFACE_COVERAGE_MIN="${SURFACE_COVERAGE_MIN:-0.005}"
SURFACE_COVERAGE_MAX="${SURFACE_COVERAGE_MAX:-0.06}"
SURFACE_MIN_VISIBLE_FRAMES="${SURFACE_MIN_VISIBLE_FRAMES:-3}"
SURFACE_MIN_VISIBILITY_RATIO="${SURFACE_MIN_VISIBILITY_RATIO:-0.5}"
SURFACE_ORIENTATION_FILTER="${SURFACE_ORIENTATION_FILTER:-none}"
SURFACE_MAX_TILT_DEGREES="${SURFACE_MAX_TILT_DEGREES:-35}"
SURFACE_MIN_CENTER_DEPTH="${SURFACE_MIN_CENTER_DEPTH:-0.0}"
SURFACE_MAX_CENTER_DEPTH="${SURFACE_MAX_CENTER_DEPTH:-0.0}"

PHYSICAL_EOT="${PHYSICAL_EOT:-1}"
PRINT_MIN="${PRINT_MIN:-0.0}"
PRINT_MAX="${PRINT_MAX:-1.0}"
EOT_BRIGHTNESS="${EOT_BRIGHTNESS:-0.15}"
EOT_CONTRAST="${EOT_CONTRAST:-0.15}"
EOT_GAMMA="${EOT_GAMMA:-0.10}"
EOT_NOISE_STD="${EOT_NOISE_STD:-0.01}"

FORCE_PREPARE_TUM10="${FORCE_PREPARE_TUM10:-0}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_APPLY="${FORCE_APPLY:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }
}

log "check paths"
require_file "$VGGT_PY"
require_file "$RECONS_PY"
require_file "$VGGT_ROOT/attack_vggt_geometry_tum10.py"
require_file "$VGGT_ROOT/scripts/prepare_tum10_frame_scenes_for_vla.py"
require_file "$VGGT_ROOT/scripts/run_vggt_clean_tum10_uniform.py"
require_file "$VGGT_ROOT/scripts/eval_vggt_tum_pose_for_recons_eval_tum10.py"
require_dir "$TUM_ROOT"
[[ -e "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }

log "settings"
echo "iterations=$ITERATIONS inner_loop=$INNER_LOOP scenes_per_iteration=$SCENES_PER_ITERATION"
echo "texture_size=$TEXTURE_SIZE patch_lr=$PATCH_LR feature_layer=$FEATURE_LAYER"
echo "plane_width=$PLANE_WIDTH plane_height=$PLANE_HEIGHT plane_distance=$PLANE_DISTANCE"
echo "plane_center=($PLANE_CENTER_X,$PLANE_CENTER_Y)"
echo "plane_mode=$PLANE_MODE clean_vggt_output_root=$TUM_CLEAN_OUT use_depth_visibility=$USE_DEPTH_VISIBILITY optimize_geometry=$OPTIMIZE_GEOMETRY"
echo "surface_candidate_grid=$SURFACE_CANDIDATE_GRID geometry_size_scales=$GEOMETRY_SIZE_SCALES geometry_roll_degrees=$GEOMETRY_ROLL_DEGREES"
echo "fused_point_stride=$FUSED_POINT_STRIDE fused_surface_candidates=$FUSED_SURFACE_CANDIDATES fused_normal_radius=$FUSED_NORMAL_RADIUS"
echo "surface_score_mode=$SURFACE_SCORE_MODE coverage=[$SURFACE_COVERAGE_MIN,$SURFACE_COVERAGE_MAX] min_visible_frames=$SURFACE_MIN_VISIBLE_FRAMES min_visibility_ratio=$SURFACE_MIN_VISIBILITY_RATIO"
echo "surface_orientation_filter=$SURFACE_ORIENTATION_FILTER max_tilt=$SURFACE_MAX_TILT_DEGREES center_depth=[$SURFACE_MIN_CENTER_DEPTH,$SURFACE_MAX_CENTER_DEPTH]"
echo "physical_eot=$PHYSICAL_EOT print=[$PRINT_MIN,$PRINT_MAX] brightness=$EOT_BRIGHTNESS contrast=$EOT_CONTRAST gamma=$EOT_GAMMA noise_std=$EOT_NOISE_STD"

log "prepare TUM images links"
for seq_dir in "$TUM_ROOT"/rgbd_dataset_freiburg3_*; do
  if [[ -d "$seq_dir/rgb_90" ]]; then
    ln -sfn rgb_90 "$seq_dir/images"
  fi
done

log "prepare uniform TUM-10 manifest"
prepare_args=()
if [[ "$FORCE_PREPARE_TUM10" == "1" ]]; then
  prepare_args=(--overwrite)
fi
"$VGGT_PY" "$VGGT_ROOT/scripts/prepare_tum10_frame_scenes_for_vla.py" \
  --tum_root "$TUM_ROOT" \
  --out_root "$TUM10_FRAME_SCENES" \
  --frame_count "$TUM_FRAME_COUNT" \
  "${prepare_args[@]}"

log "run clean VGGT on TUM-10 uniform frames"
clean_args=()
if [[ "$FORCE_CLEAN" != "1" ]]; then
  clean_args=(--skip_existing)
fi
"$VGGT_PY" "$VGGT_ROOT/scripts/run_vggt_clean_tum10_uniform.py" \
  --tum_root "$TUM_ROOT" \
  --output_root "$TUM_CLEAN_OUT" \
  --frame_manifest "$TUM10_FRAME_MANIFEST" \
  --ckpt "$CKPT" \
  --seed "$SEED" \
  "${clean_args[@]}"

log "run GT-geometry-aware planar patch attack"
patch_args=()
if [[ "$FORCE_TRAIN" != "1" && -f "$TUM_GEOM_OUT/geometry_patch/geometry_patch_texture.npz" ]]; then
  patch_args=(--texture_path "$TUM_GEOM_OUT/geometry_patch/geometry_patch_texture.npz")
  log "reuse geometry patch -> ${patch_args[1]}"
fi
apply_args=()
if [[ "$FORCE_APPLY" != "1" && "$FORCE_TRAIN" != "1" ]]; then
  apply_args=(--skip_existing_outputs)
fi
geometry_args=(--plane_mode "$PLANE_MODE")
if [[ "$USE_DEPTH_VISIBILITY" == "1" ]]; then
  geometry_args+=(--use_depth_visibility)
fi
if [[ "$OPTIMIZE_GEOMETRY" == "1" ]]; then
  geometry_args+=(--optimize_geometry)
fi
if [[ "$PHYSICAL_EOT" == "1" ]]; then
  geometry_args+=(--physical_eot)
fi
"$VGGT_PY" "$VGGT_ROOT/attack_vggt_geometry_tum10.py" \
  --tum_root "$TUM_ROOT" \
  --output_dir "$TUM_GEOM_OUT" \
  --frame_manifest "$TUM10_FRAME_MANIFEST" \
  --ckpt "$CKPT" \
  --texture_size "$TEXTURE_SIZE" \
  --iterations "$ITERATIONS" \
  --inner_loop "$INNER_LOOP" \
  --scenes_per_iteration "$SCENES_PER_ITERATION" \
  --patch_lr "$PATCH_LR" \
  --feature_layer "$FEATURE_LAYER" \
  --plane_width "$PLANE_WIDTH" \
  --plane_height "$PLANE_HEIGHT" \
  --plane_distance "$PLANE_DISTANCE" \
  --plane_center_x "$PLANE_CENTER_X" \
  --plane_center_y "$PLANE_CENTER_Y" \
  --clean_vggt_output_root "$TUM_CLEAN_OUT" \
  --vggt_point_conf_percentile "$VGGT_POINT_CONF_PERCENTILE" \
  --surface_candidate_grid "$SURFACE_CANDIDATE_GRID" \
  --surface_search_margin "$SURFACE_SEARCH_MARGIN" \
  --geometry_size_scales="$GEOMETRY_SIZE_SCALES" \
  --geometry_roll_degrees="$GEOMETRY_ROLL_DEGREES" \
  --fused_point_stride "$FUSED_POINT_STRIDE" \
  --fused_max_points "$FUSED_MAX_POINTS" \
  --fused_surface_candidates "$FUSED_SURFACE_CANDIDATES" \
  --fused_normal_radius "$FUSED_NORMAL_RADIUS" \
  --fused_min_neighbors "$FUSED_MIN_NEIGHBORS" \
  --fused_max_neighbors "$FUSED_MAX_NEIGHBORS" \
  --fused_max_plane_residual "$FUSED_MAX_PLANE_RESIDUAL" \
  --surface_score_mode "$SURFACE_SCORE_MODE" \
  --surface_coverage_min "$SURFACE_COVERAGE_MIN" \
  --surface_coverage_max "$SURFACE_COVERAGE_MAX" \
  --surface_min_visible_frames "$SURFACE_MIN_VISIBLE_FRAMES" \
  --surface_min_visibility_ratio "$SURFACE_MIN_VISIBILITY_RATIO" \
  --surface_orientation_filter "$SURFACE_ORIENTATION_FILTER" \
  --surface_max_tilt_degrees "$SURFACE_MAX_TILT_DEGREES" \
  --surface_min_center_depth "$SURFACE_MIN_CENTER_DEPTH" \
  --surface_max_center_depth "$SURFACE_MAX_CENTER_DEPTH" \
  --visibility_depth_margin "$VISIBILITY_DEPTH_MARGIN" \
  --print_min "$PRINT_MIN" \
  --print_max "$PRINT_MAX" \
  --eot_brightness "$EOT_BRIGHTNESS" \
  --eot_contrast "$EOT_CONTRAST" \
  --eot_gamma "$EOT_GAMMA" \
  --eot_noise_std "$EOT_NOISE_STD" \
  --seed "$SEED" \
  "${geometry_args[@]}" \
  "${patch_args[@]}" \
  "${apply_args[@]}"

if [[ "$RUN_EVAL" != "1" ]]; then
  log "skip recons_eval because RUN_EVAL=$RUN_EVAL"
  exit 0
fi

log "evaluate TUM-10 clean and geometry-aware attack"
"$RECONS_PY" "$VGGT_ROOT/scripts/eval_vggt_tum_pose_for_recons_eval_tum10.py" \
  --vggt_output_root "$TUM_CLEAN_OUT" \
  --model_name "$TUM_CLEAN_MODEL" \
  --recons_root "$RECONS_ROOT" \
  --overwrite
"$RECONS_PY" "$VGGT_ROOT/scripts/eval_vggt_tum_pose_for_recons_eval_tum10.py" \
  --vggt_output_root "$TUM_GEOM_OUT" \
  --model_name "$TUM_GEOM_MODEL" \
  --recons_root "$RECONS_ROOT" \
  --overwrite

log "all done"
echo "TUM clean:    $RECONS_ROOT/outputs/relpose-distance/tum10-metric-$TUM_CLEAN_MODEL.csv"
echo "TUM geometry: $RECONS_ROOT/outputs/relpose-distance/tum10-metric-$TUM_GEOM_MODEL.csv"
