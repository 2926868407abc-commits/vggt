#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
RECONS_ROOT="${RECONS_ROOT:-/mnt/data/wangqq/recons_eval}"
VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"
RECONS_PY="${RECONS_PY:-/mnt/data/wangqq/conda_envs/recons_eval/bin/python3}"
CKPT="${CKPT:-$VGGT_ROOT/checkpoints/VGGT-1B}"

NRGBD_ROOT="${NRGBD_ROOT:-$RECONS_ROOT/data/nrgbd}"
NRGBD_SEQ_MAP="${NRGBD_SEQ_MAP:-$RECONS_ROOT/datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf500.json}"
NRGBD_GEOM_SCENES="${NRGBD_GEOM_SCENES:-$VGGT_ROOT/data/nrgbd_sparse_geometry_scenes}"
NRGBD_GEOM_MANIFEST="${NRGBD_GEOM_MANIFEST:-$NRGBD_GEOM_SCENES/nrgbd_geometry_frame_manifest.json}"
SCENE_PATTERN="${SCENE_PATTERN:-*}"

OUT_BASE="${OUT_BASE:-$VGGT_ROOT/outputs_attack_geometry_aware_nrgbd}"
NRGBD_GEOM_RUN_NAME="${NRGBD_GEOM_RUN_NAME:-nrgbd_sparse_gt_depth_geometry_feature_l3}"
NRGBD_GEOM_OUT="$OUT_BASE/$NRGBD_GEOM_RUN_NAME"
NRGBD_GEOM_MODEL="${NRGBD_GEOM_MODEL:-vggt_nrgbd_sparse_gt_depth_geometry_feature_l3}"

ITERATIONS="${ITERATIONS:-200}"
INNER_LOOP="${INNER_LOOP:-10}"
SCENES_PER_ITERATION="${SCENES_PER_ITERATION:-1}"
PATCH_LR="${PATCH_LR:-0.001}"
TEXTURE_SIZE="${TEXTURE_SIZE:-128}"
TEXTURE_INIT="${TEXTURE_INIT:-random}"
TEXTURE_INIT_IMAGE="${TEXTURE_INIT_IMAGE:-}"
FREEZE_TEXTURE="${FREEZE_TEXTURE:-0}"
FEATURE_LAYER="${FEATURE_LAYER:-aggregator_final}"
ATTACK_LOSS="${ATTACK_LOSS:-feature_l1}"
ACTIVATION_CHECKPOINT="${ACTIVATION_CHECKPOINT:-0}"
SEED="${SEED:-0}"

PLANE_MODE="${PLANE_MODE:-fused_depth_surface}"
PLANE_WIDTH="${PLANE_WIDTH:-0.30}"
PLANE_HEIGHT="${PLANE_HEIGHT:-0.30}"
PLANE_DISTANCE="${PLANE_DISTANCE:-2.0}"
PLANE_CENTER_X="${PLANE_CENTER_X:-0.0}"
PLANE_CENTER_Y="${PLANE_CENTER_Y:-0.0}"
USE_DEPTH_VISIBILITY="${USE_DEPTH_VISIBILITY:-1}"
OPTIMIZE_GEOMETRY="${OPTIMIZE_GEOMETRY:-1}"
SURFACE_CANDIDATE_GRID="${SURFACE_CANDIDATE_GRID:-5}"
SURFACE_SEARCH_MARGIN="${SURFACE_SEARCH_MARGIN:-0.18}"
GEOMETRY_SIZE_SCALES="${GEOMETRY_SIZE_SCALES:-0.75,1.0,1.25}"
GEOMETRY_ROLL_DEGREES="${GEOMETRY_ROLL_DEGREES:--30,-15,0,15,30}"
VISIBILITY_DEPTH_MARGIN="${VISIBILITY_DEPTH_MARGIN:-0.05}"

FUSED_POINT_STRIDE="${FUSED_POINT_STRIDE:-20}"
FUSED_MAX_POINTS="${FUSED_MAX_POINTS:-8000}"
FUSED_SURFACE_CANDIDATES="${FUSED_SURFACE_CANDIDATES:-128}"
FUSED_NORMAL_RADIUS="${FUSED_NORMAL_RADIUS:-0.25}"
FUSED_MIN_NEIGHBORS="${FUSED_MIN_NEIGHBORS:-16}"
FUSED_MAX_NEIGHBORS="${FUSED_MAX_NEIGHBORS:-256}"
FUSED_MAX_PLANE_RESIDUAL="${FUSED_MAX_PLANE_RESIDUAL:-0.08}"

SURFACE_SCORE_MODE="${SURFACE_SCORE_MODE:-natural}"
SURFACE_COVERAGE_MIN="${SURFACE_COVERAGE_MIN:-0.003}"
SURFACE_COVERAGE_MAX="${SURFACE_COVERAGE_MAX:-0.08}"
SURFACE_MIN_VISIBLE_FRAMES="${SURFACE_MIN_VISIBLE_FRAMES:-1}"
SURFACE_MIN_VISIBILITY_RATIO="${SURFACE_MIN_VISIBILITY_RATIO:-0.4}"
SURFACE_ORIENTATION_FILTER="${SURFACE_ORIENTATION_FILTER:-none}"
SURFACE_MAX_TILT_DEGREES="${SURFACE_MAX_TILT_DEGREES:-50}"
SURFACE_MIN_CENTER_DEPTH="${SURFACE_MIN_CENTER_DEPTH:-0.5}"
SURFACE_MAX_CENTER_DEPTH="${SURFACE_MAX_CENTER_DEPTH:-6.0}"
SURFACE_SUPPORT_CHECK="${SURFACE_SUPPORT_CHECK:-1}"
SURFACE_SUPPORT_ABS_TOLERANCE="${SURFACE_SUPPORT_ABS_TOLERANCE:-0.08}"
SURFACE_SUPPORT_REL_TOLERANCE="${SURFACE_SUPPORT_REL_TOLERANCE:-0.05}"
SURFACE_MIN_SUPPORT_RATIO="${SURFACE_MIN_SUPPORT_RATIO:-0.45}"
SURFACE_STRENGTH_SEARCH="${SURFACE_STRENGTH_SEARCH:-0}"
SURFACE_STRENGTH_CANDIDATES="${SURFACE_STRENGTH_CANDIDATES:-8}"
SURFACE_STRENGTH_STEPS="${SURFACE_STRENGTH_STEPS:-8}"
SURFACE_STRENGTH_LR="${SURFACE_STRENGTH_LR:-0.002}"
SURFACE_STRENGTH_TEXTURE_INIT="${SURFACE_STRENGTH_TEXTURE_INIT:-checker}"
SURFACE_STRENGTH_REGULARIZATION_WEIGHT="${SURFACE_STRENGTH_REGULARIZATION_WEIGHT:-0.2}"
NATURAL_AUTO_RELAX="${NATURAL_AUTO_RELAX:-1}"
NATURAL_RELAX_MAX_COVERAGE="${NATURAL_RELAX_MAX_COVERAGE:-0.10}"
NATURAL_RELAX_MIN_VISIBLE_FRAMES="${NATURAL_RELAX_MIN_VISIBLE_FRAMES:-1}"
NATURAL_RELAX_MIN_VISIBILITY_RATIO="${NATURAL_RELAX_MIN_VISIBILITY_RATIO:-0.25}"
NATURAL_RELAX_ORIENTATION_FILTER="${NATURAL_RELAX_ORIENTATION_FILTER:-none}"
NATURAL_RELAX_MAX_TILT_DEGREES="${NATURAL_RELAX_MAX_TILT_DEGREES:-65}"
NATURAL_RELAX_MIN_CENTER_DEPTH="${NATURAL_RELAX_MIN_CENTER_DEPTH:-0.4}"
NATURAL_RELAX_MIN_SUPPORT_RATIO="${NATURAL_RELAX_MIN_SUPPORT_RATIO:-0.30}"

PHYSICAL_EOT="${PHYSICAL_EOT:-1}"
PRINT_MIN="${PRINT_MIN:-0.0}"
PRINT_MAX="${PRINT_MAX:-1.0}"
EOT_BRIGHTNESS="${EOT_BRIGHTNESS:-0.15}"
EOT_CONTRAST="${EOT_CONTRAST:-0.15}"
EOT_GAMMA="${EOT_GAMMA:-0.10}"
EOT_NOISE_STD="${EOT_NOISE_STD:-0.01}"
TV_WEIGHT="${TV_WEIGHT:-0.001}"
PRINTABILITY_WEIGHT="${PRINTABILITY_WEIGHT:-0.001}"
PRINTABLE_COLOR_LEVELS="${PRINTABLE_COLOR_LEVELS:-4}"
LOW_FREQUENCY_WEIGHT="${LOW_FREQUENCY_WEIGHT:-0.0}"
LOW_FREQUENCY_KERNEL="${LOW_FREQUENCY_KERNEL:-9}"
NATURAL_REFERENCE_IMAGE="${NATURAL_REFERENCE_IMAGE:-}"
NATURAL_REFERENCE_WEIGHT="${NATURAL_REFERENCE_WEIGHT:-0.0}"

FORCE_PREPARE="${FORCE_PREPARE:-0}"
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
require_file "$VGGT_ROOT/scripts/prepare_nrgbd_geometry_scenes.py"
require_file "$RECONS_ROOT/scripts/eval_vggt_nrgbd_mv_recon_for_recons_eval.py"
require_file "$NRGBD_SEQ_MAP"
require_dir "$NRGBD_ROOT"
[[ -e "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }

log "settings"
echo "iterations=$ITERATIONS inner_loop=$INNER_LOOP scenes_per_iteration=$SCENES_PER_ITERATION"
echo "attack_loss=$ATTACK_LOSS feature_layer=$FEATURE_LAYER texture_init=$TEXTURE_INIT patch_lr=$PATCH_LR"
echo "plane_mode=$PLANE_MODE plane_size=($PLANE_WIDTH,$PLANE_HEIGHT)m use_depth_visibility=$USE_DEPTH_VISIBILITY"
echo "surface_score_mode=$SURFACE_SCORE_MODE coverage=[$SURFACE_COVERAGE_MIN,$SURFACE_COVERAGE_MAX]"
echo "surface_strength_search=$SURFACE_STRENGTH_SEARCH candidates=$SURFACE_STRENGTH_CANDIDATES steps=$SURFACE_STRENGTH_STEPS"
echo "output=$NRGBD_GEOM_OUT model=$NRGBD_GEOM_MODEL"

log "prepare NRGBD geometry scenes"
prepare_args=()
if [[ "$FORCE_PREPARE" == "1" ]]; then
  prepare_args=(--overwrite)
fi
"$VGGT_PY" "$VGGT_ROOT/scripts/prepare_nrgbd_geometry_scenes.py" \
  --nrgbd_root "$NRGBD_ROOT" \
  --seq_id_map "$NRGBD_SEQ_MAP" \
  --out_root "$NRGBD_GEOM_SCENES" \
  --manifest_out "$NRGBD_GEOM_MANIFEST" \
  "${prepare_args[@]}"

log "run NRGBD GT-geometry-aware planar patch attack"
patch_args=()
if [[ "$FORCE_TRAIN" != "1" && -f "$NRGBD_GEOM_OUT/geometry_patch/geometry_patch_texture.npz" ]]; then
  patch_args=(--texture_path "$NRGBD_GEOM_OUT/geometry_patch/geometry_patch_texture.npz")
  log "reuse geometry patch -> ${patch_args[1]}"
fi
apply_args=()
if [[ "$FORCE_APPLY" != "1" && "$FORCE_TRAIN" != "1" ]]; then
  apply_args=(--skip_existing_outputs)
fi
flag_args=()
if [[ "$USE_DEPTH_VISIBILITY" == "1" ]]; then
  flag_args+=(--use_depth_visibility)
fi
if [[ "$OPTIMIZE_GEOMETRY" == "1" ]]; then
  flag_args+=(--optimize_geometry)
fi
if [[ "$PHYSICAL_EOT" == "1" ]]; then
  flag_args+=(--physical_eot)
fi
if [[ "$SURFACE_STRENGTH_SEARCH" == "1" ]]; then
  flag_args+=(--surface_strength_search)
fi
if [[ "$NATURAL_AUTO_RELAX" == "1" ]]; then
  flag_args+=(--natural_auto_relax)
fi
if [[ "$SURFACE_SUPPORT_CHECK" == "1" ]]; then
  flag_args+=(--surface_support_check)
fi
if [[ "$FREEZE_TEXTURE" == "1" ]]; then
  flag_args+=(--freeze_texture)
fi
if [[ "$ACTIVATION_CHECKPOINT" == "1" ]]; then
  flag_args+=(--activation_checkpoint)
fi
texture_args=()
if [[ "$TEXTURE_INIT_IMAGE" != "" ]]; then
  texture_args+=(--texture_init_image "$TEXTURE_INIT_IMAGE")
fi
natural_args=()
if [[ "$NATURAL_REFERENCE_IMAGE" != "" ]]; then
  natural_args+=(--natural_reference_image "$NATURAL_REFERENCE_IMAGE")
fi

"$VGGT_PY" "$VGGT_ROOT/attack_vggt_geometry_tum10.py" \
  --tum_root "$NRGBD_GEOM_SCENES" \
  --scene_pattern "$SCENE_PATTERN" \
  --output_dir "$NRGBD_GEOM_OUT" \
  --frame_manifest "$NRGBD_GEOM_MANIFEST" \
  --ckpt "$CKPT" \
  --gt_name groundtruth_90.txt \
  --fx 554.2562584220408 \
  --fy 554.2562584220408 \
  --cx 320 \
  --cy 240 \
  --depth_scale 1000 \
  --texture_size "$TEXTURE_SIZE" \
  --texture_init "$TEXTURE_INIT" \
  "${texture_args[@]}" \
  --iterations "$ITERATIONS" \
  --inner_loop "$INNER_LOOP" \
  --scenes_per_iteration "$SCENES_PER_ITERATION" \
  --patch_lr "$PATCH_LR" \
  --feature_layer "$FEATURE_LAYER" \
  --attack_loss "$ATTACK_LOSS" \
  --plane_mode "$PLANE_MODE" \
  --plane_width "$PLANE_WIDTH" \
  --plane_height "$PLANE_HEIGHT" \
  --plane_distance "$PLANE_DISTANCE" \
  --plane_center_x "$PLANE_CENTER_X" \
  --plane_center_y "$PLANE_CENTER_Y" \
  --surface_candidate_grid "$SURFACE_CANDIDATE_GRID" \
  --surface_search_margin "$SURFACE_SEARCH_MARGIN" \
  --geometry_size_scales="$GEOMETRY_SIZE_SCALES" \
  --geometry_roll_degrees="$GEOMETRY_ROLL_DEGREES" \
  --visibility_depth_margin "$VISIBILITY_DEPTH_MARGIN" \
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
  --surface_support_abs_tolerance "$SURFACE_SUPPORT_ABS_TOLERANCE" \
  --surface_support_rel_tolerance "$SURFACE_SUPPORT_REL_TOLERANCE" \
  --surface_min_support_ratio "$SURFACE_MIN_SUPPORT_RATIO" \
  --surface_strength_candidates "$SURFACE_STRENGTH_CANDIDATES" \
  --surface_strength_steps "$SURFACE_STRENGTH_STEPS" \
  --surface_strength_lr "$SURFACE_STRENGTH_LR" \
  --surface_strength_texture_init "$SURFACE_STRENGTH_TEXTURE_INIT" \
  --surface_strength_regularization_weight "$SURFACE_STRENGTH_REGULARIZATION_WEIGHT" \
  --natural_relax_max_coverage "$NATURAL_RELAX_MAX_COVERAGE" \
  --natural_relax_min_visible_frames "$NATURAL_RELAX_MIN_VISIBLE_FRAMES" \
  --natural_relax_min_visibility_ratio "$NATURAL_RELAX_MIN_VISIBILITY_RATIO" \
  --natural_relax_orientation_filter "$NATURAL_RELAX_ORIENTATION_FILTER" \
  --natural_relax_max_tilt_degrees "$NATURAL_RELAX_MAX_TILT_DEGREES" \
  --natural_relax_min_center_depth "$NATURAL_RELAX_MIN_CENTER_DEPTH" \
  --natural_relax_min_support_ratio "$NATURAL_RELAX_MIN_SUPPORT_RATIO" \
  --print_min "$PRINT_MIN" \
  --print_max "$PRINT_MAX" \
  --eot_brightness "$EOT_BRIGHTNESS" \
  --eot_contrast "$EOT_CONTRAST" \
  --eot_gamma "$EOT_GAMMA" \
  --eot_noise_std "$EOT_NOISE_STD" \
  --tv_weight "$TV_WEIGHT" \
  --printability_weight "$PRINTABILITY_WEIGHT" \
  --printable_color_levels "$PRINTABLE_COLOR_LEVELS" \
  --low_frequency_weight "$LOW_FREQUENCY_WEIGHT" \
  --low_frequency_kernel "$LOW_FREQUENCY_KERNEL" \
  "${natural_args[@]}" \
  --natural_reference_weight "$NATURAL_REFERENCE_WEIGHT" \
  --seed "$SEED" \
  "${flag_args[@]}" \
  "${patch_args[@]}" \
  "${apply_args[@]}"

if [[ "$RUN_EVAL" == "1" ]]; then
  log "evaluate NRGBD point map"
  cd "$RECONS_ROOT"
  "$RECONS_PY" "$RECONS_ROOT/scripts/eval_vggt_nrgbd_mv_recon_for_recons_eval.py" \
    --vggt_output_root "$NRGBD_GEOM_OUT" \
    --dataset_name NRGBD-sparse \
    --model_name "$NRGBD_GEOM_MODEL" \
    --pred_key depth_unproject \
    --overwrite \
    --no_save_ply
fi

log "all done"
echo "NRGBD geometry: $RECONS_ROOT/outputs/mv_recon/NRGBD-sparse-metric-$NRGBD_GEOM_MODEL.csv"
