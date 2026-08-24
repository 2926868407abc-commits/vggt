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
SCENE_PATTERN="${SCENE_PATTERN:-rgbd_dataset_freiburg3_*}"

OUT_BASE="${OUT_BASE:-$VGGT_ROOT/outputs_attack_geometry_aware_tum10}"
TUM_CLEAN_OUT="$OUT_BASE/tum10_clean_uniform_l3"
TUM_GEOM_RUN_NAME="${TUM_GEOM_RUN_NAME:-tum10_vggt_pointmap_geometry_feature_l3}"
TUM_GEOM_OUT="$OUT_BASE/$TUM_GEOM_RUN_NAME"

TUM_CLEAN_MODEL="${TUM_CLEAN_MODEL:-vggt_tum10_clean_uniform_l3_geomrun}"
TUM_GEOM_MODEL="${TUM_GEOM_MODEL:-vggt_tum10_vggt_pointmap_geometry_feature_l3}"

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
POSE_REVERSE_REFERENCE="${POSE_REVERSE_REFERENCE:-gt}"
POSE_BAD_REFERENCE="${POSE_BAD_REFERENCE:-gt}"
POSE_DRIFT_X_M="${POSE_DRIFT_X_M:-0.5}"
POSE_DRIFT_Y_M="${POSE_DRIFT_Y_M:-0.0}"
POSE_DRIFT_Z_M="${POSE_DRIFT_Z_M:-0.0}"
POSE_DRIFT_YAW_DEGREES="${POSE_DRIFT_YAW_DEGREES:-0.0}"
POSE_TRANSLATION_SCALE="${POSE_TRANSLATION_SCALE:-2.0}"
POSE_YAW_DEGREES="${POSE_YAW_DEGREES:-30.0}"
PIECEWISE_GAUGE_FAMILY="${PIECEWISE_GAUGE_FAMILY:-}"
PIECEWISE_GAUGE_MAGNITUDE="${PIECEWISE_GAUGE_MAGNITUDE:-1.0}"
ORTHOGONAL_MODE_ORDER="${ORTHOGONAL_MODE_ORDER:-2}"
ORTHOGONAL_MODE_AXIS="${ORTHOGONAL_MODE_AXIS:-0}"
JOINT_POSE_WEIGHT="${JOINT_POSE_WEIGHT:-1.0}"
JOINT_DEPTH_WEIGHT="${JOINT_DEPTH_WEIGHT:-1.0}"
JOINT_POINT_WEIGHT="${JOINT_POINT_WEIGHT:-1.0}"
JOINT_DEPTH_CONF_WEIGHT="${JOINT_DEPTH_CONF_WEIGHT:-0.0}"
JOINT_POINT_CONF_WEIGHT="${JOINT_POINT_CONF_WEIGHT:-0.0}"
JOINT_TRACK_WEIGHT="${JOINT_TRACK_WEIGHT:-0.0}"
JOINT_TRACK_GRID_ROWS="${JOINT_TRACK_GRID_ROWS:-6}"
JOINT_TRACK_GRID_COLS="${JOINT_TRACK_GRID_COLS:-8}"
JOINT_TRACK_QUERY_MARGIN="${JOINT_TRACK_QUERY_MARGIN:-0.10}"
JOINT_TRACK_ITERS="${JOINT_TRACK_ITERS:-2}"
JOINT_TRACK_MIN_VISIBILITY="${JOINT_TRACK_MIN_VISIBILITY:-0.20}"
JOINT_TRACK_VISIBILITY_WEIGHT="${JOINT_TRACK_VISIBILITY_WEIGHT:-0.10}"
JOINT_TRACK_CONFIDENCE_WEIGHT="${JOINT_TRACK_CONFIDENCE_WEIGHT:-0.10}"
FILTER_BUDGET_JSON="${FILTER_BUDGET_JSON:-$VGGT_ROOT/configs/tum10_filter_budgets.json}"
FILTER_BUDGET_FRACTION="${FILTER_BUDGET_FRACTION:-0.80}"
BUDGET_CONSTRAINTS="${BUDGET_CONSTRAINTS:-conf_std,conf_frac_floor,head_disagree_rel,reproj_rel_err,track}"
BUDGET_POSE_OBJECTIVE="${BUDGET_POSE_OBJECTIVE:-untargeted_gt}"
BUDGET_DUAL_INIT="${BUDGET_DUAL_INIT:-0.0}"
BUDGET_DUAL_LR="${BUDGET_DUAL_LR:-1.0}"
BUDGET_DUAL_MAX="${BUDGET_DUAL_MAX:-100.0}"
BUDGET_DUAL_DECAY="${BUDGET_DUAL_DECAY:-1.0}"
BUDGET_RHO="${BUDGET_RHO:-1.0}"
BUDGET_CONF_FLOOR_TEMPERATURE="${BUDGET_CONF_FLOOR_TEMPERATURE:-0.02}"
BUDGET_REPROJ_STRIDE="${BUDGET_REPROJ_STRIDE:-8}"
BUDGET_TRACK_PIXELS="${BUDGET_TRACK_PIXELS:-2.0}"
POSE_ROTATION_WEIGHT="${POSE_ROTATION_WEIGHT:-1.0}"
POSE_TRANSLATION_WEIGHT="${POSE_TRANSLATION_WEIGHT:-1.0}"
SEED="${SEED:-0}"

PLANE_WIDTH="${PLANE_WIDTH:-0.6}"
PLANE_HEIGHT="${PLANE_HEIGHT:-0.6}"
PLANE_DISTANCE="${PLANE_DISTANCE:-2.0}"
PLANE_CENTER_X="${PLANE_CENTER_X:-0.0}"
PLANE_CENTER_Y="${PLANE_CENTER_Y:-0.0}"
MANUAL_ANCHOR_COORDINATES="${MANUAL_ANCHOR_COORDINATES:-normalized}"
MANUAL_ANCHOR_X="${MANUAL_ANCHOR_X:-0.5}"
MANUAL_ANCHOR_Y="${MANUAL_ANCHOR_Y:-0.5}"
MANUAL_ANCHOR_FRAME="${MANUAL_ANCHOR_FRAME:-0}"
MANUAL_ANCHOR_SEARCH_RADIUS="${MANUAL_ANCHOR_SEARCH_RADIUS:-12}"
MANUAL_ANCHOR_ROLL_DEGREES="${MANUAL_ANCHOR_ROLL_DEGREES:-0}"
MANUAL_QUAD_COORDINATES="${MANUAL_QUAD_COORDINATES:-normalized}"
MANUAL_QUAD_XY="${MANUAL_QUAD_XY:-}"
MANUAL_QUAD_DEPTH_SAMPLE_STRIDE="${MANUAL_QUAD_DEPTH_SAMPLE_STRIDE:-2}"
MANUAL_QUAD_FIT_SHRINK="${MANUAL_QUAD_FIT_SHRINK:-0.70}"
MANUAL_QUAD_PLANE_INLIER_TOLERANCE="${MANUAL_QUAD_PLANE_INLIER_TOLERANCE:-0.06}"
MANUAL_QUAD_MIN_INLIER_RATIO="${MANUAL_QUAD_MIN_INLIER_RATIO:-0.25}"
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
SURFACE_SUPPORT_CHECK="${SURFACE_SUPPORT_CHECK:-0}"
SURFACE_SUPPORT_ABS_TOLERANCE="${SURFACE_SUPPORT_ABS_TOLERANCE:-0.08}"
SURFACE_SUPPORT_REL_TOLERANCE="${SURFACE_SUPPORT_REL_TOLERANCE:-0.05}"
SURFACE_MIN_SUPPORT_RATIO="${SURFACE_MIN_SUPPORT_RATIO:-0.6}"
SURFACE_STRENGTH_SEARCH="${SURFACE_STRENGTH_SEARCH:-0}"
SURFACE_STRENGTH_CANDIDATES="${SURFACE_STRENGTH_CANDIDATES:-1}"
SURFACE_STRENGTH_STEPS="${SURFACE_STRENGTH_STEPS:-0}"
SURFACE_STRENGTH_LR="${SURFACE_STRENGTH_LR:-0.002}"
SURFACE_STRENGTH_TEXTURE_INIT="${SURFACE_STRENGTH_TEXTURE_INIT:-random}"
SURFACE_STRENGTH_REGULARIZATION_WEIGHT="${SURFACE_STRENGTH_REGULARIZATION_WEIGHT:-1.0}"
NATURAL_AUTO_RELAX="${NATURAL_AUTO_RELAX:-0}"
NATURAL_RELAX_MAX_COVERAGE="${NATURAL_RELAX_MAX_COVERAGE:-0.08}"
NATURAL_RELAX_MIN_VISIBLE_FRAMES="${NATURAL_RELAX_MIN_VISIBLE_FRAMES:-2}"
NATURAL_RELAX_MIN_VISIBILITY_RATIO="${NATURAL_RELAX_MIN_VISIBILITY_RATIO:-0.25}"
NATURAL_RELAX_ORIENTATION_FILTER="${NATURAL_RELAX_ORIENTATION_FILTER:-fronto_or_tabletop}"
NATURAL_RELAX_MAX_TILT_DEGREES="${NATURAL_RELAX_MAX_TILT_DEGREES:-50}"
NATURAL_RELAX_MIN_CENTER_DEPTH="${NATURAL_RELAX_MIN_CENTER_DEPTH:-1.0}"
NATURAL_RELAX_MIN_SUPPORT_RATIO="${NATURAL_RELAX_MIN_SUPPORT_RATIO:-0.3}"

PHYSICAL_EOT="${PHYSICAL_EOT:-1}"
PRINT_MIN="${PRINT_MIN:-0.0}"
PRINT_MAX="${PRINT_MAX:-1.0}"
EOT_BRIGHTNESS="${EOT_BRIGHTNESS:-0.15}"
EOT_CONTRAST="${EOT_CONTRAST:-0.15}"
EOT_GAMMA="${EOT_GAMMA:-0.10}"
EOT_NOISE_STD="${EOT_NOISE_STD:-0.01}"
EOT_WARMUP_FRACTION="${EOT_WARMUP_FRACTION:-0.25}"
EOT_GEO_TRANSLATE="${EOT_GEO_TRANSLATE:-0.02}"
EOT_GEO_SCALE="${EOT_GEO_SCALE:-0.03}"
EOT_GEO_ROTATE_DEGREES="${EOT_GEO_ROTATE_DEGREES:-2.0}"
EOT_GEO_PERSPECTIVE="${EOT_GEO_PERSPECTIVE:-0.01}"
EOT_SAMPLES="${EOT_SAMPLES:-1}"
TV_WEIGHT="${TV_WEIGHT:-0.0}"
PRINTABILITY_WEIGHT="${PRINTABILITY_WEIGHT:-0.0}"
PRINTABLE_COLOR_LEVELS="${PRINTABLE_COLOR_LEVELS:-8}"
LOW_FREQUENCY_WEIGHT="${LOW_FREQUENCY_WEIGHT:-0.0}"
LOW_FREQUENCY_KERNEL="${LOW_FREQUENCY_KERNEL:-9}"
NATURAL_REFERENCE_IMAGE="${NATURAL_REFERENCE_IMAGE:-}"
NATURAL_REFERENCE_WEIGHT="${NATURAL_REFERENCE_WEIGHT:-0.0}"

FORCE_PREPARE_TUM10="${FORCE_PREPARE_TUM10:-0}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_APPLY="${FORCE_APPLY:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_GAUGE_DIAG="${RUN_GAUGE_DIAG:-1}"
RUN_CONSISTENCY_CHECK="${RUN_CONSISTENCY_CHECK:-1}"
FREEZE_MODEL_PARAMETERS="${FREEZE_MODEL_PARAMETERS:-1}"

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
echo "texture_size=$TEXTURE_SIZE texture_init=$TEXTURE_INIT freeze_texture=$FREEZE_TEXTURE patch_lr=$PATCH_LR feature_layer=$FEATURE_LAYER"
echo "texture_init_image=$TEXTURE_INIT_IMAGE"
echo "activation_checkpoint=$ACTIVATION_CHECKPOINT"
echo "attack_loss=$ATTACK_LOSS pose_reverse_reference=$POSE_REVERSE_REFERENCE pose_weights=(rot:$POSE_ROTATION_WEIGHT,trans:$POSE_TRANSLATION_WEIGHT)"
echo "filter_budget json=$FILTER_BUDGET_JSON fraction=$FILTER_BUDGET_FRACTION constraints=$BUDGET_CONSTRAINTS pose_objective=$BUDGET_POSE_OBJECTIVE"
echo "pose_bad_reference=$POSE_BAD_REFERENCE drift=($POSE_DRIFT_X_M,$POSE_DRIFT_Y_M,$POSE_DRIFT_Z_M)m drift_yaw=$POSE_DRIFT_YAW_DEGREES scale=$POSE_TRANSLATION_SCALE yaw=$POSE_YAW_DEGREES"
echo "plane_width=$PLANE_WIDTH plane_height=$PLANE_HEIGHT plane_distance=$PLANE_DISTANCE"
echo "plane_center=($PLANE_CENTER_X,$PLANE_CENTER_Y)"
echo "scene_pattern=$SCENE_PATTERN manual_anchor=($MANUAL_ANCHOR_X,$MANUAL_ANCHOR_Y) frame=$MANUAL_ANCHOR_FRAME roll=$MANUAL_ANCHOR_ROLL_DEGREES"
echo "manual_quad coordinates=$MANUAL_QUAD_COORDINATES xy=$MANUAL_QUAD_XY depth_sample_stride=$MANUAL_QUAD_DEPTH_SAMPLE_STRIDE"
echo "manual_quad robust_fit shrink=$MANUAL_QUAD_FIT_SHRINK inlier_tol=$MANUAL_QUAD_PLANE_INLIER_TOLERANCE min_inlier_ratio=$MANUAL_QUAD_MIN_INLIER_RATIO"
echo "plane_mode=$PLANE_MODE clean_vggt_output_root=$TUM_CLEAN_OUT use_depth_visibility=$USE_DEPTH_VISIBILITY optimize_geometry=$OPTIMIZE_GEOMETRY"
echo "surface_candidate_grid=$SURFACE_CANDIDATE_GRID geometry_size_scales=$GEOMETRY_SIZE_SCALES geometry_roll_degrees=$GEOMETRY_ROLL_DEGREES"
echo "fused_point_stride=$FUSED_POINT_STRIDE fused_surface_candidates=$FUSED_SURFACE_CANDIDATES fused_normal_radius=$FUSED_NORMAL_RADIUS"
echo "surface_score_mode=$SURFACE_SCORE_MODE coverage=[$SURFACE_COVERAGE_MIN,$SURFACE_COVERAGE_MAX] min_visible_frames=$SURFACE_MIN_VISIBLE_FRAMES min_visibility_ratio=$SURFACE_MIN_VISIBILITY_RATIO"
echo "surface_orientation_filter=$SURFACE_ORIENTATION_FILTER max_tilt=$SURFACE_MAX_TILT_DEGREES center_depth=[$SURFACE_MIN_CENTER_DEPTH,$SURFACE_MAX_CENTER_DEPTH]"
echo "surface_support_check=$SURFACE_SUPPORT_CHECK abs_tol=$SURFACE_SUPPORT_ABS_TOLERANCE rel_tol=$SURFACE_SUPPORT_REL_TOLERANCE min_support=$SURFACE_MIN_SUPPORT_RATIO"
echo "surface_strength_search=$SURFACE_STRENGTH_SEARCH candidates=$SURFACE_STRENGTH_CANDIDATES steps=$SURFACE_STRENGTH_STEPS strength_lr=$SURFACE_STRENGTH_LR"
echo "natural_auto_relax=$NATURAL_AUTO_RELAX max_coverage=$NATURAL_RELAX_MAX_COVERAGE min_visible_frames=$NATURAL_RELAX_MIN_VISIBLE_FRAMES min_visibility=$NATURAL_RELAX_MIN_VISIBILITY_RATIO orientation=$NATURAL_RELAX_ORIENTATION_FILTER"
echo "physical_eot=$PHYSICAL_EOT print=[$PRINT_MIN,$PRINT_MAX] brightness=$EOT_BRIGHTNESS contrast=$EOT_CONTRAST gamma=$EOT_GAMMA noise_std=$EOT_NOISE_STD warmup_fraction=$EOT_WARMUP_FRACTION"
echo "eot_geo translate=$EOT_GEO_TRANSLATE scale=$EOT_GEO_SCALE rotate=$EOT_GEO_ROTATE_DEGREES perspective=$EOT_GEO_PERSPECTIVE samples=$EOT_SAMPLES"
echo "regularization tv=$TV_WEIGHT printability=$PRINTABILITY_WEIGHT levels=$PRINTABLE_COLOR_LEVELS low_frequency=$LOW_FREQUENCY_WEIGHT kernel=$LOW_FREQUENCY_KERNEL"
echo "natural_reference image=$NATURAL_REFERENCE_IMAGE weight=$NATURAL_REFERENCE_WEIGHT"
echo "run_eval=$RUN_EVAL run_gauge_diag=$RUN_GAUGE_DIAG run_consistency_check=$RUN_CONSISTENCY_CHECK"

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
if [[ "$SURFACE_STRENGTH_SEARCH" == "1" ]]; then
  geometry_args+=(--surface_strength_search)
fi
if [[ "$NATURAL_AUTO_RELAX" == "1" ]]; then
  geometry_args+=(--natural_auto_relax)
fi
if [[ "$SURFACE_SUPPORT_CHECK" == "1" ]]; then
  geometry_args+=(--surface_support_check)
fi
if [[ "$FREEZE_TEXTURE" == "1" ]]; then
  geometry_args+=(--freeze_texture)
fi
if [[ "$ACTIVATION_CHECKPOINT" == "1" ]]; then
  geometry_args+=(--activation_checkpoint)
fi
if [[ "$FREEZE_MODEL_PARAMETERS" != "1" ]]; then
  geometry_args+=(--no_freeze_model_parameters)
fi
if [[ -n "$PIECEWISE_GAUGE_FAMILY" ]]; then
  geometry_args+=(--piecewise_gauge_family "$PIECEWISE_GAUGE_FAMILY")
fi
texture_args=()
if [[ -n "$TEXTURE_INIT_IMAGE" ]]; then
  texture_args+=(--texture_init_image "$TEXTURE_INIT_IMAGE")
fi
if [[ -n "$NATURAL_REFERENCE_IMAGE" ]]; then
  texture_args+=(--natural_reference_image "$NATURAL_REFERENCE_IMAGE")
fi
"$VGGT_PY" "$VGGT_ROOT/attack_vggt_geometry_tum10.py" \
  --tum_root "$TUM_ROOT" \
  --scene_pattern "$SCENE_PATTERN" \
  --output_dir "$TUM_GEOM_OUT" \
  --frame_manifest "$TUM10_FRAME_MANIFEST" \
  --ckpt "$CKPT" \
  --texture_size "$TEXTURE_SIZE" \
  --texture_init "$TEXTURE_INIT" \
  --iterations "$ITERATIONS" \
  --inner_loop "$INNER_LOOP" \
  --scenes_per_iteration "$SCENES_PER_ITERATION" \
  --patch_lr "$PATCH_LR" \
  --feature_layer "$FEATURE_LAYER" \
  --attack_loss "$ATTACK_LOSS" \
  --pose_reverse_reference "$POSE_REVERSE_REFERENCE" \
  --pose_bad_reference "$POSE_BAD_REFERENCE" \
  --pose_drift_x_m "$POSE_DRIFT_X_M" \
  --pose_drift_y_m "$POSE_DRIFT_Y_M" \
  --pose_drift_z_m "$POSE_DRIFT_Z_M" \
  --pose_drift_yaw_degrees "$POSE_DRIFT_YAW_DEGREES" \
  --pose_translation_scale "$POSE_TRANSLATION_SCALE" \
  --pose_yaw_degrees "$POSE_YAW_DEGREES" \
  --piecewise_gauge_magnitude "$PIECEWISE_GAUGE_MAGNITUDE" \
  --orthogonal_mode_order "$ORTHOGONAL_MODE_ORDER" \
  --orthogonal_mode_axis "$ORTHOGONAL_MODE_AXIS" \
  --joint_pose_weight "$JOINT_POSE_WEIGHT" \
  --joint_depth_weight "$JOINT_DEPTH_WEIGHT" \
  --joint_point_weight "$JOINT_POINT_WEIGHT" \
  --joint_depth_conf_weight "$JOINT_DEPTH_CONF_WEIGHT" \
  --joint_point_conf_weight "$JOINT_POINT_CONF_WEIGHT" \
  --joint_track_weight "$JOINT_TRACK_WEIGHT" \
  --joint_track_grid_rows "$JOINT_TRACK_GRID_ROWS" \
  --joint_track_grid_cols "$JOINT_TRACK_GRID_COLS" \
  --joint_track_query_margin "$JOINT_TRACK_QUERY_MARGIN" \
  --joint_track_iters "$JOINT_TRACK_ITERS" \
  --joint_track_min_visibility "$JOINT_TRACK_MIN_VISIBILITY" \
  --joint_track_visibility_weight "$JOINT_TRACK_VISIBILITY_WEIGHT" \
  --joint_track_confidence_weight "$JOINT_TRACK_CONFIDENCE_WEIGHT" \
  --filter_budget_json "$FILTER_BUDGET_JSON" \
  --filter_budget_fraction "$FILTER_BUDGET_FRACTION" \
  --budget_constraints "$BUDGET_CONSTRAINTS" \
  --budget_pose_objective "$BUDGET_POSE_OBJECTIVE" \
  --budget_dual_init "$BUDGET_DUAL_INIT" \
  --budget_dual_lr "$BUDGET_DUAL_LR" \
  --budget_dual_max "$BUDGET_DUAL_MAX" \
  --budget_dual_decay "$BUDGET_DUAL_DECAY" \
  --budget_rho "$BUDGET_RHO" \
  --budget_conf_floor_temperature "$BUDGET_CONF_FLOOR_TEMPERATURE" \
  --budget_reproj_stride "$BUDGET_REPROJ_STRIDE" \
  --budget_track_pixels "$BUDGET_TRACK_PIXELS" \
  --pose_rotation_weight "$POSE_ROTATION_WEIGHT" \
  --pose_translation_weight "$POSE_TRANSLATION_WEIGHT" \
  --plane_width "$PLANE_WIDTH" \
  --plane_height "$PLANE_HEIGHT" \
  --plane_distance "$PLANE_DISTANCE" \
  --plane_center_x "$PLANE_CENTER_X" \
  --plane_center_y "$PLANE_CENTER_Y" \
  --manual_anchor_coordinates "$MANUAL_ANCHOR_COORDINATES" \
  --manual_anchor_x "$MANUAL_ANCHOR_X" \
  --manual_anchor_y "$MANUAL_ANCHOR_Y" \
  --manual_anchor_frame "$MANUAL_ANCHOR_FRAME" \
  --manual_anchor_search_radius "$MANUAL_ANCHOR_SEARCH_RADIUS" \
  --manual_anchor_roll_degrees "$MANUAL_ANCHOR_ROLL_DEGREES" \
  --manual_quad_coordinates "$MANUAL_QUAD_COORDINATES" \
  --manual_quad_xy "$MANUAL_QUAD_XY" \
  --manual_quad_depth_sample_stride "$MANUAL_QUAD_DEPTH_SAMPLE_STRIDE" \
  --manual_quad_fit_shrink "$MANUAL_QUAD_FIT_SHRINK" \
  --manual_quad_plane_inlier_tolerance "$MANUAL_QUAD_PLANE_INLIER_TOLERANCE" \
  --manual_quad_min_inlier_ratio "$MANUAL_QUAD_MIN_INLIER_RATIO" \
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
  --visibility_depth_margin "$VISIBILITY_DEPTH_MARGIN" \
  --print_min "$PRINT_MIN" \
  --print_max "$PRINT_MAX" \
  --eot_brightness "$EOT_BRIGHTNESS" \
  --eot_contrast "$EOT_CONTRAST" \
  --eot_gamma "$EOT_GAMMA" \
  --eot_noise_std "$EOT_NOISE_STD" \
  --eot_warmup_fraction "$EOT_WARMUP_FRACTION" \
  --eot_geo_translate "$EOT_GEO_TRANSLATE" \
  --eot_geo_scale "$EOT_GEO_SCALE" \
  --eot_geo_rotate_degrees "$EOT_GEO_ROTATE_DEGREES" \
  --eot_geo_perspective "$EOT_GEO_PERSPECTIVE" \
  --eot_samples "$EOT_SAMPLES" \
  --tv_weight "$TV_WEIGHT" \
  --printability_weight "$PRINTABILITY_WEIGHT" \
  --printable_color_levels "$PRINTABLE_COLOR_LEVELS" \
  --low_frequency_weight "$LOW_FREQUENCY_WEIGHT" \
  --low_frequency_kernel "$LOW_FREQUENCY_KERNEL" \
  --natural_reference_weight "$NATURAL_REFERENCE_WEIGHT" \
  --seed "$SEED" \
  "${texture_args[@]}" \
  "${geometry_args[@]}" \
  "${patch_args[@]}" \
  "${apply_args[@]}"

if [[ "$RUN_CONSISTENCY_CHECK" == "1" ]]; then
  # Recompute the run's own attack loss on the saved prediction. A large gap means
  # the trained attack did not survive into what actually gets evaluated -- the EOT
  # jitter is applied during training but not when the outputs are written.
  # Report only; it never fails the run.
  log "train/test consistency check"
  "$VGGT_PY" "$VGGT_ROOT/scripts/check_train_test_consistency.py" \
    --vggt_output_root "$TUM_GEOM_OUT" \
    --tum_root "$TUM_ROOT" \
    --scene_pattern "$SCENE_PATTERN" \
    --out_csv "$TUM_GEOM_OUT/train_test_consistency.csv" || true
fi

if [[ "$RUN_GAUGE_DIAG" == "1" ]]; then
  # Gauge diagnostics: how much of the trajectory damage is a global Sim(3) that
  # the Sim(3)-aligned ATE discards. Writes its own CSV only; it does not touch
  # the ATE/RPE evaluation or its outputs.
  log "gauge diagnostics (gauge_absorbed_frac / sim3_scale_vs_clean)"
  "$RECONS_PY" "$VGGT_ROOT/scripts/eval_gauge_absorbed_frac.py" \
    --vggt_output_root "$TUM_GEOM_OUT" \
    --model_name "$TUM_GEOM_MODEL" \
    --recons_root "$RECONS_ROOT" \
    --tum_root "$TUM_ROOT" \
    --clean_vggt_output_root "$TUM_CLEAN_OUT" \
    --scene_pattern "$SCENE_PATTERN"
fi

if [[ "$RUN_EVAL" != "1" ]]; then
  log "skip recons_eval because RUN_EVAL=$RUN_EVAL"
  exit 0
fi

log "evaluate TUM-10 clean and geometry-aware attack"
"$RECONS_PY" "$VGGT_ROOT/scripts/eval_vggt_tum_pose_for_recons_eval_tum10.py" \
  --vggt_output_root "$TUM_CLEAN_OUT" \
  --model_name "$TUM_CLEAN_MODEL" \
  --recons_root "$RECONS_ROOT" \
  --scene_pattern "$SCENE_PATTERN" \
  --overwrite
"$RECONS_PY" "$VGGT_ROOT/scripts/eval_vggt_tum_pose_for_recons_eval_tum10.py" \
  --vggt_output_root "$TUM_GEOM_OUT" \
  --model_name "$TUM_GEOM_MODEL" \
  --recons_root "$RECONS_ROOT" \
  --scene_pattern "$SCENE_PATTERN" \
  --overwrite

log "all done"
echo "TUM clean:    $RECONS_ROOT/outputs/relpose-distance/tum10-metric-$TUM_CLEAN_MODEL.csv"
echo "TUM geometry: $RECONS_ROOT/outputs/relpose-distance/tum10-metric-$TUM_GEOM_MODEL.csv"
if [[ "$RUN_GAUGE_DIAG" == "1" ]]; then
  echo "TUM gauge:    $RECONS_ROOT/outputs/relpose-distance/tum10-gauge-$TUM_GEOM_MODEL.csv"
fi
