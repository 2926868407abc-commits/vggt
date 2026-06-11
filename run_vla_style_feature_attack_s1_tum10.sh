#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
RECONS_ROOT="${RECONS_ROOT:-/mnt/data/wangqq/recons_eval}"
VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"
RECONS_PY="${RECONS_PY:-/mnt/data/wangqq/conda_envs/recons_eval/bin/python3}"
CKPT="${CKPT:-$VGGT_ROOT/checkpoints/VGGT-1B}"

ITERATIONS="${ITERATIONS:-2000}"
INNER_LOOP="${INNER_LOOP:-50}"
PATCH_LR="${PATCH_LR:-0.001}"
PATCH_AREA_RATIO="${PATCH_AREA_RATIO:-0.05}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-20}"
ROTATION_DEGREES="${ROTATION_DEGREES:-30}"
SHEAR="${SHEAR:-0.2}"
GEOMETRY_PROB="${GEOMETRY_PROB:-0.8}"
FEATURE_LAYER="${FEATURE_LAYER:-aggregator_final}"
SEED="${SEED:-0}"

PATCH_X="${PATCH_X:--1}"
PATCH_Y="${PATCH_Y:--1}"
NYU_PATCH_X="${NYU_PATCH_X:-$PATCH_X}"
NYU_PATCH_Y="${NYU_PATCH_Y:-$PATCH_Y}"
BONN_PATCH_X="${BONN_PATCH_X:-$PATCH_X}"
BONN_PATCH_Y="${BONN_PATCH_Y:-$PATCH_Y}"
TUM_PATCH_X="${TUM_PATCH_X:-$PATCH_X}"
TUM_PATCH_Y="${TUM_PATCH_Y:-$PATCH_Y}"

NYU_SCENES="${NYU_SCENES:-$VGGT_ROOT/data/nyu_v2_recons_eval_scenes}"
BONN_SCENES="${BONN_SCENES:-$VGGT_ROOT/data/bonn_monodepth_scenes}"
TUM_ROOT="${TUM_ROOT:-$RECONS_ROOT/data/tum}"
TUM10_FRAME_SCENES="${TUM10_FRAME_SCENES:-$VGGT_ROOT/data/tum_dynamics_10frame_individual_scenes}"
TUM10_FRAME_MANIFEST="${TUM10_FRAME_MANIFEST:-$TUM10_FRAME_SCENES/tum10_frame_manifest.json}"
TUM_FRAME_COUNT="${TUM_FRAME_COUNT:-10}"

SINGLE_FRAME_SCENES_PER_ITERATION="${SINGLE_FRAME_SCENES_PER_ITERATION:-1}"
TUM_FRAME_S1_SCENES_PER_ITERATION="${TUM_FRAME_S1_SCENES_PER_ITERATION:-1}"
TUM_FRAME_S10_SCENES_PER_ITERATION="${TUM_FRAME_S10_SCENES_PER_ITERATION:-10}"

FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_APPLY="${FORCE_APPLY:-0}"
FORCE_PREPARE_TUM10="${FORCE_PREPARE_TUM10:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

OUT_BASE="${OUT_BASE:-$VGGT_ROOT/outputs_attack_vla_style_s1_tum10}"
NYU_OUT="$OUT_BASE/nyu_v2_vla_universal_feature_s1_l3"
BONN_OUT="$OUT_BASE/bonn_vla_universal_feature_s1_l3"
TUM_FRAME_S1_OUT="$OUT_BASE/tum10_frame_s1_vla_universal_feature_l3"
TUM_FRAME_S10_OUT="$OUT_BASE/tum10_frame_s10_vla_universal_feature_l3"
TUM_CLEAN_OUT="$OUT_BASE/tum10_clean_uniform_l3"

NYU_MODEL="vggt_nyu_v2_vla_universal_feature_s1_l3"
BONN_MODEL="vggt_bonn_vla_universal_feature_s1_l3"
TUM_FRAME_S1_MODEL="vggt_tum10_frame_s1_vla_universal_feature_l3"
TUM_FRAME_S10_MODEL="vggt_tum10_frame_s10_vla_universal_feature_l3"
TUM_CLEAN_MODEL="vggt_tum10_clean_uniform_l3"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }
}

run_vla_attack() {
  local out_dir="$1"
  shift
  local attack_args=("$@")
  local patch_path="$out_dir/universal_patch/universal_patch.npz"
  local patch_args=()
  local apply_args=()

  if [[ "$FORCE_TRAIN" != "1" && -f "$patch_path" ]]; then
    patch_args=(--patch_path "$patch_path")
    log "reuse universal patch -> $patch_path"
  else
    log "train universal patch -> $patch_path"
  fi

  if [[ "$FORCE_APPLY" != "1" && "$FORCE_TRAIN" != "1" ]]; then
    apply_args=(--skip_existing_outputs)
  fi

  (
    cd "$VGGT_ROOT"
    "$VGGT_PY" attack_vggt_vla_style.py \
      "${attack_args[@]}" \
      "${patch_args[@]}" \
      "${apply_args[@]}" \
      --output_dir "$out_dir" \
      --ckpt "$CKPT" \
      --feature_layer "$FEATURE_LAYER" \
      --iterations "$ITERATIONS" \
      --inner_loop "$INNER_LOOP" \
      --patch_lr "$PATCH_LR" \
      --patch_area_ratio "$PATCH_AREA_RATIO" \
      --scheduler "$SCHEDULER" \
      --warmup_iterations "$WARMUP_ITERATIONS" \
      --rotation_degrees "$ROTATION_DEGREES" \
      --shear "$SHEAR" \
      --geometry_prob "$GEOMETRY_PROB" \
      --seed "$SEED"
  )
}

log "check paths"
require_file "$VGGT_PY"
require_file "$RECONS_PY"
require_file "$VGGT_ROOT/attack_vggt_vla_style.py"
require_file "$VGGT_ROOT/scripts/prepare_tum10_frame_scenes_for_vla.py"
require_file "$VGGT_ROOT/scripts/eval_vggt_tum_pose_for_recons_eval_tum10.py"
require_file "$VGGT_ROOT/scripts/run_vggt_clean_tum10_uniform.py"
require_dir "$RECONS_ROOT"
require_dir "$NYU_SCENES"
require_dir "$BONN_SCENES"
require_dir "$TUM_ROOT"
[[ -e "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }

log "settings"
echo "iterations=$ITERATIONS inner_loop=$INNER_LOOP total_updates=$((ITERATIONS * INNER_LOOP))"
echo "single_frame_scenes_per_iteration=$SINGLE_FRAME_SCENES_PER_ITERATION"
echo "tum_frame_count=$TUM_FRAME_COUNT"
echo "tum_frame_s1_scenes_per_iteration=$TUM_FRAME_S1_SCENES_PER_ITERATION"
echo "tum_frame_s10_scenes_per_iteration=$TUM_FRAME_S10_SCENES_PER_ITERATION"
echo "out_base=$OUT_BASE"

log "prepare TUM images links"
for seq_dir in "$TUM_ROOT"/rgbd_dataset_freiburg3_*; do
  if [[ -d "$seq_dir/rgb_90" ]]; then
    ln -sfn rgb_90 "$seq_dir/images"
  fi
done

log "prepare TUM-10 single-frame training scenes -> $TUM10_FRAME_SCENES"
prepare_args=()
if [[ "$FORCE_PREPARE_TUM10" == "1" ]]; then
  prepare_args=(--overwrite)
fi
"$VGGT_PY" "$VGGT_ROOT/scripts/prepare_tum10_frame_scenes_for_vla.py" \
  --tum_root "$TUM_ROOT" \
  --out_root "$TUM10_FRAME_SCENES" \
  --frame_count "$TUM_FRAME_COUNT" \
  "${prepare_args[@]}"

log "run clean VGGT on TUM-10 uniform frames -> $TUM_CLEAN_OUT"
clean_args=()
if [[ "$FORCE_APPLY" != "1" && "$FORCE_TRAIN" != "1" ]]; then
  clean_args=(--skip_existing)
fi
"$VGGT_PY" "$VGGT_ROOT/scripts/run_vggt_clean_tum10_uniform.py" \
  --tum_root "$TUM_ROOT" \
  --output_root "$TUM_CLEAN_OUT" \
  --frame_manifest "$TUM10_FRAME_MANIFEST" \
  --ckpt "$CKPT" \
  --seed "$SEED" \
  "${clean_args[@]}"

log "NYU-v2 VLA-style universal feature patch, scenes_per_iteration=1"
run_vla_attack "$NYU_OUT" \
  --dataset nyu-v2 \
  --train_scenes_root "$NYU_SCENES" \
  --train_scene_pattern "nyu_*" \
  --eval_scenes_root "$NYU_SCENES" \
  --eval_scene_pattern "nyu_*" \
  --max_frames 1 \
  --patch_x "$NYU_PATCH_X" \
  --patch_y "$NYU_PATCH_Y" \
  --scenes_per_iteration "$SINGLE_FRAME_SCENES_PER_ITERATION"

log "Bonn VLA-style universal feature patch, scenes_per_iteration=1"
run_vla_attack "$BONN_OUT" \
  --dataset bonn \
  --train_scenes_root "$BONN_SCENES" \
  --train_scene_pattern "rgbd_bonn_*__*" \
  --eval_scenes_root "$BONN_SCENES" \
  --eval_scene_pattern "rgbd_bonn_*__*" \
  --max_frames 1 \
  --patch_x "$BONN_PATCH_X" \
  --patch_y "$BONN_PATCH_Y" \
  --scenes_per_iteration "$SINGLE_FRAME_SCENES_PER_ITERATION"

log "TUM-10 frame-update VLA-style patch, scenes_per_iteration=1"
run_vla_attack "$TUM_FRAME_S1_OUT" \
  --dataset tum-dynamics-10frame-frameupdate-s1 \
  --train_scenes_root "$TUM10_FRAME_SCENES" \
  --train_scene_pattern "rgbd_dataset_freiburg3_*__frame_*" \
  --eval_scenes_root "$TUM_ROOT" \
  --eval_scene_pattern "rgbd_dataset_freiburg3_*" \
  --max_frames "$TUM_FRAME_COUNT" \
  --train_max_frames 1 \
  --eval_max_frames "$TUM_FRAME_COUNT" \
  --frame_manifest "$TUM10_FRAME_MANIFEST" \
  --patch_x "$TUM_PATCH_X" \
  --patch_y "$TUM_PATCH_Y" \
  --scenes_per_iteration "$TUM_FRAME_S1_SCENES_PER_ITERATION"

log "TUM-10 frame-update VLA-style patch, scenes_per_iteration=10"
run_vla_attack "$TUM_FRAME_S10_OUT" \
  --dataset tum-dynamics-10frame-frameupdate-s10 \
  --train_scenes_root "$TUM10_FRAME_SCENES" \
  --train_scene_pattern "rgbd_dataset_freiburg3_*__frame_*" \
  --eval_scenes_root "$TUM_ROOT" \
  --eval_scene_pattern "rgbd_dataset_freiburg3_*" \
  --max_frames "$TUM_FRAME_COUNT" \
  --train_max_frames 1 \
  --eval_max_frames "$TUM_FRAME_COUNT" \
  --frame_manifest "$TUM10_FRAME_MANIFEST" \
  --patch_x "$TUM_PATCH_X" \
  --patch_y "$TUM_PATCH_Y" \
  --scenes_per_iteration "$TUM_FRAME_S10_SCENES_PER_ITERATION"

if [[ "$RUN_EVAL" != "1" ]]; then
  log "skip recons_eval because RUN_EVAL=$RUN_EVAL"
  exit 0
fi

log "convert and evaluate NYU-v2"
(
  cd "$RECONS_ROOT"
  "$RECONS_PY" scripts/prepare_nyu_v2_vggt_monodepth_for_recons_eval.py \
    --vggt_output_root "$NYU_OUT" \
    --pred_out "$RECONS_ROOT/outputs/monodepth/$NYU_MODEL/nyu-v2" \
    --scene_pattern "nyu_*" \
    --overwrite
  "$RECONS_PY" monodepth/eval.py \
    "eval_models=[$NYU_MODEL]" \
    'eval_datasets=[nyu-v2]' \
    output_dir="$RECONS_ROOT/outputs/monodepth" \
    save_suffix=nyu_v2_vla_universal_feature_s1_l3
)

log "convert and evaluate Bonn"
(
  cd "$RECONS_ROOT"
  "$RECONS_PY" scripts/prepare_bonn_monodepth_for_recons_eval.py \
    --vggt_output_root "$BONN_OUT" \
    --pred_out "$RECONS_ROOT/outputs/monodepth/$BONN_MODEL/bonn" \
    --scene_pattern "rgbd_bonn_*__*" \
    --overwrite
  "$RECONS_PY" monodepth/eval.py \
    "eval_models=[$BONN_MODEL]" \
    'eval_datasets=[bonn]' \
    output_dir="$RECONS_ROOT/outputs/monodepth" \
    save_suffix=bonn_vla_universal_feature_s1_l3
)

log "evaluate TUM-10 frame-update variants"
"$RECONS_PY" "$VGGT_ROOT/scripts/eval_vggt_tum_pose_for_recons_eval_tum10.py" \
  --vggt_output_root "$TUM_CLEAN_OUT" \
  --model_name "$TUM_CLEAN_MODEL" \
  --recons_root "$RECONS_ROOT" \
  --overwrite
"$RECONS_PY" "$VGGT_ROOT/scripts/eval_vggt_tum_pose_for_recons_eval_tum10.py" \
  --vggt_output_root "$TUM_FRAME_S1_OUT" \
  --model_name "$TUM_FRAME_S1_MODEL" \
  --recons_root "$RECONS_ROOT" \
  --overwrite
"$RECONS_PY" "$VGGT_ROOT/scripts/eval_vggt_tum_pose_for_recons_eval_tum10.py" \
  --vggt_output_root "$TUM_FRAME_S10_OUT" \
  --model_name "$TUM_FRAME_S10_MODEL" \
  --recons_root "$RECONS_ROOT" \
  --overwrite

log "all done"
echo "NYU:        $RECONS_ROOT/outputs/monodepth/nyu-v2-metric-nyu_v2_vla_universal_feature_s1_l3.csv"
echo "Bonn:       $RECONS_ROOT/outputs/monodepth/bonn-metric-bonn_vla_universal_feature_s1_l3.csv"
echo "TUM clean:  $RECONS_ROOT/outputs/relpose-distance/tum10-metric-$TUM_CLEAN_MODEL.csv"
echo "TUM s1:     $RECONS_ROOT/outputs/relpose-distance/tum10-metric-$TUM_FRAME_S1_MODEL.csv"
echo "TUM s10:    $RECONS_ROOT/outputs/relpose-distance/tum10-metric-$TUM_FRAME_S10_MODEL.csv"
