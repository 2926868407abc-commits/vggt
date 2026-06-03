#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
RECONS_ROOT="${RECONS_ROOT:-/mnt/data/wangqq/recons_eval}"
VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"
RECONS_PY="${RECONS_PY:-/mnt/data/wangqq/conda_envs/recons_eval/bin/python3}"
CKPT="${CKPT:-$VGGT_ROOT/checkpoints/VGGT-1B}"

# VLA-Attacker paper-aligned patch-training defaults.
ITERATIONS="${ITERATIONS:-2000}"
INNER_LOOP="${INNER_LOOP:-50}"
PATCH_LR="${PATCH_LR:-0.001}"
PATCH_AREA_RATIO="${PATCH_AREA_RATIO:-0.05}"

# The paper specifies random shear/rotation but not their exact ranges.
# These values follow the authors' official implementation.
ROTATION_DEGREES="${ROTATION_DEGREES:-30}"
SHEAR="${SHEAR:-0.2}"
GEOMETRY_PROB="${GEOMETRY_PROB:-0.8}"
FEATURE_LAYER="${FEATURE_LAYER:-aggregator_final}"

# Training positions are randomized. VLA-Attacker selects a paste location for
# each evaluation suite; VGGT has no matching task-specific location, so the
# default evaluation adaptation uses a configurable center.
PATCH_X="${PATCH_X:--1}"
PATCH_Y="${PATCH_Y:--1}"
NYU_PATCH_X="${NYU_PATCH_X:-$PATCH_X}"
NYU_PATCH_Y="${NYU_PATCH_Y:-$PATCH_Y}"
BONN_PATCH_X="${BONN_PATCH_X:-$PATCH_X}"
BONN_PATCH_Y="${BONN_PATCH_Y:-$PATCH_Y}"
TUM_PATCH_X="${TUM_PATCH_X:-$PATCH_X}"
TUM_PATCH_Y="${TUM_PATCH_Y:-$PATCH_Y}"
NRGBD_PATCH_X="${NRGBD_PATCH_X:-$PATCH_X}"
NRGBD_PATCH_Y="${NRGBD_PATCH_Y:-$PATCH_Y}"
SEED="${SEED:-0}"
SINGLE_FRAME_SCENES_PER_ITERATION="${SINGLE_FRAME_SCENES_PER_ITERATION:-8}"
MULTI_FRAME_SCENES_PER_ITERATION="${MULTI_FRAME_SCENES_PER_ITERATION:-1}"

FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_APPLY="${FORCE_APPLY:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

NYU_SCENES="${NYU_SCENES:-$VGGT_ROOT/data/nyu_v2_recons_eval_scenes}"
BONN_SCENES="${BONN_SCENES:-$VGGT_ROOT/data/bonn_monodepth_scenes}"
TUM_ROOT="${TUM_ROOT:-$RECONS_ROOT/data/tum}"
NRGBD_SCENES="${NRGBD_SCENES:-$VGGT_ROOT/data/nrgbd_sparse_mv_recon_scenes}"

OUT_BASE="${OUT_BASE:-$VGGT_ROOT/outputs_attack_vla_style}"
NYU_OUT="$OUT_BASE/nyu_v2_vla_universal_feature_l3"
BONN_OUT="$OUT_BASE/bonn_vla_universal_feature_l3"
TUM_OUT="$OUT_BASE/tum_vla_universal_feature_l3"
NRGBD_OUT="$OUT_BASE/nrgbd_sparse_vla_universal_feature_l3"

NYU_MODEL="vggt_nyu_v2_vla_universal_feature_l3"
BONN_MODEL="vggt_bonn_vla_universal_feature_l3"
TUM_MODEL="vggt_tum_vla_universal_feature_l3"
NRGBD_MODEL="vggt_nrgbd_sparse_vla_universal_feature_l3"

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
require_dir "$RECONS_ROOT"
require_dir "$NYU_SCENES"
require_dir "$BONN_SCENES"
require_dir "$TUM_ROOT"
require_dir "$NRGBD_SCENES"
[[ -e "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }

log "VLA-style universal patch settings"
echo "iterations=$ITERATIONS inner_loop=$INNER_LOOP total_updates=$((ITERATIONS * INNER_LOOP))"
echo "patch_lr=$PATCH_LR patch_area_ratio=$PATCH_AREA_RATIO feature_layer=$FEATURE_LAYER"
echo "rotation_degrees=$ROTATION_DEGREES shear=$SHEAR geometry_prob=$GEOMETRY_PROB"
echo "training_position=random"
echo "evaluation_position_nyu=($NYU_PATCH_X,$NYU_PATCH_Y), where -1 centers the patch"
echo "evaluation_position_bonn=($BONN_PATCH_X,$BONN_PATCH_Y)"
echo "evaluation_position_tum=($TUM_PATCH_X,$TUM_PATCH_Y)"
echo "evaluation_position_nrgbd=($NRGBD_PATCH_X,$NRGBD_PATCH_Y)"
echo "single_frame_scenes_per_iteration=$SINGLE_FRAME_SCENES_PER_ITERATION"
echo "multi_frame_scenes_per_iteration=$MULTI_FRAME_SCENES_PER_ITERATION"
echo "single_frame_scene_backprops=$((ITERATIONS * INNER_LOOP * SINGLE_FRAME_SCENES_PER_ITERATION))"
echo "multi_frame_scene_backprops=$((ITERATIONS * INNER_LOOP * MULTI_FRAME_SCENES_PER_ITERATION))"

log "prepare TUM images links"
for seq_dir in "$TUM_ROOT"/rgbd_dataset_freiburg3_*; do
  if [[ -d "$seq_dir/rgb_90" ]]; then
    ln -sfn rgb_90 "$seq_dir/images"
  fi
done

log "NYU-v2 VLA-style universal feature patch"
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

log "Bonn VLA-style universal feature patch"
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

log "TUM-dynamics VLA-style universal feature patch"
run_vla_attack "$TUM_OUT" \
  --dataset tum-dynamics \
  --train_scenes_root "$TUM_ROOT" \
  --train_scene_pattern "rgbd_dataset_freiburg3_*" \
  --eval_scenes_root "$TUM_ROOT" \
  --eval_scene_pattern "rgbd_dataset_freiburg3_*" \
  --max_frames 90 \
  --patch_x "$TUM_PATCH_X" \
  --patch_y "$TUM_PATCH_Y" \
  --scenes_per_iteration "$MULTI_FRAME_SCENES_PER_ITERATION" \
  --activation_checkpoint

log "Neural-RGBD sparse VLA-style universal feature patch"
run_vla_attack "$NRGBD_OUT" \
  --dataset nrgbd-sparse \
  --train_scenes_root "$NRGBD_SCENES" \
  --train_scene_pattern "*" \
  --eval_scenes_root "$NRGBD_SCENES" \
  --eval_scene_pattern "*" \
  --max_frames 999 \
  --patch_x "$NRGBD_PATCH_X" \
  --patch_y "$NRGBD_PATCH_Y" \
  --scenes_per_iteration "$MULTI_FRAME_SCENES_PER_ITERATION"

if [[ "$RUN_EVAL" != "1" ]]; then
  log "skip recons_eval because RUN_EVAL=$RUN_EVAL"
  exit 0
fi

log "convert NYU-v2 depth predictions"
(
  cd "$RECONS_ROOT"
  "$RECONS_PY" scripts/prepare_nyu_v2_vggt_monodepth_for_recons_eval.py \
    --vggt_output_root "$NYU_OUT" \
    --pred_out "$RECONS_ROOT/outputs/monodepth/$NYU_MODEL/nyu-v2" \
    --scene_pattern "nyu_*" \
    --overwrite
)

log "evaluate NYU-v2 monodepth"
(
  cd "$RECONS_ROOT"
  "$RECONS_PY" monodepth/eval.py \
    "eval_models=[$NYU_MODEL]" \
    'eval_datasets=[nyu-v2]' \
    output_dir="$RECONS_ROOT/outputs/monodepth" \
    save_suffix=nyu_v2_vla_universal_feature_l3
)

log "convert Bonn depth predictions"
(
  cd "$RECONS_ROOT"
  "$RECONS_PY" scripts/prepare_bonn_monodepth_for_recons_eval.py \
    --vggt_output_root "$BONN_OUT" \
    --pred_out "$RECONS_ROOT/outputs/monodepth/$BONN_MODEL/bonn" \
    --scene_pattern "rgbd_bonn_*__*" \
    --overwrite
)

log "evaluate Bonn monodepth"
(
  cd "$RECONS_ROOT"
  "$RECONS_PY" monodepth/eval.py \
    "eval_models=[$BONN_MODEL]" \
    'eval_datasets=[bonn]' \
    output_dir="$RECONS_ROOT/outputs/monodepth" \
    save_suffix=bonn_vla_universal_feature_l3
)

log "evaluate TUM camera pose"
(
  cd "$RECONS_ROOT"
  "$RECONS_PY" scripts/eval_vggt_tum_pose_for_recons_eval.py \
    --vggt_output_root "$TUM_OUT" \
    --model_name "$TUM_MODEL" \
    --overwrite
)

log "evaluate Neural-RGBD sparse point map"
(
  cd "$RECONS_ROOT"
  "$RECONS_PY" scripts/eval_vggt_nrgbd_mv_recon_for_recons_eval.py \
    --vggt_output_root "$NRGBD_OUT" \
    --dataset_name NRGBD-sparse \
    --model_name "$NRGBD_MODEL" \
    --pred_key depth_unproject \
    --overwrite \
    --no_save_ply
)

log "all done"
echo "NYU:   $RECONS_ROOT/outputs/monodepth/nyu-v2-metric-nyu_v2_vla_universal_feature_l3.csv"
echo "Bonn:  $RECONS_ROOT/outputs/monodepth/bonn-metric-bonn_vla_universal_feature_l3.csv"
echo "TUM:   $RECONS_ROOT/outputs/relpose-distance/tum-metric-$TUM_MODEL.csv"
echo "NRGBD: $RECONS_ROOT/outputs/mv_recon/NRGBD-sparse-metric-$NRGBD_MODEL.csv"
