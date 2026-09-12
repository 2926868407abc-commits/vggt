#!/usr/bin/env bash
# Parameterised entry point for the auto-placement prototype.
#
# PHASE=planes : extract candidate planes + contact sheet + position grid
# PHASE=scan   : run frozen-texture VGGT inference at every valid grid position
# PHASE=score  : compute ATE/RPE and render both heat maps
# PHASE=probe  : select spread positions for short/long correlation runs
# PHASE=all    : run all four phases (includes the expensive GPU scan)
set -Eeuo pipefail

VGGT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$VGGT_ROOT"

PHASE="${PHASE:-planes}"
SCENE="${SCENE:-rgbd_dataset_freiburg3_sitting_halfsphere}"
AREA="${AREA:-0.2}"
ASPECT="${ASPECT:-1.5534}"
GRID="${GRID:-8}"
CELL="${CELL:-0.025}"
MIN_FILL="${MIN_FILL:-0.95}"
MIN_CANDIDATE_VISIBLE_FRAMES="${MIN_CANDIDATE_VISIBLE_FRAMES:-6}"
MIN_CANDIDATE_SUPPORTED_FRAMES="${MIN_CANDIDATE_SUPPORTED_FRAMES:-6}"
PLANES_DIR="${PLANES_DIR:-$VGGT_ROOT/outputs/candidate_planes}"
GT_DIR="${GT_DIR:-$VGGT_ROOT/outputs/tum_gt_point_track}"
CLEAN_ROOT="${CLEAN_ROOT:-$VGGT_ROOT/outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3}"
TUM_ROOT="${TUM_ROOT:-/mnt/data/wangqq/recons_eval/data/tum}"
RECONS_ROOT="${RECONS_ROOT:-/mnt/data/wangqq/recons_eval}"
VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"
RECONS_PY="${RECONS_PY:-/mnt/data/wangqq/conda_envs/recons_eval/bin/python3}"
TEXTURE="${TEXTURE:-}"
SEED_SD="${SEED_SD:-}"

mkdir -p "$PLANES_DIR"

run_planes() {
  "$VGGT_PY" -B scripts/extract_candidate_planes.py \
    --scene "$SCENE" --tum_root "$TUM_ROOT" --gt_dir "$GT_DIR" \
    --clean_root "$CLEAN_ROOT" --out_dir "$PLANES_DIR"
  "$VGGT_PY" -B scripts/viz_candidate_planes.py \
    --scene "$SCENE" --tum_root "$TUM_ROOT" --gt_dir "$GT_DIR" \
    --clean_root "$CLEAN_ROOT" --planes_dir "$PLANES_DIR"
  "$VGGT_PY" -B scripts/gen_scan_candidates.py \
    --scene "$SCENE" --area "$AREA" --aspect "$ASPECT" --grid "$GRID" \
    --cell "$CELL" --min_fill "$MIN_FILL" \
    --planes_dir "$PLANES_DIR" --gt_dir "$GT_DIR"
  "$VGGT_PY" -B scripts/filter_scan_candidates.py \
    --scene "$SCENE" --area "$AREA" --tum_root "$TUM_ROOT" \
    --planes_dir "$PLANES_DIR" --clean_root "$CLEAN_ROOT" \
    --min_visible_frames "$MIN_CANDIDATE_VISIBLE_FRAMES" \
    --min_supported_frames "$MIN_CANDIDATE_SUPPORTED_FRAMES"
}

run_scan() {
  local texture_args=()
  [[ -n "$TEXTURE" ]] && texture_args=(--texture "$TEXTURE")
  "$VGGT_PY" -B scripts/run_position_scan.py \
    --scene "$SCENE" --area "$AREA" --tum_root "$TUM_ROOT" \
    --planes_dir "$PLANES_DIR" --clean_root "$CLEAN_ROOT" \
    --spec "$PLANES_DIR/${SCENE}_validated_a${AREA}.json" \
    "${texture_args[@]}"
}

run_score() {
  local seed_args=()
  [[ -n "$SEED_SD" ]] && seed_args=(--seed-sd "$SEED_SD")
  "$RECONS_PY" -B scripts/score_scan.py \
    --scene "$SCENE" --area "$AREA" --planes-dir "$PLANES_DIR" \
    --runs-root "$VGGT_ROOT/outputs_attack_geometry_aware_tum10" \
    --recons-root "$RECONS_ROOT" --tum-root "$TUM_ROOT" \
    "${seed_args[@]}"
}

run_probe() {
  "$VGGT_PY" -B scripts/pick_probe_positions.py \
    --scene "$SCENE" --area "$AREA" --planes-dir "$PLANES_DIR" \
    --input "$PLANES_DIR/${SCENE}_validated_a${AREA}.json"
}

case "$PHASE" in
  planes) run_planes ;;
  scan) run_scan ;;
  score) run_score ;;
  probe) run_probe ;;
  all) run_planes; run_scan; run_score; run_probe ;;
  *) echo "Unknown PHASE=$PHASE (use planes, scan, score, probe, or all)" >&2; exit 2 ;;
esac
