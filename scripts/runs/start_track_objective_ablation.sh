#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
LOG_DIR="$V/logs/track_objective_ablation"
mkdir -p "$LOG_DIR"

nohup bash "$V/scripts/runs/launch_track_objective_ablation.sh" \
  capped_all 1 300 >"$LOG_DIR/capped_all_u300.log" 2>&1 &
echo $! >"$LOG_DIR/capped_all_u300.pid"

nohup bash "$V/scripts/runs/launch_track_objective_ablation.sh" \
  eval_epe 3 300 >"$LOG_DIR/eval_epe_u300.log" 2>&1 &
echo $! >"$LOG_DIR/eval_epe_u300.pid"

echo "capped_all pid=$(cat "$LOG_DIR/capped_all_u300.pid") gpu=1"
echo "eval_epe   pid=$(cat "$LOG_DIR/eval_epe_u300.pid") gpu=3"
