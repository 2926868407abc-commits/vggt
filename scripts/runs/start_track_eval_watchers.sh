#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
LOG_DIR="$V/logs/track_objective_ablation"

nohup bash "$V/scripts/runs/wait_then_eval_track.sh" \
  "$LOG_DIR/capped_all_u300.pid" 1 \
  trackonly_v5_k10_firsthalf_sitting_xyz_capped_all_s0_u300 \
  >"$LOG_DIR/capped_all_u300_eval_watch.log" 2>&1 &
echo $! >"$LOG_DIR/capped_all_u300_eval_watch.pid"

nohup bash "$V/scripts/runs/wait_then_eval_track.sh" \
  "$LOG_DIR/eval_epe_u300.pid" 3 \
  trackonly_v5_k10_firsthalf_sitting_xyz_eval_epe_s0_u300 \
  >"$LOG_DIR/eval_epe_u300_eval_watch.log" 2>&1 &
echo $! >"$LOG_DIR/eval_epe_u300_eval_watch.pid"

echo "evaluation watchers armed"
