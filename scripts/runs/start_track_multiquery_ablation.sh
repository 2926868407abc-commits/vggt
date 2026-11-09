#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
LOG_DIR="$V/logs/track_multiquery_ablation"
mkdir -p "$LOG_DIR"

MEAN_RUN=trackonly_v5_k10_firsthalf_sitting_xyz_multiquery6_mean_adaptive_s0_umax3000
ROBUST_RUN=trackonly_v5_k10_firsthalf_sitting_xyz_multiquery6_robust_adaptive_s0_umax3000

nohup bash "$V/scripts/runs/launch_track_multiquery_mean_adaptive.sh" \
  1 3000 "$MEAN_RUN" > "$LOG_DIR/$MEAN_RUN.log" 2>&1 &
echo $! > "$LOG_DIR/$MEAN_RUN.pid"

nohup bash "$V/scripts/runs/launch_track_multiquery_robust_adaptive.sh" \
  3 3000 "$ROBUST_RUN" > "$LOG_DIR/$ROBUST_RUN.log" 2>&1 &
echo $! > "$LOG_DIR/$ROBUST_RUN.pid"

printf '%s\n' \
  "mean pid=$(cat "$LOG_DIR/$MEAN_RUN.pid") log=$LOG_DIR/$MEAN_RUN.log" \
  "robust pid=$(cat "$LOG_DIR/$ROBUST_RUN.pid") log=$LOG_DIR/$ROBUST_RUN.log"
