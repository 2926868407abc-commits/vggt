#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
TRAIN_SESSION="${1:-perseq_split_u1000}"
STEPS="${2:-1000}"
REPAIR_SESSION="${3:-perseq_split_gpu3_repair}"

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null \
   || tmux has-session -t "$REPAIR_SESSION" 2>/dev/null; do
  sleep 60
done

# Re-run the complete matrix once with the repaired loss.  Completed textures are
# skipped, so this only catches workers that may have failed on a track-empty
# random batch before they loaded the repair.
REPAIR_LOG="$V/logs/per_scene_split_matrix_u${STEPS}.repair_all.log"
bash "$V/scripts/runs/run_per_scene_split_matrix.sh" "$STEPS" \
  >"$REPAIR_LOG" 2>&1
grep -q "all 16 per-scene split runs completed" "$REPAIR_LOG"

exec bash "$V/scripts/runs/run_per_scene_split_eval_matrix.sh" "$STEPS"
