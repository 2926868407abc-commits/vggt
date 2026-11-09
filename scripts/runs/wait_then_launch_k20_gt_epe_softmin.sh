#!/usr/bin/env bash
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
GPU="${1:?usage: $0 GPU BLOCKING_PID [MAX_STEPS]}"
BLOCKING_PID="${2:?usage: $0 GPU BLOCKING_PID [MAX_STEPS]}"
MAX_STEPS="${3:-8000}"
QUEUE_LOG="$V/logs/joint_context_10_20_30/k20_per_scene/sitting_static.gt_epe_softmin.queue.log"

mkdir -p "$(dirname "$QUEUE_LOG")"
echo "[queue-start] $(date -Is) gpu=$GPU blocking_pid=$BLOCKING_PID" >>"$QUEUE_LOG"

while kill -0 "$BLOCKING_PID" 2>/dev/null; do
  sleep 60
done

# A training/evaluation watcher may claim the same GPU immediately after the
# outer three-stage process exits.  Do not race it: launch only when the card
# has no compute process at all.
while nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | grep -q '[0-9]'; do
  sleep 60
done

echo "[queue-launch] $(date -Is) gpu=$GPU" >>"$QUEUE_LOG"
exec "$V/scripts/runs/launch_k20_gt_epe_softmin_static.sh" "$GPU" "$MAX_STEPS"
