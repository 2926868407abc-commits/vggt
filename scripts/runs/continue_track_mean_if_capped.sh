#!/usr/bin/env bash
set -Eeuo pipefail

V="/mnt/data/wangqq/vggt"
OLD_PID="${1:?usage: $0 OLD_PID GPU NEW_MAX RUN_NAME OLD_LOG}"
GPU="${2:?usage: $0 OLD_PID GPU NEW_MAX RUN_NAME OLD_LOG}"
NEW_MAX="${3:?usage: $0 OLD_PID GPU NEW_MAX RUN_NAME OLD_LOG}"
RUN_NAME="${4:?usage: $0 OLD_PID GPU NEW_MAX RUN_NAME OLD_LOG}"
OLD_LOG="${5:?usage: $0 OLD_PID GPU NEW_MAX RUN_NAME OLD_LOG}"

while kill -0 "$OLD_PID" 2>/dev/null; do
  sleep 30
done

if grep -Fq '[adaptive] converged' "$OLD_LOG"; then
  echo "[continuation] original run converged before its safety ceiling"
  exit 0
fi

echo "[continuation] original run hit its safety ceiling; resuming to max=$NEW_MAX"
exec "$V/scripts/runs/launch_track_multiquery_mean_adaptive.sh" \
  "$GPU" "$NEW_MAX" "$RUN_NAME"
