#!/usr/bin/env bash
# Start one formal profile only after a preceding run has completed evaluation,
# figures and its atomic Markdown record.  Safe for one queue per GPU.
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 WAIT_RUN NEXT_PROFILE GPU" >&2
  exit 2
fi

WAIT_RUN="$1"
NEXT_PROFILE="$2"
GPU="$3"
VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
REPORT="${REPORT:-$VGGT_ROOT/reports/baseline_experiments/baseline_experiment_record.md}"
MARKER="<!-- BASELINE-RUN:${WAIT_RUN}:END -->"

while ! grep -Fq "$MARKER" "$REPORT" 2>/dev/null; do
  sleep 60
done

cd "$VGGT_ROOT"
exec env PROFILE="$NEXT_PROFILE" STAGE=formal STEPS=1000 GPU="$GPU" \
  bash scripts/runs/paper_baselines.sh
