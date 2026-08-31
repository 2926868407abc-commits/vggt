#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VGGT_ROOT"

run_name="${1:-mp_full_s0}"
gpu="${2:-1}"
iterations="${3:-1000}"
rotation_weight="${4:-5.0}"
attack_loss="${5:-mirage_projected}"
direction_scale="${6:-0.0}"
seed="${7:-0}"
log="logs/${run_name}.log"
pid_file="logs/${run_name}.pid"

mkdir -p logs
if [[ -f "$pid_file" ]]; then
  old_pid="$(cat "$pid_file")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "$run_name already running as PID $old_pid" >&2
    exit 1
  fi
fi

CUDA_VISIBLE_DEVICES="$gpu" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PROJECTED_RUN_NAME="$run_name" \
PROJECTED_ITERATIONS="$iterations" \
PROJECTED_INNER_LOOP=1 \
PROJECTED_POSE_ROTATION_WEIGHT="$rotation_weight" \
PROJECTED_ATTACK_LOSS="$attack_loss" \
PROJECTED_DIRECTION_SCALE="$direction_scale" \
PROJECTED_SEED="$seed" \
setsid bash "$VGGT_ROOT/run_mirage_projected_smoke.sh" >"$log" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$pid_file"
echo "started $run_name pid=$pid gpu=$gpu iterations=$iterations rotation_weight=$rotation_weight attack_loss=$attack_loss direction_scale=$direction_scale seed=$seed log=$log"
