#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VGGT_ROOT"

steps="${ABLATION_STEPS:-200}"
direction_scale="${ABLATION_DIRECTION_SCALE:-0.004719185642898083}"
seed="${ABLATION_SEED:-0}"
master_log="logs/mp_pose_weight_ablation_${steps}.status"

mkdir -p logs
: >"$master_log"

run_and_wait() {
  local run_name="$1"
  local gpu="$2"
  local rotation_weight="$3"
  local attack_loss="$4"
  local history="outputs_attack_geometry_aware_tum10/${run_name}/geometry_patch/training_history.jsonl"
  local pid_file="logs/${run_name}.pid"

  echo "$(date '+%F %T') START ${run_name} gpu=${gpu} rot=${rotation_weight} loss=${attack_loss}" >>"$master_log"
  bash "$VGGT_ROOT/launch_mirage_projected_full.sh" \
    "$run_name" "$gpu" "$steps" "$rotation_weight" "$attack_loss" \
    "$direction_scale" "$seed" >>"$master_log" 2>&1

  local pid
  pid="$(cat "$pid_file")"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
  done

  local lines=0
  if [[ -f "$history" ]]; then
    lines="$(wc -l <"$history")"
  fi
  if [[ "$lines" -ne "$steps" ]]; then
    echo "$(date '+%F %T') FAIL ${run_name} history=${lines}/${steps}" >>"$master_log"
    return 1
  fi
  echo "$(date '+%F %T') DONE ${run_name} history=${lines}/${steps}" >>"$master_log"
}

(
  run_and_wait "mp_ab_r0_s${seed}_${steps}" 0 0.0 mirage_projected
  run_and_wait "mp_ab_r10_s${seed}_${steps}" 0 10.0 mirage_projected
) &
queue0=$!

(
  run_and_wait "mp_ab_r1_s${seed}_${steps}" 1 1.0 mirage_projected
  run_and_wait "mp_ab_split_s${seed}_${steps}" 1 5.0 mirage_projected_split_pose
) &
queue1=$!

run_and_wait "mp_ab_r2_s${seed}_${steps}" 2 2.0 mirage_projected &
queue2=$!

run_and_wait "mp_ab_r5_s${seed}_${steps}" 3 5.0 mirage_projected &
queue3=$!

status=0
for queue in "$queue0" "$queue1" "$queue2" "$queue3"; do
  wait "$queue" || status=1
done

if [[ "$status" -ne 0 ]]; then
  echo "$(date '+%F %T') ABLATION_FAILED" >>"$master_log"
  exit 1
fi

echo "$(date '+%F %T') ABLATION_COMPLETE" >>"$master_log"
