#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
GPU_INDEX="${GPU_INDEX:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

while true; do
  free_mib="$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  echo "[$(date '+%F %T')] GPU $GPU_INDEX free=${free_mib} MiB; need ${MIN_FREE_MIB} MiB"
  if (( free_mib >= MIN_FREE_MIB )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

cd "$VGGT_ROOT"
exec env \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  PYTHONUNBUFFERED=1 \
  ACTIVATION_CHECKPOINT=0 \
  bash "$VGGT_ROOT/run_geometry_aware_nrgbd_hazard.sh"
