#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VGGT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

out_dir="outputs/diag_multitask_gradients"
mkdir -p "$out_dir"

VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"
"$VGGT_PY" \
  scripts/diag_multitask_gradients.py "$@" \
  2>&1 | tee "$out_dir/run.log"
