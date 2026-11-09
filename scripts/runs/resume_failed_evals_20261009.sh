#!/usr/bin/env bash
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
MODE="${1:?usage: $0 k30-k20|point-k10}"
EVAL="$V/scripts/runs/eval_per_scene_formal_w30_s20.sh"
RESUME="$V/scripts/runs/resume_autodl_final_stage_20261004.sh"
XYZ=rgbd_dataset_freiburg3_sitting_xyz
STATIC=rgbd_dataset_freiburg3_sitting_static
K30=jointctx_v5_k30_firsthalf_xyz_fivetarget_uncapped_adaptive_s0_umax8000
K20=jointctx_v5_k20_firsthalf_sitting_static_fivetarget_uncapped_adaptive_s0_umax8000
POINT=pointab_v5_k10_firsthalf_xyz_pointonly_cold_adaptive_s0_umax8000
K10=jointctx_v5_k10_firsthalf_sitting_static_fivetarget_uncapped_adaptive_s0_umax8000

case "$MODE" in
  k30-k20)
    bash "$EVAL" 1 "$XYZ" "$K30"
    bash "$EVAL" 1 "$STATIC" "$K20"
    ;;
  point-k10)
    bash "$EVAL" 3 "$XYZ" "$POINT"
    bash "$RESUME" 3 10 "$STATIC" "$K10" 8000
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
