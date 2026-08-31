#!/usr/bin/env bash
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
RECONS_ROOT="${RECONS_ROOT:-/mnt/data/wangqq/recons_eval}"
VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"

"$VGGT_PY" "$VGGT_ROOT/scripts/detect_tum_monitor_quad.py" \
  --tum_root "$RECONS_ROOT/data/tum" \
  --manifest "$VGGT_ROOT/data/tum_dynamics_10frame_individual_scenes/tum10_frame_manifest.json" \
  --scene "${SCENE:-rgbd_dataset_freiburg3_sitting_static}" \
  --frame_rank "${FRAME_RANK:-0}" \
  --out_dir "${OUT_DIR:-$VGGT_ROOT/outputs_attack_geometry_aware_tum10/monitor_quad_detection}" \
  --roi "${ROI:-0.38,0.26,0.78,0.68}" \
  --prefer "${PREFER:-0.58,0.45}" \
  --dark_threshold "${DARK_THRESHOLD:-72}" \
  --shrink "${SHRINK:-0.88}"
