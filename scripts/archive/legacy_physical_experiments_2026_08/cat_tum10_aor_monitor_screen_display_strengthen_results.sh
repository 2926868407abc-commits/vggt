#!/usr/bin/env bash
set -Eeuo pipefail

RECONS_ROOT="${RECONS_ROOT:-/mnt/data/wangqq/recons_eval}"
OUT="$RECONS_ROOT/outputs/relpose-distance"

cat "$OUT/tum10-metric-vggt_tum10_sitting_static_clean_uniform_l3.csv"

for MODEL in \
  vggt_tum10_sitting_static_aor_monitor_screen_display_refined_base_1000 \
  vggt_tum10_sitting_static_aor_monitor_screen_display_refined_ref002_1000 \
  vggt_tum10_sitting_static_aor_monitor_screen_display_refined_lr003_1000 \
  vggt_tum10_sitting_static_aor_monitor_screen_display_refined_2000_2000 \
  vggt_tum10_sitting_static_aor_monitor_screen_display_refined_noeot_1000
do
  echo
  echo "== $MODEL =="
  cat "$OUT/tum10-metric-${MODEL}.csv"
done

