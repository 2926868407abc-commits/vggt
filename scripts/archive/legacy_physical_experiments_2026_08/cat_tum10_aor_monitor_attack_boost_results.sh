#!/usr/bin/env bash
set -Eeuo pipefail

RECONS_ROOT="${RECONS_ROOT:-/mnt/data/wangqq/recons_eval}"
OUT="$RECONS_ROOT/outputs/relpose-distance"

cat "$OUT/tum10-metric-vggt_tum10_sitting_static_clean_uniform_l3.csv"

for MODEL in \
  vggt_tum10_sitting_static_aor_monitor_screen_display_boost_oldquad_lr003_ref002_2000 \
  vggt_tum10_sitting_static_aor_monitor_screen_display_boost_oldquad_3000_ref002_3000 \
  vggt_tum10_sitting_static_aor_monitor_screen_display_boost_refined_weakeot_lr003_2000 \
  vggt_tum10_sitting_static_aor_monitor_screen_display_boost_oldquad_trans3_lr003_2000
do
  echo
  echo "== $MODEL =="
  cat "$OUT/tum10-metric-${MODEL}.csv"
done

