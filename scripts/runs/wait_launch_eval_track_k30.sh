#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
GPU="${1:-3}"
SCENE=rgbd_dataset_freiburg3_sitting_xyz
SMOKE_RUN=trackonly_v5_k30_firsthalf_sitting_xyz_selectiveheads_warm_smoke_s0_u2
DIAG_RUN=trackonly_v5_k30_firsthalf_sitting_xyz_multiquery2_warm_adaptive_s0_umax500
LOG_DIR="$V/logs/track_k30_warm_diagnostic"
mkdir -p "$LOG_DIR"

gpu_busy() {
  nvidia-smi -i "$GPU" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'
}

echo "[wait-start] $(date -Is) gpu=$GPU" | tee "$LOG_DIR/status.log"
while gpu_busy; do
  echo "[wait-busy] $(date -Is) gpu=$GPU" >> "$LOG_DIR/status.log"
  sleep 60
done

echo "[smoke-start] $(date -Is) gpu=$GPU" | tee -a "$LOG_DIR/status.log"
ADAPTIVE_AUDIT_EVERY=1 ADAPTIVE_CHECKPOINT_EVERY=1 \
  bash "$V/scripts/runs/launch_track_k30_warm_adaptive.sh" \
    "$GPU" 2 1 "$SMOKE_RUN" \
    >"$LOG_DIR/smoke.log" 2>&1
test -f "$V/outputs_attack_geometry_aware_tum10/$SMOKE_RUN/geometry_patch/geometry_patch_texture.npz"
echo "[smoke-done] $(date -Is)" | tee -a "$LOG_DIR/status.log"

echo "[diagnostic-start] $(date -Is) sets_per_update=2 max_updates=500" \
  | tee -a "$LOG_DIR/status.log"
bash "$V/scripts/runs/launch_track_k30_warm_adaptive.sh" \
  "$GPU" 500 2 "$DIAG_RUN" \
  >"$LOG_DIR/diagnostic.log" 2>&1
test -f "$V/outputs_attack_geometry_aware_tum10/$DIAG_RUN/geometry_patch/geometry_patch_texture.npz"
echo "[diagnostic-done] $(date -Is)" | tee -a "$LOG_DIR/status.log"

echo "[heldout-eval-start] $(date -Is) protocol=w30_s20" \
  | tee -a "$LOG_DIR/status.log"
bash "$V/scripts/runs/run_controlled_baseline_eval_w30_s20.sh" \
  "$GPU" "$SCENE" "$DIAG_RUN"
echo "[heldout-eval-done] $(date -Is)" | tee -a "$LOG_DIR/status.log"
