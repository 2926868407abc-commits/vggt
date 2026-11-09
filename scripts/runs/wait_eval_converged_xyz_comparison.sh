#!/usr/bin/env bash
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
SCENE=rgbd_dataset_freiburg3_sitting_xyz
MAX_STEPS="${1:-6000}"
MIRAGE_RUN="formal_xyz_v3_mirage_joint_warm_adaptive_s0_umax${MAX_STEPS}"
SPOO_RUN="formal_xyz_v3_spoo_stage2_adaptive_s17_umax${MAX_STEPS}"
LOG="$V/logs/formal_xyz_comparison/eval_w30_s20.log"
TRAIN_SESSION="${TRAIN_SESSION:-formal_xyz_cmp_v3}"

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  sleep 60
done

MIRAGE_TEXTURE="$V/outputs_attack_geometry_aware_tum10/$MIRAGE_RUN/geometry_patch/geometry_patch_texture.npz"
SPOO_TEXTURE="$V/outputs_attack_geometry_aware_tum10/$SPOO_RUN/geometry_patch/geometry_patch_texture.npz"
test -f "$MIRAGE_TEXTURE"
test -f "$SPOO_TEXTURE"

# A fixed step cap is not convergence.  Refuse to evaluate or report unless
# all three stages ended through the adaptive early-stop condition.
"$V/../conda_envs/vggt/bin/python3" \
  "$V/scripts/summarize_formal_xyz_comparison.py" check \
  --root "$V" --max_steps "$MAX_STEPS"

{
  echo "[comparison-eval-start] $(date -Is) protocol=w30_s20"
  "$V/scripts/runs/run_controlled_baseline_eval.sh" 0 "$SCENE" "$MIRAGE_RUN" &
  MIRAGE_PID=$!
  "$V/scripts/runs/run_controlled_baseline_eval.sh" 1 "$SCENE" "$SPOO_RUN" &
  SPOO_PID=$!
  wait "$MIRAGE_PID"
  wait "$SPOO_PID"
  "$V/../conda_envs/vggt/bin/python3" \
    "$V/scripts/summarize_formal_xyz_comparison.py" report \
    --root "$V" --max_steps "$MAX_STEPS"
  echo "[comparison-eval-done] $(date -Is) protocol=w30_s20"
} >"$LOG" 2>&1
