#!/usr/bin/env bash
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
TRAIN_SESSION="${1:?usage: $0 TRAIN_SESSION GPU SCENE RUN_NAME}"
GPU="${2:?usage: $0 TRAIN_SESSION GPU SCENE RUN_NAME}"
SCENE="${3:?usage: $0 TRAIN_SESSION GPU SCENE RUN_NAME}"
RUN_NAME="${4:?usage: $0 TRAIN_SESSION GPU SCENE RUN_NAME}"
ROOT="$V/outputs_attack_geometry_aware_tum10/$RUN_NAME"
TEXTURE="$ROOT/geometry_patch/geometry_patch_texture.npz"
LOG_DIR="$V/logs/joint_context_10_20_30/eval_queues"
LOG="$LOG_DIR/${RUN_NAME}.queue.log"

mkdir -p "$LOG_DIR"
echo "[eval-queue-armed] $(date -Is) train_session=$TRAIN_SESSION gpu=$GPU run=$RUN_NAME" >>"$LOG"

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  sleep 30
done

# The three-stage launcher exits successfully only after its final stage has
# converged.  Refuse to evaluate a failed or incomplete run.
test -f "$TEXTURE"
echo "[eval-queue-start] $(date -Is) gpu=$GPU run=$RUN_NAME" >>"$LOG"
bash "$V/scripts/runs/eval_per_scene_formal_w30_s20.sh" \
  "$GPU" "$SCENE" "$RUN_NAME"
echo "[eval-queue-done] $(date -Is) gpu=$GPU run=$RUN_NAME" >>"$LOG"
