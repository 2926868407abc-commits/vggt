#!/usr/bin/env bash
set -Eeuo pipefail

V="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
cd "$V"

LOG_DIR="$V/logs/missing_context_ablation_20261009"
mkdir -p "$LOG_DIR"

launch_one() {
  local session="$1"
  local gpu="$2"
  local context="$3"
  local scene="$4"
  local run="$5"
  local log="$LOG_DIR/${session}.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip-existing-session] $session"
    return 0
  fi

  tmux new-session -d -s "$session" \
    "cd '$V' && \
     echo '[orchestration-start] '$(date -Is) gpu=$gpu context=$context scene=$scene >>'$log' && \
     bash scripts/runs/launch_joint_context_scene_three_stage_autodl_resume_20261004.sh \
       '$gpu' '$context' '$scene' 8000 >>'$log' 2>&1 && \
     bash scripts/runs/eval_per_scene_formal_w30_s20.sh \
       '$gpu' '$scene' '$run' >>'$log' 2>&1 && \
     echo '[orchestration-complete] '$(date -Is) run=$run >>'$log'"

  echo "[launched] session=$session gpu=$gpu context=$context scene=$scene run=$run"
}

# Complete the controlled same-sequence context-length ablations:
# sitting_static already has K=10 (running) and K=20 (complete), so add K=30.
launch_one \
  missing_static_k30 \
  0 \
  30 \
  rgbd_dataset_freiburg3_sitting_static \
  jointctx_v5_k30_firsthalf_sitting_static_fivetarget_uncapped_adaptive_s0_umax8000

# sitting_xyz already has the current K=30 run. Add the missing K=20 run.
launch_one \
  missing_xyz_k20 \
  1 \
  20 \
  rgbd_dataset_freiburg3_sitting_xyz \
  jointctx_v5_k20_firsthalf_sitting_xyz_fivetarget_uncapped_adaptive_s0_umax8000

tmux ls
