#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
cd "$V"
mkdir -p logs/autodl_resume_20261004

if ! tmux has-session -t autodl_resume_k20 2>/dev/null; then
  tmux new-session -d -s autodl_resume_k20 \
    'cd /mnt/data/wangqq/vggt && bash scripts/runs/launch_joint_context_scene_three_stage_autodl_resume_20261004.sh 1 20 rgbd_dataset_freiburg3_sitting_static 8000 && bash scripts/runs/eval_per_scene_formal_w30_s20.sh 1 rgbd_dataset_freiburg3_sitting_static jointctx_v5_k20_firsthalf_sitting_static_fivetarget_uncapped_adaptive_s0_umax8000'
fi

if ! tmux has-session -t autodl_resume_k30 2>/dev/null; then
  tmux new-session -d -s autodl_resume_k30 \
    'cd /mnt/data/wangqq/vggt && bash scripts/runs/resume_autodl_final_stage_20261004.sh 2 30 rgbd_dataset_freiburg3_sitting_xyz jointctx_v5_k30_firsthalf_xyz_fivetarget_uncapped_adaptive_s0_umax8000 8000'
fi

if ! tmux has-session -t autodl_resume_point_k10queue 2>/dev/null; then
  tmux new-session -d -s autodl_resume_point_k10queue \
    'cd /mnt/data/wangqq/vggt && bash scripts/runs/run_point_single_target_ablation_autodl_resume_20261004.sh 3 8000 && bash scripts/runs/resume_autodl_final_stage_20261004.sh 3 10 rgbd_dataset_freiburg3_sitting_static jointctx_v5_k10_firsthalf_sitting_static_fivetarget_uncapped_adaptive_s0_umax8000 8000'
fi

tmux ls
