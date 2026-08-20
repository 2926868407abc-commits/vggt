#!/usr/bin/env bash
# Recalibrate the regulariser weights for the monitor placement.
#
# The published weights were measured on the automatic fused_depth_surface patch
# (0.36x0.24 m, 1.2% of frame). The monitor patch is 0.63x0.41 m and ~3% of frame,
# so |g_attack| is a different size and the old weights no longer equalise the
# |g_attack|/|g_reg| ratio across losses.
#
# One difference from the earlier calibration on purpose: --physical_eot 0. The old
# run measured gradients with EOT on while the ablation trained with it off, so the
# weights it produced described a configuration nobody ran.
set -Eeuo pipefail
cd /mnt/data/wangqq/vggt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/mnt/data/wangqq/conda_envs/vggt/bin/python3

QUAD_COMMON="--manual_quad_coordinates normalized --manual_quad_depth_sample_stride 1 \
--manual_quad_fit_shrink 0.75 --manual_quad_plane_inlier_tolerance 0.06 \
--manual_quad_min_inlier_ratio 0.60 --surface_support_rel_tolerance 0.01 \
--fused_max_plane_residual 0.02 --surface_min_visible_frames 4 \
--surface_min_visibility_ratio 0.30 --surface_coverage_min 0.002 --surface_coverage_max 0.08"

run() {
  local tag="$1" scene="$2" quad="$3" gpu="$4"
  echo "[calib] $tag on gpu $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" $PY scripts/calibrate_attack_reg_balance.py \
    --scene "$scene" --physical_eot 0 \
    --plane_mode depth_manual_quad_surface \
    --extra "--plane_mode depth_manual_quad_surface --manual_quad_xy $quad $QUAD_COMMON" \
    > "/tmp/calib_$tag.log" 2>&1 &
}

run mon_xyz  rgbd_dataset_freiburg3_sitting_xyz \
    0.5330,0.4520,0.7380,0.4670,0.7320,0.6340,0.5450,0.6220 0
run mon_half rgbd_dataset_freiburg3_sitting_halfsphere \
    0.3603,0.2939,0.5682,0.3063,0.5644,0.4943,0.3603,0.4795 1
run post_half rgbd_dataset_freiburg3_sitting_halfsphere \
    0.2220,0.1680,0.3450,0.1720,0.3430,0.4870,0.2240,0.4830 2
wait
echo CALIB_MONITOR_DONE
for t in mon_xyz mon_half post_half; do
  echo; echo "================ $t"
  sed -n '/patch coverage/,$p' "/tmp/calib_$t.log" | grep -v "^ *\[mem\]"
done
