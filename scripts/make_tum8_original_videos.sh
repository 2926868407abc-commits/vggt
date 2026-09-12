#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/data/wangqq/recons_eval/data/tum_all_sync
OUT=/mnt/data/wangqq/vggt/outputs/tum_universal_all_original_videos

mkdir -p "$OUT"

scenes=(
  rgbd_dataset_freiburg3_sitting_static
  rgbd_dataset_freiburg3_sitting_xyz
  rgbd_dataset_freiburg3_sitting_rpy
  rgbd_dataset_freiburg3_sitting_halfsphere
  rgbd_dataset_freiburg3_walking_static
  rgbd_dataset_freiburg3_walking_xyz
  rgbd_dataset_freiburg3_walking_rpy
  rgbd_dataset_freiburg3_walking_halfsphere
)

for scene in "${scenes[@]}"; do
  input="$ROOT/$scene/images/*.png"
  output="$OUT/$scene.mp4"
  echo "[video] $scene -> $output"
  ffmpeg -hide_banner -loglevel error -y \
    -framerate 30 -pattern_type glob -i "$input" \
    -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' \
    -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
    -movflags +faststart "$output"
done

echo "[done] $OUT"
