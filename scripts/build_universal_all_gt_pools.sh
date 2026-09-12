#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
PY=/mnt/data/wangqq/conda_envs/recons_eval/bin/python3
TUM=/mnt/data/wangqq/recons_eval/data/tum_all_sync
POOLS="$V/data/tum_universal_all/universal_train_firsthalf_pools.json"
OUT="$V/outputs/tum_gt_pool_universal_all_firsthalf"
LOG_ROOT="$V/logs/universal_all_gt_pool"
mkdir -p "$OUT" "$LOG_ROOT"

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

running=0
for scene in "${scenes[@]}"; do
  if [[ -f "$OUT/$scene/pool_meta.json" ]]; then
    echo "skip complete pool $scene"
    continue
  fi
  frames="$($PY -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))[sys.argv[2]])))' "$POOLS" "$scene")"
  log="$LOG_ROOT/$scene.log"
  "$PY" "$V/scripts/build_tum_gt_pool.py" \
    --tum_root "$TUM" \
    --scene "$scene" \
    --frames "$frames" \
    --gt_name groundtruth.txt \
    --out_dir "$OUT" \
    --lazy_tracks >"$log" 2>&1 &
  running=$((running + 1))
  if (( running == 4 )); then
    wait
    running=0
  fi
done
wait

for scene in "${scenes[@]}"; do
  test -f "$OUT/$scene/pool_meta.json"
done
echo "built ${#scenes[@]} all-frame first-half GT pools -> $OUT"
