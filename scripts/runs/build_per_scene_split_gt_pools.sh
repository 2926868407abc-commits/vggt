#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
PY=/mnt/data/wangqq/conda_envs/vggt/bin/python3
TUM=/mnt/data/wangqq/recons_eval/data/tum_all_sync
SPLITS_CSV="${1:-firsthalf,evenodd}"
JOBS="${GT_BUILD_JOBS:-4}"
IFS=',' read -r -a SPLITS <<< "$SPLITS_CSV"

SCENES=(
  rgbd_dataset_freiburg3_sitting_static
  rgbd_dataset_freiburg3_sitting_xyz
  rgbd_dataset_freiburg3_sitting_rpy
  rgbd_dataset_freiburg3_sitting_halfsphere
  rgbd_dataset_freiburg3_walking_static
  rgbd_dataset_freiburg3_walking_xyz
  rgbd_dataset_freiburg3_walking_rpy
  rgbd_dataset_freiburg3_walking_halfsphere
)

pool_config() {
  case "$1" in
    firsthalf)
      POOLS="$V/data/tum_per_scene_splits/firsthalf_train_pools.json"
      OUT="$V/outputs/tum_gt_pool_universal_all_firsthalf"
      ;;
    evenodd)
      POOLS="$V/data/tum_per_scene_splits/evenodd_train_pools.json"
      OUT="$V/outputs/tum_gt_pool_per_scene_even"
      ;;
    *)
      echo "unknown split: $1" >&2
      return 2
      ;;
  esac
}

build_one() {
  local split="$1"
  local scene="$2"
  local pools="$3"
  local out="$4"
  local log_root="$V/logs/per_scene_gt_pool_${split}"
  mkdir -p "$out" "$log_root"
  if [[ -f "$out/$scene/pool_meta.json" ]]; then
    echo "[skip-complete] split=$split scene=$scene"
    return 0
  fi
  echo "[build] split=$split scene=$scene"
  "$PY" "$V/scripts/build_tum_gt_pool.py" \
    --tum_root "$TUM" \
    --scene "$scene" \
    --pool_json "$pools" \
    --pool_key "$scene" \
    --gt_name groundtruth.txt \
    --out_dir "$out" \
    --lazy_tracks \
    --overwrite >"$log_root/$scene.log" 2>&1
  test -f "$out/$scene/pool_meta.json"
  echo "[done] split=$split scene=$scene"
}

export -f build_one
export V PY TUM

for split in "${SPLITS[@]}"; do
  pool_config "$split"
  test -f "$POOLS"
  export split POOLS OUT
  printf '%s\n' "${SCENES[@]}" | xargs -P "$JOBS" -I '{}' \
    bash -c 'build_one "$split" "$1" "$POOLS" "$OUT"' _ '{}'
done

for split in "${SPLITS[@]}"; do
  pool_config "$split"
  for scene in "${SCENES[@]}"; do
    test -f "$OUT/$scene/pool_meta.json"
  done
done

echo "[all-complete] splits=$SPLITS_CSV"
