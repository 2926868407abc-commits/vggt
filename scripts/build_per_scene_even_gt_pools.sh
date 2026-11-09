#!/usr/bin/env bash
set -Eeuo pipefail

V=/mnt/data/wangqq/vggt
PY="${PY:-/mnt/data/wangqq/conda_envs/recons_eval/bin/python3}"
TUM=/mnt/data/wangqq/recons_eval/data/tum_all_sync
POOLS="$V/data/tum_per_scene_splits/evenodd_train_pools.json"
OUT="$V/outputs/tum_gt_pool_per_scene_even"
LOG_ROOT="$V/logs/per_scene_even_gt_pool"
mkdir -p "$OUT" "$LOG_ROOT"

mapfile -t scenes < <("$PY" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1])).keys()))' "$POOLS")

build_one() {
  local scene="$1"
  local frames
  if [[ -f "$OUT/$scene/pool_meta.json" ]]; then
    echo "skip complete pool $scene"
    return
  fi
  frames="$("$PY" -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))[sys.argv[2]])))' "$POOLS" "$scene")"
  "$PY" "$V/scripts/build_tum_gt_pool.py" \
    --tum_root "$TUM" \
    --scene "$scene" \
    --frames "$frames" \
    --gt_name groundtruth.txt \
    --out_dir "$OUT" \
    --lazy_tracks >"$LOG_ROOT/$scene.log" 2>&1
}

running=0
jobs="${GT_POOL_JOBS:-4}"
if (( jobs < 1 )); then
  echo "GT_POOL_JOBS must be at least 1" >&2
  exit 2
fi
for scene in "${scenes[@]}"; do
  build_one "$scene" &
  running=$((running + 1))
  if (( running == jobs )); then
    wait
    running=0
  fi
done
wait

"$PY" - "$POOLS" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

pools = json.load(open(sys.argv[1]))
root = Path(sys.argv[2])
for scene, expected in pools.items():
    meta_path = root / scene / "pool_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    actual = json.loads(meta_path.read_text())["frame_indices"]
    if actual != expected:
        raise AssertionError(f"{scene}: GT frames disagree with frozen even-frame pool")
print(f"validated {len(pools)} even-frame GT pools -> {root}")
PY
