#!/usr/bin/env bash
set -u

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
RECONS_ROOT="${RECONS_ROOT:-/mnt/data/wangqq/recons_eval}"
VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"
RECONS_PY="${RECONS_PY:-/mnt/data/wangqq/conda_envs/recons_eval/bin/python3}"
CKPT="${CKPT:-$VGGT_ROOT/checkpoints/VGGT-1B}"

NYU_SCENES="${NYU_SCENES:-$VGGT_ROOT/data/nyu_v2_recons_eval_scenes}"
BONN_RAW="${BONN_RAW:-$RECONS_ROOT/data/bonn}"
BONN_DATASET="${BONN_DATASET:-$BONN_RAW/rgbd_bonn_dataset}"
BONN_SCENES="${BONN_SCENES:-$VGGT_ROOT/data/bonn_monodepth_scenes}"
TUM_ROOT="${TUM_ROOT:-$RECONS_ROOT/data/tum}"
NRGBD_RAW="${NRGBD_RAW:-$RECONS_ROOT/data/nrgbd}"
NRGBD_SCENES="${NRGBD_SCENES:-$VGGT_ROOT/data/nrgbd_sparse_mv_recon_scenes}"

missing=0

ok() {
  printf '[OK]      %s\n' "$*"
}

warn() {
  printf '[WARN]    %s\n' "$*"
}

missing_item() {
  printf '[MISSING] %s\n' "$*"
  missing=$((missing + 1))
}

check_file() {
  [[ -f "$1" ]] && ok "$2: $1" || missing_item "$2: $1"
}

check_dir() {
  [[ -d "$1" ]] && ok "$2: $1" || missing_item "$2: $1"
}

check_glob() {
  local pattern="$1"
  local label="$2"
  if compgen -G "$pattern" >/dev/null; then
    local count
    count=$(find $(dirname "$pattern") -maxdepth 1 -name "$(basename "$pattern")" 2>/dev/null | wc -l)
    ok "$label: matched $count item(s), pattern=$pattern"
  else
    missing_item "$label: no match, pattern=$pattern"
  fi
}

printf 'VGGT_ROOT=%s\n' "$VGGT_ROOT"
printf 'RECONS_ROOT=%s\n\n' "$RECONS_ROOT"

echo '== Environments =='
check_file "$VGGT_PY" "VGGT python"
check_file "$RECONS_PY" "recons_eval python"
if [[ -f "$CKPT" ]]; then
  ok "VGGT checkpoint file: $CKPT"
elif [[ -d "$CKPT" ]]; then
  ok "VGGT checkpoint directory: $CKPT"
else
  missing_item "VGGT checkpoint path: $CKPT"
fi

echo
echo '== Core Code =='
check_file "$VGGT_ROOT/attack_vggt_new1.py" "VGGT attack script"
check_file "$VGGT_ROOT/attack_vggt_vla_style.py" "VLA-style universal patch script"
if [[ -f "$VGGT_ROOT/attack_vggt_new1.py" ]]; then
  grep -q "torch.optim.Adam" "$VGGT_ROOT/attack_vggt_new1.py" \
    && ok "patch attack uses Adam optimizer" \
    || warn "attack script does not contain torch.optim.Adam; patch may still be old PGD-style"
fi
check_file "$VGGT_ROOT/run_feature_attack_eval_all.sh" "full attack/eval bash"
check_file "$VGGT_ROOT/run_vla_style_feature_attack_eval_all.sh" "VLA-style attack/eval bash"
check_file "$RECONS_ROOT/monodepth/eval.py" "monodepth eval"
check_file "$RECONS_ROOT/datasets/preprocess/prepare_tum.py" "TUM prepare.py"
check_file "$RECONS_ROOT/datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf500.json" "NRGBD sparse seq-id map"
check_file "$RECONS_ROOT/relpose/evo_utils.py" "pose metrics helper"
check_file "$RECONS_ROOT/datasets/nrgbd.py" "NRGBD dataset loader"

echo
echo '== Optional Helper Scripts In recons_eval/scripts =='
for f in \
  export_nyu_v2_scenes.py \
  prepare_nyu_v2_monodepth_scenes.py \
  prepare_nyu_v2_vggt_monodepth_for_recons_eval.py \
  prepare_bonn_monodepth_scenes.py \
  prepare_bonn_monodepth_for_recons_eval.py \
  prepare_nrgbd_mv_recon_scenes.py \
  eval_vggt_tum_pose_for_recons_eval.py \
  eval_vggt_nrgbd_mv_recon_for_recons_eval.py
do
  if [[ -f "$RECONS_ROOT/scripts/$f" ]]; then
    ok "helper script exists: scripts/$f"
  else
    warn "helper script absent: scripts/$f"
  fi
done
echo '[INFO]    Missing optional helper scripts are okay for run_feature_attack_eval_all.sh because conversion/eval bridges are embedded in that bash.'

echo
echo '== NYU-v2 =='
check_dir "$NYU_SCENES" "NYU VGGT scene root"
check_glob "$NYU_SCENES/nyu_*" "NYU scenes"
check_glob "$NYU_SCENES/nyu_*/images/*" "NYU scene images"
check_dir "$RECONS_ROOT/data/nyu-v2/val/nyu_images" "NYU recons_eval images"
check_dir "$RECONS_ROOT/data/nyu-v2/val/nyu_depths" "NYU recons_eval GT depths"

echo
echo '== Bonn =='
check_dir "$BONN_RAW" "Bonn raw/prepared root"
check_dir "$BONN_DATASET" "Bonn rgbd_bonn_dataset root"
if compgen -G "$BONN_DATASET/rgbd_bonn_*/rgb_110/*.png" >/dev/null; then
  ok "Bonn rgb_110 frames exist under $BONN_DATASET"
else
  warn "Bonn rgb_110 frames not found under $BONN_DATASET; run datasets/preprocess/prepare_bonn.py or prepare_bonn.sh first"
fi
if [[ -d "$BONN_SCENES" ]]; then
  ok "Bonn VGGT scene root: $BONN_SCENES"
  if compgen -G "$BONN_SCENES/rgbd_bonn_*__*/images/*" >/dev/null; then
    ok "Bonn VGGT scene images exist"
  else
    warn "Bonn scene root exists but no scene images matched"
  fi
else
  warn "Bonn VGGT scene root absent; full bash will create it from rgb_110 frames if raw Bonn data exists"
fi

echo
echo '== TUM-dynamics =='
check_dir "$TUM_ROOT" "TUM root"
check_glob "$TUM_ROOT/rgbd_dataset_freiburg3_*" "TUM sequences"
if compgen -G "$TUM_ROOT/rgbd_dataset_freiburg3_*/rgb_90/*.png" >/dev/null; then
  ok "TUM rgb_90 frames exist"
else
  warn "TUM rgb_90 frames not found; full bash will try prepare_tum.py"
fi
if compgen -G "$TUM_ROOT/rgbd_dataset_freiburg3_*/groundtruth_90.txt" >/dev/null; then
  ok "TUM groundtruth_90.txt files exist"
else
  warn "TUM groundtruth_90.txt not found; full bash will try prepare_tum.py"
fi
if compgen -G "$TUM_ROOT/rgbd_dataset_freiburg3_*/images" >/dev/null; then
  ok "TUM images links/folders exist"
else
  warn "TUM images links absent; full bash will create images -> rgb_90"
fi

echo
echo '== Neural-RGBD Sparse =='
check_dir "$NRGBD_RAW" "Neural-RGBD raw root"
check_file "$RECONS_ROOT/datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf500.json" "Neural-RGBD sparse seq map"
if [[ -d "$NRGBD_SCENES" ]]; then
  ok "Neural-RGBD VGGT scene root: $NRGBD_SCENES"
  if compgen -G "$NRGBD_SCENES/*/images/*.png" >/dev/null; then
    ok "Neural-RGBD VGGT scene images exist"
  else
    warn "Neural-RGBD scene root exists but no scene images matched"
  fi
else
  warn "Neural-RGBD VGGT scene root absent; full bash will create it from seq-id map if raw data exists"
fi

echo
echo '== Existing Attack Outputs =='
for d in \
  "$VGGT_ROOT/outputs_attack/nyu_v2_feature_global_l3" \
  "$VGGT_ROOT/outputs_attack/nyu_v2_feature_patch_adam_l3" \
  "$VGGT_ROOT/outputs_attack/bonn_feature_global_l3" \
  "$VGGT_ROOT/outputs_attack/bonn_feature_patch_adam_l3" \
  "$VGGT_ROOT/outputs_attack/tum_feature_global_l3" \
  "$VGGT_ROOT/outputs_attack/tum_feature_patch_adam_l3" \
  "$VGGT_ROOT/outputs_attack/nrgbd_sparse_feature_global_l3" \
  "$VGGT_ROOT/outputs_attack/nrgbd_sparse_feature_patch_adam_l3"
do
  if [[ -f "$d/attack_batch_summary.json" ]]; then
    ok "existing attack output: $d"
  else
    warn "attack output not complete yet: $d"
  fi
done

echo
if [[ "$missing" -eq 0 ]]; then
  echo "[SUMMARY] Required files/directories look present. Warnings may still be okay."
  exit 0
else
  echo "[SUMMARY] Missing required item count: $missing"
  exit 1
fi
