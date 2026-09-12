#!/usr/bin/env bash
# One paper-faithful VGGT baseline profile per invocation.  Placement is read
# from a renderer-preflighted JSON that was frozen without attack scores.
set -Eeuo pipefail

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
VGGT_PY="${VGGT_PY:-/mnt/data/wangqq/conda_envs/vggt/bin/python3}"
RECONS_PY="${RECONS_PY:-/mnt/data/wangqq/conda_envs/recons_eval/bin/python3}"
SCENE="${SCENE:-rgbd_dataset_freiburg3_sitting_halfsphere}"
PROFILE="${PROFILE:-aparate_monitor}"
STEPS="${STEPS:-1000}"
SEED="${SEED:-}"
GPU="${GPU:-0}"
STAGE="${STAGE:-formal}"
RUN_TAG="${RUN_TAG:-}"
PLACEMENTS="${PLACEMENTS:-$VGGT_ROOT/configs/paper_baseline_placements_${SCENE}.json}"
REPORT_DIR="${REPORT_DIR:-$VGGT_ROOT/reports/baseline_experiments}"
TRACK_DIR="$REPORT_DIR/tracks"
CLEAN_RUN="tum10_clean_uniform_l3"

cd "$VGGT_ROOT"
mkdir -p logs "$REPORT_DIR/figures" "$TRACK_DIR"

if [[ ! -f "$PLACEMENTS" ]]; then
  "$VGGT_PY" -B scripts/build_paper_baseline_placements.py \
    --scene "$SCENE" --out "$PLACEMENTS"
fi
if ! "$VGGT_PY" -B - "$PLACEMENTS" <<'PY'
import json,sys
data=json.load(open(sys.argv[1],encoding="utf-8"))
rows=data.get("profiles",[])
raise SystemExit(0 if rows and all("preflight" in row for row in rows) else 1)
PY
then
  "$VGGT_PY" -B scripts/preflight_paper_baseline_placements.py \
    --placements "$PLACEMENTS"
fi

mapfile -t fields < <("$VGGT_PY" -B - "$PLACEMENTS" "$PROFILE" <<'PY'
import json,sys
data=json.load(open(sys.argv[1],encoding="utf-8"))
row=next((x for x in data["profiles"] if x["name"]==sys.argv[2]),None)
if row is None: raise SystemExit(f"unknown profile: {sys.argv[2]}")
if not row.get("preflight",{}).get("passed",False):
    raise SystemExit(f"preflight did not pass: {sys.argv[2]}")
for key in ("attack_loss","explicit_quad_world","baseline_focus_quad_world","carrier"):
    print(row.get(key,""))
PY
)
METHOD="${fields[0]}"
QUAD="${fields[1]}"
FOCUS_QUAD="${fields[2]}"
CARRIER="${fields[3]}"

if [[ "$STAGE" == "formal" && "$STEPS" != "1000" ]]; then
  echo "Formal table is locked to 1000 updates; got STEPS=$STEPS" >&2
  exit 2
fi
if [[ "$STAGE" != "formal" && "$STAGE" != "smoke" ]]; then
  echo "STAGE must be formal or smoke" >&2
  exit 2
fi

TEXTURE_SIZE=64
TEXTURE_INIT=random
TEXTURE_INIT_IMAGE=""
PATCH_LR=0.01
PHYSICAL_EOT=1
EOT_SAMPLES=1
EOT_BRIGHTNESS=0.10
EOT_CONTRAST=0.20
EOT_SATURATION=0.0
EOT_HUE=0.0
EOT_GAMMA=0.0
EOT_NOISE_STD=0.10
EOT_WARMUP_FRACTION=0.0
EOT_GEO_ROTATE_DEGREES=20
EOT_GEO_SCALE=0.03
EOT_GEO_TRANSLATE=0.02
EOT_GEO_PERSPECTIVE=0.01
BASELINE_FOCUS_SCALE=2.0
BASELINE_WEATHER_STRENGTH=0.0
BASELINE_TEXTURE_REPEATS=6
BASELINE_CONVERTED_TEXTURE_SIZE=4096
BASELINE_TEXTURE_UPSAMPLE_FACTOR=8
TEXTURE_INIT_PRESERVE_ASPECT=0
BASELINE_SPOO_NATURALNESS=0
BASELINE_SPOO_MASK_SHARPNESS=1.0
BASELINE_SPOO_BORDER_LR=1.0
BASELINE_SPOO_MASK_UPDATE_INTERVAL=20
SCHEDULER=none
WARMUP_ITERATIONS=0

case "$METHOD" in
  baseline_spoo_depth)
    # Official example: Warnning.png is both content and style and is also the
    # optimized image. VGG19 content/style, TV, second-stage matting realism,
    # L-BFGS texture and late-start single-edge Adam are all enabled in code.
    TEXTURE_INIT=image
    TEXTURE_INIT_IMAGE="$VGGT_ROOT/assets/hazard_textures/mde_attack_warnning.png"
    TEXTURE_SIZE=600
    TEXTURE_INIT_PRESERVE_ASPECT=1
    BASELINE_SPOO_NATURALNESS=1
    PATCH_LR=0.3689
    # The source controller updates for 10,000 closures: one source pixel per
    # closure and a mask-weight update every 20 closures.  The common VGGT
    # table allows 1,000 costly model updates, so preserve controller progress
    # with a 10x pixel step and a 2-update interval.  This is the explicit
    # equal-budget adapter; texture still receives exactly 1,000 gradients.
    BASELINE_SPOO_BORDER_LR=10.0
    BASELINE_SPOO_MASK_UPDATE_INTERVAL=2
    # SPOO varies road backgrounds but does not add the generic photometric or
    # projective EOT used by this repository's own method.
    PHYSICAL_EOT=0
    ;;
  baseline_aparate_depth)
    # APARATE: random patch, Adam 0.01, linear bbox scale 0.2, +/-20 degree,
    # multiplicative brightness/contrast and bounded uniform +/-0.1 noise.
    # It stays centred as in the paper; translation/perspective jitter is not
    # stated and is disabled. The paper states random scaling but publishes no
    # range, so +/-3% is recorded as a VGGT adapter choice.
    EOT_GAMMA=0.0
    EOT_GEO_TRANSLATE=0.0
    EOT_GEO_PERSPECTIVE=0.0
    ;;
  baseline_3d2fool_depth)
    TEXTURE_SIZE=128
    TEXTURE_INIT=random
    PATCH_LR=0.01
    EOT_BRIGHTNESS=0.20
    EOT_CONTRAST=0.10
    EOT_SATURATION=0.10
    EOT_HUE=0.05
    EOT_NOISE_STD=0.0
    # Independent flips and optional +90-degree rotation already happen inside
    # 3D2Fool's texture-conversion operator.  Do not rotate it a second time
    # through the generic grid EOT.
    EOT_GEO_ROTATE_DEGREES=0
    EOT_GEO_TRANSLATE=0.0
    EOT_GEO_SCALE=0.0
    EOT_GEO_PERSPECTIVE=0.0
    # CARLA map/weather labels do not exist in TUM; disable the old generic
    # weather proxy instead of mislabelling it as public 3D2Fool logic.
    BASELINE_WEATHER_STRENGTH=0.0
    ;;
  baseline_vo_mprms|baseline_vo_rms)
    # Paper Algorithms 1-2 initialise P ~ Uniform(0,1). The public PGD class has
    # a zero-valued fallback when rand_init is omitted, but that fallback is not
    # the initialization specified by the paper's reported method.
    TEXTURE_INIT=random
    PATCH_LR=0.00392156862745098
    PHYSICAL_EOT=0
    ;;
  *)
    echo "Unsupported paper baseline method: $METHOD" >&2
    exit 2
    ;;
esac

if [[ -z "$SEED" ]]; then
  # SPOO's public parser fixes seed 17. The other papers do not disclose a
  # seed; use one shared reproducibility seed for their random initializers.
  if [[ "$METHOD" == "baseline_spoo_depth" ]]; then
    SEED=17
  else
    SEED=0
  fi
fi

tag_fragment=""
if [[ -n "$RUN_TAG" ]]; then
  tag_fragment="${RUN_TAG}_"
fi
run_name="baseline_${STAGE}_${tag_fragment}${PROFILE}_s${SEED}_u${STEPS}"
run_root="$VGGT_ROOT/outputs_attack_geometry_aware_tum10/$run_name"
log_path="$VGGT_ROOT/logs/${run_name}.log"
figure_dir="$REPORT_DIR/figures/$run_name"
eval_csv="$REPORT_DIR/${run_name}_four_tasks.csv"

echo "[paper-baseline] profile=$PROFILE method=$METHOD carrier=$CARRIER stage=$STAGE updates=$STEPS"
texture_image_args=()
if [[ -n "$TEXTURE_INIT_IMAGE" ]]; then
  texture_image_args+=(TEXTURE_INIT_IMAGE="$TEXTURE_INIT_IMAGE")
fi

env CUDA_VISIBLE_DEVICES="$GPU" \
  SCENE_PATTERN="$SCENE" \
  ITERATIONS="$STEPS" INNER_LOOP=1 SCENES_PER_ITERATION=1 \
  PATCH_LR="$PATCH_LR" TEXTURE_SIZE="$TEXTURE_SIZE" \
  TEXTURE_INIT="$TEXTURE_INIT" \
  TEXTURE_INIT_PRESERVE_ASPECT="$TEXTURE_INIT_PRESERVE_ASPECT" \
  SCHEDULER="$SCHEDULER" WARMUP_ITERATIONS="$WARMUP_ITERATIONS" \
  PLANE_MODE=explicit_world_quad EXPLICIT_QUAD_WORLD="$QUAD" \
  BASELINE_FOCUS_QUAD_WORLD="$FOCUS_QUAD" \
  OPTIMIZE_GEOMETRY=0 USE_DEPTH_VISIBILITY=1 VISIBILITY_DEPTH_MARGIN=0.08 \
  SURFACE_SUPPORT_CHECK=1 SURFACE_SUPPORT_ABS_TOLERANCE=0.06 \
  SURFACE_SUPPORT_REL_TOLERANCE=0.02 SURFACE_MIN_SUPPORT_RATIO=0.50 \
  ATTACK_LOSS="$METHOD" SEED="$SEED" TUM_GEOM_RUN_NAME="$run_name" \
  TV_WEIGHT=0 PRINTABILITY_WEIGHT=0 LOW_FREQUENCY_WEIGHT=0 \
  NATURAL_REFERENCE_WEIGHT=0 \
  PHYSICAL_EOT="$PHYSICAL_EOT" EOT_SAMPLES="$EOT_SAMPLES" \
  EOT_BRIGHTNESS="$EOT_BRIGHTNESS" EOT_CONTRAST="$EOT_CONTRAST" \
  EOT_SATURATION="$EOT_SATURATION" EOT_HUE="$EOT_HUE" \
  EOT_GAMMA="$EOT_GAMMA" \
  EOT_WARMUP_FRACTION="$EOT_WARMUP_FRACTION" \
  EOT_NOISE_STD="$EOT_NOISE_STD" EOT_GEO_ROTATE_DEGREES="$EOT_GEO_ROTATE_DEGREES" \
  EOT_GEO_SCALE="$EOT_GEO_SCALE" EOT_GEO_TRANSLATE="$EOT_GEO_TRANSLATE" \
  EOT_GEO_PERSPECTIVE="$EOT_GEO_PERSPECTIVE" \
  BASELINE_FOCUS_SCALE="$BASELINE_FOCUS_SCALE" \
  BASELINE_SPOO_NATURALNESS="$BASELINE_SPOO_NATURALNESS" \
  BASELINE_SPOO_MASK_SHARPNESS="$BASELINE_SPOO_MASK_SHARPNESS" \
  BASELINE_SPOO_BORDER_LR="$BASELINE_SPOO_BORDER_LR" \
  BASELINE_SPOO_MASK_UPDATE_INTERVAL="$BASELINE_SPOO_MASK_UPDATE_INTERVAL" \
  BASELINE_TEXTURE_REPEATS="$BASELINE_TEXTURE_REPEATS" \
  BASELINE_CONVERTED_TEXTURE_SIZE="$BASELINE_CONVERTED_TEXTURE_SIZE" \
  BASELINE_TEXTURE_UPSAMPLE_FACTOR="$BASELINE_TEXTURE_UPSAMPLE_FACTOR" \
  BASELINE_WEATHER_STRENGTH="$BASELINE_WEATHER_STRENGTH" \
  SKIP_TUM_PREPARE=1 SKIP_CLEAN_VGGT=1 \
  FORCE_TRAIN=1 FORCE_APPLY=1 RUN_EVAL=0 RUN_GAUGE_DIAG=0 RUN_CONSISTENCY_CHECK=0 \
  "${texture_image_args[@]}" \
  bash run_geometry_aware_tum10.sh >"$log_path" 2>&1

# A formal SPOO run is invalid if the pixel-space one-edge update overshoots
# the paper's 1/9 target by a large amount.  The public code stops immediately
# after crossing the target, so one-pixel tolerance is sufficient; 0.01 is a
# deliberately looser guard for the soft tanh boundary.
if [[ "$STAGE" == "formal" && "$METHOD" == "baseline_spoo_depth" ]]; then
  "$VGGT_PY" -B - "$run_root/geometry_patch/geometry_patch_meta.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1], encoding="utf-8"))["paper_baseline"]
target = float(meta["spoo_target_area_fraction"])
area = float(meta["spoo_final_area_fraction"])
if not (target - 0.03 <= area <= target + 0.01):
    raise SystemExit(
        f"SPOO final soft area {area:.6f} missed target {target:.6f}; "
        "formal result rejected"
    )
print(f"[paper-baseline] SPOO area gate passed: {area:.6f} vs {target:.6f}")
PY
fi

dump_runs="$run_name"
if [[ ! -f "$TRACK_DIR/${CLEAN_RUN}__${SCENE}.npz" ]]; then
  dump_runs="$CLEAN_RUN,$run_name"
fi
CUDA_VISIBLE_DEVICES="$GPU" "$VGGT_PY" -B scripts/dump_tracks.py \
  --runs "$dump_runs" --scene "$SCENE" --out_dir "$TRACK_DIR" \
  --gt_dir "$VGGT_ROOT/outputs/tum_gt_point_track" \
  --clean "$VGGT_ROOT/outputs_attack_geometry_aware_tum10/$CLEAN_RUN"

"$RECONS_PY" -B scripts/eval_four_tasks.py \
  --runs "$CLEAN_RUN,$run_name" --scene "$SCENE" --track_dir "$TRACK_DIR" \
  --gt_dir "$VGGT_ROOT/outputs/tum_gt_point_track" \
  --clean "$CLEAN_RUN" --out_csv "$eval_csv" --require_render_ok

"$VGGT_PY" -B scripts/visualize_tum10_geometry_patch.py \
  --geometry_output_root "$run_root" --out_dir "$figure_dir" \
  --scene_pattern "$SCENE" --frames all --contact_columns 5 \
  --contact_sheet_only --no_outline --no_label --no_outline_frames

"$VGGT_PY" -B scripts/record_baseline_experiment.py \
  --run "$run_name" --scene "$SCENE" --profile "$PROFILE" \
  --clean_run "$CLEAN_RUN" \
  --placements "$PLACEMENTS" --eval_csv "$eval_csv" \
  --figures_dir "$figure_dir" --log "$log_path" --stage "$STAGE"

echo "run=$run_name"
echo "report=$REPORT_DIR/baseline_experiment_record.md"
echo "figures=$figure_dir"
