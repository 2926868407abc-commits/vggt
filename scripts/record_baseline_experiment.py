"""Append one completed paper-baseline run to the shared Markdown record."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path


VG = Path("/mnt/data/wangqq/vggt")
RUNS = VG / "outputs_attack_geometry_aware_tum10"
REPORT = VG / "reports/baseline_experiments/baseline_experiment_record.md"


def read_eval_row(path: Path, run: str) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("run") == run:
                return dict(row)
    return {}


REQUIRED_METRICS = (
    "ATE",
    "RPE_rot_deg",
    "depth_absrel",
    "point_acc",
    "track_epe",
    "track_epe_mean",
    "track_vis_f1",
    "render_ok",
)


def fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—" if value in (None, "") else str(value)
    return f"{number:.{digits}f}"


def replace_block(text: str, run: str, block: str) -> str:
    start = f"<!-- BASELINE-RUN:{run}:START -->"
    end = f"<!-- BASELINE-RUN:{run}:END -->"
    if start in text and end in text:
        prefix, tail = text.split(start, 1)
        _, suffix = tail.split(end, 1)
        return prefix.rstrip() + "\n\n" + block + suffix
    return text.rstrip() + "\n\n" + block + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--eval_csv", type=Path, required=True)
    parser.add_argument("--figures_dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--clean_run", default="tum10_clean_uniform_l3")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()

    run_root = RUNS / args.run
    patch_dir = run_root / "geometry_patch"
    summary_path = run_root / args.scene / "attack_summary.json"
    meta_path = patch_dir / "geometry_patch_meta.json"
    for path in (summary_path, meta_path, args.placements, args.log):
        if not path.exists():
            raise FileNotFoundError(path)

    placement_spec = json.loads(args.placements.read_text(encoding="utf-8"))
    placement = next(
        row for row in placement_spec["profiles"] if row["name"] == args.profile
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics = read_eval_row(args.eval_csv, args.run)
    clean_metrics = read_eval_row(args.eval_csv, args.clean_run)
    if not metrics:
        raise RuntimeError(f"No four-task evaluation row for {args.run} in {args.eval_csv}")
    if not clean_metrics:
        raise RuntimeError(
            f"No clean four-task evaluation row for {args.clean_run} in {args.eval_csv}"
        )
    missing_metrics = [key for key in REQUIRED_METRICS if metrics.get(key, "") == ""]
    missing_clean_metrics = [
        key for key in REQUIRED_METRICS if clean_metrics.get(key, "") == ""
    ]
    if missing_metrics or missing_clean_metrics:
        raise RuntimeError(
            f"Incomplete four-task rows; attack missing {missing_metrics}, "
            f"clean missing {missing_clean_metrics}"
        )
    history_path = patch_dir / "training_history.jsonl"
    last_history = {}
    if history_path.exists():
        lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            last_history = json.loads(lines[-1])

    initial = patch_dir / "initial_patch.png"
    optimized = patch_dir / "optimized_patch.png"
    contact = args.figures_dir / args.scene / "contact_sheet_patch.png"
    for path in (initial, optimized, contact):
        if not path.exists():
            raise FileNotFoundError(f"Required baseline visual is missing: {path}")

    paper = meta.get("paper_baseline", {}) or {}
    geometry = summary.get("geometry", {})
    preflight = placement.get("preflight", {})
    eot = meta.get("physical_eot", {})
    block_lines = [
        f"<!-- BASELINE-RUN:{args.run}:START -->",
        f"## {args.run}",
        "",
        f"- 阶段：{'正式 1000 更新' if args.stage == 'formal' else '短冒烟，只验代码与摆放'}",
        f"- 方法配置：`{args.profile}` / `{meta.get('attack_target')}`",
        f"- 场景：`{args.scene}`",
        f"- 载体与摆放：{placement['carrier']}；{placement['placement_rule']}",
        f"- 论文一致性边界：{placement['source_fidelity']}",
        f"- 外接物理尺寸：{placement['patch_width_m']:.3f} × {placement['patch_height_m']:.3f} m；外接面积 {placement['outer_quad_area_m2']:.4f} m²",
        f"- 纹理：初始化 `{meta.get('texture_init')}`，分辨率 `{meta.get('texture_shape')}`",
        f"- 优化：`{meta.get('optimizer')}`，学习率 {meta.get('patch_lr')}，总更新数 {meta.get('total_updates')}",
        f"- EOT：{'开' if eot.get('enabled') else '关'}；采样数 {eot.get('samples', 1)}",
        f"- EOT 参数：亮度 {eot.get('brightness')}，对比度 {eot.get('contrast')}，噪声 {eot.get('noise_std')}，平移 {eot.get('geo_translate')}，尺度 {eot.get('geo_scale')}，旋转 {eot.get('geo_rotate_degrees')}°，透视 {eot.get('geo_perspective')}",
        f"- 渲染检查：预检通过 `{preflight.get('passed')}`；训练输出平均覆盖率 {fmt(geometry.get('mask_coverage_mean'), 5)}",
        f"- 日志：`{args.log.resolve()}`",
    ]
    if meta.get("attack_target") == "baseline_spoo_depth":
        block_lines.extend(
            [
                f"- SPOO 目标面积比例：{paper.get('spoo_target_area_fraction')}；最终边界 `[l,r,t,b]` = `{paper.get('spoo_final_borders_lrtb')}`",
                f"- SPOO 区域是否已到 1/9 并冻结：`{paper.get('spoo_region_frozen')}`；末步面积比例 {fmt(last_history.get('spoo_area_fraction'), 5)}",
            ]
        )
    elif meta.get("attack_target") == "baseline_aparate_depth":
        block_lines.append(
            f"- APARATE 正则：patch 外权重 {paper.get('aparate_outside_weight')}，TV {paper.get('aparate_tv_weight')}，NPS {paper.get('aparate_nps_weight')}"
        )
    elif meta.get("attack_target") == "baseline_3d2fool_depth":
        block_lines.append(
            f"- 3D²Fool 适配：128 种子重复 {paper.get('texture_repeats')} 次并裁成 {paper.get('converted_texture_size')}；TV {paper.get('3d2fool_tv_weight')}，NPS {paper.get('3d2fool_nps_weight')}，天气强度 {paper.get('weather_strength')}"
        )
    elif meta.get("attack_target") in ("baseline_vo_mprms", "baseline_vo_rms"):
        block_lines.append("- Passive VO 更新：PGD-sign；固定屏幕位置；不启用 EOT。")

    block_lines.extend(
        [
            "",
            "### 统一四任务结果",
            "",
            "| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
        "",
            "### 本实验三张图（服务器绝对路径）",
            "",
            f"1. Patch 初始图：`{initial.resolve()}`",
            f"2. Patch 优化后图：`{optimized.resolve()}`",
            f"3. 优化后 patch 在 10 帧中的无红边合图：`{contact.resolve()}`",
            "",
            "> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。",
            f"<!-- BASELINE-RUN:{args.run}:END -->",
        ]
    )

    def metric_row(label: str, values: dict[str, str]) -> str:
        return "| " + " | ".join(
            [
                label,
                fmt(values.get("ATE")),
                fmt(values.get("RPE_rot_deg")),
                fmt(values.get("depth_absrel")),
                fmt(values.get("point_acc")),
                fmt(values.get("track_epe"), 2),
                fmt(values.get("track_epe_mean"), 2),
                fmt(values.get("track_vis_f1"), 3),
                str(values.get("render_ok", "—")),
            ]
        ) + " |"

    # The first empty string belongs to the heading.  Insert after the metrics
    # header/separator by locating the separator explicitly.
    separator_index = block_lines.index(
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|"
    )
    block_lines[separator_index + 1:separator_index + 1] = [
        metric_row("Clean", clean_metrics),
        metric_row("Attack", metrics),
    ]
    block = "\n".join(block_lines)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.report.with_suffix(args.report.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if args.report.exists():
            report_text = args.report.read_text(encoding="utf-8")
        else:
            report_text = "# 论文 Baseline 迁移到 VGGT：逐实验记录\n"
        updated = replace_block(report_text, args.run, block)
        temporary = args.report.with_name(f".{args.report.name}.{os.getpid()}.tmp")
        temporary.write_text(updated, encoding="utf-8")
        temporary.replace(args.report)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    print(f"recorded {args.run} -> {args.report}")


if __name__ == "__main__":
    main()
