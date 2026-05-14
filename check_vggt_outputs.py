"""
VGGT Output Inspector
=====================
检查 batch_vggt_inference.py 跑出来的结果：
- 4 类输出（cameras / depth / pointmap / tracks）的 shape 和数值范围
- 置信度分布
- 自动生成 depth + confidence 可视化 PNG
- 标记可疑/异常的 scene

用法：
    python check_vggt_outputs.py --output_root outputs/test
    python check_vggt_outputs.py --output_root outputs/test --viz  # 加可视化
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def check_one(npz_path: Path):
    """检查单个 scene 的输出。返回 dict 包含所有诊断信息。"""
    info = {"scene": npz_path.parent.name}
    try:
        d = np.load(npz_path)
    except Exception as e:
        info["error"] = f"load failed: {e}"
        return info

    info["keys"] = list(d.files)

    # ---- 1. Cameras ----
    if "extrinsic" in d and "intrinsic" in d:
        ext, ins = d["extrinsic"], d["intrinsic"]
        info["camera"] = {
            "ext_shape": str(ext.shape),
            "int_shape": str(ins.shape),
            "n_frames": ext.shape[0],
            # 检查相机位置（每帧外参 -R^T @ t 算 camera center）
            "trans_range": f"[{ext[:, :, 3].min():.2f}, {ext[:, :, 3].max():.2f}]",
            "focal_mean": f"{ins[:, 0, 0].mean():.1f}",
        }
    else:
        info["camera"] = "MISSING"

    # ---- 2. Depth ----
    if "depth" in d:
        dep = d["depth"].astype(np.float32)
        conf = d.get("depth_conf", None)
        # 过滤无效值
        valid = (dep > 0) & np.isfinite(dep)
        info["depth"] = {
            "shape": str(dep.shape),
            "range": f"[{dep[valid].min():.3f}, {dep[valid].max():.3f}]" if valid.any() else "ALL_INVALID",
            "median": f"{np.median(dep[valid]):.3f}" if valid.any() else "—",
            "valid_ratio": f"{valid.mean()*100:.1f}%",
        }
        if conf is not None:
            c = conf.astype(np.float32)
            info["depth"]["conf_range"] = f"[{c.min():.3f}, {c.max():.3f}]"
            info["depth"]["conf_median"] = f"{np.median(c):.3f}"

    # ---- 3. Point Map ----
    if "point_map" in d:
        pm = d["point_map"].astype(np.float32)
        info["pointmap"] = {
            "shape": str(pm.shape),
            "xyz_range": (
                f"x[{pm[..., 0].min():.2f},{pm[..., 0].max():.2f}] "
                f"y[{pm[..., 1].min():.2f},{pm[..., 1].max():.2f}] "
                f"z[{pm[..., 2].min():.2f},{pm[..., 2].max():.2f}]"
            ),
        }
        if "point_cloud_unproj" in d:
            info["pointmap"]["unproj_shape"] = str(d["point_cloud_unproj"].shape)

    # ---- 4. Tracks ----
    if "tracks" in d:
        tr = d["tracks"]
        info["tracks"] = {
            "shape": str(tr.shape),
            "n_query": tr.shape[-2] if tr.ndim >= 2 else 1,
        }
        if "track_visibility" in d:
            vis = d["track_visibility"]
            info["tracks"]["vis_mean"] = f"{vis.mean():.3f}"

    # ---- 健康判断 ----
    flags = []
    if "depth" in info and isinstance(info["depth"], dict):
        # depth 全是 0 或全是 inf 都是异常
        rng = info["depth"].get("range", "")
        if "ALL_INVALID" in rng:
            flags.append("depth_all_invalid")
        if info["depth"]["valid_ratio"].rstrip("%") == "0.0":
            flags.append("no_valid_depth")
    info["flags"] = flags
    info["healthy"] = (len(flags) == 0)
    return info


def make_viz(npz_path: Path, out_path: Path, max_frames: int = 8):
    """生成 depth + confidence 可视化拼图。"""
    if not HAS_MPL:
        return
    d = np.load(npz_path)
    if "depth" not in d:
        return
    dep = d["depth"].astype(np.float32)
    conf = d.get("depth_conf", None)
    if conf is not None:
        conf = conf.astype(np.float32)

    # 处理 shape: 可能是 (N, H, W, 1) 或 (N, H, W)
    if dep.ndim == 4 and dep.shape[-1] == 1:
        dep = dep[..., 0]
    N = dep.shape[0]
    n_show = min(N, max_frames)
    idx = np.linspace(0, N - 1, n_show, dtype=int)

    rows = 2 if conf is not None else 1
    fig, axes = plt.subplots(rows, n_show, figsize=(2.5 * n_show, 2.5 * rows))
    if rows == 1:
        axes = axes[None, :]

    for j, i in enumerate(idx):
        # depth
        valid = (dep[i] > 0) & np.isfinite(dep[i])
        if valid.any():
            vmin, vmax = np.percentile(dep[i][valid], [2, 98])
        else:
            vmin, vmax = 0, 1
        axes[0, j].imshow(dep[i], cmap="turbo", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"f{i}", fontsize=8)
        axes[0, j].axis("off")
        # confidence
        if conf is not None:
            c = conf[i] if conf.ndim == 3 else conf[i, ..., 0]
            axes[1, j].imshow(c, cmap="viridis")
            axes[1, j].axis("off")

    axes[0, 0].set_ylabel("depth", fontsize=9)
    if conf is not None:
        axes[1, 0].set_ylabel("conf", fontsize=9)
    plt.suptitle(npz_path.parent.name, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True,
                    help="batch_vggt_inference.py 的 --output_root 目录")
    ap.add_argument("--viz", action="store_true",
                    help="生成 depth/conf 可视化 PNG（需 matplotlib）")
    args = ap.parse_args()

    root = Path(args.output_root)
    scenes = sorted(p.parent for p in root.glob("*/vggt_outputs.npz"))

    print(f"\n{'=' * 80}")
    print(f" VGGT 输出诊断报告  ({len(scenes)} scenes under {root})")
    print(f"{'=' * 80}\n")

    all_info = []
    for s in scenes:
        npz = s / "vggt_outputs.npz"
        info = check_one(npz)
        all_info.append(info)

        # 打印
        print(f"━━ {info['scene']} ━━")
        if "error" in info:
            print(f"  ❌ {info['error']}")
            continue

        if isinstance(info.get("camera"), dict):
            c = info["camera"]
            print(f"  📷 cameras   N={c['n_frames']}  "
                  f"ext{c['ext_shape']}  int{c['int_shape']}  "
                  f"focal_mean={c['focal_mean']}px")
        if isinstance(info.get("depth"), dict):
            dp = info["depth"]
            line = f"  📏 depth     {dp['shape']}  range={dp['range']}  median={dp['median']}  valid={dp['valid_ratio']}"
            if "conf_range" in dp:
                line += f"  conf={dp['conf_range']}"
            print(line)
        if isinstance(info.get("pointmap"), dict):
            pm = info["pointmap"]
            print(f"  🌐 pointmap  {pm['shape']}")
            print(f"     {pm['xyz_range']}")
        if isinstance(info.get("tracks"), dict):
            tr = info["tracks"]
            print(f"  🎯 tracks    {tr['shape']}  n_query={tr['n_query']}", end="")
            if "vis_mean" in tr:
                print(f"  vis_mean={tr['vis_mean']}", end="")
            print()

        if info["flags"]:
            print(f"  ⚠️  flags: {', '.join(info['flags'])}")
        else:
            print(f"  ✅ 看起来正常")

        # 可视化
        if args.viz:
            viz_path = s / "depth_viz.png"
            try:
                make_viz(npz, viz_path)
                print(f"  🖼  -> {viz_path.relative_to(root.parent)}")
            except Exception as e:
                print(f"  ⚠️  viz failed: {e}")
        print()

    # 汇总
    healthy = sum(1 for i in all_info if i.get("healthy"))
    print(f"{'=' * 80}")
    print(f"  汇总：{healthy}/{len(all_info)} 个 scene 通过自动检查")
    bad = [i["scene"] for i in all_info if not i.get("healthy")]
    if bad:
        print(f"  ⚠️ 需要人工检查：{', '.join(bad)}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
