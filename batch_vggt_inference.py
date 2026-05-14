"""
VGGT Batch Inference Script
============================
批量处理 50-100 个多视角场景，记录 VGGT 的 4 类输出：
    1. Camera parameters (extrinsic + intrinsic)
    2. Depth maps + confidence
    3. Point maps + confidence (+ unprojected point cloud)
    4. 3D point tracks (image-center query)

使用方法：
    python batch_vggt_inference.py \
        --scenes_root /data/vggt_scenes/all \
        --output_root /data/vggt_outputs \
        --max_frames 40 \
        --save_ply

输出结构：
    /data/vggt_outputs/
    ├── scene_001/
    │   ├── vggt_outputs.npz       # 主结果
    │   ├── pointcloud.ply         # 可视化点云
    │   └── meta.json              # 元信息（耗时、显存、状态）
    └── summary.csv                # 总表

Author: PanoNav project
"""

import os
import glob
import json
import time
import argparse
import traceback
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


# ---------- helpers ----------

def find_images(scene_dir: Path):
    """支持 scene/images/*.* 和 scene/*.* 两种结构。"""
    img_dir = scene_dir / "images"
    if not img_dir.is_dir():
        img_dir = scene_dir
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for e in exts:
        paths.extend(glob.glob(str(img_dir / e)))
    return sorted(paths)


def subsample(paths, max_n):
    """等距下采样到 max_n 张。"""
    if len(paths) <= max_n:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_n, dtype=int)
    return [paths[i] for i in idx]


def save_ply(points, colors, out_path):
    """保存点云为 .ply ASCII。points: (N,3), colors: (N,3) 0-255."""
    n = len(points)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(out_path, "w") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# ---------- core ----------

def process_scene(model, scene_dir: Path, out_root: Path, args, device, dtype):
    """处理单个 scene，返回 meta dict。"""
    scene_name = scene_dir.name
    out_dir = out_root / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"scene": scene_name, "status": "init", "n_frames": 0}

    image_paths = find_images(scene_dir)
    if len(image_paths) == 0:
        meta["status"] = "no_images"
        return meta

    image_paths = subsample(image_paths, args.max_frames)
    meta["n_frames"] = len(image_paths)

    t0 = time.time()
    try:
        images = load_and_preprocess_images(image_paths).to(device)  # (N, 3, H, W)
        images_b = images[None]  # add batch dim

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                agg_tokens, ps_idx = model.aggregator(images_b)

            # 1. Cameras
            pose_enc = model.camera_head(agg_tokens)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images.shape[-2:]
            )

            # 2. Depth
            depth, depth_conf = model.depth_head(agg_tokens, images_b, ps_idx)

            # 3. Point maps
            pmap, pmap_conf = model.point_head(agg_tokens, images_b, ps_idx)

            # 也做 depth+pose 的反投影点云（论文推荐，通常更准）
            pcd_unproj = unproject_depth_map_to_point_map(
                depth.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
            )

            # 4. Tracks: 用第一帧中心点做 query（你也可以改成多点）
            H, W = images.shape[-2:]
            query = torch.FloatTensor([[W / 2, H / 2]]).to(device)
            tracks, vis_score, conf_score = model.track_head(
                agg_tokens, images_b, ps_idx, query_points=query[None]
            )

        # ---- 保存主结果 ----
        np.savez_compressed(
            out_dir / "vggt_outputs.npz",
            extrinsic=extrinsic.squeeze(0).cpu().numpy().astype(np.float32),
            intrinsic=intrinsic.squeeze(0).cpu().numpy().astype(np.float32),
            depth=depth.squeeze(0).cpu().numpy().astype(np.float16),
            depth_conf=depth_conf.squeeze(0).cpu().numpy().astype(np.float16),
            point_map=pmap.squeeze(0).cpu().numpy().astype(np.float16),
            point_conf=pmap_conf.squeeze(0).cpu().numpy().astype(np.float16),
            point_cloud_unproj=pcd_unproj.astype(np.float16),
            tracks=tracks[-1].cpu().numpy() if isinstance(tracks, list)
                   else tracks.cpu().numpy(),
            track_visibility=vis_score.cpu().numpy(),
            track_confidence=conf_score.cpu().numpy(),
            image_paths=np.array([os.path.basename(p) for p in image_paths]),
        )

        # ---- 可选：导出可视化点云 ----
        if args.save_ply:
            # 用 depth 反投影 + 置信度过滤
            conf_np = depth_conf.squeeze(0).cpu().numpy()
            mask = conf_np > np.quantile(conf_np, 0.3)  # 保留高置信 70%

            imgs_np = (images.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            pts = pcd_unproj.reshape(-1, 3)
            cols = imgs_np.reshape(-1, 3)
            m = mask.reshape(-1)
            pts, cols = pts[m], cols[m]

            # 进一步下采样防止文件过大
            if len(pts) > 500_000:
                ridx = np.random.choice(len(pts), 500_000, replace=False)
                pts, cols = pts[ridx], cols[ridx]
            save_ply(pts, cols, out_dir / "pointcloud.ply")

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        meta.update({
            "status": "ok",
            "time_sec": round(time.time() - t0, 2),
            "peak_gpu_gb": round(peak_mem, 2),
        })

    except torch.cuda.OutOfMemoryError:
        meta["status"] = "oom"
        meta["error"] = f"OOM with {meta['n_frames']} frames"
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = str(e)
        meta["traceback"] = traceback.format_exc()
    finally:
        torch.cuda.empty_cache()

    # 写 meta
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_root", required=True,
                    help="父目录，下面每个子目录是一个 scene")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--max_frames", type=int, default=40,
                    help="单 scene 最多帧数（防 OOM）。40 帧在 24GB 卡安全")
    ap.add_argument("--scene_pattern", default="*",
                    help="scene 目录的 glob，比如 'tnt_*' 只跑 T&T")
    ap.add_argument("--save_ply", action="store_true",
                    help="是否额外保存可视化点云 .ply")
    ap.add_argument("--ckpt", default="facebook/VGGT-1B",
                    help="HF 模型路径，可换 VGGT-1B-Commercial")
    ap.add_argument("--skip_existing", action="store_true",
                    help="跳过已处理（output_root/scene/meta.json 存在）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if torch.cuda.is_available() and
             torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    print(f"[cfg] device={device}, dtype={dtype}, ckpt={args.ckpt}")

    print(f"[model] loading {args.ckpt} ...")
    model = VGGT.from_pretrained(args.ckpt).to(device).eval()

    scenes_root = Path(args.scenes_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted(
        d for d in scenes_root.glob(args.scene_pattern) if d.is_dir()
    )
    print(f"[data] found {len(scene_dirs)} scenes under {scenes_root}")

    all_meta = []
    for scene_dir in tqdm(scene_dirs, desc="VGGT"):
        if args.skip_existing and (output_root / scene_dir.name / "meta.json").exists():
            with open(output_root / scene_dir.name / "meta.json") as f:
                all_meta.append(json.load(f))
            continue
        meta = process_scene(model, scene_dir, output_root, args, device, dtype)
        all_meta.append(meta)
        # 实时打印状态
        s = meta["status"]
        if s == "ok":
            print(f"  ✓ {meta['scene']:30s}  "
                  f"{meta['n_frames']:3d} frames  "
                  f"{meta['time_sec']:.1f}s  "
                  f"{meta['peak_gpu_gb']:.1f} GB")
        else:
            print(f"  ✗ {meta['scene']:30s}  [{s}]  "
                  f"{meta.get('error', '')[:80]}")

    # ---- 写 summary.csv ----
    import csv
    csv_path = output_root / "summary.csv"
    fields = ["scene", "status", "n_frames", "time_sec", "peak_gpu_gb", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in all_meta:
            w.writerow({k: m.get(k, "") for k in fields})

    n_ok = sum(1 for m in all_meta if m["status"] == "ok")
    print(f"\n[done] {n_ok}/{len(all_meta)} succeeded.")
    print(f"[done] summary -> {csv_path}")


if __name__ == "__main__":
    main()
