"""Detect a monitor/screen quadrilateral in a TUM frame for physical patch placement.

This is a small helper for the AoR-style screen-carrier experiment. It detects
dark rectangular regions in the VGGT-preprocessed frame, saves a visualization
with candidate boxes, and writes an env file with the best normalized quad:

    SCREEN_INNER_QUAD_XY=x0,y0,x1,y1,x2,y2,x3,y3

The output is intentionally human-checkable. Black clothes, monitor bezels, and
depth noise can fool pure automation, so the candidate sheet should be checked
before launching a long run.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def find_images(root: Path) -> list[Path]:
    for subdir in ("images", "rgb_90", "rgb"):
        image_dir = root / subdir
        if image_dir.exists():
            paths = sorted(
                list(image_dir.glob("*.png"))
                + list(image_dir.glob("*.jpg"))
                + list(image_dir.glob("*.jpeg"))
            )
            if paths:
                return paths
    return sorted(
        list(root.glob("*.png"))
        + list(root.glob("*.jpg"))
        + list(root.glob("*.jpeg"))
    )


def preprocess_like_vggt(image_path: Path, target_size: int = 518) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    new_width = target_size
    new_height = round(height * (new_width / width) / 14) * 14
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    if new_height > target_size:
        start_y = (new_height - target_size) // 2
        image = image.crop((0, start_y, new_width, start_y + target_size))
    return image


def load_manifest_frame(tum_root: Path, manifest_path: Path, scene: str, frame_rank: int) -> Path:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if scene in manifest:
            names = manifest[scene].get("image_names", [])
            if names:
                frame_rank = min(max(frame_rank, 0), len(names) - 1)
                for subdir in ("images", "rgb_90", "rgb"):
                    path = tum_root / scene / subdir / names[frame_rank]
                    if path.exists():
                        return path
    images = find_images(tum_root / scene)
    if not images:
        raise FileNotFoundError(f"No images found for {tum_root / scene}")
    frame_rank = min(max(frame_rank, 0), len(images) - 1)
    return images[frame_rank]


def connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[tuple[int, int, int, int, int]] = []
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if seen[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
        seen[start_y, start_x] = True
        count = 0
        x0 = x1 = start_x
        y0 = y1 = start_y
        while queue:
            y, x = queue.popleft()
            count += 1
            x0 = min(x0, x)
            x1 = max(x1, x)
            y0 = min(y0, y)
            y1 = max(y1, y)
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        comps.append((x0, y0, x1 + 1, y1 + 1, count))
    return comps


def bbox_to_quad(x0: int, y0: int, x1: int, y1: int, width: int, height: int, shrink: float) -> np.ndarray:
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half_w = 0.5 * (x1 - x0) * shrink
    half_h = 0.5 * (y1 - y0) * shrink
    quad = np.asarray(
        [
            [cx - half_w, cy - half_h],
            [cx + half_w, cy - half_h],
            [cx + half_w, cy + half_h],
            [cx - half_w, cy + half_h],
        ],
        dtype=np.float64,
    )
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    return quad


def normalized_quad(quad: np.ndarray, width: int, height: int) -> list[float]:
    out = quad.copy()
    out[:, 0] /= max(1, width - 1)
    out[:, 1] /= max(1, height - 1)
    return [float(f"{value:.6f}") for value in out.reshape(-1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum_root", default="/mnt/data/wangqq/recons_eval/data/tum")
    parser.add_argument("--manifest", default="/mnt/data/wangqq/vggt/data/tum_dynamics_10frame_individual_scenes/tum10_frame_manifest.json")
    parser.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_static")
    parser.add_argument("--frame_rank", type=int, default=0)
    parser.add_argument("--out_dir", default="/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/monitor_quad_detection")
    parser.add_argument("--roi", default="0.38,0.26,0.78,0.68", help="normalized x0,y0,x1,y1 search ROI")
    parser.add_argument("--prefer", default="0.58,0.45", help="normalized preferred monitor center")
    parser.add_argument("--dark_threshold", type=float, default=72.0)
    parser.add_argument("--min_area", type=int, default=350)
    parser.add_argument("--shrink", type=float, default=0.88)
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()

    tum_root = Path(args.tum_root)
    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = load_manifest_frame(tum_root, manifest, args.scene, args.frame_rank)
    image = preprocess_like_vggt(image_path)
    arr = np.asarray(image, dtype=np.float32)
    height, width = arr.shape[:2]

    roi = [float(v) for v in args.roi.split(",")]
    rx0 = int(np.clip(roi[0], 0, 1) * (width - 1))
    ry0 = int(np.clip(roi[1], 0, 1) * (height - 1))
    rx1 = int(np.clip(roi[2], 0, 1) * (width - 1))
    ry1 = int(np.clip(roi[3], 0, 1) * (height - 1))
    prefer = [float(v) for v in args.prefer.split(",")]
    prefer_x = prefer[0] * (width - 1)
    prefer_y = prefer[1] * (height - 1)

    gray = arr.mean(axis=2)
    # Screens are dark and fairly low-saturation. The max-min gate removes many
    # colored clothes/posters while keeping black display regions.
    chroma = arr.max(axis=2) - arr.min(axis=2)
    mask = (gray < args.dark_threshold) & (chroma < 55.0)
    roi_mask = np.zeros((height, width), dtype=bool)
    roi_mask[ry0:ry1, rx0:rx1] = True
    mask &= roi_mask
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    mask = np.asarray(mask_img, dtype=np.uint8) > 0

    candidates = []
    for x0, y0, x1, y1, count in connected_components(mask):
        bw = x1 - x0
        bh = y1 - y0
        box_area = max(1, bw * bh)
        fill = count / box_area
        aspect = bw / max(1, bh)
        if count < args.min_area or bw < 18 or bh < 18:
            continue
        if aspect < 0.55 or aspect > 2.4:
            continue
        if fill < 0.35:
            continue
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        dist = ((cx - prefer_x) / width) ** 2 + ((cy - prefer_y) / height) ** 2
        darkness = 1.0 - float(gray[y0:y1, x0:x1].mean() / 255.0)
        area_score = min(1.0, box_area / (0.08 * width * height))
        score = 1.5 * darkness + 1.0 * fill + 0.8 * area_score - 2.5 * dist
        quad = bbox_to_quad(x0, y0, x1, y1, width, height, args.shrink)
        candidates.append(
            {
                "score": float(score),
                "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "area": int(count),
                "fill_ratio": float(fill),
                "aspect": float(aspect),
                "center_norm": [float(cx / (width - 1)), float(cy / (height - 1))],
                "quad_xy_normalized": normalized_quad(quad, width, height),
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    candidates = candidates[: args.top_k]
    if not candidates:
        raise RuntimeError("No monitor-like dark rectangle was detected. Adjust --roi/--dark_threshold.")

    vis = image.convert("RGBA")
    draw = ImageDraw.Draw(vis)
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange", "deepskyblue", "white"]
    draw.rectangle((rx0, ry0, rx1, ry1), outline="white", width=2)
    for idx, cand in enumerate(candidates):
        color = colors[idx % len(colors)]
        x0, y0, x1, y1 = cand["bbox_xyxy"]
        quad_vals = np.asarray(cand["quad_xy_normalized"], dtype=np.float64).reshape(4, 2)
        quad_px = quad_vals.copy()
        quad_px[:, 0] *= width - 1
        quad_px[:, 1] *= height - 1
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        draw.line([tuple(p) for p in quad_px] + [tuple(quad_px[0])], fill=color, width=4)
        draw.text((x0, max(0, y0 - 14)), f"{idx}: {cand['score']:.2f}", fill=color)

    result = {
        "source_image": str(image_path),
        "scene": args.scene,
        "frame_rank": args.frame_rank,
        "tensor_size": [width, height],
        "roi": roi,
        "best_quad_env": ",".join(f"{v:.6f}" for v in candidates[0]["quad_xy_normalized"]),
        "candidates": candidates,
    }
    json_path = out_dir / f"{args.scene}_frame{args.frame_rank:02d}_monitor_quad_candidates.json"
    png_path = out_dir / f"{args.scene}_frame{args.frame_rank:02d}_monitor_quad_candidates.png"
    env_path = out_dir / "best_monitor_quad.env"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    vis.convert("RGB").save(png_path)
    env_path.write_text(f"SCREEN_INNER_QUAD_XY={result['best_quad_env']}\n", encoding="utf-8")
    print(f"source: {image_path}")
    print(f"saved: {json_path}")
    print(f"saved: {png_path}")
    print(f"saved: {env_path}")
    print(f"SCREEN_INNER_QUAD_XY={result['best_quad_env']}")


if __name__ == "__main__":
    main()
