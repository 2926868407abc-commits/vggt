"""Visualize GT-geometry-aware TUM-10 planar patch projections.

This script reads geometry-aware attack outputs written by
attack_vggt_geometry_tum10.py and renders:

* the learned patch texture
* per-frame VGGT-preprocessed images with the projected patch composited
* per-frame outline-only overlays
* per-sequence contact sheets

It does not run VGGT and does not require a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def preprocess_like_vggt(image_path: str, target_size: int = 518) -> Image.Image:
    image = Image.open(image_path)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    image = image.convert("RGB")

    width, height = image.size
    new_width = target_size
    new_height = round(height * (new_width / width) / 14) * 14
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    if new_height > target_size:
        start_y = (new_height - target_size) // 2
        image = image.crop((0, start_y, new_width, start_y + target_size))
    return image


def perspective_coefficients(dst: np.ndarray, src: np.ndarray) -> list[float]:
    """Return PIL PERSPECTIVE coefficients mapping output xy to source uv."""
    rows = []
    rhs = []
    for (x, y), (u, v) in zip(dst, src):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        rhs.append(v)
    coeffs = np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64))
    return coeffs.astype(float).tolist()


def warp_texture_to_quad(texture: Image.Image, quad: np.ndarray, image_size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    tex_w, tex_h = texture.size
    src = np.asarray(
        [
            [0.0, 0.0],
            [tex_w - 1.0, 0.0],
            [tex_w - 1.0, tex_h - 1.0],
            [0.0, tex_h - 1.0],
        ],
        dtype=np.float64,
    )
    coeffs = perspective_coefficients(quad.astype(np.float64), src)
    warped = texture.transform(
        image_size,
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )
    white = Image.new("L", texture.size, 255)
    mask = white.transform(
        image_size,
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=0,
    )
    return warped.convert("RGBA"), mask


def alpha_mask(mask: Image.Image, alpha: float) -> Image.Image:
    arr = np.asarray(mask, dtype=np.float32)
    arr = np.clip(arr * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def mask_array_to_image(mask: np.ndarray, image_size: tuple[int, int]) -> Image.Image:
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[0]
    mask = np.clip(mask, 0.0, 1.0)
    mask_image = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L")
    if mask_image.size != image_size:
        mask_image = mask_image.resize(image_size, Image.Resampling.NEAREST)
    return mask_image


def draw_label(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    pad = 5
    rect = (8, 8, 8 + bbox[2] - bbox[0] + pad * 2, 8 + bbox[3] - bbox[1] + pad * 2)
    draw.rectangle(rect, fill=(0, 0, 0, 170))
    draw.text((8 + pad, 8 + pad), text, fill=(255, 255, 255, 255), font=font)


def composite_patch(
    base: Image.Image,
    texture: Image.Image,
    quad: np.ndarray,
    alpha: float,
    label: str,
    visibility_mask: np.ndarray | None = None,
) -> Image.Image:
    out = base.convert("RGBA")
    warped, quad_mask = warp_texture_to_quad(texture, quad, out.size)
    mask = quad_mask
    if visibility_mask is not None:
        visible = mask_array_to_image(visibility_mask, out.size)
        mask = Image.fromarray(
            np.minimum(np.asarray(quad_mask, dtype=np.uint8), np.asarray(visible, dtype=np.uint8)),
            mode="L",
        )
    out = Image.composite(warped, out, alpha_mask(mask, alpha))
    draw = ImageDraw.Draw(out)
    points = [tuple(map(float, p)) for p in quad.tolist()]
    draw.line(points + [points[0]], fill=(255, 40, 40, 255), width=3)
    draw_label(out, label)
    return out.convert("RGB")


def outline_only(base: Image.Image, quad: np.ndarray, label: str) -> Image.Image:
    out = base.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    points = [tuple(map(float, p)) for p in quad.tolist()]
    draw.polygon(points, fill=(255, 40, 40, 55))
    draw.line(points + [points[0]], fill=(255, 40, 40, 255), width=3)
    out = Image.alpha_composite(out, overlay)
    draw_label(out, label)
    return out.convert("RGB")


def contact_sheet(images: list[Image.Image], columns: int, thumb_width: int) -> Image.Image | None:
    if not images:
        return None
    thumbs = []
    for image in images:
        thumb = image.copy()
        ratio = thumb_width / thumb.width
        thumb = thumb.resize((thumb_width, max(1, round(thumb.height * ratio))), Image.Resampling.BICUBIC)
        thumbs.append(thumb)
    rows = int(np.ceil(len(thumbs) / columns))
    cell_w = max(img.width for img in thumbs)
    cell_h = max(img.height for img in thumbs)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (245, 245, 245))
    for i, thumb in enumerate(thumbs):
        x = (i % columns) * cell_w
        y = (i // columns) * cell_h
        sheet.paste(thumb, (x, y))
    return sheet


def parse_frame_selection(selection: str, n_frames: int) -> list[int]:
    if selection == "all":
        return list(range(n_frames))
    indices = []
    for item in selection.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            idx += n_frames
        if idx < 0 or idx >= n_frames:
            raise ValueError(f"Frame index {item} out of range for {n_frames} frames")
        indices.append(idx)
    return sorted(set(indices))


def visualize_sequence(
    seq_dir: Path,
    texture: Image.Image,
    out_root: Path,
    frame_selection: str,
    alpha: float,
    contact_columns: int,
    thumb_width: int,
) -> int:
    summary_path = seq_dir / "attack_summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    image_paths = summary["image_paths"]
    frame_indices = summary["frame_indices"]
    corners = summary["geometry"]["projected_corners"]
    coverages = summary["geometry"]["mask_coverage_per_frame"]
    mask_path = seq_dir / summary.get("outputs", {}).get("visibility_masks", "geometry_visibility_masks.npz")
    masks = None
    if mask_path.exists():
        with np.load(mask_path) as data:
            masks = np.asarray(data["masks"], dtype=np.float32)
    selected = parse_frame_selection(frame_selection, len(image_paths))

    seq_out = out_root / seq_dir.name
    overlay_dir = seq_out / "overlay"
    outline_dir = seq_out / "outline"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    outline_dir.mkdir(parents=True, exist_ok=True)

    sheet_images = []
    for i in selected:
        base = preprocess_like_vggt(image_paths[i])
        quad = np.asarray(corners[i], dtype=np.float64)
        label = f"{seq_dir.name} | frame={frame_indices[i]} | cov={coverages[i]:.4f}"

        visibility_mask = masks[i] if masks is not None else None
        over = composite_patch(base, texture, quad, alpha, label, visibility_mask)
        outline = outline_only(base, quad, label)
        stem = f"{i:02d}_frame_{int(frame_indices[i]):06d}"
        over.save(overlay_dir / f"{stem}_patch.png")
        outline.save(outline_dir / f"{stem}_outline.png")
        sheet_images.append(over)

    sheet = contact_sheet(sheet_images, contact_columns, thumb_width)
    if sheet is not None:
        sheet.save(seq_out / "contact_sheet_patch.png")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry_output_root",
        default="/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/tum10_vggt_pointmap_geometry_feature_l3",
        help="Directory containing geometry_patch/ and per-sequence attack_summary.json files.",
    )
    parser.add_argument(
        "--out_dir",
        default="/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/visualizations",
    )
    parser.add_argument("--scene_pattern", default="rgbd_dataset_freiburg3_*")
    parser.add_argument("--frames", default="all", help="'all' or comma-separated frame positions, e.g. 0,4,9")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--contact_columns", type=int, default=5)
    parser.add_argument("--thumb_width", type=int, default=260)
    args = parser.parse_args()

    root = Path(args.geometry_output_root)
    out_dir = Path(args.out_dir)
    texture_path = root / "geometry_patch" / "geometry_patch_texture.png"
    if not texture_path.exists():
        raise FileNotFoundError(f"Missing texture image: {texture_path}")

    texture = Image.open(texture_path).convert("RGB")
    out_dir.mkdir(parents=True, exist_ok=True)
    texture.save(out_dir / "geometry_patch_texture.png")

    scene_dirs = sorted(
        path for path in root.glob(args.scene_pattern) if (path / "attack_summary.json").exists()
    )
    if not scene_dirs:
        raise ValueError(f"No attack summaries matched {root / args.scene_pattern}")

    total = 0
    for seq_dir in scene_dirs:
        count = visualize_sequence(
            seq_dir,
            texture,
            out_dir,
            args.frames,
            args.alpha,
            args.contact_columns,
            args.thumb_width,
        )
        total += count
        print(f"[visualized] {seq_dir.name}: {count} frame(s)")

    print(f"[done] wrote {total} frame visualization(s) -> {out_dir}")


if __name__ == "__main__":
    main()
