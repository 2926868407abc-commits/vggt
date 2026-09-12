"""Render the exact base/deployed initializers used by paper baseline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms.functional import to_pil_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack_vggt_geometry_tum10 import (
    initialize_texture,
    paper_base_texture_preview,
    prepare_texture_for_render,
)
from attack_vggt_new1 import set_random_seeds


PROFILES = (
    {
        "name": "spoo",
        "attack_loss": "baseline_spoo_depth",
        "seed": 17,
        "texture_size": 600,
        "texture_init": "image",
        "preserve_aspect": True,
    },
    {
        "name": "aparate",
        "attack_loss": "baseline_aparate_depth",
        "seed": 0,
        "texture_size": 64,
        "texture_init": "random",
        "preserve_aspect": False,
    },
    {
        "name": "3d2fool",
        "attack_loss": "baseline_3d2fool_depth",
        "seed": 0,
        "texture_size": 128,
        "texture_init": "random",
        "preserve_aspect": False,
    },
    {
        "name": "passive_vo",
        "attack_loss": "baseline_vo_mprms",
        "seed": 0,
        "texture_size": 64,
        "texture_init": "random",
        "preserve_aspect": False,
    },
)


def initializer_args(profile: dict, warning: Path) -> argparse.Namespace:
    return argparse.Namespace(
        attack_loss=profile["attack_loss"],
        texture_size=profile["texture_size"],
        texture_init=profile["texture_init"],
        texture_init_image=str(warning) if profile["texture_init"] == "image" else None,
        texture_init_preserve_aspect=profile["preserve_aspect"],
        print_min=0.0,
        print_max=1.0,
        baseline_texture_repeats=6,
        baseline_converted_texture_size=4096,
        baseline_texture_upsample_factor=8,
        baseline_vo_blur_kernel=9,
        physical_eot=False,
    )


def contact_sheet(rows: list[tuple[str, Image.Image, Image.Image]]) -> Image.Image:
    cell_w, cell_h, label_h = 300, 230, 26
    sheet = Image.new("RGB", (cell_w * 2, (cell_h + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (name, base, deployed) in enumerate(rows):
        y = row * (cell_h + label_h)
        for col, (kind, image) in enumerate((("base variable", base), ("deployed preview", deployed))):
            fitted = ImageOps.contain(image.convert("RGB"), (cell_w - 16, cell_h - 16))
            x0 = col * cell_w + (cell_w - fitted.width) // 2
            y0 = y + label_h + (cell_h - fitted.height) // 2
            sheet.paste(fitted, (x0, y0))
            draw.text((col * cell_w + 8, y + 6), f"{name}: {kind}", fill="black")
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warning", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"warning_source": str(args.warning.resolve()), "profiles": []}
    rows = []
    for profile in PROFILES:
        set_random_seeds(int(profile["seed"]))
        run_args = initializer_args(profile, args.warning)
        texture = initialize_texture(run_args, torch.device("cpu"), requires_grad=False)
        base = to_pil_image(paper_base_texture_preview(texture, run_args))
        deployed_tensor = prepare_texture_for_render(
            texture, run_args, training=False, n_frames=1
        )[0].detach().cpu()
        deployed = to_pil_image(deployed_tensor)
        base_path = args.out_dir / f"{profile['name']}_initial_patch.png"
        deployed_path = args.out_dir / f"{profile['name']}_initial_deployed_patch.png"
        base.save(base_path)
        deployed.save(deployed_path)
        rows.append((profile["name"], base, deployed))
        manifest["profiles"].append(
            {
                **profile,
                "raw_tensor_shape": list(texture.shape),
                "base_image_size": list(base.size),
                "deployed_image_size": list(deployed.size),
                "base": str(base_path.resolve()),
                "deployed": str(deployed_path.resolve()),
            }
        )

    contact_sheet(rows).save(args.out_dir / "all_initializers_contact_sheet.png")
    (args.out_dir / "initializers_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.out_dir.resolve())


if __name__ == "__main__":
    main()
