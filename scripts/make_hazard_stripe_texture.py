from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw


def parse_color(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError("color must be RRGGBB or #RRGGBB")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def make_hazard_stripes(
    size: int,
    stripe_width: int,
    angle_degrees: float,
    yellow: tuple[int, int, int],
    black: tuple[int, int, int],
) -> Image.Image:
    canvas_size = int(math.ceil(size * math.sqrt(2) * 2))
    image = Image.new("RGB", (canvas_size, canvas_size), yellow)
    draw = ImageDraw.Draw(image)

    spacing = stripe_width * 2
    for x in range(-canvas_size, canvas_size * 2, spacing):
        polygon = [
            (x, 0),
            (x + stripe_width, 0),
            (x + stripe_width + canvas_size, canvas_size),
            (x + canvas_size, canvas_size),
        ]
        draw.polygon(polygon, fill=black)

    rotated = image.rotate(angle_degrees, resample=Image.Resampling.BICUBIC, expand=False)
    left = (canvas_size - size) // 2
    top = (canvas_size - size) // 2
    return rotated.crop((left, top, left + size, top + size))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a printable black/yellow hazard-stripe patch texture.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--stripe_width", type=int, default=24)
    parser.add_argument("--angle_degrees", type=float, default=0.0)
    parser.add_argument("--yellow", type=parse_color, default=parse_color("ffe500"))
    parser.add_argument("--black", type=parse_color, default=parse_color("050505"))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image = make_hazard_stripes(
        size=args.size,
        stripe_width=args.stripe_width,
        angle_degrees=args.angle_degrees,
        yellow=args.yellow,
        black=args.black,
    )
    image.save(out)
    print(out)


if __name__ == "__main__":
    main()
