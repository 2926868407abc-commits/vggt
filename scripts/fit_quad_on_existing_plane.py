#!/usr/bin/env python3
"""Move a rendered quad within its existing stable 3-D carrier plane.

The source image quad is the projection of ``world_quad`` in an anchor frame.
The target image quad is converted to source-quad coordinates by a homography,
then bilinearly interpolated between the four original world corners.  This
changes the attached rectangle without fitting a new plane from unreliable LCD
depth samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_values(text: str, count: int) -> np.ndarray:
    values = np.asarray([float(value) for value in text.split(",")], dtype=np.float64)
    if values.size != count:
        raise ValueError(f"expected {count} comma-separated values, got {values.size}")
    return values


def bilinear_world(corners: np.ndarray, uv: np.ndarray) -> np.ndarray:
    u = uv[:, 0:1]
    v = uv[:, 1:2]
    tl, tr, br, bl = corners
    return ((1.0 - u) * (1.0 - v) * tl
            + u * (1.0 - v) * tr
            + u * v * br
            + (1.0 - u) * v * bl)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_quad", required=True)
    parser.add_argument("--source_projected", required=True)
    parser.add_argument("--target_projected", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    world = parse_values(Path(args.world_quad).read_text().strip(), 12).reshape(4, 3)
    source = parse_values(args.source_projected, 8).reshape(4, 2).astype(np.float32)
    target = parse_values(args.target_projected, 8).reshape(4, 2).astype(np.float32)
    unit = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    image_to_unit = cv2.getPerspectiveTransform(source, unit)
    uv = cv2.perspectiveTransform(target[None, ...], image_to_unit)[0].astype(np.float64)
    adjusted = bilinear_world(world, uv)
    Path(args.output).write_text(
        ",".join(f"{value:.9f}" for value in adjusted.reshape(-1)) + "\n",
        encoding="utf-8",
    )
    print("source", source.tolist())
    print("target", target.tolist())
    print("plane uv", uv.tolist())
    print("world", adjusted.tolist())


if __name__ == "__main__":
    main()
