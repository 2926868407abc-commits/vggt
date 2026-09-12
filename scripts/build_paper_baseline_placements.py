"""Freeze paper-specific VGGT baseline placements before texture training.

This stage deliberately never reads an attack score.  It translates each source
paper's placement rule into the already extracted TUM RGB-D planes:

* SPOO: optimise a rectangle inside the full target carrier (starts full size).
* APARATE: fixed patch at the target carrier centre, linear scale 0.2.
* 3D2Fool: texture the full visible target surface (planar TUM adaptation).
* Passive VO: fixed partition/screen, scaled to the paper's real-data patch FOV.

The output JSON is immutable experiment input, not a leaderboard of locations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VG = Path(__file__).resolve().parents[1]
PLANES = VG / "outputs/candidate_planes"


def plane_quad(plane: dict, *, scale_u: float = 1.0, scale_v: float = 1.0) -> np.ndarray:
    centre = np.asarray(plane["centre"], dtype=np.float64)
    u = np.asarray(plane["u"], dtype=np.float64)
    v = np.asarray(plane["v"], dtype=np.float64)
    lo = np.asarray(plane["uv_min"], dtype=np.float64)
    hi = np.asarray(plane["uv_max"], dtype=np.float64)
    local_centre = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * np.asarray([scale_u, scale_v], dtype=np.float64)
    u0, v0 = local_centre - half
    u1, v1 = local_centre + half
    return np.stack(
        [
            centre + u0 * u + v0 * v,
            centre + u1 * u + v0 * v,
            centre + u1 * u + v1 * v,
            centre + u0 * u + v1 * v,
        ]
    )


def quad_string(quad: np.ndarray) -> str:
    return ",".join(f"{value:.9f}" for value in np.asarray(quad).reshape(-1))


def quad_dimensions(quad: np.ndarray) -> tuple[float, float, float]:
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 3)
    width = float(np.linalg.norm(quad[1] - quad[0]))
    height = float(np.linalg.norm(quad[3] - quad[0]))
    return width, height, width * height


def find_plane(planes: list[dict], index: int) -> dict:
    if index < 0 or index >= len(planes):
        raise IndexError(f"Plane index {index} is outside 0..{len(planes) - 1}")
    return planes[index]


def profile(
    name: str,
    method: str,
    carrier_name: str,
    carrier_index: int,
    patch_quad: np.ndarray,
    focus_quad: np.ndarray | None,
    placement_rule: str,
    source_fidelity: str,
) -> dict:
    width, height, area = quad_dimensions(patch_quad)
    return {
        "name": name,
        "attack_loss": method,
        "carrier": carrier_name,
        "source_plane_index": int(carrier_index),
        "placement_rule": placement_rule,
        "source_fidelity": source_fidelity,
        "explicit_quad_world": quad_string(patch_quad),
        "baseline_focus_quad_world": quad_string(focus_quad) if focus_quad is not None else "",
        "patch_width_m": width,
        "patch_height_m": height,
        "outer_quad_area_m2": area,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    parser.add_argument("--planes", type=Path, default=None)
    parser.add_argument("--monitor_plane", type=int, default=2)
    parser.add_argument("--table_plane", type=int, default=1)
    parser.add_argument("--partition_plane", type=int, default=7)
    parser.add_argument("--aparate_scale", type=float, default=0.2)
    parser.add_argument("--vo_fov", type=float, default=0.088)
    parser.add_argument(
        "--vo_partition_scale",
        type=float,
        default=0.90,
        help="linear fraction of the partition extent; preflight records actual FOV",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.planes or PLANES / f"{args.scene}_planes.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    planes = data["candidates"]

    monitor = find_plane(planes, args.monitor_plane)
    table = find_plane(planes, args.table_plane)
    partition = find_plane(planes, args.partition_plane)
    carriers = {
        "monitor": (args.monitor_plane, monitor),
        "table": (args.table_plane, table),
        "partition": (args.partition_plane, partition),
    }

    profiles: list[dict] = []
    for carrier_name in ("monitor", "table", "partition"):
        plane_index, carrier = carriers[carrier_name]
        full = plane_quad(carrier)
        row = profile(
            f"spoo_{carrier_name}",
            "baseline_spoo_depth",
            carrier_name,
            plane_index,
            full,
            full,
            "SPOO learns [l,r,t,b] inside the full frozen carrier and stops at area ratio 1/9.",
            "Paper region optimization; TUM carrier plane replaces the paper's segmented object.",
        )
        row["spoo_target_area_fraction"] = 1.0 / 9.0
        profiles.append(row)

    for carrier_name in ("monitor", "table"):
        plane_index, carrier = carriers[carrier_name]
        full = plane_quad(carrier)
        patch = plane_quad(
            carrier,
            scale_u=args.aparate_scale,
            scale_v=args.aparate_scale,
        )
        row = profile(
            f"aparate_{carrier_name}",
            "baseline_aparate_depth",
            carrier_name,
            plane_index,
            patch,
            full,
            f"APARATE centre placement with linear object scale {args.aparate_scale:g}.",
            "Paper loss/centre/scale; frozen RGB-D carrier mask replaces YOLO object detection.",
        )
        row["aparate_scale"] = float(args.aparate_scale)
        profiles.append(row)

    full_monitor = plane_quad(monitor)
    row = profile(
        "3d2fool_monitor_visible_surface",
        "baseline_3d2fool_depth",
        "monitor",
        args.monitor_plane,
        full_monitor,
        None,
        "Texture the complete visible monitor carrier.",
        "Visible-surface planar adaptation; TUM has no complete 3-D vehicle mesh.",
    )
    row.update({"texture_seed_size": 128, "texture_repeats": 6, "converted_size": 512})
    profiles.append(row)

    vo_quad = plane_quad(
        partition,
        scale_u=args.vo_partition_scale,
        scale_v=args.vo_partition_scale,
    )
    row = profile(
        "passive_vo_partition",
        "baseline_vo_mprms",
        "partition",
        args.partition_plane,
        vo_quad,
        None,
        "Fixed planar screen/partition selected before optimization.",
        "Paper MPRMS and fixed-screen placement; linear scale is frozen before training.",
    )
    row["target_projected_fov"] = float(args.vo_fov)
    row["partition_linear_scale"] = float(args.vo_partition_scale)
    profiles.append(row)

    output = args.out or VG / "configs" / f"paper_baseline_placements_{args.scene}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene": args.scene,
        "source_planes": str(source.resolve()),
        "selection_uses_attack_scores": False,
        "plane_labels": {
            "monitor": args.monitor_plane,
            "table": args.table_plane,
            "partition": args.partition_plane,
        },
        "profiles": profiles,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in profiles:
        print(
            f"{row['name']:<38} {row['attack_loss']:<26} "
            f"{row['patch_width_m']:.3f}x{row['patch_height_m']:.3f}m"
        )
    print(f"-> {output}")


if __name__ == "__main__":
    main()
