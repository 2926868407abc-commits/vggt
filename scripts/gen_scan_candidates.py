"""Stage 2 setup: enumerate the 8x8 position grid on every qualified plane.

Per the revised plan section 6.1 the area is one absolute physical value shared by
all planes, so a plane cannot win by hosting a bigger patch. The manual monitor
patch (0.263 m², 74% of its own plane) stays a separate baseline row and does not
take part in the scan.

Section 6.1 also requires at least 16 valid candidates per plane for a heat map to
mean anything, so that is checked here rather than after spending GPU time: a centre
is only valid when the whole rectangle lies inside the plane's occupied region, not
merely inside its bounding box -- an L-shaped desk should not accept a patch hanging
over the notch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VG = Path(__file__).resolve().parents[1]
PL = VG / "outputs/candidate_planes"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    ap.add_argument("--planes_dir", type=Path, default=PL)
    ap.add_argument(
        "--gt_dir",
        type=Path,
        default=VG / "outputs/tum_gt_point_track",
    )
    ap.add_argument("--area", type=float, default=0.05)
    ap.add_argument("--aspect", type=float, default=1.5534,
                    help="w/h, kept at the existing patch's ratio")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--cell", type=float, default=0.05,
                    help="occupancy cell size, matching the extractor")
    ap.add_argument("--min_fill", type=float, default=0.85,
                    help="fraction of the rectangle that must sit on occupied surface")
    ap.add_argument("--min_valid", type=int, default=16)
    ap.add_argument("--min_visible_frames", type=int, default=6)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    scene = cli.scene
    meta = json.loads(
        (cli.planes_dir / f"{scene}_planes.json").read_text(encoding="utf-8")
    )
    npz = np.load(cli.planes_dir / f"{scene}_planes.npz", allow_pickle=True)
    gt_c2w = npz["gt_c2w"]

    w = float(np.sqrt(cli.area * cli.aspect))
    h = float(np.sqrt(cli.area / cli.aspect))
    print(f"{scene}\n面积 {cli.area} m² -> {w:.3f} x {h:.3f} m  (长宽比 {cli.aspect})")
    print(f"网格 {cli.grid}x{cli.grid}，矩形需 {cli.min_fill:.0%} 落在实际表面上\n")

    # the occupied cells per plane, rebuilt from the stored point set
    gt = np.load(cli.gt_dir / f"{scene}_gt.npz", allow_pickle=True)
    world = gt["point_map"][gt["point_valid"]]
    keys = np.floor(world / 0.02).astype(np.int64)
    _, uniq = np.unique(keys, axis=0, return_index=True)
    cloud = world[np.sort(uniq)]

    out = []
    print(f"{'平面':>4}{'面积m²':>9}{'可见帧':>7}{'有效位置':>9}{'占比':>8}  说明")
    print("-" * 62)
    for pi, c in enumerate(meta["candidates"]):
        if c["visible_frames"] < cli.min_visible_frames:
            print(f"{pi:>4}{c['area_m2']:>9.3f}{c['visible_frames']:>7}"
                  f"{'—':>9}{'—':>8}  可见帧不足，跳过")
            continue
        centre = np.asarray(c["centre"])
        nrm = np.asarray(c["normal"])
        u, v = np.asarray(c["u"]), np.asarray(c["v"])
        lo, hi = np.asarray(c["uv_min"]), np.asarray(c["uv_max"])
        ext = hi - lo
        if ext[0] < w or ext[1] < h:
            print(f"{pi:>4}{c['area_m2']:>9.3f}{c['visible_frames']:>7}"
                  f"{'—':>9}{'—':>8}  放不下 {w:.2f}x{h:.2f}，跳过")
            continue

        rel = cloud - centre
        on = np.abs(rel @ nrm) < 0.02
        uu, vv = rel @ u, rel @ v
        inplane = on & (uu >= lo[0]) & (uu <= hi[0]) & (vv >= lo[1]) & (vv <= hi[1])
        occ = set(map(tuple, np.floor(
            np.stack([uu[inplane], vv[inplane]], -1) / cli.cell).astype(np.int64)))

        cus = np.linspace(lo[0] + w / 2, hi[0] - w / 2, cli.grid)
        cvs = np.linspace(lo[1] + h / 2, hi[1] - h / 2, cli.grid)
        valid = []
        for iu, cu in enumerate(cus):
            for iv, cv in enumerate(cvs):
                us = np.arange(cu - w / 2, cu + w / 2, cli.cell)
                vs = np.arange(cv - h / 2, cv + h / 2, cli.cell)
                if len(us) == 0 or len(vs) == 0:
                    continue
                gu, gv = np.meshgrid(us, vs, indexing="ij")
                cells = np.stack([np.floor(gu.ravel() / cli.cell),
                                  np.floor(gv.ravel() / cli.cell)], -1).astype(np.int64)
                fill = np.mean([tuple(k) in occ for k in cells])
                if fill < cli.min_fill:
                    continue
                # four world corners, ordered like the manual quad
                corners = np.stack([
                    centre + (cu - w / 2) * u + (cv - h / 2) * v,
                    centre + (cu + w / 2) * u + (cv - h / 2) * v,
                    centre + (cu + w / 2) * u + (cv + h / 2) * v,
                    centre + (cu - w / 2) * u + (cv + h / 2) * v,
                ])
                valid.append({"plane": pi, "iu": iu, "iv": iv,
                              "cu": float(cu), "cv": float(cv),
                              "fill": float(fill),
                              "quad": corners.reshape(-1).tolist()})
        frac = len(valid) / (cli.grid ** 2)
        note = "✅" if len(valid) >= cli.min_valid else f"< {cli.min_valid}，热力图不可信"
        print(f"{pi:>4}{c['area_m2']:>9.3f}{c['visible_frames']:>7}"
              f"{len(valid):>9}{frac:>8.0%}  {note}")
        if len(valid) >= cli.min_valid:
            out.extend(valid)

    dst = Path(cli.out or (cli.planes_dir / f"{scene}_scan_a{cli.area}.json"))
    dst.write_text(json.dumps({"scene": scene, "area_m2": cli.area,
                               "w": w, "h": h, "grid": cli.grid,
                               "candidates": out}, indent=1), encoding="utf-8")
    planes = sorted({c["plane"] for c in out})
    print(f"\n合格平面 {planes}，候选总数 {len(out)}")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
