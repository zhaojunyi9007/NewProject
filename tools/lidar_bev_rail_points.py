#!/usr/bin/env python3
"""Export rail_probability BEV cells as optimizer-friendly 3D rail samples."""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np


def export_lidar_bev_rail_points(
    npz_path: str,
    out_path: str,
    min_prob: float = 0.15,
    stride_cells: int = 2,
    max_points: int = 8000,
    reference_z: float = 0.0,
) -> int:
    z = np.load(npz_path)
    if "rail_probability" not in z.files:
        return 0
    rail = np.asarray(z["rail_probability"], dtype=np.float32)
    if rail.ndim != 2 or rail.size == 0:
        return 0
    x0 = float(np.asarray(z["bev_xmin"]).reshape(-1)[0]) if "bev_xmin" in z.files else 0.0
    y0 = float(np.asarray(z["bev_ymin"]).reshape(-1)[0]) if "bev_ymin" in z.files else 0.0
    res = float(np.asarray(z["bev_resolution"]).reshape(-1)[0]) if "bev_resolution" in z.files else 0.2
    stride = max(1, int(stride_cells))

    yy, xx = np.where(rail >= float(min_prob))
    if yy.size == 0:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# x y z confidence\n")
        return 0
    keep = ((yy % stride) == 0) & ((xx % stride) == 0)
    yy, xx = yy[keep], xx[keep]
    conf = rail[yy, xx]
    if yy.size > int(max_points) > 0:
        order = np.argsort(conf)[::-1][: int(max_points)]
        yy, xx, conf = yy[order], xx[order], conf[order]

    xs = x0 + (xx.astype(np.float64) + 0.5) * res
    ys = y0 + (yy.astype(np.float64) + 0.5) * res
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# x y z confidence\n")
        for x, y, c in zip(xs, ys, conf):
            f.write(f"{float(x):.6f} {float(y):.6f} {float(reference_z):.6f} {float(c):.6f}\n")
    return int(len(conf))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path")
    ap.add_argument("out_path")
    ap.add_argument("--min-prob", type=float, default=0.15)
    ap.add_argument("--stride-cells", type=int, default=2)
    ap.add_argument("--max-points", type=int, default=8000)
    ap.add_argument("--reference-z", type=float, default=0.0)
    args = ap.parse_args(argv)
    n = export_lidar_bev_rail_points(
        args.npz_path,
        args.out_path,
        min_prob=args.min_prob,
        stride_cells=args.stride_cells,
        max_points=args.max_points,
        reference_z=args.reference_z,
    )
    print(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
