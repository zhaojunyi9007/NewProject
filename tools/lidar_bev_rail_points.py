#!/usr/bin/env python3
"""Export refined LiDAR rail BEV cells as optimizer-friendly 3D rail samples."""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional

import numpy as np


def _load_scalar(z: Any, key: str, default: float) -> float:
    return float(np.asarray(z[key]).reshape(-1)[0]) if key in z.files else float(default)


def _load_oracle_rail(oracle_npz_path: Optional[str]) -> Optional[np.ndarray]:
    if not oracle_npz_path:
        return None
    try:
        z = np.load(oracle_npz_path)
    except OSError:
        return None
    for key in ("rail_from_label_track", "rail_from_likelihood", "rail"):
        if key in z.files:
            arr = np.asarray(z[key], dtype=np.float32)
            if arr.ndim == 2 and arr.size:
                return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return None


def _resize_like(src: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if src.shape == shape:
        return src.astype(np.float32, copy=False)
    import cv2

    return cv2.resize(src.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def refine_lidar_rail_probability(
    rail: np.ndarray,
    min_prob: float = 0.15,
    oracle_rail: Optional[np.ndarray] = None,
    oracle_overlap_dilate_cells: int = 3,
    min_component_cells: int = 20,
    debug_out: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Denoise LiDAR rail probability using thresholding, components, and optional oracle overlap."""
    import cv2

    arr = np.nan_to_num(np.asarray(rail, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    raw_nonzero = float((arr >= float(min_prob)).mean()) if arr.size else 0.0

    mask = arr >= float(min_prob)
    oracle_used = False
    oracle_nonzero = 0.0
    if oracle_rail is not None and np.asarray(oracle_rail).size:
        oracle = _resize_like(np.asarray(oracle_rail, dtype=np.float32), arr.shape)
        oracle = np.nan_to_num(oracle, nan=0.0, posinf=1.0, neginf=0.0)
        oracle_mask = oracle > 1e-4
        oracle_nonzero = float(oracle_mask.mean()) if oracle_mask.size else 0.0
        if oracle_mask.any():
            oracle_prob_floor = max(float(min_prob), 0.35)
            mask &= arr >= oracle_prob_floor
            dilate = max(0, int(oracle_overlap_dilate_cells))
            if dilate > 0:
                k = 2 * dilate + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                oracle_mask = cv2.dilate(oracle_mask.astype(np.uint8), kernel, iterations=1) > 0
            mask &= oracle_mask
            oracle_used = True

    refined = np.zeros_like(arr, dtype=np.float32)
    component_count = 0
    kept_components = 0
    if mask.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        component_count = max(0, int(n) - 1)
        min_cells = max(1, int(min_component_cells))
        for lab in range(1, n):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_cells:
                continue
            comp = labels == lab
            refined[comp] = arr[comp]
            kept_components += 1

    refined_nonzero = float((refined > 1e-4).mean()) if refined.size else 0.0
    fallback_used = False
    if refined_nonzero <= 0.0 and raw_nonzero > 0.0:
        # Avoid making the rail term disappear when oracle/geometry refinement is over-strict
        # or when the image/LiDAR BEV grids do not overlap well enough yet.
        refined = np.where(arr >= float(min_prob), arr, 0.0).astype(np.float32)
        refined_nonzero = float((refined > 1e-4).mean()) if refined.size else 0.0
        fallback_used = True
    if debug_out is not None:
        debug_out.update(
            {
                "lidar_rail_raw_nonzero_ratio": raw_nonzero,
                "lidar_rail_refined_nonzero_ratio": refined_nonzero,
                "lidar_rail_oracle_used": bool(oracle_used),
                "lidar_rail_oracle_nonzero_ratio": oracle_nonzero,
                "lidar_rail_component_count": float(component_count),
                "lidar_rail_kept_component_count": float(kept_components),
                "lidar_rail_refine_fallback_used": bool(fallback_used),
                "lidar_rail_min_prob": float(min_prob),
                "lidar_rail_min_component_cells": float(max(1, int(min_component_cells))),
            }
        )
    return refined


def export_lidar_bev_rail_points(
    npz_path: str,
    out_path: str,
    min_prob: float = 0.15,
    stride_cells: int = 2,
    max_points: int = 8000,
    reference_z: float = 0.0,
    oracle_npz_path: Optional[str] = None,
    oracle_overlap_dilate_cells: int = 3,
    min_component_cells: int = 1,
    debug_path: Optional[str] = None,
    refined_png_path: Optional[str] = None,
) -> int:
    z = np.load(npz_path)
    if "rail_probability_refined" in z.files:
        rail = np.asarray(z["rail_probability_refined"], dtype=np.float32)
        debug: dict[str, Any] = {"lidar_rail_refine_source": "rail_probability_refined"}
    elif "rail_probability" in z.files:
        debug = {"lidar_rail_refine_source": "rail_probability"}
        rail = refine_lidar_rail_probability(
            np.asarray(z["rail_probability"], dtype=np.float32),
            min_prob=min_prob,
            oracle_rail=_load_oracle_rail(oracle_npz_path),
            oracle_overlap_dilate_cells=oracle_overlap_dilate_cells,
            min_component_cells=min_component_cells,
            debug_out=debug,
        )
    else:
        return 0
    if rail.ndim != 2 or rail.size == 0:
        return 0
    x0 = _load_scalar(z, "bev_xmin", 0.0)
    y0 = _load_scalar(z, "bev_ymin", 0.0)
    res = _load_scalar(z, "bev_resolution", 0.2)
    stride = max(1, int(stride_cells))

    yy, xx = np.where(rail >= float(min_prob))
    if yy.size == 0:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# x y z confidence\n")
        if debug_path:
            debug.update({"rail_sample_count": 0.0})
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2)
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

    if refined_png_path:
        import cv2

        cv2.imwrite(refined_png_path, np.clip(rail * 255.0, 0, 255).astype(np.uint8))
    if debug_path:
        debug.update(
            {
                "rail_sample_count": float(len(conf)),
                "rail_sample_stride_cells": float(stride),
                "rail_sample_min_prob": float(min_prob),
                "rail_sample_max_points": float(max_points),
            }
        )
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)
    return int(len(conf))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path")
    ap.add_argument("out_path")
    ap.add_argument("--min-prob", type=float, default=0.15)
    ap.add_argument("--stride-cells", type=int, default=2)
    ap.add_argument("--max-points", type=int, default=8000)
    ap.add_argument("--reference-z", type=float, default=0.0)
    ap.add_argument("--oracle-npz", default="")
    ap.add_argument("--oracle-overlap-dilate-cells", type=int, default=3)
    ap.add_argument("--min-component-cells", type=int, default=20)
    ap.add_argument("--debug-path", default="")
    ap.add_argument("--refined-png", default="")
    args = ap.parse_args(argv)
    n = export_lidar_bev_rail_points(
        args.npz_path,
        args.out_path,
        min_prob=args.min_prob,
        stride_cells=args.stride_cells,
        max_points=args.max_points,
        reference_z=args.reference_z,
        oracle_npz_path=args.oracle_npz or None,
        oracle_overlap_dilate_cells=args.oracle_overlap_dilate_cells,
        min_component_cells=args.min_component_cells,
        debug_path=args.debug_path or None,
        refined_png_path=args.refined_png or None,
    )
    print(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
