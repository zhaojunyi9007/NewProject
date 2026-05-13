#!/usr/bin/env python3
"""Export refined LiDAR rail BEV cells as optimizer-friendly 3D rail samples."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class MetricGrid:
    xmin: float
    ymin: float
    resolution: float
    nx: int
    ny: int


@dataclass(frozen=True)
class OracleRail:
    rail: np.ndarray
    grid: MetricGrid


def _load_scalar(z: Any, key: str, default: float) -> float:
    return float(np.asarray(z[key]).reshape(-1)[0]) if key in z.files else float(default)


def _load_range_start(z: Any, key: str, default: float) -> float:
    if key not in z.files:
        return float(default)
    arr = np.asarray(z[key]).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _default_grid(arr: np.ndarray) -> MetricGrid:
    ny, nx = arr.shape
    return MetricGrid(xmin=0.0, ymin=0.0, resolution=1.0, nx=int(nx), ny=int(ny))


def _grid_from_npz(z: Any, arr: np.ndarray, prefix: str, default_res: float = 1.0) -> MetricGrid:
    ny, nx = arr.shape
    if prefix == "image":
        xmin = _load_range_start(z, "bev_x_range", 0.0)
        ymin = _load_range_start(z, "bev_y_range", 0.0)
    else:
        xmin = _load_scalar(z, "bev_xmin", 0.0)
        ymin = _load_scalar(z, "bev_ymin", 0.0)
    return MetricGrid(
        xmin=float(xmin),
        ymin=float(ymin),
        resolution=max(_load_scalar(z, "bev_resolution", default_res), 1e-6),
        nx=int(nx),
        ny=int(ny),
    )


def _load_oracle_rail(oracle_npz_path: Optional[str]) -> Optional[OracleRail]:
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
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
                return OracleRail(rail=arr, grid=_grid_from_npz(z, arr, "image", default_res=1.0))
    return None


def _resample_to_grid(src: np.ndarray, src_grid: MetricGrid, dst_grid: MetricGrid) -> np.ndarray:
    src = np.nan_to_num(np.asarray(src, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    out = np.zeros((dst_grid.ny, dst_grid.nx), dtype=np.float32)
    for iy in range(dst_grid.ny):
        y = dst_grid.ymin + (float(iy) + 0.5) * dst_grid.resolution
        v = int(np.floor((y - src_grid.ymin) / src_grid.resolution))
        if v < 0 or v >= src_grid.ny:
            continue
        for ix in range(dst_grid.nx):
            x = dst_grid.xmin + (float(ix) + 0.5) * dst_grid.resolution
            u = int(np.floor((x - src_grid.xmin) / src_grid.resolution))
            if 0 <= u < src_grid.nx:
                out[iy, ix] = src[v, u]
    return out


def _mask_bbox_m(mask: np.ndarray, grid: MetricGrid) -> list[float]:
    yy, xx = np.where(mask)
    if yy.size == 0:
        return []
    return [
        float(grid.xmin + float(xx.min()) * grid.resolution),
        float(grid.ymin + float(yy.min()) * grid.resolution),
        float(grid.xmin + float(xx.max() + 1) * grid.resolution),
        float(grid.ymin + float(yy.max() + 1) * grid.resolution),
    ]


def _mask_centroid_m(mask: np.ndarray, grid: MetricGrid) -> Optional[tuple[float, float]]:
    yy, xx = np.where(mask)
    if yy.size == 0:
        return None
    xs = grid.xmin + (xx.astype(np.float64) + 0.5) * grid.resolution
    ys = grid.ymin + (yy.astype(np.float64) + 0.5) * grid.resolution
    return float(xs.mean()), float(ys.mean())


def _overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(np.logical_or(a, b).sum())
    if denom <= 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(denom)


def _write_single_channel_bev_bin(path: str, rail: np.ndarray, x0: float, y0: float, res: float) -> None:
    arr = np.nan_to_num(np.asarray(rail, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    ny, nx = arr.shape
    with open(path, "wb") as f:
        f.write(b"EDGEBEV1")
        f.write(struct.pack("iii", int(nx), int(ny), 1))
        f.write(struct.pack("ffff", float(x0), float(y0), float(res), 0.0))
        f.write(arr.astype(np.float32, copy=False).tobytes(order="C"))


def refine_lidar_rail_probability(
    rail: np.ndarray,
    min_prob: float = 0.15,
    oracle_rail: Optional[np.ndarray | OracleRail] = None,
    lidar_grid: Optional[MetricGrid] = None,
    oracle_overlap_dilate_cells: int = 3,
    min_component_cells: int = 20,
    debug_out: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Denoise LiDAR rail probability using thresholding, components, and optional oracle overlap."""
    import cv2

    arr = np.nan_to_num(np.asarray(rail, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    if lidar_grid is None:
        lidar_grid = _default_grid(arr)
    raw_nonzero = float((arr >= float(min_prob)).mean()) if arr.size else 0.0

    mask = arr >= float(min_prob)
    oracle_used = False
    oracle_nonzero = 0.0
    oracle_grid = _default_grid(arr)
    metric_overlap = 0.0
    metric_lidar_bbox = _mask_bbox_m(mask, lidar_grid)
    metric_image_bbox: list[float] = []
    metric_centroid_delta: list[float] = []
    if oracle_rail is not None:
        if isinstance(oracle_rail, OracleRail):
            oracle_src = oracle_rail.rail
            oracle_grid = oracle_rail.grid
        else:
            oracle_src = np.asarray(oracle_rail, dtype=np.float32)
            oracle_grid = _default_grid(oracle_src)
        oracle_used = True
        oracle = _resample_to_grid(oracle_src, oracle_grid, lidar_grid) if oracle_src.size else np.zeros_like(arr)
        oracle = np.nan_to_num(oracle, nan=0.0, posinf=1.0, neginf=0.0)
        oracle_mask = oracle > 1e-4
        oracle_nonzero = float(oracle_mask.mean()) if oracle_mask.size else 0.0
        metric_overlap = _overlap_ratio(mask, oracle_mask)
        metric_image_bbox = _mask_bbox_m(oracle_mask, lidar_grid)
        lidar_centroid = _mask_centroid_m(mask, lidar_grid)
        image_centroid = _mask_centroid_m(oracle_mask, lidar_grid)
        if lidar_centroid is not None and image_centroid is not None:
            metric_centroid_delta = [
                float(lidar_centroid[0] - image_centroid[0]),
                float(lidar_centroid[1] - image_centroid[1]),
            ]
        oracle_prob_floor = max(float(min_prob), 0.35)
        mask &= arr >= oracle_prob_floor
        dilate = max(0, int(oracle_overlap_dilate_cells))
        if oracle_mask.any() and dilate > 0:
            k = 2 * dilate + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            oracle_mask = cv2.dilate(oracle_mask.astype(np.uint8), kernel, iterations=1) > 0
        mask &= oracle_mask

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

    pre_fallback_refined_nonzero = float((refined > 1e-4).mean()) if refined.size else 0.0
    refined_nonzero = pre_fallback_refined_nonzero
    fallback_used = False
    if refined_nonzero <= 0.0 and raw_nonzero > 0.0:
        # Preserve samples for diagnostics, but mark oracle mismatch explicitly.
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
                "rail_refinement_valid": bool(not (fallback_used and oracle_used)),
                "rail_refinement_mismatch": bool(fallback_used and oracle_used),
                "rail_refinement_empty": bool(pre_fallback_refined_nonzero <= 0.0),
                "lidar_rail_min_prob": float(min_prob),
                "lidar_rail_min_component_cells": float(max(1, int(min_component_cells))),
                "oracle_resampling_mode": "metric_cell_center",
                "image_bev_xmin": float(oracle_grid.xmin),
                "image_bev_ymin": float(oracle_grid.ymin),
                "image_bev_resolution": float(oracle_grid.resolution),
                "lidar_bev_xmin": float(lidar_grid.xmin),
                "lidar_bev_ymin": float(lidar_grid.ymin),
                "lidar_bev_resolution": float(lidar_grid.resolution),
                "metric_overlap_ratio": float(metric_overlap),
                "metric_lidar_rail_bbox_m": metric_lidar_bbox,
                "metric_image_rail_bbox_m": metric_image_bbox,
                "metric_centroid_delta_m": metric_centroid_delta,
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
    refined_bin_path: Optional[str] = None,
) -> int:
    z = np.load(npz_path)
    x0 = _load_scalar(z, "bev_xmin", 0.0)
    y0 = _load_scalar(z, "bev_ymin", 0.0)
    res = _load_scalar(z, "bev_resolution", 0.2)
    if "rail_probability_refined" in z.files:
        rail = np.asarray(z["rail_probability_refined"], dtype=np.float32)
        debug: dict[str, Any] = {"lidar_rail_refine_source": "rail_probability_refined"}
    elif "rail_probability" in z.files:
        debug = {"lidar_rail_refine_source": "rail_probability"}
        raw_rail = np.asarray(z["rail_probability"], dtype=np.float32)
        rail = refine_lidar_rail_probability(
            raw_rail,
            min_prob=min_prob,
            oracle_rail=_load_oracle_rail(oracle_npz_path),
            lidar_grid=_grid_from_npz(z, raw_rail, "lidar", default_res=0.2),
            oracle_overlap_dilate_cells=oracle_overlap_dilate_cells,
            min_component_cells=min_component_cells,
            debug_out=debug,
        )
    else:
        return 0
    if rail.ndim != 2 or rail.size == 0:
        return 0

    if refined_bin_path:
        _write_single_channel_bev_bin(refined_bin_path, rail, x0, y0, res)
    if refined_png_path:
        import cv2

        cv2.imwrite(refined_png_path, np.clip(rail * 255.0, 0, 255).astype(np.uint8))

    sample_rail = np.zeros_like(rail, dtype=np.float32) if debug.get("rail_refinement_mismatch", False) else rail
    stride = max(1, int(stride_cells))
    yy, xx = np.where(sample_rail >= float(min_prob))
    if yy.size == 0:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# x y z confidence\n")
        if debug_path:
            debug.update({"rail_sample_count": 0.0, "rail_sample_count_for_optimizer": 0})
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2)
        return 0

    keep = ((yy % stride) == 0) & ((xx % stride) == 0)
    yy, xx = yy[keep], xx[keep]
    conf = sample_rail[yy, xx]
    if yy.size > int(max_points) > 0:
        order = np.argsort(conf)[::-1][: int(max_points)]
        yy, xx, conf = yy[order], xx[order], conf[order]

    xs = x0 + (xx.astype(np.float64) + 0.5) * res
    ys = y0 + (yy.astype(np.float64) + 0.5) * res
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# x y z confidence\n")
        for x, y, c in zip(xs, ys, conf):
            f.write(f"{float(x):.6f} {float(y):.6f} {float(reference_z):.6f} {float(c):.6f}\n")

    if debug_path:
        debug.update(
            {
                "rail_sample_count": float(len(conf)),
                "rail_sample_count_for_optimizer": int(len(conf)),
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
    ap.add_argument("--refined-bin", default="")
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
        refined_bin_path=args.refined_bin or None,
    )
    print(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
