#!/usr/bin/env python3
"""Diagnose image/LiDAR rail BEV alignment overlap, axis transforms, and coarse cell shift."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MetricGrid:
    xmin: float
    ymin: float
    resolution: float
    nx: int
    ny: int


def _load_scalar(z, key: str, default: float) -> float:
    return float(np.asarray(z[key]).reshape(-1)[0]) if key in z.files else float(default)


def _load_range_start(z, key: str, default: float) -> float:
    if key not in z.files:
        return float(default)
    arr = np.asarray(z[key]).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _grid_for(arr: np.ndarray, xmin: float, ymin: float, resolution: float) -> MetricGrid:
    ny, nx = arr.shape
    return MetricGrid(float(xmin), float(ymin), max(float(resolution), 1e-6), int(nx), int(ny))


def _load_image_rail_with_grid(path: str) -> tuple[np.ndarray, MetricGrid]:
    z = np.load(path)
    for key in ("rail_from_label_track", "rail_from_likelihood", "rail"):
        if key in z.files:
            arr = np.asarray(z[key], dtype=np.float32)
            if arr.ndim == 2:
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
                return arr, _grid_for(
                    arr,
                    _load_range_start(z, "bev_x_range", 0.0),
                    _load_range_start(z, "bev_y_range", 0.0),
                    _load_scalar(z, "bev_resolution", 1.0),
                )
    arr = np.zeros((1, 1), dtype=np.float32)
    return arr, _grid_for(arr, 0.0, 0.0, 1.0)


def _load_image_rail(path: str) -> np.ndarray:
    return _load_image_rail_with_grid(path)[0]


def _load_lidar_raw_with_grid(path: str) -> tuple[np.ndarray, MetricGrid]:
    z = np.load(path)
    if "rail_probability_refined" in z.files:
        arr = np.asarray(z["rail_probability_refined"], dtype=np.float32)
    elif "rail_probability" in z.files:
        arr = np.asarray(z["rail_probability"], dtype=np.float32)
    else:
        arr = np.zeros((1, 1), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr, _grid_for(
        arr,
        _load_scalar(z, "bev_xmin", 0.0),
        _load_scalar(z, "bev_ymin", 0.0),
        _load_scalar(z, "bev_resolution", 1.0),
    )


def _load_lidar_raw(path: str) -> np.ndarray:
    return _load_lidar_raw_with_grid(path)[0]


def _load_refined_png(path: str | None, shape: tuple[int, int]) -> np.ndarray:
    if not path or not Path(path).is_file():
        return np.zeros(shape, dtype=np.float32)
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(shape, dtype=np.float32)
    arr = img.astype(np.float32) / 255.0
    return _resize_like(arr, shape)


def _resize_like(src: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if src.shape == shape:
        return src.astype(np.float32, copy=False)
    import cv2

    return cv2.resize(src.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


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


def _mask_centroid_m(mask: np.ndarray, grid: MetricGrid) -> tuple[float, float] | None:
    yy, xx = np.where(mask)
    if yy.size == 0:
        return None
    xs = grid.xmin + (xx.astype(np.float64) + 0.5) * grid.resolution
    ys = grid.ymin + (yy.astype(np.float64) + 0.5) * grid.resolution
    return float(xs.mean()), float(ys.mean())


def _metric_debug_fields(
    image: np.ndarray,
    image_grid: MetricGrid,
    lidar: np.ndarray,
    lidar_grid: MetricGrid,
    threshold: float,
) -> dict:
    im = image > threshold
    lm = lidar > threshold
    ci = _mask_centroid_m(im, image_grid)
    cl = _mask_centroid_m(lm, lidar_grid)
    delta = []
    if ci is not None and cl is not None:
        delta = [float(cl[0] - ci[0]), float(cl[1] - ci[1])]
    return {
        "oracle_resampling_mode": "metric_cell_center",
        "image_bev_xmin": float(image_grid.xmin),
        "image_bev_ymin": float(image_grid.ymin),
        "image_bev_resolution": float(image_grid.resolution),
        "lidar_bev_xmin": float(lidar_grid.xmin),
        "lidar_bev_ymin": float(lidar_grid.ymin),
        "lidar_bev_resolution": float(lidar_grid.resolution),
        "metric_lidar_rail_bbox_m": _mask_bbox_m(lm, lidar_grid),
        "metric_image_rail_bbox_m": _mask_bbox_m(im, image_grid),
        "metric_centroid_delta_m": delta,
    }


def _apply_transform(src: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return src
    if name == "flip_x":
        return np.fliplr(src)
    if name == "flip_y":
        return np.flipud(src)
    if name == "swap_xy":
        return src.T
    if name == "swap_xy_flip_x":
        return np.fliplr(src.T)
    if name == "swap_xy_flip_y":
        return np.flipud(src.T)
    raise ValueError(f"unknown transform: {name}")


def _overlap(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    ma = a > threshold
    mb = b > threshold
    denom = int(np.logical_or(ma, mb).sum())
    if denom <= 0:
        return 0.0
    return float(np.logical_and(ma, mb).sum()) / float(denom)


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    y0_src = max(0, -dy)
    y1_src = min(h, h - dy)
    x0_src = max(0, -dx)
    x1_src = min(w, w - dx)
    y0_dst = max(0, dy)
    y1_dst = min(h, h + dy)
    x0_dst = max(0, dx)
    x1_dst = min(w, w + dx)
    if y1_src > y0_src and x1_src > x0_src:
        out[y0_dst:y1_dst, x0_dst:x1_dst] = mask[y0_src:y1_src, x0_src:x1_src]
    return out


def _best_shift(image: np.ndarray, lidar: np.ndarray, threshold: float, max_shift: int) -> tuple[list[int], float]:
    import cv2

    h, w = image.shape
    scale = max(1, int(np.ceil(max(h, w) / 400.0)))
    if scale > 1:
        small_shape = (max(1, w // scale), max(1, h // scale))
        image_s = cv2.resize(image.astype(np.float32), small_shape, interpolation=cv2.INTER_AREA)
        lidar_s = cv2.resize(lidar.astype(np.float32), small_shape, interpolation=cv2.INTER_AREA)
    else:
        image_s = image
        lidar_s = lidar

    im = image_s > threshold
    lm = lidar_s > threshold
    best = [0, 0]
    best_score = _overlap(image_s, lidar_s, threshold)
    search = max(1, int(np.ceil(max_shift / scale)))
    step = max(1, int(search // 50))
    for dy in range(-search, search + 1, step):
        for dx in range(-search, search + 1, step):
            shifted = _shift_mask(lm, dy, dx)
            denom = int(np.logical_or(im, shifted).sum())
            score = 0.0 if denom <= 0 else float(np.logical_and(im, shifted).sum()) / float(denom)
            if score > best_score:
                best_score = score
                best = [int(dx * scale), int(dy * scale)]
    return best, float(best_score)


def diagnose_arrays(
    image: np.ndarray,
    lidar_raw: np.ndarray,
    lidar_refined: np.ndarray | None = None,
    threshold: float = 1e-4,
    max_shift_cells: int = 80,
    min_overlap: float = 0.15,
    test_transforms: bool = True,
) -> dict:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    raw_base = np.nan_to_num(lidar_raw.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    refined_base = raw_base if lidar_refined is None or not np.asarray(lidar_refined).any() else np.nan_to_num(
        lidar_refined.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0
    )
    transforms = ["identity", "flip_x", "flip_y", "swap_xy", "swap_xy_flip_x", "swap_xy_flip_y"] if test_transforms else ["identity"]
    candidates = []
    for name in transforms:
        raw = _resize_like(_apply_transform(raw_base, name), image.shape)
        refined = _resize_like(_apply_transform(refined_base, name), image.shape)
        shift, score = _best_shift(image, refined, threshold, int(max_shift_cells))
        candidates.append(
            {
                "transform_name": name,
                "overlap_ratio_raw": _overlap(image, raw, threshold),
                "overlap_ratio_refined": _overlap(image, refined, threshold),
                "best_shift_cells": shift,
                "best_shift_score": score,
            }
        )
    best = max(candidates, key=lambda x: float(x["best_shift_score"])) if candidates else {}
    return {
        "image_rail_nonzero_ratio": float((image > threshold).mean()),
        "lidar_rail_raw_nonzero_ratio": float((_resize_like(raw_base, image.shape) > threshold).mean()),
        "lidar_rail_refined_nonzero_ratio": float((_resize_like(refined_base, image.shape) > threshold).mean()),
        "overlap_ratio_raw": float(candidates[0]["overlap_ratio_raw"]) if candidates else 0.0,
        "overlap_ratio_refined": float(candidates[0]["overlap_ratio_refined"]) if candidates else 0.0,
        "best_transform": best.get("transform_name", "identity"),
        "best_shift_cells": best.get("best_shift_cells", [0, 0]),
        "best_shift_score": float(best.get("best_shift_score", 0.0)),
        "rail_bev_alignment_valid": bool(float(best.get("best_shift_score", 0.0)) >= float(min_overlap)),
        "rail_bev_alignment_min_overlap": float(min_overlap),
        "transform_candidates": candidates,
    }


def diagnose(
    image_pseudo_bev: str,
    lidar_bev_maps: str,
    refined_png: str | None = None,
    threshold: float = 1e-4,
    max_shift_cells: int = 80,
    min_overlap: float = 0.15,
    test_transforms: bool = True,
) -> dict:
    image, image_grid = _load_image_rail_with_grid(image_pseudo_bev)
    lidar_raw, lidar_grid = _load_lidar_raw_with_grid(lidar_bev_maps)
    lidar_refined = _load_refined_png(refined_png, lidar_raw.shape) if refined_png else None
    lidar_raw_on_image_grid = _resample_to_grid(lidar_raw, lidar_grid, image_grid)
    lidar_refined_on_image_grid = (
        _resample_to_grid(lidar_refined, lidar_grid, image_grid) if lidar_refined is not None else None
    )
    out = diagnose_arrays(
        image,
        lidar_raw_on_image_grid,
        lidar_refined_on_image_grid,
        threshold=threshold,
        max_shift_cells=max_shift_cells,
        min_overlap=min_overlap,
        test_transforms=test_transforms,
    )
    out.update(_metric_debug_fields(image, image_grid, lidar_raw, lidar_grid, threshold))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pseudo-bev", required=True)
    ap.add_argument("--lidar-bev-maps", required=True)
    ap.add_argument("--lidar-rail-refined-png", default="")
    ap.add_argument("--threshold", type=float, default=1e-4)
    ap.add_argument("--max-shift-cells", type=int, default=80)
    ap.add_argument("--min-overlap", type=float, default=0.15)
    ap.add_argument("--no-axis-transforms", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    obj = diagnose(
        args.image_pseudo_bev,
        args.lidar_bev_maps,
        args.lidar_rail_refined_png or None,
        threshold=args.threshold,
        max_shift_cells=args.max_shift_cells,
        min_overlap=args.min_overlap,
        test_transforms=not args.no_axis_transforms,
    )
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
