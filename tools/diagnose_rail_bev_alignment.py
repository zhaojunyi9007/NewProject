#!/usr/bin/env python3
"""Diagnose image/LiDAR rail BEV alignment overlap and coarse cell shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_image_rail(path: str) -> np.ndarray:
    z = np.load(path)
    for key in ("rail_from_label_track", "rail_from_likelihood", "rail"):
        if key in z.files:
            arr = np.asarray(z[key], dtype=np.float32)
            if arr.ndim == 2:
                return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.zeros((1, 1), dtype=np.float32)


def _load_lidar_raw(path: str) -> np.ndarray:
    z = np.load(path)
    if "rail_probability_refined" in z.files:
        arr = np.asarray(z["rail_probability_refined"], dtype=np.float32)
    elif "rail_probability" in z.files:
        arr = np.asarray(z["rail_probability"], dtype=np.float32)
    else:
        arr = np.zeros((1, 1), dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def _load_refined_png(path: str | None, shape: tuple[int, int]) -> np.ndarray:
    if not path or not Path(path).is_file():
        return np.zeros(shape, dtype=np.float32)
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(shape, dtype=np.float32)
    arr = img.astype(np.float32) / 255.0
    if arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return arr


def _resize_like(src: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if src.shape == shape:
        return src.astype(np.float32, copy=False)
    import cv2

    return cv2.resize(src.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


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


def diagnose(image_pseudo_bev: str, lidar_bev_maps: str, refined_png: str | None = None, threshold: float = 1e-4, max_shift_cells: int = 80) -> dict:
    image = _load_image_rail(image_pseudo_bev)
    lidar_raw = _resize_like(_load_lidar_raw(lidar_bev_maps), image.shape)
    lidar_refined = _load_refined_png(refined_png, image.shape)
    if not lidar_refined.any():
        lidar_refined = lidar_raw
    best_shift, best_shift_score = _best_shift(image, lidar_refined, threshold, int(max_shift_cells))
    return {
        "image_rail_nonzero_ratio": float((image > threshold).mean()),
        "lidar_rail_raw_nonzero_ratio": float((lidar_raw > threshold).mean()),
        "lidar_rail_refined_nonzero_ratio": float((lidar_refined > threshold).mean()),
        "overlap_ratio_raw": _overlap(image, lidar_raw, threshold),
        "overlap_ratio_refined": _overlap(image, lidar_refined, threshold),
        "best_shift_cells": best_shift,
        "best_shift_score": best_shift_score,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pseudo-bev", required=True)
    ap.add_argument("--lidar-bev-maps", required=True)
    ap.add_argument("--lidar-rail-refined-png", default="")
    ap.add_argument("--threshold", type=float, default=1e-4)
    ap.add_argument("--max-shift-cells", type=int, default=80)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    obj = diagnose(
        args.image_pseudo_bev,
        args.lidar_bev_maps,
        args.lidar_rail_refined_png or None,
        threshold=args.threshold,
        max_shift_cells=args.max_shift_cells,
    )
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
