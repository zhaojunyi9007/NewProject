#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 pseudo_bev.npz 中的 rail 导出为与 lidar_extractor 一致的 EDGEBEV1 单通道二进制。"""

from __future__ import annotations

import struct
from typing import Any, Dict, Optional

import numpy as np


def _select_rail_layer(z, source: str) -> tuple[np.ndarray, str]:
    source = (source or "label_track").strip().lower()
    if source == "label_track":
        for key in ("rail_from_label_track", "rail_from_likelihood", "rail"):
            if key in z.files:
                return np.asarray(z[key], dtype=np.float32), key
    if source == "likelihood":
        for key in ("rail_from_likelihood", "rail_from_label_track", "rail"):
            if key in z.files:
                return np.asarray(z[key], dtype=np.float32), key
    if source == "fused":
        layers = [np.asarray(z[k], dtype=np.float32) for k in ("rail_from_label_track", "rail_from_likelihood", "rail") if k in z.files]
        if layers:
            return np.maximum.reduce(layers), "fused"
    if "rail" in z.files:
        return np.asarray(z["rail"], dtype=np.float32), "rail"
    raise KeyError("npz missing usable rail layer")


def export_image_rail_bin(
    npz_path: str,
    out_bin: str,
    bev_cfg: Dict[str, Any],
    debug_out: Optional[Dict[str, Any]] = None,
) -> bool:
    try:
        z = np.load(npz_path)
    except OSError as e:
        print(f"[bev_export] Cannot load {npz_path}: {e}")
        return False
    try:
        rail, source_used = _select_rail_layer(z, str(bev_cfg.get("image_rail_source", "label_track")))
    except KeyError as e:
        print(f"[bev_export] {e}")
        return False
    x0, x1 = [float(x) for x in bev_cfg["x_range"]]
    y0, y1 = [float(x) for x in bev_cfg["y_range"]]
    res = float(bev_cfg.get("resolution", 0.1))
    nx = max(1, int(np.ceil((x1 - x0) / res)))
    ny = max(1, int(np.ceil((y1 - y0) / res)))
    import cv2

    if rail.shape != (ny, nx):
        rail = cv2.resize(rail, (nx, ny), interpolation=cv2.INTER_LINEAR)
    rail = np.nan_to_num(rail.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    rail = np.clip(rail, 0.0, 1.0)
    pre_nonzero = float((rail > 1e-4).mean()) if rail.size else 0.0

    threshold = float(bev_cfg.get("image_rail_threshold", 0.0) or 0.0)
    if threshold > 0.0:
        rail = np.where(rail >= threshold, rail, 0.0).astype(np.float32)

    dilate_cells = int(bev_cfg.get("image_rail_dilate_cells", 0) or 0)
    if dilate_cells > 0 and np.any(rail > 0):
        k = 2 * dilate_cells + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        rail = cv2.dilate(rail, kernel, iterations=1)

    blur_sigma = float(bev_cfg.get("image_rail_blur_sigma_cells", 0.0) or 0.0)
    if blur_sigma > 0.0 and np.any(rail > 0):
        rail = cv2.GaussianBlur(rail, (0, 0), blur_sigma)
        rail = np.clip(rail, 0.0, 1.0)

    post_nonzero = float((rail > 1e-4).mean()) if rail.size else 0.0
    if debug_out is not None:
        debug_out.update(
            {
                "image_rail_source": source_used,
                "image_rail_requested_source": str(bev_cfg.get("image_rail_source", "label_track")),
                "image_rail_threshold": threshold,
                "image_rail_dilate_cells": dilate_cells,
                "image_rail_blur_sigma_cells": blur_sigma,
                "image_rail_pre_nonzero_ratio": pre_nonzero,
                "image_rail_nonzero_ratio": post_nonzero,
                "image_rail_max": float(rail.max()) if rail.size else 0.0,
                "image_rail_mean": float(rail.mean()) if rail.size else 0.0,
            }
        )

    with open(out_bin, "wb") as f:
        f.write(b"EDGEBEV1")
        f.write(struct.pack("iii", nx, ny, 1))
        f.write(struct.pack("ffff", float(x0), float(y0), res, 0.0))
        f.write(rail.tobytes())
    return True
