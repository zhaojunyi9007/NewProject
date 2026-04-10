#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单帧可观测性打分（Phase 6.3）：便于在 Python 侧快速迭代。"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np


def _safe_mean_max_prob(npy_path: str) -> Optional[float]:
    try:
        z = np.load(npy_path)
    except OSError:
        return None
    if z.ndim != 3:
        return None
    pmax = float(np.mean(np.max(z, axis=2)))
    return max(0.0, min(1.0, pmax))


def _line_count(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
    return n


def _rail_centerlines_count(path: str) -> int:
    """
    Count points in rail_centerlines_2d.txt.
    Format (Phase 2): poly_id u v, blank lines between polylines, '#' comments allowed.
    We count valid (poly_id,u,v) rows as signal strength.
    """
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                int(parts[0])
                int(parts[1])
                int(parts[2])
            except ValueError:
                continue
            n += 1
    return n


def _safe_nonzero_ratio_u8_png(path: str) -> Optional[float]:
    """
    Read a png as u8 and compute non-zero pixel ratio.
    Returns None on failure.
    """
    try:
        import cv2  # local import: optional dependency at runtime
    except Exception:
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None
    nz = float(np.count_nonzero(img))
    return max(0.0, min(1.0, nz / float(img.size)))


def _file_nonempty(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 32
    except OSError:
        return False


def compute_frame_observability(
    frame_id: int,
    *,
    lidar_dir: str,
    image_features_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> tuple[float, Dict[str, float]]:
    """
    综合：语义概率峰值、2D 线数量、边缘图、语义图存在性、可见类别数（近似）。
    返回 (score in [0,1], debug dict)。
    """
    _ = config
    fid = f"{frame_id:010d}"
    base_l = os.path.join(lidar_dir, fid)
    base_i = os.path.join(image_features_dir, fid)

    dbg: Dict[str, float] = {}
    score = 0.25

    npy_sem = os.path.join(base_i, "semantic_probs.npy")
    if os.path.isfile(npy_sem):
        mm = _safe_mean_max_prob(npy_sem)
        if mm is not None:
            dbg["mean_max_semantic_prob"] = mm
            score += 0.22 * mm
        try:
            z = np.load(npy_sem)
            if z.ndim == 3:
                C = z.shape[2]
                visible = int(np.sum(np.mean(z.reshape(-1, C), axis=0) > 0.05))
                dbg["visible_class_bins"] = float(visible)
                score += 0.08 * min(1.0, visible / 10.0)
        except OSError:
            pass

    # Phase 2 (sam_2d): observability no longer depends on lines_2d.txt.
    # Prefer rail_centerlines_2d.txt; fallback to rail_centerline.png non-zero ratio.
    rail_txt = os.path.join(base_i, "rail_centerlines_2d.txt")
    rail_png = os.path.join(base_i, "rail_centerline.png")
    n_rail_pts = _rail_centerlines_count(rail_txt)
    dbg["rail_centerline_points"] = float(n_rail_pts)
    if n_rail_pts > 0:
        score += 0.12 * min(1.0, n_rail_pts / 600.0)
        dbg["rail_centerline_source"] = 1.0  # txt
    else:
        rr = _safe_nonzero_ratio_u8_png(rail_png)
        if rr is not None:
            dbg["rail_centerline_nonzero_ratio"] = rr
            score += 0.12 * min(1.0, rr / 0.08)
            dbg["rail_centerline_source"] = 2.0  # png
        else:
            dbg["rail_centerline_source"] = 0.0  # missing

    if _file_nonempty(os.path.join(base_i, "edge_map.png")):
        score += 0.08
        dbg["has_edge_map"] = 1.0
    else:
        dbg["has_edge_map"] = 0.0

    if _file_nonempty(os.path.join(base_i, "semantic_argmax.png")):
        score += 0.06
        dbg["has_semantic_argmax"] = 1.0
    else:
        dbg["has_semantic_argmax"] = 0.0

    # LiDAR：BEV / 语义点存在
    if _file_nonempty(base_l + "_bev_maps.npz") or _file_nonempty(base_l + "_bev_channels.bin"):
        score += 0.08
        dbg["has_bev"] = 1.0
    else:
        dbg["has_bev"] = 0.0

    if _file_nonempty(base_l + "_semantic_points.txt"):
        score += 0.07
        dbg["has_semantic_points"] = 1.0
    else:
        dbg["has_semantic_points"] = 0.0

    score = max(0.0, min(1.0, score))
    dbg["observability_raw"] = score
    return score, dbg
