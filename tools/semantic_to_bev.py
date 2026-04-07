#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将图像平面语义概率投影到 LiDAR 参考平面上的 pseudo-BEV 栅格。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _bilinear_remap_image(
    image_hw: np.ndarray, map_x: np.ndarray, map_y: np.ndarray
) -> np.ndarray:
    """image_hw: H x W float32；map_x/map_y 与输出同形状，单位为像素坐标。"""
    h, w = image_hw.shape[:2]
    return cv2.remap(
        image_hw.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def semantic_probs_to_pseudo_bev(
    semantic_probs: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: Tuple[np.ndarray, np.ndarray],
    bev_config: Dict[str, Any],
    dataset_meta: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    将语义概率反投影到轨顶/地面参考平面 Z=reference_z（LiDAR 坐标系）上的 BEV 栅格。

    Parameters
    ----------
    semantic_probs : (H_i, W_i, C) float32
    intrinsics : 3x3 K
    extrinsics : (rvec (3,), tvec (3,)) LiDAR -> 相机（与 OpenCV projectPoints 一致）
    bev_config : x_range, y_range, resolution；可选 z_range 忽略，平面用 reference_z
    dataset_meta : semantic_classes, reference_z
    """
    rvec, tvec = extrinsics
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    K = np.asarray(intrinsics, dtype=np.float64)

    x0, x1 = [float(x) for x in bev_config["x_range"]]
    y0, y1 = [float(x) for x in bev_config["y_range"]]
    res = float(bev_config.get("resolution", 0.1))
    reference_z = float(dataset_meta.get("reference_z", 0.0))

    nx = max(1, int(np.ceil((x1 - x0) / res)))
    ny = max(1, int(np.ceil((y1 - y0) / res)))

    classes: List[str] = list(dataset_meta.get("semantic_classes", []))

    def ci(name: str) -> int:
        return classes.index(name) if name in classes else -1

    Hi, Wi, C = semantic_probs.shape
    map_x = np.zeros((ny, nx), dtype=np.float32)
    map_y = np.zeros((ny, nx), dtype=np.float32)
    valid = np.zeros((ny, nx), dtype=np.bool_)

    for j in range(ny):
        Y = y0 + (j + 0.5) * res
        for i in range(nx):
            X = x0 + (i + 0.5) * res
            Pl = np.array([X, Y, reference_z], dtype=np.float64)
            Pc = R @ Pl + t
            if Pc[2] <= 0.05:
                continue
            u = (K[0, 0] * Pc[0] / Pc[2]) + K[0, 2]
            v = (K[1, 1] * Pc[1] / Pc[2]) + K[1, 2]
            if u < 0 or v < 0 or u >= Wi - 1 or v >= Hi - 1:
                continue
            map_x[j, i] = float(u)
            map_y[j, i] = float(v)
            valid[j, i] = True

    out: Dict[str, np.ndarray] = {}
    out["bev_valid_mask"] = valid.astype(np.float32)
    out["bev_x_range"] = np.array([x0, x1], dtype=np.float32)
    out["bev_y_range"] = np.array([y0, y1], dtype=np.float32)
    out["bev_resolution"] = np.array([res], dtype=np.float32)

    def sample_channel(name: str, combine: Optional[List[str]] = None) -> np.ndarray:
        if combine:
            acc = np.zeros((Hi, Wi), dtype=np.float32)
            for n in combine:
                k = ci(n)
                if k >= 0:
                    acc += semantic_probs[:, :, k]
            ch = acc
        else:
            k = ci(name)
            if k < 0:
                return np.zeros((ny, nx), dtype=np.float32)
            ch = semantic_probs[:, :, k]
        warped = _bilinear_remap_image(ch, map_x, map_y)
        warped = np.where(valid, warped, 0.0)
        return warped.astype(np.float32)

    out["rail"] = sample_channel("rail", combine=["rail", "ballast"])
    out["pole_signal"] = sample_channel("", combine=["pole", "signal"])
    out["platform_building"] = sample_channel("", combine=["platform", "building"])
    out["road"] = sample_channel("road")

    return out

