#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 pseudo_bev.npz 中的 rail 导出为与 lidar_extractor 一致的 EDGEBEV1 单通道二进制。"""

from __future__ import annotations

import struct
from typing import Any, Dict

import numpy as np


def export_image_rail_bin(npz_path: str, out_bin: str, bev_cfg: Dict[str, Any]) -> bool:
    try:
        z = np.load(npz_path)
    except OSError as e:
        print(f"[bev_export] Cannot load {npz_path}: {e}")
        return False
    if "rail" not in z.files:
        print("[bev_export] npz missing 'rail'")
        return False
    rail = np.asarray(z["rail"], dtype=np.float32)
    x0, x1 = [float(x) for x in bev_cfg["x_range"]]
    y0, y1 = [float(x) for x in bev_cfg["y_range"]]
    res = float(bev_cfg.get("resolution", 0.1))
    nx = max(1, int(np.ceil((x1 - x0) / res)))
    ny = max(1, int(np.ceil((y1 - y0) / res)))
    import cv2

    if rail.shape != (ny, nx):
        rail = cv2.resize(rail, (nx, ny), interpolation=cv2.INTER_LINEAR)

    with open(out_bin, "wb") as f:
        f.write(b"EDGEBEV1")
        f.write(struct.pack("iii", nx, ny, 1))
        f.write(struct.pack("ffff", float(x0), float(y0), res, 0.0))
        f.write(rail.tobytes())
    return True
