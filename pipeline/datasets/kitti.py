#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KITTI dataset adapter: path layout + cam_to_cam / velo_to_cam calibration."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.datasets.resolver import KittiResolver
from pipeline.datasets.base import DatasetAdapter


def parse_velo_to_cam_file(calib_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not calib_path:
        return None
    if not os.path.exists(calib_path):
        print(f"[Warning] LiDAR外参文件不存在: {calib_path}")
        return None

    r_vals = None
    t_vals = None
    tr_vals = None
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("R:"):
                r_vals = [float(x) for x in line.replace("R:", "").split()]
            elif line.startswith("T:"):
                t_vals = [float(x) for x in line.replace("T:", "").split()]
            elif line.startswith("Tr_velo_to_cam:"):
                tr_vals = [float(x) for x in line.replace("Tr_velo_to_cam:", "").split()]

    if tr_vals and len(tr_vals) == 12:
        r_mat = np.array(tr_vals[:9], dtype=np.float64).reshape(3, 3)
        t_vec = np.array(tr_vals[9:], dtype=np.float64)
        return r_mat, t_vec

    if not r_vals or not t_vals or len(r_vals) != 9 or len(t_vals) != 3:
        print(f"[Warning] LiDAR外参文件格式异常: {calib_path}")
        return None

    r_mat = np.array(r_vals, dtype=np.float64).reshape(3, 3)
    t_vec = np.array(t_vals, dtype=np.float64)
    return r_mat, t_vec


def parse_cam_to_cam_file(calib_path: str) -> Dict[str, np.ndarray]:
    if not calib_path or not os.path.exists(calib_path):
        return {}
    parsed: Dict[str, np.ndarray] = {}
    with open(calib_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key not in {"R_rect_00", "R0_rect", "R_rect_02", "R_02", "T_02", "P_rect_00", "P0", "P_rect_02", "P2"}:
                continue
            if not value.strip():
                continue
            try:
                values = [float(x) for x in value.strip().split()]
            except ValueError:
                continue
            if key in {"R_rect_00", "R0_rect", "R_rect_02", "R_02"} and len(values) == 9:
                parsed[key] = np.array(values, dtype=np.float64).reshape(3, 3)
            elif key in {"T_02"} and len(values) == 3:
                parsed[key] = np.array(values, dtype=np.float64)
            elif key in {"P_rect_00", "P0", "P_rect_02", "P2"} and len(values) == 12:
                parsed[key] = np.array(values, dtype=np.float64).reshape(3, 4)
    return parsed


def compose_extrinsic_to_target_camera(
    r_cam0: np.ndarray,
    t_cam0: np.ndarray,
    cam_to_cam: Dict[str, np.ndarray],
    target_camera: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if target_camera != "cam2":
        return r_cam0, t_cam0

    r_02 = cam_to_cam.get("R_02")
    t_02 = cam_to_cam.get("T_02")
    if r_02 is not None and t_02 is not None:
        r_target = r_02 @ r_cam0
        t_target = r_02 @ t_cam0 + t_02
        return r_target, t_target

    p0 = cam_to_cam.get("P_rect_00") or cam_to_cam.get("P0")
    p2 = cam_to_cam.get("P_rect_02") or cam_to_cam.get("P2")
    if p0 is not None and p2 is not None and abs(p0[0, 0]) > 1e-9 and abs(p2[0, 0]) > 1e-9:
        tx0 = p0[0, 3] / p0[0, 0]
        tx2 = p2[0, 3] / p2[0, 0]
        baseline_x = tx2 - tx0
        t_target = t_cam0 + np.array([baseline_x, 0.0, 0.0], dtype=np.float64)
        print("[Warning] cam_to_cam缺少R_02/T_02，使用P_rect基线近似Cam0->Cam2平移")
        return r_cam0, t_target

    print("[Warning] 无法从cam_to_cam构造Cam0->Cam2，回退使用Cam0外参")
    return r_cam0, t_cam0


def load_velo_to_cam_extrinsic(config: Dict[str, Any]) -> Optional[Tuple[List[float], List[float]]]:
    velo_path = config["data"].get("velo_to_cam_file", "")
    parsed = parse_velo_to_cam_file(velo_path)
    if parsed is None:
        return None

    r_cam0, t_cam0 = parsed
    init_cfg = config.get("calibration", {}).get("extrinsic_init", {})
    target_camera = str(init_cfg.get("target_camera", "cam2")).lower()
    cam_to_cam_path = config["data"].get("calib_file", "")
    cam_to_cam = parse_cam_to_cam_file(cam_to_cam_path)
    r_target, t_target = compose_extrinsic_to_target_camera(r_cam0, t_cam0, cam_to_cam, target_camera)

    r_vec, _ = cv2.Rodrigues(r_target)
    print(f"[Info] 初始外参目标相机: {target_camera}")
    return r_vec.reshape(-1).tolist(), t_target.reshape(-1).tolist()


def load_kitti_intrinsics(calib_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load K, R_rect, P_rect from calib_cam_to_cam.txt (P2 / R_rect_00)."""
    default_k = np.array([[721.5, 0, 609.5], [0, 721.5, 172.8], [0, 0, 1]], dtype=np.float64)
    default_r_rect = np.eye(3, dtype=np.float64)
    default_p_rect = np.array(
        [[721.5, 0, 609.5, 0.0], [0, 721.5, 172.8, 0.0], [0, 0, 1, 0.0]], dtype=np.float64
    )

    if not calib_file or not os.path.exists(calib_file):
        print("[Warning] No calib_file provided; using default camera intrinsics (KITTI typical values)")
        return default_k, default_r_rect, default_p_rect

    try:
        K = None
        R_rect = None
        P_rect = None
        with open(calib_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("P2:") or line.startswith("P_rect_02:"):
                    parts = line.split(":", 1)
                    if len(parts) >= 2:
                        values_str = parts[1].strip()
                        values = list(map(float, values_str.split()))
                        if len(values) == 12:
                            P_rect = np.array(values, dtype=np.float64).reshape(3, 4)
                            K = P_rect[:, :3].copy()
                elif line.startswith("R0_rect:") or line.startswith("R_rect_00:"):
                    parts = line.split(":", 1)
                    if len(parts) >= 2:
                        values_str = parts[1].strip()
                        values = list(map(float, values_str.split()))
                        if len(values) == 9:
                            R_rect = np.array(values, dtype=np.float64).reshape(3, 3)
        if K is None or P_rect is None:
            raise ValueError("P2/P_rect_02 not found in calibration file.")
        if R_rect is None:
            R_rect = default_r_rect
        print(f"[Info] Loaded camera intrinsics and rectification from {calib_file}")
        return K, R_rect, P_rect
    except Exception as e:
        print(f"[Error] Failed to parse KITTI calibration file: {e}")
        print("[Warning] Using default camera intrinsics")
        return default_k, default_r_rect, default_p_rect


class KittiAdapter(DatasetAdapter):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._resolver = KittiResolver(config)

    def resolve_image(self, frame_id: int) -> Optional[str]:
        return self._resolver.resolve_image(frame_id)

    def resolve_lidar(self, frame_id: int) -> Optional[str]:
        p = self._resolver.resolve_lidar(frame_id)
        return p if isinstance(p, str) else None

    def list_available_frames(self) -> List[int]:
        return self._resolver.list_available_frames()

    def load_intrinsics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        calib_file = str(self._config.get("data", {}).get("calib_file", "") or "")
        return load_kitti_intrinsics(calib_file)

    def load_initial_extrinsic(self) -> Optional[Tuple[List[float], List[float]]]:
        return load_velo_to_cam_extrinsic(self._config)

    def get_optimizer_env(self) -> Dict[str, str]:
        return {"EDGECALIB_DATASET_FORMAT": "kitti"}

    def apply_derived_paths(self, config: Dict[str, Any]) -> None:
        _ = config
