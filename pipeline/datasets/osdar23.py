#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OSDaR23 dataset adapter: counter_timestamp filenames + calibration.txt."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.datasets.resolver import OSDaRResolver
from pipeline.datasets.base import DatasetAdapter


def load_osdar23_init_extrinsic(calib_path: str, camera_folder: str) -> Optional[Tuple[List[float], List[float]]]:
    """
    Parse OSDaR23 calibration.txt for initial LiDAR->optical extrinsic.
    File gives T_cam_to_parent; merged lidar is parent; then body->optical fixed rotation.
    """
    if not calib_path or not os.path.exists(calib_path):
        return None
    want = (camera_folder or "rgb_center").strip()
    in_cam = False
    cam_match = False
    T_cam_to_parent = None
    with open(calib_path, "r", encoding="utf-8") as f:
        lines = list(f)
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "CAMERA":
            in_cam = True
            cam_match = False
            i += 1
            continue
        if in_cam and s.startswith("data_folder:"):
            folder = s.split(":", 1)[1].strip()
            cam_match = folder == want
            i += 1
            continue
        if in_cam and cam_match and s.startswith("homogeneous transform:"):
            rest = s[len("homogeneous transform:") :].strip()
            nums: List[float] = []
            for x in _extract_numbers(rest):
                if len(nums) < 16:
                    nums.append(x)
            j = i + 1
            while len(nums) < 16 and j < len(lines):
                for x in _extract_numbers(lines[j]):
                    if len(nums) < 16:
                        nums.append(x)
                j += 1
            if len(nums) < 16:
                break
            T_cam_to_parent = np.array(nums[:16], dtype=np.float64).reshape(4, 4)
            break
        i += 1
    if T_cam_to_parent is None:
        return None

    T_lidar_to_body = np.linalg.inv(T_cam_to_parent)

    R_body_to_optical = np.array(
        [
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
        ],
        dtype=np.float64,
    )
    T_body_to_optical = np.eye(4, dtype=np.float64)
    T_body_to_optical[:3, :3] = R_body_to_optical

    T_lidar_to_cam = T_body_to_optical @ T_lidar_to_body
    R = T_lidar_to_cam[:3, :3]
    t = T_lidar_to_cam[:3, 3]
    print(f"  [OSDaR23 calib] T_cam_to_parent t={T_cam_to_parent[:3,3].tolist()}")
    print(f"  [OSDaR23 calib] T_lidar_to_optical t={t.tolist()}")
    rvec, _ = cv2.Rodrigues(R)
    print(f"  [OSDaR23 calib] rvec={rvec.reshape(-1).tolist()}")
    return rvec.reshape(-1).tolist(), t.reshape(-1).tolist()


def _extract_numbers(line: str) -> List[float]:
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
    return [float(x) for x in nums]


def load_osdar23_intrinsics(calib_file: str, camera_folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load K, R_rect=I, P_rect=[K|0] for the given camera data_folder (e.g. rgb_center).
    Aligns with C++ optimizer OSDaR branch.
    """
    if not calib_file or not os.path.exists(calib_file):
        print("[Warning] OSDaR23 calib_file missing; fallback to default intrinsics")
        K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        R_rect = np.eye(3, dtype=np.float64)
        P_rect = np.hstack([K, np.zeros((3, 1), dtype=np.float64)])
        return K, R_rect, P_rect

    want = (camera_folder or "rgb_center").strip()
    in_cam = False
    cam_match = False
    K: Optional[np.ndarray] = None

    with open(calib_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip() for ln in f.readlines()]

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "CAMERA":
            in_cam = True
            cam_match = False
            i += 1
            continue
        if in_cam and s.startswith("data_folder:"):
            folder = s.split(":", 1)[1].strip()
            cam_match = folder == want
            i += 1
            continue
        if in_cam and cam_match and s.strip().startswith("camera_matrix:"):
            rest = s[len("camera_matrix:") :].strip()
            nums: List[float] = []
            for x in _extract_numbers(rest):
                if len(nums) < 9:
                    nums.append(x)
            j = i + 1
            while len(nums) < 9 and j < len(lines):
                for x in _extract_numbers(lines[j]):
                    if len(nums) < 9:
                        nums.append(x)
                j += 1
            if len(nums) < 9:
                break
            K = np.array(nums[:9], dtype=np.float64).reshape(3, 3)
            break
        i += 1

    if K is None:
        raise RuntimeError(f"Cannot parse OSDaR23 camera_matrix for data_folder={want} in {calib_file}")

    R_rect = np.eye(3, dtype=np.float64)
    P_rect = np.hstack([K, np.zeros((3, 1), dtype=np.float64)])
    return K, R_rect, P_rect


class OSDaR23Adapter(DatasetAdapter):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._resolver = OSDaRResolver(config)

    def resolve_image(self, frame_id: int) -> Optional[str]:
        return self._resolver.resolve_image(frame_id)

    def resolve_lidar(self, frame_id: int) -> Optional[str]:
        p = self._resolver.resolve_lidar(frame_id)
        return p if isinstance(p, str) else None

    def list_available_frames(self) -> List[int]:
        return self._resolver.list_available_frames()

    def load_intrinsics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        calib_file = str(self._config.get("data", {}).get("calib_file", "") or "")
        cam = str(self._config.get("data", {}).get("image_sensor", "rgb_center") or "rgb_center")
        return load_osdar23_intrinsics(calib_file, cam)

    def load_initial_extrinsic(self) -> Optional[Tuple[List[float], List[float]]]:
        calib_file = str(self._config.get("data", {}).get("calib_file", "") or "")
        cam = str(self._config.get("data", {}).get("image_sensor", "rgb_center") or "rgb_center")
        return load_osdar23_init_extrinsic(calib_file, cam)

    def get_optimizer_env(self) -> Dict[str, str]:
        cam = str(self._config.get("data", {}).get("image_sensor", "rgb_center") or "rgb_center")
        return {
            "EDGECALIB_OSDAR_CAMERA": cam,
            "EDGECALIB_DATASET_FORMAT": "osdar23",
        }

    def apply_derived_paths(self, config: Dict[str, Any]) -> None:
        data = config.setdefault("data", {})
        fmt = str(data.get("dataset_format", "kitti") or "kitti").lower()
        root = str(data.get("osdar_sequence_root", "") or "").strip()
        if fmt in {"osdar23", "osdar"} and root:
            sensor = str(data.get("image_sensor", "rgb_center") or "rgb_center").strip() or "rgb_center"
            data["image_dir"] = os.path.join(root, sensor)
            data["velodyne_dir"] = os.path.join(root, "lidar")
            data["calib_file"] = os.path.join(root, "calibration.txt")
