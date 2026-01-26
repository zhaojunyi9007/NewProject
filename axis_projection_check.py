#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Axis projection sanity check.

Projects LiDAR axis points into the image using the given extrinsic and
camera intrinsics (from KITTI calib file). This helps verify whether the
LiDAR->camera coordinate conventions are consistent.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


def load_kitti_calib(calib_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not calib_file or not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")
    K = None
    R_rect = None
    P_rect = None
    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("P2:") or line.startswith("P_rect_02:"):
                parts = line.split(":")
                values = list(map(float, parts[1].strip().split()))
                if len(values) == 12:
                    P_rect = np.array(values, dtype=np.float64).reshape(3, 4)
                    K = P_rect[:, :3].copy()
            if line.startswith("R0_rect:") or line.startswith("R_rect_00:"):
                parts = line.split(":")
                values = list(map(float, parts[1].strip().split()))
                if len(values) == 9:
                    R_rect = np.array(values, dtype=np.float64).reshape(3, 3)
    if K is None or P_rect is None:
        raise ValueError("Failed to parse P2/P_rect_02 from calibration file.")
    if R_rect is None:
        R_rect = np.eye(3, dtype=np.float64)
    return K, R_rect, P_rect
                    

def project_point(p_lidar: np.ndarray,
                  R_rect: np.ndarray,
                  P_rect: np.ndarray,
                  R: np.ndarray,
                  t: np.ndarray) -> tuple[bool, np.ndarray]:
    p_cam = R @ p_lidar + t
    p_rect = R_rect @ p_cam
    if p_rect[2] <= 0:
        return False, p_rect
    uv = P_rect @ np.hstack([p_rect, 1.0])
    return True, np.array([uv[0] / uv[2], uv[1] / uv[2], p_cam[2]], dtype=np.float64)



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project LiDAR axis points to check coordinate consistency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--calib_file", required=True, help="KITTI calib_cam_to_cam.txt")
    parser.add_argument("--r_vec", nargs=3, type=float, required=True, help="Rotation vector (rx ry rz) in radians.")
    parser.add_argument("--t_vec", nargs=3, type=float, required=True, help="Translation vector (tx ty tz) in meters.")
    parser.add_argument("--axis_len", type=float, default=10.0, help="Axis length in LiDAR meters.")
    args = parser.parse_args()

    K, R_rect, P_rect = load_kitti_calib(args.calib_file)
    r_vec = np.array(args.r_vec, dtype=np.float64)
    t_vec = np.array(args.t_vec, dtype=np.float64)
    R, _ = cv2.Rodrigues(r_vec)

    axes = {
        "+X": np.array([args.axis_len, 0.0, 0.0], dtype=np.float64),
        "-X": np.array([-args.axis_len, 0.0, 0.0], dtype=np.float64),
        "+Y": np.array([0.0, args.axis_len, 0.0], dtype=np.float64),
        "-Y": np.array([0.0, -args.axis_len, 0.0], dtype=np.float64),
        "+Z": np.array([0.0, 0.0, args.axis_len], dtype=np.float64),
        "-Z": np.array([0.0, 0.0, -args.axis_len], dtype=np.float64),
    }

    print("=== Axis Projection Check ===")
    print(f"K:\n{K}")
    print(f"R_rect:\n{R_rect}")
    print(f"P_rect:\n{P_rect}")
    print(f"r_vec: {r_vec}")
    print(f"t_vec: {t_vec}")
    print("")

    for name, p_lidar in axes.items():
        ok, proj = project_point(p_lidar, R_rect, P_rect, R, t_vec)
        if not ok:
            print(f"{name}: behind camera (p_cam.z <= 0) -> p_cam={proj}")
            continue
        u, v, z = proj
        print(f"{name}: u={u:.2f}, v={v:.2f}, z={z:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())