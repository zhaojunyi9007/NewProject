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


def load_kitti_calib(calib_file: str) -> np.ndarray:
    if not calib_file or not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")
    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("P2:") or line.startswith("P_rect_02:"):
                parts = line.split(":")
                values = list(map(float, parts[1].strip().split()))
                if len(values) == 12:
                    return np.array(
                        [
                            [values[0], values[1], values[2]],
                            [values[4], values[5], values[6]],
                            [values[8], values[9], values[10]],
                        ],
                        dtype=np.float64,
                    )
    raise ValueError("Failed to parse P2/P_rect_02 from calibration file.")


def project_point(p_lidar: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[bool, np.ndarray]:
    p_cam = R @ p_lidar + t
    if p_cam[2] <= 0:
        return False, p_cam
    uv = K @ p_cam
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

    K = load_kitti_calib(args.calib_file)
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
    print(f"r_vec: {r_vec}")
    print(f"t_vec: {t_vec}")
    print("")

    for name, p_lidar in axes.items():
        ok, proj = project_point(p_lidar, K, R, t_vec)
        if not ok:
            print(f"{name}: behind camera (p_cam.z <= 0) -> p_cam={proj}")
            continue
        u, v, z = proj
        print(f"{name}: u={u:.2f}, v={v:.2f}, z={z:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())