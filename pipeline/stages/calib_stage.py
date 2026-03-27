#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

import cv2
import numpy as np

from pipeline.context import RuntimeContext
from pipeline.optimizer_constraint_adapter import get_optimizer_constraint_adapter


def _load_velo_to_cam_extrinsic(config):
    calib_path = config["data"].get("velo_to_cam_file", "")
    if not calib_path:
        return None
    if not os.path.exists(calib_path):
        print(f"[Warning] LiDAR外参文件不存在: {calib_path}")
        return None

    r_vals = None
    t_vals = None
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("R:"):
                r_vals = [float(x) for x in line.replace("R:", "").split()]
            elif line.startswith("T:"):
                t_vals = [float(x) for x in line.replace("T:", "").split()]

    if not r_vals or not t_vals or len(r_vals) != 9 or len(t_vals) != 3:
        print(f"[Warning] LiDAR外参文件格式异常: {calib_path}")
        return None

    r_mat = np.array(r_vals, dtype=np.float64).reshape(3, 3)
    t_vec = np.array(t_vals, dtype=np.float64)
    r_vec, _ = cv2.Rodrigues(r_mat)
    return r_vec.reshape(-1).tolist(), t_vec.reshape(-1).tolist()


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段3] 两阶段标定优化")
    print("=" * 40)

    sam_dir = context.config["data"]["sam_output_dir"]
    lidar_dir = context.config["data"]["lidar_output_dir"]
    calib_dir = context.config["data"]["calib_output_dir"]
    calib_file = context.config["data"]["calib_file"]
    history_cfg = context.config.get("calibration", {}).get("temporal_validation", {}).get("history_file", "")
    history_file = ""
    if history_cfg:
        history_file = str(history_cfg)
        if not os.path.isabs(history_file):
            history_file = os.path.join(calib_dir, history_file)
        os.makedirs(os.path.dirname(history_file) or ".", exist_ok=True)

    init_r = context.config["calibration"]["initial_extrinsic"]["rotation"]
    init_t = context.config["calibration"]["initial_extrinsic"]["translation"]
    velo_to_cam = _load_velo_to_cam_extrinsic(context.config)
    if velo_to_cam:
        init_r, init_t = velo_to_cam
        print("[Info] 使用calib_velo_to_cam.txt中的R/T作为初始外参")

    adapter = get_optimizer_constraint_adapter(context.config)
    optimizer_env, has_ab_overrides = adapter.build_env(context.config, os.environ.copy())
    print(f"[Info] 优化约束适配器: {adapter.name}")
    if has_ab_overrides:
        print("[Info] 已加载 calibration.ab_experiment 参数并传递给 optimizer")

    for frame_id in context.frame_ids:
        feature_base = os.path.join(lidar_dir, f"{frame_id:010d}")
        sam_base = os.path.join(sam_dir, f"{frame_id:010d}")
        output_file = os.path.join(calib_dir, f"{frame_id:010d}_calib_result.txt")

        if not os.path.exists(f"{feature_base}_points.txt"):
            print(f"[Warning] 特征文件不存在，跳过帧 {frame_id:010d}")
            continue

        print(f"\n优化帧 {frame_id:010d}...")
        cmd = [
            "./build/optimizer",
            feature_base,
            sam_base,
            calib_file if os.path.exists(calib_file) else "",
            str(init_r[0]), str(init_r[1]), str(init_r[2]),
            str(init_t[0]), str(init_t[1]), str(init_t[2]),
            output_file,
        ]
        if history_file:
            cmd.append(history_file)
        subprocess.run(cmd, check=True, env=optimizer_env)

    print(f"\n[完成] 标定结果已保存到: {calib_dir}")