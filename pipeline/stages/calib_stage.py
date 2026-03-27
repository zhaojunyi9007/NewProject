#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

import cv2
import numpy as np

from pipeline.context import RuntimeContext
from pipeline.optimizer_constraint_adapter import get_optimizer_constraint_adapter


def _parse_velo_to_cam_file(calib_path):
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


def _parse_cam_to_cam_file(calib_path):
    if not calib_path or not os.path.exists(calib_path):
        return {}
    parsed = {}
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
                # Skip metadata lines such as calib_time: 09-Jan-2012 ...
                continue
            if key in {"R_rect_00", "R0_rect", "R_rect_02", "R_02"} and len(values) == 9:
                parsed[key] = np.array(values, dtype=np.float64).reshape(3, 3)
            elif key in {"T_02"} and len(values) == 3:
                parsed[key] = np.array(values, dtype=np.float64)
            elif key in {"P_rect_00", "P0", "P_rect_02", "P2"} and len(values) == 12:
                parsed[key] = np.array(values, dtype=np.float64).reshape(3, 4)
    return parsed


def _compose_extrinsic_to_target_camera(r_cam0, t_cam0, cam_to_cam, target_camera):
    if target_camera != "cam2":
        return r_cam0, t_cam0

    r_02 = cam_to_cam.get("R_02")
    t_02 = cam_to_cam.get("T_02")
    if r_02 is not None and t_02 is not None:
        r_target = r_02 @ r_cam0
        t_target = r_02 @ t_cam0 + t_02
        return r_target, t_target

    # Fallback: derive x-baseline from projection matrices (rectified approximation).
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


def _load_velo_to_cam_extrinsic(config):
    velo_path = config["data"].get("velo_to_cam_file", "")
    parsed = _parse_velo_to_cam_file(velo_path)
    if parsed is None:
        return None

    r_cam0, t_cam0 = parsed
    init_cfg = config.get("calibration", {}).get("extrinsic_init", {})
    target_camera = str(init_cfg.get("target_camera", "cam2")).lower()
    cam_to_cam_path = config["data"].get("calib_file", "")
    cam_to_cam = _parse_cam_to_cam_file(cam_to_cam_path)
    r_target, t_target = _compose_extrinsic_to_target_camera(r_cam0, t_cam0, cam_to_cam, target_camera)

    r_vec, _ = cv2.Rodrigues(r_target)
    print(f"[Info] 初始外参目标相机: {target_camera}")
    return r_vec.reshape(-1).tolist(), t_target.reshape(-1).tolist()


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