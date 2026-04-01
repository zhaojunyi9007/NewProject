#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

from pipeline.context import RuntimeContext
from pipeline.datasets import get_adapter
from pipeline.optimizer.constraint_adapter import get_optimizer_constraint_adapter


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

    ds = get_adapter(context.config)
    ext = ds.load_initial_extrinsic()
    if ext:
        init_r, init_t = ext
        ds_fmt = str(context.config.get("data", {}).get("dataset_format", "kitti") or "kitti").lower()
        if ds_fmt in {"osdar23", "osdar"}:
            cam_folder = str(context.config.get("data", {}).get("image_sensor", "rgb_center") or "rgb_center")
            print(f"[Info] 使用OSDaR23 calibration.txt 读取初始外参, camera={cam_folder}")
        else:
            print("[Info] 使用calib_velo_to_cam.txt中的R/T作为初始外参")

    adapter = get_optimizer_constraint_adapter(context.config)
    optimizer_env, has_ab_overrides = adapter.build_env(context.config, os.environ.copy())
    for k, v in ds.get_optimizer_env().items():
        optimizer_env[k] = v

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
        print(f"  logical_frame_id={frame_id:010d}")
        print(f"  feature_base={feature_base}")
        print(f"  sam_base={sam_base}")
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
