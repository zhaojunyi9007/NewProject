#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

from pipeline.context import RuntimeContext
from pipeline.dataset_resolver import get_dataset_resolver


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段4] 结果可视化")
    print("=" * 40)

    lidar_dir = context.config["data"]["lidar_output_dir"]
    calib_dir = context.config["data"]["calib_output_dir"]
    visual_dir = context.config["data"]["visual_output_dir"]
    calib_file = context.config["data"].get("calib_file", "")
    resolver = get_dataset_resolver(context.config)

    for frame_id in context.frame_ids:
        img_path = resolver.resolve_image(frame_id)
        feature_base = os.path.join(lidar_dir, f"{frame_id:010d}")
        calib_result_file = os.path.join(calib_dir, f"{frame_id:010d}_calib_result.txt")
        output_path = os.path.join(visual_dir, f"{frame_id:010d}_result.png")

        if not img_path or not os.path.exists(img_path):
            print(f"[Warning] 图像不存在，跳过帧 {frame_id:010d}: {img_path}")
            continue

        if not os.path.exists(calib_result_file):
            print(f"[Warning] 标定结果不存在，跳过帧 {frame_id:010d}")
            continue

        with open(calib_result_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) < 3:
            print(f"[Warning] 标定结果格式异常(行数不足)，跳过帧 {frame_id:010d}: {calib_result_file}")
            continue

        r_vec = lines[1].split()
        t_vec = lines[2].split()
        if len(r_vec) != 3 or len(t_vec) != 3:
            print(f"[Warning] 标定结果格式异常(R/T维度错误)，跳过帧 {frame_id:010d}: {calib_result_file}")
            continue

        print(f"可视化帧 {frame_id:010d}...")
        print(f"  logical_frame_id={frame_id:010d}")
        print(f"  source_image={img_path}")
        print(f"  feature_base={feature_base}")
        cmd = [
            "python", "visual_result.py",
            "--img", img_path,
            "--feature_base", feature_base,
            "--calib_file", calib_file if os.path.exists(calib_file) else "",
            "--r_vec", *r_vec,
            "--t_vec", *t_vec,
            "--output", output_path,
        ]
        subprocess.run(cmd, check=True)

    print(f"\n[完成] 可视化结果已保存到: {visual_dir}")
