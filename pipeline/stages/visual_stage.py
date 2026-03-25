#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

from pipeline.context import RuntimeContext


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段4] 结果可视化")
    print("=" * 40)

    image_dir = context.config["data"]["image_dir"]
    lidar_dir = context.config["data"]["lidar_output_dir"]
    calib_dir = context.config["data"]["calib_output_dir"]
    visual_dir = context.config["data"]["visual_output_dir"]
    calib_file = context.config["data"].get("calib_file", "")

    for frame_id in context.frame_ids:
        img_path = os.path.join(image_dir, f"{frame_id:010d}.png")
        feature_base = os.path.join(lidar_dir, f"{frame_id:010d}")
        calib_result_file = os.path.join(calib_dir, f"{frame_id:010d}_calib_result.txt")
        output_path = os.path.join(visual_dir, f"{frame_id:010d}_result.png")

        if not os.path.exists(calib_result_file):
            print(f"[Warning] 标定结果不存在，跳过帧 {frame_id:010d}")
            continue

        with open(calib_result_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            r_vec = lines[1].strip().split()
            t_vec = lines[2].strip().split()

        print(f"可视化帧 {frame_id:010d}...")
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
