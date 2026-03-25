#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

from pipeline.context import RuntimeContext


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段2] LiDAR特征提取 (含NDT多帧融合)")
    print("=" * 40)

    velodyne_dir = context.config["data"]["velodyne_dir"]
    output_dir = context.config["data"]["lidar_output_dir"]
    fusion_window = context.config["frames"]["fusion_window"]

    for i, frame_id in enumerate(context.frame_ids):
        fusion_frames = []
        for j in range(max(0, i - fusion_window + 1), i + 1):
            if j < len(context.frame_ids):
                fusion_frames.append(context.frame_ids[j])

        bin_paths = [os.path.join(velodyne_dir, f"{fid:010d}.bin") for fid in fusion_frames]
        if not all(os.path.exists(p) for p in bin_paths):
            print(f"[Warning] 部分点云文件不存在，跳过帧 {frame_id:010d}")
            continue

        output_base = os.path.join(output_dir, f"{frame_id:010d}")

        print(f"\n处理帧 {frame_id:010d}，融合 {len(bin_paths)} 帧...")
        print(f"  融合帧: {[f'{fid:010d}' for fid in fusion_frames]}")

        cmd = ["./build/lidar_extractor", *bin_paths, output_base]
        subprocess.run(cmd, check=True)

    print(f"\n[完成] LiDAR特征已保存到: {output_dir}")