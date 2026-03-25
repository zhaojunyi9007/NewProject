#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

from pipeline.context import RuntimeContext


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段1] SAM图像特征提取")
    print("=" * 40)

    image_dir = context.config["data"]["image_dir"]
    output_dir = context.config["data"]["sam_output_dir"]
    checkpoint = context.config["sam"]["checkpoint_path"]

    for frame_id in context.frame_ids:
        img_path = os.path.join(image_dir, f"{frame_id:010d}.png")
        if not os.path.exists(img_path):
            print(f"[Warning] 图像不存在: {img_path}")
            continue

        print(f"\n处理帧 {frame_id:010d}...")
        cmd = [
            "python", "python/run_sam.py",
            "--image", img_path,
            "--checkpoint", checkpoint,
            "--output_dir", output_dir,
        ]
        subprocess.run(cmd, check=True)

    print(f"\n[完成] SAM特征已保存到: {output_dir}")