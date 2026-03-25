#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from pipeline.context import RuntimeContext
from pipeline.feature_plugin_registry import get_feature_plugin
from python.features.interfaces import FeatureFrameContext


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段1] SAM图像特征提取")
    print("=" * 40)

    image_dir = context.config["data"]["image_dir"]
    output_dir = context.config["data"]["sam_output_dir"]
    plugin = get_feature_plugin(context.config)
    print(f"[Info] SAM特征插件: {plugin.name}")

    for frame_id in context.frame_ids:
        img_path = os.path.join(image_dir, f"{frame_id:010d}.png")
        if not os.path.exists(img_path):
            print(f"[Warning] 图像不存在: {img_path}")
            continue

        print(f"\n处理帧 {frame_id:010d}...")
        plugin.run_frame(
            FeatureFrameContext(
                frame_id=frame_id,
                image_path=img_path,
                output_dir=output_dir,
                config=context.config,
            )
        )

    print(f"\n[完成] SAM特征已保存到: {output_dir}")
