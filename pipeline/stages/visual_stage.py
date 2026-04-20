#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys

from pipeline.context import RuntimeContext
from pipeline.datasets import get_adapter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段4] 结果可视化")
    print("=" * 40)

    lidar_dir = context.config["data"]["lidar_output_dir"]
    calib_dir = context.config["data"]["calib_output_dir"]
    visual_dir = context.config["data"]["visual_output_dir"]
    calib_file = context.config["data"].get("calib_file", "")
    ds_fmt = str(context.config.get("data", {}).get("dataset_format", "osdar23") or "osdar23").lower()
    img_sensor = str(context.config.get("data", {}).get("image_sensor", "") or "")
    adapter = get_adapter(context.config)

    for frame_id in context.frame_ids:
        img_path = adapter.resolve_image(frame_id)
        feature_base = os.path.join(lidar_dir, f"{frame_id:010d}")
        calib_result_file = os.path.join(calib_dir, f"{frame_id:010d}_calib_result.txt")
        output_path = os.path.join(visual_dir, f"{frame_id:010d}_result.png")

        if not img_path or not os.path.exists(img_path):
            print(f"[Warning] 图像不存在，跳过帧 {frame_id:010d}: {img_path}")
            continue

        if not os.path.exists(calib_result_file):
            print(f"[Warning] 标定结果不存在，跳过帧 {frame_id:010d}")
            continue

        kv: dict[str, str] = {}
        with open(calib_result_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if ":" in s:
                    k, v = s.split(":", 1)
                    kv[k.strip()] = v.strip()
        if "r" not in kv or "t" not in kv:
            print(f"[Warning] 标定结果格式异常(缺少 r/t)，跳过帧 {frame_id:010d}: {calib_result_file}")
            continue
        r_vec = kv["r"].split()
        t_vec = kv["t"].split()
        if len(r_vec) != 3 or len(t_vec) != 3:
            print(f"[Warning] 标定结果格式异常(R/T维度错误)，跳过帧 {frame_id:010d}: {calib_result_file}")
            continue

        print(f"可视化帧 {frame_id:010d}...")
        print(f"  logical_frame_id={frame_id:010d}")
        print(f"  source_image={img_path}")
        print(f"  feature_base={feature_base}")
        cmd = [
            sys.executable, "tools/visualize.py",
            "--img", img_path,
            "--feature_base", feature_base,
            "--calib_file", calib_file if os.path.exists(calib_file) else "",
            "--dataset_format", ds_fmt,
            "--r_vec", *r_vec,
            "--t_vec", *t_vec,
            "--output", output_path,
        ]
        if img_sensor:
            cmd.extend(["--image_sensor", img_sensor])

        vis_cfg = context.config.get("visualization") or {}
        if bool(vis_cfg.get("enable_diag_panels", True)):
            img_feat = os.path.join(
                context.config["data"].get("image_features_output_dir", "") or "",
                f"{frame_id:010d}",
            )
            ref_dir = ""
            if context.paths:
                ref_dir = context.paths.get("refinement", "") or ""
            if not ref_dir:
                ref_dir = context.config["data"].get("refinement_output_dir", "") or ""
            cmd.extend(
                [
                    "--diag",
                    "bev",
                    "--diag",
                    "semantic",
                    "--diag",
                    "refine",
                    "--image_features_frame",
                    os.path.abspath(img_feat),
                    "--sam_frame_dir",
                    os.path.abspath(img_feat),
                    "--refinement_dir",
                    os.path.abspath(ref_dir) if ref_dir else "",
                ]
            )

        subprocess.run(cmd, check=True, cwd=_REPO_ROOT)

    print(f"\n[完成] 可视化结果已保存到: {visual_dir}")
