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
    lidar_cfg = context.config.get("lidar", {})
    ndt_cfg = lidar_cfg.get("ndt", {}) if isinstance(lidar_cfg.get("ndt", {}), dict) else {}
    temporal_filter_cfg = lidar_cfg.get("temporal_filter", {}) if isinstance(lidar_cfg.get("temporal_filter", {}), dict) else {}

    extractor_env = os.environ.copy()
    env_map = {
        "EDGECALIB_LIDAR_VOXEL_SIZE": lidar_cfg.get("voxel_size"),
        "EDGECALIB_LIDAR_NORMAL_K": lidar_cfg.get("normal_k"),
        "EDGECALIB_LIDAR_PCA_RADIUS": lidar_cfg.get("pca_radius"),
        "EDGECALIB_LIDAR_NDT_TRANS_EPS": ndt_cfg.get("transformation_epsilon"),
        "EDGECALIB_LIDAR_NDT_STEP_SIZE": ndt_cfg.get("step_size"),
        "EDGECALIB_LIDAR_NDT_RESOLUTION": ndt_cfg.get("resolution"),
        "EDGECALIB_LIDAR_NDT_MAX_ITERS": ndt_cfg.get("max_iterations"),
        "EDGECALIB_LIDAR_TEMPORAL_POS_THRESH": temporal_filter_cfg.get("position_threshold"),
        "EDGECALIB_LIDAR_TEMPORAL_PROJ_THRESH": temporal_filter_cfg.get("position_threshold"),
        "EDGECALIB_LIDAR_TEMPORAL_STATIC_WEIGHT": temporal_filter_cfg.get("static_weight"),
        "EDGECALIB_LIDAR_TEMPORAL_DYNAMIC_WEIGHT": temporal_filter_cfg.get("dynamic_weight"),
        "EDGECALIB_LIDAR_GROUND_Z_MIN": lidar_cfg.get("ground_z_min"),
        "EDGECALIB_LIDAR_GROUND_Z_MAX": lidar_cfg.get("ground_z_max"),
        "EDGECALIB_LIDAR_RAIL_RANSAC_THRESHOLD": lidar_cfg.get("rail_ransac_threshold"),
        "EDGECALIB_LIDAR_RAIL_MAX_LINES": lidar_cfg.get("rail_max_lines"),
        "EDGECALIB_LIDAR_POLE_Z_MIN": lidar_cfg.get("pole_z_min"),
        "EDGECALIB_LIDAR_POLE_Z_MAX": lidar_cfg.get("pole_z_max"),
        "EDGECALIB_LIDAR_POLE_RANSAC_THRESHOLD": lidar_cfg.get("pole_ransac_threshold"),
        "EDGECALIB_LIDAR_POLE_VERTICAL_TOLERANCE": lidar_cfg.get("pole_vertical_tolerance"),
        "EDGECALIB_LIDAR_POLE_MAX_LINES": lidar_cfg.get("pole_max_lines"),
    }
    for key, value in env_map.items():
        if value is not None:
            extractor_env[key] = str(value)

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
        subprocess.run(cmd, check=True, env=extractor_env)

    print(f"\n[完成] LiDAR特征已保存到: {output_dir}")