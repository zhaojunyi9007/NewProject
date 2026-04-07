#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys

from pipeline.context import RuntimeContext
from pipeline.datasets import get_adapter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TOOLS = os.path.join(_REPO_ROOT, "tools")
_LIDAR_EXTRACTOR = os.path.join(_REPO_ROOT, "build", "lidar_extractor")


def _pack_bev_npz(output_base: str) -> None:
    script = os.path.join(_TOOLS, "pack_lidar_bev_npz.py")
    if not os.path.isfile(script):
        print(f"[Warning] 未找到 {script}，跳过 bev_maps.npz 打包")
        return
    r = subprocess.run([sys.executable, script, output_base], check=False)
    if r.returncode != 0:
        print(f"[Warning] bev_maps.npz 打包失败 (exit={r.returncode})，可检查 {output_base}_bev_meta.json")


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段2] LiDAR特征提取 (含NDT多帧融合)")
    print("=" * 40)

    output_dir = context.config["data"]["lidar_output_dir"]
    fusion_window = context.config["frames"]["fusion_window"]
    lidar_cfg = context.config.get("lidar", {})
    ndt_cfg = lidar_cfg.get("ndt", {}) if isinstance(lidar_cfg.get("ndt", {}), dict) else {}
    temporal_filter_cfg = (
        lidar_cfg.get("temporal_filter", {}) if isinstance(lidar_cfg.get("temporal_filter", {}), dict) else {}
    )
    phase3_cfg = lidar_cfg.get("phase3", {}) if isinstance(lidar_cfg.get("phase3", {}), dict) else {}
    dataset_meta = context.config.get("dataset") or {}
    adapter = get_adapter(context.config)

    extractor_env = os.environ.copy()
    ds_fmt = str(context.config.get("data", {}).get("dataset_format", "kitti") or "kitti").lower()
    extractor_env["EDGECALIB_DATASET_FORMAT"] = ds_fmt

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

    phase3_enabled = bool(phase3_cfg.get("enabled", False))
    ref_z = float(phase3_cfg.get("reference_plane_z", dataset_meta.get("reference_z", 0.0)))

    if phase3_enabled:
        extractor_env["EDGECALIB_LIDAR_PHASE3"] = "1"
        extractor_env["EDGECALIB_LIDAR_REFERENCE_PLANE_Z"] = str(ref_z)
        extractor_env["EDGECALIB_LIDAR_RAIL_BAND_ZMIN"] = str(phase3_cfg.get("rail_band_zmin", -0.5))
        extractor_env["EDGECALIB_LIDAR_RAIL_BAND_ZMAX"] = str(phase3_cfg.get("rail_band_zmax", 0.5))
        extractor_env["EDGECALIB_LIDAR_BEV_XMIN"] = str(phase3_cfg.get("bev_xmin", 0.0))
        extractor_env["EDGECALIB_LIDAR_BEV_XMAX"] = str(phase3_cfg.get("bev_xmax", 100.0))
        extractor_env["EDGECALIB_LIDAR_BEV_YMIN"] = str(phase3_cfg.get("bev_ymin", -25.0))
        extractor_env["EDGECALIB_LIDAR_BEV_YMAX"] = str(phase3_cfg.get("bev_ymax", 25.0))
        extractor_env["EDGECALIB_LIDAR_BEV_RESOLUTION"] = str(phase3_cfg.get("bev_resolution", 0.2))
        extractor_env["EDGECALIB_LIDAR_RAIL_BEV_PROB_THRESH"] = str(phase3_cfg.get("rail_bev_prob_thresh", 0.12))
        extractor_env["EDGECALIB_LIDAR_RAIL_BEV_MIN_CELLS"] = str(phase3_cfg.get("rail_bev_min_cells", 40.0))
        extractor_env["EDGECALIB_LIDAR_RAIL_BEV_SMOOTH_K"] = str(int(phase3_cfg.get("rail_bev_smooth_k", 5)))
        extractor_env["EDGECALIB_LIDAR_RAIL_BEV_MAX_SEGMENTS"] = str(int(phase3_cfg.get("rail_bev_max_segments", 6)))
        extractor_env["EDGECALIB_LIDAR_VERT_CLUSTER_TOL"] = str(phase3_cfg.get("vert_cluster_tolerance", 0.35))
        extractor_env["EDGECALIB_LIDAR_VERT_MIN_CLUSTER"] = str(int(phase3_cfg.get("vert_min_cluster", 25)))
        if "use_legacy_range_image" in phase3_cfg:
            extractor_env["EDGECALIB_LIDAR_USE_LEGACY_RANGE_IMAGE"] = (
                "1" if phase3_cfg.get("use_legacy_range_image") else "0"
            )
        if phase3_cfg.get("bev_edge_gradient_ratio") is not None:
            extractor_env["EDGECALIB_LIDAR_BEV_EDGE_GRAD_RATIO"] = str(
                phase3_cfg.get("bev_edge_gradient_ratio", 0.35)
            )

    for i, frame_id in enumerate(context.frame_ids):
        fusion_frames = []
        for j in range(max(0, i - fusion_window + 1), i + 1):
            if j < len(context.frame_ids):
                fusion_frames.append(context.frame_ids[j])

        cloud_paths = []
        missing = False
        for fid in fusion_frames:
            p = adapter.resolve_lidar(fid)
            if not p or not os.path.exists(p):
                missing = True
                break
            cloud_paths.append(p)
        if missing:
            print(f"[Warning] 部分点云文件不存在，跳过帧 {frame_id:010d}")
            continue

        output_base = os.path.join(output_dir, f"{frame_id:010d}")

        print(f"\n处理帧 {frame_id:010d}，融合 {len(cloud_paths)} 帧...")
        print(f"  logical_frame_id={frame_id:010d}")
        print(f"  source_lidar={cloud_paths[-1] if cloud_paths else None}")
        print(f"  融合帧: {[f'{fid:010d}' for fid in fusion_frames]}")
        print(f"  融合点云: {cloud_paths}")
        if phase3_enabled:
            print("  [Phase3] BEV + 语义点 + 聚类立柱 / BEV 多段轨道")

        lidar_bin = _LIDAR_EXTRACTOR if os.path.isfile(_LIDAR_EXTRACTOR) else os.path.join("build", "lidar_extractor")
        cmd = [lidar_bin]
        if phase3_enabled:
            if phase3_cfg.get("save_semantic_points", True):
                cmd.append("--save_semantic_points")
            if phase3_cfg.get("save_bev_maps", True):
                cmd.append("--save_bev_maps")
        cmd.extend(cloud_paths)
        cmd.append(output_base)

        subprocess.run(cmd, check=True, env=extractor_env)

        if phase3_enabled and phase3_cfg.get("save_bev_maps", True):
            _pack_bev_npz(output_base)

    print(f"\n[完成] LiDAR特征已保存到: {output_dir}")
