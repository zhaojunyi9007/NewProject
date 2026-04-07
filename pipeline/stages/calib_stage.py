#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import sys
import time

from pipeline.context import RuntimeContext
from pipeline.datasets import get_adapter
from pipeline.optimizer.constraint_adapter import get_optimizer_constraint_adapter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _parse_calib_breakdown(path: str) -> dict:
    out: dict = {}
    score = None
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("#") and "Score:" in s:
                try:
                    score = float(s.split("Score:", 1)[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            if s.startswith("#") and "semantic_js_divergence:" in s:
                try:
                    out["semantic_js_divergence"] = float(s.split(":", 1)[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            if s.startswith("#") and "semantic_hist_similarity:" in s:
                try:
                    out["semantic_hist_similarity"] = float(s.split(":", 1)[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            if s.startswith("#") and "edge_term_norm:" in s:
                try:
                    out["edge_term_norm"] = float(s.split(":", 1)[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
    if score is not None:
        out["optimizer_score"] = score
    return out


def _parse_calib_pose(path: str):
    nums: list[float] = []
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for part in s.split():
                try:
                    nums.append(float(part))
                except ValueError:
                    pass
    if len(nums) < 6:
        return None
    return {"rvec": nums[0:3], "tvec": nums[3:6]}


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段3] 两阶段标定优化")
    print("=" * 40)

    sam_dir = context.config["data"]["sam_output_dir"]
    lidar_dir = context.config["data"]["lidar_output_dir"]
    calib_dir = context.config["data"]["calib_output_dir"]
    calib_file = context.config["data"]["calib_file"]
    temporal_cfg = context.config.get("calibration", {}).get("temporal_validation", {})
    history_enabled = bool(temporal_cfg.get("enabled", False))
    history_cfg = temporal_cfg.get("history_file", "")
    history_file = ""
    if history_enabled and history_cfg:
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

    bev_cfg = context.config.get("bev") or {}
    bev_by_frame = getattr(context, "bev_pose_by_frame", None) or {}

    adapter = get_optimizer_constraint_adapter(context.config)
    optimizer_env, has_ab_overrides = adapter.build_env(context.config, os.environ.copy())
    for k, v in ds.get_optimizer_env().items():
        optimizer_env[k] = v

    print(f"[Info] 优化约束适配器: {adapter.name}")
    if has_ab_overrides:
        print("[Info] 已加载 calibration.ab_experiment 参数并传递给 optimizer")
    if history_file:
        print(f"[Info] 时序平滑已启用，history_file={history_file}")
    else:
        print("[Info] 时序平滑已禁用，不加载历史标定结果")

    sys.path.insert(0, os.path.join(_REPO_ROOT, "tools"))
    from score_breakdown_io import write_unified_debug_json  # noqa: E402

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
        r_use, t_use = list(init_r), list(init_t)
        if bool(bev_cfg.get("enabled", False)) and frame_id in bev_by_frame:
            pose_bev = bev_by_frame[frame_id]
            r_use = pose_bev["rvec"]
            t_use = pose_bev["tvec"]
            print("[Info] 本帧使用 BEV 粗初始化位姿作为 optimizer 初值")
        cmd = [
            "./build/optimizer",
            feature_base,
            sam_base,
            calib_file if os.path.exists(calib_file) else "",
            str(r_use[0]), str(r_use[1]), str(r_use[2]),
            str(t_use[0]), str(t_use[1]), str(t_use[2]),
            output_file,
        ]
        if history_file:
            cmd.append(history_file)
        if not os.path.isabs(cmd[0]):
            cmd[0] = os.path.join(_REPO_ROOT, "build", "optimizer")
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, env=optimizer_env, cwd=_REPO_ROOT)
        elapsed = time.perf_counter() - t0

        pose_out = _parse_calib_pose(output_file)
        if pose_out:
            context.current_pose_semantic = pose_out
        br = _parse_calib_breakdown(output_file)
        write_unified_debug_json(
            os.path.join(calib_dir, f"{frame_id:010d}_debug_score_breakdown.json"),
            stage="calib",
            frame_id=f"{frame_id:010d}",
            input_pose={"rvec": list(r_use), "tvec": list(t_use)},
            output_pose=pose_out,
            breakdown=br,
            elapsed_sec=elapsed,
            meta={"optimizer_argv_len": len(cmd)},
        )

    print(f"\n[完成] 标定结果已保存到: {calib_dir}")
