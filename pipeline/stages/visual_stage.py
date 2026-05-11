#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys

from pipeline.context import RuntimeContext
from pipeline.datasets import get_adapter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _read_calib_meta(path: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if ":" in s:
                k, v = s.split(":", 1)
                kv[k.strip()] = v.strip()
    return kv


def _read_calib_pose(path: str):
    kv = _read_calib_meta(path)
    if "r" not in kv or "t" not in kv:
        return None
    r_vec = kv["r"].split()
    t_vec = kv["t"].split()
    return (r_vec, t_vec) if len(r_vec) == 3 and len(t_vec) == 3 else None


def _calib_meta_valid(kv: dict[str, str]) -> bool:
    try:
        return float((kv.get("final_pose_valid", "1") or "1").split()[0]) >= 0.5
    except (TypeError, ValueError):
        return True


def _initial_pose_from_context(context: RuntimeContext, adapter):
    ext = None
    if hasattr(adapter, "load_initial_extrinsic"):
        try:
            ext = adapter.load_initial_extrinsic()
        except Exception:
            ext = None
    if isinstance(ext, dict) and "rvec" in ext and "tvec" in ext:
        return ([str(float(x)) for x in ext["rvec"][:3]], [str(float(x)) for x in ext["tvec"][:3]])
    if isinstance(ext, (list, tuple)) and len(ext) >= 2:
        r0, t0 = ext[0], ext[1]
        if len(r0) >= 3 and len(t0) >= 3:
            return ([str(float(x)) for x in r0[:3]], [str(float(x)) for x in t0[:3]])
    cfg_ext = (context.config.get("calibration") or {}).get("initial_extrinsic") or {}
    r = cfg_ext.get("rotation", [0.0, 0.0, 0.0])
    t = cfg_ext.get("translation", [0.0, 0.0, 0.0])
    return ([str(float(x)) for x in r[:3]], [str(float(x)) for x in t[:3]])


def _read_window_pose(path: str):
    nums = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for part in s.split():
                nums.append(part)
    return (nums[:3], nums[3:6]) if len(nums) >= 6 else None


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段4] 结果可视化")
    print("=" * 40)

    lidar_dir = context.config["data"]["lidar_output_dir"]
    calib_dir = context.config["data"]["calib_output_dir"]
    visual_dir = context.config["data"]["visual_output_dir"]
    calib_file = context.config["data"].get("calib_file", "")
    ds_fmt = str(context.config.get("data", {}).get("dataset_format", "osdar23") or "osdar23").lower()
    img_sensor = str(context.config.get("data", {}).get("image_sensor", "rgb_center") or "rgb_center")
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

        vis_cfg = context.config.get("visualization") or {}
        pose_source = str(vis_cfg.get("pose_source", "refined") or "refined")
        calib_meta = _read_calib_meta(calib_result_file)
        calib_valid = _calib_meta_valid(calib_meta)
        pose = None
        used_pose_source = "calibration"
        invalid_reason = calib_meta.get("invalid_reason", "invalid_calibration_pose")
        reject_invalid = bool(vis_cfg.get("reject_invalid_pose", True))
        invalid_fallback = str(vis_cfg.get("invalid_pose_fallback", "initial") or "initial")
        if not calib_valid and reject_invalid and invalid_fallback == "initial":
            pose = _initial_pose_from_context(context, adapter)
            used_pose_source = "initial_due_to_invalid_calib"
        else:
            if pose_source == "refined":
                ref_root = (context.paths or {}).get("refinement") or context.config["data"].get("refinement_output_dir", "")
                refined_pose = os.path.join(ref_root, f"{frame_id:010d}_window_pose.txt")
                if os.path.isfile(refined_pose):
                    pose = _read_window_pose(refined_pose)
                    used_pose_source = "refined"
            if pose is None:
                pose = _read_calib_pose(calib_result_file)
                used_pose_source = "calibration"
        if pose is None:
            print(f"[Warning] 标定结果格式异常(缺少 r/t)，跳过帧 {frame_id:010d}: {calib_result_file}")
            continue
        r_vec, t_vec = pose

        print(f"可视化帧 {frame_id:010d}...")
        print(f"  visualization_pose_source={used_pose_source}")
        if not calib_valid:
            print(f"  invalid_calibration_pose=1 reason={invalid_reason}")
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
            "--overlay-json-rail",
            "--overlay-lidar-rail-samples",
        ]
        if img_sensor:
            cmd.extend(["--image_sensor", img_sensor])

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
