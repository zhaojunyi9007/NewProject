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


def _existing_path(path: str) -> str:
    p = str(path or "").strip()
    if not p:
        return ""
    candidates = [p]
    if not os.path.isabs(p):
        candidates.append(os.path.join(_REPO_ROOT, p))
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


def _select_visualization_calib_file(calib_meta: dict[str, str], fallback_calib_file: str) -> tuple[str, str, bool]:
    optimizer_calib = _existing_path(calib_meta.get("optimizer_calib_file", ""))
    if optimizer_calib:
        return optimizer_calib, calib_meta.get("optimizer_intrinsics_source", "optimizer_calib_file") or "optimizer_calib_file", True
    fallback = _existing_path(fallback_calib_file)
    return fallback, "config_calib_file" if fallback else "missing", False


def _calib_meta_valid(kv: dict[str, str]) -> bool:
    try:
        return float((kv.get("final_pose_valid", "1") or "1").split()[0]) >= 0.5
    except (TypeError, ValueError):
        return True


def _initial_pose_from_context(context: RuntimeContext, adapter):
    ext = None
    label_cfg = context.config.get("label_assist") or {}
    if bool(label_cfg.get("enabled", False)) and bool(label_cfg.get("use_openlabel_extrinsic", True)) and hasattr(adapter, "load_label_assist_extrinsic"):
        try:
            ext = adapter.load_label_assist_extrinsic()
        except Exception:
            ext = None
    if ext is None and hasattr(adapter, "load_initial_extrinsic"):
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
        out_root, _ = os.path.splitext(output_path)

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
        visualization_calib_file, visualization_intrinsics_source, optimizer_calib_file_found = _select_visualization_calib_file(
            calib_meta,
            calib_file,
        )
        print(f"  visualization_intrinsics_source={visualization_intrinsics_source}")
        print(f"  visualization_calib_file={visualization_calib_file}")
        print(f"  optimizer_calib_file_found={1 if optimizer_calib_file_found else 0}")
        cmd = [
            sys.executable, "tools/visualize.py",
            "--img", img_path,
            "--feature_base", feature_base,
            "--calib_file", visualization_calib_file,
            "--dataset_format", ds_fmt,
            "--r_vec", *r_vec,
            "--t_vec", *t_vec,
            "--output", output_path,
            "--overlay-lidar-rail-samples",
        ]
        if bool(vis_cfg.get("overlay_json_rail", False)):
            cmd.append("--overlay-json-rail")
        if img_sensor:
            cmd.extend(["--image_sensor", img_sensor])
        label_cfg = context.config.get("label_assist") or {}
        strong_label_features = f"{feature_base}_label_strong_features.tsv"
        if bool(label_cfg.get("strong_label_static_overlay_enabled", True)) and os.path.isfile(strong_label_features):
            cmd.extend([
                "--strong-label-features",
                strong_label_features,
                "--strong-static-overlay-output",
                out_root + "_diag_static_strong.png",
            ])
        if bool(vis_cfg.get("lidar_projection_enabled", True)):
            cmd.extend([
                "--lidar-projection-output",
                out_root + "_lidar_projection.png",
                "--lidar-projection-point-source",
                str(vis_cfg.get("lidar_projection_point_source", "all") or "all"),
                "--lidar-projection-color-mode",
                str(vis_cfg.get("lidar_projection_color_mode", "depth") or "depth"),
                "--lidar-projection-max-points",
                str(int(vis_cfg.get("lidar_projection_max_points", 120000) or 120000)),
                "--lidar-projection-point-radius",
                str(int(vis_cfg.get("lidar_projection_point_radius", 2) or 2)),
                "--lidar-projection-background",
                str(vis_cfg.get("lidar_projection_background", "grayscale") or "grayscale"),
                "--lidar-projection-background-alpha",
                str(float(vis_cfg.get("lidar_projection_background_alpha", 0.55))),
                "--lidar-projection-depth-min-m",
                str(float(vis_cfg.get("lidar_projection_depth_min_m", 5.0))),
                "--lidar-projection-depth-max-m",
                str(float(vis_cfg.get("lidar_projection_depth_max_m", 120.0))),
                "--pose-source",
                used_pose_source,
            ])

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
