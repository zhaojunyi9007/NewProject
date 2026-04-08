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
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if ":" not in s:
                continue
            k, v = s.split(":", 1)
            k = k.strip()
            v = v.strip().split()[0] if v.strip() else ""
            if not v:
                continue
            try:
                out[k] = float(v)
            except ValueError:
                continue
    return out


def _parse_calib_pose(path: str):
    if not os.path.isfile(path):
        return None
    kv: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or ":" not in s:
                continue
            k, v = s.split(":", 1)
            kv[k.strip()] = v.strip()
    if "r" not in kv or "t" not in kv:
        return None
    try:
        r = [float(x) for x in kv["r"].split()[:3]]
        t = [float(x) for x in kv["t"].split()[:3]]
    except ValueError:
        return None
    if len(r) != 3 or len(t) != 3:
        return None
    return {"rvec": r, "tvec": t}


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段3] 两阶段标定优化")
    print("=" * 40)

    image_root = context.paths.get("image_features") if context.paths else ""
    if not image_root:
        image_root = context.config["data"]["image_features_output_dir"]
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
    sem_cfg = context.config.get("semantic_calib") or {}
    sem_enabled = bool(sem_cfg.get("enabled", False))
    img_cfg = context.config.get("image_features") or {}
    sem_classes = list(img_cfg.get("semantic_classes", []) or [])
    cw = sem_cfg.get("class_weights") or {}
    class_weights = [float(cw.get(name, 1.0)) for name in sem_classes]
    pyr = sem_cfg.get("pyramid_scales") or [1.0, 0.5, 0.25]
    pyramid_scales_s = ",".join(str(float(x)) for x in pyr)
    class_weights_s = ",".join(str(float(x)) for x in class_weights) if class_weights else ""

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
    from export_semantic_probs_bin import npy_to_edgecalib_bin  # noqa: E402

    for frame_id in context.frame_ids:
        feature_base = os.path.join(lidar_dir, f"{frame_id:010d}")
        frame_dir = os.path.join(image_root, f"{frame_id:010d}")
        sam_base = os.path.join(frame_dir, "optimizer")
        output_file = os.path.join(calib_dir, f"{frame_id:010d}_calib_result.txt")

        if not os.path.exists(f"{feature_base}_points.txt"):
            print(f"[Warning] 特征文件不存在，跳过帧 {frame_id:010d}")
            continue

        print(f"\n优化帧 {frame_id:010d}...")
        print(f"  logical_frame_id={frame_id:010d}")
        print(f"  feature_base={feature_base}")
        print(f"  image_features_dir={frame_dir}")
        print(f"  optimizer_base={sam_base}")
        r_use, t_use = list(init_r), list(init_t)
        if bool(bev_cfg.get("enabled", False)) and frame_id in bev_by_frame:
            pose_bev = bev_by_frame[frame_id]
            r_use = pose_bev["rvec"]
            t_use = pose_bev["tvec"]
            print("[Info] 本帧使用 BEV 粗初始化位姿作为 optimizer 初值")

        # Phase B6: pass semantic inputs (probabilities + semantic points + BEV init pose) to optimizer CLI.
        sem_npy = os.path.join(frame_dir, "semantic_probs.npy")
        sem_bin = os.path.join(frame_dir, "semantic_probs.edgecalib.bin")
        sem_pts = f"{feature_base}_semantic_points.txt"
        bev_pose = ""
        if bool(bev_cfg.get("enabled", False)) and context.paths and context.paths.get("bev_init"):
            cand = os.path.join(context.paths["bev_init"], f"{frame_id:010d}", "pose_after_bev.txt")
            if os.path.isfile(cand):
                bev_pose = cand

        use_sem = sem_enabled and os.path.isfile(sem_npy) and os.path.isfile(sem_pts)
        if use_sem:
            ok = npy_to_edgecalib_bin(sem_npy, sem_bin)
            if not ok:
                print("[Warning] semantic_probs.npy 导出 EDGESEM1 失败，回退 legacy optimizer 调用")
                use_sem = False

        if use_sem:
            cmd = [
                os.path.join(_REPO_ROOT, "build", "optimizer"),
                "--lidar_feature_base",
                feature_base,
                "--sam_feature_base",
                sam_base,
                "--calib_file",
                calib_file if os.path.exists(calib_file) else "",
                "--init_rx",
                str(r_use[0]),
                "--init_ry",
                str(r_use[1]),
                "--init_rz",
                str(r_use[2]),
                "--init_tx",
                str(t_use[0]),
                "--init_ty",
                str(t_use[1]),
                "--init_tz",
                str(t_use[2]),
                "--output_file",
                output_file,
                "--semantic_probs",
                sem_bin,
                "--lidar_semantic_points",
                sem_pts,
                "--semantic_js_weight",
                str(float(sem_cfg.get("semantic_js_weight", 3.0))),
                "--histogram_weight",
                str(float(sem_cfg.get("histogram_weight", 0.5))),
                "--pyramid_scales",
                pyramid_scales_s,
            ]
            if class_weights_s:
                cmd.extend(["--class_weights", class_weights_s])
            mode = str(sem_cfg.get("optimize_mode", "full_calib") or "full_calib")
            cmd.extend(["--mode", mode])
            if bev_pose:
                cmd.extend(["--init_pose_from_bev", bev_pose])
        else:
            if sem_enabled:
                print("[Warning] semantic_calib.enabled=true 但缺少 semantic_probs.npy 或 semantic_points.txt，回退 legacy")
            cmd = [
                os.path.join(_REPO_ROOT, "build", "optimizer"),
                feature_base,
                sam_base,
                calib_file if os.path.exists(calib_file) else "",
                str(r_use[0]),
                str(r_use[1]),
                str(r_use[2]),
                str(t_use[0]),
                str(t_use[1]),
                str(t_use[2]),
                output_file,
            ]
            if history_file:
                cmd.append(history_file)

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
