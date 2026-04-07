#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 4：BEV 粗初始化（调用 build/bev_initializer，写入 bev_init/<frame>/）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from pipeline.context import RuntimeContext
from pipeline.datasets import get_adapter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TOOLS = os.path.join(_REPO_ROOT, "tools")
_BUILD_BEV_INIT = os.path.join(_REPO_ROOT, "build", "bev_initializer")


def _parse_pose_after_bev(path: str):
    nums: list[float] = []
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
    print("[阶段 bev] BEV 粗初始化 (Phase 4)")
    print("=" * 40)

    bev_cfg = context.config.get("bev") or {}
    if not bool(bev_cfg.get("enabled", False)):
        print("[Info] bev.enabled=false，跳过")
        return

    out_root = context.paths.get("bev_init") or context.config.get("data", {}).get("bev_init_output_dir", "")
    if not out_root:
        print("[Error] 缺少 bev_init 输出目录")
        return
    os.makedirs(out_root, exist_ok=True)

    img_root = context.paths.get("image_features") or context.config.get("data", {}).get("image_features_output_dir", "")
    lidar_root = context.paths.get("lidar") or context.config.get("data", {}).get("lidar_output_dir", "")
    if not img_root or not lidar_root:
        print("[Error] 需要 image_features_output_dir 与 lidar_output_dir")
        return

    sys.path.insert(0, _TOOLS)
    from bev_export_image_bin import export_image_rail_bin  # noqa: E402
    from score_breakdown_io import write_unified_debug_json  # noqa: E402

    bev_init = str(bev_cfg.get("initializer_binary") or "").strip()
    exe = bev_init if bev_init and os.path.isfile(bev_init) else _BUILD_BEV_INIT
    if not os.path.isfile(exe):
        print(f"[Error] 未找到可执行文件 bev_initializer: {exe}（请先 cmake 编译）")
        return

    adapter = get_adapter(context.config)
    ext = adapter.load_initial_extrinsic()
    if not ext:
        ie = context.config.get("calibration", {}).get("initial_extrinsic", {})
        rvec = [float(x) for x in ie.get("rotation", [0.0, 0.0, 0.0])]
        tvec = [float(x) for x in ie.get("translation", [0.0, 0.0, 0.0])]
    else:
        rvec, tvec = ext
        rvec = [float(x) for x in rvec]
        tvec = [float(x) for x in tvec]

    env = os.environ.copy()
    env["EDGECALIB_BEV_YAW_MIN_DEG"] = str(bev_cfg.get("yaw_min_deg", -6.0))
    env["EDGECALIB_BEV_YAW_MAX_DEG"] = str(bev_cfg.get("yaw_max_deg", 6.0))
    env["EDGECALIB_BEV_YAW_STEP_DEG"] = str(bev_cfg.get("yaw_step_deg", 1.0))
    env["EDGECALIB_BEV_TX_MIN_M"] = str(bev_cfg.get("tx_min_m", -2.0))
    env["EDGECALIB_BEV_TX_MAX_M"] = str(bev_cfg.get("tx_max_m", 2.0))
    env["EDGECALIB_BEV_TY_MIN_M"] = str(bev_cfg.get("ty_min_m", -2.0))
    env["EDGECALIB_BEV_TY_MAX_M"] = str(bev_cfg.get("ty_max_m", 2.0))
    env["EDGECALIB_BEV_TRANS_STEP_M"] = str(bev_cfg.get("trans_step_m", 0.5))

    last_pose = None
    context.bev_pose_by_frame.clear()
    for frame_id in context.frame_ids:
        fid = f"{frame_id:010d}"
        frame_dir = os.path.join(out_root, fid)
        os.makedirs(frame_dir, exist_ok=True)

        pseudo_npz = os.path.join(os.path.abspath(img_root), fid, "pseudo_bev.npz")
        lidar_bin = os.path.join(os.path.abspath(lidar_root), f"{fid}_bev_channels.bin")
        if not os.path.isfile(pseudo_npz):
            print(f"[Warning] 缺少 {pseudo_npz}，跳过帧 {fid}")
            continue
        if not os.path.isfile(lidar_bin):
            print(f"[Warning] 缺少 {lidar_bin}（需 LiDAR Phase3 生成 bev），跳过帧 {fid}")
            continue

        img_bin = os.path.join(frame_dir, "image_rail_bev.bin")
        if not export_image_rail_bin(pseudo_npz, img_bin, bev_cfg):
            print(f"[Warning] 导出 image BEV 失败: {fid}")
            continue

        init_path = os.path.join(frame_dir, "init_pose.txt")
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("# rvec (3)\n")
            f.write(f"{rvec[0]} {rvec[1]} {rvec[2]}\n")
            f.write("# tvec (3)\n")
            f.write(f"{tvec[0]} {tvec[1]} {tvec[2]}\n")

        cmd = [exe, lidar_bin, img_bin, init_path, frame_dir, fid]
        print(f"  BEV init: frame={fid}")
        print(f"    {' '.join(cmd)}")
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, check=True, env=env, cwd=_REPO_ROOT)
        except subprocess.CalledProcessError as e:
            print(f"[Error] bev_initializer 失败 (exit={e.returncode})，跳过帧 {fid}")
            continue
        elapsed = time.perf_counter() - t0

        after_path = os.path.join(frame_dir, "pose_after_bev.txt")
        bev_dbg = os.path.join(frame_dir, "debug_bev_score.json")
        breakdown: dict = {}
        if os.path.isfile(bev_dbg):
            try:
                with open(bev_dbg, "r", encoding="utf-8") as f:
                    breakdown = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        if os.path.isfile(after_path):
            parsed = _parse_pose_after_bev(after_path)
            if parsed:
                last_pose = parsed
                context.bev_pose_by_frame[frame_id] = parsed
            write_unified_debug_json(
                os.path.join(frame_dir, "debug_score_breakdown.json"),
                stage="bev",
                frame_id=fid,
                input_pose={"rvec": list(rvec), "tvec": list(tvec)},
                output_pose=parsed,
                breakdown=breakdown if breakdown else None,
                elapsed_sec=elapsed,
                meta={"bev_initializer": os.path.basename(exe)},
            )

    if last_pose:
        context.current_pose_bev = last_pose
        print(f"[Info] current_pose_bev 已更新（末帧）: rvec={last_pose['rvec']}")
    else:
        print("[Warning] 未得到任何 pose_after_bev.txt")

    print(f"\n[完成] BEV 粗初始化输出目录: {out_root}")
