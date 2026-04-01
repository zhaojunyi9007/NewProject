#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小 line constraints A/B 实验脚本

目的（中文）：
- 在不改默认主流程的前提下，对同一帧做 line off / line on 两次标定，
  自动汇总 final cost、active line blocks、最终外参，并生成叠加可视化图。
- 该脚本是“额外能力”，不会影响 run_pipeline.py 默认行为。
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
import yaml


def load_cfg(path: str):
    from pipeline.context import load_runtime_config
    return load_runtime_config(path)


def load_velo_to_cam_extrinsic(calib_path: str):
    """读取 calib_velo_to_cam.txt 的 R/T（与 run_pipeline 保持一致）"""
    if not calib_path or not os.path.exists(calib_path):
        return None
    r_vals = None
    t_vals = None
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("R:"):
                r_vals = [float(x) for x in line.replace("R:", "").split()]
            elif line.startswith("T:"):
                t_vals = [float(x) for x in line.replace("T:", "").split()]
    if not r_vals or not t_vals or len(r_vals) != 9 or len(t_vals) != 3:
        return None

    # 仅做最小依赖：延用 OpenCV Rodrigues 的外部脚本结果会更稳。
    # 这里避免引入 cv2 依赖，直接回退 None，交给配置初值。
    return None


def parse_metrics(stdout: str):
    """从 optimizer 输出中提取 final cost 与 active line blocks"""
    final_cost = None
    active_line_blocks = 0

    m_cost = re.search(r"Final cost:\s*([0-9eE+\-.]+)", stdout)
    if m_cost:
        final_cost = float(m_cost.group(1))

    m_line = re.search(r"Line constraints added:\s*\d+,\s*active at init:\s*(\d+)", stdout)
    if m_line:
        active_line_blocks = int(m_line.group(1))

    return final_cost, active_line_blocks


def read_result_rt(result_file: str):
    """读取 optimizer 输出的最终外参（平滑后保存值）"""
    with open(result_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.startswith("#")]
    if len(lines) < 2:
        raise RuntimeError(f"Invalid calib result file: {result_file}")
    r = [float(x) for x in lines[0].split()[:3]]
    t = [float(x) for x in lines[1].split()[:3]]
    return r, t


def run_one(cfg, frame_id: int, mode: str, out_dir: str, env_base: dict, init_r, init_t, line_debug: bool):
    assert mode in ("off", "on")

    data = cfg["data"]
    sam_dir = data["sam_output_dir"]
    lidar_dir = data["lidar_output_dir"]
    calib_file = data.get("calib_file", "")

    feature_base = os.path.join(lidar_dir, f"{frame_id:010d}")
    sam_base = os.path.join(sam_dir, f"{frame_id:010d}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    result_file = os.path.join(out_dir, f"{frame_id:010d}_line_{mode}_calib_result.txt")

    env = env_base.copy()
    env["EDGECALIB_USE_LINE_CONSTRAINT"] = "1" if mode == "on" else "0"
    if line_debug:
        env["EDGECALIB_LOG_LINE_DEBUG"] = "1"

    cmd = [
        "./build/optimizer",
        feature_base,
        sam_base,
        calib_file if os.path.exists(calib_file) else "",
        str(init_r[0]), str(init_r[1]), str(init_r[2]),
        str(init_t[0]), str(init_t[1]), str(init_t[2]),
        result_file,
    ]

    print(f"\n[LineAB] Running mode={mode} frame={frame_id:010d}")
    proc = subprocess.run(cmd, check=True, env=env, text=True, capture_output=True)
    print(proc.stdout)

    final_cost, active_blocks = parse_metrics(proc.stdout)
    r, t = read_result_rt(result_file)

    # 生成叠加图
    img_path = os.path.join(data["image_dir"], f"{frame_id:010d}.png")
    vis_out = os.path.join(out_dir, f"{frame_id:010d}_line_{mode}_overlay.png")
    vis_cmd = [
        sys.executable, "tools/visualize.py",
        "--img", img_path,
        "--feature_base", feature_base,
        "--calib_file", calib_file if os.path.exists(calib_file) else "",
        "--r_vec", str(r[0]), str(r[1]), str(r[2]),
        "--t_vec", str(t[0]), str(t[1]), str(t[2]),
        "--output", vis_out,
    ]
    subprocess.run(vis_cmd, check=True, env=env)

    return {
        "mode": mode,
        "final_cost": final_cost,
        "active_line_blocks": active_blocks,
        "r": r,
        "t": t,
        "result_file": result_file,
        "overlay": vis_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Line constraints ON/OFF 最小A/B实验")
    parser.add_argument("--config", default="configs/kitti.yaml")
    parser.add_argument("--frame_id", type=int, default=None, help="默认取 config.frames.frame_ids 的第一帧")
    parser.add_argument("--output_dir", default="result/ablation_line")
    parser.add_argument(
        "--init_calib_result",
        default="",
        help="可选：指定初始外参文件（两行：r 和 t）。不填则自动尝试 result/calibration/<frame>_calib_result.txt。"
    )
    parser.add_argument("--line_debug", action="store_true", help="开启每条3D线的初始化诊断日志（behind/unmatched/threshold等）")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    frame_id = args.frame_id
    if frame_id is None:
        frame_ids = cfg.get("frames", {}).get("frame_ids", [])
        if not frame_ids:
            raise RuntimeError("No frame_id provided and config.frames.frame_ids is empty")
        frame_id = int(frame_ids[0])

    # 为避免 line A/B 实验被“坏初值”污染，优先使用已跑通的标定结果作为 A/B 共同初值。
    # 这样才能真正比较 line off/on 的影响，而不是比较“初值太差导致都退化”。
    init_r = cfg["calibration"]["initial_extrinsic"]["rotation"]
    init_t = cfg["calibration"]["initial_extrinsic"]["translation"]
    init_result_file = args.init_calib_result.strip()
    if not init_result_file:
        init_result_file = os.path.join(cfg["data"]["calib_output_dir"], f"{frame_id:010d}_calib_result.txt")
    if init_result_file and os.path.exists(init_result_file):
        try:
            init_r, init_t = read_result_rt(init_result_file)
            print(f"[LineAB] 使用初始外参文件: {init_result_file}")
            print(f"[LineAB] init R: {init_r}")
            print(f"[LineAB] init T: {init_t}")
        except Exception as e:
            print(f"[LineAB][Warning] 读取初始外参文件失败，回退 config 初值: {e}")

    env_base = os.environ.copy()

    off_res = run_one(cfg, frame_id, "off", args.output_dir, env_base, init_r, init_t, args.line_debug)
    on_res = run_one(cfg, frame_id, "on", args.output_dir, env_base, init_r, init_t, args.line_debug)

    print("\n================ Line Constraints A/B Summary ================")
    for res in (off_res, on_res):
        print(f"mode={res['mode']}")
        print(f"  final_cost: {res['final_cost']}")
        print(f"  active_line_blocks: {res['active_line_blocks']}")
        print(f"  final_R: {res['r']}")
        print(f"  final_T: {res['t']}")
        print(f"  overlay: {res['overlay']}")


if __name__ == "__main__":
    main()