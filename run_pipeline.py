#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EdgeCalib v2.0 主控脚本
Stage-3: 仅保留 CLI + Runner 调用
"""
import argparse

from pipeline.context import (
    apply_cli_semantic_overrides,
    load_runtime_config,
    parse_frame_ids,
    validate_config,
)
from pipeline.runner import PipelineRunner



def main():
    parser = argparse.ArgumentParser(description="EdgeCalib v2.0 完整标定流程")
    parser.add_argument("--config", default="configs/osdar23.yaml", help="配置文件路径")
    parser.add_argument("--stage", choices=["sam", "lidar", "calib", "visual", "all"], default="all", help="执行阶段")
    parser.add_argument("--skip-sam", action="store_true", help="跳过SAM提取")
    parser.add_argument("--skip-lidar", action="store_true", help="跳过LiDAR提取")
    parser.add_argument("--skip-calib", action="store_true", help="跳过标定优化")
    parser.add_argument("--skip-visual", action="store_true", help="跳过可视化")
    parser.add_argument("--frame-ids", default=None, help="CLI覆盖帧ID列表，逗号分隔，如 0,10,20")
    parser.add_argument("--fusion-window", type=int, default=None, help="CLI覆盖多帧融合窗口")
    parser.add_argument("--result-dir", default=None, help="CLI覆盖 result_dir（仅入口级覆盖）")
    
    args = parser.parse_args()
    
    cli_overrides = [
        (("frames", "frame_ids"), parse_frame_ids(args.frame_ids)),
        (("frames", "fusion_window"), args.fusion_window),
        (("data", "result_dir"), args.result_dir),
    ]
    config = load_runtime_config(args.config, cli_overrides=cli_overrides)
    apply_cli_semantic_overrides(config, result_dir=args.result_dir, frame_ids_text=args.frame_ids)
    validate_config(config)
    runner = PipelineRunner(config)
    
    if args.stage == "sam":
        runner.run_sam_extraction()
    elif args.stage == "lidar":
        runner.run_lidar_extraction()
    elif args.stage == "calib":
        runner.run_calibration()
    elif args.stage == "visual":
        runner.run_visualization()
    else: 
        runner.run_all(
            skip_sam=args.skip_sam,
            skip_lidar=args.skip_lidar,
            skip_calib=args.skip_calib,
            skip_visual=args.skip_visual,
        )


if __name__ == "__main__":
    main()
