#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys

from pipeline.context import RuntimeContext, create_output_dirs, get_frame_list
from pipeline.stages import calib_stage, lidar_stage, sam_stage, visual_stage


class PipelineRunner:
    def __init__(self, config):
        self.context = RuntimeContext(config=config, frame_ids=[])
        create_output_dirs(self.context.config)
        self.context.frame_ids = get_frame_list(self.context.config)

        print("=== EdgeCalib v2.0 Pipeline ===")
        print(f"处理帧数: {len(self.context.frame_ids)}")
        print(f"帧ID列表: {self.context.frame_ids}")
        print(f"输出目录: {self.context.config['data']['result_dir']}")
        print("=" * 40)

    def run_sam_extraction(self):
        sam_stage.run(self.context)

    def run_lidar_extraction(self):
        lidar_stage.run(self.context)

    def run_calibration(self):
        calib_stage.run(self.context)

    def run_visualization(self):
        visual_stage.run(self.context)

    def run_all(self, skip_sam=False, skip_lidar=False, skip_calib=False, skip_visual=False):
        try:
            if not skip_sam:
                self.run_sam_extraction()
            if not skip_lidar:
                self.run_lidar_extraction()
            if not skip_calib:
                self.run_calibration()
            if not skip_visual:
                self.run_visualization()

            print("\n" + "=" * 40)
            print("✅ 全部流程执行完成!")
            print("=" * 40)
            print(f"结果保存在: {self.context.config['data']['result_dir']}")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ 执行失败: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            sys.exit(1)