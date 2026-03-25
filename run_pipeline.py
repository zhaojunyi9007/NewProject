#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EdgeCalib v2.0 主控脚本
按照README2.0方案执行完整的标定流程
"""

import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path

import cv2
import numpy as np

from pipeline.context import load_runtime_config, parse_frame_ids, validate_config
from pipeline.optimizer_env_adapter import build_optimizer_env

class EdgeCalibPipeline:
    def __init__(self, config_path="config.yaml", config=None):
        """初始化标定流程"""
        if config is not None:
            self.config = config
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        # 创建输出目录
        self._create_output_dirs()
        
        # 获取要处理的帧列表
        self.frame_ids = self._get_frame_list()
        
        print(f"=== EdgeCalib v2.0 Pipeline ===")
        print(f"处理帧数: {len(self.frame_ids)}")
        print(f"帧ID列表: {self.frame_ids}")
        print(f"输出目录: {self.config['data']['result_dir']}")
        print("=" * 40)

    def _load_velo_to_cam_extrinsic(self):
        """从calib_velo_to_cam.txt加载外参，并转换为angle-axis"""
        calib_path = self.config["data"].get("velo_to_cam_file", "")
        if not calib_path:
            return None
        if not os.path.exists(calib_path):
            print(f"[Warning] LiDAR外参文件不存在: {calib_path}")
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
            print(f"[Warning] LiDAR外参文件格式异常: {calib_path}")
            return None

        r_mat = np.array(r_vals, dtype=np.float64).reshape(3, 3)
        t_vec = np.array(t_vals, dtype=np.float64)
        r_vec, _ = cv2.Rodrigues(r_mat)
        return r_vec.reshape(-1).tolist(), t_vec.reshape(-1).tolist()
    
    def _create_output_dirs(self):
        """创建输出目录结构"""
        dirs = [
            self.config['data']['result_dir'],
            self.config['data']['sam_output_dir'],
            self.config['data']['lidar_output_dir'],
            self.config['data']['calib_output_dir'],
            self.config['data']['visual_output_dir']
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        print(f"[Info] 输出目录已创建")
    
    def _get_frame_list(self):
        """获取要处理的帧列表"""
        mode = self.config['frames']['mode']
        
        if mode == "select":
            return self.config['frames']['frame_ids']
        elif mode == "all":
            # 扫描velodyne目录获取所有.bin文件
            velodyne_dir = self.config['data']['velodyne_dir']
            bin_files = sorted(Path(velodyne_dir).glob("*.bin"))
            return [int(f.stem) for f in bin_files]
        else:
            raise ValueError(f"未知的处理模式: {mode}")
    
    def run_sam_extraction(self):
        """阶段1: 运行SAM特征提取"""
        print("\n" + "=" * 40)
        print("[阶段1] SAM图像特征提取")
        print("=" * 40)
        
        image_dir = self.config['data']['image_dir']
        output_dir = self.config['data']['sam_output_dir']
        checkpoint = self.config['sam']['checkpoint_path']
        
        for frame_id in self.frame_ids:
            img_path = os.path.join(image_dir, f"{frame_id:010d}.png")
            if not os.path.exists(img_path):
                print(f"[Warning] 图像不存在: {img_path}")
                continue
            
            print(f"\n处理帧 {frame_id:010d}...")
            cmd = [
                "python", "python/run_sam.py",
                "--image", img_path,
                "--checkpoint", checkpoint,
                "--output_dir", output_dir
            ]
            subprocess.run(cmd, check=True)
        
        print(f"\n[完成] SAM特征已保存到: {output_dir}")
    
    def run_lidar_extraction(self):
        """阶段2: 运行LiDAR特征提取（含NDT融合）"""
        print("\n" + "=" * 40)
        print("[阶段2] LiDAR特征提取 (含NDT多帧融合)")
        print("=" * 40)
        
        velodyne_dir = self.config['data']['velodyne_dir']
        output_dir = self.config['data']['lidar_output_dir']
        fusion_window = self.config['frames']['fusion_window']
        
        for i, frame_id in enumerate(self.frame_ids):
            # 确定融合窗口: 当前帧 + 前N-1帧
            fusion_frames = []
            for j in range(max(0, i - fusion_window + 1), i + 1):
                if j < len(self.frame_ids):
                    fusion_frames.append(self.frame_ids[j])
            
            # 构建点云文件路径
            bin_paths = [os.path.join(velodyne_dir, f"{fid:010d}.bin") for fid in fusion_frames]
            
            # 检查文件存在性
            if not all(os.path.exists(p) for p in bin_paths):
                print(f"[Warning] 部分点云文件不存在，跳过帧 {frame_id:010d}")
                continue
            
            output_base = os.path.join(output_dir, f"{frame_id:010d}")
            
            print(f"\n处理帧 {frame_id:010d}，融合 {len(bin_paths)} 帧...")
            print(f"  融合帧: {[f'{fid:010d}' for fid in fusion_frames]}")
            
            # 调用C++程序
            cmd = ["./build/lidar_extractor", *bin_paths, output_base]
            subprocess.run(cmd, check=True)
        
        print(f"\n[完成] LiDAR特征已保存到: {output_dir}")
    
    def run_calibration(self):
        """阶段3: 运行标定优化"""
        print("\n" + "=" * 40)
        print("[阶段3] 两阶段标定优化")
        print("=" * 40)
        
        sam_dir = self.config['data']['sam_output_dir']
        lidar_dir = self.config['data']['lidar_output_dir']
        calib_dir = self.config['data']['calib_output_dir']
        calib_file = self.config['data']['calib_file']
        
        init_r = self.config['calibration']['initial_extrinsic']['rotation']
        init_t = self.config['calibration']['initial_extrinsic']['translation']
        velo_to_cam = self._load_velo_to_cam_extrinsic()
        if velo_to_cam:
            init_r, init_t = velo_to_cam
            print("[Info] 使用calib_velo_to_cam.txt中的R/T作为初始外参")

        # A/B实验参数（通过统一 adapter 传递给 C++ optimizer）
        optimizer_env, has_ab_overrides = build_optimizer_env(
            self.config.get("calibration", {}),
            os.environ.copy()
        )
        if has_ab_overrides:
            print("[Info] 已加载 calibration.ab_experiment 参数并传递给 optimizer")
        
        for frame_id in self.frame_ids:
            feature_base = os.path.join(lidar_dir, f"{frame_id:010d}")
            sam_base = os.path.join(sam_dir, f"{frame_id:010d}")
            output_file = os.path.join(calib_dir, f"{frame_id:010d}_calib_result.txt")
            
            # 检查特征文件是否存在
            if not os.path.exists(f"{feature_base}_points.txt"):
                print(f"[Warning] 特征文件不存在，跳过帧 {frame_id:010d}")
                continue
            
            print(f"\n优化帧 {frame_id:010d}...")
            
            # 调用C++优化器
            cmd = [
                "./build/optimizer",
                feature_base,
                sam_base,
                calib_file if os.path.exists(calib_file) else "",
                str(init_r[0]), str(init_r[1]), str(init_r[2]),
                str(init_t[0]), str(init_t[1]), str(init_t[2]),
                output_file
            ]
            subprocess.run(cmd, check=True, env=optimizer_env)
        
        print(f"\n[完成] 标定结果已保存到: {calib_dir}")
    
    def run_visualization(self):
        """阶段4: 可视化结果"""
        print("\n" + "=" * 40)
        print("[阶段4] 结果可视化")
        print("=" * 40)
        
        image_dir = self.config['data']['image_dir']
        lidar_dir = self.config['data']['lidar_output_dir']
        calib_dir = self.config['data']['calib_output_dir']
        visual_dir = self.config['data']['visual_output_dir']
        calib_file = self.config['data'].get('calib_file', '')
        
        for frame_id in self.frame_ids:
            img_path = os.path.join(image_dir, f"{frame_id:010d}.png")
            feature_base = os.path.join(lidar_dir, f"{frame_id:010d}")
            calib_result_file = os.path.join(calib_dir, f"{frame_id:010d}_calib_result.txt")
            output_path = os.path.join(visual_dir, f"{frame_id:010d}_result.png")
            
            if not os.path.exists(calib_result_file):
                print(f"[Warning] 标定结果不存在，跳过帧 {frame_id:010d}")
                continue
            
            # 读取标定结果
            with open(calib_result_file, 'r') as f:
                lines = f.readlines()
                r_vec = lines[1].strip().split()
                t_vec = lines[2].strip().split()
            
            print(f"可视化帧 {frame_id:010d}...")
            cmd = [
                "python", "visual_result.py",
                "--img", img_path,
                "--feature_base", feature_base,
                "--calib_file", calib_file if os.path.exists(calib_file) else "",
                "--r_vec", *r_vec,
                "--t_vec", *t_vec,
                "--output", output_path
            ]
            subprocess.run(cmd, check=True)
        
        print(f"\n[完成] 可视化结果已保存到: {visual_dir}")
    
    def run_all(self, skip_sam=False, skip_lidar=False, skip_calib=False, skip_visual=False):
        """运行完整流程"""
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
            print(f"结果保存在: {self.config['data']['result_dir']}")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 执行失败: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="EdgeCalib v2.0 完整标定流程")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--stage", choices=["sam", "lidar", "calib", "visual", "all"], 
                        default="all", help="执行阶段")
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
    validate_config(config)

    pipeline = EdgeCalibPipeline(args.config, config=config)
    
    if args.stage == "sam":
        pipeline.run_sam_extraction()
    elif args.stage == "lidar":
        pipeline.run_lidar_extraction()
    elif args.stage == "calib":
        pipeline.run_calibration()
    elif args.stage == "visual":
        pipeline.run_visualization()
    else:  # all
        pipeline.run_all(
            skip_sam=args.skip_sam,
            skip_lidar=args.skip_lidar,
            skip_calib=args.skip_calib,
            skip_visual=args.skip_visual
        )


if __name__ == "__main__":
    main()
