import cv2
import numpy as np
import os
import argparse
import sys

def load_kitti_calib(calib_file):
    """
    加载KITTI标定文件
    如果文件不存在或解析失败，返回默认值
    支持P2:和P_rect_02:格式
    """
    if not calib_file or not os.path.exists(calib_file):
        print("[Warning] Using default camera intrinsics (KITTI typical values)")
        K = np.array([[721.5, 0, 609.5],
                      [0, 721.5, 172.8],
                      [0, 0, 1]])
        return K
    
    try:
        with open(calib_file, 'r') as f:
            for line in f:
                # 支持P2:和P_rect_02:格式
                if line.startswith('P2:') or line.startswith('P_rect_02:'):
                    # 提取key和values
                    parts = line.split(':')
                    if len(parts) >= 2:
                        values_str = parts[1].strip()
                        values = list(map(float, values_str.split()))
                        if len(values) == 12:
                            # P2 是 3x4 投影矩阵，提取左上3x3作为内参
                            K = np.array([[values[0], values[1], values[2]],
                                          [values[4], values[5], values[6]],
                                          [values[8], values[9], values[10]]])
                            print(f"[Info] Loaded camera intrinsics from {calib_file}")
                            return K
    except Exception as e:
        print(f"[Error] Failed to parse calibration file: {e}")
    
    # 返回默认值
    print("[Warning] Using default camera intrinsics")
    K = np.array([[721.5, 0, 609.5],
                  [0, 721.5, 172.8],
                  [0, 0, 1]])
    return K

def load_features(feature_base):
    """
    加载点特征和线特征
    """
    points = []
    lines_3d = []
    
    # 加载点特征
    points_file = feature_base + "_points.txt"
    if os.path.exists(points_file):
        try:
            with open(points_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        vals = list(map(float, line.strip().split()))
                        if len(vals) >= 7:  # x y z i nx ny nz [label] [weight]
                            # 只取前7个值(x y z i nx ny nz)用于可视化
                            points.append(vals[:7])
            print(f"[Info] Loaded {len(points)} point features")
        except Exception as e:
            print(f"[Warning] Failed to load points: {e}")
    else:
        print(f"[Warning] Point features file not found: {points_file}")
    
    # 加载3D线特征
    lines_file = feature_base + "_lines_3d.txt"
    if os.path.exists(lines_file):
        try:
            with open(lines_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        vals = list(map(float, line.strip().split()))
                        if len(vals) >= 7:  # x1 y1 z1 x2 y2 z2 type
                            lines_3d.append(vals[:7])
            print(f"[Info] Loaded {len(lines_3d)} 3D line features")
        except Exception as e:
            print(f"[Warning] Failed to load 3D lines: {e}")
    else:
        print(f"[Warning] 3D line features file not found: {lines_file}")
    
    return points, lines_3d

def project_3d_lines(img, lines_3d, K, R, t):
    """
    将3D线投影到图像上
    
    投影公式（与C++代码一致）:
    1. 坐标变换: p_cam = R @ p_lidar + t  (LiDAR坐标系 -> 相机坐标系)
    2. 投影: uv = K @ p_cam
    3. 像素坐标: u = uv[0]/uv[2], v = uv[1]/uv[2]
    """
    if not lines_3d:
        return img
    
    h, w = img.shape[:2]
    
    for line in lines_3d:
        p1 = np.array([line[0], line[1], line[2]])
        p2 = np.array([line[3], line[4], line[5]])
        l_type = int(line[6])
        
        # Transform: LiDAR坐标系 -> 相机坐标系
        p1_c = R @ p1 + t
        p2_c = R @ p2 + t
        
        # 检查点是否在相机前方（z > 0.1）
        if p1_c[2] < 0.1 or p2_c[2] < 0.1: 
            continue
        
        # Project: 3D点 -> 2D像素坐标
        uv1 = K @ p1_c
        u1, v1 = int(uv1[0]/uv1[2]), int(uv1[1]/uv1[2])
        
        uv2 = K @ p2_c
        u2, v2 = int(uv2[0]/uv2[2]), int(uv2[1]/uv2[2])
        
        # 检查投影点是否在图像范围内
        if (0 <= u1 < w and 0 <= v1 < h) or (0 <= u2 < w and 0 <= v2 < h):
            # Color: 0(Rail)=Green, 1(Pole)=Red
            color = (0, 255, 0) if l_type == 0 else (0, 0, 255)
            thickness = 2
            
            # Draw
            cv2.line(img, (u1, v1), (u2, v2), color, thickness)
    
    return img

def project_points(img, points, K, R, t, subsample=5):
    """
    将3D点投影到图像上
    
    投影公式（与C++代码一致）:
    1. 坐标变换: p_cam = R @ p_lidar + t  (LiDAR坐标系 -> 相机坐标系)
    2. 投影: uv = K @ p_cam
    3. 像素坐标: u = uv[0]/uv[2], v = uv[1]/uv[2]
    
    Args:
        subsample: 每隔几个点绘制一次，避免图像过于密集
    """
    if not points:
        return img
    
    h, w = img.shape[:2]
    projected_count = 0
    
    for i, pt in enumerate(points):
        # 子采样
        if i % subsample != 0:
            continue
        
        p = np.array([pt[0], pt[1], pt[2]])
        # Transform: LiDAR坐标系 -> 相机坐标系
        p_c = R @ p + t
        
        # 检查点是否在相机前方（z > 0.1）
        if p_c[2] < 0.1: 
            continue
        
        # Project: 3D点 -> 2D像素坐标
        uv = K @ p_c
        u, v = int(uv[0]/uv[2]), int(uv[1]/uv[2])
        
        if 0 <= u < w and 0 <= v < h:
            # Yellow points
            cv2.circle(img, (u, v), 2, (0, 255, 255), -1)
            projected_count += 1
    
    print(f"[Info] Projected {projected_count} points to image")
    return img

def add_legend(img):
    """
    在图像上添加图例
    """
    # 创建图例区域
    legend_h = 80
    legend = np.zeros((legend_h, img.shape[1], 3), dtype=np.uint8)
    
    # 添加文字和图例
    cv2.putText(legend, "Legend:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.circle(legend, (20, 45), 3, (0, 255, 255), -1)
    cv2.putText(legend, "LiDAR Points", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.line(legend, (200, 45), (250, 45), (0, 255, 0), 2)
    cv2.putText(legend, "Rail Lines", (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.line(legend, (400, 45), (450, 45), (0, 0, 255), 2)
    cv2.putText(legend, "Pole Lines", (460, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 拼接到原图
    result = np.vstack([img, legend])
    return result

def load_calib_result(calib_result_file):
    """
    从标定结果文件中读取外参
    返回: (r_vec, t_vec) 或 None
    
    支持的格式:
    1. EdgeCalib v2.0 格式:
       # EdgeCalib v2.0 Calibration Result
       <rx> <ry> <rz>
       <tx> <ty> <tz>
    
    2. 旧格式 (向后兼容):
       Rotation (rx, ry, rz): <rx> <ry> <rz>
       Translation (tx, ty, tz): <tx> <ty> <tz>
    """
    if not os.path.exists(calib_result_file):
        return None
    
    try:
        with open(calib_result_file, 'r') as f:
            lines = f.readlines()
            r_vec = None
            t_vec = None
            
            # 尝试解析 EdgeCalib v2.0 格式（简单格式）
            data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
            if len(data_lines) >= 2:
                # 第一行是旋转，第二行是平移
                r_vals = list(map(float, data_lines[0].split()))
                t_vals = list(map(float, data_lines[1].split()))
                if len(r_vals) == 3 and len(t_vals) == 3:
                    r_vec = np.array(r_vals)
                    t_vec = np.array(t_vals)
                    print(f"[Info] Loaded calibration result from {calib_result_file} (EdgeCalib v2.0 format)")
                    return r_vec, t_vec
            
            # 尝试解析旧格式
            for idx, line in enumerate(lines):
                line = line.strip()
                if line.startswith('Rotation (rx, ry, rz):'):
                    parts = line.split(':')
                    if len(parts) == 2:
                        vals = list(map(float, parts[1].strip().split()))
                        if len(vals) == 3:
                            r_vec = np.array(vals)
                
                elif line.startswith('Translation (tx, ty, tz):'):
                    parts = line.split(':')
                    if len(parts) == 2:
                        vals = list(map(float, parts[1].strip().split()))
                        if len(vals) == 3:
                            t_vec = np.array(vals)
            
            if r_vec is not None and t_vec is not None:
                print(f"[Info] Loaded calibration result from {calib_result_file} (legacy format)")
                return r_vec, t_vec
    except Exception as e:
        print(f"[Warning] Failed to parse calibration result: {e}")
    
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Visualize HSR Lidar-Camera Calibration Result",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用标定结果文件自动加载外参
  python3 visual_result.py --img image.png --feature_base result/0000000000
  
  # 手动指定外参
  python3 visual_result.py --img image.png --feature_base result/0000000000 \\
      --r_vec 0.01 0.02 0.03 --t_vec 0.0 -0.3 1.8
  
  # 指定相机标定文件
  python3 visual_result.py --img image.png --feature_base result/0000000000 \\
      --calib_file /path/to/calib.txt
        """)
    
    parser.add_argument("--img", type=str, required=True, 
                        help="Original Image Path (e.g., /gz-data/dataset/.../image_02/data/0000000000.png)")
    parser.add_argument("--feature_base", type=str, required=True, 
                        help="Path to features (e.g., result/0000000000)")
    parser.add_argument("--calib_file", type=str, default="", 
                        help="KITTI calibration file (optional, e.g., calib_cam_to_cam.txt)")
    parser.add_argument("--r_vec", type=float, nargs=3, default=None, 
                        help="Rotation Vector (rx ry rz) in radians. If not provided, will try to load from <feature_base>_calib_result.txt")
    parser.add_argument("--t_vec", type=float, nargs=3, default=None, 
                        help="Translation Vector (tx ty tz) in meters. If not provided, will try to load from <feature_base>_calib_result.txt")
    parser.add_argument("--output", type=str, default="", 
                        help="Output image path (default: result/visual_<frame_id>.png)")
    parser.add_argument("--subsample", type=int, default=5, 
                        help="Point subsampling factor (draw every N-th point)")
    
    args = parser.parse_args()

    print("=== HSR Lidar-Camera Calibration Visualization ===")
    print("")
    
    # 读取图像
    if not os.path.exists(args.img):
        print(f"[Error] Image file not found: {args.img}")
        return 1
    
    img = cv2.imread(args.img)
    if img is None:
        print(f"[Error] Cannot read image: {args.img}")
        return 1

    print(f"[Info] Image loaded: {img.shape[1]}x{img.shape[0]}")

    # 1. 加载特征
    points, lines_3d = load_features(args.feature_base)
    
    if not points and not lines_3d:
        print("[Error] No features loaded. Nothing to visualize.")
        print(f"[Info] Expected files: {args.feature_base}_points.txt, {args.feature_base}_lines_3d.txt")
        return 1

    # 2. 加载相机内参
    K = load_kitti_calib(args.calib_file)
    print(f"[Info] Camera intrinsics K:\n{K}")

    # 3. 准备外参
    # 尝试从标定结果文件读取
    r_vec = None
    t_vec = None
    
    if args.r_vec is None or args.t_vec is None:
        # 尝试多个可能的标定结果文件路径
        # 1. 直接在feature_base同目录
        calib_result_file = f"{args.feature_base}_calib_result.txt"
        result = load_calib_result(calib_result_file)
        
        # 2. 如果在lidar_features目录，尝试从calibration目录读取
        if result is None:
            feature_dir = os.path.dirname(args.feature_base)
            if feature_dir and 'lidar_features' in feature_dir:
                # 替换lidar_features为calibration
                calib_dir = feature_dir.replace('lidar_features', 'calibration')
                frame_id = os.path.basename(args.feature_base)
                calib_result_file = os.path.join(calib_dir, f"{frame_id}_calib_result.txt")
                result = load_calib_result(calib_result_file)
        
        # 3. 如果在result目录下，尝试从calibration子目录读取
        if result is None:
            feature_dir = os.path.dirname(args.feature_base)
            if feature_dir:
                # 尝试在result目录下查找calibration子目录
                result_dir = os.path.dirname(feature_dir) if os.path.basename(feature_dir) in ['lidar_features', 'sam_features'] else feature_dir
                calib_dir = os.path.join(result_dir, 'calibration')
                frame_id = os.path.basename(args.feature_base)
                calib_result_file = os.path.join(calib_dir, f"{frame_id}_calib_result.txt")
                result = load_calib_result(calib_result_file)
        
        if result is not None:
            r_vec, t_vec = result
        else:
            print(f"[Warning] Calibration result file not found or invalid")
            print(f"[Warning] Tried: {args.feature_base}_calib_result.txt")
            print("[Warning] Using default values (identity transform)")
            r_vec = np.array([0.0, 0.0, 0.0])
            t_vec = np.array([0.0, 0.0, 0.0])
    else:
        r_vec = np.array(args.r_vec)
        t_vec = np.array(args.t_vec)
    
    R, _ = cv2.Rodrigues(r_vec)
    
    print(f"[Info] Rotation (angle-axis): {r_vec}")
    print(f"[Info] Translation (meters): {t_vec}")

    # 4. 绘制投影
    print("\n[Projecting Features]")
    
    # 先画点 (作为背景)
    if points:
        img = project_points(img, points, K, R, t_vec, subsample=args.subsample)
    
    # 再画线 (更显眼)
    if lines_3d:
        img = project_3d_lines(img, lines_3d, K, R, t_vec)

    # 5. 添加图例
    img = add_legend(img)

    # 6. 保存结果
    # 如果没有指定输出路径，自动生成到result/visualization目录
    if not args.output:
        # 提取frame_id
        frame_id = os.path.basename(args.feature_base)
        feature_dir = os.path.dirname(args.feature_base)
        
        # 尝试找到result目录
        if feature_dir:
            # 如果在lidar_features或sam_features目录下，向上找到result目录
            if os.path.basename(feature_dir) in ['lidar_features', 'sam_features']:
                result_dir = os.path.dirname(feature_dir)
                output_dir = os.path.join(result_dir, 'visualization')
            else:
                # 否则使用feature_dir的父目录或当前目录
                output_dir = os.path.join(feature_dir, 'visualization') if feature_dir else 'result/visualization'
        else:
            output_dir = 'result/visualization'
        
        args.output = os.path.join(output_dir, f"{frame_id}_result.png")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cv2.imwrite(args.output, img)
    print(f"\n[Success] Result saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
