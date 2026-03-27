import cv2
import numpy as np
import os
import argparse
import sys

def load_kitti_calib(calib_file):
    """
    加载KITTI标定文件
    如果文件不存在或解析失败，返回默认值
    支持P2/P_rect_02和R0_rect/R_rect_00格式
    """
    default_k = np.array([[721.5, 0, 609.5],
                          [0, 721.5, 172.8],
                          [0, 0, 1]])
    default_r_rect = np.eye(3)
    default_p_rect = np.array([[721.5, 0, 609.5, 0.0],
                               [0, 721.5, 172.8, 0.0],
                               [0, 0, 1, 0.0]])

    if not calib_file or not os.path.exists(calib_file):
        print("[Warning] No calib_file provided; using default camera intrinsics (KITTI typical values)")
        
        return default_k, default_r_rect, default_p_rect
    
    try:
        K = None
        R_rect = None
        P_rect = None

        with open(calib_file, 'r') as f:
            for line in f:
                line = line.strip()
                # 支持P2:和P_rect_02:格式
                if line.startswith('P2:') or line.startswith('P_rect_02:'):                    
                    parts = line.split(':')
                    if len(parts) >= 2:
                        values_str = parts[1].strip()
                        values = list(map(float, values_str.split()))
                        if len(values) == 12:
                            P_rect = np.array(values, dtype=np.float64).reshape(3, 4)
                            K = P_rect[:, :3].copy()
                elif line.startswith('R0_rect:') or line.startswith('R_rect_00:'):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        values_str = parts[1].strip()
                        values = list(map(float, values_str.split()))
                        if len(values) == 9:
                            R_rect = np.array(values, dtype=np.float64).reshape(3, 3)
        if K is None or P_rect is None:
            raise ValueError("P2/P_rect_02 not found in calibration file.")
        if R_rect is None:
            R_rect = default_r_rect
        print(f"[Info] Loaded camera intrinsics and rectification from {calib_file}")
        return K, R_rect, P_rect
                            
    except Exception as e:
        print(f"[Error] Failed to parse calibration file: {e}")
    

    print("[Warning] Using default camera intrinsics")
    return default_k, default_r_rect, default_p_rect

def load_features(feature_base, point_source="edge"):
    """
    加载点特征和线特征
    """
    points = []
    lines_3d = []
    
    # 加载点特征：
    # 1) 优先使用 _points.txt（Stage 8 标准输出：x y z i nx ny nz label weight）
    # 2) 若不存在则回退到 _edge_points.txt（Stage 9 输出：x y z intensity）
    edge_file = feature_base + "_edge_points.txt"
    all_file = feature_base + "_points.txt"

    if point_source == "all":
        points_candidates = [all_file, edge_file]
    elif point_source == "auto":
        points_candidates = [edge_file, all_file]
    else:  # 默认 edge
        points_candidates = [edge_file, all_file]
    
    points_file_used = None
    for points_file in points_candidates:
        if not os.path.exists(points_file):
            continue
        try:
            with open(points_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        vals = list(map(float, line.strip().split()))
                        # - _points.txt: 至少 7 列（x y z i nx ny nz ...）
                        # - _edge_points.txt: 至少 4 列（x y z intensity）
                        # 可视化投影只需要前三列坐标。
                        if len(vals) >= 3:  
                            points.append(vals[:3])
            points_file_used = points_file
            print(f"[Info] Loaded {len(points)} point features from: {points_file_used}")
            break
        except Exception as e:
            print(f"[Warning] Failed to load points from {points_file}: {e}")

    if points_file_used is None:
        print(f"[Warning] Point features files not found: {points_candidates[0]} or {points_candidates[1]}")
    
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

def project_3d_lines(img, lines_3d, K, R_rect, P_rect, R, t):
    """
    将3D线投影到图像上
    投影公式(使用KITTI官方方法):
    1. 坐标变换: p_cam = R @ p_lidar + t  (LiDAR坐标系 -> 相机坐标系)
    2. 整流: p_rect = R_rect @ p_cam
    3. 投影: uv = P_rect @ [p_rect; 1]
    4. 像素坐标: u = uv[0]/uv[2], v = uv[1]/uv[2]   
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
        p1_rect = R_rect @ p1_c
        p2_rect = R_rect @ p2_c
        
        # 检查点是否在相机前方（z > 0.1）
        if p1_rect[2] < 0.1 and p2_rect[2] < 0.1:
            continue # 如果两个端点都在相机后面，才跳过

        # 如果只有一个点在后面，按比例把它截断在相机前方 (z=0.1) 处
        if p1_rect[2] < 0.1:
            ratio = (0.1 - p2_rect[2]) / (p1_rect[2] - p2_rect[2])
            p1_rect = p2_rect + ratio * (p1_rect - p2_rect)
        elif p2_rect[2] < 0.1:
            ratio = (0.1 - p1_rect[2]) / (p2_rect[2] - p1_rect[2])
            p2_rect = p1_rect + ratio * (p2_rect - p1_rect)
        
        # Project: 3D点 -> 2D像素坐标
        uv1 = P_rect @ np.hstack([p1_rect, 1.0])
        u1, v1 = int(uv1[0]/uv1[2]), int(uv1[1]/uv1[2])
        
        uv2 = P_rect @ np.hstack([p2_rect, 1.0])
        u2, v2 = int(uv2[0]/uv2[2]), int(uv2[1]/uv2[2])
        
        # 检查投影点是否在图像范围内
        if (0 <= u1 < w and 0 <= v1 < h) or (0 <= u2 < w and 0 <= v2 < h):
            # Color: 0(Rail)=Green, 1(Pole)=Red
            color = (0, 255, 0) if l_type == 0 else (0, 0, 255)
            thickness = 2
            
            # Draw
            cv2.line(img, (u1, v1), (u2, v2), color, thickness)
    
    return img

def project_points(img, points, K, R_rect, P_rect, R, t, subsample=5):
    """
    将3D点投影到图像上
    
    投影公式(使用KITTI官方方法):
    1. 坐标变换: p_cam = R @ p_lidar + t  (LiDAR坐标系 -> 相机坐标系)
    2. 整流: p_rect = R_rect @ p_cam
    3. 投影: uv = P_rect @ [p_rect; 1]
    4. 像素坐标: u = uv[0]/uv[2], v = uv[1]/uv[2]

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
        p_rect = R_rect @ p_c
        
        # 检查点是否在相机前方（z > 0.1）
        if p_rect[2] < 0.1: 
            continue
        
        # Project: 3D点 -> 2D像素坐标
        uv = P_rect @ np.hstack([p_rect, 1.0])
        u, v = int(uv[0]/uv[2]), int(uv[1]/uv[2])
        
        if 0 <= u < w and 0 <= v < h:
            # Yellow points
            cv2.circle(img, (u, v), 2, (0, 255, 255), -1)
            projected_count += 1
    
    print(f"[Info] Projected {projected_count} points to image")
    return img

def compute_projection_stats(points, R_rect, P_rect, R, t, img_w, img_h):
    """
    计算投影统计信息（用于外参方向A/B验证）
    返回:
      total: 原始点数
      projected_success: z>0.1 的点数（可投影到成像平面前方）
      in_image: 投影后落在图像内的点数
      behind: 在相机后方的点数
      behind_ratio: behind / total
    """
    total = len(points)
    projected_success = 0
    in_image = 0
    behind = 0

    if total == 0:
        return {
            "total": 0,
            "projected_success": 0,
            "in_image": 0,
            "behind": 0,
            "behind_ratio": 0.0,
        }

    for pt in points:
        p = np.array([pt[0], pt[1], pt[2]], dtype=np.float64)
        p_cam = R @ p + t
        p_rect = R_rect @ p_cam

        if p_rect[2] < 0.1:
            behind += 1
            continue

        projected_success += 1
        uv = P_rect @ np.hstack([p_rect, 1.0])
        u = uv[0] / uv[2]
        v = uv[1] / uv[2]
        if 0 <= u < img_w and 0 <= v < img_h:
            in_image += 1

    return {
        "total": total,
        "projected_success": projected_success,
        "in_image": in_image,
        "behind": behind,
        "behind_ratio": behind / float(total),
    }

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
    parser.add_argument("--skip_rectification", action="store_true",
                    help="Skip applying R_rect (assume extrinsic already in rectified camera frame).")                    
    parser.add_argument("--output", type=str, default="", 
                        help="Output image path (default: result/visual_<frame_id>.png)")
    parser.add_argument("--subsample", type=int, default=5, 
                        help="Point subsampling factor (draw every N-th point)")
    parser.add_argument("--point_source", type=str, default="edge", choices=["edge", "all", "auto"],
                        help="Point source for visualization: edge(optimizer target), all(full points), auto(fallback).")
    parser.add_argument("--ab_extrinsic_compare", action="store_true",
                        help="外参方向A/B验证：比较当前外参与逆外参的投影统计与叠加图。")
    parser.add_argument("--rectify_compare", action="store_true",
                        help="整流约定A/B验证：比较使用R_rect与跳过R_rect的投影统计与叠加图。")
    
    args = parser.parse_args()

    print("=== HSR Lidar-Camera Calibration Visualization ===")
    print("")
    if args.calib_file:
        print(f"[Info] calib_file provided: {args.calib_file}")
    else:
        print("[Warning] calib_file not provided; using default intrinsics unless auto-loaded")
    
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
    points, lines_3d = load_features(args.feature_base, point_source=args.point_source)
    
    if not points and not lines_3d:
        print("[Error] No features loaded. Nothing to visualize.")
        print(f"[Info] Expected files: {args.feature_base}_points.txt (or _edge_points.txt), {args.feature_base}_lines_3d.txt")
        return 1

    # 2. 加载相机内参/整流/投影矩阵
    K, R_rect_loaded, P_rect = load_kitti_calib(args.calib_file)
    R_rect = R_rect_loaded
    if args.skip_rectification:
        print("[Info] Skipping rectification (R_rect = Identity)")
        R_rect = np.eye(3)
    print(f"[Info] Camera intrinsics K:\n{K}")
    print(f"[Info] Rectification R_rect:\n{R_rect}")
    print(f"[Info] Projection P_rect:\n{P_rect}")

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

    # 整流约定A/B验证（最小侵入，不改变默认主流程）
    # A: 使用R_rect（默认流程）
    # B: 跳过R_rect（等价于 --skip_rectification）
    if args.rectify_compare:
        h, w = img.shape[:2]
        R_use = R
        t_use = t_vec
        R_rect_a = R_rect_loaded
        R_rect_b = np.eye(3)

        stats_a = compute_projection_stats(points, R_rect_a, P_rect, R_use, t_use, w, h)
        stats_b = compute_projection_stats(points, R_rect_b, P_rect, R_use, t_use, w, h)

        print("\n[Rectify Compare] Rectification convention check")
        print("  A (use R_rect):")
        print(f"    total={stats_a['total']}, projected_success={stats_a['projected_success']}, "
              f"in_image={stats_a['in_image']}, behind_ratio={stats_a['behind_ratio']:.6f}")
        print("  B (skip R_rect):")
        print(f"    total={stats_b['total']}, projected_success={stats_b['projected_success']}, "
              f"in_image={stats_b['in_image']}, behind_ratio={stats_b['behind_ratio']:.6f}")

        img_a = img.copy()
        if points:
            img_a = project_points(img_a, points, K, R_rect_a, P_rect, R_use, t_use, subsample=args.subsample)
        if lines_3d:
            img_a = project_3d_lines(img_a, lines_3d, K, R_rect_a, P_rect, R_use, t_use)
        img_a = add_legend(img_a)

        img_b = img.copy()
        if points:
            img_b = project_points(img_b, points, K, R_rect_b, P_rect, R_use, t_use, subsample=args.subsample)
        if lines_3d:
            img_b = project_3d_lines(img_b, lines_3d, K, R_rect_b, P_rect, R_use, t_use)
        img_b = add_legend(img_b)

        if not args.output:
            frame_id = os.path.basename(args.feature_base)
            output_dir = 'result/visualization'
            os.makedirs(output_dir, exist_ok=True)
            base_output = os.path.join(output_dir, f"{frame_id}_rectify_compare")
        else:
            out_root, out_ext = os.path.splitext(args.output)
            if not out_ext:
                out_ext = ".png"
            base_output = out_root

        output_a = f"{base_output}_A_use_rrect.png"
        output_b = f"{base_output}_B_skip_rrect.png"
        os.makedirs(os.path.dirname(output_a) if os.path.dirname(output_a) else ".", exist_ok=True)
        os.makedirs(os.path.dirname(output_b) if os.path.dirname(output_b) else ".", exist_ok=True)
        cv2.imwrite(output_a, img_a)
        cv2.imwrite(output_b, img_b)
        print(f"[Rectify Compare] Saved overlay A: {output_a}")
        print(f"[Rectify Compare] Saved overlay B: {output_b}")
        return 0

    # 外参方向A/B验证（最小侵入，不改变默认主流程）
    # A: 当前外参（假定 LiDAR -> Camera）
    # B: 逆外参（用于验证是否存在方向约定错误）
    if args.ab_extrinsic_compare:
        h, w = img.shape[:2]
        R_a = R
        t_a = t_vec
        R_b = R_a.T
        t_b = -R_a.T @ t_a

        stats_a = compute_projection_stats(points, R_rect, P_rect, R_a, t_a, w, h)
        stats_b = compute_projection_stats(points, R_rect, P_rect, R_b, t_b, w, h)

        print("\n[AB Compare] Extrinsic direction check")
        print("  A (current, LiDAR->Camera):")
        print(f"    total={stats_a['total']}, projected_success={stats_a['projected_success']}, "
              f"in_image={stats_a['in_image']}, behind_ratio={stats_a['behind_ratio']:.6f}")
        print("  B (inverse transform):")
        print(f"    total={stats_b['total']}, projected_success={stats_b['projected_success']}, "
              f"in_image={stats_b['in_image']}, behind_ratio={stats_b['behind_ratio']:.6f}")

        # 叠加图输出（A/B各一张）
        img_a = img.copy()
        if points:
            img_a = project_points(img_a, points, K, R_rect, P_rect, R_a, t_a, subsample=args.subsample)
        if lines_3d:
            img_a = project_3d_lines(img_a, lines_3d, K, R_rect, P_rect, R_a, t_a)
        img_a = add_legend(img_a)

        img_b = img.copy()
        if points:
            img_b = project_points(img_b, points, K, R_rect, P_rect, R_b, t_b, subsample=args.subsample)
        if lines_3d:
            img_b = project_3d_lines(img_b, lines_3d, K, R_rect, P_rect, R_b, t_b)
        img_b = add_legend(img_b)

        # 基于输出路径生成A/B文件名
        if not args.output:
            frame_id = os.path.basename(args.feature_base)
            output_dir = 'result/visualization'
            os.makedirs(output_dir, exist_ok=True)
            base_output = os.path.join(output_dir, f"{frame_id}_ab_compare")
        else:
            out_root, out_ext = os.path.splitext(args.output)
            if not out_ext:
                out_ext = ".png"
            base_output = out_root

        output_a = f"{base_output}_A_current.png"
        output_b = f"{base_output}_B_inverse.png"
        os.makedirs(os.path.dirname(output_a) if os.path.dirname(output_a) else ".", exist_ok=True)
        os.makedirs(os.path.dirname(output_b) if os.path.dirname(output_b) else ".", exist_ok=True)
        cv2.imwrite(output_a, img_a)
        cv2.imwrite(output_b, img_b)
        print(f"[AB Compare] Saved overlay A: {output_a}")
        print(f"[AB Compare] Saved overlay B: {output_b}")
        return 0

    # 4. 绘制投影
    print("\n[Projecting Features]")
    
    # 先画点 (作为背景)
    if points:
        img = project_points(img, points, K, R_rect, P_rect, R, t_vec, subsample=args.subsample)
    
    # 再画线 (更显眼)
    if lines_3d:
        img = project_3d_lines(img, lines_3d, K, R_rect, P_rect, R, t_vec)

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
