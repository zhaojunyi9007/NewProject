import cv2
import numpy as np
import os
import argparse
import sys

# Ensure repo root is importable when running as a script:
# `python tools/visualize.py ...` puts `tools/` on sys.path, not the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline.datasets import get_adapter

def _infer_sam_base_from_feature_base(feature_base: str) -> str:
    """
    Infer sam feature base path from lidar feature_base.
    Typical:
      result/lidar_features/0000000088  -> result/sam_features/0000000088
    """
    feature_dir = os.path.dirname(feature_base)
    if feature_dir and "lidar_features" in feature_dir:
        sam_dir = feature_dir.replace("lidar_features", "sam_features")
        return os.path.join(sam_dir, os.path.basename(feature_base))
    # Fallback: assume sibling directory name
    return feature_base

def load_edge_dist_map(sam_base: str):
    """
    Load SAM edge distance transform map produced by sam stage.
    Expected file: <sam_base>_edge_dist.png (float/uint, normalized 0..1 ideally).
    Returns: (dist_map_float01, raw_map) or (None, None)
    """
    if not sam_base:
        return None, None
    cand = f"{sam_base}_edge_dist.png"
    if not os.path.exists(cand):
        return None, None
    raw = cv2.imread(cand, cv2.IMREAD_UNCHANGED)
    if raw is None or raw.size == 0:
        return None, None
    dist = raw.astype(np.float32)
    # Robust normalization: map to [0,1] if not already.
    dmin = float(np.nanmin(dist))
    dmax = float(np.nanmax(dist))
    if dmax - dmin > 1e-12:
        dist01 = (dist - dmin) / (dmax - dmin)
    else:
        dist01 = np.zeros_like(dist, dtype=np.float32)
    dist01 = np.clip(dist01, 0.0, 1.0)
    return dist01, raw

def overlay_edge_dist(img_bgr: np.ndarray, dist01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    Overlay edge distance map on top of the RGB image.
    Lower distance (closer to edges) is shown hotter.
    """
    if dist01 is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if dist01.shape[0] != h or dist01.shape[1] != w:
        dist01 = cv2.resize(dist01, (w, h), interpolation=cv2.INTER_LINEAR)
    # Invert so edges (low dist) become high intensity
    inv = (1.0 - dist01) * 255.0
    inv_u8 = inv.astype(np.uint8)
    heat = cv2.applyColorMap(inv_u8, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(img_bgr, 1.0, heat, float(alpha), 0.0)


def load_rail_dist_map(sam_base: str):
    """
    Phase 7 (sam_2d): load rail distance map (<sam_base>_rail_dist.png), normalized to [0,1].
    """
    if not sam_base:
        return None, None
    cand = f"{sam_base}_rail_dist.png"
    if not os.path.exists(cand):
        return None, None
    raw = cv2.imread(cand, cv2.IMREAD_UNCHANGED)
    if raw is None or raw.size == 0:
        return None, None
    dist = raw.astype(np.float32)
    dmin = float(np.nanmin(dist))
    dmax = float(np.nanmax(dist))
    if dmax - dmin > 1e-12:
        dist01 = (dist - dmin) / (dmax - dmin)
    else:
        dist01 = np.zeros_like(dist, dtype=np.float32)
    dist01 = np.clip(dist01, 0.0, 1.0)
    return dist01, raw


def overlay_rail_region_centerline(img_bgr: np.ndarray, sam_base: str, alpha_region: float = 0.28) -> np.ndarray:
    """
    Phase 7 (sam_2d): overlay rail_region (green) and rail_centerline (yellow) — replaces LSD line_map as track cue.
    """
    if not sam_base:
        return img_bgr
    h, w = img_bgr.shape[:2]
    region = cv2.imread(f"{sam_base}_rail_region.png", cv2.IMREAD_GRAYSCALE)
    cl = cv2.imread(f"{sam_base}_rail_centerline.png", cv2.IMREAD_GRAYSCALE)
    out = img_bgr.copy()
    if region is not None and region.size > 0:
        rg = cv2.resize(region, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (rg > 0).astype(np.float32)
        green = np.zeros_like(out)
        green[:, :, 1] = 255
        m3 = mask[:, :, None]
        out = (out.astype(np.float32) * (1.0 - alpha_region * m3) + green.astype(np.float32) * (alpha_region * m3)).astype(
            np.uint8
        )
    if cl is not None and cl.size > 0:
        cg = cv2.resize(cl, (w, h), interpolation=cv2.INTER_NEAREST)
        m = cg > 0
        yl = np.array([0, 255, 255], dtype=np.float32)
        om = out[m].astype(np.float32)
        out[m] = (om * 0.45 + yl * 0.55).astype(np.uint8)
    return out


def edge_alignment_stats(points, R_rect, P_rect, R, t, img_w, img_h, dist01: np.ndarray):
    """
    Sample edge distance map at projected point locations.
    Returns dict with counts and quantiles for in-image projected points.
    dist01 is normalized to [0,1] where 0 means on-edge (best).
    """
    if dist01 is None or len(points) == 0:
        return None
    # Ensure map matches image size
    if dist01.shape[0] != img_h or dist01.shape[1] != img_w:
        dist01 = cv2.resize(dist01, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    samples = []
    behind = 0
    oob = 0
    for pt in points:
        p = np.array([pt[0], pt[1], pt[2]], dtype=np.float64)
        p_cam = R @ p + t
        p_rect = R_rect @ p_cam
        if p_rect[2] < 0.1:
            behind += 1
            continue
        uv = P_rect @ np.hstack([p_rect, 1.0])
        u = uv[0] / uv[2]
        v = uv[1] / uv[2]
        if not (0 <= u < img_w and 0 <= v < img_h):
            oob += 1
            continue
        ui = int(u)
        vi = int(v)
        samples.append(float(dist01[vi, ui]))

    if len(samples) == 0:
        return {
            "n_total": len(points),
            "n_sampled": 0,
            "behind": behind,
            "oob": oob,
        }

    arr = np.array(samples, dtype=np.float32)
    qs = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    return {
        "n_total": len(points),
        "n_sampled": int(arr.size),
        "behind": behind,
        "oob": oob,
        "mean": float(arr.mean()),
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
        "ratio_le_0.05": float((arr <= 0.05).mean()),
        "ratio_le_0.10": float((arr <= 0.10).mean()),
        "ratio_le_0.20": float((arr <= 0.20).mean()),
    }

def _infer_dataset_format_from_calib_file(calib_file: str) -> str:
    if not calib_file or not os.path.exists(calib_file):
        return "osdar23"
    try:
        with open(calib_file, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(256 * 1024).lower()
        # OSDaR23 calibration.txt typically contains these fields (not always in first 4KB).
        if ("intrinsics_pinhole" in head) or ("pose_wrt_parent" in head) or ("data_folder:" in head and "camera_matrix" in head):
            return "osdar23"
    except Exception:
        pass
    return "osdar23"


def _infer_osdar_camera_from_img_path(img_path: str) -> str:
    # OSDaR23 layout: <sequence_root>/<camera_folder>/<counter>_<timestamp>.png
    if not img_path:
        return "rgb_center"
    parent = os.path.basename(os.path.dirname(img_path))
    return parent or "rgb_center"


def load_calib_via_adapter(calib_file: str, dataset_format: str | None = None, image_sensor: str | None = None):
    """
    Load (K, R_rect, P_rect) via dataset adapters.
    This avoids ad-hoc parsing and keeps behavior consistent with the pipeline/optimizer.
    """
    fmt = (dataset_format or "").strip().lower() or _infer_dataset_format_from_calib_file(calib_file)
    cam = (image_sensor or "").strip() or "rgb_center"
    cfg = {"data": {"dataset_format": fmt, "calib_file": calib_file, "image_sensor": cam}}
    ds = get_adapter(cfg)
    return ds.load_intrinsics()

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
            
            # Phase A3：稳定 key-value 格式
            kv = {}
            for line in lines:
                s = line.strip()
                if not s or ":" not in s:
                    continue
                k, v = s.split(":", 1)
                kv[k.strip()] = v.strip()
            if "r" in kv and "t" in kv:
                r_vals = list(map(float, kv["r"].split()[:3]))
                t_vals = list(map(float, kv["t"].split()[:3]))
                if len(r_vals) == 3 and len(t_vals) == 3:
                    r_vec = np.array(r_vals)
                    t_vec = np.array(t_vals)
                    print(f"[Info] Loaded calibration result from {calib_result_file} (Phase A3 kv format)")
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
  python3 tools/visualize.py --img image.png --feature_base result/0000000000
  
  # 手动指定外参
  python3 tools/visualize.py --img image.png --feature_base result/0000000000 \\
      --r_vec 0.01 0.02 0.03 --t_vec 0.0 -0.3 1.8
  
  # 指定相机标定文件
  python3 tools/visualize.py --img image.png --feature_base result/0000000000 \\
      --calib_file /path/to/calib.txt
        """)
    
    parser.add_argument("--img", type=str, required=True, 
                        help="Original Image Path (e.g., /gz-data/dataset/.../image_02/data/0000000000.png)")
    parser.add_argument("--feature_base", type=str, required=True, 
                        help="Path to features (e.g., result/0000000000)")
    parser.add_argument("--calib_file", type=str, default="", 
                        help="Calibration file (optional, e.g., calibration.txt)")
    parser.add_argument("--dataset_format", type=str, default="",
                        help="可选：强制指定数据集格式（osdar23 / osdar）。不填则根据 calib_file 自动推断。")
    parser.add_argument("--image_sensor", type=str, default="",
                        help="可选：OSDaR23 相机 data_folder（如 rgb_center）。不填则尝试从 --img 路径推断。")
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
    parser.add_argument("--overlay_edge_dist", action="store_true",
                        help="Overlay SAM edge distance map on output image (auto-infer sam_base from feature_base).")
    parser.add_argument("--edge_dist_alpha", type=float, default=0.35,
                        help="Alpha for edge distance overlay (0..1).")
    parser.add_argument(
        "--overlay_rail_maps",
        action="store_true",
        help="Phase 7 (sam_2d): overlay rail_region + rail_centerline (replaces LSD line_map as main track cue).",
    )
    parser.add_argument(
        "--overlay_rail_dist",
        action="store_true",
        help="Phase 7 (sam_2d): overlay rail_dist heatmap (same semantics as edge_dist: low=on track).",
    )
    parser.add_argument("--rail_dist_alpha", type=float, default=0.35, help="Alpha for rail_dist overlay (0..1).")
    parser.add_argument("--diag", action="append", choices=["bev", "semantic", "refine", "rail"],
                        help="Phase 7：额外生成诊断拼图（bev/semantic/refine/rail，可重复指定）")
    parser.add_argument("--image_features_frame", type=str, default="",
                        help="image_features 下单帧目录，如 .../image_features/0000000012（用于 BEV 拼图）")
    parser.add_argument("--sam_frame_dir", type=str, default="",
                        help="SAM 单帧目录（含 semantic_argmax.png 时优先）或留空由 feature_base 推断 sam_features")
    parser.add_argument("--refinement_dir", type=str, default="",
                        help="refinement 输出目录（含 state.json，用于精修曲线）")

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

    img_clean = img.copy()
    print(f"[Info] Image loaded: {img.shape[1]}x{img.shape[0]}")

    # 1. 加载特征
    points, lines_3d = load_features(args.feature_base, point_source=args.point_source)
    
    if not points and not lines_3d:
        print("[Error] No features loaded. Nothing to visualize.")
        print(f"[Info] Expected files: {args.feature_base}_points.txt (or _edge_points.txt), {args.feature_base}_lines_3d.txt")
        return 1

    # 2. 加载相机内参/整流/投影矩阵（KITTI / OSDaR23 自动识别）
    # Prefer dataset adapter to keep behavior aligned with the pipeline.
    # For OSDaR23, allow selecting camera intrinsics by image_sensor (data_folder).
    img_sensor = args.image_sensor.strip() or _infer_osdar_camera_from_img_path(args.img)
    ds_fmt = args.dataset_format.strip().lower() or _infer_dataset_format_from_calib_file(args.calib_file)
    K, R_rect_loaded, P_rect = load_calib_via_adapter(args.calib_file, dataset_format=ds_fmt, image_sensor=img_sensor)
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

    # Optional overlays: Phase 7 (sam_2d) rail_region / rail_centerline / rail_dist; legacy edge_dist.
    sam_base = _infer_sam_base_from_feature_base(args.feature_base)
    dist01, _ = load_edge_dist_map(sam_base)
    rail_dist01, _ = load_rail_dist_map(sam_base)
    if args.overlay_rail_maps:
        img = overlay_rail_region_centerline(img, sam_base)
    if rail_dist01 is not None and args.overlay_rail_dist:
        img = overlay_edge_dist(img, rail_dist01, alpha=args.rail_dist_alpha)
    if dist01 is not None and args.overlay_edge_dist:
        img = overlay_edge_dist(img, dist01, alpha=args.edge_dist_alpha)
    
    # 先画点 (作为背景)
    if points:
        img = project_points(img, points, K, R_rect, P_rect, R, t_vec, subsample=args.subsample)
    
    # 再画线 (更显眼)
    if lines_3d:
        img = project_3d_lines(img, lines_3d, K, R_rect, P_rect, R, t_vec)

    # 4b. Phase 7 诊断图（在图例之前，使用干净背景做语义拼图）
    if args.diag:
        _tools_d = os.path.dirname(os.path.abspath(__file__))
        if _tools_d not in sys.path:
            sys.path.insert(0, _tools_d)
        import visualize_diag as vd  # noqa: E402

        out_root, out_ext = os.path.splitext(args.output)
        if not out_ext:
            out_ext = ".png"
        sam_d = args.sam_frame_dir.strip() or _infer_sam_base_from_feature_base(args.feature_base)
        if "bev" in args.diag and args.image_features_frame.strip():
            vd.render_bev_panel(args.feature_base, args.image_features_frame.strip(), out_root + "_diag_bev.png")
        elif "bev" in args.diag:
            print("[Warning] --diag bev 需要 --image_features_frame")
        if "semantic" in args.diag:
            vd.render_semantic_panel(
                img_clean,
                args.feature_base,
                sam_d,
                sam_d,
                K,
                R_rect,
                P_rect,
                R,
                t_vec,
                out_root + "_diag_semantic.png",
            )
        if "refine" in args.diag and args.refinement_dir.strip():
            vd.render_refine_curves(args.refinement_dir.strip(), out_root + "_diag_refine.png")
        elif "refine" in args.diag:
            print("[Warning] --diag refine 需要 --refinement_dir")
        if "rail" in args.diag:
            vd.render_rail_panel(sam_d, out_root + "_diag_rail.png")

    # 5. 添加图例
    img = add_legend(img)

    # Print quantitative edge-alignment stats (samples DT values at projected points).
    if dist01 is not None and points:
        h, w = img.shape[:2]
        stats = edge_alignment_stats(points, R_rect, P_rect, R, t_vec, w, h - 80, dist01)
        # h includes legend stacked; use original image height (h-80)
        if stats and stats.get("n_sampled", 0) > 0:
            print("\n[Edge Alignment Stats] (edge_dist at projected points; 0=on-edge, 1=far)")
            print(f"  sampled={stats['n_sampled']}/{stats['n_total']}, behind={stats['behind']}, oob={stats['oob']}")
            print(f"  mean={stats['mean']:.4f}, p50={stats['p50']:.4f}, p25/p75={stats['p25']:.4f}/{stats['p75']:.4f}")
            print(f"  ratio<=0.05={stats['ratio_le_0.05']:.3f}, <=0.10={stats['ratio_le_0.10']:.3f}, <=0.20={stats['ratio_le_0.20']:.3f}")
    if rail_dist01 is not None and points and args.overlay_rail_dist:
        h, w = img.shape[:2]
        rstats = edge_alignment_stats(points, R_rect, P_rect, R, t_vec, w, h - 80, rail_dist01)
        if rstats and rstats.get("n_sampled", 0) > 0:
            print("\n[Rail Alignment Stats] (rail_dist at projected points; 0=near centerline, 1=far)")
            print(f"  sampled={rstats['n_sampled']}/{rstats['n_total']}, mean={rstats['mean']:.4f}, p50={rstats['p50']:.4f}")

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
