import numpy as np
import cv2
import torch
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class FeatureExtractor:
    """
    统一的特征提取器:使用SAM进行分割,提取mask、2D线特征等
    """
     def __init__(
        self,
        checkpoint_path,
        model_type="vit_h",
        device=None,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=500,
        heuristics=None,
    ):
        # 加载 SAM 模型
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[SAM] Initializing model ({model_type}) on {self.device}...")
        
        # 检查checkpoint文件是否存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # 加载SAM模型
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        
        # 设置 16x16 的网格提示生成掩码
        self.mask_generator = SamAutomaticMaskGenerator(
            sam, 
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=min_mask_region_area,
        )
        self.heuristics = {
            "min_mask_area_ratio": 0.001,
            "max_background_area_ratio": 0.15,
            "sky_mask_bottom_ratio": 0.3,
            "ground_region_top_ratio": 0.5,
            "flat_ground_aspect_ratio": 3.0,
            "structural_aspect_ratio": 3.0,
            "contour_stddev_threshold": 10.0,
            "min_arc_length_ratio": 0.06,
            "global_min_line_length_ratio": 0.015,
            "ground_min_line_length_ratio": 0.06,
            "sky_min_line_length_ratio": 0.03,
            "fused_top_black_ratio": 0.20,
            "fused_bottom_black_ratio": 0.80,
            "distance_max_ratio": 0.15,
        }
        if isinstance(heuristics, dict):
            self.heuristics.update(heuristics)
        
        print(f"[SAM] Model loaded successfully")

    def extract_edges(self, image):
        """
        提取边缘,返回二值边缘图、权重图和mask id图
        """
        # 1. 生成原始掩码
        print("[SAM] Generating masks...")
        masks = self.mask_generator.generate(image)
        print(f"[SAM] Generated {len(masks)} masks")
        
        # 2. 初始化最终边缘图
        h, w = image.shape[:2]
        final_edge_map = np.zeros((h, w), dtype=np.uint8)
        # 引入注意力权重图，记录每个边缘点的质量
        weight_map = np.zeros((h, w), dtype=np.float32)
        mask_id_map = np.zeros((h, w), dtype=np.uint16)
        stability_map = np.zeros((h, w), dtype=np.float32)

        # --- 新增：计算动态基准尺度 ---
        image_area = h * w
        max_dim = max(h, w)
        min_area_thresh = image_area * float(self.heuristics["min_mask_area_ratio"])
        min_arc_length = max_dim * float(self.heuristics["min_arc_length_ratio"])
        # ----------------------------
 
        for idx, mask in enumerate(masks):
            # 提取掩码元数据
            m_bool = mask['segmentation']
            stability = mask['stability_score'] # SAM 的边缘稳定性得分
            bbox = mask['bbox'] # [x, y, w, h]
            mask_id = mask.get('id', idx + 1)

            x, y, bw, bh = bbox
            area = mask.get('area', bw * bh)

            # 过滤1：超大面积背景
            if area > image_area * float(self.heuristics["max_background_area_ratio"]):
                continue
            # 过滤2：极小噪点 (修改为动态比例)
            if area < min_area_thresh:
                continue

            # 过滤3：纯天上物体（物体最底端 y+bh 都在图像上半部，多为树冠）
            if (y + bh) < h * float(self.heuristics["sky_mask_bottom_ratio"]):
                continue
            # 过滤4：地面扁平物体（马路上的巨大横向阴影、斑马线）
            if y > h * float(self.heuristics["ground_region_top_ratio"]) and bw > bh * float(self.heuristics["flat_ground_aspect_ratio"]):
                continue
        
            # 几何过滤逻辑 (空旷场景优先保留长条形结构如护栏、轨道)
            bw, bh = bbox[2], bbox[3]
            aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
            # 长宽比较大的 mask 通常代表道路边界或护栏
            is_structural = aspect_ratio > 3.0 

            m = m_bool.astype(np.uint8)
            if not m.flags['C_CONTIGUOUS']:
                m = np.ascontiguousarray(m)

            # 使用形态学梯度/Sobel 提取掩码边界
            grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            sobel_x = cv2.Sobel(grad, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(grad, cv2.CV_32F, 0, 1, ksize=3)
            edge_strength = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            edge_strength = (edge_strength > 0).astype(np.uint8) * 255

            # 提取物体轮廓
            contours, _ = cv2.findContours(edge_strength, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                edge_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(edge_mask, [cnt], -1, 255, 1)
                pixels = image[edge_mask == 255]
            
                if len(pixels) == 0: 
                    continue

                # 仅保留高置信度的边界 (修改 arcLength 阈值为动态比例)
                if np.std(pixels) > float(self.heuristics["contour_stddev_threshold"]) and cv2.arcLength(cnt, True) >= min_arc_length:
                    weight = stability * (1.5 if is_structural else 1.0)
                    cv2.drawContours(final_edge_map, [cnt], -1, 255, 1)
                    cv2.drawContours(weight_map, [cnt], -1, float(weight), 1)

            # 生成mask id map (保留稳定性更高的掩码)
            update_mask = m_bool & (stability > stability_map)
            if np.any(update_mask):
                mask_id_map[update_mask] = mask_id
                stability_map[update_mask] = stability

        return final_edge_map, weight_map, mask_id_map
        

    def get_distance_transform(self, edge_map):
        """
        计算距离变换，使用固定尺度而非自适应归一化
        这样可以保证多帧之间的残差尺度一致
        """
        print("[SAM] Computing Distance Transform...")
        # dist_map 的每个像素值代表该点到最近边缘的距离（单位：像素）
        # 输入要求：边缘是白色(255)，背景是黑色(0) -> 需要反转一下给 distanceTransform
        # distanceTransform 计算的是"到零像素的距离"，所以我们要把边缘变成0
        dist_map = cv2.distanceTransform(255 - edge_map, cv2.DIST_L2, 5)

        # --- 修改：使用基于图像尺寸的动态归一化尺度 ---
        h, w = edge_map.shape[:2]
        # 默认使用最大边长的 15% 作为距离场截断值，也可通过环境变量强行指定
        default_max_dist = max(h, w) * float(self.heuristics["distance_max_ratio"]) 
        FIXED_MAX_DIST = float(os.environ.get("SAM_MAX_DIST", default_max_dist))
        
        # 使用固定的最大距离进行归一化（单位：像素）
        # KITTI图像尺寸约 1242x375，选择200像素作为最大有效距离
        # 这样可以保证不同帧之间的残差尺度一致
        dist_map = np.clip(dist_map, 0, FIXED_MAX_DIST) / FIXED_MAX_DIST
        
        actual_max = np.max(dist_map * FIXED_MAX_DIST)
        print(f"[SAM] Distance map normalized with fixed scale")
        print(f"      Fixed max: {FIXED_MAX_DIST}px, Actual max: {actual_max:.2f}px, Range: [0, 1]")
        
        return dist_map

    def extract_lines_2d(self, image, edge_map=None, return_mask=False):
        """
        使用LSD (Line Segment Detector) 提取2D线特征
        返回格式: [(u1, v1, u2, v2, type), ...]
        type: 0=Horizontal, 1=Vertical
        """
        print("[SAM] Extracting 2D line features...") 

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        
        # 如果没有提供edge_map，先提取边缘
        if edge_map is None:
            edge_map, _, _ = self.extract_edges(image)
        
        # 使用OpenCV的LSD检测器（基于灰度图）
        lsd = cv2.createLineSegmentDetector(0)
        lines, _, _, _ = lsd.detect(gray)
        
        if lines is None:
            print("[SAM] No lines detected")
            return []
        
        # 过滤和分类线段
        lines_2d = []
        h, w = image.shape[:2]
        line_mask = np.zeros((h, w), dtype=np.uint8)
        min_length = 20  # 最小线段长度（像素）
        top_region = int(h * 0.4)  # 图像上方区域（电缆）

        # --- 新增：计算动态线段长度阈值 ---
        max_dim = max(h, w)
        min_length = max_dim * float(self.heuristics["global_min_line_length_ratio"])
        min_ground_length = max_dim * float(self.heuristics["ground_min_line_length_ratio"])
        min_sky_length = max_dim * float(self.heuristics["sky_min_line_length_ratio"])
        # --------------------------------
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算线段长度
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < min_length:
                continue
            
            # 计算线段方向，判断水平/垂直
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            # 判断类型：dy/dx > 2 为垂直，dx/dy > 2 为水平
            if dy > dx * 2:
                line_type = 1  # Vertical
            elif dx > dy * 2:
                line_type = 0  # Horizontal
            else:
                continue  # 跳过斜线

            # ===== 新增的严厉过滤逻辑 =====
            # 计算这条线的中心点Y坐标
            mid_y = (y1 + y2) / 2.0
            
            # 判断线在图像的下半部(地面)还是上半部(天空/建筑)
            if mid_y > h * float(self.heuristics["ground_region_top_ratio"]): 
                # 【情况A：在地面】
                # 地面上有很多斑马线、阴影。我们只想要铁轨！
                # 铁轨通常很长，所以把长度小于 80 像素的短线全部干掉。
                if length < min_ground_length:
                    continue 
            else:
                # 【情况B：在天上或背景里】
                # 天上的横线可能是电缆，但也可能是树枝。把小于 40 的短横线干掉。
                if line_type == 0 and length < min_sky_length: 
                    continue
            # ==============================
            
            # 如果能活过上面的重重过滤，才把它加进最终的特征里
            lines_2d.append((x1, y1, x2, y2, line_type))
            cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
        
        print(f"[SAM] Extracted {len(lines_2d)} 2D line features")
        if return_mask:
            return lines_2d, line_mask
        return lines_2d

    def build_edge_attraction_field(self, image):
        """
        融合 SAM 边缘与 LSD 直线，生成边缘吸引场与权重图
        """
        edge_map, weight_map, mask_id_map = self.extract_edges(image)
        lines_2d, line_mask = self.extract_lines_2d(image, edge_map, return_mask=True)
        
        fused_edge_map = cv2.bitwise_or(edge_map, line_mask)
        h_img, w_img = fused_edge_map.shape
        # 强制抹黑顶部 20% (纯天空和树顶边缘)
        fused_edge_map[0:int(h_img * float(self.heuristics["fused_top_black_ratio"])), :] = 0
        # 强制抹黑底部 20% (本车引擎盖、近处地面的车道线)
        fused_edge_map[int(h_img * float(self.heuristics["fused_bottom_black_ratio"])):h_img, :] = 0

        dist_map = self.get_distance_transform(fused_edge_map)
        dist_map = cv2.GaussianBlur(dist_map, (5, 5), 0)
        weight_map = cv2.GaussianBlur(weight_map, (5, 5), 0)
        return dist_map, weight_map, edge_map, line_mask, fused_edge_map, lines_2d, mask_id_map

    def process_image(self, image_path, output_dir):
        """
        处理单张图像，生成所有必要的特征文件
        生成文件：
        - xxx_lines_2d.txt: 2D线特征 (用于Fine)
        - xxx_edge_map.png: SAM边缘图
        - xxx_line_map.png: LSD直线图
        - xxx_edge_fused.png: 融合边缘图
        - xxx_edge_dist.png: 边缘吸引场 (16-bit PNG)
        - xxx_edge_weight.png: 边缘权重图 (16-bit PNG)
        - xxx_mask_ids.png: SAM Mask ID图 (16-bit PNG)
        - xxx_semantic_map.png: 语义ID图 (16-bit PNG)
        """
        print(f"\n[Processing] {image_path}")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"[Error] Cannot read image: {image_path}")
            return False
        
        # 提取文件名（不含扩展名）
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_base = os.path.join(output_dir, filename)

        # 1. 生成融合边缘与吸引场
        dist_map, weight_map, edge_map, line_mask, fused_edge_map, lines_2d, mask_id_map = (
            self.build_edge_attraction_field(image)
        )
        cv2.imwrite(output_base + "_edge_map.png", edge_map)
        cv2.imwrite(output_base + "_line_map.png", line_mask)
        cv2.imwrite(output_base + "_edge_fused.png", fused_edge_map)

        dist_u16 = np.clip(dist_map, 0, 1) * 65535.0
        cv2.imwrite(output_base + "_edge_dist.png", dist_u16.astype(np.uint16))

        max_weight = float(weight_map.max()) if weight_map.size else 0.0
        if max_weight > 0:
            weight_norm = np.clip(weight_map / max_weight, 0, 1)
        else:
            weight_norm = np.zeros_like(weight_map, dtype=np.float32)
        weight_u16 = weight_norm * 65535.0
        cv2.imwrite(output_base + "_edge_weight.png", weight_u16.astype(np.uint16))

        cv2.imwrite(output_base + "_mask_ids.png", mask_id_map.astype(np.uint16))
        cv2.imwrite(output_base + "_semantic_map.png", mask_id_map.astype(np.uint16))

        print(f"[Saved] {output_base}_edge_map.png")
        print(f"[Saved] {output_base}_line_map.png")
        print(f"[Saved] {output_base}_edge_fused.png")
        print(f"[Saved] {output_base}_edge_dist.png")
        print(f"[Saved] {output_base}_edge_weight.png")
        print(f"[Saved] {output_base}_mask_ids.png")
        print(f"[Saved] {output_base}_semantic_map.png")

        # 2. 提取2D线特征
        with open(output_base + "_lines_2d.txt", 'w') as f:
            f.write("# 2D Line Features: u1 v1 u2 v2 type (0=Horizontal, 1=Vertical)\n")
            for line in lines_2d:
                f.write(f"{line[0]:.2f} {line[1]:.2f} {line[2]:.2f} {line[3]:.2f} {line[4]}\n")      
        print(f"[Saved] {output_base}_lines_2d.txt ({len(lines_2d)} lines)")
        
        print(f"[Complete] {filename}")
        return True


# 保持向后兼容性的别名
SAMEdgeExtractor = FeatureExtractor
