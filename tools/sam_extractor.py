import numpy as np
import cv2
import torch
import os
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

_tools_dir = os.path.dirname(os.path.abspath(__file__))
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)
from semantic_to_bev import semantic_probs_to_pseudo_bev


def _semantic_class_indices(semantic_classes, names):
    out = []
    for n in names:
        if n in semantic_classes:
            out.append(semantic_classes.index(n))
    return out


def _mask_to_class_weights(mask_dict, h, w, semantic_classes):
    """Heuristic soft class weights for one SAM mask -> vector len C."""
    C = len(semantic_classes)
    wts = np.zeros(C, dtype=np.float32)
    bbox = mask_dict["bbox"]
    x, y, bw, bh = bbox
    area = float(mask_dict.get("area", bw * bh))
    image_area = float(h * w)
    cy = (y + bh * 0.5) / max(h, 1)
    aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
    ar = area / (image_area + 1e-8)

    def add(name, val):
        if name in semantic_classes:
            wts[semantic_classes.index(name)] += val

    if cy < 0.26:
        add("sky", 3.0)
    if cy > 0.36 and aspect > 2.0 and bw >= bh * 0.9:
        add("rail", 2.8)
        add("ballast", 1.6)
    if aspect < 0.52 and bh > bw * 1.15 and 0.08 < cy < 0.92:
        add("pole", 2.6)
    if ar > 0.012 and aspect < 5.0:
        add("building", 2.0)
    if cy > 0.52 and aspect > 1.4 and bw >= bh:
        add("road", 1.8)
    if 0.008 < ar < 0.09 and cy > 0.42:
        add("vehicle", 1.6)
    if 0.25 < cy < 0.72 and 0.002 < ar < 0.06:
        add("vegetation", 1.2)
    if cy < 0.36 and ar < 0.012:
        add("signal", 1.8)
    if 0.012 < ar < 0.11 and 0.28 < cy < 0.72:
        add("platform", 1.2)

    if wts.sum() < 1e-6:
        wts[:] = 1.0 / C
    else:
        wts = wts / (wts.sum() + 1e-8)
    return wts


def extract_semantic_probabilities(image, masks, config):
    """
    由 SAM masks 估计每像素语义概率（启发式，与类别顺序一致）。
    config 需含 semantic_classes: List[str]
    返回 (semantic_probs, semantic_logits)，形状 H×W×C，float32。
    """
    h, w = image.shape[:2]
    semantic_classes = list(config.get("semantic_classes", []))
    if not semantic_classes:
        semantic_classes = [
            "rail",
            "ballast",
            "pole",
            "signal",
            "platform",
            "building",
            "road",
            "vehicle",
            "vegetation",
            "sky",
        ]
    C = len(semantic_classes)
    acc = np.zeros((h, w, C), dtype=np.float32)
    for mask in masks:
        seg = mask.get("segmentation")
        if seg is None:
            continue
        m = seg.astype(np.float32)
        cw = _mask_to_class_weights(mask, h, w, semantic_classes)
        acc += m[:, :, None] * cw[None, None, :]

    s = acc.sum(axis=2, keepdims=True) + 1e-8
    probs = acc / s
    empty = acc.sum(axis=2) < 1e-5
    if np.any(empty):
        uni = np.ones(C, dtype=np.float32) / C
        probs[empty] = uni
    sky_i = semantic_classes.index("sky") if "sky" in semantic_classes else -1
    if sky_i >= 0:
        for row in range(int(h * 0.22)):
            probs[row, :, sky_i] += 0.25
    probs = probs / (probs.sum(axis=2, keepdims=True) + 1e-8)
    logits = np.log(np.clip(probs, 1e-8, 1.0))
    return probs.astype(np.float32), logits.astype(np.float32)


def build_semantic_edge_map(semantic_probs, config):
    """
    在 rail / pole / platform / building 概率通道上取梯度能量并融合为 uint8 边缘图。
    """
    semantic_classes = list(config.get("semantic_classes", []))
    names = ("rail", "pole", "platform", "building")
    idxs = [semantic_classes.index(n) for n in names if n in semantic_classes]
    if not idxs:
        h, w = semantic_probs.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)
    energy = np.zeros(semantic_probs.shape[:2], dtype=np.float32)
    for i in idxs:
        ch = semantic_probs[:, :, i].astype(np.float32)
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        energy += np.sqrt(gx * gx + gy * gy)
    emax = float(energy.max()) + 1e-8
    out = np.clip(energy / emax * 255.0, 0, 255).astype(np.uint8)
    return out


def _line_semantic_support_map(semantic_probs, config):
    """融合 rail/ballast、pole/signal、platform/building 边界用于 LSD 门控。"""
    semantic_classes = list(config.get("semantic_classes", []))
    h, w = semantic_probs.shape[:2]
    rail_names = list(config.get("rail_class_names", ["rail", "ballast"]))
    vert_names = list(config.get("vertical_structure_classes", ["pole", "signal"]))
    ri = _semantic_class_indices(semantic_classes, rail_names)
    vi = _semantic_class_indices(semantic_classes, vert_names)
    pi = _semantic_class_indices(semantic_classes, ["platform", "building"])

    rail_p = np.zeros((h, w), dtype=np.float32)
    for i in ri:
        rail_p += semantic_probs[:, :, i]
    pole_p = np.zeros((h, w), dtype=np.float32)
    for i in vi:
        pole_p += semantic_probs[:, :, i]
    struct = np.zeros((h, w), dtype=np.float32)
    for i in pi:
        struct += semantic_probs[:, :, i]
    gx = cv2.Sobel(struct, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(struct, cv2.CV_32F, 0, 1, ksize=3)
    struct_edge = np.sqrt(gx * gx + gy * gy)
    smax = float(struct_edge.max()) + 1e-8
    struct_edge = struct_edge / smax

    support = np.maximum(rail_p, pole_p) + 0.45 * struct_edge
    support = np.clip(support, 0.0, 1.0)
    return support.astype(np.float32)


def extract_lines_2d(
    image,
    edge_map,
    config,
    semantic_probs=None,
    heuristics=None,
    image_features_cfg=None,
    require_edge_overlap=None,
):
    """
    LSD 2D 线段；可选用语义支持图门控；保留斜线（type=2）。
    config: 可为空 dict；image_features_cfg 覆盖 bottom_crop、门控阈值等。
    返回 list of (u1,v1,u2,v2,type,semantic_support)，type: 0=水平,1=垂直,2=斜向。
    """
    heuristics = heuristics or {}
    ifc = image_features_cfg if isinstance(image_features_cfg, dict) else {}
    bottom_crop = float(ifc.get("bottom_crop_ratio_for_edges", config.get("bottom_crop_ratio_for_edges", 0.0)))
    restrict_sem = bool(ifc.get("restrict_lsd_by_semantics", config.get("restrict_lsd_by_semantics", True)))
    keep_diag = bool(ifc.get("keep_diagonal_lines", config.get("keep_diagonal_lines", True)))
    support_thresh = float(ifc.get("line_semantic_support_threshold", config.get("line_semantic_support_threshold", 0.12)))
    if require_edge_overlap is None:
        require_edge_overlap = semantic_probs is not None

    print("[SAM] Extracting 2D line features...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]

    support_map = None
    if semantic_probs is not None and restrict_sem:
        support_map = _line_semantic_support_map(semantic_probs, {**config, **ifc})
        if bottom_crop > 1e-6:
            y0 = int(h * (1.0 - bottom_crop))
            support_map[y0:h, :] = 0.0
        ksz = max(3, int(round(min(h, w) * 0.002)) | 1)
        support_map = cv2.dilate(support_map, np.ones((ksz, ksz), np.uint8))

    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(gray)
    if lines is None:
        print("[SAM] No lines detected")
        return []

    lines_2d = []
    line_mask = np.zeros((h, w), dtype=np.uint8)
    max_dim = max(h, w)
    min_length = max_dim * float(heuristics.get("global_min_line_length_ratio", 0.015))
    min_ground_length = max_dim * float(heuristics.get("ground_min_line_length_ratio", 0.06))
    min_sky_length = max_dim * float(heuristics.get("sky_min_line_length_ratio", 0.03))
    ground_top = float(heuristics.get("ground_region_top_ratio", 0.5))

    em_crop = edge_map
    if bottom_crop > 1e-6 and edge_map is not None:
        y0 = int(h * (1.0 - bottom_crop))
        em_crop = edge_map.copy()
        em_crop[y0:h, :] = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_length:
            continue

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy > dx * 2:
            line_type = 1
        elif dx > dy * 2:
            line_type = 0
        else:
            line_type = 2
            if not keep_diag:
                continue

        mid_y = (y1 + y2) * 0.5
        if mid_y > h * ground_top:
            if length < min_ground_length:
                continue
        else:
            if line_type == 0 and length < min_sky_length:
                continue

        sem_score = 1.0
        if support_map is not None:
            n = max(8, int(length / 5.0))
            xs = np.linspace(x1, x2, n)
            ys = np.linspace(y1, y2, n)
            vals = []
            for xi, yi in zip(xs, ys):
                u, v = int(np.clip(xi, 0, w - 1)), int(np.clip(yi, 0, h - 1))
                vals.append(support_map[v, u])
            sem_score = float(np.mean(vals)) if vals else 0.0
            if sem_score < support_thresh:
                continue
        if require_edge_overlap and em_crop is not None and np.any(em_crop > 0):
            n = max(6, int(length / 8.0))
            xs = np.linspace(x1, x2, n)
            ys = np.linspace(y1, y2, n)
            on_edge = 0
            for xi, yi in zip(xs, ys):
                u, v = int(np.clip(xi, 0, w - 1)), int(np.clip(yi, 0, h - 1))
                if em_crop[v, u] > 0:
                    on_edge += 1
            if on_edge < max(1, n // 6):
                continue

        lines_2d.append((x1, y1, x2, y2, line_type, sem_score))
        cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)

    print(f"[SAM] Extracted {len(lines_2d)} 2D line features")
    return lines_2d


def save_image_feature_bundle(output_dir, bundle, prefix=""):
    """
    保存图像特征包到 output_dir（prefix 用于扁平文件名前缀，可为空）。
    bundle: dict，含 semantic_probs, semantic_logits, edge_map, edge_weight_u16,
      line_mask, lines_2d, rail_png, pole_png, semantic_argmax, pseudo_bev_path 等。
    """
    os.makedirs(output_dir, exist_ok=True)
    if "semantic_probs" in bundle:
        np.save(os.path.join(output_dir, "semantic_probs.npy"), bundle["semantic_probs"])
    if "semantic_logits" in bundle and bundle.get("save_logits", True):
        np.save(os.path.join(output_dir, "semantic_logits.npy"), bundle["semantic_logits"])
    if "semantic_argmax" in bundle:
        cv2.imwrite(os.path.join(output_dir, "semantic_argmax.png"), bundle["semantic_argmax"])
    if "rail_prob_png" in bundle:
        cv2.imwrite(os.path.join(output_dir, "rail_prob.png"), bundle["rail_prob_png"])
    if "pole_prob_png" in bundle:
        cv2.imwrite(os.path.join(output_dir, "pole_prob.png"), bundle["pole_prob_png"])
    if "edge_map" in bundle:
        cv2.imwrite(os.path.join(output_dir, "edge_map.png"), bundle["edge_map"])
    if "edge_weight_u16" in bundle:
        cv2.imwrite(os.path.join(output_dir, "edge_weight.png"), bundle["edge_weight_u16"])
    if "line_mask" in bundle:
        cv2.imwrite(os.path.join(output_dir, "line_map.png"), bundle["line_mask"])
    if "lines_2d" in bundle:
        with open(os.path.join(output_dir, "lines_2d.txt"), "w", encoding="utf-8") as f:
            f.write("# u1 v1 u2 v2 type semantic_support (type: 0=H, 1=V, 2=diagonal)\n")
            for t in bundle["lines_2d"]:
                if len(t) >= 6:
                    f.write(f"{t[0]:.2f} {t[1]:.2f} {t[2]:.2f} {t[3]:.2f} {t[4]} {t[5]:.4f}\n")
                else:
                    f.write(f"{t[0]:.2f} {t[1]:.2f} {t[2]:.2f} {t[3]:.2f} {t[4]} 1.0000\n")
    if "pseudo_bev" in bundle and bundle["pseudo_bev"] is not None:
        path = os.path.join(output_dir, "pseudo_bev.npz")
        np.savez_compressed(path, **bundle["pseudo_bev"])


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

    def extract_edges(self, image, masks=None):
        """
        提取边缘,返回二值边缘图、权重图和mask id图。
        若传入 masks（与 generate 相同结构），则跳过重复 SAM 推理。
        """
        if masks is None:
            print("[SAM] Generating masks...")
            masks = self.mask_generator.generate(image)
        print(f"[SAM] Using {len(masks)} masks for edge extraction")
        
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
        使用 LSD 提取 2D 线（兼容旧行为：仅水平/垂直，无语义门控）。
        返回 [(u1, v1, u2, v2, type), ...]，type: 0=Horizontal, 1=Vertical
        """
        if edge_map is None:
            edge_map, _, _ = self.extract_edges(image)
        legacy_cfg = {
            "restrict_lsd_by_semantics": False,
            "keep_diagonal_lines": False,
            "bottom_crop_ratio_for_edges": 0.0,
        }
        lines_6 = extract_lines_2d(
            image,
            edge_map,
            legacy_cfg,
            semantic_probs=None,
            heuristics=self.heuristics,
            image_features_cfg=legacy_cfg,
            require_edge_overlap=False,
        )
        lines_2d = [(a[0], a[1], a[2], a[3], a[4]) for a in lines_6]
        line_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for ln in lines_2d:
            cv2.line(line_mask, (int(ln[0]), int(ln[1])), (int(ln[2]), int(ln[3])), 255, 1)
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

    def process_image(self, image_path, output_dir, output_prefix=None):
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
        
        # 输出前缀：优先使用调用者传入的 output_prefix（逻辑帧号），
        # 否则回退到原始图像文件名 stem（兼容旧行为）。
        if output_prefix:
            filename = output_prefix
        else:
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

    def process_image_feature_bundle(
        self,
        image_path,
        frame_bundle_dir,
        sam_output_base,
        image_features_cfg,
        bev_cfg,
        intrinsics,
        rvec,
        tvec,
        dataset_meta,
    ):
        """
        语义优先：在 frame_bundle_dir 写入 Phase 2 全量产物，
        并写入 sam_output_base_* 与 process_image 一致，供 C++ 优化器读取。
        """
        print(f"\n[Processing semantic bundle] {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"[Error] Cannot read image: {image_path}")
            return False

        merged = dict(image_features_cfg or {})
        merged.setdefault(
            "semantic_classes",
            [
                "rail",
                "ballast",
                "pole",
                "signal",
                "platform",
                "building",
                "road",
                "vehicle",
                "vegetation",
                "sky",
            ],
        )

        print("[SAM] Generating masks (single forward)...")
        masks = self.mask_generator.generate(image)

        probs, logits = extract_semantic_probabilities(image, masks, merged)
        edge_sem = build_semantic_edge_map(probs, merged)

        edge_map_sam, weight_map, mask_id_map = self.extract_edges(image, masks=masks)

        lines_full = extract_lines_2d(
            image,
            edge_sem,
            merged,
            semantic_probs=probs,
            heuristics=self.heuristics,
            image_features_cfg=merged,
            require_edge_overlap=True,
        )

        h, w = image.shape[:2]
        line_mask = np.zeros((h, w), dtype=np.uint8)
        for ln in lines_full:
            cv2.line(
                line_mask,
                (int(ln[0]), int(ln[1])),
                (int(ln[2]), int(ln[3])),
                255,
                1,
            )

        fused_edge_map = cv2.bitwise_or(edge_sem, line_mask)
        fused_edge_map = cv2.bitwise_or(fused_edge_map, edge_map_sam)
        fused_edge_map[0 : int(h * float(self.heuristics["fused_top_black_ratio"])), :] = 0
        bottom_crop = float(merged.get("bottom_crop_ratio_for_edges", 0.0))
        if bottom_crop > 1e-6:
            fused_edge_map[int(h * (1.0 - bottom_crop)) : h, :] = 0
        else:
            fused_edge_map[int(h * float(self.heuristics["fused_bottom_black_ratio"])) : h, :] = 0

        dist_map = self.get_distance_transform(fused_edge_map)
        dist_map = cv2.GaussianBlur(dist_map, (5, 5), 0)
        weight_map = cv2.GaussianBlur(weight_map, (5, 5), 0)

        max_weight = float(weight_map.max()) if weight_map.size else 0.0
        if max_weight > 0:
            weight_norm = np.clip(weight_map / max_weight, 0, 1)
        else:
            weight_norm = np.zeros_like(weight_map, dtype=np.float32)
        weight_u16 = (weight_norm * 65535.0).astype(np.uint16)
        dist_u16 = (np.clip(dist_map, 0, 1) * 65535.0).astype(np.uint16)

        classes = list(merged["semantic_classes"])
        argmax = np.argmax(probs, axis=2).astype(np.uint8)
        denom = max(1, len(classes) - 1)
        argmax_vis = (argmax.astype(np.float32) / float(denom) * 255.0).astype(np.uint8)

        def _ci(name):
            return classes.index(name) if name in classes else -1

        ri, bi = _ci("rail"), _ci("ballast")
        rail_ch = np.zeros((h, w), dtype=np.float32)
        if ri >= 0:
            rail_ch += probs[:, :, ri]
        if bi >= 0:
            rail_ch += probs[:, :, bi]
        rail_png = np.clip(rail_ch * 255.0, 0, 255).astype(np.uint8)

        pi, si = _ci("pole"), _ci("signal")
        pole_ch = np.zeros((h, w), dtype=np.float32)
        if pi >= 0:
            pole_ch += probs[:, :, pi]
        if si >= 0:
            pole_ch += probs[:, :, si]
        pole_png = np.clip(pole_ch * 255.0, 0, 255).astype(np.uint8)

        dm = dict(dataset_meta or {})
        dm.setdefault("semantic_classes", classes)
        bev_cfg = bev_cfg if isinstance(bev_cfg, dict) else {}
        pseudo = semantic_probs_to_pseudo_bev(probs, intrinsics, (rvec, tvec), bev_cfg, dm)

        bundle = {
            "semantic_probs": probs,
            "semantic_logits": logits,
            "save_logits": merged.get("save_logits", True),
            "semantic_argmax": argmax_vis,
            "rail_prob_png": rail_png,
            "pole_prob_png": pole_png,
            "edge_map": edge_sem,
            "edge_weight_u16": weight_u16,
            "line_mask": line_mask,
            "lines_2d": lines_full,
            "pseudo_bev": pseudo,
        }
        os.makedirs(frame_bundle_dir, exist_ok=True)
        save_image_feature_bundle(frame_bundle_dir, bundle)

        out_dir = os.path.dirname(sam_output_base)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(sam_output_base + "_edge_map.png", edge_map_sam)
        cv2.imwrite(sam_output_base + "_line_map.png", line_mask)
        cv2.imwrite(sam_output_base + "_edge_fused.png", fused_edge_map)
        cv2.imwrite(sam_output_base + "_edge_dist.png", dist_u16)
        cv2.imwrite(sam_output_base + "_edge_weight.png", weight_u16)
        cv2.imwrite(sam_output_base + "_mask_ids.png", mask_id_map.astype(np.uint16))
        cv2.imwrite(sam_output_base + "_semantic_map.png", mask_id_map.astype(np.uint16))

        with open(sam_output_base + "_lines_2d.txt", "w", encoding="utf-8") as f:
            f.write("# u1 v1 u2 v2 type semantic_support (type: 0=H, 1=V, 2=diagonal)\n")
            for t in lines_full:
                if len(t) >= 6:
                    f.write(f"{t[0]:.2f} {t[1]:.2f} {t[2]:.2f} {t[3]:.2f} {t[4]} {t[5]:.4f}\n")
                else:
                    f.write(f"{t[0]:.2f} {t[1]:.2f} {t[2]:.2f} {t[3]:.2f} {t[4]} 1.0000\n")

        print(f"[Saved] bundle -> {frame_bundle_dir}")
        print(f"[Saved] optimizer inputs -> {sam_output_base}_*")
        return True


# 保持向后兼容性的别名
SAMEdgeExtractor = FeatureExtractor
