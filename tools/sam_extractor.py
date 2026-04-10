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

# Keep 2D line class_id aligned with cpp/include/common.h: SemanticIdRailway.
SEM_UNKNOWN = 0
SEM_RAIL_LIKE = 1
SEM_BALLAST_GROUND = 2
SEM_VERTICAL_STRUCTURE = 3
SEM_PLATFORM_OR_BUILDING = 4
SEM_VEHICLE_LIKE = 5
SEM_VEGETATION_LIKE = 6


def _semantic_class_indices(semantic_classes, names):
    out = []
    for n in names:
        if n in semantic_classes:
            out.append(semantic_classes.index(n))
    return out


def _line_class_to_railway_semantic_id(mean_probs, semantic_classes):
    """Map image semantic channel space to SemanticIdRailway."""
    if mean_probs is None or len(mean_probs) == 0:
        return SEM_UNKNOWN, 0.0

    def group_prob(names):
        idxs = _semantic_class_indices(semantic_classes, names)
        if not idxs:
            return 0.0
        return float(sum(float(mean_probs[i]) for i in idxs))

    group_scores = {
        SEM_RAIL_LIKE: group_prob(["rail"]),
        SEM_BALLAST_GROUND: group_prob(["ballast", "road"]),
        SEM_VERTICAL_STRUCTURE: group_prob(["pole", "signal"]),
        SEM_PLATFORM_OR_BUILDING: group_prob(["platform", "building"]),
        SEM_VEHICLE_LIKE: group_prob(["vehicle"]),
        SEM_VEGETATION_LIKE: group_prob(["vegetation"]),
    }
    best_id = max(group_scores, key=group_scores.get)
    best_score = float(group_scores[best_id])
    if best_score <= 1e-6:
        return SEM_UNKNOWN, 0.0
    return int(best_id), best_score


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


def _build_rail_probability_maps(semantic_probs, config):
    """
    Build rail probability maps from semantic_probs.
    Returns:
      rail_prob: float32 [H,W] in [0,1]
    """
    semantic_classes = list(config.get("semantic_classes", []))
    rail_names = list(config.get("rail_class_names", ["rail", "ballast"]))
    idxs = _semantic_class_indices(semantic_classes, rail_names)
    h, w = semantic_probs.shape[:2]
    rail_prob = np.zeros((h, w), dtype=np.float32)
    for i in idxs:
        rail_prob += semantic_probs[:, :, i].astype(np.float32)
    return np.clip(rail_prob, 0.0, 1.0)


def _build_rail_region_from_masks(rail_prob, config):
    """
    Build a binary rail region mask from rail_prob using seed/support thresholds + morphology + CC filtering.
    Returns uint8 mask in {0,255}.
    """
    h, w = rail_prob.shape[:2]
    seed_th = float(config.get("rail_seed_threshold", 0.42))
    sup_th = float(config.get("rail_support_threshold", 0.22))
    min_area = int(config.get("rail_mask_min_area", 400))
    k_close = int(config.get("rail_close_kernel", 9))
    k_open = int(config.get("rail_open_kernel", 5))
    top_ignore = float(config.get("rail_top_ignore_ratio", 0.18))
    bottom_keep = float(config.get("rail_bottom_keep_ratio", 0.75))
    min_h_ratio = float(config.get("rail_component_min_height_ratio", 0.05))

    seed = (rail_prob >= seed_th).astype(np.uint8) * 255
    support = (rail_prob >= sup_th).astype(np.uint8) * 255

    # Focus vertical ROI: ignore top & optionally ignore very bottom outside keep region.
    y_top = int(h * top_ignore)
    y_bottom_keep = int(h * bottom_keep)
    if y_top > 0:
        seed[0:y_top, :] = 0
        support[0:y_top, :] = 0
    if y_bottom_keep > 0 and y_bottom_keep < h:
        seed[y_bottom_keep:h, :] = 0
        support[y_bottom_keep:h, :] = 0

    # Region grow: keep support connected to seeds.
    grow = np.zeros((h + 2, w + 2), np.uint8)
    seed_ff = seed.copy()
    sup_ff = support.copy()
    cv2.floodFill(sup_ff, grow, (0, 0), 0)  # ensure background stable
    # Use morphology to roughly connect seed points before CC.
    if k_close >= 3:
        kk = k_close | 1
        seed_ff = cv2.morphologyEx(seed_ff, cv2.MORPH_CLOSE, np.ones((kk, kk), np.uint8))
    # Connected components on support; keep components that intersect seed.
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats((sup_ff > 0).astype(np.uint8), connectivity=8)
    out = np.zeros((h, w), np.uint8)
    min_h = max(1, int(h * min_h_ratio))
    for l in range(1, nlab):
        area = int(stats[l, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        y = int(stats[l, cv2.CC_STAT_TOP])
        hh = int(stats[l, cv2.CC_STAT_HEIGHT])
        if hh < min_h:
            continue
        comp = (labels == l)
        if np.any(comp & (seed_ff > 0)):
            out[comp] = 255

    if k_open >= 3:
        kk = k_open | 1
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((kk, kk), np.uint8))
    if k_close >= 3:
        kk = k_close | 1
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((kk, kk), np.uint8))
    return out


def _skeletonize_binary_mask(mask_u8):
    """Zhang-Suen thinning for uint8 mask {0,255}. Returns uint8 skeleton {0,255}."""
    img = (mask_u8 > 0).astype(np.uint8)
    h, w = img.shape
    if h < 3 or w < 3:
        return (img * 255).astype(np.uint8)

    def neighbors(x, y):
        p2 = img[x - 1, y]
        p3 = img[x - 1, y + 1]
        p4 = img[x, y + 1]
        p5 = img[x + 1, y + 1]
        p6 = img[x + 1, y]
        p7 = img[x + 1, y - 1]
        p8 = img[x, y - 1]
        p9 = img[x - 1, y - 1]
        return (p2, p3, p4, p5, p6, p7, p8, p9)

    def transitions(ps):
        n = 0
        for i in range(8):
            if ps[i] == 0 and ps[(i + 1) % 8] == 1:
                n += 1
        return n

    changed = True
    while changed:
        changed = False
        to_del = []
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if img[x, y] != 1:
                    continue
                ps = neighbors(x, y)
                s = sum(ps)
                if s < 2 or s > 6:
                    continue
                if transitions(ps) != 1:
                    continue
                if ps[0] * ps[2] * ps[4] != 0:
                    continue
                if ps[2] * ps[4] * ps[6] != 0:
                    continue
                to_del.append((x, y))
        if to_del:
            for x, y in to_del:
                img[x, y] = 0
            changed = True

        to_del = []
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if img[x, y] != 1:
                    continue
                ps = neighbors(x, y)
                s = sum(ps)
                if s < 2 or s > 6:
                    continue
                if transitions(ps) != 1:
                    continue
                if ps[0] * ps[2] * ps[6] != 0:
                    continue
                if ps[0] * ps[4] * ps[6] != 0:
                    continue
                to_del.append((x, y))
        if to_del:
            for x, y in to_del:
                img[x, y] = 0
            changed = True

    return (img * 255).astype(np.uint8)


def _extract_centerline_polylines(skel_u8, config):
    """
    Extract rough polylines from skeleton pixels by connected components.
    Returns list of list[(u,v)].
    """
    min_len = int(config.get("rail_skeleton_min_length_px", 40))
    sk = (skel_u8 > 0).astype(np.uint8)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(sk, connectivity=8)
    polys = []
    for l in range(1, nlab):
        area = int(stats[l, cv2.CC_STAT_AREA])
        if area < min_len:
            continue
        ys, xs = np.where(labels == l)
        if xs.size < min_len:
            continue
        # Sort by image row then col for a stable polyline order.
        order = np.lexsort((xs, ys))
        pts = [(int(xs[i]), int(ys[i])) for i in order]
        polys.append(pts)
    return polys


def _build_distance_map_from_centerline(centerline_u8, config):
    """
    Build normalized distance map to centerline.
    Returns float32 [H,W] in [0,1].
    """
    h, w = centerline_u8.shape[:2]
    max_dim = max(h, w)
    max_ratio = float(config.get("rail_dist_max_ratio", 0.08))
    max_dist = max(1.0, max_dim * max_ratio)
    inv = (centerline_u8 == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)
    dist = np.clip(dist, 0.0, float(max_dist)) / float(max_dist)
    return dist


def save_image_feature_bundle(output_dir, bundle, prefix=""):
    """
    保存图像特征包到 output_dir（prefix 用于扁平文件名前缀，可为空）。
    bundle: dict，含 semantic_probs, semantic_logits, edge_map, edge_weight_u16,
      rail_region/centerline/dist/weight/centerlines_2d, rail_png, pole_png, semantic_argmax, pseudo_bev_path 等。
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
    if "rail_region_u8" in bundle:
        cv2.imwrite(os.path.join(output_dir, "rail_region.png"), bundle["rail_region_u8"])
    if "rail_centerline_u8" in bundle:
        cv2.imwrite(os.path.join(output_dir, "rail_centerline.png"), bundle["rail_centerline_u8"])
    if "rail_dist_u16" in bundle:
        cv2.imwrite(os.path.join(output_dir, "rail_dist.png"), bundle["rail_dist_u16"])
    if "rail_weight_u16" in bundle:
        cv2.imwrite(os.path.join(output_dir, "rail_weight.png"), bundle["rail_weight_u16"])
    if "rail_centerlines_2d" in bundle:
        with open(os.path.join(output_dir, "rail_centerlines_2d.txt"), "w", encoding="utf-8") as f:
            f.write("# rail centerlines 2d polylines: poly_id u v\n")
            for pid, poly in enumerate(bundle["rail_centerlines_2d"]):
                for (u, v) in poly:
                    f.write(f"{pid} {u} {v}\n")
                f.write("\n")
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

    def build_edge_attraction_field(self, image):
        """
        仅使用 SAM 边缘生成边缘吸引场与权重图（Phase 8：删除 LSD 线特征链路）。
        """
        edge_map, weight_map, mask_id_map = self.extract_edges(image)
        fused_edge_map = edge_map.copy()
        h_img, w_img = fused_edge_map.shape
        # 强制抹黑顶部 20% (纯天空和树顶边缘)
        fused_edge_map[0:int(h_img * float(self.heuristics["fused_top_black_ratio"])), :] = 0
        # 强制抹黑底部 20% (本车引擎盖、近处地面的车道线)
        fused_edge_map[int(h_img * float(self.heuristics["fused_bottom_black_ratio"])):h_img, :] = 0

        dist_map = self.get_distance_transform(fused_edge_map)
        dist_map = cv2.GaussianBlur(dist_map, (5, 5), 0)
        weight_map = cv2.GaussianBlur(weight_map, (5, 5), 0)
        return dist_map, weight_map, edge_map, np.zeros_like(edge_map, dtype=np.uint8), fused_edge_map, [], mask_id_map

    def process_image(self, image_path, output_dir, output_prefix=None):
        """
        处理单张图像，生成所有必要的特征文件
        生成文件：
        - xxx_edge_map.png: SAM边缘图
        - xxx_edge_fused.png: 融合边缘图
        - xxx_edge_dist.png: 边缘吸引场 (16-bit PNG)
        - xxx_edge_weight.png: 边缘权重图 (16-bit PNG)
        - xxx_mask_ids.png: SAM Mask ID图 (16-bit PNG)
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
        dist_map, weight_map, edge_map, _line_mask, fused_edge_map, _lines_2d, mask_id_map = (
            self.build_edge_attraction_field(image)
        )
        cv2.imwrite(output_base + "_edge_map.png", edge_map)
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

        print(f"[Saved] {output_base}_edge_map.png")
        print(f"[Saved] {output_base}_edge_fused.png")
        print(f"[Saved] {output_base}_edge_dist.png")
        print(f"[Saved] {output_base}_edge_weight.png")
        print(f"[Saved] {output_base}_mask_ids.png")

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

        h, w = image.shape[:2]
        fused_edge_map = cv2.bitwise_or(edge_sem, edge_map_sam)
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

        # Phase 1 (sam_2d): rail region + centerline + dist + weight
        rail_prob = _build_rail_probability_maps(probs, merged)
        rail_region = _build_rail_region_from_masks(rail_prob, merged)
        rail_centerline = _skeletonize_binary_mask(rail_region)
        rail_centerlines_2d = _extract_centerline_polylines(rail_centerline, merged)
        rail_dist = _build_distance_map_from_centerline(rail_centerline, merged)
        # rail_weight: dilated centerline weighted by rail_prob
        dil_k = int(merged.get("rail_weight_dilate_kernel", 9))
        dil_k = max(3, dil_k | 1)
        rail_cl_dil = cv2.dilate(rail_centerline, np.ones((dil_k, dil_k), np.uint8))
        rail_weight = np.clip((rail_cl_dil > 0).astype(np.float32) * rail_prob, 0.0, 1.0)

        rail_dist_u16 = (np.clip(rail_dist, 0, 1) * 65535.0).astype(np.uint16)
        rail_weight_u16 = (np.clip(rail_weight, 0, 1) * 65535.0).astype(np.uint16)

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
            "rail_region_u8": rail_region,
            "rail_centerline_u8": rail_centerline,
            "rail_dist_u16": rail_dist_u16,
            "rail_weight_u16": rail_weight_u16,
            "rail_centerlines_2d": rail_centerlines_2d,
            "pseudo_bev": pseudo,
        }
        os.makedirs(frame_bundle_dir, exist_ok=True)
        save_image_feature_bundle(frame_bundle_dir, bundle)

        out_dir = os.path.dirname(sam_output_base)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(sam_output_base + "_edge_map.png", edge_map_sam)
        cv2.imwrite(sam_output_base + "_edge_fused.png", fused_edge_map)
        cv2.imwrite(sam_output_base + "_edge_dist.png", dist_u16)
        cv2.imwrite(sam_output_base + "_edge_weight.png", weight_u16)
        cv2.imwrite(sam_output_base + "_mask_ids.png", mask_id_map.astype(np.uint16))
        cv2.imwrite(sam_output_base + "_semantic_map.png", argmax_vis)
        cv2.imwrite(sam_output_base + "_rail_region.png", rail_region)
        cv2.imwrite(sam_output_base + "_rail_centerline.png", rail_centerline)
        cv2.imwrite(sam_output_base + "_rail_dist.png", rail_dist_u16)
        cv2.imwrite(sam_output_base + "_rail_weight.png", rail_weight_u16)
        with open(sam_output_base + "_rail_centerlines_2d.txt", "w", encoding="utf-8") as f:
            f.write("# rail centerlines 2d polylines: poly_id u v\n")
            for pid, poly in enumerate(rail_centerlines_2d):
                for (u, v) in poly:
                    f.write(f"{pid} {u} {v}\n")
                f.write("\n")

        print(f"[Saved] bundle -> {frame_bundle_dir}")
        print(f"[Saved] optimizer inputs -> {sam_output_base}_*")
        return True


# 保持向后兼容性的别名
SAMEdgeExtractor = FeatureExtractor
