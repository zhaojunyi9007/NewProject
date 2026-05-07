    h, w = shape
    fitted = [_fit_line(c) for c in curves]
    intersections: List[np.ndarray] = []
    for i in range(len(fitted)):
        for j in range(i + 1, len(fitted)):
            ok, pt = _line_intersection(fitted[i][0], fitted[i][1], fitted[j][0], fitted[j][1])
            if not ok:
                continue
            if -2.0 * w <= pt[0] <= 3.0 * w and -1.5 * h <= pt[1] <= 1.2 * h:
                intersections.append(pt)
    if not intersections:
        return None, 0.0

    angle_th = float(_cfg(config, "vp_angle_thresh_deg", 10.0))
    best_vp = None
    best_inliers = -1
    best_score = -1.0
    for vp in intersections:
        angles = [_angle_to_vp(c, vp) for c in curves]
        inliers = sum(a <= angle_th for a in angles)
        score = inliers - 0.002 * max(0.0, float(vp[1]) - h * 0.8)
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_vp = vp
    if best_vp is None:
        return None, 0.0
    return best_vp.astype(np.float32), float(best_inliers) / max(1, len(curves))


def _smooth_curve(curve: Dict[str, Any], config: Dict[str, Any]) -> List[Point]:
    pts = curve["points"].astype(np.float32)
    if pts.shape[0] < 2:
        return []
    # Fit u=f(v), which is stable for rail curves under forward-facing cameras.
    order = np.argsort(pts[:, 1])
    pts = pts[order]
    ys = pts[:, 1]
    xs = pts[:, 0]
    degree = int(_cfg(config, "fit_poly_degree", 2))
    degree = max(1, min(3, degree))
    if len(np.unique(np.round(ys).astype(np.int32))) <= degree + 1:
        return [(int(round(x)), int(round(y))) for x, y in pts]
    try:
        coeff = np.polyfit(ys, xs, degree)
    except np.linalg.LinAlgError:
        return [(int(round(x)), int(round(y))) for x, y in pts]
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    step = max(1.0, float(_cfg(config, "polyline_sample_step_px", 2.0)))
    y_new = np.arange(y0, y1 + 0.5 * step, step, dtype=np.float32)
    x_new = np.polyval(coeff, y_new)
    h_pad = int(_cfg(config, "polyline_clip_pad_px", 4))
    out: List[Point] = []
    for x, y in zip(x_new, y_new):
        if -h_pad <= x and -h_pad <= y:
            out.append((int(round(x)), int(round(y))))
    return out


def _rasterize_polylines(polylines: Sequence[Sequence[Point]], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    for poly in polylines:
        pts = np.array([(u, v) for u, v in poly if 0 <= u < w and 0 <= v < h], dtype=np.int32)
        if pts.shape[0] >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, 255, 1, cv2.LINE_AA)
    return out


def _distance_map(centerline_u8: np.ndarray, max_ratio: float) -> np.ndarray:
    h, w = centerline_u8.shape[:2]
    max_dist = max(1.0, max(h, w) * float(max_ratio))
    inv = (centerline_u8 == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)
    return np.clip(dist, 0.0, max_dist) / max_dist


def refine_rail_lines(image_bgr: np.ndarray, rail_prob: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    h, w = rail_prob.shape[:2]
    method = "steger_sam_roi"
    enabled = bool(_cfg(config, "enabled", True))
    if not enabled:
        quality = {
            "enabled": False,
            "method": method,
            "line_count": 0,
            "total_length_px": 0.0,
            "quality_score": 0.0,
            "disable_reason": "rail_refinement_disabled",
        }
        return {
            "rail_centerline_u8": np.zeros((h, w), dtype=np.uint8),
            "rail_centerlines_2d": [],
            "rail_dist": np.ones((h, w), dtype=np.float32),
            "rail_weight": np.zeros((h, w), dtype=np.float32),
            "quality": quality,
        }

    gray = _enhance_gray(image_bgr, config)
    candidate_mask, response = _steger_candidate_map(gray, rail_prob, config)
    curves = _component_curves(candidate_mask, response, rail_prob, config)
    vp, vp_ratio = _estimate_vanishing_point(curves, (h, w), config)

    angle_th = float(_cfg(config, "vp_angle_thresh_deg", 10.0))
    min_prob = float(_cfg(config, "min_mean_rail_prob", 0.08))
    if vp is not None:
        filtered = [c for c in curves if _angle_to_vp(c, vp) <= angle_th and c["mean_rail_prob"] >= min_prob]
    else:
        filtered = [c for c in curves if c["mean_rail_prob"] >= min_prob]

    if bool(_cfg(config, "prefer_main_pair", True)) and len(filtered) > 2:
        filtered.sort(key=lambda c: (c["length"] * (0.4 + c["mean_response"]) * (0.3 + c["mean_rail_prob"])), reverse=True)
        filtered = filtered[: max(2, int(_cfg(config, "max_output_lines", 2)))]

    polylines = [_smooth_curve(c, config) for c in filtered]
    polylines = [p for p in polylines if len(p) >= 2]
    centerline = _rasterize_polylines(polylines, (h, w))
    dist = _distance_map(centerline, float(config.get("rail_dist_max_ratio", 0.08)))

    dil_k = int(config.get("rail_weight_dilate_kernel", _cfg(config, "rail_weight_dilate_kernel", 9)))
    dil_k = max(3, dil_k | 1)
    dil = cv2.dilate(centerline, np.ones((dil_k, dil_k), np.uint8))
    resp_blur = cv2.GaussianBlur(response, (0, 0), sigmaX=1.0)
    weight = (dil > 0).astype(np.float32) * resp_blur * (0.3 + 0.7 * np.clip(rail_prob, 0.0, 1.0))
    weight = np.clip(weight, 0.0, 1.0)

    total_length = float(sum(_polyline_span(np.array(poly, dtype=np.float32)) for poly in polylines if len(poly) >= 2))
    mean_resp = float(np.mean([c["mean_response"] for c in filtered])) if filtered else 0.0
    mean_prob = float(np.mean([c["mean_rail_prob"] for c in filtered])) if filtered else 0.0
    rail_dist_valid_ratio = float(np.mean(dist < 0.1)) if dist.size else 0.0
    rail_weight_valid_ratio = float(np.mean(weight > 1e-4)) if weight.size else 0.0

    min_lines = int(_cfg(config, "min_line_count", 2))
    min_length = float(_cfg(config, "min_total_length_px", 1200.0))
    min_quality = float(_cfg(config, "min_quality_score", 0.45))
    score = 0.0
    score += min(1.0, len(polylines) / max(1, min_lines)) * 0.25
    score += min(1.0, total_length / max(1.0, min_length)) * 0.25
    score += min(1.0, vp_ratio / 0.4) * 0.20
    score += min(1.0, mean_resp / 0.20) * 0.15
    score += min(1.0, mean_prob / max(0.02, min_prob)) * 0.15
    score = float(np.clip(score, 0.0, 1.0))

    reasons = []
    if len(polylines) < min_lines:
        reasons.append("line_count_low")
    if total_length < min_length:
        reasons.append("total_length_low")
    if score < min_quality:
        reasons.append("quality_score_low")
    enabled_out = not reasons

    quality = {
        "enabled": bool(enabled_out),
        "method": method,
        "line_count": int(len(polylines)),
        "total_length_px": float(total_length),
        "mean_response": float(mean_resp),
        "mean_rail_prob_on_lines": float(mean_prob),
        "vanishing_point": None if vp is None else [float(vp[0]), float(vp[1])],
        "vp_inlier_ratio": float(vp_ratio),
        "rail_dist_valid_ratio": float(rail_dist_valid_ratio),
        "rail_weight_valid_ratio": float(rail_weight_valid_ratio),
        "quality_score": float(score),
        "disable_reason": ",".join(reasons),
    }
    return {
        "rail_centerline_u8": centerline,
        "rail_centerlines_2d": polylines,
        "rail_dist": dist.astype(np.float32),
        "rail_weight": weight.astype(np.float32),
        "quality": quality,
        "candidate_mask": (candidate_mask * 255).astype(np.uint8),
        "response": response.astype(np.float32),
    }