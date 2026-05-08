"""Rail line refinement using SAM rail probability as a weak prior.

The extractor is intentionally self-contained: it does not depend on trained
rail-specific models and keeps the old optimizer outputs compatible.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


def _cfg(config: Dict[str, Any], key: str, default: Any) -> Any:
    ref_cfg = config.get("rail_refinement", {}) if isinstance(config, dict) else {}
    if isinstance(ref_cfg, dict) and key in ref_cfg:
        return ref_cfg[key]
    return config.get(key, default) if isinstance(config, dict) else default


def _enhance_gray(image_bgr: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clip = float(_cfg(config, "clahe_clip_limit", 2.0))
    tile = int(_cfg(config, "clahe_tile_grid", 8))
    tile = max(2, tile)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    enhanced = clahe.apply(gray)
    return enhanced.astype(np.float32) / 255.0


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.nanmin(arr)) if arr.size else 0.0
    hi = float(np.nanmax(arr)) if arr.size else 0.0
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _steger_candidate_map(
    gray01: np.ndarray,
    rail_prob: np.ndarray,
    config: Dict[str, Any],
    lidar_prior: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate Steger line response from Hessian eigen structure.

    We keep the response continuous for weighting, then create a sparse mask
    from subpixel-valid candidates. Both bright and dark ridges are represented
    by the absolute strongest Hessian eigenvalue.
    """
    h, w = gray01.shape[:2]
    sigma = float(_cfg(config, "steger_sigma", 1.4))
    sigma = max(0.5, sigma)
    blurred = cv2.GaussianBlur(gray01, (0, 0), sigmaX=sigma, sigmaY=sigma)

    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
    gxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
    gyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)

    trace = gxx + gyy
    diff = gxx - gyy
    root = np.sqrt(np.maximum(diff * diff + 4.0 * gxy * gxy, 0.0))
    l1 = 0.5 * (trace + root)
    l2 = 0.5 * (trace - root)
    use_l1 = np.abs(l1) >= np.abs(l2)
    lam = np.where(use_l1, l1, l2)

    # Eigenvector for strongest eigenvalue. For [a b; b c], vector can be
    # [b, lambda-a]; fall back to x-axis for near-zero vectors.
    nx = gxy
    ny = lam - gxx
    n_norm = np.sqrt(nx * nx + ny * ny)
    fallback = n_norm < 1e-6
    nx = np.where(fallback, 1.0, nx / np.maximum(n_norm, 1e-6))
    ny = np.where(fallback, 0.0, ny / np.maximum(n_norm, 1e-6))

    denom = gxx * nx * nx + 2.0 * gxy * nx * ny + gyy * ny * ny
    numer = gx * nx + gy * ny
    offset = -numer / (denom + 1e-6)
    subpixel_ok = np.abs(offset) <= float(_cfg(config, "max_subpixel_offset", 0.6))

    sam_prior = np.clip(rail_prob.astype(np.float32), 0.0, 1.0)
    use_lidar_prior = lidar_prior is not None and bool(_cfg(config, "use_lidar_bev_prior", False))
    lidar_prior01 = None
    if use_lidar_prior:
        lidar_prior01 = np.clip(lidar_prior.astype(np.float32), 0.0, 1.0)
        if lidar_prior01.shape[:2] != (h, w):
            lidar_prior01 = cv2.resize(lidar_prior01, (w, h), interpolation=cv2.INTER_LINEAR)

    weak_roi_th = float(_cfg(config, "weak_roi_threshold", 0.08))
    top_ignore = float(_cfg(config, "rail_top_ignore_ratio", 0.05))
    bottom_keep = float(_cfg(config, "rail_bottom_keep_ratio", 0.98))
    roi = np.ones((h, w), dtype=bool)
    if top_ignore > 0:
        roi[: int(h * top_ignore), :] = False
    if 0 < bottom_keep < 1:
        roi[int(h * bottom_keep) :, :] = False

    response = _normalize01(np.abs(lam))
    lidar_w = float(_cfg(config, "lidar_prior_weight", 0.0)) if lidar_prior01 is not None else 0.0
    sam_w = float(_cfg(config, "sam_prior_weight", 1.0))
    resp_w = float(_cfg(config, "steger_response_weight", 0.0))
    wsum = max(1e-6, lidar_w + sam_w + resp_w)
    rail_likelihood = (sam_w * sam_prior + resp_w * response) / wsum
    if lidar_prior01 is not None:
        rail_likelihood = rail_likelihood + (lidar_w * lidar_prior01) / wsum
        prior_th = float(_cfg(config, "lidar_prior_roi_threshold", 1e-4))
        support = lidar_prior01 > prior_th
        dilate_px = int(_cfg(config, "candidate_roi_dilate_px", 45))
        if dilate_px > 1:
            k = max(3, dilate_px | 1)
            support = cv2.dilate(support.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        roi &= support
    else:
        roi &= sam_prior >= weak_roi_th

    rail_likelihood = np.clip(rail_likelihood, 0.0, 1.0)
    weighted_response = response * (0.25 + 0.75 * rail_likelihood)
    min_response = float(_cfg(config, "min_response", 0.03))
    candidate_core = (weighted_response >= min_response) & subpixel_ok & roi

    # Thin the response by non-maximum suppression along the normal direction.
    yy, xx = np.indices((h, w), dtype=np.float32)
    x1 = np.clip(np.rint(xx + nx).astype(np.int32), 0, w - 1)
    y1 = np.clip(np.rint(yy + ny).astype(np.int32), 0, h - 1)
    x2 = np.clip(np.rint(xx - nx).astype(np.int32), 0, w - 1)
    y2 = np.clip(np.rint(yy - ny).astype(np.int32), 0, h - 1)
    nms = (weighted_response >= weighted_response[y1, x1]) & (weighted_response >= weighted_response[y2, x2])
    if lidar_prior01 is not None:
        # A projected BEV prior is already spatially selective; keep strong
        # candidates inside it even when pixel-grid NMS breaks long rails into
        # isolated samples.
        strong_prior = rail_likelihood >= float(_cfg(config, "lidar_prior_keep_threshold", 0.20))
        mask = candidate_core & (nms | strong_prior)
    else:
        strong_prior = sam_prior >= weak_roi_th
        mask = candidate_core & (nms | strong_prior)

    return mask.astype(np.uint8), weighted_response.astype(np.float32), rail_likelihood.astype(np.float32)


def _component_curves(
    candidate_mask: np.ndarray,
    response: np.ndarray,
    rail_prob: np.ndarray,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    dilate_k = int(_cfg(config, "candidate_connect_kernel", 3))
    dilate_k = max(1, dilate_k | 1)
    connected = candidate_mask
    if dilate_k >= 3:
        connected = cv2.dilate(connected, np.ones((dilate_k, dilate_k), np.uint8))

    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    min_len = int(_cfg(config, "min_curve_length_px", 120))
    max_samples = int(_cfg(config, "max_curve_samples", 900))
    curves: List[Dict[str, Any]] = []
    for lab in range(1, nlab):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < max(8, min_len // 4):
            continue
        comp = labels == lab
        ys, xs = np.where(comp & (candidate_mask > 0))
        if xs.size < max(8, min_len // 5):
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        length = _polyline_span(pts)
        if length < min_len:
            continue
        if pts.shape[0] > max_samples:
            take = np.linspace(0, pts.shape[0] - 1, max_samples).astype(np.int32)
            pts = pts[take]
        rr = response[ys, xs]
        rp = rail_prob[ys, xs]
        curves.append(
            {
                "points": pts,
                "length": float(length),
                "mean_response": float(np.mean(rr)) if rr.size else 0.0,
                "mean_rail_prob": float(np.mean(rp)) if rp.size else 0.0,
            }
        )
    return curves


def _polyline_span(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    x_span = float(np.max(points_xy[:, 0]) - np.min(points_xy[:, 0]))
    y_span = float(np.max(points_xy[:, 1]) - np.min(points_xy[:, 1]))
    return math.hypot(x_span, y_span)


def _fit_line(curve: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    pts = curve["points"].astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    v = np.array([float(vx), float(vy)], dtype=np.float32)
    v /= max(float(np.linalg.norm(v)), 1e-6)
    p0 = np.array([float(x0), float(y0)], dtype=np.float32)
    return p0, v


def _line_intersection(p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> Tuple[bool, np.ndarray]:
    a = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]], dtype=np.float32)
    b = p2 - p1
    det = float(np.linalg.det(a))
    if abs(det) < 1e-4:
        return False, np.zeros(2, dtype=np.float32)
    t = np.linalg.solve(a, b)[0]
    return True, p1 + t * v1


def _angle_to_vp(curve: Dict[str, Any], vp: np.ndarray) -> float:
    p0, v = _fit_line(curve)
    to_vp = vp - p0
    n = float(np.linalg.norm(to_vp))
    if n < 1e-6:
        return 180.0
    to_vp /= n
    c = abs(float(np.dot(v, to_vp)))
    c = min(1.0, max(-1.0, c))
    return math.degrees(math.acos(c))


def _estimate_vanishing_point(curves: List[Dict[str, Any]], shape: Tuple[int, int], config: Dict[str, Any]) -> Tuple[np.ndarray | None, float]:
    if len(curves) < 2:
        return None, 0.0
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


def _config_array(config: Dict[str, Any], key: str, shape: Tuple[int, ...]) -> np.ndarray | None:
    if key not in config:
        return None
    try:
        arr = np.asarray(config[key], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.size != int(np.prod(shape)):
        return None
    return arr.reshape(shape)


def _interp_polyline_at_y(poly: Sequence[Point], ys: np.ndarray) -> np.ndarray | None:
    pts = np.array(poly, dtype=np.float32)
    if pts.shape[0] < 2:
        return None
    order = np.argsort(pts[:, 1])
    pts = pts[order]
    unique_y, unique_idx = np.unique(pts[:, 1], return_index=True)
    if unique_y.size < 2:
        return None
    xs = pts[unique_idx, 0]
    if ys[0] < unique_y[0] or ys[-1] > unique_y[-1]:
        return None
    out_x = np.interp(ys, unique_y, xs).astype(np.float32)
    return np.stack([out_x, ys.astype(np.float32)], axis=1)


def _backproject_to_lidar_plane(
    uv: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    reference_z: float,
) -> np.ndarray | None:
    if uv.size == 0:
        return None
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1).astype(np.float64))
    Rt = R.T
    Kinv = np.linalg.inv(K.astype(np.float64))
    rays = (Kinv @ np.column_stack([uv[:, 0], uv[:, 1], np.ones(len(uv))]).T).T
    rt = Rt @ tvec.reshape(3).astype(np.float64)
    dirs_l = (Rt @ rays.T).T
    denom = dirs_l[:, 2]
    valid = np.abs(denom) > 1e-8
    if not np.any(valid):
        return None
    s = np.zeros(len(uv), dtype=np.float64)
    s[valid] = (float(reference_z) + rt[2]) / denom[valid]
    valid &= s > 0
    if np.count_nonzero(valid) < 3:
        return None
    pts = (Rt @ (rays[valid] * s[valid, None] - tvec.reshape(1, 3)).T).T
    return pts


def _gauge_for_pair(poly_a: Sequence[Point], poly_b: Sequence[Point], config: Dict[str, Any]) -> float | None:
    K = _config_array(config, "_intrinsics", (3, 3))
    rvec = _config_array(config, "_rvec", (3,))
    tvec = _config_array(config, "_tvec", (3,))
    if K is None or rvec is None or tvec is None:
        return None
    a = np.array(poly_a, dtype=np.float32)
    b = np.array(poly_b, dtype=np.float32)
    y0 = max(float(np.min(a[:, 1])), float(np.min(b[:, 1])))
    y1 = min(float(np.max(a[:, 1])), float(np.max(b[:, 1])))
    if y1 <= y0 + 4:
        return None
    ys = np.linspace(y0, y1, int(_cfg(config, "gauge_sample_count", 25)), dtype=np.float32)
    uv_a = _interp_polyline_at_y(poly_a, ys)
    uv_b = _interp_polyline_at_y(poly_b, ys)
    if uv_a is None or uv_b is None:
        return None
    reference_z = float(config.get("_reference_z", 0.0))
    pts_a = _backproject_to_lidar_plane(uv_a, K, rvec, tvec, reference_z)
    pts_b = _backproject_to_lidar_plane(uv_b, K, rvec, tvec, reference_z)
    if pts_a is None or pts_b is None:
        return None
    n = min(len(pts_a), len(pts_b))
    if n < 3:
        return None
    dist = np.linalg.norm(pts_a[:n, :2] - pts_b[:n, :2], axis=1)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return None
    return float(np.median(dist))


def _select_main_pair(
    curves: List[Dict[str, Any]],
    vp: np.ndarray | None,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(curves) < 2:
        return curves, {"pair_score": 0.0, "gauge_median_m": None, "gauge_error_m": None}

    polylines = [_smooth_curve(c, config) for c in curves]
    items = [(c, p) for c, p in zip(curves, polylines) if len(p) >= 2]
    if len(items) < 2:
        return [c for c, _ in items], {"pair_score": 0.0, "gauge_median_m": None, "gauge_error_m": None}

    gauge_m = float(_cfg(config, "track_gauge_m", 1.435))
    gauge_tol = float(_cfg(config, "track_gauge_tolerance_m", 0.45))
    best = None
    best_valid = None
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            c1, p1 = items[i]
            c2, p2 = items[j]
            total_len = float(c1["length"] + c2["length"])
            mean_resp = 0.5 * float(c1["mean_response"] + c2["mean_response"])
            mean_prior = 0.5 * float(c1["mean_rail_prob"] + c2["mean_rail_prob"])
            vp_score = 1.0
            if vp is not None:
                angle_th = max(1e-6, float(_cfg(config, "vp_angle_thresh_deg", 10.0)))
                vp_score = max(0.0, 1.0 - (_angle_to_vp(c1, vp) + _angle_to_vp(c2, vp)) / (2.0 * angle_th))
            gauge = _gauge_for_pair(p1, p2, config)
            gauge_err = None if gauge is None else abs(gauge - gauge_m)
            gauge_score = 0.5 if gauge_err is None else max(0.0, 1.0 - gauge_err / max(1e-6, gauge_tol))
            score = (
                min(1.0, total_len / max(1.0, float(_cfg(config, "min_total_length_px", 1200.0)))) * 0.30
                + min(1.0, mean_resp / 0.20) * 0.20
                + min(1.0, mean_prior / max(0.02, float(_cfg(config, "min_mean_rail_prob", 0.08)))) * 0.20
                + vp_score * 0.15
                + gauge_score * 0.15
            )
            info = {
                "pair_score": float(score),
                "gauge_median_m": None if gauge is None else float(gauge),
                "gauge_error_m": None if gauge_err is None else float(gauge_err),
            }
            candidate = (score, [c1, c2], info)
            if best is None or score > best[0]:
                best = candidate
            if gauge_err is None or gauge_err <= gauge_tol:
                if best_valid is None or score > best_valid[0]:
                    best_valid = candidate

    chosen = best_valid or best
    if chosen is None:
        return [c for c, _ in items[:2]], {"pair_score": 0.0, "gauge_median_m": None, "gauge_error_m": None}
    return chosen[1], chosen[2]


def refine_rail_lines(
    image_bgr: np.ndarray,
    rail_prob: np.ndarray,
    config: Dict[str, Any],
    lidar_prior: np.ndarray | None = None,
) -> Dict[str, Any]:
    h, w = rail_prob.shape[:2]
    lidar_prior_used = lidar_prior is not None and bool(_cfg(config, "use_lidar_bev_prior", False))
    method = "lidar_bev_sam_steger_vp_gauge" if lidar_prior_used else "steger_sam_roi"
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
    candidate_mask, response, rail_likelihood = _steger_candidate_map(gray, rail_prob, config, lidar_prior=lidar_prior)
    curves = _component_curves(candidate_mask, response, rail_likelihood, config)
    vp, vp_ratio = _estimate_vanishing_point(curves, (h, w), config)

    angle_th = float(_cfg(config, "vp_angle_thresh_deg", 10.0))
    min_prob = float(_cfg(config, "min_mean_rail_prob", 0.08))
    if vp is not None:
        filtered = [c for c in curves if _angle_to_vp(c, vp) <= angle_th and c["mean_rail_prob"] >= min_prob]
    else:
        filtered = [c for c in curves if c["mean_rail_prob"] >= min_prob]

    pair_info = {"pair_score": 0.0, "gauge_median_m": None, "gauge_error_m": None}
    if bool(_cfg(config, "prefer_main_pair", True)) and len(filtered) >= 2:
        filtered, pair_info = _select_main_pair(filtered, vp, config)
    elif len(filtered) > 2:
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
    weight = (dil > 0).astype(np.float32) * resp_blur * (0.3 + 0.7 * np.clip(rail_likelihood, 0.0, 1.0))
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
    gauge_err = pair_info.get("gauge_error_m")
    if gauge_err is not None and gauge_err > float(_cfg(config, "track_gauge_tolerance_m", 0.45)):
        reasons.append("gauge_error_high")
    if score < min_quality:
        reasons.append("quality_score_low")
    enabled_out = not reasons

    quality = {
        "enabled": bool(enabled_out),
        "method": method,
        "lidar_prior_used": bool(lidar_prior_used),
        "line_count": int(len(polylines)),
        "total_length_px": float(total_length),
        "mean_response": float(mean_resp),
        "mean_rail_prob_on_lines": float(mean_prob),
        "vanishing_point": None if vp is None else [float(vp[0]), float(vp[1])],
        "vp_inlier_ratio": float(vp_ratio),
        "pair_score": float(pair_info.get("pair_score", 0.0) or 0.0),
        "gauge_median_m": pair_info.get("gauge_median_m"),
        "gauge_error_m": pair_info.get("gauge_error_m"),
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
        "rail_likelihood": rail_likelihood.astype(np.float32),
    }
