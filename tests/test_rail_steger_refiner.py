import os
import sys

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from rail_steger_refiner import refine_rail_lines  # noqa: E402


def test_refine_rail_lines_extracts_synthetic_rails():
    h, w = 360, 640
    image = np.zeros((h, w, 3), dtype=np.uint8)
    rail_prob = np.zeros((h, w), dtype=np.float32)

    left = np.array([[220, 330], [280, 250], [315, 170], [330, 90]], dtype=np.int32)
    right = np.array([[420, 330], [365, 250], [335, 170], [330, 90]], dtype=np.int32)
    cv2.polylines(image, [left.reshape(-1, 1, 2), right.reshape(-1, 1, 2)], False, (230, 230, 230), 3)
    cv2.polylines(rail_prob, [left.reshape(-1, 1, 2), right.reshape(-1, 1, 2)], False, 0.35, 19)

    cfg = {
        "rail_dist_max_ratio": 0.08,
        "rail_weight_dilate_kernel": 7,
        "rail_refinement": {
            "enabled": True,
            "weak_roi_threshold": 0.05,
            "steger_sigma": 1.0,
            "min_response": 0.01,
            "min_curve_length_px": 60,
            "min_mean_rail_prob": 0.04,
            "vp_angle_thresh_deg": 18.0,
            "min_total_length_px": 100,
            "min_quality_score": 0.2,
            "max_output_lines": 4,
        },
    }

    out = refine_rail_lines(image, rail_prob, cfg)

    assert out["rail_centerline_u8"].shape == (h, w)
    assert out["rail_dist"].shape == (h, w)
    assert out["rail_weight"].shape == (h, w)
    assert len(out["rail_centerlines_2d"]) >= 2
    assert out["quality"]["line_count"] >= 2
    assert out["quality"]["quality_score"] > 0.2


def test_lidar_prior_recovers_rails_when_sam_prior_is_weak():
    h, w = 360, 640
    image = np.zeros((h, w, 3), dtype=np.uint8)
    sam_prior = np.zeros((h, w), dtype=np.float32) + 0.01
    lidar_prior = np.zeros((h, w), dtype=np.float32)

    left = np.array([[230, 335], [280, 250], [314, 170], [330, 90]], dtype=np.int32)
    right = np.array([[410, 335], [365, 250], [338, 170], [330, 90]], dtype=np.int32)
    cv2.polylines(image, [left.reshape(-1, 1, 2), right.reshape(-1, 1, 2)], False, (235, 235, 235), 3)
    cv2.polylines(lidar_prior, [left.reshape(-1, 1, 2), right.reshape(-1, 1, 2)], False, 1.0, 31)

    cfg = {
        "rail_dist_max_ratio": 0.08,
        "rail_weight_dilate_kernel": 7,
        "rail_refinement": {
            "enabled": True,
            "use_lidar_bev_prior": True,
            "lidar_prior_weight": 0.75,
            "sam_prior_weight": 0.15,
            "steger_response_weight": 0.10,
            "candidate_roi_dilate_px": 25,
            "steger_sigma": 1.0,
            "min_response": 0.01,
            "min_curve_length_px": 60,
            "min_mean_rail_prob": 0.02,
            "vp_angle_thresh_deg": 20.0,
            "min_total_length_px": 100,
            "min_quality_score": 0.2,
            "max_output_lines": 4,
        },
    }

    out = refine_rail_lines(image, sam_prior, cfg, lidar_prior=lidar_prior)

    assert out["quality"]["lidar_prior_used"] is True
    assert out["quality"]["line_count"] >= 2
    assert len(out["rail_centerlines_2d"]) >= 2
    assert out["rail_weight"].mean() > 0.0


def test_label_track_prior_dominates_weak_sam_and_lidar_priors():
    h, w = 360, 640
    image = np.zeros((h, w, 3), dtype=np.uint8)
    sam_prior = np.zeros((h, w), dtype=np.float32) + 0.01
    lidar_prior = np.zeros((h, w), dtype=np.float32)
    label_prior = np.zeros((h, w), dtype=np.float32)

    left = np.array([[230, 335], [280, 250], [314, 170], [330, 90]], dtype=np.int32)
    right = np.array([[410, 335], [365, 250], [338, 170], [330, 90]], dtype=np.int32)
    cv2.polylines(image, [left.reshape(-1, 1, 2), right.reshape(-1, 1, 2)], False, (235, 235, 235), 3)
    cv2.polylines(label_prior, [left.reshape(-1, 1, 2), right.reshape(-1, 1, 2)], False, 1.0, 31)

    cfg = {
        "rail_dist_max_ratio": 0.08,
        "rail_weight_dilate_kernel": 7,
        "rail_refinement": {
            "enabled": True,
            "use_label_track_prior": True,
            "label_track_prior_weight": 0.80,
            "use_lidar_bev_prior": True,
            "lidar_prior_weight": 0.12,
            "sam_prior_weight": 0.04,
            "steger_response_weight": 0.04,
            "candidate_roi_dilate_px": 25,
            "steger_sigma": 1.0,
            "min_response": 0.01,
            "min_curve_length_px": 60,
            "min_mean_rail_prob": 0.02,
            "vp_angle_thresh_deg": 20.0,
            "min_total_length_px": 100,
            "min_quality_score": 0.2,
            "max_output_lines": 4,
        },
    }

    out = refine_rail_lines(image, sam_prior, cfg, lidar_prior=lidar_prior, label_track_prior=label_prior)

    assert out["quality"]["label_track_prior_used"] is True
    assert out["quality"]["line_count"] >= 2
    assert len(out["rail_centerlines_2d"]) >= 2
    assert out["rail_likelihood"][label_prior > 0].mean() > out["rail_likelihood"][label_prior == 0].mean()


def test_label_track_all_lines_are_output_and_locally_refined_by_steger():
    h, w = 360, 640
    image = np.zeros((h, w, 3), dtype=np.uint8)
    sam_prior = np.zeros((h, w), dtype=np.float32) + 0.01
    label_prior = np.zeros((h, w), dtype=np.float32)

    label_lines = [
        [(228, 335), (278, 250), (312, 170), (328, 90)],
        [(412, 335), (367, 250), (340, 170), (332, 90)],
        [(120, 350), (170, 260), (215, 170), (250, 90)],
    ]
    image_lines = [
        np.array([(x + 6, y) for x, y in line], dtype=np.int32) for line in label_lines
    ]
    for pts in image_lines:
        cv2.polylines(image, [pts.reshape(-1, 1, 2)], False, (235, 235, 235), 3)
    for line in label_lines:
        cv2.polylines(label_prior, [np.array(line, dtype=np.int32).reshape(-1, 1, 2)], False, 1.0, 31)

    cfg = {
        "rail_dist_max_ratio": 0.08,
        "rail_weight_dilate_kernel": 7,
        "rail_refinement": {
            "enabled": True,
            "use_label_track_prior": True,
            "label_track_output_lines": "all",
            "label_track_prior_weight": 0.80,
            "sam_prior_weight": 0.04,
            "steger_response_weight": 0.04,
            "label_track_steger_search_radius_px": 10,
            "label_track_polyline_sample_step_px": 4,
            "steger_sigma": 1.0,
            "min_response": 0.01,
            "min_mean_rail_prob": 0.02,
            "vp_angle_thresh_deg": 35.0,
            "min_total_length_px": 100,
            "min_quality_score": 0.1,
        },
    }

    out = refine_rail_lines(image, sam_prior, cfg, label_track_prior=label_prior, label_track_polylines=label_lines)

    assert out["quality"]["method"] == "label_track_steger_local_all"
    assert out["quality"]["line_count"] == 3
    assert out["quality"]["all_line_count"] == 3
    assert len(out["rail_centerlines_2d"]) == 3
    assert any(abs(out["rail_centerlines_2d"][0][0][0] - label_lines[0][0][0]) > 0 for _ in [0])


def test_missing_lidar_prior_falls_back_to_sam_only():
    h, w = 240, 420
    image = np.zeros((h, w, 3), dtype=np.uint8)
    sam_prior = np.zeros((h, w), dtype=np.float32)
    left = np.array([[150, 220], [180, 150], [205, 80]], dtype=np.int32)
    right = np.array([[280, 220], [245, 150], [215, 80]], dtype=np.int32)
    cv2.polylines(image, [left.reshape(-1, 1, 2), right.reshape(-1, 1, 2)], False, (230, 230, 230), 3)
    cv2.polylines(sam_prior, [left.reshape(-1, 1, 2), right.reshape(-1, 1, 2)], False, 0.35, 17)

    cfg = {
        "rail_dist_max_ratio": 0.08,
        "rail_refinement": {
            "enabled": True,
            "use_lidar_bev_prior": True,
            "weak_roi_threshold": 0.05,
            "steger_sigma": 1.0,
            "min_response": 0.01,
            "min_curve_length_px": 45,
            "min_mean_rail_prob": 0.04,
            "vp_angle_thresh_deg": 22.0,
            "min_total_length_px": 80,
            "min_quality_score": 0.15,
        },
    }

    out = refine_rail_lines(image, sam_prior, cfg, lidar_prior=None)

    assert out["quality"]["lidar_prior_used"] is False
    assert "rail_centerline_u8" in out
    assert out["rail_dist"].shape == (h, w)


def test_gauge_constraint_rejects_implausible_pair():
    h, w = 360, 640
    image = np.zeros((h, w, 3), dtype=np.uint8)
    sam_prior = np.zeros((h, w), dtype=np.float32) + 0.01
    lidar_prior = np.zeros((h, w), dtype=np.float32)

    left = np.array([[170, 330], [205, 230], [230, 100]], dtype=np.int32)
    right = np.array([[500, 330], [465, 230], [430, 100]], dtype=np.int32)
    cv2.polylines(image, [left.reshape(-1, 1, 2), right.reshape(-1, 1, 2)], False, (235, 235, 235), 3)
    cv2.polylines(lidar_prior, [left.reshape(-1, 1, 2), right.reshape(-1, 1, 2)], False, 1.0, 25)

    cfg = {
        "_intrinsics": np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 100.0], [0.0, 0.0, 1.0]]),
        "_rvec": np.zeros(3),
        "_tvec": np.array([0.0, 0.0, 5.0]),
        "_reference_z": 0.0,
        "rail_dist_max_ratio": 0.08,
        "rail_refinement": {
            "enabled": True,
            "use_lidar_bev_prior": True,
            "lidar_prior_weight": 0.75,
            "sam_prior_weight": 0.15,
            "steger_response_weight": 0.10,
            "candidate_roi_dilate_px": 25,
            "steger_sigma": 1.0,
            "min_response": 0.01,
            "min_curve_length_px": 60,
            "min_mean_rail_prob": 0.02,
            "vp_angle_thresh_deg": 30.0,
            "track_gauge_m": 1.435,
            "track_gauge_tolerance_m": 0.45,
            "min_total_length_px": 100,
            "min_quality_score": 0.0,
        },
    }

    out = refine_rail_lines(image, sam_prior, cfg, lidar_prior=lidar_prior)

    assert out["quality"]["gauge_error_m"] is not None
    assert out["quality"]["gauge_error_m"] > 0.45
    assert out["quality"]["enabled"] is False
    assert "gauge_error_high" in out["quality"]["disable_reason"]
