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