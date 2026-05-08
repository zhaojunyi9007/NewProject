import json
import os
import sys

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from sam_extractor import build_osdar_label_track_prior  # noqa: E402


def test_osdar_label_track_prior_rasterizes_track_polyline(tmp_path):
    label_path = tmp_path / "sequence_labels.json"
    label_doc = {
        "openlabel": {
            "objects": {
                "track-1": {"type": "track"},
                "car-1": {"type": "road_vehicle"},
            },
            "frames": {
                "12": {
                    "objects": {
                        "track-1": {
                            "object_data": {
                                "poly2d": [
                                    {
                                        "name": "rgb_center__poly2d__track",
                                        "coordinate_system": "rgb_center",
                                        "val": [20, 90, 60, 50, 100, 20],
                                    }
                                ]
                            }
                        },
                        "car-1": {
                            "object_data": {
                                "poly2d": [
                                    {
                                        "name": "rgb_center__poly2d__road_vehicle",
                                        "coordinate_system": "rgb_center",
                                        "val": [1, 1, 5, 5],
                                    }
                                ]
                            }
                        },
                    }
                }
            },
        }
    }
    label_path.write_text(json.dumps(label_doc), encoding="utf-8")
    cfg = {
        "rail_refinement": {
            "use_label_track_prior": True,
            "label_track_dilate_px": 9,
            "label_track_blur_sigma_px": 0,
            "label_track_min_valid_ratio": 0.0001,
        }
    }

    prior, stats, polylines = build_osdar_label_track_prior((120, 160, 3), str(label_path), 12, "rgb_center", cfg)

    assert prior is not None
    assert prior.shape == (120, 160)
    assert stats["label_track_prior_used"] is True
    assert stats["label_track_polyline_count"] == 1
    assert stats["label_track_valid_ratio"] > 0.0
    assert len(polylines) == 1
    assert polylines[0][0] == (20, 90)
    assert float(prior.max()) == 1.0


def test_osdar_label_track_prior_missing_json_falls_back():
    cfg = {"rail_refinement": {"use_label_track_prior": True}}

    prior, stats, polylines = build_osdar_label_track_prior((120, 160, 3), "/missing/file.json", 12, "rgb_center", cfg)

    assert prior is None
    assert polylines == []
    assert stats["label_track_prior_used"] is False
    assert stats["label_track_polyline_count"] == 0
