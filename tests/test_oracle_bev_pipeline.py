import os
import sys

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from lidar_bev_rail_points import export_lidar_bev_rail_points  # noqa: E402
from semantic_to_bev import semantic_probs_to_pseudo_bev  # noqa: E402


def test_label_track_prior_is_written_to_pseudo_bev():
    h, w = 20, 20
    probs = np.zeros((h, w, 2), dtype=np.float32)
    label_prior = np.zeros((h, w), dtype=np.float32)
    label_prior[:5, :5] = 1.0
    K = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]])
    rvec = np.zeros(3)
    tvec = np.array([0.0, 0.0, 10.0])
    bev_cfg = {"x_range": [0.0, 2.0], "y_range": [0.0, 2.0], "resolution": 1.0}
    meta = {"semantic_classes": ["rail", "ballast"], "reference_z": 0.0}

    out = semantic_probs_to_pseudo_bev(
        probs,
        K,
        (rvec, tvec),
        bev_cfg,
        meta,
        extra_rail_priors={"label_track": label_prior},
    )

    assert "rail_from_label_track" in out
    assert out["rail_from_label_track"].max() > 0.0
    assert np.all(out["rail"] >= out["rail_from_label_track"])


def test_lidar_bev_rail_probability_exports_points(tmp_path):
    npz = tmp_path / "bev_maps.npz"
    out = tmp_path / "rail_points.txt"
    rail = np.zeros((4, 5), dtype=np.float32)
    rail[1, 2] = 0.2
    rail[3, 4] = 0.9
    np.savez_compressed(npz, rail_probability=rail, bev_xmin=1.0, bev_ymin=-2.0, bev_resolution=0.5)

    n = export_lidar_bev_rail_points(str(npz), str(out), min_prob=0.15, stride_cells=1, max_points=10, reference_z=0.3)

    lines = [ln for ln in out.read_text().splitlines() if ln and not ln.startswith("#")]
    assert n == 2
    assert len(lines) == 2
    assert all(len(ln.split()) == 4 for ln in lines)
