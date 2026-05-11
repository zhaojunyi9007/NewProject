import os
import struct
import sys

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from lidar_bev_rail_points import export_lidar_bev_rail_points  # noqa: E402
from bev_export_image_bin import export_image_rail_bin  # noqa: E402
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


def test_bev_export_prefers_sparse_label_track_prior(tmp_path):
    npz = tmp_path / "pseudo_bev.npz"
    out = tmp_path / "image_rail_bev.bin"
    rail = np.ones((8, 8), dtype=np.float32) * 0.6
    label_track = np.zeros((8, 8), dtype=np.float32)
    label_track[2:4, 3:5] = 1.0
    np.savez_compressed(npz, rail=rail, rail_from_label_track=label_track)

    ok = export_image_rail_bin(
        str(npz),
        str(out),
        {
            "x_range": [0.0, 8.0],
            "y_range": [0.0, 8.0],
            "resolution": 1.0,
            "image_rail_source": "label_track",
            "image_rail_threshold": 0.2,
            "image_rail_dilate_cells": 0,
            "image_rail_blur_sigma_cells": 0.0,
        },
    )

    assert ok
    raw = out.read_bytes()
    assert raw[:8] == b"EDGEBEV1"
    nx, ny, nch = struct.unpack("iii", raw[8:20])
    assert (nx, ny, nch) == (8, 8, 1)
    arr = np.frombuffer(raw[36:], dtype=np.float32).reshape(ny, nx)
    assert int((arr > 0).sum()) == 4
    assert arr[2:4, 3:5].min() > 0.9



def test_refined_lidar_rail_bev_uses_oracle_overlap_to_reduce_noise(tmp_path):
    from lidar_bev_rail_points import refine_lidar_rail_probability

    rail = np.ones((40, 40), dtype=np.float32) * 0.2
    rail[:, 8:10] = 0.9
    rail[:, 18:20] = 0.85
    oracle = np.zeros((40, 40), dtype=np.float32)
    oracle[:, 8:10] = 1.0
    oracle[:, 18:20] = 1.0
    debug = {}

    refined = refine_lidar_rail_probability(
        rail,
        min_prob=0.15,
        oracle_rail=oracle,
        oracle_overlap_dilate_cells=1,
        min_component_cells=10,
        debug_out=debug,
    )

    assert refined.shape == rail.shape
    assert debug["lidar_rail_raw_nonzero_ratio"] > 0.9
    assert debug["lidar_rail_refined_nonzero_ratio"] < 0.2
    assert refined[:, 8:10].max() > 0.5
    assert refined[:, 18:20].max() > 0.5


def test_oracle_rail_hard_gate_marks_low_visibility_invalid():
    from pipeline.stages.calib_stage import _apply_final_rail_hard_gate

    breakdown = {"rail_visible_count": 18.0, "rail_visible_ratio": 0.034}
    cfg = {
        "oracle_rail_hard_gate": True,
        "min_final_rail_visible_count": 50,
        "min_final_rail_visible_ratio": 0.08,
        "reject_pose_on_rail_gate_fail": True,
    }

    gated = _apply_final_rail_hard_gate(breakdown, cfg, oracle_rail=True)

    assert gated["rail_gate_failed"] == 1.0
    assert gated["final_pose_valid"] == 0.0
    assert "rail_visible" in gated["invalid_reason"]



def test_export_lidar_bev_rail_points_writes_refined_bev_bin(tmp_path):
    npz = tmp_path / "bev_maps.npz"
    out = tmp_path / "rail_points.txt"
    refined_bin = tmp_path / "rail_refined.bin"
    rail = np.zeros((3, 4), dtype=np.float32)
    rail[1, 1:3] = 0.8
    np.savez_compressed(npz, rail_probability=rail, bev_xmin=2.0, bev_ymin=-1.0, bev_resolution=0.25)

    n = export_lidar_bev_rail_points(
        str(npz),
        str(out),
        min_prob=0.15,
        stride_cells=1,
        max_points=10,
        refined_bin_path=str(refined_bin),
    )

    assert n == 2
    raw = refined_bin.read_bytes()
    assert raw[:8] == b"EDGEBEV1"
    nx, ny, nch = struct.unpack("iii", raw[8:20])
    assert (nx, ny, nch) == (4, 3, 1)
    arr = np.frombuffer(raw[36:], dtype=np.float32).reshape(ny, nx)
    assert arr[1, 1:3].min() > 0.7


def test_rail_refinement_mismatch_is_reported_when_oracle_clears_all(tmp_path):
    from lidar_bev_rail_points import export_lidar_bev_rail_points

    npz = tmp_path / "bev_maps.npz"
    oracle_npz = tmp_path / "pseudo_bev.npz"
    out = tmp_path / "rail_points.txt"
    dbg = tmp_path / "rail_debug.json"
    rail = np.zeros((20, 20), dtype=np.float32)
    rail[:, 2:4] = 0.9
    oracle = np.zeros((20, 20), dtype=np.float32)
    oracle[:, 15:17] = 1.0
    np.savez_compressed(npz, rail_probability=rail, bev_xmin=0.0, bev_ymin=0.0, bev_resolution=1.0)
    np.savez_compressed(oracle_npz, rail_from_label_track=oracle)

    export_lidar_bev_rail_points(
        str(npz),
        str(out),
        min_prob=0.15,
        stride_cells=1,
        oracle_npz_path=str(oracle_npz),
        oracle_overlap_dilate_cells=0,
        min_component_cells=5,
        debug_path=str(dbg),
    )

    obj = __import__("json").loads(dbg.read_text())
    assert obj["rail_refinement_valid"] is False
    assert obj["rail_refinement_mismatch"] is True
    assert obj["lidar_rail_refine_fallback_used"] is True


def test_bev_stage_prefers_nonempty_refined_lidar_rail_bin(tmp_path):
    from pipeline.stages.bev_stage import _select_lidar_bev_input

    raw = tmp_path / "0000000012_bev_channels.bin"
    refined = tmp_path / "0000000012_rail_bev_refined.bin"
    raw.write_bytes(b"raw")
    refined.write_bytes(b"EDGEBEV1" + struct.pack("iii", 1, 1, 1) + struct.pack("ffff", 0.0, 0.0, 1.0, 0.0) + np.array([1.0], dtype=np.float32).tobytes())

    selected, source, ratio = _select_lidar_bev_input(str(raw), str(refined), {"use_refined_lidar_rail": True})

    assert selected == str(refined)
    assert source == "refined"
    assert ratio == 1.0


def test_overlay_json_rail_centerlines_parses_poly_id_u_v(tmp_path):
    from tools.visualize import overlay_json_rail_centerlines

    frame = tmp_path / "frame"
    frame.mkdir()
    (frame / "rail_centerlines_2d.txt").write_text("# poly_id u v\n0 10 20\n0 20 20\n1 30 40\n1 40 40\n", encoding="utf-8")
    img = np.zeros((60, 60, 3), dtype=np.uint8)

    out = overlay_json_rail_centerlines(img, str(frame))

    # Cyan line should be near actual u coordinates, not at x=0/poly_id.
    assert out[20, 10].sum() > 0 or out[20, 20].sum() > 0
    assert out[:, 0].sum() == 0
