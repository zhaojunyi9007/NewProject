import json
import os

import numpy as np

from pipeline.datasets import get_adapter
from pipeline.datasets.osdar23 import load_osdar23_intrinsics


def _write_calibration(path, camera="rgb_highres_center", cy=1486.0):
    path.write_text(
        "\n".join(
            [
                "CAMERA",
                f"data_folder: {camera}",
                "camera_matrix: [7267.0, 0, 2056.0;",
                f" 0, 7267.0, {cy};",
                " 0, 0, 1]",
                "homogeneous transform:",
                "[1, 0, 0, 0;",
                " 0, 1, 0, 0;",
                " 0, 0, 1, 0;",
                " 0, 0, 0, 1]",
            ]
        ),
        encoding="utf-8",
    )


def _write_openlabel(path, camera="rgb_highres_center", cy=1232.0):
    data = {
        "openlabel": {
            "streams": {
                camera: {
                    "stream_properties": {
                        "intrinsics_pinhole": {
                            "camera_matrix": [
                                7267.0,
                                0.0,
                                2056.0,
                                0.0,
                                0.0,
                                7267.0,
                                cy,
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                                0.0,
                            ]
                        }
                    }
                }
            }
        }
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_osdar23_intrinsics_prefers_openlabel_camera_matrix(tmp_path):
    calib = tmp_path / "calibration.txt"
    label = tmp_path / "labels.json"
    _write_calibration(calib, cy=1486.0)
    _write_openlabel(label, cy=1232.0)

    K, R_rect, P_rect = load_osdar23_intrinsics(str(calib), "rgb_highres_center", label_json=str(label))

    assert np.allclose(R_rect, np.eye(3))
    assert P_rect.shape == (3, 4)
    assert K[0, 0] == 7267.0
    assert K[0, 2] == 2056.0
    assert K[1, 2] == 1232.0


def test_osdar23_intrinsics_falls_back_to_calibration_txt_without_openlabel(tmp_path):
    calib = tmp_path / "calibration.txt"
    _write_calibration(calib, cy=1486.0)

    K, _, _ = load_osdar23_intrinsics(str(calib), "rgb_highres_center", label_json=str(tmp_path / "missing.json"))

    assert K[1, 2] == 1486.0


def test_osdar23_adapter_resolves_sequence_openlabel_intrinsics(tmp_path):
    root = tmp_path / "1_calibration_1.1"
    root.mkdir()
    calib = root / "calibration.txt"
    label = root / "1_calibration_1.1_labels.json"
    _write_calibration(calib, cy=1486.0)
    _write_openlabel(label, cy=1232.0)

    cfg = {
        "data": {
            "dataset_format": "osdar23",
            "osdar_sequence_root": str(root),
            "calib_file": str(calib),
            "image_sensor": "rgb_highres_center",
        }
    }

    K, _, _ = get_adapter(cfg).load_intrinsics()

    assert K[1, 2] == 1232.0
