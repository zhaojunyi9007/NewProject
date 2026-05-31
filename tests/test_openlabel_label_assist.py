import json
import os
import pathlib
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(ROOT, "tools")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from openlabel_label_assist import export_label_assist  # noqa: E402
from pipeline.datasets.osdar23 import load_osdar23_openlabel_extrinsic  # noqa: E402


def _write_openlabel(path):
    data = {
        "openlabel": {
            "objects": {
                "veh1": {"type": "road_vehicle", "name": "car"},
                "ped1": {"type": "person", "name": "person"},
                "trk1": {"type": "track", "name": "track"},
                "pole1": {"type": "catenary_pole", "name": "pole"},
            },
            "frames": {
                "12": {
                    "objects": {
                        "veh1": {
                            "object_data": {
                                "bbox": [{"name": "rgb_center__bbox", "coordinate_system": "rgb_center", "val": [40, 30, 20, 12]}],
                                "cuboid": [{"name": "lidar__cuboid", "coordinate_system": "lidar", "val": [8, 1, 0.8, 0, 0, 0, 1, 4, 2, 1.6]}],
                            }
                        },
                        "ped1": {
                            "object_data": {
                                "bbox": [{"name": "rgb_center__bbox", "coordinate_system": "rgb_center", "val": [12, 25, 6, 16]}],
                                "cuboid": [{"name": "lidar__cuboid", "coordinate_system": "lidar", "val": [5, -1, 0.9, 0, 0, 0, 1, 0.6, 0.6, 1.8]}],
                            }
                        },
                        "trk1": {
                            "object_data": {
                                "poly2d": [{"name": "rgb_center__poly", "coordinate_system": "rgb_center", "val": [5, 50, 40, 40, 75, 50]}],
                                "vec": [{"name": "lidar__vec", "coordinate_system": "lidar", "val": [0, 0, 10, 0]}],
                            }
                        },
                        "pole1": {
                            "object_data": {
                                "bbox": [{"name": "rgb_center__bbox", "coordinate_system": "rgb_center", "val": [70, 20, 6, 24]}],
                                "cuboid": [{"name": "lidar__cuboid", "coordinate_system": "lidar", "val": [12, 2, 2, 0, 0, 0, 1, 0.4, 0.4, 4]}],
                            }
                        },
                    }
                }
            },
        }
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_openlabel_label_assist_exports_teacher_maps_and_points(tmp_path):
    label_json = tmp_path / "labels.json"
    _write_openlabel(label_json)
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((64, 96, 3), dtype=np.uint8))

    sam_base = str(tmp_path / "sam" / "0000000012")
    frame_dir = str(tmp_path / "label_features" / "0000000012")
    lidar_base = str(tmp_path / "lidar" / "0000000012")

    summary = export_label_assist(str(label_json), 12, str(image_path), "rgb_center", sam_base, frame_dir, lidar_base)

    assert summary["label_assist_enabled"] is True
    assert summary["paired_counts"]["vehicle"] == 1
    assert summary["paired_counts"]["person"] == 1
    assert summary["paired_counts"]["track"] == 1
    assert os.path.isfile(sam_base + "_label_vehicle_weight.png")
    assert os.path.isfile(sam_base + "_label_person_dist.png")
    assert os.path.isfile(os.path.join(frame_dir, "debug_label_assist.json"))

    points = (tmp_path / "lidar" / "0000000012_label_object_points.txt").read_text(encoding="utf-8")
    assert " 5 " in points  # SEM_VEHICLE_LIKE
    assert " 7 " in points  # SEM_PERSON_LIKE
    assert " 3 " in points  # SEM_VERTICAL_STRUCTURE
    assert " 1 " in points  # SEM_RAIL_LIKE from lidar track vec

    summary_path = tmp_path / "label_features" / "0000000012" / "debug_label_assist.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["label_track_point_count"] > 0



def test_openlabel_cuboid_sampling_applies_quaternion_rotation(tmp_path):
    label_json = tmp_path / "labels.json"
    # 90 degree yaw around z: axis-aligned size 4x2 should rotate to bbox 2x4 in x/y extents.
    qz = 2 ** 0.5 / 2
    qw = 2 ** 0.5 / 2
    data = {
        "openlabel": {
            "objects": {"veh1": {"type": "road_vehicle", "name": "car"}},
            "frames": {
                "12": {
                    "objects": {
                        "veh1": {
                            "object_data": {
                                "bbox": [{"name": "rgb_center__bbox", "coordinate_system": "rgb_center", "val": [40, 30, 20, 12]}],
                                "cuboid": [{"name": "lidar__cuboid", "coordinate_system": "lidar", "val": [0, 0, 0, 0, 0, qz, qw, 4, 2, 1]}],
                            }
                        }
                    }
                }
            },
        }
    }
    label_json.write_text(json.dumps(data), encoding="utf-8")
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((64, 96, 3), dtype=np.uint8))
    sam_base = str(tmp_path / "sam" / "0000000012")
    frame_dir = str(tmp_path / "label_features" / "0000000012")
    lidar_base = str(tmp_path / "lidar" / "0000000012")

    export_label_assist(str(label_json), 12, str(image_path), "rgb_center", sam_base, frame_dir, lidar_base)
    pts = []
    for line in (tmp_path / "lidar" / "0000000012_label_object_points.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        x, y, z, cls, *_ = line.split()
        pts.append((float(x), float(y), float(z), int(cls)))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    assert max(xs) - min(xs) < 2.1
    assert max(ys) - min(ys) > 3.9


def test_openlabel_teacher_points_are_cropped_to_visible_x_range(tmp_path):
    label_json = tmp_path / "labels.json"
    data = {
        "openlabel": {
            "objects": {
                "pole1": {"type": "catenary_pole", "name": "near pole"},
                "veh_far": {"type": "road_vehicle", "name": "far vehicle"},
            },
            "frames": {
                "12": {
                    "objects": {
                        "pole1": {
                            "object_data": {
                                "bbox": [{"name": "rgb_center__bbox", "coordinate_system": "rgb_center", "val": [40, 30, 8, 20]}],
                                "cuboid": [{"name": "lidar__cuboid", "coordinate_system": "lidar", "val": [10, 0, 2, 0, 0, 0, 1, 0.4, 0.4, 4]}],
                            }
                        },
                        "veh_far": {
                            "object_data": {
                                "bbox": [{"name": "rgb_center__bbox", "coordinate_system": "rgb_center", "val": [60, 30, 16, 10]}],
                                "cuboid": [{"name": "lidar__cuboid", "coordinate_system": "lidar", "val": [200, 0, 1, 0, 0, 0, 1, 4, 2, 2]}],
                            }
                        },
                    }
                }
            },
        }
    }
    label_json.write_text(json.dumps(data), encoding="utf-8")
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((64, 96, 3), dtype=np.uint8))
    sam_base = str(tmp_path / "sam" / "0000000012")
    frame_dir = str(tmp_path / "label_features" / "0000000012")
    lidar_base = str(tmp_path / "lidar" / "0000000012")

    summary = export_label_assist(
        str(label_json),
        12,
        str(image_path),
        "rgb_center",
        sam_base,
        frame_dir,
        lidar_base,
        teacher_visible_xmax_m=120.0,
    )

    assert summary["label_teacher_raw_point_counts"]["vehicle"] > 0
    assert summary["label_vehicle_point_count"] == 0
    assert summary["label_static_point_count"] > 0
    pts = (tmp_path / "lidar" / "0000000012_label_object_points.txt").read_text(encoding="utf-8")
    assert " 5 " not in pts
    assert " 3 " in pts



def test_openlabel_label_assist_exports_strong_feature_json_and_tsv(tmp_path):
    label_json = tmp_path / "labels.json"
    _write_openlabel(label_json)
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((64, 96, 3), dtype=np.uint8))

    sam_base = str(tmp_path / "sam" / "0000000012")
    frame_dir = str(tmp_path / "label_features" / "0000000012")
    lidar_base = str(tmp_path / "lidar" / "0000000012")

    summary = export_label_assist(
        str(label_json),
        12,
        str(image_path),
        "rgb_center",
        sam_base,
        frame_dir,
        lidar_base,
        strong_features_enabled=True,
        max_track_samples_per_object=5,
    )

    assert summary["strong_features_enabled"] is True
    assert summary["strong_label_feature_count"] >= 2
    assert summary["strong_label_feature_counts"]["track"] == 1
    assert summary["strong_label_feature_counts"]["catenary_pole"] == 1

    strong_json = json.loads((tmp_path / "lidar" / "0000000012_label_strong_features.json").read_text(encoding="utf-8"))
    classes = {f["class_type"] for f in strong_json["features"]}
    assert {"track", "catenary_pole"}.issubset(classes)

    tsv = (tmp_path / "lidar" / "0000000012_label_strong_features.tsv").read_text(encoding="utf-8")
    assert "track\ttrk1\tpoint" in tsv
    assert "catenary_pole\tpole1\taxis" in tsv


def test_openlabel_strong_features_skip_unpaired_switch_and_export_buffer_stop(tmp_path):
    label_json = tmp_path / "labels.json"
    data = {
        "openlabel": {
            "objects": {
                "sw1": {"type": "switch", "name": "switch"},
                "buf1": {"type": "buffer_stop", "name": "buffer"},
            },
            "frames": {
                "12": {
                    "objects": {
                        "sw1": {
                            "object_data": {
                                "poly2d": [{"name": "rgb_center__poly", "coordinate_system": "rgb_center", "val": [10, 40, 30, 42, 50, 45]}]
                            }
                        },
                        "buf1": {
                            "object_data": {
                                "bbox": [{"name": "rgb_center__bbox", "coordinate_system": "rgb_center", "val": [40, 30, 20, 12]}],
                                "cuboid": [{"name": "lidar__cuboid", "coordinate_system": "lidar", "val": [12, 0, 1, 0, 0, 0, 1, 2, 1, 2]}],
                            }
                        },
                    }
                }
            },
        }
    }
    label_json.write_text(json.dumps(data), encoding="utf-8")
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((64, 96, 3), dtype=np.uint8))
    sam_base = str(tmp_path / "sam" / "0000000012")
    frame_dir = str(tmp_path / "label_features" / "0000000012")
    lidar_base = str(tmp_path / "lidar" / "0000000012")

    summary = export_label_assist(str(label_json), 12, str(image_path), "rgb_center", sam_base, frame_dir, lidar_base)
    assert summary["strong_label_feature_counts"]["switch"] == 0
    assert summary["strong_label_feature_counts"]["buffer_stop"] == 1
    tsv = (tmp_path / "lidar" / "0000000012_label_strong_features.tsv").read_text(encoding="utf-8")
    assert "buffer_stop\tbuf1" in tsv
    assert "switch\tsw1" not in tsv


def test_openlabel_strong_features_use_per_class_visible_ranges(tmp_path):
    label_json = tmp_path / "labels.json"
    data = {
        "openlabel": {
            "objects": {
                "trk1": {"type": "track", "name": "track"},
                "sw1": {"type": "switch", "name": "switch"},
                "pole_near": {"type": "catenary_pole", "name": "near pole"},
                "pole_mid": {"type": "catenary_pole", "name": "mid pole"},
                "buf_far": {"type": "buffer_stop", "name": "far buffer"},
            },
            "frames": {
                "12": {
                    "objects": {
                        "trk1": {
                            "object_data": {
                                "poly2d": [{"name": "rgb_center__poly", "coordinate_system": "rgb_center", "val": [5, 50, 40, 40, 75, 50]}],
                                "vec": [{"name": "lidar__vec", "coordinate_system": "lidar", "val": [0, 0, 0, 130, 0, 0]}],
                            }
                        },
                        "sw1": {
                            "object_data": {
                                "poly2d": [{"name": "rgb_center__poly", "coordinate_system": "rgb_center", "val": [10, 40, 30, 42, 50, 45]}],
                                "vec": [{"name": "lidar__vec", "coordinate_system": "lidar", "val": [20, 0, 0, 130, 0, 0]}],
                            }
                        },
                        "pole_near": {
                            "object_data": {
                                "bbox": [{"name": "rgb_center__bbox", "coordinate_system": "rgb_center", "val": [70, 20, 6, 24]}],
                                "cuboid": [{"name": "lidar__cuboid", "coordinate_system": "lidar", "val": [105, 2, 2, 0, 0, 0, 1, 0.4, 0.4, 4]}],
                            }
                        },
                        "pole_mid": {
                            "object_data": {
                                "bbox": [{"name": "rgb_center__bbox", "coordinate_system": "rgb_center", "val": [50, 20, 6, 24]}],
                                "cuboid": [{"name": "lidar__cuboid", "coordinate_system": "lidar", "val": [150, 3, 2, 0, 0, 0, 1, 0.4, 0.4, 4]}],
                            }
                        },
                        "buf_far": {
                            "object_data": {
                                "bbox": [{"name": "rgb_center__bbox", "coordinate_system": "rgb_center", "val": [40, 30, 20, 12]}],
                                "cuboid": [{"name": "lidar__cuboid", "coordinate_system": "lidar", "val": [210, 0, 1, 0, 0, 0, 1, 2, 1, 2]}],
                            }
                        },
                    }
                }
            },
        }
    }
    label_json.write_text(json.dumps(data), encoding="utf-8")
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((64, 96, 3), dtype=np.uint8))
    sam_base = str(tmp_path / "sam" / "0000000012")
    frame_dir = str(tmp_path / "label_features" / "0000000012")
    lidar_base = str(tmp_path / "lidar" / "0000000012")

    summary = export_label_assist(
        str(label_json),
        12,
        str(image_path),
        "rgb_center",
        sam_base,
        frame_dir,
        lidar_base,
        strong_features_enabled=True,
        teacher_visible_xmax_m=120.0,
        track_visible_xmax_m=120.0,
        switch_visible_xmax_m=120.0,
        catenary_pole_visible_xmax_m=160.0,
        buffer_stop_visible_xmax_m=240.0,
    )

    assert summary["strong_label_feature_counts"]["track"] == 1
    assert summary["strong_label_feature_counts"]["switch"] == 1
    assert summary["strong_label_feature_counts"]["catenary_pole"] == 2
    assert summary["strong_label_feature_counts"]["buffer_stop"] == 1
    assert summary["strong_label_filtered_by_range_counts"]["track"] >= 1
    assert summary["strong_label_filtered_by_range_counts"]["switch"] >= 1
    tsv = (tmp_path / "lidar" / "0000000012_label_strong_features.tsv").read_text(encoding="utf-8")
    assert "buffer_stop\tbuf_far" in tsv
    assert "catenary_pole\tpole_mid" in tsv


def test_openlabel_extrinsic_uses_inverse_sensor_pose_then_optical_rotation(tmp_path):
    label_json = tmp_path / "labels.json"
    data = {
        "openlabel": {
            "coordinate_systems": {
                "rgb_center": {
                    "type": "sensor",
                    "parent": "base",
                    "pose_wrt_parent": {
                        "translation": [1.0, 2.0, 3.0],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            }
        }
    }
    label_json.write_text(json.dumps(data), encoding="utf-8")

    rvec, tvec = load_osdar23_openlabel_extrinsic(str(label_json), "rgb_center")
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    expected_R = np.asarray([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    expected_t = expected_R @ np.asarray([-1.0, -2.0, -3.0], dtype=np.float64)

    assert np.allclose(R, expected_R, atol=1e-8)
    assert np.allclose(np.asarray(tvec, dtype=np.float64), expected_t, atol=1e-8)


def test_real_openlabel_extrinsic_differs_from_calibration_when_dataset_available():
    label_json = "/gz-data/OSDaR23/1_calibration_1.1/1_calibration_1.1_labels.json"
    if not os.path.isfile(label_json):
        return
    ext = load_osdar23_openlabel_extrinsic(label_json, "rgb_center")
    assert ext is not None
    rvec, tvec = ext
    assert len(rvec) == 3
    assert len(tvec) == 3
    # The known failure mode was silently using calibration.txt instead of OpenLABEL.
    assert abs(float(tvec[0])) < 0.5
    assert abs(float(tvec[1]) - 2.054) < 0.1


def test_strong_label_costs_are_added_to_optimizer_source():
    source = (pathlib.Path(ROOT) / "cpp" / "edge_calibrator.cpp").read_text(encoding="utf-8")
    assert "AutoDiffCostFunction<TrackPolylineProjectionCost" in source
    assert "AutoDiffCostFunction<PoleCenterlineProjectionCost" in source
    assert "AutoDiffCostFunction<BufferStopBBoxProjectionCost" in source
    assert "strong_residuals_added_to_optimizer" in source

