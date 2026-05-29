import json
import os
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
