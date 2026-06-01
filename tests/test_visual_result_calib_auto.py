import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

from pipeline.datasets import get_adapter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import visualize  # noqa: E402


class VisualResultCalibAutoTest(unittest.TestCase):
    def test_detects_osdar_calib_and_parses_camera_matrix(self):
        calib_txt = "\n".join(
            [
                "CAMERA",
                "data_folder: rgb_center",
                "intrinsics_pinhole:",
                "camera_matrix:",
                "1200 0 800",
                "0 1200 450",
                "0 0 1",
                "homogeneous transform:",
                "1 0 0 0",
                "0 1 0 0",
                "0 0 1 0",
                "0 0 0 1",
            ]
        )
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "calibration.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(calib_txt)
            cfg = {"data": {"dataset_format": "osdar23", "calib_file": p, "image_sensor": "rgb_center"}}
            K, R_rect, P_rect = get_adapter(cfg).load_intrinsics()
            self.assertTrue(np.allclose(R_rect, np.eye(3)))
            self.assertTrue(np.allclose(K, np.array([[1200, 0, 800], [0, 1200, 450], [0, 0, 1]], dtype=np.float64)))
            self.assertEqual(P_rect.shape, (3, 4))

    def test_lidar_projection_outputs_depth_colored_image_and_debug(self):
        with tempfile.TemporaryDirectory() as d:
            img = np.full((80, 120, 3), 80, dtype=np.uint8)
            points = [
                [5.0, -0.5, 0.0],
                [8.0, 0.0, 0.0],
                [12.0, 0.5, 0.0],
                [-2.0, 0.0, 0.0],
            ]
            K = np.array([[30.0, 0.0, 60.0], [0.0, 30.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            P = np.hstack([K, np.zeros((3, 1), dtype=np.float64)])
            R_rect = np.eye(3, dtype=np.float64)
            R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
            t = np.zeros(3, dtype=np.float64)
            out = os.path.join(d, "projection.png")
            debug = os.path.join(d, "projection_debug.json")

            ok = visualize.render_lidar_projection(
                img,
                points,
                K,
                R_rect,
                P,
                R,
                t,
                out,
                debug,
                point_source_used="all",
                visualization_calib_file="openlabel_calib.txt",
                pose_source="calibration",
                max_points=120000,
                point_radius=1,
                background="grayscale",
            )

            self.assertTrue(ok)
            self.assertTrue(os.path.isfile(out))
            self.assertTrue(os.path.isfile(debug))
            rendered = cv2.imread(out)
            self.assertIsNotNone(rendered)
            self.assertGreater(int((rendered != rendered[:, :, :1]).sum()), 0)
            import json

            payload = json.load(open(debug, encoding="utf-8"))
            self.assertEqual(payload["point_source_used"], "all")
            self.assertEqual(payload["total_points"], 4)
            self.assertEqual(payload["projected_points"], 3)
            self.assertEqual(payload["behind_count"], 1)
            self.assertEqual(payload["visualization_calib_file"], "openlabel_calib.txt")


if __name__ == "__main__":
    unittest.main()

