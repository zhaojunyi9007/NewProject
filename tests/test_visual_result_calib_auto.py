import os
import tempfile
import unittest

import numpy as np

from pipeline.datasets import get_adapter


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


if __name__ == "__main__":
    unittest.main()

