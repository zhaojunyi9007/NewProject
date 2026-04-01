import os
import tempfile
import unittest

import numpy as np

import importlib.util


_HAS_CV2 = importlib.util.find_spec("cv2") is not None
if _HAS_CV2:
    from visual_result import load_calib_auto


class VisualResultCalibAutoTest(unittest.TestCase):
    @unittest.skipUnless(_HAS_CV2, "cv2 not installed; skip visual_result calib auto test")
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
            K, R_rect, P_rect = load_calib_auto(p)
            self.assertTrue(np.allclose(R_rect, np.eye(3)))
            self.assertTrue(np.allclose(K, np.array([[1200, 0, 800], [0, 1200, 450], [0, 0, 1]], dtype=np.float64)))
            self.assertEqual(P_rect.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()

