import os
import tempfile
import unittest

import numpy as np

import importlib.util


_HAS_CV2 = importlib.util.find_spec("cv2") is not None
if _HAS_CV2:
    from pipeline.datasets.osdar23 import load_osdar23_init_extrinsic


class OSDaRCalibParseTest(unittest.TestCase):
    @unittest.skipUnless(_HAS_CV2, "cv2 not installed; skip extrinsic rodrigues parse test")
    def test_load_osdar23_init_extrinsic_inverts_transform(self):
        # OSDaR23 codepath:
        #   1) T_lidar_to_body = inv(T_cam_to_parent)
        #   2) T_lidar_to_optical = T_body_to_optical * T_lidar_to_body
        # so rvec is NOT expected to be zero even if T_cam_to_parent rotation is identity.
        #
        # For an identity rotation + translation (1,2,3):
        #   T_lidar_to_body translation = (-1,-2,-3)
        #   optical-X = -body-Y, optical-Y = -body-Z, optical-Z = body-X
        # => t_optical = (2,3,-1)
        calib_txt = "\n".join(
            [
                "CAMERA",
                "data_folder: rgb_center",
                "intrinsics_pinhole:",
                "camera_matrix:",
                "1000 0 960",
                "0 1000 540",
                "0 0 1",
                "homogeneous transform:",
                "1 0 0 1",
                "0 1 0 2",
                "0 0 1 3",
                "0 0 0 1",
            ]
        )

        with tempfile.TemporaryDirectory() as d:
            calib_path = os.path.join(d, "calibration.txt")
            with open(calib_path, "w", encoding="utf-8") as f:
                f.write(calib_txt)

            rvec, tvec = load_osdar23_init_extrinsic(calib_path, "rgb_center")
            self.assertIsNotNone(rvec)
            self.assertIsNotNone(tvec)

            self.assertEqual(len(rvec), 3)
            self.assertTrue(np.all(np.isfinite(np.array(rvec, dtype=np.float64))))
            self.assertTrue(np.allclose(np.array(tvec, dtype=np.float64), np.array([2.0, 3.0, -1.0]), atol=1e-9))

    @unittest.skipUnless(_HAS_CV2, "cv2 not installed; skip extrinsic rodrigues parse test")
    def test_returns_none_when_camera_folder_not_found(self):
        calib_txt = "\n".join(
            [
                "CAMERA",
                "data_folder: rgb_left",
                "homogeneous transform:",
                "1 0 0 0",
                "0 1 0 0",
                "0 0 1 0",
                "0 0 0 1",
            ]
        )
        with tempfile.TemporaryDirectory() as d:
            calib_path = os.path.join(d, "calibration.txt")
            with open(calib_path, "w", encoding="utf-8") as f:
                f.write(calib_txt)
            self.assertIsNone(load_osdar23_init_extrinsic(calib_path, "rgb_center"))


if __name__ == "__main__":
    unittest.main()

