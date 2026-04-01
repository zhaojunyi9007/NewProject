import os
import tempfile
import unittest

from pipeline.context import get_frame_list


class FrameListOSDaRAllModeTest(unittest.TestCase):
    def _touch(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\x00")

    def test_frames_all_uses_osdar_resolver_scan(self):
        with tempfile.TemporaryDirectory() as d:
            img_dir = os.path.join(d, "rgb_center")
            os.makedirs(img_dir, exist_ok=True)
            # OSDaR23 naming: {counter}_{timestamp}.png
            self._touch(os.path.join(img_dir, "7_100.png"))
            self._touch(os.path.join(img_dir, "7_200.png"))  # duplicate counter
            self._touch(os.path.join(img_dir, "9_000.png"))

            cfg = {
                "data": {
                    "dataset_format": "osdar23",
                    "image_dir": img_dir,
                    # velodyne_dir not needed for list_available_frames()
                    "velodyne_dir": os.path.join(d, "lidar"),
                },
                "frames": {"mode": "all"},
            }

            frames = get_frame_list(cfg)
            self.assertEqual(frames, [7, 9])


if __name__ == "__main__":
    unittest.main()

