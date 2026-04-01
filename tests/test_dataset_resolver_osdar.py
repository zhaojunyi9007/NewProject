import os
import tempfile
import unittest

from pipeline.dataset_resolver import OSDaRResolver, get_dataset_resolver


class DatasetResolverOSDaRTest(unittest.TestCase):
    def _touch(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\x00")

    def test_list_and_resolve_latest_default(self):
        with tempfile.TemporaryDirectory() as d:
            img_dir = os.path.join(d, "rgb_center")
            lidar_dir = os.path.join(d, "lidar")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lidar_dir, exist_ok=True)

            # counter=12 has duplicates (timestamps)
            self._touch(os.path.join(img_dir, "012_20240101.png"))
            self._touch(os.path.join(img_dir, "012_20240102.png"))
            self._touch(os.path.join(lidar_dir, "012_20240101.pcd"))
            self._touch(os.path.join(lidar_dir, "012_20240103.pcd"))
            # counter=13 single
            self._touch(os.path.join(img_dir, "013_20240101.png"))
            self._touch(os.path.join(lidar_dir, "013_20240101.pcd"))

            cfg = {
                "data": {
                    "dataset_format": "osdar23",
                    "image_dir": img_dir,
                    "velodyne_dir": lidar_dir,
                }
            }
            resolver = get_dataset_resolver(cfg)
            self.assertIsInstance(resolver, OSDaRResolver)

            self.assertEqual(resolver.list_available_frames(), [12, 13])
            # Should match even when filenames carry leading zeros (012_*)
            self.assertTrue(resolver.resolve_image(12).endswith("012_20240102.png"))
            self.assertTrue(resolver.resolve_lidar(12).endswith("012_20240103.pcd"))

    def test_resolve_earliest_policy(self):
        with tempfile.TemporaryDirectory() as d:
            img_dir = os.path.join(d, "rgb_center")
            os.makedirs(img_dir, exist_ok=True)
            self._touch(os.path.join(img_dir, "5_0002.png"))
            self._touch(os.path.join(img_dir, "5_0001.png"))

            cfg = {
                "data": {
                    "dataset_format": "osdar23",
                    "image_dir": img_dir,
                    "velodyne_dir": os.path.join(d, "lidar"),
                    "osdar_duplicate_policy": "earliest",
                }
            }
            resolver = get_dataset_resolver(cfg)
            self.assertTrue(resolver.resolve_image(5).endswith("5_0001.png"))


if __name__ == "__main__":
    unittest.main()

