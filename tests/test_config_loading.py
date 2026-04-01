import unittest

from pipeline.context import load_runtime_config, validate_config


class ConfigLoadingTest(unittest.TestCase):
    def test_default_config_valid(self):
        cfg = load_runtime_config("configs/kitti.yaml")
        validate_config(cfg)
        self.assertIn("sam", cfg)
        self.assertIn("calibration", cfg)


if __name__ == "__main__":
    unittest.main()