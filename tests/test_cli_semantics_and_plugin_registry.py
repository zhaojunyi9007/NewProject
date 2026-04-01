import unittest

from pipeline.context import apply_cli_semantic_overrides
from pipeline.sam.plugin_registry import get_feature_plugin

class CliSemanticsAndPluginRegistryTest(unittest.TestCase):
    def test_result_dir_override_relinks_stage_outputs(self):
        cfg = {
            "data": {
                "result_dir": "result",
                "sam_output_dir": "result/sam_features",
                "lidar_output_dir": "result/lidar_features",
                "calib_output_dir": "result/calibration",
                "visual_output_dir": "result/visualization",
            },
            "frames": {"mode": "select", "frame_ids": [0]},
        }

        apply_cli_semantic_overrides(cfg, result_dir="/tmp/new_result", frame_ids_text=None)

        self.assertEqual(cfg["data"]["result_dir"], "/tmp/new_result")
        self.assertEqual(cfg["data"]["sam_output_dir"], "/tmp/new_result/sam_features")
        self.assertEqual(cfg["data"]["lidar_output_dir"], "/tmp/new_result/lidar_features")
        self.assertEqual(cfg["data"]["calib_output_dir"], "/tmp/new_result/calibration")
        self.assertEqual(cfg["data"]["visual_output_dir"], "/tmp/new_result/visualization")

    def test_frame_ids_override_forces_select_mode(self):
        cfg = {"data": {}, "frames": {"mode": "all"}}

        apply_cli_semantic_overrides(cfg, result_dir=None, frame_ids_text="0,10")

        self.assertEqual(cfg["frames"]["mode"], "select")

    def test_mask_alignment_enabled_wraps_base_plugin(self):
        cfg = {
            "sam": {
                "feature_plugin": "sam_subprocess",
                "mask_alignment": {"enabled": True, "output_dir": "result/mask_alignment"},
            }
        }

        plugin = get_feature_plugin(cfg)
        self.assertEqual(plugin.name, "mask_alignment(sam_subprocess)")


if __name__ == "__main__":
    unittest.main()