import unittest
from unittest.mock import patch

from pipeline.context import RuntimeContext
from pipeline.stages import sam_stage


class SingleFrameSmokeTest(unittest.TestCase):
    @patch("python.features.plugins.sam_subprocess_plugin.subprocess.run")
    @patch("pipeline.stages.sam_stage.os.path.exists", return_value=True)
    def test_sam_stage_single_frame_smoke(self, _mock_exists, mock_run):
        context = RuntimeContext(
            config={
                "data": {"image_dir": "dataset", "sam_output_dir": "result/sam_features"},
                "sam": {
                    "checkpoint_path": "/tmp/fake_checkpoint.pth",
                    "feature_plugin": "sam_subprocess",
                    "mask_alignment": {"enabled": False},
                },
            },
            frame_ids=[0],
        )
        sam_stage.run(context)
        self.assertEqual(mock_run.call_count, 1)


if __name__ == "__main__":
    unittest.main()