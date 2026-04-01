import os
import tempfile
import unittest
from unittest.mock import patch

from pipeline.context import RuntimeContext
from pipeline.stages import lidar_stage


class LidarStageOSDaRSmokeTest(unittest.TestCase):
    @patch("pipeline.stages.lidar_stage.subprocess.run")
    def test_lidar_stage_builds_pcd_command(self, mock_run):
        with tempfile.TemporaryDirectory() as d:
            img_dir = os.path.join(d, "rgb_center")
            lidar_dir = os.path.join(d, "lidar")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lidar_dir, exist_ok=True)

            # Create fake pcd files (existence check only)
            for fid in [10, 11, 12]:
                p = os.path.join(lidar_dir, f"{fid:03d}_123456.pcd")
                with open(p, "wb") as f:
                    f.write(b"\x00")

            cfg = {
                "data": {
                    "dataset_format": "osdar23",
                    "image_dir": img_dir,
                    "velodyne_dir": lidar_dir,
                    "lidar_output_dir": os.path.join(d, "out_lidar"),
                },
                "frames": {"fusion_window": 2},
                "lidar": {},
            }
            context = RuntimeContext(config=cfg, frame_ids=[10, 11, 12])

            lidar_stage.run(context)

            # Should run once per frame (3 frames)
            self.assertEqual(mock_run.call_count, 3)

            # Last call should include two fused frames: 11 and 12
            last_args, last_kwargs = mock_run.call_args
            cmd = last_args[0]
            self.assertEqual(cmd[0], "./build/lidar_extractor")
            self.assertTrue(cmd[1].endswith("011_123456.pcd"))
            self.assertTrue(cmd[2].endswith("012_123456.pcd"))
            self.assertTrue(cmd[3].endswith("0000000012"))


if __name__ == "__main__":
    unittest.main()

