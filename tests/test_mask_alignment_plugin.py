import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from pipeline.sam.interfaces import FeatureFrameContext
from pipeline.sam.mask_alignment_plugin import MaskAlignmentFeaturePlugin


class _DummyBasePlugin:
    @property
    def name(self):
        return "dummy"

    def run_frame(self, context: FeatureFrameContext) -> None:
        tag = f"{context.frame_id:010d}"
        base = os.path.join(context.output_dir, tag)
        mask = np.zeros((8, 8), dtype=np.uint16)
        mask[2:6, 2:6] = 1
        edge = np.zeros((8, 8), dtype=np.uint8)
        edge[3:5, :] = 255
        Image.fromarray(mask).save(base + "_mask_ids.png")
        Image.fromarray(edge).save(base + "_edge_map.png")
        Image.fromarray(edge).save(base + "_rail_centerline.png")


class MaskAlignmentPluginTest(unittest.TestCase):
    def test_plugin_generates_metrics_when_enabled(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "sam")
            metrics_dir = os.path.join(td, "metrics")
            os.makedirs(out_dir, exist_ok=True)

            plugin = MaskAlignmentFeaturePlugin(base_plugin=_DummyBasePlugin(), output_dir=metrics_dir)
            ctx = FeatureFrameContext(frame_id=0, image_path="", output_dir=out_dir, config={})
            plugin.run_frame(ctx)

            metrics_path = os.path.join(metrics_dir, "0000000000_mask_alignment.json")
            self.assertTrue(os.path.exists(metrics_path))
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("alignment_score", data)
            self.assertGreaterEqual(data["alignment_score"], 0.0)


if __name__ == "__main__":
    unittest.main()