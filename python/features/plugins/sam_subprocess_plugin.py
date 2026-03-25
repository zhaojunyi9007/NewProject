#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess

from python.features.interfaces import FeatureFrameContext


class SamSubprocessFeaturePlugin:
    """默认特征插件：保持原行为，调用 python/run_sam.py。"""

    @property
    def name(self) -> str:
        return "sam_subprocess"

    def run_frame(self, context: FeatureFrameContext) -> None:
        checkpoint = context.config["sam"]["checkpoint_path"]
        cmd = [
            "python", "python/run_sam.py",
            "--image", context.image_path,
            "--checkpoint", checkpoint,
            "--output_dir", context.output_dir,
        ]
        subprocess.run(cmd, check=True)