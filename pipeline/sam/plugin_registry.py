#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict

from python.features.mask_alignment_plugin import MaskAlignmentFeaturePlugin
from python.features.plugins.sam_subprocess_plugin import SamSubprocessFeaturePlugin


def get_feature_plugin(config: Dict[str, Any]):
    plugin_name = config.get("sam", {}).get("feature_plugin", "sam_subprocess")
    plugin = None
    if plugin_name == "sam_subprocess":
        plugin = SamSubprocessFeaturePlugin()
    else:
        raise ValueError(f"未知特征插件: {plugin_name}")

    mask_alignment_cfg = config.get("sam", {}).get("mask_alignment", {})
    if mask_alignment_cfg.get("enabled", False):
        output_dir = str(mask_alignment_cfg.get("output_dir", "result/mask_alignment"))
        plugin = MaskAlignmentFeaturePlugin(base_plugin=plugin, output_dir=output_dir)
    return plugin