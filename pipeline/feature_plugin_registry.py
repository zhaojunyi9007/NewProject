#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict

from python.features.plugins.sam_subprocess_plugin import SamSubprocessFeaturePlugin


def get_feature_plugin(config: Dict[str, Any]):
    plugin_name = config.get("sam", {}).get("feature_plugin", "sam_subprocess")
    if plugin_name == "sam_subprocess":
        return SamSubprocessFeaturePlugin()
    raise ValueError(f"未知特征插件: {plugin_name}")