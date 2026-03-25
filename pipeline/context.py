#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 context helpers: config loading, CLI overrides and validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Tuple

import yaml


ConfigPath = Tuple[str, ...]


def _set_nested(config: Dict[str, Any], path: ConfigPath, value: Any) -> None:
    node = config
    for key in path[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[path[-1]] = value


def load_runtime_config(config_path: str, cli_overrides: Iterable[Tuple[ConfigPath, Any]] | None = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式错误（非字典）: {config_path}")

    merged = deepcopy(config)
    if cli_overrides:
        for path, value in cli_overrides:
            if value is None:
                continue
            _set_nested(merged, path, value)
    return merged


def validate_config(config: Dict[str, Any]) -> None:
    required_paths = [
        ("data", "result_dir"),
        ("data", "sam_output_dir"),
        ("data", "lidar_output_dir"),
        ("data", "calib_output_dir"),
        ("data", "visual_output_dir"),
        ("frames", "mode"),
        ("calibration", "initial_extrinsic", "rotation"),
        ("calibration", "initial_extrinsic", "translation"),
    ]
    for path in required_paths:
        node = config
        for key in path:
            if not isinstance(node, dict) or key not in node:
                dotted = ".".join(path)
                raise ValueError(f"配置缺少必填项: {dotted}")
            node = node[key]

    mode = config["frames"]["mode"]
    if mode not in {"select", "all"}:
        raise ValueError(f"frames.mode 仅支持 select/all，当前: {mode}")

    if mode == "select":
        frame_ids = config["frames"].get("frame_ids")
        if not isinstance(frame_ids, list) or not frame_ids:
            raise ValueError("frames.mode=select 时，frames.frame_ids 必须是非空列表")


def parse_frame_ids(frame_ids_text: str | None) -> list[int] | None:
    if not frame_ids_text:
        return None
    items = [x.strip() for x in frame_ids_text.split(",") if x.strip()]
    if not items:
        return None
    return [int(x) for x in items]