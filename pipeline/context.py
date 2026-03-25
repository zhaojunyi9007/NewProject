#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 context helpers: config loading, CLI overrides and validation."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, Iterable, Tuple
import os

import yaml


ConfigPath = Tuple[str, ...]


@dataclass
class RuntimeContext:
    config: Dict[str, Any]
    frame_ids: list[int]


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


def apply_cli_semantic_overrides(config: Dict[str, Any], result_dir: str | None, frame_ids_text: str | None) -> None:
    data_cfg = config.setdefault("data", {})
    frames_cfg = config.setdefault("frames", {})

    if result_dir:
        data_cfg["result_dir"] = result_dir
        linked_dirs = {
            "sam_output_dir": "sam_features",
            "lidar_output_dir": "lidar_features",
            "calib_output_dir": "calibration",
            "visual_output_dir": "visualization",
        }
        for key, default_leaf in linked_dirs.items():
            current_path = str(data_cfg.get(key, "")).strip()
            leaf = os.path.basename(current_path) if current_path else default_leaf
            data_cfg[key] = os.path.join(result_dir, leaf)

    if frame_ids_text:
        frames_cfg["mode"] = "select"


def create_output_dirs(config: Dict[str, Any]) -> None:
    data_cfg = config.get("data", {})
    output_keys = [
        "result_dir",
        "sam_output_dir",
        "lidar_output_dir",
        "calib_output_dir",
        "visual_output_dir",
    ]
    for key in output_keys:
        path = data_cfg.get(key)
        if path:
            os.makedirs(path, exist_ok=True)


def get_frame_list(config: Dict[str, Any]) -> list[int]:
    frames_cfg = config.get("frames", {})
    mode = frames_cfg.get("mode", "select")
    if mode == "select":
        return [int(x) for x in frames_cfg.get("frame_ids", [])]

    image_dir = config.get("data", {}).get("image_dir", "")
    if not image_dir or not os.path.isdir(image_dir):
        return []

    frame_ids: list[int] = []
    for name in os.listdir(image_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".png" or not stem.isdigit():
            continue
        frame_ids.append(int(stem))
    frame_ids.sort()
    return frame_ids
