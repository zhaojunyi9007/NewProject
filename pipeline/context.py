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


def apply_data_paths_from_dataset(config: Dict[str, Any]) -> None:
    """
    When dataset_format is OSDaR23 and osdar_sequence_root is set, derive standard paths:
    image_dir = <root>/<image_sensor>, velodyne_dir = <root>/lidar, calib_file = <root>/calibration.txt
    KITTI paths in yaml are left unchanged when dataset_format=kitti or root is empty.
    """
    data = config.setdefault("data", {})
    fmt = str(data.get("dataset_format", "kitti") or "kitti").lower()
    root = str(data.get("osdar_sequence_root", "") or "").strip()
    if fmt in {"osdar23", "osdar"} and root:
        sensor = str(data.get("image_sensor", "rgb_center") or "rgb_center").strip() or "rgb_center"
        data["image_dir"] = os.path.join(root, sensor)
        data["velodyne_dir"] = os.path.join(root, "lidar")
        data["calib_file"] = os.path.join(root, "calibration.txt")


def prepare_runtime_config(config: Dict[str, Any]) -> None:
    """Apply dataset-specific path rules before validation and frame listing."""
    apply_data_paths_from_dataset(config)


def validate_config(config: Dict[str, Any]) -> None:
    # Ensure dataset-derived paths are applied before validation.
    prepare_runtime_config(config)
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

    ds_fmt = str(config.get("data", {}).get("dataset_format", "kitti") or "kitti").lower()
    if ds_fmt not in {"kitti", "osdar23", "osdar"}:
        raise ValueError(f"data.dataset_format 仅支持 kitti / osdar23，当前: {ds_fmt}")

    if mode == "select":
        frame_ids = config["frames"].get("frame_ids")
        if not isinstance(frame_ids, list) or not frame_ids:
            raise ValueError("frames.mode=select 时，frames.frame_ids 必须是非空列表")
        _log_resolved_paths(config, [int(x) for x in frame_ids[:10]])


def _log_resolved_paths(config: Dict[str, Any], frame_ids: list[int]) -> None:
    if not frame_ids:
        return
    try:
        from pipeline.dataset_resolver import get_dataset_resolver

        resolver = get_dataset_resolver(config)
    except Exception as e:
        print(f"[Warning] Dataset resolver init failed: {e}")
        return

    print("[Info] Frame source resolution preview (up to 10 frames):")
    for fid in frame_ids:
        img = resolver.resolve_image(fid)
        lidar = resolver.resolve_lidar(fid)
        print(f"  frame_id={fid} image={img} lidar={lidar}")


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
    from pipeline.dataset_resolver import get_dataset_resolver

    frames_cfg = config.get("frames", {})
    mode = frames_cfg.get("mode", "select")
    if mode == "select":
        return [int(x) for x in frames_cfg.get("frame_ids", [])]

    prepare_runtime_config(config)
    resolver = get_dataset_resolver(config)
    frame_ids = resolver.list_available_frames()
    _log_resolved_paths(config, frame_ids[:10])
    return frame_ids
