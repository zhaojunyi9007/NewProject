#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 context helpers: config loading, CLI overrides and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple
import os

import yaml

from pipeline.datasets import get_adapter


ConfigPath = Tuple[str, ...]


@dataclass
class RuntimeContext:
    """Per-run pipeline state. Phase 1+ pose fields filled by later stages."""

    config: Dict[str, Any]
    frame_ids: list[int]
    current_pose_init: Any = None
    current_pose_bev: Any = None
    bev_pose_by_frame: Dict[int, Any] = field(default_factory=dict)
    current_pose_semantic: Any = None
    current_pose_refined: Any = None
    frame_bundle_ids: Optional[List[int]] = None
    paths: Dict[str, str] = field(default_factory=dict)


def _set_nested(config: Dict[str, Any], path: ConfigPath, value: Any) -> None:
    node = config
    for key in path[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[path[-1]] = value


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge two dictionaries.
    - Dict values merge recursively
    - Non-dict values (including lists) are overridden
    """
    out: Dict[str, Any] = deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _load_yaml_with_base(config_path: str, _stack: Tuple[str, ...] = ()) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误（非字典）: {config_path}")

    bases = cfg.get("_base")
    if not bases:
        return cfg

    if config_path in _stack:
        chain = " -> ".join(_stack + (config_path,))
        raise ValueError(f"检测到配置 _base 循环引用: {chain}")

    if isinstance(bases, str):
        base_list = [bases]
    elif isinstance(bases, list) and all(isinstance(x, str) for x in bases):
        base_list = list(bases)
    else:
        raise ValueError(f"_base 必须是字符串或字符串列表: {config_path}")

    merged_base: Dict[str, Any] = {}
    here = os.path.dirname(os.path.abspath(config_path))
    for rel in base_list:
        base_path = rel
        if not os.path.isabs(base_path):
            base_path = os.path.join(here, rel)
        base_cfg = _load_yaml_with_base(base_path, _stack=_stack + (config_path,))
        merged_base = _deep_merge_dicts(merged_base, base_cfg)

    child = dict(cfg)
    child.pop("_base", None)
    return _deep_merge_dicts(merged_base, child)


def load_runtime_config(config_path: str, cli_overrides: Iterable[Tuple[ConfigPath, Any]] | None = None) -> Dict[str, Any]:
    merged = _load_yaml_with_base(config_path)
    if cli_overrides:
        for path, value in cli_overrides:
            if value is None:
                continue
            _set_nested(merged, path, value)
    return merged


def prepare_runtime_config(config: Dict[str, Any]) -> None:
    """Apply dataset-specific path rules before validation and frame listing."""
    get_adapter(config).apply_derived_paths(config)


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

    ds_fmt = str(config.get("data", {}).get("dataset_format", "osdar23") or "osdar23").lower()
    if ds_fmt not in {"osdar23", "osdar"}:
        raise ValueError(f"data.dataset_format 仅支持 osdar23/osdar，当前: {ds_fmt}")

    if mode == "select":
        frame_ids = config["frames"].get("frame_ids")
        if not isinstance(frame_ids, list) or not frame_ids:
            raise ValueError("frames.mode=select 时，frames.frame_ids 必须是非空列表")
        _log_resolved_paths(config, [int(x) for x in frame_ids[:10]])


def _log_resolved_paths(config: Dict[str, Any], frame_ids: list[int]) -> None:
    if not frame_ids:
        return
    try:
        adapter = get_adapter(config)
    except Exception as e:
        print(f"[Warning] Dataset adapter init failed: {e}")
        return

    print("[Info] Frame source resolution preview (up to 10 frames):")
    for fid in frame_ids:
        img = adapter.resolve_image(fid)
        lidar = adapter.resolve_lidar(fid)
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
            "image_features_output_dir": "image_features",
            "bev_init_output_dir": "bev_init",
            "refinement_output_dir": "refinement",
        }
        for key, default_leaf in linked_dirs.items():
            current_path = str(data_cfg.get(key, "")).strip()
            leaf = os.path.basename(current_path) if current_path else default_leaf
            data_cfg[key] = os.path.join(result_dir, leaf)

    if frame_ids_text:
        frames_cfg["mode"] = "select"


def resolve_stage_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """Resolve canonical output directories for each pipeline stage."""
    data = config.get("data", {})
    rd = str(data.get("result_dir", "result") or "result")

    def _abs(key: str, default_leaf: str) -> str:
        p = data.get(key)
        if p is not None and str(p).strip():
            return os.path.abspath(str(p))
        return os.path.abspath(os.path.join(rd, default_leaf))

    return {
        "result_dir": os.path.abspath(rd),
        "sam": _abs("sam_output_dir", "sam_features"),
        "lidar": _abs("lidar_output_dir", "lidar_features"),
        "calib": _abs("calib_output_dir", "calibration"),
        "visual": _abs("visual_output_dir", "visualization"),
        "image_features": _abs("image_features_output_dir", "image_features"),
        "bev_init": _abs("bev_init_output_dir", "bev_init"),
        "refinement": _abs("refinement_output_dir", "refinement"),
    }


def attach_stage_paths_to_context(context: RuntimeContext) -> None:
    context.paths = resolve_stage_paths(context.config)


def resolve_frame_bundle_window(config: Dict[str, Any], center_frame_id: int) -> List[int]:
    """Return frame IDs for a sliding window around center_frame_id (refine.window_size)."""
    refine_cfg = config.get("refine", {}) if isinstance(config.get("refine"), dict) else {}
    w = int(refine_cfg.get("window_size", 5) or 5)
    w = max(1, w)
    half = w // 2
    return [center_frame_id - half + i for i in range(w)]


def load_refinement_state_if_exists(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load persisted refinement state if present. Phase 1: optional JSON stub."""
    paths = resolve_stage_paths(config)
    cand = os.path.join(paths["refinement"], "state.json")
    if not os.path.isfile(cand):
        return None
    try:
        import json

        with open(cand, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def create_output_dirs(config: Dict[str, Any]) -> None:
    paths = resolve_stage_paths(config)
    for _k, path in paths.items():
        if path and _k != "result_dir":
            os.makedirs(path, exist_ok=True)
    rd = paths.get("result_dir")
    if rd:
        os.makedirs(rd, exist_ok=True)


def get_frame_list(config: Dict[str, Any]) -> list[int]:
    frames_cfg = config.get("frames", {})
    mode = frames_cfg.get("mode", "select")
    if mode == "select":
        return [int(x) for x in frames_cfg.get("frame_ids", [])]

    prepare_runtime_config(config)
    adapter = get_adapter(config)
    frame_ids = adapter.list_available_frames()
    _log_resolved_paths(config, frame_ids[:10])
    return frame_ids
