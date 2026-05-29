#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图像语义特征阶段：语义概率、语义边缘、LSD、pseudo-BEV（Phase 2）。"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict

import numpy as np

from pipeline.context import RuntimeContext
from pipeline.datasets import get_adapter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TOOLS = os.path.join(_REPO_ROOT, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from sam_extractor import FeatureExtractor  # noqa: E402



def _resolve_label_json(cfg: Dict[str, Any]) -> str:
    label_cfg = cfg.get("label_assist") or {}
    explicit = str(label_cfg.get("label_json", "") or "").strip()
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(_REPO_ROOT, explicit)
    root = str(cfg.get("data", {}).get("osdar_sequence_root", "") or "").strip()
    if root:
        tagged = os.path.basename(root.rstrip(os.sep))
        candidates = [
            os.path.join(root, f"{tagged}_labels.json"),
            os.path.join(root, "1_calibration_1.1_labels.json"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
    fallback = os.path.join(_REPO_ROOT, "1_calibration_1.1_labels.json")
    return fallback if os.path.isfile(fallback) else ""


def _export_label_assist_if_enabled(cfg: Dict[str, Any], frame_id: int, image_path: str, frame_dir: str, sam_base: str, lidar_base: str) -> None:
    label_cfg = cfg.get("label_assist") or {}
    if not bool(label_cfg.get("enabled", False)):
        return
    label_json = _resolve_label_json(cfg)
    if not label_json or not os.path.isfile(label_json):
        print("[Warning] label_assist.enabled=true but label JSON was not found; continuing unsupervised path")
        return
    tool = os.path.join(_TOOLS, "openlabel_label_assist.py")
    if not os.path.isfile(tool):
        print(f"[Warning] Missing label assist exporter: {tool}")
        return
    cmd = [
        sys.executable,
        tool,
        "--label-json", label_json,
        "--frame-id", str(frame_id),
        "--image", image_path,
        "--image-sensor", str(cfg.get("data", {}).get("image_sensor", "rgb_center") or "rgb_center"),
        "--sam-base", sam_base,
        "--frame-dir", frame_dir,
        "--lidar-base", lidar_base,
    ]
    subprocess.run(cmd, check=True)


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段 image_features] 图像语义特征（语义优先流水线）")
    print("=" * 40)

    cfg: Dict[str, Any] = context.config
    img_cfg: Dict[str, Any] = dict(cfg.get("image_features") or {})
    if not img_cfg.get("enabled", False):
        print("[Info] image_features.enabled=false，跳过")
        return

    sam_cfg: Dict[str, Any] = dict(cfg.get("sam") or {})
    bev_cfg: Dict[str, Any] = dict(cfg.get("bev") or {})
    paths = context.paths or {}
    out_root = paths.get("image_features") or cfg.get("data", {}).get("image_features_output_dir", "")
    sam_root = paths.get("sam") or cfg.get("data", {}).get("sam_output_dir", "")
    label_root = paths.get("label_features") or cfg.get("data", {}).get("label_features_output_dir", "")
    if not out_root:
        print("[Error] 缺少 image_features 输出路径")
        return
    if not sam_root:
        print("[Error] 缺少 sam_output_dir（optimizer 图像输入输出前缀）")
        return

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(sam_root, exist_ok=True)
    if label_root:
        os.makedirs(label_root, exist_ok=True)

    ckpt = str(sam_cfg.get("checkpoint_path", "") or "").strip()
    if not ckpt or not os.path.isfile(ckpt):
        print(f"[Error] SAM checkpoint 无效或不存在: {ckpt}")
        return

    heuristics = dict(sam_cfg.get("heuristics") or {})

    extractor = FeatureExtractor(
        checkpoint_path=ckpt,
        model_type=str(sam_cfg.get("model_type", "vit_h")),
        device=None,
        points_per_side=int(sam_cfg.get("points_per_side", 16)),
        pred_iou_thresh=float(sam_cfg.get("pred_iou_thresh", 0.86)),
        stability_score_thresh=float(sam_cfg.get("stability_score_thresh", 0.92)),
        min_mask_region_area=int(sam_cfg.get("min_mask_region_area", 500)),
        heuristics=heuristics,
    )

    adapter = get_adapter(cfg)
    K, _, _ = adapter.load_intrinsics()
    ext = adapter.load_initial_extrinsic()
    if ext:
        rvec = np.asarray(ext[0], dtype=np.float64).reshape(3)
        tvec = np.asarray(ext[1], dtype=np.float64).reshape(3)
    else:
        ie = cfg.get("calibration", {}).get("initial_extrinsic", {})
        rvec = np.asarray(ie.get("rotation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        tvec = np.asarray(ie.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)

    ds_meta = cfg.get("dataset") or {}
    reference_z = float(ds_meta.get("reference_z", 0.0))
    dataset_meta = {
        "reference_z": reference_z,
        "semantic_classes": list(img_cfg.get("semantic_classes", [])),
        "dataset_format": str(cfg.get("data", {}).get("dataset_format", "")),
        "osdar_sequence_root": str(cfg.get("data", {}).get("osdar_sequence_root", "") or ""),
        "image_sensor": str(cfg.get("data", {}).get("image_sensor", "rgb_center") or "rgb_center"),
    }

    for frame_id in context.frame_ids:
        img_path = adapter.resolve_image(frame_id)
        if not img_path or not os.path.isfile(img_path):
            print(f"[Warning] 图像不存在，跳过帧 {frame_id:010d}: {img_path}")
            continue

        frame_dir = os.path.join(out_root, f"{frame_id:010d}")
        # 统一 optimizer 图像输入前缀：sam_output_dir/<frame_id>
        sam_base = os.path.join(sam_root, f"{frame_id:010d}")
        lidar_root = paths.get("lidar") or cfg.get("data", {}).get("lidar_output_dir", "")
        lidar_bev_path = os.path.join(lidar_root, f"{frame_id:010d}_bev_maps.npz") if lidar_root else ""

        print(f"\n处理帧 {frame_id:010d}...")
        print(f"  image={img_path}")
        print(f"  bundle_dir={frame_dir}")
        print(f"  optimizer_base={sam_base}")
        if lidar_bev_path:
            print(f"  lidar_bev={lidar_bev_path}")

        ok = extractor.process_image_feature_bundle(
            img_path,
            frame_dir,
            sam_base,
            img_cfg,
            bev_cfg,
            K,
            rvec,
            tvec,
            dataset_meta,
            lidar_bev_path=lidar_bev_path,
            frame_id=frame_id,
        )
        if not ok:
            print(f"[Warning] 帧 {frame_id:010d} 特征提取失败")
        else:
            lidar_base = os.path.join(lidar_root, f"{frame_id:010d}") if lidar_root else ""
            label_frame_dir = os.path.join(label_root, f"{frame_id:010d}") if label_root else frame_dir
            _export_label_assist_if_enabled(cfg, frame_id, img_path, label_frame_dir, sam_base, lidar_base)

    print(f"\n[完成] 图像语义特征已保存到: {out_root}")
