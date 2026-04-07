#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图像语义特征阶段：语义概率、语义边缘、LSD、pseudo-BEV（Phase 2）。"""

from __future__ import annotations

import os
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
    sam_dir = cfg.get("data", {}).get("sam_output_dir", "")
    if not out_root or not sam_dir:
        print("[Error] 缺少 image_features 或 sam_output_dir 路径")
        return

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(sam_dir, exist_ok=True)

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
    }

    for frame_id in context.frame_ids:
        img_path = adapter.resolve_image(frame_id)
        if not img_path or not os.path.isfile(img_path):
            print(f"[Warning] 图像不存在，跳过帧 {frame_id:010d}: {img_path}")
            continue

        frame_dir = os.path.join(out_root, f"{frame_id:010d}")
        sam_base = os.path.join(sam_dir, f"{frame_id:010d}")

        print(f"\n处理帧 {frame_id:010d}...")
        print(f"  image={img_path}")
        print(f"  bundle_dir={frame_dir}")
        print(f"  sam_base={sam_base}")

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
        )
        if not ok:
            print(f"[Warning] 帧 {frame_id:010d} 特征提取失败")

    print(f"\n[完成] 图像语义特征已保存到: {out_root}")
