#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from python.features.interfaces import FeatureFrameContext, FeaturePlugin


@dataclass
class MaskAlignmentFeaturePlugin:
    """
    Mask alignment 插件（不替换原算法）：
    1) 先调用基础插件完成原有特征提取
    2) 在输出结果上计算额外对齐指标并写入日志/JSON
    """

    base_plugin: FeaturePlugin
    output_dir: str = "result/mask_alignment"

    @property
    def name(self) -> str:
        return f"mask_alignment({self.base_plugin.name})"

    def run_frame(self, context: FeatureFrameContext) -> None:
        self.base_plugin.run_frame(context)
        self._evaluate_and_dump(context)

    def _evaluate_and_dump(self, context: FeatureFrameContext) -> None:
        frame_tag = f"{context.frame_id:010d}"
        base = os.path.join(context.output_dir, frame_tag)
        mask_path = base + "_mask_ids.png"
        edge_path = base + "_edge_map.png"
        line_path = base + "_line_map.png"

        if not (os.path.exists(mask_path) and os.path.exists(edge_path)):
            print(f"[MaskAlignment][Warning] 缺少输出文件，跳过评估: frame={frame_tag}")
            return

        try:
            mask_img = np.array(Image.open(mask_path))
            edge_img = np.array(Image.open(edge_path))
            line_img = np.array(Image.open(line_path)) if os.path.exists(line_path) else None
        except Exception:
            print(f"[MaskAlignment][Warning] 读取输出失败，跳过评估: frame={frame_tag}")
            return

        if edge_img.ndim == 3:
            edge_img = edge_img[..., 0]
        if line_img is not None and line_img.ndim == 3:
            line_img = line_img[..., 0]

        mask_bin = (mask_img > 0).astype(np.uint8)
        edge_bin = (edge_img > 0).astype(np.uint8)
        line_bin = (line_img > 0).astype(np.uint8) if line_img is not None else np.zeros_like(edge_bin)
        fused_edges = np.clip(edge_bin + line_bin, 0, 1)

        # 简单稳健指标：边缘位于掩码区域比例 + 覆盖率
        covered_edges = int((fused_edges * mask_bin).sum())
        total_edges = int(fused_edges.sum())
        total_mask = int(mask_bin.sum())
        image_area = int(mask_bin.shape[0] * mask_bin.shape[1])

        edge_in_mask_ratio = float(covered_edges / total_edges) if total_edges > 0 else 0.0
        mask_coverage_ratio = float(total_mask / image_area) if image_area > 0 else 0.0
        alignment_score = 0.7 * edge_in_mask_ratio + 0.3 * mask_coverage_ratio

        os.makedirs(self.output_dir, exist_ok=True)
        metrics = {
            "frame_id": context.frame_id,
            "edge_pixels": total_edges,
            "covered_edge_pixels": covered_edges,
            "edge_in_mask_ratio": edge_in_mask_ratio,
            "mask_coverage_ratio": mask_coverage_ratio,
            "alignment_score": alignment_score,
            "plugin": self.name,
        }
        metrics_path = os.path.join(self.output_dir, f"{frame_tag}_mask_alignment.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(
            f"[MaskAlignment] frame={frame_tag} "
            f"score={alignment_score:.4f} edge_in_mask={edge_in_mask_ratio:.4f} "
            f"mask_cov={mask_coverage_ratio:.4f}"
        )