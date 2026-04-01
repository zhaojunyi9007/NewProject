#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
import sys

from python.features.interfaces import FeatureFrameContext


class SamSubprocessFeaturePlugin:
    """默认特征插件：保持原行为，调用 python/run_sam.py。"""

    @property
    def name(self) -> str:
        return "sam_subprocess"

    def run_frame(self, context: FeatureFrameContext) -> None:
        sam_cfg = context.config["sam"]
        checkpoint = sam_cfg["checkpoint_path"]
        heuristics_cfg = sam_cfg.get("heuristics", {})
        cmd = [
            sys.executable, "python/run_sam.py",
            "--image", context.image_path,
            "--checkpoint", checkpoint,
            "--output_dir", context.output_dir,
            "--output_prefix", f"{context.frame_id:010d}",
            "--model_type", str(sam_cfg.get("model_type", "vit_h")),
            "--points_per_side", str(sam_cfg.get("points_per_side", 16)),
            "--pred_iou_thresh", str(sam_cfg.get("pred_iou_thresh", 0.86)),
            "--stability_score_thresh", str(sam_cfg.get("stability_score_thresh", 0.92)),
            "--min_mask_region_area", str(sam_cfg.get("min_mask_region_area", 500)),
            "--min_mask_area_ratio", str(heuristics_cfg.get("min_mask_area_ratio", 0.001)),
            "--max_background_area_ratio", str(heuristics_cfg.get("max_background_area_ratio", 0.15)),
            "--sky_mask_bottom_ratio", str(heuristics_cfg.get("sky_mask_bottom_ratio", 0.3)),
            "--ground_region_top_ratio", str(heuristics_cfg.get("ground_region_top_ratio", 0.5)),
            "--flat_ground_aspect_ratio", str(heuristics_cfg.get("flat_ground_aspect_ratio", 3.0)),
            "--structural_aspect_ratio", str(heuristics_cfg.get("structural_aspect_ratio", 3.0)),
            "--contour_stddev_threshold", str(heuristics_cfg.get("contour_stddev_threshold", 10.0)),
            "--min_arc_length_ratio", str(heuristics_cfg.get("min_arc_length_ratio", 0.06)),
            "--global_min_line_length_ratio", str(heuristics_cfg.get("global_min_line_length_ratio", 0.015)),
            "--ground_min_line_length_ratio", str(heuristics_cfg.get("ground_min_line_length_ratio", 0.06)),
            "--sky_min_line_length_ratio", str(heuristics_cfg.get("sky_min_line_length_ratio", 0.03)),
            "--fused_top_black_ratio", str(heuristics_cfg.get("fused_top_black_ratio", 0.20)),
            "--fused_bottom_black_ratio", str(heuristics_cfg.get("fused_bottom_black_ratio", 0.80)),
            "--distance_max_ratio", str(heuristics_cfg.get("distance_max_ratio", 0.15)),
        ]
        subprocess.run(cmd, check=True)