#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 7：各阶段统一的 score breakdown / 调试元数据 JSON 结构。"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional


SCHEMA_VERSION = 1


def write_unified_debug_json(
    path: str,
    *,
    stage: str,
    frame_id: Optional[str] = None,
    input_pose: Optional[Dict[str, Any]] = None,
    output_pose: Optional[Dict[str, Any]] = None,
    breakdown: Optional[Dict[str, Any]] = None,
    elapsed_sec: Optional[float] = None,
    confidence: Optional[float] = None,
    observability: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    统一字段（Phase 7.3）：
    - stage: bev | calib | refine
    - input_pose / output_pose: { "rvec": [3], "tvec": [3] }
    - breakdown: 子项分数（含 BEV/语义 JS/边缘等，按阶段填充）
    - elapsed_sec, confidence, observability
    """
    doc: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "unix_time": time.time(),
    }
    if frame_id is not None:
        doc["frame_id"] = str(frame_id)
    if input_pose is not None:
        doc["input_pose"] = input_pose
    if output_pose is not None:
        doc["output_pose"] = output_pose
    if breakdown is not None:
        doc["breakdown"] = breakdown
    if elapsed_sec is not None:
        doc["elapsed_sec"] = float(elapsed_sec)
    if confidence is not None:
        doc["confidence"] = float(confidence)
    if observability is not None:
        doc["observability"] = float(observability)
    if meta:
        doc["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
