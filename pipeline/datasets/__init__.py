#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict

from pipeline.datasets.base import DatasetAdapter
from pipeline.datasets.kitti import KittiAdapter
from pipeline.datasets.osdar23 import OSDaR23Adapter


def get_adapter(config: Dict[str, Any]) -> DatasetAdapter:
    fmt = str(config.get("data", {}).get("dataset_format", "kitti") or "kitti").lower()
    if fmt in {"osdar23", "osdar"}:
        return OSDaR23Adapter(config)
    return KittiAdapter(config)


__all__ = ["DatasetAdapter", "KittiAdapter", "OSDaR23Adapter", "get_adapter"]
