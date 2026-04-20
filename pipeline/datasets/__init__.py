#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict

from pipeline.datasets.base import DatasetAdapter
from pipeline.datasets.osdar23 import OSDaR23Adapter


def get_adapter(config: Dict[str, Any]) -> DatasetAdapter:
    fmt = str(config.get("data", {}).get("dataset_format", "osdar23") or "osdar23").lower()
    if fmt in {"osdar23", "osdar"}:
        return OSDaR23Adapter(config)
    raise ValueError(f"仅支持 OsDaR23 数据集（data.dataset_format=osdar23/osdar），当前: {fmt}")


__all__ = ["DatasetAdapter", "OSDaR23Adapter", "get_adapter"]
