#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass
class FeatureFrameContext:
    frame_id: int
    image_path: str
    output_dir: str
    config: Dict[str, Any]


class FeaturePlugin(Protocol):
    @property
    def name(self) -> str:
        ...

    def run_frame(self, context: FeatureFrameContext) -> None:
        ...

