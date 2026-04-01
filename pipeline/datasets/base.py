#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset adapter abstraction: KITTI vs OSDaR23 differences live here."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DatasetAdapter(ABC):
    """Encapsulates dataset-specific path discovery, calibration I/O, and optimizer env."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    @abstractmethod
    def resolve_image(self, frame_id: int) -> Optional[str]:
        ...

    @abstractmethod
    def resolve_lidar(self, frame_id: int) -> Optional[str]:
        ...

    @abstractmethod
    def list_available_frames(self) -> List[int]:
        ...

    @abstractmethod
    def load_intrinsics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (K_3x3, R_rect_3x3, P_rect_3x4)."""

    @abstractmethod
    def load_initial_extrinsic(self) -> Optional[Tuple[List[float], List[float]]]:
        """Return (rvec_3, tvec_3) or None to use YAML initial_extrinsic."""

    @abstractmethod
    def get_optimizer_env(self) -> Dict[str, str]:
        """Extra env vars for the C++ optimizer subprocess."""

    def apply_derived_paths(self, config: Dict[str, Any]) -> None:
        """Fill derived data paths before validation (default: no-op)."""
        _ = config
