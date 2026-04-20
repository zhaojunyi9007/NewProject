#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset path resolution for KITTI vs OSDaR23 (I/O discovery only).

OSDaR23 filenames: ``{counter}_{timestamp}.png`` / ``{counter}_{timestamp}.pcd``
KITTI filenames: ``{frame_id:010d}.png`` / ``{frame_id:010d}.bin``
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

PathOrNone = Optional[str]
PathsOrNone = Union[str, List[str], None]


class DatasetResolver(ABC):
    """Resolve per-frame image / lidar source paths from config."""

    @abstractmethod
    def resolve_image(self, frame_id: int) -> PathOrNone:
        ...

    @abstractmethod
    def resolve_lidar(self, frame_id: int) -> PathsOrNone:
        ...

    @abstractmethod
    def list_available_frames(self) -> List[int]:
        ...

class OSDaRResolver(DatasetResolver):
    """
    OSDaR23: ``{counter}_{timestamp}.png`` / ``{counter}_{timestamp}.pcd``.

    When multiple files share the same counter, pick one by ``osdar_duplicate_policy``:
    ``latest`` (default) = lexicographically last filename, ``earliest`` = first.
    """

    _PREFIX_RE = re.compile(r"^(\d+)_")

    def __init__(self, config: Dict[str, Any]) -> None:
        data = config.get("data", {})
        self._image_dir = str(data.get("image_dir", "") or "").strip()
        self._velo_dir = str(data.get("velodyne_dir", "") or "").strip()
        policy = str(data.get("osdar_duplicate_policy", "latest") or "latest").lower()
        self._duplicate_policy = "earliest" if policy == "earliest" else "latest"

    def _pick_one(self, paths: List[str]) -> Optional[str]:
        if not paths:
            return None
        paths = sorted(paths)
        if self._duplicate_policy == "earliest":
            return paths[0]
        return paths[-1]

    def _resolve_by_prefix_int(self, directory: str, frame_id: int, exts: tuple[str, ...]) -> Optional[str]:
        """
        Resolve OSDaR23 files robustly:
        filenames are `{counter}_{timestamp}.*` where `counter` may have leading zeros (e.g., `012_...`).
        We match by parsing the numeric prefix and comparing `int(prefix) == frame_id`.
        """
        if not directory or not os.path.isdir(directory):
            return None
        fid = int(frame_id)
        candidates: List[str] = []
        for name in os.listdir(directory):
            low = name.lower()
            if not any(low.endswith(ext) for ext in exts):
                continue
            m = self._PREFIX_RE.match(name)
            if not m:
                continue
            try:
                if int(m.group(1)) != fid:
                    continue
            except ValueError:
                continue
            candidates.append(os.path.join(directory, name))
        return self._pick_one(candidates)

    def resolve_image(self, frame_id: int) -> PathOrNone:
        return self._resolve_by_prefix_int(self._image_dir, frame_id, (".png", ".jpg", ".jpeg"))

    def resolve_lidar(self, frame_id: int) -> PathsOrNone:
        return self._resolve_by_prefix_int(self._velo_dir, frame_id, (".pcd",))

    def list_available_frames(self) -> List[int]:
        if not self._image_dir or not os.path.isdir(self._image_dir):
            return []
        ids = set()
        for name in os.listdir(self._image_dir):
            m = self._PREFIX_RE.match(name)
            if not m:
                continue
            try:
                ids.add(int(m.group(1)))
            except ValueError:
                continue
        return sorted(ids)

