#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

from pipeline.optimizer_env_adapter import build_optimizer_env


class OptimizerConstraintAdapter(Protocol):
    @property
    def name(self) -> str:
        ...

    def build_env(self, config: Dict[str, Any], base_env: Dict[str, str]) -> Tuple[Dict[str, str], bool]:
        ...


@dataclass
class DefaultOptimizerConstraintAdapter:
    """默认约束适配器：保持现有 ab_experiment -> ENV 行为。"""

    @property
    def name(self) -> str:
        return "legacy_env_constraints"

    def build_env(self, config: Dict[str, Any], base_env: Dict[str, str]) -> Tuple[Dict[str, str], bool]:
        calibration_cfg = config.get("calibration", {})
        return build_optimizer_env(calibration_cfg, base_env)


def get_optimizer_constraint_adapter(config: Dict[str, Any]) -> OptimizerConstraintAdapter:
    adapter_name = config.get("calibration", {}).get("constraint_adapter", "legacy_env_constraints")
    if adapter_name == "legacy_env_constraints":
        return DefaultOptimizerConstraintAdapter()
    raise ValueError(f"未知优化器约束适配器: {adapter_name}")