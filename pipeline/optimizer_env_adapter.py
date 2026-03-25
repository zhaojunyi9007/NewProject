#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build optimizer env from calibration config without changing behavior."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple


ENV_MAP = {
    "opt_translation": "EDGECALIB_OPT_TRANSLATION",
    "use_line_constraint": "EDGECALIB_USE_LINE_CONSTRAINT",
    "line_match_threshold": "EDGECALIB_LINE_MATCH_THRESHOLD",
    "line_soft_penalty": "EDGECALIB_LINE_SOFT_PENALTY",
    "line_soft_cap": "EDGECALIB_LINE_SOFT_CAP",
    "t_prior_weight": "EDGECALIB_T_PRIOR_WEIGHT",
    "w_consistency": "EDGECALIB_W_CONSISTENCY",
    "coarse_ty_range": "EDGECALIB_COARSE_TY_RANGE",
    "coarse_tz_range": "EDGECALIB_COARSE_TZ_RANGE",
    "coarse_tx_range": "EDGECALIB_COARSE_TX_RANGE",
    "coarse_ty_step": "EDGECALIB_COARSE_TY_STEP",
    "coarse_tz_step": "EDGECALIB_COARSE_TZ_STEP",
    "coarse_tx_step": "EDGECALIB_COARSE_TX_STEP",
    "log_line_debug": "EDGECALIB_LOG_LINE_DEBUG",
}


def build_optimizer_env(calibration_cfg: Dict[str, Any], base_env: Dict[str, str] | None = None) -> Tuple[Dict[str, str], bool]:
    env = dict(base_env or os.environ.copy())
    for env_key in ENV_MAP.values():
        env.pop(env_key, None)
    ab_cfg = calibration_cfg.get("ab_experiment", {}) if isinstance(calibration_cfg, dict) else {}
    applied = False
    for key, env_key in ENV_MAP.items():
        if key in ab_cfg and ab_cfg[key] is not None:
            env[env_key] = str(ab_cfg[key])
            applied = True
    return env, applied