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
    "coarse_angle_range": "EDGECALIB_COARSE_ANGLE_RANGE",
    "coarse_angle_step": "EDGECALIB_COARSE_ANGLE_STEP",
    "coarse_ty_range": "EDGECALIB_COARSE_TY_RANGE",
    "coarse_tz_range": "EDGECALIB_COARSE_TZ_RANGE",
    "coarse_tx_range": "EDGECALIB_COARSE_TX_RANGE",
    "coarse_ty_step": "EDGECALIB_COARSE_TY_STEP",
    "coarse_tz_step": "EDGECALIB_COARSE_TZ_STEP",
    "coarse_tx_step": "EDGECALIB_COARSE_TX_STEP",
    "log_line_debug": "EDGECALIB_LOG_LINE_DEBUG",
    "fine_margin_switch_inimage": "EDGECALIB_FINE_MARGIN_SWITCH_INIMAGE",
    "fine_margin_low": "EDGECALIB_FINE_MARGIN_LOW",
    "fine_margin_high": "EDGECALIB_FINE_MARGIN_HIGH",
    "t_prior_disable_inimage_thresh": "EDGECALIB_T_PRIOR_DISABLE_INIMAGE_THRESH",
    "t_prior_weaken_inimage_thresh": "EDGECALIB_T_PRIOR_WEAKEN_INIMAGE_THRESH",
    "t_prior_weaken_factor": "EDGECALIB_T_PRIOR_WEAKEN_FACTOR",
    "t_prior_disable_oob_thresh": "EDGECALIB_T_PRIOR_DISABLE_OOB_THRESH",
    "t_prior_weaken_oob_thresh": "EDGECALIB_T_PRIOR_WEAKEN_OOB_THRESH",
    "t_prior_weaken_oob_factor": "EDGECALIB_T_PRIOR_WEAKEN_OOB_FACTOR",
}


def build_optimizer_env(calibration_cfg: Dict[str, Any], base_env: Dict[str, str] | None = None) -> Tuple[Dict[str, str], bool]:
    env = dict(base_env or os.environ.copy())
    for env_key in ENV_MAP.values():
        env.pop(env_key, None)

    # First pass: apply values from the main calibration config section.
    # This allows YAML keys like coarse_ty_range to reach the optimizer
    # without requiring the ab_experiment sub-section.
    if isinstance(calibration_cfg, dict):
        for key, env_key in ENV_MAP.items():
            val = calibration_cfg.get(key)
            if val is not None:
                env[env_key] = str(val)

    # Second pass: ab_experiment overrides have higher priority.
    ab_cfg = calibration_cfg.get("ab_experiment", {}) if isinstance(calibration_cfg, dict) else {}
    applied = False
    for key, env_key in ENV_MAP.items():
        if key in ab_cfg and ab_cfg[key] is not None:
            env[env_key] = str(ab_cfg[key])
            applied = True
    return env, applied