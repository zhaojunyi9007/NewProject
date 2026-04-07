#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多帧滑动窗口精修状态：持久化、窗口查询与更新策略（Phase 6）。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FrameRefinementEntry:
    frame_id: int
    rvec: List[float]
    tvec: List[float]
    calib_score: Optional[float] = None
    semantic_js: Optional[float] = None
    semantic_hist: Optional[float] = None
    edge_term: Optional[float] = None
    observability: float = 0.0
    refined_rvec: Optional[List[float]] = None
    refined_tvec: Optional[List[float]] = None


@dataclass
class RefinementState:
    version: int = 1
    anchor_rvec: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    anchor_tvec: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    last_success_frame_id: Optional[int] = None
    last_update_frame_id: Optional[int] = None
    history: List[FrameRefinementEntry] = field(default_factory=list)


def _entry_from_dict(d: Dict[str, Any]) -> FrameRefinementEntry:
    return FrameRefinementEntry(
        frame_id=int(d["frame_id"]),
        rvec=[float(x) for x in d["rvec"]],
        tvec=[float(x) for x in d["tvec"]],
        calib_score=_f(d.get("calib_score")),
        semantic_js=_f(d.get("semantic_js")),
        semantic_hist=_f(d.get("semantic_hist")),
        edge_term=_f(d.get("edge_term")),
        observability=float(d.get("observability", 0.0)),
        refined_rvec=_v3(d.get("refined_rvec")),
        refined_tvec=_v3(d.get("refined_tvec")),
    )


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _v3(x: Any) -> Optional[List[float]]:
    if x is None or not isinstance(x, (list, tuple)) or len(x) < 3:
        return None
    return [float(x[0]), float(x[1]), float(x[2])]


def load_state(path: str) -> Optional[RefinementState]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    hist = raw.get("history") or []
    entries = [_entry_from_dict(x) for x in hist if isinstance(x, dict)]
    st = RefinementState(
        version=int(raw.get("version", 1)),
        anchor_rvec=_v3(raw.get("anchor_rvec")) or [0.0, 0.0, 0.0],
        anchor_tvec=_v3(raw.get("anchor_tvec")) or [0.0, 0.0, 0.0],
        last_success_frame_id=raw.get("last_success_frame_id"),
        last_update_frame_id=raw.get("last_update_frame_id"),
        history=entries,
    )
    if st.last_success_frame_id is not None:
        st.last_success_frame_id = int(st.last_success_frame_id)
    if st.last_update_frame_id is not None:
        st.last_update_frame_id = int(st.last_update_frame_id)
    return st


def save_state(path: str, state: RefinementState) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = asdict(state)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_frame_result(
    state: RefinementState,
    frame_id: int,
    rvec: List[float],
    tvec: List[float],
    *,
    calib_score: Optional[float] = None,
    score_breakdown: Optional[Dict[str, float]] = None,
    observability: float = 0.0,
    refined_rvec: Optional[List[float]] = None,
    refined_tvec: Optional[List[float]] = None,
) -> None:
    """追加或替换同 frame_id 的记录（保持按 frame_id 递增时可去重）。"""
    sb = score_breakdown or {}
    entry = FrameRefinementEntry(
        frame_id=int(frame_id),
        rvec=list(rvec),
        tvec=list(tvec),
        calib_score=calib_score,
        semantic_js=sb.get("semantic_js"),
        semantic_hist=sb.get("semantic_hist"),
        edge_term=sb.get("edge_term"),
        observability=float(observability),
        refined_rvec=list(refined_rvec) if refined_rvec is not None else None,
        refined_tvec=list(refined_tvec) if refined_tvec is not None else None,
    )
    for i, e in enumerate(state.history):
        if e.frame_id == frame_id:
            state.history[i] = entry
            return
    state.history.append(entry)


def get_active_window(state: RefinementState, window_size: int) -> List[FrameRefinementEntry]:
    """取 history 末尾 window_size 条（已有时间顺序假定）。"""
    w = max(1, int(window_size))
    return state.history[-w:] if state.history else []


def should_update(
    observability: float,
    min_observability_score: float,
    frame_id: int,
    last_update_frame_id: Optional[int],
    update_stride: int,
) -> bool:
    if observability < float(min_observability_score):
        return False
    stride = max(1, int(update_stride))
    if last_update_frame_id is None:
        return True
    return (frame_id - int(last_update_frame_id)) >= stride


def mean_pose_window(entries: List[FrameRefinementEntry]) -> tuple[List[float], List[float]]:
    """简单算术平均（rvec/tvec）；条目至少 1。"""
    n = len(entries)
    if n == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    rs = [0.0, 0.0, 0.0]
    ts = [0.0, 0.0, 0.0]
    for e in entries:
        for k in range(3):
            rs[k] += e.rvec[k]
            ts[k] += e.tvec[k]
    for k in range(3):
        rs[k] /= n
        ts[k] /= n
    return rs, ts


def clamp_pose_delta(
    r_new: List[float],
    t_new: List[float],
    r_ref: List[float],
    t_ref: List[float],
    max_deg: float,
    max_m: float,
) -> tuple[List[float], List[float]]:
    """将相对 ref 的变化限制在 max_deg（轴角范数近似）与 max_m（平移范数）内。"""
    import math

    dr = [r_new[i] - r_ref[i] for i in range(3)]
    dt = [t_new[i] - t_ref[i] for i in range(3)]
    ang = math.sqrt(dr[0] ** 2 + dr[1] ** 2 + dr[2] ** 2)
    tm = math.sqrt(dt[0] ** 2 + dt[1] ** 2 + dt[2] ** 2)
    max_rad = math.radians(max_deg)
    sr = 1.0
    st = 1.0
    if ang > 1e-12 and ang > max_rad:
        sr = max_rad / ang
    if tm > 1e-12 and tm > max_m:
        st = max_m / tm
    r_out = [r_ref[i] + dr[i] * sr for i in range(3)]
    t_out = [t_ref[i] + dt[i] * st for i in range(3)]
    return r_out, t_out


def temporal_blend(
    r_mean: List[float],
    t_mean: List[float],
    r_prev: List[float],
    t_prev: List[float],
    lam: float,
) -> tuple[List[float], List[float]]:
    """refined = (1-lam)*mean + lam*prev"""
    a = float(lam)
    a = max(0.0, min(1.0, a))
    r = [(1.0 - a) * r_mean[i] + a * r_prev[i] for i in range(3)]
    t = [(1.0 - a) * t_mean[i] + a * t_prev[i] for i in range(3)]
    return r, t
