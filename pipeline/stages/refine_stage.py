#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多帧滑动窗口时间精修（Phase 6）。"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

from pipeline.context import RuntimeContext

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from pipeline.observability import compute_frame_observability
from pipeline.refinement_state import (
    RefinementState,
    append_frame_result,
    clamp_pose_delta,
    get_active_window,
    load_state,
    mean_pose_window,
    save_state,
    should_update,
    temporal_blend,
)


def _parse_calib_result(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    kv: Dict[str, str] = {}
    breakdown: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if ":" in s:
                k, v = s.split(":", 1)
                kv[k.strip()] = v.strip()
                # capture numeric breakdown for debug
                vv = v.strip().split()[0] if v.strip() else ""
                if vv:
                    try:
                        breakdown[k.strip()] = float(vv)
                    except ValueError:
                        pass
    if "r" not in kv or "t" not in kv:
        return None
    try:
        r = [float(x) for x in kv["r"].split()[:3]]
        t = [float(x) for x in kv["t"].split()[:3]]
    except ValueError:
        return None
    if len(r) != 3 or len(t) != 3:
        return None
    score = breakdown.get("Score")
    return {
        "rvec": r,
        "tvec": t,
        "score": score,
        "breakdown": breakdown,
    }


def _save_trajectory_png(
    path: str,
    frames: List[int],
    r_raw: List[List[float]],
    r_ref: List[List[float]],
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if len(frames) < 2:
        return False
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    labels = ["rx", "ry", "rz"]
    for k in range(3):
        ax[k].plot(frames, [r[k] for r in r_raw], "o-", label="semantic/calib", alpha=0.7)
        ax[k].plot(frames, [r[k] for r in r_ref], "s-", label="refined", alpha=0.8)
        ax[k].set_ylabel(labels[k])
        ax[k].grid(True, alpha=0.3)
    ax[-1].set_xlabel("frame_id")
    ax[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段 refine] 多帧滑动窗口精修 (Phase 6)")
    print("=" * 40)

    out = context.paths.get("refinement") or context.config.get("data", {}).get("refinement_output_dir", "")
    if not out:
        print("[Error] 缺少 refinement 输出目录")
        return
    os.makedirs(out, exist_ok=True)

    sys.path.insert(0, os.path.join(_REPO_ROOT, "tools"))
    from score_breakdown_io import write_unified_debug_json  # noqa: E402

    cfg = context.config.get("refine") or {}
    window_size = int(cfg.get("window_size", 5) or 5)
    min_obs = float(cfg.get("min_observability_score", 0.35))
    max_deg = float(cfg.get("max_pose_jump_deg", 0.8))
    max_m = float(cfg.get("max_pose_jump_m", 0.15))
    lam = float(cfg.get("temporal_smoothing_lambda", 0.4))
    stride = int(cfg.get("update_stride", 1) or 1)

    state_path = os.path.join(out, "state.json")
    state = load_state(state_path) or RefinementState()

    lidar_dir = context.config["data"]["lidar_output_dir"]
    image_features_dir = context.config["data"]["image_features_output_dir"]
    calib_dir = context.config["data"]["calib_output_dir"]

    prev_refined_r: Optional[List[float]] = None
    prev_refined_t: Optional[List[float]] = None
    if state.history:
        last = state.history[-1]
        if last.refined_rvec and last.refined_tvec:
            prev_refined_r = list(last.refined_rvec)
            prev_refined_t = list(last.refined_tvec)

    all_frames: List[int] = []
    r_raw_series: List[List[float]] = []
    r_ref_series: List[List[float]] = []

    for frame_id in sorted(context.frame_ids):
        fid = f"{frame_id:010d}"
        calib_path = os.path.join(calib_dir, f"{fid}_calib_result.txt")
        parsed = _parse_calib_result(calib_path)
        if not parsed:
            print(f"[Warning] 无标定结果，跳过帧 {fid}")
            continue

        obs, dbg = compute_frame_observability(
            frame_id, lidar_dir=lidar_dir, image_features_dir=image_features_dir, config=context.config
        )
        br = parsed.get("breakdown") or {}

        append_frame_result(
            state,
            frame_id,
            parsed["rvec"],
            parsed["tvec"],
            calib_score=parsed.get("score"),
            score_breakdown={k: float(v) for k, v in br.items() if v is not None},
            observability=obs,
        )

        window = get_active_window(state, window_size)
        r_sem = parsed["rvec"]
        t_sem = parsed["tvec"]

        do_update = should_update(obs, min_obs, frame_id, state.last_update_frame_id, stride)

        if do_update and len(window) >= 1:
            r_mean, t_mean = mean_pose_window(window)
            if prev_refined_r is not None and prev_refined_t is not None:
                refined_r, refined_t = temporal_blend(r_mean, t_mean, prev_refined_r, prev_refined_t, lam)
            else:
                refined_r, refined_t = list(r_mean), list(t_mean)

            if state.last_success_frame_id is None:
                state.anchor_rvec = list(r_sem)
                state.anchor_tvec = list(t_sem)

            refined_r, refined_t = clamp_pose_delta(
                refined_r, refined_t, state.anchor_rvec, state.anchor_tvec, max_deg, max_m
            )
            if prev_refined_r is not None and prev_refined_t is not None:
                refined_r, refined_t = clamp_pose_delta(
                    refined_r, refined_t, prev_refined_r, prev_refined_t, max_deg, max_m
                )

            state.last_update_frame_id = frame_id
            state.last_success_frame_id = frame_id
            state.anchor_rvec = list(refined_r)
            state.anchor_tvec = list(refined_t)
        else:
            if prev_refined_r is not None and prev_refined_t is not None:
                refined_r, refined_t = list(prev_refined_r), list(prev_refined_t)
            else:
                refined_r, refined_t = list(r_sem), list(t_sem)

        for e in state.history:
            if e.frame_id == frame_id:
                e.refined_rvec = list(refined_r)
                e.refined_tvec = list(refined_t)
                e.observability = obs
                break

        prev_refined_r = list(refined_r)
        prev_refined_t = list(refined_t)

        save_state(state_path, state)

        win_ids = [e.frame_id for e in window] if window else [frame_id]
        debug = {
            "frame_id": frame_id,
            "window_frame_ids": win_ids,
            "observability": obs,
            "observability_debug": dbg,
            "refined_rvec": refined_r,
            "refined_tvec": refined_t,
            "semantic_rvec": r_sem,
            "semantic_tvec": t_sem,
            "params": {
                "window_size": window_size,
                "min_observability_score": min_obs,
                "max_pose_jump_deg": max_deg,
                "max_pose_jump_m": max_m,
                "temporal_smoothing_lambda": lam,
                "update_stride": stride,
            },
        }
        with open(os.path.join(out, f"{fid}_window_debug.json"), "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2, ensure_ascii=False)

        bd = dict(parsed.get("breakdown") or {})
        if parsed.get("score") is not None:
            bd["calib_score"] = float(parsed["score"])
        write_unified_debug_json(
            os.path.join(out, f"{fid}_debug_score_breakdown.json"),
            stage="refine",
            frame_id=fid,
            input_pose={"rvec": list(r_sem), "tvec": list(t_sem)},
            output_pose={"rvec": list(refined_r), "tvec": list(refined_t)},
            breakdown=bd if bd else None,
            observability=obs,
            meta={"window_debug": f"{fid}_window_debug.json"},
        )

        with open(os.path.join(out, f"{fid}_window_pose.txt"), "w", encoding="utf-8") as f:
            f.write("# refined rvec (3)\n")
            f.write(f"{refined_r[0]} {refined_r[1]} {refined_r[2]}\n")
            f.write("# refined tvec (3)\n")
            f.write(f"{refined_t[0]} {refined_t[1]} {refined_t[2]}\n")

        all_frames.append(frame_id)
        r_raw_series.append(list(r_sem))
        r_ref_series.append(list(refined_r))

        print(f"  frame={fid} observability={obs:.3f} refined_rx={refined_r[0]:.4f}")

    context.current_pose_refined = {"rvec": prev_refined_r, "tvec": prev_refined_t} if prev_refined_r else None

    png_path = os.path.join(out, f"{context.frame_ids[-1]:010d}_trajectory.png") if context.frame_ids else None
    if png_path and len(all_frames) >= 2:
        if _save_trajectory_png(png_path, all_frames, r_raw_series, r_ref_series):
            print(f"[Info] 轨迹图: {png_path}")
        else:
            print("[Info] 未安装 matplotlib，跳过 trajectory png")

    print(f"\n[完成] refinement 状态: {state_path}")
    print(f"输出目录: {out}")
