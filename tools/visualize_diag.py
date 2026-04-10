#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 7：BEV / 语义对齐 / 精修 诊断图（由 visualize.py 调用）。"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m, M = float(np.nanmin(x)), float(np.nanmax(x))
    if M - m < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - m) / (M - m), 0.0, 1.0)


def render_bev_panel(lidar_base: str, image_feat_frame: str, out_path: str) -> bool:
    """A：LiDAR BEV rail、图像 pseudo-BEV rail、叠加。"""
    npz_l = lidar_base + "_bev_maps.npz"
    if not os.path.isfile(npz_l):
        print(f"[diag_bev] missing {npz_l}")
        return False
    zl = np.load(npz_l)
    rail_l = None
    for key in ("rail_probability", "rail"):
        if key in zl.files:
            rail_l = np.asarray(zl[key], dtype=np.float32)
            break
    if rail_l is None:
        print("[diag_bev] no rail channel in bev_maps.npz")
        return False
    pb = os.path.join(image_feat_frame, "pseudo_bev.npz")
    if not os.path.isfile(pb):
        print(f"[diag_bev] missing {pb}")
        return False
    z = np.load(pb)
    if "rail" not in z.files:
        print("[diag_bev] pseudo_bev.npz has no 'rail'")
        return False
    rail_i = np.asarray(z["rail"], dtype=np.float32)
    h, w = rail_l.shape[:2]
    if rail_i.shape[:2] != (h, w):
        rail_i = cv2.resize(rail_i, (w, h), interpolation=cv2.INTER_LINEAR)
    a = (_norm01(rail_l) * 255.0).astype(np.uint8)
    b = (_norm01(rail_i) * 255.0).astype(np.uint8)
    a3 = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    b3 = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    ov = cv2.addWeighted(a3, 0.5, b3, 0.5, 0.0)
    panel = np.hstack([a3, b3, ov])
    cv2.putText(panel, "LiDAR rail BEV", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(panel, "Image pseudo-BEV", (w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(panel, "Overlay 0.5/0.5", (2 * w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, panel)
    print(f"[diag_bev] wrote {out_path}")
    return True


def _find_semantic_argmax(sam_frame_dir: str, sam_prefix: str) -> Optional[np.ndarray]:
    cands: List[str] = []
    if os.path.isdir(sam_frame_dir):
        cands.extend(
            [
                os.path.join(sam_frame_dir, "semantic_argmax.png"),
                os.path.join(sam_frame_dir, "semantic_argmax.jpg"),
            ]
        )
    cands.extend(glob.glob(sam_prefix + "_semantic_argmax.png"))
    cands.extend(glob.glob(sam_prefix + "_semantic_map.png"))
    for p in cands:
        if p and os.path.isfile(p):
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is not None and im.size > 0:
                return im
    return None


def _load_semantic_points(path: str) -> List[Tuple[np.ndarray, int]]:
    out: List[Tuple[np.ndarray, int]] = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 9:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                sid = int(float(parts[-1]))
            except ValueError:
                continue
            out.append((np.array([x, y, z], dtype=np.float64), sid))
    return out


def render_semantic_panel(
    img_bgr: np.ndarray,
    feature_base: str,
    sam_frame_dir: str,
    sam_prefix: str,
    K: np.ndarray,
    R_rect: np.ndarray,
    P_rect: np.ndarray,
    R: np.ndarray,
    t_vec: np.ndarray,
    out_path: str,
) -> bool:
    """B：语义 argmax、LiDAR 投影语义 scatter、不一致热图（近似）。"""
    argmax = _find_semantic_argmax(sam_frame_dir, sam_prefix)
    if argmax is None:
        print("[diag_semantic] semantic_argmax / semantic_map not found")
        return False
    if argmax.ndim == 2:
        arg_u8 = (np.clip(argmax.astype(np.float32), 0, 65535) / 256.0).astype(np.uint8)
        arg_color = cv2.applyColorMap(arg_u8, cv2.COLORMAP_JET)
        lab_map = argmax.astype(np.int32)
    elif argmax.ndim == 3:
        arg_color = argmax[:, :, :3].copy()
        lab_map = np.argmax(argmax, axis=2).astype(np.int32)
    else:
        print("[diag_semantic] unsupported argmax shape")
        return False
    h, w = img_bgr.shape[:2]
    if arg_color.shape[0] != h or arg_color.shape[1] != w:
        arg_color = cv2.resize(arg_color, (w, h), interpolation=cv2.INTER_NEAREST)
        lab_map = cv2.resize(lab_map.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    pts = _load_semantic_points(feature_base + "_semantic_points.txt")
    if not pts:
        pts = []
        pf = feature_base + "_points.txt"
        if os.path.isfile(pf):
            with open(pf, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    parts = s.split()
                    if len(parts) < 8:
                        continue
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        lab = int(float(parts[7])) if len(parts) > 7 else 0
                    except ValueError:
                        continue
                    pts.append((np.array([x, y, z], dtype=np.float64), lab))

    scatter = np.zeros((h, w), dtype=np.float32)
    mismatch = np.zeros((h, w), dtype=np.float32)
    lid_overlay = img_bgr.copy()

    def proj(p: np.ndarray) -> Optional[Tuple[int, int]]:
        p_cam = R @ p + t_vec
        p_rect = R_rect @ p_cam
        if p_rect[2] < 0.1:
            return None
        uv = P_rect @ np.hstack([p_rect, 1.0])
        u = int(uv[0] / uv[2])
        v = int(uv[1] / uv[2])
        if 0 <= u < w and 0 <= v < h:
            return u, v
        return None

    for p, sid in pts:
        uv = proj(p)
        if uv is None:
            continue
        u, v = uv
        hue = int(180 * (abs(sid) % 16) / 16.0)
        cv2.circle(lid_overlay, (u, v), 2, (int(hue), 200, 200), -1)
        scatter[v, u] = max(scatter[v, u], 1.0)
        img_lab = int(lab_map[v, u])
        if img_lab > 0 and sid > 0 and img_lab != sid:
            mismatch[v, u] = 1.0

    mm = cv2.GaussianBlur(mismatch, (21, 21), 0)
    mm_n = _norm01(mm)
    heat = (mm_n * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)
    panel = np.hstack([arg_color, lid_overlay, heat_bgr])
    cv2.putText(panel, "Image argmax/labels", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel, "LiDAR semantic proj", (w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel, "Mismatch heat", (2 * w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, panel)
    print(f"[diag_semantic] wrote {out_path}")
    return True


def render_refine_curves(refinement_dir: str, out_path: str) -> bool:
    """C：轨迹与可观测性 / 标定分曲线。"""
    state_path = os.path.join(refinement_dir, "state.json")
    if not os.path.isfile(state_path):
        print(f"[diag_refine] missing {state_path}")
        return False
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    hist = st.get("history") or []
    if len(hist) < 1:
        print("[diag_refine] empty history")
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[diag_refine] matplotlib not installed")
        return False

    frames = [int(h["frame_id"]) for h in hist]
    obs = [float(h.get("observability", 0.0)) for h in hist]
    scores = [h.get("calib_score") for h in hist]
    scores_f = [float(s) if s is not None else float("nan") for s in scores]
    rxs = [float(h["rvec"][0]) for h in hist]
    rxf = [
        float(h["refined_rvec"][0]) if h.get("refined_rvec") else float(h["rvec"][0]) for h in hist
    ]

    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(frames, rxf, "g.-", label="refined rx")
    ax[0].plot(frames, rxs, "b.--", alpha=0.5, label="semantic rx")
    ax[0].set_ylabel("rx")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(frames, obs, "m.-")
    ax[1].set_ylabel("observability")
    ax[1].grid(True, alpha=0.3)
    ax[2].plot(frames, scores_f, "c.-")
    ax[2].set_ylabel("calib_score")
    ax[2].set_xlabel("frame_id")
    ax[2].grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[diag_refine] wrote {out_path}")
    return True


def render_rail_panel(sam_prefix: str, out_path: str) -> bool:
    """Phase 7 (sam_2d): rail_region、rail_centerline、rail_dist 三联图。"""
    paths = [
        (sam_prefix + "_rail_region.png", "region", False),
        (sam_prefix + "_rail_centerline.png", "centerline", False),
        (sam_prefix + "_rail_dist.png", "dist", True),
    ]
    for p, _label, _ in paths:
        if not os.path.isfile(p):
            print(f"[diag_rail] missing {p}")
            return False
    imgs: List[np.ndarray] = []
    for p, _label, is_dist in paths:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None or im.size == 0:
            print(f"[diag_rail] cannot read {p}")
            return False
        if im.ndim == 2:
            g = im.astype(np.float32)
        elif im.ndim == 3 and im.shape[2] >= 3:
            g = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            g = im[:, :, 0].astype(np.float32)
        if is_dist:
            g = _norm01(g) * 255.0
        else:
            g = np.clip(g, 0.0, 255.0)
        g_u8 = g.astype(np.uint8)
        imgs.append(cv2.cvtColor(g_u8, cv2.COLOR_GRAY2BGR))
    h = min(x.shape[0] for x in imgs)
    w = min(x.shape[1] for x in imgs)
    resized = [cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in imgs]
    panel = np.hstack(resized)
    cv2.putText(panel, "rail_region", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(panel, "rail_centerline", (w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(panel, "rail_dist", (2 * w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, panel)
    print(f"[diag_rail] wrote {out_path}")
    return True
