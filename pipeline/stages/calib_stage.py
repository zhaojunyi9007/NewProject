#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict

from pipeline.context import RuntimeContext
from pipeline.datasets import get_adapter
from pipeline.optimizer.constraint_adapter import get_optimizer_constraint_adapter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_OPTIMIZER_ARG_SUPPORT_CACHE: dict[tuple[str, str], bool] = {}


def _optimizer_binary_supports_arg(optimizer_path: str, arg_name: str) -> bool:
    key = (os.path.abspath(optimizer_path), arg_name)
    if key in _OPTIMIZER_ARG_SUPPORT_CACHE:
        return _OPTIMIZER_ARG_SUPPORT_CACHE[key]
    try:
        with open(optimizer_path, "rb") as f:
            data = f.read()
        supported = arg_name.encode("utf-8") in data
    except OSError:
        supported = True
    _OPTIMIZER_ARG_SUPPORT_CACHE[key] = supported
    return supported


def _parse_calib_breakdown(path: str) -> dict:
    out: dict = {}
    if not os.path.isfile(path):
        return {}
    string_keys = {
        "extrinsic_source",
        "final_gate_source",
        "invalid_reason",
        "optimizer_calib_file",
        "optimizer_intrinsics_source",
        "selected_init_source",
        "stage_a_reject_reason",
        "stage_b_guard_reason",
        "stage_b_switch_skipped_reason",
        "stage_b_track_skipped_reason",
        "strong_label_revert_reason",
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if ":" not in s:
                continue
            k, v = s.split(":", 1)
            k = k.strip()
            if k in {"r", "t"}:
                # Pose vectors are parsed by _parse_calib_pose; do not pollute breakdown with scalar fragments.
                continue
            raw_v = v.strip()
            if k in string_keys:
                if raw_v:
                    out[k] = raw_v
                continue
            v = raw_v.split()[0] if raw_v else ""
            if not v:
                continue
            try:
                out[k] = float(v)
            except ValueError:
                continue
    return out


def _stage_a_reject_reason(
    *,
    static_before: float,
    static_after: float,
    pose_jump_m: float,
    yaw_jump_deg: float,
    tolerance: float,
    max_pose_jump_m: float,
    max_yaw_jump_deg: float,
) -> str:
    if static_before <= 0.0:
        return "no_static_residuals"
    if static_after + max(0.0, tolerance) < static_before:
        return "static_score_degraded"
    if pose_jump_m > max_pose_jump_m:
        return "pose_jump_exceeded"
    if yaw_jump_deg > max_yaw_jump_deg:
        return "yaw_jump_exceeded"
    return "none"


def _extract_strong_label_debug(breakdown: dict) -> dict:
    keys = (
        "strong_label_feature_count",
        "strong_label_object_count",
        "strong_label_residual_count",
        "strong_residuals_added_to_optimizer",
        "strong_label_optimizer_residual_count",
        "strong_label_score_before_optimization",
        "strong_label_score_after_optimization",
        "static_score_before",
        "static_score_after_stage_a",
        "strong_label_pose_reverted",
        "strong_label_revert_reason",
        "stage_a_attempted",
        "stage_a_static_score_before",
        "stage_a_static_score_after",
        "stage_a_static_score_delta",
        "stage_a_pose_jump_m",
        "stage_a_yaw_jump_deg",
        "stage_a_rejected",
        "stage_a_reject_reason",
        "stage_b_track_used",
        "stage_b_switch_used",
        "stage_b_track_skipped_reason",
        "stage_b_switch_skipped_reason",
        "stage_b_reverted_to_stage_a",
        "stage_b_guard_reason",
        "strong_track_residual_count",
        "strong_pole_residual_count",
        "strong_switch_residual_count",
        "strong_buffer_stop_residual_count",
        "strong_track_optimizer_residual_count",
        "strong_switch_optimizer_residual_count",
        "strong_track_eval_residual_count",
        "strong_switch_eval_residual_count",
        "strong_label_score",
        "strong_track_score",
        "strong_pole_score",
        "strong_switch_score",
        "strong_buffer_stop_score",
        "strong_label_gate_used",
        "final_gate_source",
        "selected_init_source",
        "extrinsic_source",
        "openlabel_intrinsics_available",
        "optimizer_intrinsics_source",
        "intrinsics_cx_delta_px",
        "intrinsics_cy_delta_px",
        "calibration_intrinsics_cx_delta_px",
        "calibration_intrinsics_cy_delta_px",
        "optimizer_calib_file",
        "rail_bev_alignment_mismatch",
        "rail_mismatch_ignored_due_to_strong_label",
        "rail_weight_forced_low_due_to_mismatch",
        "pose_jump_from_initial_m",
        "yaw_jump_from_initial_deg",
    )
    return {k: breakdown.get(k) for k in keys if k in breakdown}


def _parse_calib_pose(path: str):
    if not os.path.isfile(path):
        return None
    kv: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or ":" not in s:
                continue
            k, v = s.split(":", 1)
            kv[k.strip()] = v.strip()
    if "r" not in kv or "t" not in kv:
        return None
    try:
        r = [float(x) for x in kv["r"].split()[:3]]
        t = [float(x) for x in kv["t"].split()[:3]]
    except ValueError:
        return None
    if len(r) != 3 or len(t) != 3:
        return None
    return {"rvec": r, "tvec": t}


def _angle_axis_delta_deg(a: list[float], b: list[float]) -> float:
    # Conservative approximation; sufficient as a gate/debug jump metric for small pose deltas.
    if len(a) != 3 or len(b) != 3:
        return 0.0
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3))) * 180.0 / math.pi


def _add_pose_jump_debug(breakdown: dict, pose_out: dict | None, init_r: list[float], init_t: list[float]) -> None:
    if not pose_out:
        return
    r = pose_out.get("rvec") or []
    t = pose_out.get("tvec") or []
    if len(r) == 3 and len(init_r) == 3:
        breakdown["rotation_jump_from_initial_deg"] = _angle_axis_delta_deg(r, init_r)
        breakdown["yaw_jump_from_initial_deg"] = abs(float(r[2]) - float(init_r[2])) * 180.0 / math.pi
    if len(t) == 3 and len(init_t) == 3:
        breakdown["pose_jump_from_initial_m"] = math.sqrt(sum((float(t[i]) - float(init_t[i])) ** 2 for i in range(3)))


def _load_json_dict(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _numbers_from_line(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]


def _read_osdar_homogeneous_transform(calib_file: str, camera_folder: str) -> list[list[float]]:
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    if not calib_file or not os.path.isfile(calib_file):
        return identity
    want = (camera_folder or "rgb_center").strip()
    try:
        with open(calib_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.rstrip() for ln in f.readlines()]
    except OSError:
        return identity
    in_cam = False
    cam_match = False
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "CAMERA":
            in_cam = True
            cam_match = False
            i += 1
            continue
        if in_cam and s.startswith("data_folder:"):
            cam_match = s.split(":", 1)[1].strip() == want
            i += 1
            continue
        if in_cam and cam_match and s.startswith("homogeneous transform:"):
            nums = _numbers_from_line(s[len("homogeneous transform:") :])
            j = i + 1
            while len(nums) < 16 and j < len(lines):
                nums.extend(_numbers_from_line(lines[j]))
                j += 1
            if len(nums) >= 16:
                return [nums[0:4], nums[4:8], nums[8:12], nums[12:16]]
            return identity
        i += 1
    return identity


def _prepare_optimizer_calib_file(
    config: Dict[str, Any],
    calib_file: str,
    calib_dir: str,
    frame_id: int,
    init_extrinsic_source: str,
) -> tuple[str, dict]:
    data_cfg = config.get("data") or {}
    fmt = str(data_cfg.get("dataset_format", "") or "").lower()
    cam = str(data_cfg.get("image_sensor", "rgb_center") or "rgb_center").strip() or "rgb_center"
    debug = {
        "openlabel_intrinsics_available": False,
        "optimizer_intrinsics_source": "calibration_txt" if calib_file and os.path.exists(calib_file) else "default",
        "intrinsics_cx_delta_px": 0.0,
        "intrinsics_cy_delta_px": 0.0,
        "calibration_intrinsics_cx_delta_px": 0.0,
        "calibration_intrinsics_cy_delta_px": 0.0,
    }
    if fmt not in {"osdar23", "osdar"} or init_extrinsic_source != "openlabel_coordinate_systems":
        return calib_file, debug

    try:
        from pipeline.datasets.osdar23 import (
            _resolve_openlabel_json,
            load_osdar23_intrinsics,
            load_osdar23_openlabel_intrinsics,
        )
    except Exception:
        return calib_file, debug

    label_json = _resolve_openlabel_json(config)
    K_openlabel = load_osdar23_openlabel_intrinsics(label_json, cam)
    if K_openlabel is None:
        return calib_file, debug
    debug["openlabel_intrinsics_available"] = True

    K_calib = None
    try:
        K_calib, _, _ = load_osdar23_intrinsics(calib_file, cam, label_json="")
    except Exception:
        K_calib = None
    if K_calib is not None:
        debug["calibration_intrinsics_cx_delta_px"] = float(K_calib[0, 2] - K_openlabel[0, 2])
        debug["calibration_intrinsics_cy_delta_px"] = float(K_calib[1, 2] - K_openlabel[1, 2])
        if abs(debug["calibration_intrinsics_cx_delta_px"]) > 1.0 or abs(debug["calibration_intrinsics_cy_delta_px"]) > 1.0:
            print(
                "[Warning] OSDaR intrinsics mismatch: "
                f"camera={cam}, cx_delta={debug['calibration_intrinsics_cx_delta_px']:.3f}px, "
                f"cy_delta={debug['calibration_intrinsics_cy_delta_px']:.3f}px; "
                "optimizer will use OpenLABEL stream intrinsics."
            )

    os.makedirs(calib_dir, exist_ok=True)
    out = os.path.join(calib_dir, f"{frame_id:010d}_openlabel_intrinsics_calib.txt")
    T = _read_osdar_homogeneous_transform(calib_file, cam)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by calib_stage.py; K comes from OpenLABEL streams.\n")
        f.write("CAMERA\n")
        f.write(f"data_folder: {cam}\n")
        f.write("camera_matrix:\n")
        f.write(f"[{K_openlabel[0, 0]:.15g}, {K_openlabel[0, 1]:.15g}, {K_openlabel[0, 2]:.15g};\n")
        f.write(f" {K_openlabel[1, 0]:.15g}, {K_openlabel[1, 1]:.15g}, {K_openlabel[1, 2]:.15g};\n")
        f.write(f" {K_openlabel[2, 0]:.15g}, {K_openlabel[2, 1]:.15g}, {K_openlabel[2, 2]:.15g}]\n")
        f.write("homogeneous transform:\n")
        f.write(f"[{T[0][0]:.15g}, {T[0][1]:.15g}, {T[0][2]:.15g}, {T[0][3]:.15g};\n")
        f.write(f" {T[1][0]:.15g}, {T[1][1]:.15g}, {T[1][2]:.15g}, {T[1][3]:.15g};\n")
        f.write(f" {T[2][0]:.15g}, {T[2][1]:.15g}, {T[2][2]:.15g}, {T[2][3]:.15g};\n")
        f.write(f" {T[3][0]:.15g}, {T[3][1]:.15g}, {T[3][2]:.15g}, {T[3][3]:.15g}]\n")
    debug["optimizer_intrinsics_source"] = "openlabel_streams"
    debug["optimizer_calib_file"] = out
    debug["intrinsics_cx_delta_px"] = 0.0
    debug["intrinsics_cy_delta_px"] = 0.0
    return out, debug


def _read_label_teacher_points(path: str, xmax_m: float = 0.0) -> list[tuple[float, float, float, int, float]]:
    out: list[tuple[float, float, float, int, float]] = []
    if not path or not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                cls = int(parts[3])
                weight = float(parts[4])
            except ValueError:
                continue
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if x < 0.0:
                continue
            if xmax_m > 0.0 and x > xmax_m:
                continue
            out.append((x, y, z, cls, max(0.0, min(1.0, weight))))
    return out


def _read_xyz_points(path: str, limit: int = 4000) -> list[tuple[float, float, float]]:
    pts: list[tuple[float, float, float]] = []
    if not path or not os.path.isfile(path):
        return pts
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
            if len(pts) >= limit:
                break
    return pts


def _load_u16_float_map(path: str):
    if not path or not os.path.isfile(path):
        return None
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    m = m.astype(np.float32)
    if m.max() > 1.5:
        m /= 65535.0
    return np.clip(m, 0.0, 1.0)


def _project_xyz(p: tuple[float, float, float], rvec: list[float], tvec: list[float], R_rect, P_rect, width: int, height: int):
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    r = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(r)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    xyz = np.asarray(p, dtype=np.float64).reshape(3)
    pc = R @ xyz + t
    pr = R_rect @ pc
    if pr[2] <= 1e-6:
        return None
    uvw = P_rect @ np.array([pr[0], pr[1], pr[2], 1.0], dtype=np.float64)
    if abs(uvw[2]) <= 1e-9:
        return None
    u = int(round(float(uvw[0] / uvw[2])))
    v = int(round(float(uvw[1] / uvw[2])))
    if u < 0 or v < 0 or u >= width or v >= height:
        return None
    return u, v


def _pose_jump_values(rvec: list[float], tvec: list[float], init_r: list[float], init_t: list[float]) -> tuple[float, float]:
    pose_jump = math.sqrt(sum((float(tvec[i]) - float(init_t[i])) ** 2 for i in range(3))) if len(tvec) == 3 and len(init_t) == 3 else 0.0
    yaw_jump = abs(float(rvec[2]) - float(init_r[2])) * 180.0 / math.pi if len(rvec) == 3 and len(init_r) == 3 else 0.0
    return pose_jump, yaw_jump


def _clip_pose_to_label_gate(pose: dict, init_r: list[float], init_t: list[float], sem_cfg: dict) -> dict | None:
    r = list(pose.get("rvec") or [])
    t = list(pose.get("tvec") or [])
    if len(r) != 3 or len(t) != 3:
        return None
    max_t = float(sem_cfg.get("max_label_assisted_pose_jump_m", 1.0))
    max_yaw = float(sem_cfg.get("max_label_assisted_yaw_jump_deg", 3.0)) * math.pi / 180.0
    dt = [float(t[i]) - float(init_t[i]) for i in range(3)]
    norm = math.sqrt(sum(x * x for x in dt))
    if max_t > 0.0 and norm > max_t:
        scale = max_t / max(norm, 1e-9)
        t = [float(init_t[i]) + dt[i] * scale for i in range(3)]
    r = [float(x) for x in r]
    dyaw = r[2] - float(init_r[2])
    if max_yaw > 0.0 and abs(dyaw) > max_yaw:
        r[2] = float(init_r[2]) + math.copysign(max_yaw, dyaw)
    return {"rvec": r, "tvec": t}


def _score_initial_pose_candidate(name: str, pose: dict, init_r: list[float], init_t: list[float], feature_base: str, sam_base: str, ds, sem_cfg: dict, label_cfg: dict) -> dict:
    r = list(pose.get("rvec") or [])
    t = list(pose.get("tvec") or [])
    out = {"source": name, "label_teacher_score": 0.0, "label_track_visible_count": 0, "label_teacher_visible_ratio": 0.0, "edge_in_image_ratio": 0.0, "selection_score": -1e9}
    if len(r) != 3 or len(t) != 3:
        out["invalid"] = "bad_pose"
        return out
    pose_jump, yaw_jump = _pose_jump_values(r, t, init_r, init_t)
    out["pose_jump_m"] = pose_jump
    out["yaw_jump_deg"] = yaw_jump
    try:
        R_rect, P_rect = None, None
        K, R_rect, P_rect = ds.load_intrinsics()
        del K
    except Exception as e:
        out["invalid"] = f"intrinsics:{e}"
        return out
    maps = {
        1: (_load_u16_float_map(sam_base + "_label_track_dist.png"), _load_u16_float_map(sam_base + "_label_track_weight.png")),
        3: (_load_u16_float_map(sam_base + "_label_static_dist.png"), _load_u16_float_map(sam_base + "_label_static_weight.png")),
        5: (_load_u16_float_map(sam_base + "_label_vehicle_dist.png"), _load_u16_float_map(sam_base + "_label_vehicle_weight.png")),
        7: (_load_u16_float_map(sam_base + "_label_person_dist.png"), _load_u16_float_map(sam_base + "_label_person_weight.png")),
    }
    first_map = next((m for pair in maps.values() for m in pair if m is not None), None)
    if first_map is None:
        out["invalid"] = "missing_label_maps"
        return out
    height, width = first_map.shape[:2]
    xmax = float(label_cfg.get("teacher_visible_xmax_m", 120.0))
    label_pts = _read_label_teacher_points(feature_base + "_label_object_points.txt", xmax)
    class_scores = {"track": 0.0, "static": 0.0, "vehicle": 0.0, "person": 0.0}
    class_visible = {"track": 0, "static": 0, "vehicle": 0, "person": 0}
    class_total = {"track": 0, "static": 0, "vehicle": 0, "person": 0}
    class_name = {1: "track", 3: "static", 5: "vehicle", 7: "person"}
    for x, y, z, cls, weight in label_pts:
        cname = class_name.get(cls)
        if not cname:
            continue
        class_total[cname] += 1
        proj = _project_xyz((x, y, z), r, t, R_rect, P_rect, width, height)
        if proj is None:
            continue
        dist, wmap = maps.get(cls, (None, None))
        if dist is None:
            continue
        u, v = proj
        img_w = float(wmap[v, u]) if wmap is not None else 1.0
        if img_w <= 1e-4:
            continue
        class_visible[cname] += 1
        class_scores[cname] += max(0.0, 1.0 - float(dist[v, u])) * img_w * weight
    norm_scores = {k: (class_scores[k] / max(1, class_total[k])) for k in class_scores}
    weights = {
        "track": float(label_cfg.get("track_weight", 1.5)),
        "static": float(label_cfg.get("static_object_weight", 1.0)),
        "vehicle": float(label_cfg.get("vehicle_weight", 0.4)),
        "person": float(label_cfg.get("person_weight", 0.2)),
    }
    denom = max(1e-9, sum(weights.values()))
    label_score = sum(weights[k] * norm_scores[k] for k in norm_scores) / denom
    visible_total = sum(class_visible.values())
    eligible_total = sum(class_total.values())
    edge_pts = _read_xyz_points(feature_base + "_edge_points.txt", 3500)
    edge_in = 0
    for pt in edge_pts:
        if _project_xyz(pt, r, t, R_rect, P_rect, width, height) is not None:
            edge_in += 1
    edge_ratio = edge_in / max(1, len(edge_pts))
    max_pose_jump = float(sem_cfg.get("max_label_assisted_pose_jump_m", 1.0))
    max_yaw_jump = float(sem_cfg.get("max_label_assisted_yaw_jump_deg", 3.0))
    jump_penalty = 0.0
    if pose_jump > max_pose_jump:
        jump_penalty += (pose_jump - max_pose_jump) * 0.25
    if yaw_jump > max_yaw_jump:
        jump_penalty += (yaw_jump - max_yaw_jump) * 0.05
    selection_score = label_score + 0.25 * min(edge_ratio / 0.20, 1.0) - jump_penalty
    out.update({
        "label_teacher_score": label_score,
        "label_teacher_eligible_count": eligible_total,
        "label_teacher_visible_count": visible_total,
        "label_teacher_visible_ratio": visible_total / max(1, eligible_total),
        "label_track_visible_count": class_visible["track"],
        "edge_in_image_ratio": edge_ratio,
        "selection_score": selection_score,
        "label_teacher_score_by_class": norm_scores,
    })
    return out


def _select_initial_pose_candidate(frame_id: int, init_r: list[float], init_t: list[float], feature_base: str, sam_base: str, ds, sem_cfg: dict, label_cfg: dict, bev_by_frame: dict, bev_candidates_by_frame: dict, label_assist_for_calib: bool) -> tuple[list[float], list[float], str, bool, list[dict]]:
    if not label_assist_for_calib or not bool(sem_cfg.get("enable_init_candidate_scoring", False)):
        if frame_id in bev_by_frame:
            pose = bev_by_frame[frame_id]
            return list(pose["rvec"]), list(pose["tvec"]), "bev_accepted", False, []
        return list(init_r), list(init_t), "original", False, []
    candidates: list[dict] = [{"source": "original_init", "pose": {"rvec": list(init_r), "tvec": list(init_t)}, "rejected": False}]
    allow_bev_candidate = (not label_assist_for_calib) or bool(sem_cfg.get("use_bev_candidate_in_label_assist", False))
    if allow_bev_candidate and frame_id in bev_by_frame:
        candidates.append({"source": "bev_accepted", "pose": bev_by_frame[frame_id], "rejected": False})
    if allow_bev_candidate and frame_id in bev_candidates_by_frame:
        cand = bev_candidates_by_frame[frame_id]
        pose = cand.get("pose", {}) if isinstance(cand, dict) else {}
        if pose.get("rvec") and pose.get("tvec"):
            rejected = str(cand.get("source", "bev_raw_rejected")) != "bev_accepted"
            candidates.append({"source": str(cand.get("source", "bev_raw_rejected")), "pose": pose, "rejected": rejected})
            clipped = _clip_pose_to_label_gate(pose, init_r, init_t, sem_cfg)
            if clipped:
                candidates.append({"source": "clipped_bev_candidate", "pose": clipped, "rejected": rejected})
    scored: list[dict] = []
    for cand in candidates:
        score = _score_initial_pose_candidate(cand["source"], cand["pose"], init_r, init_t, feature_base, sam_base, ds, sem_cfg, label_cfg)
        score["bev_candidate_rejected_by_gate"] = bool(cand.get("rejected", False))
        scored.append(score)
    original_score = next((x for x in scored if x.get("source") == "original_init"), None)
    selectable = []
    max_pose_jump = float(sem_cfg.get("max_label_assisted_pose_jump_m", 1.0))
    max_yaw_jump = float(sem_cfg.get("max_label_assisted_yaw_jump_deg", 3.0))
    for cand, score in zip(candidates, scored):
        if score.get("invalid"):
            continue
        if cand["source"] == "original_init" or cand["source"] == "bev_accepted":
            selectable.append((cand, score))
            continue
        if score.get("pose_jump_m", 0.0) <= max_pose_jump and score.get("yaw_jump_deg", 0.0) <= max_yaw_jump:
            if not original_score or score.get("selection_score", -1e9) >= original_score.get("selection_score", -1e9):
                selectable.append((cand, score))
    if not selectable:
        return list(init_r), list(init_t), "original_init", False, scored
    selected_cand, _ = max(selectable, key=lambda item: float(item[1].get("selection_score", -1e9)))
    pose = selected_cand["pose"]
    return list(pose["rvec"]), list(pose["tvec"]), str(selected_cand["source"]), bool(selected_cand.get("rejected", False)), scored



def _is_oracle_rail_enabled(rail_quality: dict, min_quality: float) -> bool:
    return (
        bool(rail_quality.get("label_track_prior_used", False))
        and bool(rail_quality.get("enabled", False))
        and float(rail_quality.get("quality_score", 0.0) or 0.0) >= float(min_quality)
    )


def _write_invalid_calib_result(path: str, rvec: list[float], tvec: list[float], reason: str, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    extra = extra or {}
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"r: {rvec[0]} {rvec[1]} {rvec[2]}\n")
        f.write(f"t: {tvec[0]} {tvec[1]} {tvec[2]}\n")
        f.write("Score: 0\n")
        f.write("semantic_js_divergence: 0\n")
        f.write("semantic_hist_similarity: 0\n")
        f.write("edge_term_norm: 0\n")
        f.write("rail_term_norm: 0\n")
        f.write("final_pose_valid: 0\n")
        f.write("rail_gate_failed: 1\n")
        f.write("optimizer_skipped: 1\n")
        f.write(f"invalid_reason: {reason}\n")
        for k, v in extra.items():
            if isinstance(v, (str, int, float, bool)):
                f.write(f"{k}: {v}\n")
            elif isinstance(v, (list, tuple)):
                f.write(f"{k}: {' '.join(str(x) for x in v)}\n")


def _has_strong_label_feature_rows(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    return True
    except OSError:
        return False
    return False


def _has_rail_mismatch(rail_debug: dict, align_debug: dict) -> bool:
    if rail_debug and not bool(rail_debug.get("rail_refinement_valid", True)):
        return True
    if align_debug and not bool(align_debug.get("rail_bev_alignment_valid", True)):
        return True
    return False


def _strong_label_can_override_rail_mismatch(label_cfg: dict, strong_label_features_path: str) -> bool:
    return (
        bool(label_cfg.get("enabled", False))
        and bool(label_cfg.get("strong_features_enabled", True))
        and _has_strong_label_feature_rows(strong_label_features_path)
    )


def _should_skip_optimizer_for_rail_mismatch(
    sem_cfg: dict,
    oracle_rail: bool,
    rail_debug: dict,
    align_debug: dict,
    label_cfg: dict | None = None,
    strong_label_features_path: str = "",
) -> tuple[bool, str]:
    if not oracle_rail or not bool(sem_cfg.get("skip_optimizer_on_rail_refinement_mismatch", False)):
        return False, ""
    if _strong_label_can_override_rail_mismatch(label_cfg or {}, strong_label_features_path):
        return False, ""
    if _has_rail_mismatch(rail_debug, align_debug):
        return True, "rail_bev_alignment_mismatch"
    return False, ""


def _apply_final_rail_hard_gate(breakdown: dict, sem_cfg: dict, oracle_rail: bool) -> dict:
    out = dict(breakdown or {})
    out.setdefault("final_pose_valid", 1.0)
    out.setdefault("rail_gate_failed", 0.0)
    out.setdefault("invalid_reason", "")
    out.setdefault("final_gate_source", "rail_or_unsupervised")
    strong_count = float(out.get("strong_label_feature_count", 0.0) or 0.0)
    if strong_count > 0 and bool(sem_cfg.get("label_assist_requires_label_gate", True)):
        strong_score = float(out.get("strong_label_score", 0.0) or 0.0)
        strong_track_count = float(out.get("strong_track_residual_count", 0.0) or 0.0)
        strong_static_count = (
            float(out.get("strong_pole_residual_count", 0.0) or 0.0)
            + float(out.get("strong_switch_residual_count", 0.0) or 0.0)
            + float(out.get("strong_buffer_stop_residual_count", 0.0) or 0.0)
        )
        min_strong_score = float(sem_cfg.get("min_strong_label_score", 0.25))
        track_primary_gate = bool(sem_cfg.get("track_use_as_primary_gate", False))
        min_track = float(sem_cfg.get("min_strong_track_residual_count", 0 if not track_primary_gate else 50))
        min_static = float(sem_cfg.get("min_strong_static_residual_count", 2))
        max_pose_jump = float(sem_cfg.get("max_label_assisted_pose_jump_m", 1.0))
        max_yaw_jump = float(sem_cfg.get("max_label_assisted_yaw_jump_deg", 3.0))
        pose_jump = float(out.get("pose_jump_from_initial_m", 0.0) or 0.0)
        yaw_jump = float(out.get("yaw_jump_from_initial_deg", 0.0) or 0.0)
        strong_ok = (
            strong_score >= min_strong_score
            and ((not track_primary_gate) or strong_track_count >= min_track)
            and strong_static_count >= min_static
            and pose_jump <= max_pose_jump
            and yaw_jump <= max_yaw_jump
        )
        out["strong_label_gate_used"] = 1.0
        if strong_ok:
            out["final_pose_valid"] = 1.0
            out["rail_gate_failed"] = 0.0
            out["final_gate_source"] = "strong_label_assisted"
            out["invalid_reason"] = ""
        else:
            reasons = []
            if strong_score < min_strong_score:
                reasons.append(f"strong_label_score={strong_score:.6f}<{min_strong_score:.6f}")
            if track_primary_gate and strong_track_count < min_track:
                reasons.append(f"strong_track_residual_count={strong_track_count:.0f}<{min_track:.0f}")
            if strong_static_count < min_static:
                reasons.append(f"strong_static_residual_count={strong_static_count:.0f}<{min_static:.0f}")
            if pose_jump > max_pose_jump:
                reasons.append(f"pose_jump_from_initial_m={pose_jump:.6f}>{max_pose_jump:.6f}")
            if yaw_jump > max_yaw_jump:
                reasons.append(f"yaw_jump_from_initial_deg={yaw_jump:.6f}>{max_yaw_jump:.6f}")
            out["final_pose_valid"] = 0.0
            out["rail_gate_failed"] = 0.0
            out["final_gate_source"] = "strong_label_rejected"
            out["invalid_reason"] = ";".join(reasons)
            out["label_gate_failed_reason"] = out["invalid_reason"]
        return out
    if not oracle_rail or not bool(sem_cfg.get("oracle_rail_hard_gate", False)):
        out["final_gate_source"] = "no_oracle_rail_gate"
        return out
    min_count = float(sem_cfg.get("min_final_rail_visible_count", sem_cfg.get("min_rail_visible_count", 50)))
    min_ratio = float(sem_cfg.get("min_final_rail_visible_ratio", sem_cfg.get("min_rail_visible_ratio", 0.08)))
    visible_count = float(out.get("rail_visible_count", 0.0) or 0.0)
    visible_ratio = float(out.get("rail_visible_ratio", 0.0) or 0.0)
    failed = visible_count < min_count or visible_ratio < min_ratio
    if failed:
        out["rail_gate_failed"] = 1.0
        object_visible = float(out.get("object_visible_count", 0.0) or 0.0)
        edge_visible = float(out.get("edge_visible_count", 0.0) or 0.0)
        semantic_used = float(out.get("semantic_term_used", 0.0) or 0.0) > 0.0
        semantic_hist = float(out.get("semantic_hist_similarity", 0.0) or 0.0)
        semantic_js = float(out.get("semantic_js_divergence", 1.0) or 1.0)
        min_object = float(sem_cfg.get("min_final_object_visible_count", 1))
        min_edge = float(sem_cfg.get("min_final_edge_visible_count", sem_cfg.get("min_edge_visible_count", 50)))
        min_sem_hist = float(sem_cfg.get("min_final_semantic_hist_similarity", 0.05))
        max_sem_js = float(sem_cfg.get("max_final_semantic_js_divergence", 0.95))
        semantic_ok = semantic_used and (semantic_hist >= min_sem_hist or semantic_js <= max_sem_js)
        label_score = float(out.get("label_teacher_score", 0.0) or 0.0)
        label_track_visible = float(out.get("label_track_visible_count", 0.0) or 0.0)
        label_static_visible = float(out.get("label_static_visible_count", 0.0) or 0.0)
        label_residual_count = float(out.get("label_residual_count", 0.0) or 0.0)
        min_label_score = float(sem_cfg.get("min_label_teacher_score", 0.25))
        min_label_track_visible = float(sem_cfg.get("min_label_track_visible_count", 20))
        min_label_residual = float(sem_cfg.get("min_label_residual_count_for_gate", 50))
        max_pose_jump = float(sem_cfg.get("max_label_assisted_pose_jump_m", 1.0))
        max_yaw_jump = float(sem_cfg.get("max_label_assisted_yaw_jump_deg", 3.0))
        pose_jump = float(out.get("pose_jump_from_initial_m", 0.0) or 0.0)
        yaw_jump = float(out.get("yaw_jump_from_initial_deg", 0.0) or 0.0)
        label_ok = (
            label_residual_count >= min_label_residual
            and label_track_visible >= min_label_track_visible
            and label_score >= min_label_score
            and pose_jump <= max_pose_jump
            and yaw_jump <= max_yaw_jump
        )
        allow_unsupervised_alt = bool(sem_cfg.get("allow_object_or_edge_to_pass_rail_gate", True)) and (
            object_visible >= min_object or edge_visible >= min_edge or semantic_ok
        )
        if bool(sem_cfg.get("label_assist_requires_label_gate", True)) and label_residual_count > 0:
            allow_alt = label_ok
        else:
            allow_alt = allow_unsupervised_alt or label_ok
        out["object_or_edge_rail_gate_bypass"] = 1.0 if allow_unsupervised_alt else 0.0
        out["semantic_rail_gate_bypass_ok"] = 1.0 if semantic_ok else 0.0
        out["label_gate_bypass_used"] = 1.0 if label_ok and allow_alt else 0.0
        if label_ok and allow_alt:
            out["final_gate_source"] = "label_assisted"
        elif label_residual_count > 0:
            out["final_gate_source"] = "label_rejected"
        elif allow_unsupervised_alt:
            out["final_gate_source"] = "unsupervised_auxiliary"
        if not label_ok:
            reasons = []
            if label_residual_count < min_label_residual:
                reasons.append(f"label_residual_count={label_residual_count:.0f}<{min_label_residual:.0f}")
            if label_track_visible < min_label_track_visible:
                reasons.append(f"label_track_visible_count={label_track_visible:.0f}<{min_label_track_visible:.0f}")
            if label_score < min_label_score:
                reasons.append(f"label_teacher_score={label_score:.6f}<{min_label_score:.6f}")
            if pose_jump > max_pose_jump:
                reasons.append(f"pose_jump_from_initial_m={pose_jump:.6f}>{max_pose_jump:.6f}")
            if yaw_jump > max_yaw_jump:
                reasons.append(f"yaw_jump_from_initial_deg={yaw_jump:.6f}>{max_yaw_jump:.6f}")
            out["label_gate_failed_reason"] = ";".join(reasons)
        else:
            out["label_gate_failed_reason"] = ""
        if bool(sem_cfg.get("reject_pose_on_rail_gate_fail", True)) and not allow_alt:
            out["final_pose_valid"] = 0.0
            out["invalid_reason"] = (
                f"rail_visible_gate_failed(count={visible_count:.0f}<{min_count:.0f} "
                f"or ratio={visible_ratio:.6f}<{min_ratio:.6f})"
            )
        else:
            if not out.get("final_gate_source") or out.get("final_gate_source") == "label_rejected":
                out["final_gate_source"] = "rail_or_auxiliary_pass"
            out["invalid_reason"] = ""
    return out




def _load_calib_stage_extrinsic(ds, config: Dict[str, Any]):
    label_cfg = config.get("label_assist") or {}
    if bool(label_cfg.get("enabled", False)) and bool(label_cfg.get("use_openlabel_extrinsic", True)):
        loader = getattr(ds, "load_label_assist_extrinsic", None)
        if callable(loader):
            ext = loader()
            if ext:
                return ext, "openlabel_coordinate_systems"
            if not bool(label_cfg.get("openlabel_extrinsic_fallback_to_calibration", True)):
                return None, "openlabel_coordinate_systems_missing"
    loader = getattr(ds, "load_initial_extrinsic", None)
    return (loader(), "calibration_txt") if callable(loader) else (None, "config_initial_extrinsic")

def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段3] 两阶段标定优化")
    print("=" * 40)

    image_root = context.paths.get("image_features") if context.paths else ""
    sam_root = context.paths.get("sam") if context.paths else ""
    if not image_root:
        image_root = context.config["data"]["image_features_output_dir"]
    if not sam_root:
        sam_root = context.config["data"]["sam_output_dir"]
    lidar_dir = context.config["data"]["lidar_output_dir"]
    calib_dir = context.config["data"]["calib_output_dir"]
    calib_file = context.config["data"]["calib_file"]
    temporal_cfg = context.config.get("calibration", {}).get("temporal_validation", {})
    history_enabled = bool(temporal_cfg.get("enabled", False))
    history_cfg = temporal_cfg.get("history_file", "")
    history_file = ""
    if history_enabled and history_cfg:
        history_file = str(history_cfg)
        if not os.path.isabs(history_file):
            history_file = os.path.join(calib_dir, history_file)
        os.makedirs(os.path.dirname(history_file) or ".", exist_ok=True)

    init_r = context.config["calibration"]["initial_extrinsic"]["rotation"]
    init_t = context.config["calibration"]["initial_extrinsic"]["translation"]

    ds = get_adapter(context.config)
    ext, init_extrinsic_source = _load_calib_stage_extrinsic(ds, context.config)
    if ext:
        init_r, init_t = ext
        cam_folder = str(context.config.get("data", {}).get("image_sensor", "rgb_center") or "rgb_center")
        print(f"[Info] 使用初始外参: source={init_extrinsic_source}, camera={cam_folder}")
    else:
        init_extrinsic_source = "config_initial_extrinsic"

    bev_cfg = context.config.get("bev") or {}
    bev_by_frame = getattr(context, "bev_pose_by_frame", None) or {}
    bev_candidates_by_frame = getattr(context, "bev_candidate_by_frame", None) or {}
    sem_cfg = context.config.get("semantic_calib") or {}
    label_cfg = context.config.get("label_assist") or {}
    label_assist_for_calib = bool(label_cfg.get("enabled", False)) and bool(label_cfg.get("use_for_calib_residual", True))
    sem_enabled = bool(sem_cfg.get("enabled", False))
    allow_legacy_fallback = bool(sem_cfg.get("allow_legacy_fallback", False))
    img_cfg = context.config.get("image_features") or {}
    sem_classes = list(img_cfg.get("semantic_classes", []) or [])
    cw = sem_cfg.get("class_weights") or {}
    class_weights = [float(cw.get(name, 1.0)) for name in sem_classes]
    pyr = sem_cfg.get("pyramid_scales") or [1.0, 0.5, 0.25]
    pyramid_scales_s = ",".join(str(float(x)) for x in pyr)
    class_weights_s = ",".join(str(float(x)) for x in class_weights) if class_weights else ""

    adapter = get_optimizer_constraint_adapter(context.config)
    optimizer_env, has_ab_overrides = adapter.build_env(context.config, os.environ.copy())
    for k, v in ds.get_optimizer_env().items():
        optimizer_env[k] = v

    print(f"[Info] 优化约束适配器: {adapter.name}")
    if has_ab_overrides:
        print("[Info] 已加载 calibration.ab_experiment 参数并传递给 optimizer")
    if history_file:
        print(f"[Info] 时序平滑已启用，history_file={history_file}")
    else:
        print("[Info] 时序平滑已禁用，不加载历史标定结果")

    sys.path.insert(0, os.path.join(_REPO_ROOT, "tools"))
    from score_breakdown_io import write_unified_debug_json  # noqa: E402
    from export_semantic_probs_bin import npy_to_edgecalib_bin  # noqa: E402
    from lidar_bev_rail_points import export_lidar_bev_rail_points  # noqa: E402

    for frame_id in context.frame_ids:
        feature_base = os.path.join(lidar_dir, f"{frame_id:010d}")
        frame_dir = os.path.join(image_root, f"{frame_id:010d}")
        sam_base = os.path.join(sam_root, f"{frame_id:010d}")
        output_file = os.path.join(calib_dir, f"{frame_id:010d}_calib_result.txt")
        optimizer_calib_file, intrinsics_debug = _prepare_optimizer_calib_file(
            context.config,
            calib_file,
            calib_dir,
            frame_id,
            init_extrinsic_source,
        )

        if not os.path.exists(f"{feature_base}_points.txt"):
            print(f"[Warning] 特征文件不存在，跳过帧 {frame_id:010d}")
            continue

        print(f"\n优化帧 {frame_id:010d}...")
        print(f"  logical_frame_id={frame_id:010d}")
        print(f"  feature_base={feature_base}")
        print(f"  image_features_dir={frame_dir}")
        print(f"  optimizer_base={sam_base}")
        r_use, t_use, selected_init_source, bev_candidate_rejected_by_gate, candidate_scores = _select_initial_pose_candidate(
            frame_id,
            list(init_r),
            list(init_t),
            feature_base,
            sam_base,
            ds,
            sem_cfg,
            label_cfg,
            bev_by_frame,
            bev_candidates_by_frame,
            label_assist_for_calib,
        )
        if selected_init_source == "bev_accepted":
            print("[Info] ???? BEV accepted candidate ?? optimizer ??")
        elif selected_init_source in {"bev_raw_rejected", "clipped_bev_candidate"}:
            print(f"[Info] label_assist candidate scoring ?? BEV ????: source={selected_init_source}")
        else:
            print(f"[Info] ??????/label-scored ??: source={selected_init_source}")

        # Phase B6: pass semantic inputs (probabilities + semantic points + BEV init pose) to optimizer CLI.
        sem_npy = os.path.join(frame_dir, "semantic_probs.npy")
        sem_bin = os.path.join(frame_dir, "semantic_probs.edgecalib.bin")
        sem_pts = f"{feature_base}_semantic_points.txt"
        bev_pose = ""
        # Only pass BEV pose to optimizer when this frame's BEV delta was accepted by the Python BEV stage.
        # This avoids silently applying pose_after_bev.txt even when rail_score was below threshold.
        if (
            selected_init_source == "bev_accepted"
            and bool(bev_cfg.get("enabled", False))
            and frame_id in bev_by_frame
            and context.paths
            and context.paths.get("bev_init")
        ):
            cand = os.path.join(context.paths["bev_init"], f"{frame_id:010d}", "pose_after_bev.txt")
            if os.path.isfile(cand):
                bev_pose = cand

        use_sem = sem_enabled and os.path.isfile(sem_npy) and os.path.isfile(sem_pts)
        if use_sem:
            ok = npy_to_edgecalib_bin(sem_npy, sem_bin)
            if not ok:
                if not allow_legacy_fallback:
                    raise RuntimeError(
                        "semantic_calib.enabled=true 且 semantic_probs 导出失败；"
                        "当前配置禁止回退 legacy。"
                    )
                print("[Warning] semantic_probs.npy 导出 EDGESEM1 失败，回退 legacy optimizer 调用")
                use_sem = False
        elif sem_enabled and not allow_legacy_fallback:
            missing = []
            if not os.path.isfile(sem_npy):
                missing.append(sem_npy)
            if not os.path.isfile(sem_pts):
                missing.append(sem_pts)
            raise FileNotFoundError(
                "semantic_calib.enabled=true 但缺少语义输入，且禁止 legacy 回退。缺失文件:\n"
                + "\n".join(missing)
            )

        if use_sem:
            bev_rail_points = f"{feature_base}_rail_bev_points.txt"
            bev_rail_count = 0
            if bool(sem_cfg.get("use_lidar_bev_rail_samples", False)):
                bev_npz = f"{feature_base}_bev_maps.npz"
                if os.path.isfile(bev_npz):
                    try:
                        rail_debug_path = f"{feature_base}_rail_bev_debug.json"
                        rail_png_path = f"{feature_base}_rail_bev_refined.png"
                        oracle_bev = os.path.join(frame_dir, "pseudo_bev.npz")
                        bev_rail_count = export_lidar_bev_rail_points(
                            bev_npz,
                            bev_rail_points,
                            min_prob=float(sem_cfg.get("lidar_bev_rail_sample_min_prob", 0.15)),
                            stride_cells=int(sem_cfg.get("lidar_bev_rail_sample_stride_cells", 2)),
                            max_points=int(sem_cfg.get("lidar_bev_rail_sample_max_points", 8000)),
                            reference_z=float((context.config.get("dataset") or {}).get("reference_z", 0.0)),
                            oracle_npz_path=oracle_bev if os.path.isfile(oracle_bev) else None,
                            oracle_overlap_dilate_cells=int(sem_cfg.get("lidar_bev_oracle_overlap_dilate_cells", 3)),
                            min_component_cells=int(sem_cfg.get("lidar_bev_rail_min_component_cells", 20)),
                            debug_path=rail_debug_path,
                            refined_png_path=rail_png_path,
                            refined_bin_path=f"{feature_base}_rail_bev_refined.bin",
                            crop_to_image_valid=bool(sem_cfg.get("lidar_bev_crop_to_image_valid", True)),
                            crop_to_image_rail_bbox=bool(sem_cfg.get("lidar_bev_crop_to_image_rail_bbox", True)),
                            image_rail_bbox_padding_m=float(sem_cfg.get("lidar_bev_image_rail_bbox_padding_m", 8.0)),
                        )
                        print(f"[Info] Exported LiDAR BEV rail samples: {bev_rail_count} -> {bev_rail_points}")
                    except Exception as exc:
                        print(f"[Warning] Export LiDAR BEV rail samples failed: {type(exc).__name__}: {exc}")

            # Phase C6: adapt rail weight based on LiDAR rail meta (switch detection / low confidence).
            effective_rail_weight = float(sem_cfg.get("rail_weight", 1.2))
            min_img_q = float(sem_cfg.get("min_image_rail_quality", 0.45))
            low_img_weight = float(sem_cfg.get("low_image_rail_quality_weight", sem_cfg.get("branch_rail_weight", 0.0)))
            rail_quality_path = os.path.join(frame_dir, "rail_quality.json")
            rail_quality = _load_json_dict(rail_quality_path)
            if not rail_quality:
                rail_quality = _load_json_dict(f"{sam_base}_rail_quality.json")
            if rail_quality:
                rq_enabled = bool(rail_quality.get("enabled", False))
                rq_score = float(rail_quality.get("quality_score", 0.0) or 0.0)
                oracle_rail = bool(rail_quality.get("label_track_prior_used", False)) and rq_enabled and rq_score >= min_img_q
                if (not rq_enabled) or rq_score < min_img_q:
                    effective_rail_weight = low_img_weight
                    print(
                        f"[Info] image rail quality low "
                        f"(enabled={rq_enabled}, score={rq_score:.3f} < {min_img_q}); "
                        f"rail_weight={effective_rail_weight}"
                    )
                elif oracle_rail:
                    min_samples = int(sem_cfg.get("min_lidar_bev_rail_samples", 200))
                    if bev_rail_count >= min_samples:
                        effective_rail_weight = float(sem_cfg.get("oracle_rail_weight", effective_rail_weight))
                    else:
                        effective_rail_weight = float(sem_cfg.get("oracle_rail_weight_fallback", 0.2))
                    print(
                        f"[Info] JSON oracle rail enabled "
                        f"(score={rq_score:.3f}, bev_samples={bev_rail_count}); "
                        f"rail_weight={effective_rail_weight}"
                    )
                else:
                    oracle_rail = False
            else:
                oracle_rail = False
            align_debug = _load_json_dict(f"{feature_base}_rail_bev_alignment_debug.json")
            rail_debug_current = _load_json_dict(f"{feature_base}_rail_bev_debug.json")
            strong_label_features_path = f"{feature_base}_label_strong_features.tsv"
            strong_label_overrides_rail_mismatch = _strong_label_can_override_rail_mismatch(label_cfg, strong_label_features_path)
            rail_mismatch_present = _has_rail_mismatch(rail_debug_current, align_debug)
            rail_mismatch_ignored_due_to_strong_label = bool(strong_label_overrides_rail_mismatch and rail_mismatch_present)
            rail_weight_forced_low_due_to_mismatch = False
            if rail_mismatch_ignored_due_to_strong_label:
                fallback_weight = float(sem_cfg.get("oracle_rail_weight_fallback", 0.0))
                effective_rail_weight = min(float(effective_rail_weight), fallback_weight)
                rail_weight_forced_low_due_to_mismatch = True
                print(
                    f"[Info] Strong label features available; ignore rail BEV mismatch and lower rail_weight={effective_rail_weight}"
                )
            skip_optimizer, skip_reason = _should_skip_optimizer_for_rail_mismatch(
                sem_cfg,
                bool(oracle_rail),
                rail_debug_current,
                align_debug,
                label_cfg,
                strong_label_features_path,
            )
            if skip_optimizer:
                extra = {
                    "rail_refinement_valid": rail_debug_current.get("rail_refinement_valid", 1),
                    "rail_bev_alignment_valid": align_debug.get("rail_bev_alignment_valid", 0),
                    "rail_bev_alignment_best_transform": align_debug.get("best_transform", ""),
                    "rail_bev_alignment_best_shift_cells": align_debug.get("best_shift_cells", [0, 0]),
                    "rail_bev_alignment_best_score": align_debug.get("best_shift_score", 0.0),
                }
                _write_invalid_calib_result(output_file, list(r_use), list(t_use), skip_reason, extra)
                br = _parse_calib_breakdown(output_file)
                br.update(intrinsics_debug)
                pose_out = _parse_calib_pose(output_file)
                write_unified_debug_json(
                    os.path.join(calib_dir, f"{frame_id:010d}_debug_score_breakdown.json"),
                    stage="calib",
                    frame_id=f"{frame_id:010d}",
                    input_pose={"rvec": list(r_use), "tvec": list(t_use)},
                    output_pose=pose_out,
                    breakdown=br,
                    elapsed_sec=0.0,
                    meta={"optimizer_skipped": True, "skip_reason": skip_reason},
                )
                context.current_pose_semantic = pose_out
                print(f"[Info] Skip optimizer for frame {frame_id:010d}: {skip_reason}")
                continue

            rail_meta_path = f"{feature_base}_rail_meta.json"
            if os.path.isfile(rail_meta_path):
                try:
                    with open(rail_meta_path, "r", encoding="utf-8") as f:
                        rm = json.load(f)
                    branch = bool(rm.get("branch_detected", False))
                    rc = float(rm.get("rail_confidence", 1.0))
                    min_rc = float(sem_cfg.get("min_rail_confidence_for_weight", 0.5))
                    if (not oracle_rail) and (branch or rc < min_rc):
                        effective_rail_weight = float(sem_cfg.get("branch_rail_weight", 0.0))
                        print(
                            f"[Info] 道岔/低置信度轨道（branch={branch}, rail_confidence={rc:.3f} < {min_rc}），"
                            f"rail_weight 降为 {effective_rail_weight}"
                        )
                except (OSError, ValueError, json.JSONDecodeError):
                    print(f"[Warning] 读取 rail_meta 失败，忽略: {rail_meta_path}")

            optimizer_bin = os.path.join(_REPO_ROOT, "build", "optimizer")
            cmd = [
                optimizer_bin,
                "--lidar_feature_base",
                feature_base,
                "--sam_feature_base",
                sam_base,
                "--calib_file",
                optimizer_calib_file if os.path.exists(optimizer_calib_file) else "",
                "--init_rx",
                str(r_use[0]),
                "--init_ry",
                str(r_use[1]),
                "--init_rz",
                str(r_use[2]),
                "--init_tx",
                str(t_use[0]),
                "--init_ty",
                str(t_use[1]),
                "--init_tz",
                str(t_use[2]),
                "--output_file",
                output_file,
                "--semantic_probs",
                sem_bin,
                "--lidar_semantic_points",
                sem_pts,
                "--semantic_js_weight",
                str(float(sem_cfg.get("semantic_js_weight", 3.0))),
                "--histogram_weight",
                str(float(sem_cfg.get("histogram_weight", 0.5))),
                "--edge_weight",
                str(float(sem_cfg.get("edge_weight", 1.0))),
                "--rail_weight",
                str(effective_rail_weight),
                "--lidar_semantic_max_points",
                str(int(sem_cfg.get("lidar_semantic_max_points", 12000))),
                "--stratified_semantic_sampling",
                "1" if bool(sem_cfg.get("stratified_semantic_sampling", False)) else "0",
                "--rail_early_reject_enabled",
                "1" if bool(sem_cfg.get("rail_early_reject_enabled", False)) else "0",
                "--rail_early_reject_visible_ratio",
                str(float(sem_cfg.get("rail_early_reject_visible_ratio", 0.02))),
                "--rail_early_reject_visible_count",
                str(int(sem_cfg.get("rail_early_reject_visible_count", 10))),
                "--optimizer_num_threads",
                str(int(sem_cfg.get("optimizer_num_threads", 0))),
                "--rail_low_visible_policy",
                str(sem_cfg.get("rail_low_visible_policy", "penalty" if oracle_rail else "zero") if oracle_rail else "zero"),
                "--edge_low_visible_policy",
                str(sem_cfg.get("edge_low_visible_policy", "zero")),
                "--min_edge_visible_count",
                str(int(sem_cfg.get("min_edge_visible_count", 50))),
                "--edge_low_visible_penalty",
                str(float(sem_cfg.get("edge_low_visible_penalty", 0.15))),
                "--min_rail_visible_count",
                str(int(sem_cfg.get("min_rail_visible_count", 50))),
                "--min_rail_visible_ratio",
                str(float(sem_cfg.get("min_rail_visible_ratio", 0.08))),
                "--rail_low_visible_penalty",
                str(float(sem_cfg.get("rail_low_visible_penalty", 0.50))),
                "--rail_visibility_residual_weight",
                str(float(sem_cfg.get("rail_visibility_residual_weight", 0.15 if oracle_rail else 0.0))),
                "--rail_oob_residual_weight",
                str(float(sem_cfg.get("rail_oob_residual_weight", 0.20 if oracle_rail else 0.0))),
                "--pyramid_scales",
                pyramid_scales_s,
            ]
            if _optimizer_binary_supports_arg(optimizer_bin, "--vehicle_object_weight"):
                cmd.extend([
                    "--vehicle_object_weight",
                    str(float(sem_cfg.get("vehicle_object_weight", 0.8))),
                    "--person_object_weight",
                    str(float(sem_cfg.get("person_object_weight", 0.5))),
                ])
            else:
                print(
                    "[Warning] build/optimizer does not support object-weight arguments; "
                    "please rebuild optimizer to enable person/vehicle object scoring."
                )
            if _optimizer_binary_supports_arg(optimizer_bin, "--label_assist_enabled"):
                cmd.extend([
                    "--label_assist_enabled",
                    "1" if label_assist_for_calib else "0",
                    "--label_object_points",
                    f"{feature_base}_label_object_points.txt",
                    "--strong_label_enabled",
                    "1" if bool(label_cfg.get("strong_features_enabled", True)) and label_assist_for_calib else "0",
                    "--label_strong_features",
                    f"{feature_base}_label_strong_features.tsv",
                    "--strong_track_weight",
                    str(float(label_cfg.get("track_weight", 2.0))),
                    "--strong_pole_weight",
                    str(float(label_cfg.get("catenary_pole_weight", 1.5))),
                    "--strong_switch_weight",
                    str(float(label_cfg.get("switch_weight", 1.2))),
                    "--strong_buffer_stop_weight",
                    str(float(label_cfg.get("buffer_stop_weight", 1.0))),
                    "--strong_label_min_translation_prior_weight",
                    str(float(sem_cfg.get("strong_label_min_translation_prior_weight", 5.0))),
                    "--strong_label_rotation_prior_weight",
                    str(float(sem_cfg.get("strong_label_rotation_prior_weight", 10.0))),
                    "--strong_label_max_pose_jump_m",
                    str(float(sem_cfg.get("max_label_assisted_pose_jump_m", 1.0))),
                    "--strong_label_max_yaw_jump_deg",
                    str(float(sem_cfg.get("max_label_assisted_yaw_jump_deg", 3.0))),
                    "--label_track_weight",
                    str(float(label_cfg.get("track_weight", 1.5))),
                    "--label_static_weight",
                    str(float(label_cfg.get("static_object_weight", 1.0))),
                    "--label_vehicle_weight",
                    str(float(label_cfg.get("vehicle_weight", 0.4))),
                    "--label_person_weight",
                    str(float(label_cfg.get("person_weight", 0.2))),
                ])
                if _optimizer_binary_supports_arg(optimizer_bin, "--strong_label_static_score_tolerance"):
                    cmd.extend([
                        "--strong_label_static_score_tolerance",
                        str(float(sem_cfg.get("strong_label_static_score_tolerance", 0.0))),
                    ])
                if _optimizer_binary_supports_arg(optimizer_bin, "--strong_stage_a_use_switch"):
                    cmd.extend([
                        "--strong_stage_a_use_switch",
                        "1" if bool(label_cfg.get("stage_a_use_switch", False)) else "0",
                    ])
                if _optimizer_binary_supports_arg(optimizer_bin, "--strong_stage_b_use_track"):
                    cmd.extend([
                        "--strong_stage_b_use_track",
                        "1" if bool(label_cfg.get("stage_b_use_track", False)) else "0",
                    ])
                if _optimizer_binary_supports_arg(optimizer_bin, "--strong_stage_b_use_switch"):
                    cmd.extend([
                        "--strong_stage_b_use_switch",
                        "1" if bool(label_cfg.get("stage_b_use_switch", False)) else "0",
                    ])
                if _optimizer_binary_supports_arg(optimizer_bin, "--strong_stage_b_track_min_score"):
                    cmd.extend([
                        "--strong_stage_b_track_min_score",
                        str(float(label_cfg.get("stage_b_track_min_score", 0.25))),
                    ])
                if _optimizer_binary_supports_arg(optimizer_bin, "--strong_stage_b_switch_min_score"):
                    cmd.extend([
                        "--strong_stage_b_switch_min_score",
                        str(float(label_cfg.get("stage_b_switch_min_score", 0.25))),
                    ])
            elif label_assist_for_calib:
                print(
                    "[Warning] build/optimizer does not support label-assisted arguments; "
                    "please rebuild optimizer to enable label teacher residuals."
                )
            if class_weights_s:
                cmd.extend(["--class_weights", class_weights_s])
            mode = str(sem_cfg.get("optimize_mode", "full_calib") or "full_calib")
            cmd.extend(["--mode", mode])
            if bev_pose:
                cmd.extend(["--init_pose_from_bev", bev_pose])
        else:
            if sem_enabled:
                print("[Warning] semantic_calib.enabled=true 但缺少 semantic_probs.npy 或 semantic_points.txt，回退 legacy")
            cmd = [
                os.path.join(_REPO_ROOT, "build", "optimizer"),
                feature_base,
                sam_base,
                optimizer_calib_file if os.path.exists(optimizer_calib_file) else "",
                str(r_use[0]),
                str(r_use[1]),
                str(r_use[2]),
                str(t_use[0]),
                str(t_use[1]),
                str(t_use[2]),
                output_file,
            ]
            if history_file:
                cmd.append(history_file)

        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, env=optimizer_env, cwd=_REPO_ROOT)
        elapsed = time.perf_counter() - t0

        pose_out = _parse_calib_pose(output_file)
        if pose_out:
            context.current_pose_semantic = pose_out
        br = _parse_calib_breakdown(output_file)
        br.update(intrinsics_debug)
        rail_debug = _load_json_dict(f"{feature_base}_rail_bev_debug.json")
        edge_debug = _load_json_dict(f"{feature_base}_edge_meta.json")
        label_debug = _load_json_dict(f"{feature_base}_debug_label_assist.json")
        if rail_debug:
            if "lidar_rail_refined_nonzero_ratio" in rail_debug:
                br["refined_lidar_rail_nonzero_ratio"] = rail_debug["lidar_rail_refined_nonzero_ratio"]
            elif "lidar_rail_raw_nonzero_ratio" in rail_debug:
                br["refined_lidar_rail_nonzero_ratio"] = rail_debug.get("lidar_rail_refined_nonzero_ratio", rail_debug.get("lidar_rail_raw_nonzero_ratio", 0.0))
        if edge_debug:
            for k in ("edge_raw_count", "edge_kept_count", "edge_range_gt_50_count", "edge_near_track_count"):
                if k in edge_debug:
                    br[k] = edge_debug[k]
        br["extrinsic_source"] = init_extrinsic_source
        if "rail_mismatch_present" in locals():
            br["rail_bev_alignment_mismatch"] = 1.0 if rail_mismatch_present else 0.0
        if "rail_mismatch_ignored_due_to_strong_label" in locals():
            br["rail_mismatch_ignored_due_to_strong_label"] = 1.0 if rail_mismatch_ignored_due_to_strong_label else 0.0
        if "rail_weight_forced_low_due_to_mismatch" in locals():
            br["rail_weight_forced_low_due_to_mismatch"] = 1.0 if rail_weight_forced_low_due_to_mismatch else 0.0
        br["selected_init_source"] = selected_init_source
        br["bev_candidate_rejected_by_gate"] = 1.0 if bev_candidate_rejected_by_gate else 0.0
        br["candidate_scores"] = candidate_scores
        if candidate_scores:
            selected_score = next((x for x in candidate_scores if x.get("source") == selected_init_source), {})
            for _k in ("label_teacher_eligible_count", "label_teacher_visible_ratio"):
                if _k in selected_score:
                    br[f"init_{_k}"] = selected_score[_k]
        _add_pose_jump_debug(br, pose_out, list(init_r), list(init_t))
        if label_debug:
            for k in (
                "label_assist_enabled",
                "label_feature_used",
                "unsupervised_feature_used",
                "image_track_vs_label_score",
                "sam_vehicle_vs_label_iou",
                "sam_person_vs_label_iou",
                "label_track_point_count",
                "label_static_point_count",
                "label_vehicle_point_count",
                "label_person_point_count",
                "label_teacher_eligible_count",
                "label_teacher_visible_xmax_m",
                "label_teacher_image_bbox_padding_m",
            ):
                if k in label_debug:
                    br[k] = label_debug[k]
        br = _apply_final_rail_hard_gate(br, sem_cfg, bool(locals().get("oracle_rail", False)))
        if output_file and os.path.isfile(output_file):
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"final_pose_valid: {br.get('final_pose_valid', 1.0)}\n")
                f.write(f"rail_gate_failed: {br.get('rail_gate_failed', 0.0)}\n")
                if br.get("invalid_reason"):
                    f.write(f"invalid_reason: {br.get('invalid_reason')}\n")
                for _k in ("label_gate_bypass_used", "label_gate_failed_reason", "strong_label_gate_used", "object_or_edge_rail_gate_bypass", "pose_jump_from_initial_m", "yaw_jump_from_initial_deg", "rotation_jump_from_initial_deg", "selected_init_source", "bev_candidate_rejected_by_gate", "final_gate_source", "init_label_teacher_eligible_count", "init_label_teacher_visible_ratio", "rail_bev_alignment_mismatch", "rail_mismatch_ignored_due_to_strong_label", "rail_weight_forced_low_due_to_mismatch", "openlabel_intrinsics_available", "optimizer_intrinsics_source", "intrinsics_cx_delta_px", "intrinsics_cy_delta_px", "calibration_intrinsics_cx_delta_px", "calibration_intrinsics_cy_delta_px", "optimizer_calib_file"):
                    if _k in br:
                        f.write(f"{_k}: {br[_k]}\n")
                if br.get("candidate_scores"):
                    f.write("candidate_scores_json: " + json.dumps(br["candidate_scores"], ensure_ascii=False) + "\n")
                if "refined_lidar_rail_nonzero_ratio" in br:
                    f.write(f"refined_lidar_rail_nonzero_ratio: {br['refined_lidar_rail_nonzero_ratio']}\n")
        strong_debug = _extract_strong_label_debug(br)
        if strong_debug:
            with open(os.path.join(calib_dir, f"{frame_id:010d}_strong_label_debug.json"), "w", encoding="utf-8") as f:
                json.dump(strong_debug, f, indent=2, ensure_ascii=False)
        write_unified_debug_json(
            os.path.join(calib_dir, f"{frame_id:010d}_debug_score_breakdown.json"),
            stage="calib",
            frame_id=f"{frame_id:010d}",
            input_pose={"rvec": list(r_use), "tvec": list(t_use)},
            output_pose=pose_out,
            breakdown=br,
            elapsed_sec=elapsed,
            meta={"optimizer_argv_len": len(cmd)},
        )

    print(f"\n[完成] 标定结果已保存到: {calib_dir}")
