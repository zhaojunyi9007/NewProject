
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export OSDaR23 OpenLABEL features as label-assisted teacher priors.

This tool is intentionally a sidecar: it does not replace SAM/DBSCAN outputs.
It writes optional *_label_* maps and label object point samples consumed by the
optimizer when label_assist.enabled=true.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

SEM_RAIL_LIKE = 1
SEM_VERTICAL_STRUCTURE = 3
SEM_VEHICLE_LIKE = 5
SEM_PERSON_LIKE = 7

STATIC_TYPES = {"catenary_pole", "signal_pole", "buffer_stop", "switch"}
VEHICLE_TYPES = {"road_vehicle"}
PERSON_TYPES = {"person"}
TRACK_TYPES = {"track"}
STRONG_TYPES = {"track", "catenary_pole", "switch", "buffer_stop"}


def _bbox_to_rect(val, w: int, h: int):
    if not val or len(val) < 4:
        return None
    cx, cy, bw, bh = [float(x) for x in val[:4]]
    x0 = int(round(cx - bw * 0.5)); x1 = int(round(cx + bw * 0.5))
    y0 = int(round(cy - bh * 0.5)); y1 = int(round(cy + bh * 0.5))
    x0 = max(0, min(w - 1, x0)); x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0)); y1 = max(0, min(h - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _bbox_centerline_from_rect(rect):
    x0, y0, x1, y1 = rect
    cx = 0.5 * (float(x0) + float(x1))
    return [[cx, float(y0)], [cx, float(y1)]]


def _bbox_center_from_rect(rect):
    x0, y0, x1, y1 = rect
    return [0.5 * (float(x0) + float(x1)), 0.5 * (float(y0) + float(y1))]


def _poly_points(val, w: int, h: int):
    if not val or len(val) < 4:
        return None
    pts=[]
    for i in range(0, len(val)-1, 2):
        x=int(round(float(val[i]))); y=int(round(float(val[i+1])))
        pts.append([max(0,min(w-1,x)), max(0,min(h-1,y))])
    if len(pts)<2:
        return None
    return np.asarray(pts, dtype=np.int32).reshape(-1,1,2)


def _dist_from_mask(mask: np.ndarray, cap_ratio: float = 0.08):
    mask=(mask>0).astype(np.uint8)
    if not mask.any():
        return np.ones(mask.shape, dtype=np.float32)
    inv=(mask==0).astype(np.uint8)
    dist=cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    cap=max(1.0, float(max(mask.shape))*cap_ratio)
    return np.clip(dist/cap,0.0,1.0).astype(np.float32)


def _write_u16(path: str, arr: np.ndarray):
    cv2.imwrite(path, (np.clip(arr,0,1)*65535.0).astype(np.uint16))


def _attrs_to_dict(attrs: Dict[str, Any]):
    out={}
    for group in (attrs or {}).values():
        if isinstance(group, list):
            for item in group:
                if isinstance(item, dict) and "name" in item:
                    out[item["name"]]=item.get("val")
    return out


def _extract_frame(openlabel: Dict[str, Any], frame_id: int, image_sensor: str):
    frames=openlabel.get("frames", {})
    objects_meta=openlabel.get("objects", {})
    frame=frames.get(str(frame_id))
    if not frame:
        return []
    rows=[]
    for oid, entry in (frame.get("objects") or {}).items():
        typ=(objects_meta.get(oid) or {}).get("type", "")
        name=(objects_meta.get(oid) or {}).get("name", "")
        od=entry.get("object_data", {}) or {}
        rec={"object_id": oid, "name": name, "type": typ, "rgb_bbox": [], "rgb_poly2d": [], "lidar_cuboid": [], "lidar_vec": []}
        for b in od.get("bbox", []) or []:
            if b.get("coordinate_system")==image_sensor or str(b.get("name","")).startswith(image_sensor+"__"):
                rec["rgb_bbox"].append({"val": b.get("val"), "attributes": _attrs_to_dict(b.get("attributes",{})), "name": b.get("name","")})
        for p in od.get("poly2d", []) or []:
            if p.get("coordinate_system")==image_sensor or str(p.get("name","")).startswith(image_sensor+"__"):
                rec["rgb_poly2d"].append({"val": p.get("val"), "attributes": _attrs_to_dict(p.get("attributes",{})), "name": p.get("name","")})
        for c in od.get("cuboid", []) or []:
            if c.get("coordinate_system")=="lidar" or str(c.get("name","")).startswith("lidar__"):
                rec["lidar_cuboid"].append({"val": c.get("val"), "attributes": _attrs_to_dict(c.get("attributes",{})), "name": c.get("name","")})
        for v in od.get("vec", []) or []:
            if v.get("coordinate_system")=="lidar" or str(v.get("name","")).startswith("lidar__"):
                rec["lidar_vec"].append({"val": v.get("val"), "attributes": _attrs_to_dict(v.get("attributes",{})), "name": v.get("name","")})
        rec["paired"] = bool((rec["rgb_bbox"] or rec["rgb_poly2d"]) and (rec["lidar_cuboid"] or rec["lidar_vec"]))
        rows.append(rec)
    return rows


def _class_id_for_type(typ: str):
    if typ in TRACK_TYPES:
        return SEM_RAIL_LIKE
    if typ in VEHICLE_TYPES:
        return SEM_VEHICLE_LIKE
    if typ in PERSON_TYPES:
        return SEM_PERSON_LIKE
    if typ in STATIC_TYPES:
        return SEM_VERTICAL_STRUCTURE
    return None


def _type_group(typ: str):
    if typ in TRACK_TYPES: return "track"
    if typ in VEHICLE_TYPES: return "vehicle"
    if typ in PERSON_TYPES: return "person"
    if typ in STATIC_TYPES: return "static"
    return "other"


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n <= 1e-9:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.asarray([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def _cuboid_samples(cuboid_val):
    if not cuboid_val or len(cuboid_val) < 10:
        return []
    center = np.asarray([float(v) for v in cuboid_val[:3]], dtype=np.float64)
    qx, qy, qz, qw = [float(v) for v in cuboid_val[3:7]]
    sx, sy, sz = [abs(float(v)) for v in cuboid_val[7:10]]
    rot = _quat_to_rot(qx, qy, qz, qw)
    local = [np.zeros(3, dtype=np.float64)]
    for dx in (-0.5, 0.5):
        for dy in (-0.5, 0.5):
            for dz in (-0.5, 0.5):
                local.append(np.asarray([dx*sx, dy*sy, dz*sz], dtype=np.float64))
    local.append(np.asarray([0.0, 0.0, 0.5*sz], dtype=np.float64))
    local.append(np.asarray([0.0, 0.0, -0.5*sz], dtype=np.float64))
    samples = []
    for offset in local:
        p = center + rot @ offset
        samples.append((float(p[0]), float(p[1]), float(p[2])))
    return samples


def _cuboid_geometry(cuboid_val):
    if not cuboid_val or len(cuboid_val) < 10:
        return None
    center = np.asarray([float(v) for v in cuboid_val[:3]], dtype=np.float64)
    qx, qy, qz, qw = [float(v) for v in cuboid_val[3:7]]
    sx, sy, sz = [abs(float(v)) for v in cuboid_val[7:10]]
    rot = _quat_to_rot(qx, qy, qz, qw)
    corners = []
    for dx in (-0.5, 0.5):
        for dy in (-0.5, 0.5):
            for dz in (-0.5, 0.5):
                p = center + rot @ np.asarray([dx * sx, dy * sy, dz * sz], dtype=np.float64)
                corners.append([float(p[0]), float(p[1]), float(p[2])])
    axis_top = center + rot @ np.asarray([0.0, 0.0, 0.5 * sz], dtype=np.float64)
    axis_bottom = center + rot @ np.asarray([0.0, 0.0, -0.5 * sz], dtype=np.float64)
    return {
        "center": [float(center[0]), float(center[1]), float(center[2])],
        "corners": corners,
        "axis": [[float(axis_bottom[0]), float(axis_bottom[1]), float(axis_bottom[2])],
                 [float(axis_top[0]), float(axis_top[1]), float(axis_top[2])]],
        "size": [float(sx), float(sy), float(sz)],
    }


def _is_visible_lidar_point(p, xmax_m: float):
    x, y, z = p[:3]
    if not (math.isfinite(float(x)) and math.isfinite(float(y)) and math.isfinite(float(z))):
        return False
    if float(x) < 0.0:
        return False
    if xmax_m > 0.0 and float(x) > xmax_m:
        return False
    return True


def _filter_visible_points(points, xmax_m: float):
    return [tuple(float(v) for v in p[:3]) for p in points if _is_visible_lidar_point(p, xmax_m)]


def _downsample_points(points, max_points: int):
    if max_points <= 0 or len(points) <= max_points:
        return points
    step = int(math.ceil(len(points) / float(max_points)))
    return points[::max(1, step)][:max_points]


def _tsv_row(class_type: str, object_id: str, role: str, weight: float, p1, p2, image_kind: str, image_values):
    vals = [class_type, object_id, role, f"{float(weight):.6f}"]
    vals.extend(f"{float(x):.6f}" for x in p1[:3])
    vals.extend(f"{float(x):.6f}" for x in p2[:3])
    vals.append(image_kind)
    vals.extend(str(x) if isinstance(x, int) else f"{float(x):.6f}" for x in image_values)
    return "\t".join(vals) + "\n"


def _load_pcd_xyz(path: str | None):
    if not path or not os.path.isfile(path):
        return None
    points=[]
    data=False
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line=line.strip()
                if not data:
                    if line.lower().startswith('data'):
                        data=True
                    continue
                if not line:
                    continue
                parts=line.split()
                if len(parts) < 3:
                    continue
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
    except Exception:
        return None
    return points


def _vec_samples_from_indices(vec_val, pcd_xyz, max_points: int = 1200):
    if pcd_xyz is None or not vec_val:
        return []
    out=[]
    step=max(1, int(math.ceil(len(vec_val)/max(1,max_points))))
    for raw in vec_val[::step]:
        try:
            idx=int(raw)
        except Exception:
            continue
        if 0 <= idx < len(pcd_xyz):
            out.append(pcd_xyz[idx])
    return out


def _vec_samples(vec_val, step_m: float = 0.5):
    if not vec_val or len(vec_val) < 4:
        return []
    vals = [float(v) for v in vec_val]
    pts = []
    if len(vals) >= 6:
        a = np.asarray(vals[:3], dtype=np.float64)
        b = np.asarray(vals[3:6], dtype=np.float64)
    else:
        a = np.asarray([vals[0], vals[1], 0.0], dtype=np.float64)
        b = np.asarray([vals[2], vals[3], 0.0], dtype=np.float64)
    d = b - a
    length = float(np.linalg.norm(d))
    if length <= 1e-9:
        return [(float(a[0]), float(a[1]), float(a[2]))]
    n = max(2, int(math.floor(length / max(1e-3, step_m))) + 1)
    for i in range(n):
        alpha = 0.0 if n <= 1 else i / float(n - 1)
        p = a + alpha * d
        pts.append((float(p[0]), float(p[1]), float(p[2])))
    return pts


def _iou(a: np.ndarray, b: np.ndarray):
    aa=(a>0).astype(np.uint8); bb=(b>0).astype(np.uint8)
    inter=float(np.logical_and(aa,bb).sum()); uni=float(np.logical_or(aa,bb).sum())
    return inter/uni if uni>0 else 0.0


def _fixed_body_to_optical() -> np.ndarray:
    return np.asarray([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)


def _openlabel_camera_matrix(openlabel: Dict[str, Any], image_sensor: str):
    stream = (openlabel.get("streams") or {}).get(image_sensor) or {}
    intr = ((stream.get("stream_properties") or {}).get("intrinsics_pinhole") or {})
    mat = intr.get("camera_matrix") or []
    if len(mat) >= 12:
        return np.asarray([float(mat[0]), float(mat[1]), float(mat[2]), float(mat[4]), float(mat[5]), float(mat[6]), float(mat[8]), float(mat[9]), float(mat[10])], dtype=np.float64).reshape(3, 3)
    if len(mat) >= 9:
        return np.asarray([float(v) for v in mat[:9]], dtype=np.float64).reshape(3, 3)
    return None


def _openlabel_lidar_to_optical(openlabel: Dict[str, Any], image_sensor: str):
    cs = (openlabel.get("coordinate_systems") or {}).get(image_sensor) or {}
    pose = cs.get("pose_wrt_parent") or {}
    trans = pose.get("translation") or [0.0, 0.0, 0.0]
    quat = pose.get("quaternion") or [0.0, 0.0, 0.0, 1.0]
    if len(trans) < 3 or len(quat) < 4:
        return None
    T_sensor_to_lidar = np.eye(4, dtype=np.float64)
    T_sensor_to_lidar[:3, :3] = _quat_to_rot(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    T_sensor_to_lidar[:3, 3] = np.asarray([float(v) for v in trans[:3]], dtype=np.float64)
    T_lidar_to_body = np.linalg.inv(T_sensor_to_lidar)
    T_body_to_optical = np.eye(4, dtype=np.float64)
    T_body_to_optical[:3, :3] = _fixed_body_to_optical()
    return T_body_to_optical @ T_lidar_to_body


def _project_point(K: np.ndarray, T: np.ndarray, p3):
    p = np.asarray([float(p3[0]), float(p3[1]), float(p3[2]), 1.0], dtype=np.float64)
    q = T @ p
    if q[2] <= 1e-6:
        return None
    uvw = K @ q[:3]
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


def _point_segment_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    den = vx * vx + vy * vy
    t = 0.0 if den <= 1e-9 else max(0.0, min(1.0, (wx * vx + wy * vy) / den))
    cx, cy = ax + t * vx, ay + t * vy
    return math.hypot(px - cx, py - cy)


def _polyline_dist(px, py, vals):
    n = int(float(vals[0])) if vals else 0
    coords = [float(v) for v in vals[1:1 + 2 * n]]
    if n < 2 or len(coords) < 4:
        return 1e6
    best = 1e6
    for i in range(n - 1):
        ax, ay = coords[2 * i], coords[2 * i + 1]
        bx, by = coords[2 * i + 2], coords[2 * i + 3]
        best = min(best, _point_segment_dist(px, py, ax, ay, bx, by))
    return best


def _bbox_dist(px, py, vals):
    if len(vals) < 4:
        return 1e6
    x0, y0, x1, y1 = [float(v) for v in vals[:4]]
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def _centerline_dist(px, py, vals):
    if len(vals) < 4:
        return 1e6
    return _point_segment_dist(px, py, float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))


def _score_strong_rows_with_openlabel_pose(openlabel: Dict[str, Any], image_sensor: str, strong_rows: List[str], width: int, height: int):
    K = _openlabel_camera_matrix(openlabel, image_sensor)
    T = _openlabel_lidar_to_optical(openlabel, image_sensor)
    if K is None or T is None:
        return {}, {}, {}, False
    stats: Dict[str, Dict[str, float]] = {}
    for line in strong_rows:
        if not line or line.startswith("#"):
            continue
        parts = line.strip().split("	")
        if len(parts) < 12:
            continue
        cls = parts[0]
        p3 = [float(parts[4]), float(parts[5]), float(parts[6])]
        image_kind = parts[10]
        vals = parts[11:]
        uv = _project_point(K, T, p3)
        st = stats.setdefault(cls, {"total": 0.0, "in_image": 0.0, "dist_sum": 0.0, "score_sum": 0.0})
        st["total"] += 1.0
        if uv is None:
            st["dist_sum"] += 1000.0
            continue
        u, v = uv
        if 0.0 <= u < float(width) and 0.0 <= v < float(height):
            st["in_image"] += 1.0
        if image_kind in ("polyline", "polygon"):
            d = _polyline_dist(u, v, vals)
        elif image_kind == "centerline":
            d = _centerline_dist(u, v, vals)
        elif image_kind == "bbox":
            d = _bbox_dist(u, v, vals)
        else:
            d = 1000.0
        st["dist_sum"] += d
        st["score_sum"] += math.exp(-(d * d) / (64.0 * 64.0))
    scores, mean_dists, ratios = {}, {}, {}
    valid = True
    for cls, st in stats.items():
        total = max(1.0, st["total"])
        scores[cls] = float(st["score_sum"] / total)
        mean_dists[cls] = float(st["dist_sum"] / total)
        ratios[cls] = float(st["in_image"] / total)
        if cls in {"buffer_stop", "catenary_pole"} and (scores[cls] < 0.2 or ratios[cls] < 0.5):
            valid = False
    return scores, mean_dists, ratios, valid


def export_label_assist(label_json: str, frame_id: int, image_path: str, image_sensor: str, sam_base: str, frame_dir: str, lidar_base: str|None=None, lidar_pcd_path: str|None=None, teacher_visible_xmax_m: float = 120.0, teacher_image_bbox_padding_m: float = 8.0, use_sam_refine: bool = False, strong_features_enabled: bool = True, max_track_samples_per_object: int = 800, max_pole_samples_per_object: int = 20, bbox_padding_px: int = 12, track_visible_xmax_m: float = 120.0, switch_visible_xmax_m: float = 120.0, catenary_pole_visible_xmax_m: float = 160.0, buffer_stop_visible_xmax_m: float = 240.0):
    image=cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    h,w=image.shape[:2]
    data=json.loads(Path(label_json).read_text(encoding="utf-8"))
    openlabel=data.get("openlabel", data)
    rows=_extract_frame(openlabel, frame_id, image_sensor)
    pcd_xyz=_load_pcd_xyz(lidar_pcd_path)
    masks={k:np.zeros((h,w), dtype=np.uint8) for k in ["person","vehicle","track","static"]}
    paired_counts={k:0 for k in ["person","vehicle","track","static"]}
    bbox_counts={k:0 for k in ["person","vehicle","track","static"]}
    poly_counts={k:0 for k in ["person","vehicle","track","static"]}
    point_lines=["# x y z class_id weight object_id geometry_role\n"]
    point_counts={k:0 for k in ["track","static","vehicle","person"]}
    raw_point_counts={k:0 for k in ["track","static","vehicle","person"]}
    strong_features=[]
    strong_tsv_lines=["# class_type object_id role weight x1 y1 z1 x2 y2 z2 image_kind image_values...\n"]
    strong_counts={k:0 for k in ["track","catenary_pole","switch","buffer_stop"]}
    strong_residual_point_counts={k:0 for k in ["track","catenary_pole","switch","buffer_stop"]}
    strong_raw_object_counts={k:0 for k in ["track","catenary_pole","switch","buffer_stop"]}
    strong_filtered_by_range_counts={k:0 for k in ["track","catenary_pole","switch","buffer_stop"]}
    strong_missing_rgb_counts={k:0 for k in ["track","catenary_pole","switch","buffer_stop"]}
    strong_missing_lidar_counts={k:0 for k in ["track","catenary_pole","switch","buffer_stop"]}
    teacher_visible_xmax_m = float(teacher_visible_xmax_m)
    teacher_image_bbox_padding_m = float(teacher_image_bbox_padding_m)
    strong_visible_xmax_m = {
        "track": float(track_visible_xmax_m),
        "switch": float(switch_visible_xmax_m),
        "catenary_pole": float(catenary_pole_visible_xmax_m),
        "buffer_stop": float(buffer_stop_visible_xmax_m),
    }

    def add_teacher_point(group: str, x: float, y: float, z: float, cid: int, weight: float, object_id: str, role: str) -> None:
        if group in raw_point_counts:
            raw_point_counts[group] += 1
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return
        if x < 0.0:
            return
        if teacher_visible_xmax_m > 0.0 and x > teacher_visible_xmax_m:
            return
        point_lines.append(f"{x:.6f} {y:.6f} {z:.6f} {cid} {weight:.6f} {object_id} {role}\n")
        if group in point_counts:
            point_counts[group]+=1

    def _strong_xmax(class_type: str) -> float:
        return float(strong_visible_xmax_m.get(class_type, teacher_visible_xmax_m))

    def _buffer_stop_effective_weight(pt, base_weight: float = 1.0) -> float:
        x = float(pt[0]) if pt is not None else 0.0
        decay = 1.0 - max(0.0, x - 120.0) / 160.0
        return float(base_weight) * max(0.35, min(1.0, decay))

    label_objects=[]
    for rec in rows:
        group=_type_group(rec["type"])
        if group not in masks:
            continue
        if rec.get("paired"):
            paired_counts[group]+=1
        for b in rec.get("rgb_bbox", []):
            rect=_bbox_to_rect(b.get("val"), w, h)
            if rect:
                x0,y0,x1,y1=rect
                cv2.rectangle(masks[group], (x0,y0), (x1,y1), 255, -1)
                bbox_counts[group]+=1
        for p in rec.get("rgb_poly2d", []):
            pts=_poly_points(p.get("val"), w, h)
            if pts is not None:
                if group=="track":
                    cv2.polylines(masks[group], [pts], False, 255, 5, cv2.LINE_AA)
                else:
                    cv2.fillPoly(masks[group], [pts], 255, cv2.LINE_AA)
                poly_counts[group]+=1
        cid=_class_id_for_type(rec["type"])
        if cid is not None and rec.get("paired"):
            weight={"track":1.0,"static":0.9,"vehicle":0.45,"person":0.35}.get(group,0.5)
            for ci,c in enumerate(rec.get("lidar_cuboid", [])):
                for si,(x,y,z) in enumerate(_cuboid_samples(c.get("val"))):
                    add_teacher_point(group, x, y, z, cid, weight, rec['object_id'], f"{group}_cuboid{ci}_sample{si}")
            if group == "track":
                for vi,v in enumerate(rec.get("lidar_vec", [])):
                    samples = _vec_samples_from_indices(v.get("val"), pcd_xyz) if pcd_xyz is not None else _vec_samples(v.get("val"))
                    for si,(x,y,z) in enumerate(samples):
                        add_teacher_point(group, x, y, z, cid, weight, rec['object_id'], f"{group}_vec{vi}_sample{si}")
        if strong_features_enabled and rec["type"] in STRONG_TYPES:
            class_type = rec["type"]
            strong_raw_object_counts[class_type] += 1
            has_rgb = bool(rec.get("rgb_bbox")) or bool(rec.get("rgb_poly2d"))
            has_lidar = bool(rec.get("lidar_cuboid")) or bool(rec.get("lidar_vec"))
            if not has_rgb:
                strong_missing_rgb_counts[class_type] += 1
            if not has_lidar or not rec.get("paired"):
                strong_missing_lidar_counts[class_type] += 1
            before_feature_count = len(strong_features)

            if rec.get("paired") and has_rgb and has_lidar:
                image_polys = []
                for p in rec.get("rgb_poly2d", []):
                    pts = _poly_points(p.get("val"), w, h)
                    if pts is not None:
                        image_polys.append([[float(x), float(y)] for x, y in pts.reshape(-1, 2).tolist()])
                image_rects = []
                for b in rec.get("rgb_bbox", []):
                    rect = _bbox_to_rect(b.get("val"), w, h)
                    if rect:
                        image_rects.append(rect)

                def append_feature(feature, rows):
                    strong_features.append(feature)
                    strong_counts[feature["class_type"]] += 1
                    strong_tsv_lines.extend(rows)

                if class_type == "track" and image_polys:
                    xmax_m = _strong_xmax("track")
                    for vi, v in enumerate(rec.get("lidar_vec", [])):
                        raw_samples = _vec_samples_from_indices(v.get("val"), pcd_xyz, max_points=max_track_samples_per_object) if pcd_xyz is not None else _vec_samples(v.get("val"))
                        visible_samples = _filter_visible_points(raw_samples, xmax_m)
                        strong_filtered_by_range_counts["track"] += max(0, len(raw_samples) - len(visible_samples))
                        samples = _downsample_points(visible_samples, max_track_samples_per_object)
                        if not samples:
                            continue
                        poly = image_polys[min(vi, len(image_polys) - 1)]
                        image_vals = [len(poly)] + [coord for pt in poly for coord in pt]
                        rows = [_tsv_row("track", rec["object_id"], f"point{si}", 1.0, pt, (0, 0, 0), "polyline", image_vals) for si, pt in enumerate(samples)]
                        strong_residual_point_counts["track"] += len(samples)
                        append_feature({
                            "object_id": rec["object_id"], "class_type": "track", "weight": 1.0, "paired": True,
                            "visible_x_range_m": [0.0, xmax_m],
                            "image_geometry": {"kind": "polyline", "points": poly},
                            "lidar_geometry": {"kind": "sample_points", "points": [list(map(float, p)) for p in samples]},
                        }, rows)

                elif class_type == "switch" and image_polys:
                    xmax_m = _strong_xmax("switch")
                    switch_points = []
                    for v in rec.get("lidar_vec", []):
                        switch_points.extend(_vec_samples_from_indices(v.get("val"), pcd_xyz, max_points=400) if pcd_xyz is not None else _vec_samples(v.get("val")))
                    for c in rec.get("lidar_cuboid", []):
                        switch_points.extend(_cuboid_samples(c.get("val")))
                    raw_switch_points = switch_points
                    visible_switch_points = _filter_visible_points(raw_switch_points, xmax_m)
                    strong_filtered_by_range_counts["switch"] += max(0, len(raw_switch_points) - len(visible_switch_points))
                    switch_points = _downsample_points(visible_switch_points, 400)
                    if switch_points:
                        poly = image_polys[0]
                        image_vals = [len(poly)] + [coord for pt in poly for coord in pt]
                        rows = [_tsv_row("switch", rec["object_id"], f"point{si}", 1.2, pt, (0, 0, 0), "polyline", image_vals) for si, pt in enumerate(switch_points)]
                        strong_residual_point_counts["switch"] += len(switch_points)
                        append_feature({
                            "object_id": rec["object_id"], "class_type": "switch", "weight": 1.2, "paired": True,
                            "visible_x_range_m": [0.0, xmax_m],
                            "image_geometry": {"kind": "polygon", "points": poly},
                            "lidar_geometry": {"kind": "sample_points", "points": [list(map(float, p)) for p in switch_points]},
                        }, rows)

                elif class_type == "catenary_pole" and image_rects:
                    xmax_m = _strong_xmax("catenary_pole")
                    rect = image_rects[0]
                    centerline = _bbox_centerline_from_rect(rect)
                    image_vals = [centerline[0][0], centerline[0][1], centerline[1][0], centerline[1][1]]
                    rows = []
                    axes = []
                    sample_points = []
                    for ci, c in enumerate(rec.get("lidar_cuboid", [])):
                        geom = _cuboid_geometry(c.get("val"))
                        if geom and _is_visible_lidar_point(geom["axis"][0], xmax_m) and _is_visible_lidar_point(geom["axis"][1], xmax_m):
                            axes.append(geom["axis"])
                            rows.append(_tsv_row("catenary_pole", rec["object_id"], f"axis{ci}", 1.5, geom["axis"][0], geom["axis"][1], "centerline", image_vals))
                    for v in rec.get("lidar_vec", []):
                        sample_points.extend(_vec_samples_from_indices(v.get("val"), pcd_xyz, max_points=max_pole_samples_per_object) if pcd_xyz is not None else _vec_samples(v.get("val")))
                    sample_points = _downsample_points(_filter_visible_points(sample_points, xmax_m), max_pole_samples_per_object)
                    for si, pt in enumerate(sample_points):
                        rows.append(_tsv_row("catenary_pole", rec["object_id"], f"point{si}", 1.5, pt, (0, 0, 0), "centerline", image_vals))
                    if rows:
                        strong_residual_point_counts["catenary_pole"] += len(rows)
                        append_feature({
                            "object_id": rec["object_id"], "class_type": "catenary_pole", "weight": 1.5, "paired": True,
                            "visible_x_range_m": [0.0, xmax_m],
                            "image_geometry": {"kind": "bbox_centerline", "points": centerline, "bbox": list(map(float, rect))},
                            "lidar_geometry": {"kind": "cuboid_axis", "axes": axes, "sample_points": [list(map(float, p)) for p in sample_points]},
                        }, rows)

                elif class_type == "buffer_stop" and image_rects:
                    xmax_m = _strong_xmax("buffer_stop")
                    rect = image_rects[0]
                    x0, y0, x1, y1 = rect
                    x0 = max(0, x0 - int(bbox_padding_px)); y0 = max(0, y0 - int(bbox_padding_px))
                    x1 = min(w - 1, x1 + int(bbox_padding_px)); y1 = min(h - 1, y1 + int(bbox_padding_px))
                    image_vals = [x0, y0, x1, y1]
                    rows = []
                    geoms = []
                    for ci, c in enumerate(rec.get("lidar_cuboid", [])):
                        geom = _cuboid_geometry(c.get("val"))
                        if not geom:
                            continue
                        geoms.append(geom)
                        pts = [("center", geom["center"])] + [("corner", p) for p in geom["corners"]]
                        for role, pt in pts:
                            if _is_visible_lidar_point(pt, xmax_m):
                                rows.append(_tsv_row("buffer_stop", rec["object_id"], f"{role}{ci}", _buffer_stop_effective_weight(pt, 1.0), pt, (0, 0, 0), "bbox", image_vals))
                    for v in rec.get("lidar_vec", []):
                        vec_samples = _vec_samples_from_indices(v.get("val"), pcd_xyz, max_points=80) if pcd_xyz is not None else _vec_samples(v.get("val"))
                        for si, pt in enumerate(_downsample_points(_filter_visible_points(vec_samples, xmax_m), 80)):
                            rows.append(_tsv_row("buffer_stop", rec["object_id"], f"point{si}", _buffer_stop_effective_weight(pt, 1.0), pt, (0, 0, 0), "bbox", image_vals))
                    if rows:
                        strong_residual_point_counts["buffer_stop"] += len(rows)
                        append_feature({
                            "object_id": rec["object_id"], "class_type": "buffer_stop", "weight": 1.0, "paired": True,
                            "visible_x_range_m": [0.0, xmax_m],
                            "image_geometry": {"kind": "bbox", "bbox": [float(x0), float(y0), float(x1), float(y1)]},
                            "lidar_geometry": {"kind": "cuboid_corners", "cuboids": geoms},
                        }, rows)

            if has_rgb and has_lidar and len(strong_features) == before_feature_count:
                strong_filtered_by_range_counts[class_type] += 1
        label_objects.append(rec)
    os.makedirs(os.path.dirname(sam_base), exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    if lidar_base:
        os.makedirs(os.path.dirname(lidar_base), exist_ok=True)
    openlabel_scores, openlabel_mean_dists, openlabel_in_image_ratios, label_coordinate_chain_valid = _score_strong_rows_with_openlabel_pose(openlabel, image_sensor, strong_tsv_lines[1:], w, h)
    summary={
        "label_assist_enabled": True,
        "frame_id": frame_id,
        "image_sensor": image_sensor,
        "label_json": label_json,
        "object_count": len(label_objects),
        "paired_counts": paired_counts,
        "bbox_counts": bbox_counts,
        "poly_counts": poly_counts,
        "label_feature_used": True,
        "unsupervised_feature_used": True,
        "label_track_point_count": point_counts["track"],
        "label_static_point_count": point_counts["static"],
        "label_vehicle_point_count": point_counts["vehicle"],
        "label_person_point_count": point_counts["person"],
        "label_teacher_raw_point_counts": raw_point_counts,
        "label_teacher_eligible_point_counts": point_counts,
        "label_teacher_eligible_count": int(sum(point_counts.values())),
        "label_teacher_visible_xmax_m": teacher_visible_xmax_m,
        "label_teacher_image_bbox_padding_m": teacher_image_bbox_padding_m,
        "label_sam_refine_requested": bool(use_sam_refine),
        "strong_features_enabled": bool(strong_features_enabled),
        "strong_label_feature_count": int(len(strong_features)),
        "strong_label_feature_counts": strong_counts,
        "strong_label_residual_point_counts": strong_residual_point_counts,
        "strong_label_raw_object_counts": strong_raw_object_counts,
        "strong_label_filtered_by_range_counts": strong_filtered_by_range_counts,
        "strong_label_missing_rgb_counts": strong_missing_rgb_counts,
        "strong_label_missing_lidar_counts": strong_missing_lidar_counts,
        "strong_label_visible_xmax_m": strong_visible_xmax_m,
        "extrinsic_source": "openlabel_coordinate_systems",
        "openlabel_projection_score_by_class": openlabel_scores,
        "openlabel_projection_mean_dist_px_by_class": openlabel_mean_dists,
        "openlabel_projection_in_image_ratio_by_class": openlabel_in_image_ratios,
        "label_coordinate_chain_valid": bool(label_coordinate_chain_valid),
    }
    coords=[]
    for line in point_lines[1:]:
        parts=line.split()
        if len(parts) >= 3:
            coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if coords:
        arr=np.asarray(coords, dtype=np.float64)
        summary["label_teacher_bbox_lidar_m"]={
            "xmin": float(arr[:,0].min()), "xmax": float(arr[:,0].max()),
            "ymin": float(arr[:,1].min()), "ymax": float(arr[:,1].max()),
            "zmin": float(arr[:,2].min()), "zmax": float(arr[:,2].max()),
        }
    for group,mask in masks.items():
        weight=(mask.astype(np.float32)/255.0)
        if group in ("person","vehicle"):
            k=9
            weight=cv2.GaussianBlur(weight,(k,k),0)
        dist=_dist_from_mask(mask)
        _write_u16(sam_base+f"_label_{group}_weight.png", weight)
        _write_u16(sam_base+f"_label_{group}_dist.png", dist)
        cv2.imwrite(os.path.join(frame_dir, f"label_{group}_mask.png"), mask)
        summary[f"label_{group}_valid_ratio"] = float((mask>0).mean())
    # Agreement with existing heuristic maps, if they already exist.
    for group in ("person","vehicle"):
        hp=sam_base+f"_{group}_weight.png"
        if os.path.isfile(hp):
            hm=cv2.imread(hp, cv2.IMREAD_UNCHANGED)
            if hm is not None:
                if hm.dtype==np.uint16: hm=(hm>int(0.18*65535)).astype(np.uint8)
                else: hm=(hm>int(0.18*255)).astype(np.uint8)
                summary[f"sam_{group}_vs_label_iou"]=_iou(hm, masks[group])
    # Track teacher can diagnose image rail prior by sampling existing rail_weight overlap.
    rp=sam_base+"_rail_weight.png"
    if os.path.isfile(rp):
        rm=cv2.imread(rp, cv2.IMREAD_UNCHANGED)
        if rm is not None:
            rbin=(rm>0).astype(np.uint8)
            summary["image_track_vs_label_score"]=_iou(rbin,masks["track"])
    strong_payload = {"features": strong_features, "summary": summary}
    with open(os.path.join(frame_dir,"label_objects.json"),"w",encoding="utf-8") as f:
        json.dump({"objects": label_objects, "summary": summary}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(frame_dir,"label_strong_features.json"),"w",encoding="utf-8") as f:
        json.dump(strong_payload, f, indent=2, ensure_ascii=False)
    with open(os.path.join(frame_dir,"debug_label_assist.json"),"w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(sam_base+"_debug_label_assist.json","w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if lidar_base:
        with open(lidar_base+"_label_object_points.txt","w",encoding="utf-8") as f:
            f.writelines(point_lines)
        with open(lidar_base+"_label_strong_features.tsv","w",encoding="utf-8") as f:
            f.writelines(strong_tsv_lines)
        with open(lidar_base+"_label_strong_features.json","w",encoding="utf-8") as f:
            json.dump(strong_payload, f, indent=2, ensure_ascii=False)
        with open(lidar_base+"_debug_label_assist.json","w",encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--label-json", required=True)
    ap.add_argument("--frame-id", type=int, required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--image-sensor", default="rgb_center")
    ap.add_argument("--sam-base", required=True)
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--lidar-base", default="")
    ap.add_argument("--lidar-pcd", default="")
    ap.add_argument("--teacher-visible-xmax-m", type=float, default=120.0)
    ap.add_argument("--teacher-image-bbox-padding-m", type=float, default=8.0)
    ap.add_argument("--use-sam-refine", action="store_true")
    ap.add_argument("--strong-features-enabled", action="store_true")
    ap.add_argument("--max-track-samples-per-object", type=int, default=800)
    ap.add_argument("--max-pole-samples-per-object", type=int, default=20)
    ap.add_argument("--bbox-padding-px", type=int, default=12)
    ap.add_argument("--track-visible-xmax-m", type=float, default=120.0)
    ap.add_argument("--switch-visible-xmax-m", type=float, default=120.0)
    ap.add_argument("--catenary-pole-visible-xmax-m", type=float, default=160.0)
    ap.add_argument("--buffer-stop-visible-xmax-m", type=float, default=240.0)
    args=ap.parse_args()
    summary=export_label_assist(
        args.label_json, args.frame_id, args.image, args.image_sensor,
        args.sam_base, args.frame_dir, args.lidar_base or None,
        args.lidar_pcd or None, args.teacher_visible_xmax_m,
        args.teacher_image_bbox_padding_m, args.use_sam_refine,
        args.strong_features_enabled, args.max_track_samples_per_object,
        args.max_pole_samples_per_object, args.bbox_padding_px,
        args.track_visible_xmax_m, args.switch_visible_xmax_m,
        args.catenary_pole_visible_xmax_m, args.buffer_stop_visible_xmax_m)
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
