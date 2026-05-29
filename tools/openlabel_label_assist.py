
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


def export_label_assist(label_json: str, frame_id: int, image_path: str, image_sensor: str, sam_base: str, frame_dir: str, lidar_base: str|None=None, lidar_pcd_path: str|None=None, teacher_visible_xmax_m: float = 120.0, teacher_image_bbox_padding_m: float = 8.0, use_sam_refine: bool = False):
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
    teacher_visible_xmax_m = float(teacher_visible_xmax_m)
    teacher_image_bbox_padding_m = float(teacher_image_bbox_padding_m)

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
        label_objects.append(rec)
    os.makedirs(os.path.dirname(sam_base), exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    if lidar_base:
        os.makedirs(os.path.dirname(lidar_base), exist_ok=True)
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
    with open(os.path.join(frame_dir,"label_objects.json"),"w",encoding="utf-8") as f:
        json.dump({"objects": label_objects, "summary": summary}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(frame_dir,"debug_label_assist.json"),"w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(sam_base+"_debug_label_assist.json","w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if lidar_base:
        with open(lidar_base+"_label_object_points.txt","w",encoding="utf-8") as f:
            f.writelines(point_lines)
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
    args=ap.parse_args()
    summary=export_label_assist(args.label_json,args.frame_id,args.image,args.image_sensor,args.sam_base,args.frame_dir,args.lidar_base or None,args.lidar_pcd or None,args.teacher_visible_xmax_m,args.teacher_image_bbox_padding_m,args.use_sam_refine)
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
