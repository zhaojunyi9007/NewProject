
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


def _cuboid_samples(cuboid_val):
    if not cuboid_val or len(cuboid_val) < 10:
        return []
    x,y,z = [float(v) for v in cuboid_val[:3]]
    sx,sy,sz = [abs(float(v)) for v in cuboid_val[7:10]]
    # OSDaR calibration labels in this sequence use identity-like rotations for relevant objects.
    # Use axis-aligned samples as robust teacher points; exact box orientation is not required for attraction residuals.
    samples=[(x,y,z)]
    for dx in (-0.5,0.5):
        for dy in (-0.5,0.5):
            for dz in (-0.5,0.5):
                samples.append((x+dx*sx, y+dy*sy, z+dz*sz))
    samples.append((x,y,z+0.5*sz))
    samples.append((x,y,z-0.5*sz))
    return samples


def _iou(a: np.ndarray, b: np.ndarray):
    aa=(a>0).astype(np.uint8); bb=(b>0).astype(np.uint8)
    inter=float(np.logical_and(aa,bb).sum()); uni=float(np.logical_or(aa,bb).sum())
    return inter/uni if uni>0 else 0.0


def export_label_assist(label_json: str, frame_id: int, image_path: str, image_sensor: str, sam_base: str, frame_dir: str, lidar_base: str|None=None):
    image=cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    h,w=image.shape[:2]
    data=json.loads(Path(label_json).read_text(encoding="utf-8"))
    openlabel=data.get("openlabel", data)
    rows=_extract_frame(openlabel, frame_id, image_sensor)
    masks={k:np.zeros((h,w), dtype=np.uint8) for k in ["person","vehicle","track","static"]}
    paired_counts={k:0 for k in ["person","vehicle","track","static"]}
    bbox_counts={k:0 for k in ["person","vehicle","track","static"]}
    poly_counts={k:0 for k in ["person","vehicle","track","static"]}
    point_lines=["# x y z class_id weight object_id geometry_role\n"]
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
                    point_lines.append(f"{x:.6f} {y:.6f} {z:.6f} {cid} {weight:.6f} {rec['object_id']} {group}_cuboid{ci}_sample{si}\n")
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
    args=ap.parse_args()
    summary=export_label_assist(args.label_json,args.frame_id,args.image,args.image_sensor,args.sam_base,args.frame_dir,args.lidar_base or None)
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
