#!/usr/bin/env python3
"""Extract lightweight person/vehicle-like object cluster centroids from LiDAR semantic points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEM_VERTICAL_STRUCTURE = 3
SEM_PLATFORM_OR_BUILDING = 4
SEM_VEHICLE_LIKE = 5
SEM_PERSON_LIKE = 7


def _load_points(path: str) -> np.ndarray:
    rows = []
    for line in Path(path).read_text(encoding='utf-8', errors='ignore').splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            x, y, z = map(float, parts[:3])
            weight = float(parts[8])
            sid = int(float(parts[9]))
            rows.append((x, y, z, weight, sid))
        except ValueError:
            continue
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def _components_xy(points: np.ndarray, cell: float, gap_cells: int):
    if points.size == 0:
        return []
    xy = points[:, :2]
    mn = xy.min(axis=0)
    ij = np.floor((xy - mn) / max(cell, 1e-3)).astype(np.int32)
    occupied = {tuple(v) for v in ij.tolist()}
    seen = set()
    comps = []
    neigh = [(dx, dy) for dx in range(-gap_cells, gap_cells + 1) for dy in range(-gap_cells, gap_cells + 1)]
    for key in list(occupied):
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        cells = []
        while stack:
            c = stack.pop()
            cells.append(c)
            for dx, dy in neigh:
                nb = (c[0] + dx, c[1] + dy)
                if nb in occupied and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        cell_set = set(cells)
        idx = np.array([tuple(v) in cell_set for v in ij.tolist()], dtype=bool)
        comps.append(points[idx])
    return comps


def extract(points: np.ndarray, cfg: dict) -> list[dict]:
    out = []
    if points.size == 0:
        return out
    candidate = points[(points[:, 2] > float(cfg.get('min_z', -0.5))) & (points[:, 2] < float(cfg.get('max_z', 4.0)))]
    candidate = candidate[np.isin(candidate[:, 4].astype(np.int32), [SEM_VERTICAL_STRUCTURE, SEM_PLATFORM_OR_BUILDING, SEM_VEHICLE_LIKE])]
    for comp in _components_xy(candidate, float(cfg.get('cluster_cell_m', 0.5)), int(cfg.get('cluster_gap_cells', 1))):
        if comp.shape[0] < int(cfg.get('min_cluster_points', 18)):
            continue
        mn = comp[:, :3].min(axis=0)
        mx = comp[:, :3].max(axis=0)
        dims = mx - mn
        length, width = sorted([float(dims[0]), float(dims[1])], reverse=True)
        height = float(dims[2])
        centroid = comp[:, :3].mean(axis=0)
        class_id = 0
        conf = min(1.0, comp.shape[0] / 120.0)
        if (1.2 <= height <= 2.4 and width <= 1.2 and length <= 1.8 and comp.shape[0] >= int(cfg.get('person_min_points', 12))):
            class_id = SEM_PERSON_LIKE
        elif (1.0 <= length <= 7.0 and 0.8 <= width <= 3.5 and 0.6 <= height <= 3.2 and comp.shape[0] >= int(cfg.get('vehicle_min_points', 25))):
            class_id = SEM_VEHICLE_LIKE
        if class_id:
            out.append({
                'class_id': int(class_id),
                'centroid': [float(x) for x in centroid],
                'bbox_min': [float(x) for x in mn],
                'bbox_max': [float(x) for x in mx],
                'length_m': length,
                'width_m': width,
                'height_m': height,
                'point_count': int(comp.shape[0]),
                'confidence': float(conf),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('semantic_points')
    ap.add_argument('out_txt')
    ap.add_argument('--meta-json', default='')
    args = ap.parse_args()
    pts = _load_points(args.semantic_points)
    clusters = extract(pts, {})
    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_txt, 'w', encoding='utf-8') as f:
        f.write('# class_id cx cy cz confidence\n')
        for c in clusters:
            x, y, z = c['centroid']
            f.write(f"{c['class_id']} {x:.6f} {y:.6f} {z:.6f} {c['confidence']:.6f}\n")
    if args.meta_json:
        meta = {
            'object_cluster_count': len(clusters),
            'vehicle_cluster_count': sum(1 for c in clusters if c['class_id'] == SEM_VEHICLE_LIKE),
            'person_cluster_count': sum(1 for c in clusters if c['class_id'] == SEM_PERSON_LIKE),
            'clusters': clusters,
        }
        Path(args.meta_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(len(clusters))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
