#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速检查 BEV 粗初始化：读取 LiDAR/图像 BEV 与 debug_bev_score.json，可选生成拼图。"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "tools"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bev_frame_dir", type=str, required=True, help="bev_init/<frame_id>/")
    p.add_argument("--lidar_base", type=str, default="", help="lidar_features/<frame_id> 前缀（无后缀）")
    p.add_argument("--image_feat_dir", type=str, default="", help="image_features/<frame_id>/")
    args = p.parse_args()

    d = args.bev_frame_dir
    js = os.path.join(d, "debug_bev_score.json")
    if os.path.isfile(js):
        with open(js, "r", encoding="utf-8") as f:
            j = json.load(f)
        print("[debug_bev_score.json]", json.dumps(j, indent=2, ensure_ascii=False))
    else:
        print("[Warning] 缺少 debug_bev_score.json:", js)

    lb = args.lidar_base.strip()
    imd = args.image_feat_dir.strip()
    if lb and imd:
        import visualize_diag as vd

        out = os.path.join(d, "check_bev_panel.png")
        if vd.render_bev_panel(lb, imd, out):
            print("[OK] 拼图已写:", out)
    elif lb or imd:
        print("[Info] 同时提供 --lidar_base 与 --image_feat_dir 可生成 BEV 拼图")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
