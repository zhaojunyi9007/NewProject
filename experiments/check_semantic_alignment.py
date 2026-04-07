#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查语义对齐：读取 calib 的 debug_score_breakdown.json 与语义概率文件是否存在。"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--calib_dir", type=str, required=True)
    p.add_argument("--frame_id", type=str, required=True, help="10 位帧号，如 0000000012")
    args = p.parse_args()
    fid = args.frame_id
    path = os.path.join(args.calib_dir, f"{fid}_debug_score_breakdown.json")
    if not os.path.isfile(path):
        print("[Error] 缺少", path)
        return 1
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    print(json.dumps(doc, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
