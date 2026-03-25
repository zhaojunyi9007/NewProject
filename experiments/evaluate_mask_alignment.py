#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask alignment 评估汇总脚本（实验用途，不进入默认主流程）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 mask alignment 指标")
    parser.add_argument("--metrics_dir", default="result/mask_alignment", help="mask alignment json 输出目录")
    parser.add_argument("--topk", type=int, default=5, help="输出最低分的前K帧")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    files = sorted(metrics_dir.glob("*_mask_alignment.json"))
    if not files:
        print(f"[MaskAlignmentEval] 未找到指标文件: {metrics_dir}")
        return 0

    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            rows.append(json.load(fp))

    scores = [x["alignment_score"] for x in rows]
    edge_in_mask = [x["edge_in_mask_ratio"] for x in rows]
    avg_score = sum(scores) / len(scores)
    avg_edge_in_mask = sum(edge_in_mask) / len(edge_in_mask)

    print(f"[MaskAlignmentEval] files={len(rows)} avg_score={avg_score:.4f} avg_edge_in_mask={avg_edge_in_mask:.4f}")

    worst = sorted(rows, key=lambda x: x["alignment_score"])[: max(1, args.topk)]
    print("[MaskAlignmentEval] worst frames:")
    for row in worst:
        print(
            f"  frame={int(row['frame_id']):010d} "
            f"score={row['alignment_score']:.4f} "
            f"edge_in_mask={row['edge_in_mask_ratio']:.4f} "
            f"mask_cov={row['mask_coverage_ratio']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())