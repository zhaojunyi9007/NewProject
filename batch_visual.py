#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch visual parameter search for EdgeCalib results.

Generates visualization images by sweeping pitch (r_vec[1]) and tz (t_vec[2])
around a provided center. This avoids re-running the full pipeline and only
invokes visual_result.py with different extrinsics.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path


def build_values(center: float, span: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    count = int(math.floor(span / step))
    values = []
    for i in range(-count, count + 1):
        values.append(center + i * step)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch sweep pitch/tz by calling visual_result.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--img", required=True, help="Image path.")
    parser.add_argument("--feature_base", required=True, help="Feature base path.")
    parser.add_argument("--r_vec", nargs=3, type=float, required=True,
                        help="Rotation vector (rx ry rz).")
    parser.add_argument("--t_vec", nargs=3, type=float, required=True,
                        help="Translation vector (tx ty tz).")
    parser.add_argument("--pitch_span", type=float, default=0.06,
                        help="Half span around pitch (r_vec[1]) to sweep.")
    parser.add_argument("--pitch_step", type=float, default=0.02,
                        help="Pitch sweep step size.")
    parser.add_argument("--tz_span", type=float, default=0.12,
                        help="Half span around tz (t_vec[2]) to sweep.")
    parser.add_argument("--tz_step", type=float, default=0.04,
                        help="tz sweep step size.")
    parser.add_argument("--output_dir", default="result/visual_search",
                        help="Directory to write output images.")
    parser.add_argument("--subsample", type=int, default=5,
                        help="Point subsampling factor passed to visual_result.py.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pitch_values = build_values(args.r_vec[1], args.pitch_span, args.pitch_step)
    tz_values = build_values(args.t_vec[2], args.tz_span, args.tz_step)

    total = len(pitch_values) * len(tz_values)
    idx = 0

    for pitch in pitch_values:
        for tz in tz_values:
            idx += 1
            out_name = (
                f"pitch_{pitch:+.3f}_tz_{tz:+.3f}.png"
            )
            out_path = output_dir / out_name
            cmd = [
                "python",
                "visual_result.py",
                "--img",
                args.img,
                "--feature_base",
                args.feature_base,
                "--r_vec",
                f"{args.r_vec[0]}",
                f"{pitch}",
                f"{args.r_vec[2]}",
                "--t_vec",
                f"{args.t_vec[0]}",
                f"{args.t_vec[1]}",
                f"{tz}",
                "--output",
                str(out_path),
                "--subsample",
                str(args.subsample),
            ]
            print(f"[{idx}/{total}] pitch={pitch:+.3f}, tz={tz:+.3f} -> {out_path}")
            subprocess.run(cmd, check=True)

    print(f"[Done] Wrote {total} images to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())