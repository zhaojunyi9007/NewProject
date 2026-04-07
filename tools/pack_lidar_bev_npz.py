#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 lidar_extractor 输出的 _bev_channels.bin + _bev_meta.json 打包为 bev_maps.npz。"""

from __future__ import annotations

import json
import os
import struct
import sys


def pack_bev_npz(output_base: str) -> bool:
    meta_path = output_base + "_bev_meta.json"
    if not os.path.isfile(meta_path):
        return False
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    bin_path = meta.get("binary", "")
    if not bin_path or not os.path.isfile(bin_path):
        alt = output_base + "_bev_channels.bin"
        if os.path.isfile(alt):
            bin_path = alt
        else:
            print(f"[pack_bev] Binary not found: {bin_path}", file=sys.stderr)
            return False

    import numpy as np

    with open(bin_path, "rb") as f:
        magic = f.read(8)
        if magic != b"EDGEBEV1":
            print("[pack_bev] Bad magic", magic, file=sys.stderr)
            return False
        nx, ny, nch = struct.unpack("iii", f.read(12))
        f.read(16)  # xmin, ymin, res, pad
        raw = np.frombuffer(f.read(), dtype=np.float32)
    expected = nx * ny * nch
    if raw.size != expected:
        print(f"[pack_bev] Size mismatch {raw.size} vs {expected}", file=sys.stderr)
        return False
    arr = raw.reshape((nch, ny, nx))
    names = meta.get(
        "channels",
        [
            "density",
            "max_height",
            "min_height",
            "mean_height",
            "mean_intensity",
            "verticality",
            "rail_probability",
        ],
    )
    out: dict = {names[i]: arr[i] for i in range(min(len(names), nch))}
    out["bev_xmin"] = np.float32(meta.get("xmin", 0.0))
    out["bev_ymin"] = np.float32(meta.get("ymin", 0.0))
    out["bev_resolution"] = np.float32(meta.get("resolution", 0.2))
    out_path = output_base + "_bev_maps.npz"
    np.savez_compressed(out_path, **out)
    print(f"[pack_bev] Wrote {out_path}")
    return True


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: pack_lidar_bev_npz.py <output_base>", file=sys.stderr)
        sys.exit(1)
    ok = pack_bev_npz(sys.argv[1])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
