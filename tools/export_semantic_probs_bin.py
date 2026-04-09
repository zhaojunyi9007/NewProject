#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 semantic_probs.npy (H, W, C) float32 导出为 C++ 可读的 EDGESEM1 二进制。"""

from __future__ import annotations

import struct
import sys

import numpy as np


def npy_to_edgecalib_bin(npy_path: str, out_bin: str) -> bool:
    try:
        arr = np.load(npy_path)
    except OSError as exc:
        print(f"[export_semantic_probs_bin] Cannot load {npy_path}: {exc}")
        return False

    if arr.ndim != 3:
        print("[export_semantic_probs_bin] expected HxWxC array")
        return False

    arr = np.ascontiguousarray(arr.astype(np.float32))
    h, w, c = arr.shape

    with open(out_bin, "wb") as f:
        f.write(b"EDGESEM1")
        f.write(struct.pack("iii", int(w), int(h), int(c)))
        f.write(arr.tobytes())

    return True


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "Usage: export_semantic_probs_bin.py <semantic_probs.npy> <out.edgecalib.bin>",
            file=sys.stderr,
        )
        return 1
    return 0 if npy_to_edgecalib_bin(argv[1], argv[2]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
