#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查多帧精修：打印 refinement/state.json 摘要与窗口稳定性统计。"""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refinement_dir", type=str, required=True)
    args = p.parse_args()
    path = os.path.join(args.refinement_dir, "state.json")
    if not os.path.isfile(path):
        print("[Error] 缺少", path)
        return 1
    with open(path, "r", encoding="utf-8") as f:
        st = json.load(f)
    hist = st.get("history") or []
    print("frames:", len(hist), "last_success:", st.get("last_success_frame_id"))
    if len(hist) < 2:
        return 0
    rx = [float(h["rvec"][0]) for h in hist]
    dif = [rx[i] - rx[i - 1] for i in range(1, len(rx))]
    print("mean |delta rx|:", float(sum(abs(x) for x in dif) / len(dif)))
    obs = [float(h.get("observability", 0.0)) for h in hist]
    print("observability min/mean/max:", min(obs), sum(obs) / len(obs), max(obs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
