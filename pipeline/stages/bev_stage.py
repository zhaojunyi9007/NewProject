#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 4：BEV 粗初始化（调用 build/bev_initializer，写入 bev_init/<frame>/）。"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time

from pipeline.context import RuntimeContext
from pipeline.datasets import get_adapter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TOOLS = os.path.join(_REPO_ROOT, "tools")
_BUILD_BEV_INIT = os.path.join(_REPO_ROOT, "build", "bev_initializer")


def _parse_pose_after_bev(path: str):
    nums: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for part in s.split():
                try:
                    nums.append(float(part))
                except ValueError:
                    pass
    if len(nums) < 6:
        return None
    return {"rvec": nums[0:3], "tvec": nums[3:6]}


def _load_json_dict(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _bev_rail_nonzero_ratio(path: str) -> float:
    try:
        import numpy as np

        with open(path, "rb") as f:
            if f.read(8) != b"EDGEBEV1":
                return 0.0
            nx, ny, nch = struct.unpack("iii", f.read(12))
            f.read(16)
            n = int(nx) * int(ny)
            if n <= 0 or nch <= 0:
                return 0.0
            arr = np.frombuffer(f.read(n * int(nch) * 4), dtype=np.float32)
            if arr.size < n * int(nch):
                return 0.0
            rail = arr[:n] if nch == 1 else arr[(int(nch) - 1) * n : int(nch) * n]
            return float((rail > 1e-4).mean())
    except Exception:
        return 0.0



def _select_lidar_bev_input(raw_bin: str, refined_bin: str, bev_cfg: dict) -> tuple[str, str, float]:
    """Select the LiDAR BEV input for BEV matcher, preferring non-empty refined rail BEV when enabled."""
    if bool(bev_cfg.get("use_refined_lidar_rail", False)) and refined_bin and os.path.isfile(refined_bin):
        ratio = _bev_rail_nonzero_ratio(refined_bin)
        if ratio > 0.0:
            return refined_bin, "refined", ratio
    return raw_bin, "raw_bev_channels", _bev_rail_nonzero_ratio(raw_bin)


def _maybe_export_refined_lidar_rail(
    fid: str,
    lidar_root: str,
    pseudo_npz: str,
    bev_cfg: dict,
    sem_cfg: dict,
) -> tuple[str, dict]:
    refined_bin = os.path.join(os.path.abspath(lidar_root), f"{fid}_rail_bev_refined.bin")
    debug_path = os.path.join(os.path.abspath(lidar_root), f"{fid}_rail_bev_debug.json")
    if not bool(bev_cfg.get("use_refined_lidar_rail", False)):
        return refined_bin, _load_json_dict(debug_path)
    if os.path.isfile(refined_bin) and _bev_rail_nonzero_ratio(refined_bin) > 0.0:
        return refined_bin, _load_json_dict(debug_path)

    bev_npz = os.path.join(os.path.abspath(lidar_root), f"{fid}_bev_maps.npz")
    if not os.path.isfile(bev_npz):
        return refined_bin, _load_json_dict(debug_path)
    try:
        from lidar_bev_rail_points import export_lidar_bev_rail_points  # noqa: WPS433

        export_lidar_bev_rail_points(
            bev_npz,
            os.path.join(os.path.abspath(lidar_root), f"{fid}_rail_bev_points.txt"),
            min_prob=float(sem_cfg.get("lidar_bev_rail_sample_min_prob", 0.15)),
            stride_cells=int(sem_cfg.get("lidar_bev_rail_sample_stride_cells", 2)),
            max_points=int(sem_cfg.get("lidar_bev_rail_sample_max_points", 8000)),
            reference_z=float(sem_cfg.get("lidar_bev_rail_sample_reference_z", 0.0)),
            oracle_npz_path=pseudo_npz if os.path.isfile(pseudo_npz) else None,
            oracle_overlap_dilate_cells=int(sem_cfg.get("lidar_bev_oracle_overlap_dilate_cells", 3)),
            min_component_cells=int(sem_cfg.get("lidar_bev_rail_min_component_cells", 20)),
            debug_path=debug_path,
            refined_png_path=os.path.join(os.path.abspath(lidar_root), f"{fid}_rail_bev_refined.png"),
            refined_bin_path=refined_bin,
        )
    except Exception as exc:
        print(f"[Warning] Export refined LiDAR rail BEV failed: {fid}: {exc}")
    return refined_bin, _load_json_dict(debug_path)



def _diagnose_rail_bev_alignment(fid: str, lidar_root: str, pseudo_npz: str, refined_png: str, bev_cfg: dict) -> dict:
    if not bool(bev_cfg.get("auto_test_bev_axis_transforms", False)):
        return {}
    bev_npz = os.path.join(os.path.abspath(lidar_root), f"{fid}_bev_maps.npz")
    if not os.path.isfile(bev_npz) or not os.path.isfile(pseudo_npz):
        return {}
    out_path = os.path.join(os.path.abspath(lidar_root), f"{fid}_rail_bev_alignment_debug.json")
    try:
        from diagnose_rail_bev_alignment import diagnose  # noqa: WPS433

        obj = diagnose(
            pseudo_npz,
            bev_npz,
            refined_png if os.path.isfile(refined_png) else None,
            max_shift_cells=int(bev_cfg.get("rail_alignment_max_shift_cells", 120)),
            min_overlap=float(bev_cfg.get("rail_alignment_min_overlap", 0.15)),
            test_transforms=bool(bev_cfg.get("auto_test_bev_axis_transforms", True)),
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return obj
    except Exception as exc:
        print(f"[Warning] Rail BEV alignment diagnosis failed: {fid}: {exc}")
        return {}


def _delta_within_oracle_limits(parsed: dict, base_r: list[float], base_t: list[float], bev_cfg: dict) -> tuple[bool, str]:
    max_tx = float(bev_cfg.get("oracle_max_abs_tx_m", 2.0))
    max_ty = float(bev_cfg.get("oracle_max_abs_ty_m", 1.0))
    max_yaw = float(bev_cfg.get("oracle_max_abs_yaw_deg", 3.0))
    dt = [float(parsed["tvec"][i]) - float(base_t[i]) for i in range(3)]
    if abs(dt[0]) > max_tx:
        return False, f"oracle_tx_delta_exceeds_{max_tx}"
    if abs(dt[1]) > max_ty:
        return False, f"oracle_ty_delta_exceeds_{max_ty}"
    yaw_delta_deg = abs(float(parsed["rvec"][2]) - float(base_r[2])) * 180.0 / 3.141592653589793
    if yaw_delta_deg > max_yaw:
        return False, f"oracle_yaw_delta_exceeds_{max_yaw}"
    return True, ""


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段 bev] BEV 粗初始化 (Phase 4)")
    print("=" * 40)

    bev_cfg = context.config.get("bev") or {}
    sem_cfg = context.config.get("semantic_calib") or {}
    if not bool(bev_cfg.get("enabled", False)):
        print("[Info] bev.enabled=false，跳过")
        return

    out_root = context.paths.get("bev_init") or context.config.get("data", {}).get("bev_init_output_dir", "")
    if not out_root:
        print("[Error] 缺少 bev_init 输出目录")
        return
    os.makedirs(out_root, exist_ok=True)

    img_root = context.paths.get("image_features") or context.config.get("data", {}).get("image_features_output_dir", "")
    lidar_root = context.paths.get("lidar") or context.config.get("data", {}).get("lidar_output_dir", "")
    if not img_root or not lidar_root:
        print("[Error] 需要 image_features_output_dir 与 lidar_output_dir")
        return

    sys.path.insert(0, _TOOLS)
    from bev_export_image_bin import export_image_rail_bin  # noqa: E402
    from score_breakdown_io import write_unified_debug_json  # noqa: E402

    bev_init = str(bev_cfg.get("initializer_binary") or "").strip()
    exe = bev_init if bev_init and os.path.isfile(bev_init) else _BUILD_BEV_INIT
    if not os.path.isfile(exe):
        print(f"[Error] 未找到可执行文件 bev_initializer: {exe}（请先 cmake 编译）")
        return

    adapter = get_adapter(context.config)
    ext = adapter.load_initial_extrinsic()
    if not ext:
        ie = context.config.get("calibration", {}).get("initial_extrinsic", {})
        rvec = [float(x) for x in ie.get("rotation", [0.0, 0.0, 0.0])]
        tvec = [float(x) for x in ie.get("translation", [0.0, 0.0, 0.0])]
    else:
        rvec, tvec = ext
        rvec = [float(x) for x in rvec]
        tvec = [float(x) for x in tvec]

    env = os.environ.copy()
    env["EDGECALIB_BEV_YAW_MIN_DEG"] = str(bev_cfg.get("yaw_min_deg", -6.0))
    env["EDGECALIB_BEV_YAW_MAX_DEG"] = str(bev_cfg.get("yaw_max_deg", 6.0))
    env["EDGECALIB_BEV_YAW_STEP_DEG"] = str(bev_cfg.get("yaw_step_deg", 1.0))
    env["EDGECALIB_BEV_TX_MIN_M"] = str(bev_cfg.get("tx_min_m", -2.0))
    env["EDGECALIB_BEV_TX_MAX_M"] = str(bev_cfg.get("tx_max_m", 2.0))
    env["EDGECALIB_BEV_TY_MIN_M"] = str(bev_cfg.get("ty_min_m", -2.0))
    env["EDGECALIB_BEV_TY_MAX_M"] = str(bev_cfg.get("ty_max_m", 2.0))
    env["EDGECALIB_BEV_TRANS_STEP_M"] = str(bev_cfg.get("trans_step_m", 0.5))

    last_pose = None
    context.bev_pose_by_frame.clear()
    for frame_id in context.frame_ids:
        fid = f"{frame_id:010d}"
        frame_dir = os.path.join(out_root, fid)
        os.makedirs(frame_dir, exist_ok=True)

        pseudo_npz = os.path.join(os.path.abspath(img_root), fid, "pseudo_bev.npz")
        raw_lidar_bin = os.path.join(os.path.abspath(lidar_root), f"{fid}_bev_channels.bin")
        if not os.path.isfile(pseudo_npz):
            print(f"[Warning] Missing {pseudo_npz}, skip frame {fid}")
            continue
        refined_lidar_bin, refined_dbg = _maybe_export_refined_lidar_rail(fid, lidar_root, pseudo_npz, bev_cfg, sem_cfg)
        lidar_bin, lidar_rail_source, selected_lidar_rail_ratio = _select_lidar_bev_input(
            raw_lidar_bin, refined_lidar_bin, bev_cfg
        )
        if not os.path.isfile(lidar_bin):
            print(f"[Warning] Missing {lidar_bin} (LiDAR Phase3 BEV required), skip frame {fid}")
            continue

        img_bin = os.path.join(frame_dir, "image_rail_bev.bin")
        export_dbg: dict = {}
        if not export_image_rail_bin(pseudo_npz, img_bin, bev_cfg, export_dbg):
            print(f"[Warning] 导出 image BEV 失败: {fid}")
            continue
        export_dbg["lidar_rail_nonzero_ratio"] = selected_lidar_rail_ratio
        export_dbg["lidar_rail_source"] = lidar_rail_source
        export_dbg["refined_lidar_rail_nonzero_ratio"] = _bev_rail_nonzero_ratio(refined_lidar_bin) if os.path.isfile(refined_lidar_bin) else 0.0
        export_dbg["refined_lidar_rail_fallback_used"] = bool(refined_dbg.get("lidar_rail_refine_fallback_used", False))
        export_dbg["rail_refinement_valid"] = bool(refined_dbg.get("rail_refinement_valid", True))
        export_dbg["rail_refinement_mismatch"] = bool(refined_dbg.get("rail_refinement_mismatch", False))
        align_dbg = _diagnose_rail_bev_alignment(
            fid,
            lidar_root,
            pseudo_npz,
            os.path.join(os.path.abspath(lidar_root), f"{fid}_rail_bev_refined.png"),
            bev_cfg,
        )
        if align_dbg:
            export_dbg["rail_bev_alignment_valid"] = bool(align_dbg.get("rail_bev_alignment_valid", False))
            export_dbg["rail_bev_alignment_mismatch"] = not bool(align_dbg.get("rail_bev_alignment_valid", False))
            export_dbg["rail_bev_alignment_best_transform"] = align_dbg.get("best_transform", "")
            export_dbg["rail_bev_alignment_best_shift_x_cells"] = float((align_dbg.get("best_shift_cells") or [0, 0])[0])
            export_dbg["rail_bev_alignment_best_shift_y_cells"] = float((align_dbg.get("best_shift_cells") or [0, 0])[1])
            export_dbg["rail_bev_alignment_best_score"] = float(align_dbg.get("best_shift_score", 0.0) or 0.0)
        rail_quality = _load_json_dict(os.path.join(os.path.abspath(img_root), fid, "rail_quality.json"))
        oracle_rail = (
            bool(rail_quality.get("label_track_prior_used", False))
            and bool(rail_quality.get("enabled", False))
            and float(rail_quality.get("quality_score", 0.0) or 0.0) >= float(bev_cfg.get("oracle_min_quality_score", 0.8))
        )

        init_path = os.path.join(frame_dir, "init_pose.txt")
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("# rvec (3)\n")
            f.write(f"{rvec[0]} {rvec[1]} {rvec[2]}\n")
            f.write("# tvec (3)\n")
            f.write(f"{tvec[0]} {tvec[1]} {tvec[2]}\n")

        cmd = [exe, lidar_bin, img_bin, init_path, frame_dir, fid]
        print(f"  BEV init: frame={fid}")
        print(f"    {' '.join(cmd)}")
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, check=True, env=env, cwd=_REPO_ROOT)
        except subprocess.CalledProcessError as e:
            print(f"[Error] bev_initializer 失败 (exit={e.returncode})，跳过帧 {fid}")
            continue
        elapsed = time.perf_counter() - t0

        after_path = os.path.join(frame_dir, "pose_after_bev.txt")
        bev_dbg = os.path.join(frame_dir, "debug_bev_score.json")
        breakdown: dict = {}
        if os.path.isfile(bev_dbg):
            try:
                with open(bev_dbg, "r", encoding="utf-8") as f:
                    breakdown = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        if os.path.isfile(after_path):
            parsed = _parse_pose_after_bev(after_path)
            min_rail_score = float(
                bev_cfg.get("oracle_min_rail_score_to_apply", bev_cfg.get("min_rail_score_to_apply", 0.01))
                if oracle_rail
                else bev_cfg.get("min_rail_score_to_apply", 0.01)
            )
            actual_rail_score = float(breakdown.get("rail_score", 0.0)) if breakdown else 0.0
            reject_reason = ""
            if actual_rail_score < min_rail_score:
                reject_reason = f"rail_score_below_{min_rail_score}"
                parsed = None
            elif parsed and oracle_rail:
                ok_delta, reject_reason = _delta_within_oracle_limits(parsed, rvec, tvec, bev_cfg)
                if not ok_delta:
                    parsed = None
            if parsed:
                last_pose = parsed
                context.bev_pose_by_frame[frame_id] = parsed
                breakdown["delta_applied"] = True
                breakdown["reject_reason"] = ""
                print(f"  [BEV] 帧 {fid} BEV delta 已应用：rail_score={actual_rail_score:.4f} >= {min_rail_score}")
            else:
                if not reject_reason:
                    reject_reason = "pose_parse_failed"
                print(
                    f"  [BEV] 帧 {fid} BEV delta 被拒绝：reason={reject_reason}, "
                    f"rail_score={actual_rail_score:.6f}, min_rail_score_to_apply={min_rail_score}，保留原始 init_pose"
                )
                breakdown["delta_applied"] = False
                breakdown["reject_reason"] = reject_reason
            breakdown.update(export_dbg)
            breakdown["oracle_rail_mode"] = oracle_rail
            breakdown["min_rail_score_to_apply"] = min_rail_score
            breakdown["best_score_raw"] = float(breakdown.get("best_score_raw", breakdown.get("rail_score", 0.0)))
            breakdown["best_score_norm"] = float(breakdown.get("rail_score", 0.0))
            yaw_deg = float(breakdown.get("yaw_rad", 0.0)) * 180.0 / 3.141592653589793
            tx_m = float(breakdown.get("tx_m", 0.0))
            ty_m = float(breakdown.get("ty_m", 0.0))
            eps = 1e-6
            breakdown["bev_search_hit_yaw_boundary"] = bool(
                abs(yaw_deg - float(bev_cfg.get("yaw_min_deg", -6.0))) <= eps
                or abs(yaw_deg - float(bev_cfg.get("yaw_max_deg", 6.0))) <= eps
            )
            breakdown["bev_search_hit_tx_boundary"] = bool(
                abs(tx_m - float(bev_cfg.get("tx_min_m", -2.0))) <= eps
                or abs(tx_m - float(bev_cfg.get("tx_max_m", 2.0))) <= eps
            )
            breakdown["bev_search_hit_ty_boundary"] = bool(
                abs(ty_m - float(bev_cfg.get("ty_min_m", -2.0))) <= eps
                or abs(ty_m - float(bev_cfg.get("ty_max_m", 2.0))) <= eps
            )
            try:
                with open(bev_dbg, "w", encoding="utf-8") as f:
                    json.dump(breakdown, f, ensure_ascii=False)
            except OSError:
                pass
            write_unified_debug_json(
                os.path.join(frame_dir, "debug_score_breakdown.json"),
                stage="bev",
                frame_id=fid,
                input_pose={"rvec": list(rvec), "tvec": list(tvec)},
                output_pose=parsed,
                breakdown=breakdown if breakdown else None,
                elapsed_sec=elapsed,
                meta={
                    "bev_initializer": os.path.basename(exe),
                    "rail_score": actual_rail_score,
                    "min_rail_score_to_apply": min_rail_score,
                    "bev_delta_applied": parsed is not None,
                },
            )

    if last_pose:
        context.current_pose_bev = last_pose
        print(f"[Info] current_pose_bev 已更新（末帧）: rvec={last_pose['rvec']}")
    else:
        print("[Warning] 未得到任何 pose_after_bev.txt")

    print(f"\n[完成] BEV 粗初始化输出目录: {out_root}")
