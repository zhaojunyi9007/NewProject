#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import sys
import time

from pipeline.context import RuntimeContext
from pipeline.datasets import get_adapter
from pipeline.optimizer.constraint_adapter import get_optimizer_constraint_adapter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_OPTIMIZER_ARG_SUPPORT_CACHE: dict[tuple[str, str], bool] = {}


def _optimizer_binary_supports_arg(optimizer_path: str, arg_name: str) -> bool:
    key = (os.path.abspath(optimizer_path), arg_name)
    if key in _OPTIMIZER_ARG_SUPPORT_CACHE:
        return _OPTIMIZER_ARG_SUPPORT_CACHE[key]
    try:
        with open(optimizer_path, "rb") as f:
            data = f.read()
        supported = arg_name.encode("utf-8") in data
    except OSError:
        supported = True
    _OPTIMIZER_ARG_SUPPORT_CACHE[key] = supported
    return supported


def _parse_calib_breakdown(path: str) -> dict:
    out: dict = {}
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if ":" not in s:
                continue
            k, v = s.split(":", 1)
            k = k.strip()
            if k in {"r", "t"}:
                # Pose vectors are parsed by _parse_calib_pose; do not pollute breakdown with scalar fragments.
                continue
            v = v.strip().split()[0] if v.strip() else ""
            if not v:
                continue
            try:
                out[k] = float(v)
            except ValueError:
                continue
    return out


def _parse_calib_pose(path: str):
    if not os.path.isfile(path):
        return None
    kv: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or ":" not in s:
                continue
            k, v = s.split(":", 1)
            kv[k.strip()] = v.strip()
    if "r" not in kv or "t" not in kv:
        return None
    try:
        r = [float(x) for x in kv["r"].split()[:3]]
        t = [float(x) for x in kv["t"].split()[:3]]
    except ValueError:
        return None
    if len(r) != 3 or len(t) != 3:
        return None
    return {"rvec": r, "tvec": t}


def _load_json_dict(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}



def _is_oracle_rail_enabled(rail_quality: dict, min_quality: float) -> bool:
    return (
        bool(rail_quality.get("label_track_prior_used", False))
        and bool(rail_quality.get("enabled", False))
        and float(rail_quality.get("quality_score", 0.0) or 0.0) >= float(min_quality)
    )


def _write_invalid_calib_result(path: str, rvec: list[float], tvec: list[float], reason: str, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    extra = extra or {}
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"r: {rvec[0]} {rvec[1]} {rvec[2]}\n")
        f.write(f"t: {tvec[0]} {tvec[1]} {tvec[2]}\n")
        f.write("Score: 0\n")
        f.write("semantic_js_divergence: 0\n")
        f.write("semantic_hist_similarity: 0\n")
        f.write("edge_term_norm: 0\n")
        f.write("rail_term_norm: 0\n")
        f.write("final_pose_valid: 0\n")
        f.write("rail_gate_failed: 1\n")
        f.write("optimizer_skipped: 1\n")
        f.write(f"invalid_reason: {reason}\n")
        for k, v in extra.items():
            if isinstance(v, (str, int, float, bool)):
                f.write(f"{k}: {v}\n")
            elif isinstance(v, (list, tuple)):
                f.write(f"{k}: {' '.join(str(x) for x in v)}\n")


def _should_skip_optimizer_for_rail_mismatch(sem_cfg: dict, oracle_rail: bool, rail_debug: dict, align_debug: dict) -> tuple[bool, str]:
    if not oracle_rail or not bool(sem_cfg.get("skip_optimizer_on_rail_refinement_mismatch", False)):
        return False, ""
    if rail_debug and not bool(rail_debug.get("rail_refinement_valid", True)):
        return True, "rail_bev_alignment_mismatch"
    if align_debug and not bool(align_debug.get("rail_bev_alignment_valid", True)):
        return True, "rail_bev_alignment_mismatch"
    return False, ""


def _apply_final_rail_hard_gate(breakdown: dict, sem_cfg: dict, oracle_rail: bool) -> dict:
    out = dict(breakdown or {})
    out.setdefault("final_pose_valid", 1.0)
    out.setdefault("rail_gate_failed", 0.0)
    out.setdefault("invalid_reason", "")
    if not oracle_rail or not bool(sem_cfg.get("oracle_rail_hard_gate", False)):
        return out
    min_count = float(sem_cfg.get("min_final_rail_visible_count", sem_cfg.get("min_rail_visible_count", 50)))
    min_ratio = float(sem_cfg.get("min_final_rail_visible_ratio", sem_cfg.get("min_rail_visible_ratio", 0.08)))
    visible_count = float(out.get("rail_visible_count", 0.0) or 0.0)
    visible_ratio = float(out.get("rail_visible_ratio", 0.0) or 0.0)
    failed = visible_count < min_count or visible_ratio < min_ratio
    if failed:
        out["rail_gate_failed"] = 1.0
        object_visible = float(out.get("object_visible_count", 0.0) or 0.0)
        edge_visible = float(out.get("edge_visible_count", 0.0) or 0.0)
        semantic_used = float(out.get("semantic_term_used", 0.0) or 0.0) > 0.0
        semantic_hist = float(out.get("semantic_hist_similarity", 0.0) or 0.0)
        semantic_js = float(out.get("semantic_js_divergence", 1.0) or 1.0)
        min_object = float(sem_cfg.get("min_final_object_visible_count", 1))
        min_edge = float(sem_cfg.get("min_final_edge_visible_count", sem_cfg.get("min_edge_visible_count", 50)))
        min_sem_hist = float(sem_cfg.get("min_final_semantic_hist_similarity", 0.05))
        max_sem_js = float(sem_cfg.get("max_final_semantic_js_divergence", 0.95))
        semantic_ok = semantic_used and (semantic_hist >= min_sem_hist or semantic_js <= max_sem_js)
        allow_alt = bool(sem_cfg.get("allow_object_or_edge_to_pass_rail_gate", True)) and (
            object_visible >= min_object or edge_visible >= min_edge or semantic_ok
        )
        out["object_or_edge_rail_gate_bypass"] = 1.0 if allow_alt else 0.0
        out["semantic_rail_gate_bypass_ok"] = 1.0 if semantic_ok else 0.0
        if bool(sem_cfg.get("reject_pose_on_rail_gate_fail", True)) and not allow_alt:
            out["final_pose_valid"] = 0.0
            out["invalid_reason"] = (
                f"rail_visible_gate_failed(count={visible_count:.0f}<{min_count:.0f} "
                f"or ratio={visible_ratio:.6f}<{min_ratio:.6f})"
            )
        else:
            out["invalid_reason"] = ""
    return out


def run(context: RuntimeContext) -> None:
    print("\n" + "=" * 40)
    print("[阶段3] 两阶段标定优化")
    print("=" * 40)

    image_root = context.paths.get("image_features") if context.paths else ""
    sam_root = context.paths.get("sam") if context.paths else ""
    if not image_root:
        image_root = context.config["data"]["image_features_output_dir"]
    if not sam_root:
        sam_root = context.config["data"]["sam_output_dir"]
    lidar_dir = context.config["data"]["lidar_output_dir"]
    calib_dir = context.config["data"]["calib_output_dir"]
    calib_file = context.config["data"]["calib_file"]
    temporal_cfg = context.config.get("calibration", {}).get("temporal_validation", {})
    history_enabled = bool(temporal_cfg.get("enabled", False))
    history_cfg = temporal_cfg.get("history_file", "")
    history_file = ""
    if history_enabled and history_cfg:
        history_file = str(history_cfg)
        if not os.path.isabs(history_file):
            history_file = os.path.join(calib_dir, history_file)
        os.makedirs(os.path.dirname(history_file) or ".", exist_ok=True)

    init_r = context.config["calibration"]["initial_extrinsic"]["rotation"]
    init_t = context.config["calibration"]["initial_extrinsic"]["translation"]

    ds = get_adapter(context.config)
    ext = ds.load_initial_extrinsic()
    if ext:
        init_r, init_t = ext
        cam_folder = str(context.config.get("data", {}).get("image_sensor", "rgb_center") or "rgb_center")
        print(f"[Info] 使用OSDaR23 calibration.txt 读取初始外参, camera={cam_folder}")

    bev_cfg = context.config.get("bev") or {}
    bev_by_frame = getattr(context, "bev_pose_by_frame", None) or {}
    sem_cfg = context.config.get("semantic_calib") or {}
    label_cfg = context.config.get("label_assist") or {}
    label_assist_for_calib = bool(label_cfg.get("enabled", False)) and bool(label_cfg.get("use_for_calib_residual", True))
    sem_enabled = bool(sem_cfg.get("enabled", False))
    allow_legacy_fallback = bool(sem_cfg.get("allow_legacy_fallback", False))
    img_cfg = context.config.get("image_features") or {}
    sem_classes = list(img_cfg.get("semantic_classes", []) or [])
    cw = sem_cfg.get("class_weights") or {}
    class_weights = [float(cw.get(name, 1.0)) for name in sem_classes]
    pyr = sem_cfg.get("pyramid_scales") or [1.0, 0.5, 0.25]
    pyramid_scales_s = ",".join(str(float(x)) for x in pyr)
    class_weights_s = ",".join(str(float(x)) for x in class_weights) if class_weights else ""

    adapter = get_optimizer_constraint_adapter(context.config)
    optimizer_env, has_ab_overrides = adapter.build_env(context.config, os.environ.copy())
    for k, v in ds.get_optimizer_env().items():
        optimizer_env[k] = v

    print(f"[Info] 优化约束适配器: {adapter.name}")
    if has_ab_overrides:
        print("[Info] 已加载 calibration.ab_experiment 参数并传递给 optimizer")
    if history_file:
        print(f"[Info] 时序平滑已启用，history_file={history_file}")
    else:
        print("[Info] 时序平滑已禁用，不加载历史标定结果")

    sys.path.insert(0, os.path.join(_REPO_ROOT, "tools"))
    from score_breakdown_io import write_unified_debug_json  # noqa: E402
    from export_semantic_probs_bin import npy_to_edgecalib_bin  # noqa: E402
    from lidar_bev_rail_points import export_lidar_bev_rail_points  # noqa: E402

    for frame_id in context.frame_ids:
        feature_base = os.path.join(lidar_dir, f"{frame_id:010d}")
        frame_dir = os.path.join(image_root, f"{frame_id:010d}")
        sam_base = os.path.join(sam_root, f"{frame_id:010d}")
        output_file = os.path.join(calib_dir, f"{frame_id:010d}_calib_result.txt")

        if not os.path.exists(f"{feature_base}_points.txt"):
            print(f"[Warning] 特征文件不存在，跳过帧 {frame_id:010d}")
            continue

        print(f"\n优化帧 {frame_id:010d}...")
        print(f"  logical_frame_id={frame_id:010d}")
        print(f"  feature_base={feature_base}")
        print(f"  image_features_dir={frame_dir}")
        print(f"  optimizer_base={sam_base}")
        r_use, t_use = list(init_r), list(init_t)
        if bool(bev_cfg.get("enabled", False)) and frame_id in bev_by_frame:
            pose_bev = bev_by_frame[frame_id]
            r_use = pose_bev["rvec"]
            t_use = pose_bev["tvec"]
            print("[Info] 本帧使用 BEV 粗初始化位姿作为 optimizer 初值")

        # Phase B6: pass semantic inputs (probabilities + semantic points + BEV init pose) to optimizer CLI.
        sem_npy = os.path.join(frame_dir, "semantic_probs.npy")
        sem_bin = os.path.join(frame_dir, "semantic_probs.edgecalib.bin")
        sem_pts = f"{feature_base}_semantic_points.txt"
        bev_pose = ""
        # Only pass BEV pose to optimizer when this frame's BEV delta was accepted by the Python BEV stage.
        # This avoids silently applying pose_after_bev.txt even when rail_score was below threshold.
        if bool(bev_cfg.get("enabled", False)) and frame_id in bev_by_frame and context.paths and context.paths.get("bev_init"):
            cand = os.path.join(context.paths["bev_init"], f"{frame_id:010d}", "pose_after_bev.txt")
            if os.path.isfile(cand):
                bev_pose = cand

        use_sem = sem_enabled and os.path.isfile(sem_npy) and os.path.isfile(sem_pts)
        if use_sem:
            ok = npy_to_edgecalib_bin(sem_npy, sem_bin)
            if not ok:
                if not allow_legacy_fallback:
                    raise RuntimeError(
                        "semantic_calib.enabled=true 且 semantic_probs 导出失败；"
                        "当前配置禁止回退 legacy。"
                    )
                print("[Warning] semantic_probs.npy 导出 EDGESEM1 失败，回退 legacy optimizer 调用")
                use_sem = False
        elif sem_enabled and not allow_legacy_fallback:
            missing = []
            if not os.path.isfile(sem_npy):
                missing.append(sem_npy)
            if not os.path.isfile(sem_pts):
                missing.append(sem_pts)
            raise FileNotFoundError(
                "semantic_calib.enabled=true 但缺少语义输入，且禁止 legacy 回退。缺失文件:\n"
                + "\n".join(missing)
            )

        if use_sem:
            bev_rail_points = f"{feature_base}_rail_bev_points.txt"
            bev_rail_count = 0
            if bool(sem_cfg.get("use_lidar_bev_rail_samples", False)):
                bev_npz = f"{feature_base}_bev_maps.npz"
                if os.path.isfile(bev_npz):
                    try:
                        rail_debug_path = f"{feature_base}_rail_bev_debug.json"
                        rail_png_path = f"{feature_base}_rail_bev_refined.png"
                        oracle_bev = os.path.join(frame_dir, "pseudo_bev.npz")
                        bev_rail_count = export_lidar_bev_rail_points(
                            bev_npz,
                            bev_rail_points,
                            min_prob=float(sem_cfg.get("lidar_bev_rail_sample_min_prob", 0.15)),
                            stride_cells=int(sem_cfg.get("lidar_bev_rail_sample_stride_cells", 2)),
                            max_points=int(sem_cfg.get("lidar_bev_rail_sample_max_points", 8000)),
                            reference_z=float((context.config.get("dataset") or {}).get("reference_z", 0.0)),
                            oracle_npz_path=oracle_bev if os.path.isfile(oracle_bev) else None,
                            oracle_overlap_dilate_cells=int(sem_cfg.get("lidar_bev_oracle_overlap_dilate_cells", 3)),
                            min_component_cells=int(sem_cfg.get("lidar_bev_rail_min_component_cells", 20)),
                            debug_path=rail_debug_path,
                            refined_png_path=rail_png_path,
                            refined_bin_path=f"{feature_base}_rail_bev_refined.bin",
                            crop_to_image_valid=bool(sem_cfg.get("lidar_bev_crop_to_image_valid", True)),
                            crop_to_image_rail_bbox=bool(sem_cfg.get("lidar_bev_crop_to_image_rail_bbox", True)),
                            image_rail_bbox_padding_m=float(sem_cfg.get("lidar_bev_image_rail_bbox_padding_m", 8.0)),
                        )
                        print(f"[Info] Exported LiDAR BEV rail samples: {bev_rail_count} -> {bev_rail_points}")
                    except Exception as exc:
                        print(f"[Warning] Export LiDAR BEV rail samples failed: {type(exc).__name__}: {exc}")

            # Phase C6: adapt rail weight based on LiDAR rail meta (switch detection / low confidence).
            effective_rail_weight = float(sem_cfg.get("rail_weight", 1.2))
            min_img_q = float(sem_cfg.get("min_image_rail_quality", 0.45))
            low_img_weight = float(sem_cfg.get("low_image_rail_quality_weight", sem_cfg.get("branch_rail_weight", 0.0)))
            rail_quality_path = os.path.join(frame_dir, "rail_quality.json")
            rail_quality = _load_json_dict(rail_quality_path)
            if not rail_quality:
                rail_quality = _load_json_dict(f"{sam_base}_rail_quality.json")
            if rail_quality:
                rq_enabled = bool(rail_quality.get("enabled", False))
                rq_score = float(rail_quality.get("quality_score", 0.0) or 0.0)
                oracle_rail = bool(rail_quality.get("label_track_prior_used", False)) and rq_enabled and rq_score >= min_img_q
                if (not rq_enabled) or rq_score < min_img_q:
                    effective_rail_weight = low_img_weight
                    print(
                        f"[Info] image rail quality low "
                        f"(enabled={rq_enabled}, score={rq_score:.3f} < {min_img_q}); "
                        f"rail_weight={effective_rail_weight}"
                    )
                elif oracle_rail:
                    min_samples = int(sem_cfg.get("min_lidar_bev_rail_samples", 200))
                    if bev_rail_count >= min_samples:
                        effective_rail_weight = float(sem_cfg.get("oracle_rail_weight", effective_rail_weight))
                    else:
                        effective_rail_weight = float(sem_cfg.get("oracle_rail_weight_fallback", 0.2))
                    print(
                        f"[Info] JSON oracle rail enabled "
                        f"(score={rq_score:.3f}, bev_samples={bev_rail_count}); "
                        f"rail_weight={effective_rail_weight}"
                    )
                else:
                    oracle_rail = False
            else:
                oracle_rail = False
            align_debug = _load_json_dict(f"{feature_base}_rail_bev_alignment_debug.json")
            rail_debug_current = _load_json_dict(f"{feature_base}_rail_bev_debug.json")
            skip_optimizer, skip_reason = _should_skip_optimizer_for_rail_mismatch(
                sem_cfg, bool(oracle_rail), rail_debug_current, align_debug
            )
            if skip_optimizer:
                extra = {
                    "rail_refinement_valid": rail_debug_current.get("rail_refinement_valid", 1),
                    "rail_bev_alignment_valid": align_debug.get("rail_bev_alignment_valid", 0),
                    "rail_bev_alignment_best_transform": align_debug.get("best_transform", ""),
                    "rail_bev_alignment_best_shift_cells": align_debug.get("best_shift_cells", [0, 0]),
                    "rail_bev_alignment_best_score": align_debug.get("best_shift_score", 0.0),
                }
                _write_invalid_calib_result(output_file, list(r_use), list(t_use), skip_reason, extra)
                br = _parse_calib_breakdown(output_file)
                pose_out = _parse_calib_pose(output_file)
                write_unified_debug_json(
                    os.path.join(calib_dir, f"{frame_id:010d}_debug_score_breakdown.json"),
                    stage="calib",
                    frame_id=f"{frame_id:010d}",
                    input_pose={"rvec": list(r_use), "tvec": list(t_use)},
                    output_pose=pose_out,
                    breakdown=br,
                    elapsed_sec=0.0,
                    meta={"optimizer_skipped": True, "skip_reason": skip_reason},
                )
                context.current_pose_semantic = pose_out
                print(f"[Info] Skip optimizer for frame {frame_id:010d}: {skip_reason}")
                continue

            rail_meta_path = f"{feature_base}_rail_meta.json"
            if os.path.isfile(rail_meta_path):
                try:
                    with open(rail_meta_path, "r", encoding="utf-8") as f:
                        rm = json.load(f)
                    branch = bool(rm.get("branch_detected", False))
                    rc = float(rm.get("rail_confidence", 1.0))
                    min_rc = float(sem_cfg.get("min_rail_confidence_for_weight", 0.5))
                    if (not oracle_rail) and (branch or rc < min_rc):
                        effective_rail_weight = float(sem_cfg.get("branch_rail_weight", 0.0))
                        print(
                            f"[Info] 道岔/低置信度轨道（branch={branch}, rail_confidence={rc:.3f} < {min_rc}），"
                            f"rail_weight 降为 {effective_rail_weight}"
                        )
                except (OSError, ValueError, json.JSONDecodeError):
                    print(f"[Warning] 读取 rail_meta 失败，忽略: {rail_meta_path}")

            optimizer_bin = os.path.join(_REPO_ROOT, "build", "optimizer")
            cmd = [
                optimizer_bin,
                "--lidar_feature_base",
                feature_base,
                "--sam_feature_base",
                sam_base,
                "--calib_file",
                calib_file if os.path.exists(calib_file) else "",
                "--init_rx",
                str(r_use[0]),
                "--init_ry",
                str(r_use[1]),
                "--init_rz",
                str(r_use[2]),
                "--init_tx",
                str(t_use[0]),
                "--init_ty",
                str(t_use[1]),
                "--init_tz",
                str(t_use[2]),
                "--output_file",
                output_file,
                "--semantic_probs",
                sem_bin,
                "--lidar_semantic_points",
                sem_pts,
                "--semantic_js_weight",
                str(float(sem_cfg.get("semantic_js_weight", 3.0))),
                "--histogram_weight",
                str(float(sem_cfg.get("histogram_weight", 0.5))),
                "--edge_weight",
                str(float(sem_cfg.get("edge_weight", 1.0))),
                "--rail_weight",
                str(effective_rail_weight),
                "--lidar_semantic_max_points",
                str(int(sem_cfg.get("lidar_semantic_max_points", 12000))),
                "--stratified_semantic_sampling",
                "1" if bool(sem_cfg.get("stratified_semantic_sampling", False)) else "0",
                "--rail_early_reject_enabled",
                "1" if bool(sem_cfg.get("rail_early_reject_enabled", False)) else "0",
                "--rail_early_reject_visible_ratio",
                str(float(sem_cfg.get("rail_early_reject_visible_ratio", 0.02))),
                "--rail_early_reject_visible_count",
                str(int(sem_cfg.get("rail_early_reject_visible_count", 10))),
                "--optimizer_num_threads",
                str(int(sem_cfg.get("optimizer_num_threads", 0))),
                "--rail_low_visible_policy",
                str(sem_cfg.get("rail_low_visible_policy", "penalty" if oracle_rail else "zero") if oracle_rail else "zero"),
                "--edge_low_visible_policy",
                str(sem_cfg.get("edge_low_visible_policy", "zero")),
                "--min_edge_visible_count",
                str(int(sem_cfg.get("min_edge_visible_count", 50))),
                "--edge_low_visible_penalty",
                str(float(sem_cfg.get("edge_low_visible_penalty", 0.15))),
                "--min_rail_visible_count",
                str(int(sem_cfg.get("min_rail_visible_count", 50))),
                "--min_rail_visible_ratio",
                str(float(sem_cfg.get("min_rail_visible_ratio", 0.08))),
                "--rail_low_visible_penalty",
                str(float(sem_cfg.get("rail_low_visible_penalty", 0.50))),
                "--rail_visibility_residual_weight",
                str(float(sem_cfg.get("rail_visibility_residual_weight", 0.15 if oracle_rail else 0.0))),
                "--rail_oob_residual_weight",
                str(float(sem_cfg.get("rail_oob_residual_weight", 0.20 if oracle_rail else 0.0))),
                "--pyramid_scales",
                pyramid_scales_s,
            ]
            if _optimizer_binary_supports_arg(optimizer_bin, "--vehicle_object_weight"):
                cmd.extend([
                    "--vehicle_object_weight",
                    str(float(sem_cfg.get("vehicle_object_weight", 0.8))),
                    "--person_object_weight",
                    str(float(sem_cfg.get("person_object_weight", 0.5))),
                ])
            else:
                print(
                    "[Warning] build/optimizer does not support object-weight arguments; "
                    "please rebuild optimizer to enable person/vehicle object scoring."
                )
            if _optimizer_binary_supports_arg(optimizer_bin, "--label_assist_enabled"):
                cmd.extend([
                    "--label_assist_enabled",
                    "1" if label_assist_for_calib else "0",
                    "--label_object_points",
                    f"{feature_base}_label_object_points.txt",
                    "--label_track_weight",
                    str(float(label_cfg.get("track_weight", 1.5))),
                    "--label_static_weight",
                    str(float(label_cfg.get("static_object_weight", 1.0))),
                    "--label_vehicle_weight",
                    str(float(label_cfg.get("vehicle_weight", 0.4))),
                    "--label_person_weight",
                    str(float(label_cfg.get("person_weight", 0.2))),
                ])
            elif label_assist_for_calib:
                print(
                    "[Warning] build/optimizer does not support label-assisted arguments; "
                    "please rebuild optimizer to enable label teacher residuals."
                )
            if class_weights_s:
                cmd.extend(["--class_weights", class_weights_s])
            mode = str(sem_cfg.get("optimize_mode", "full_calib") or "full_calib")
            cmd.extend(["--mode", mode])
            if bev_pose:
                cmd.extend(["--init_pose_from_bev", bev_pose])
        else:
            if sem_enabled:
                print("[Warning] semantic_calib.enabled=true 但缺少 semantic_probs.npy 或 semantic_points.txt，回退 legacy")
            cmd = [
                os.path.join(_REPO_ROOT, "build", "optimizer"),
                feature_base,
                sam_base,
                calib_file if os.path.exists(calib_file) else "",
                str(r_use[0]),
                str(r_use[1]),
                str(r_use[2]),
                str(t_use[0]),
                str(t_use[1]),
                str(t_use[2]),
                output_file,
            ]
            if history_file:
                cmd.append(history_file)

        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, env=optimizer_env, cwd=_REPO_ROOT)
        elapsed = time.perf_counter() - t0

        pose_out = _parse_calib_pose(output_file)
        if pose_out:
            context.current_pose_semantic = pose_out
        br = _parse_calib_breakdown(output_file)
        rail_debug = _load_json_dict(f"{feature_base}_rail_bev_debug.json")
        edge_debug = _load_json_dict(f"{feature_base}_edge_meta.json")
        label_debug = _load_json_dict(f"{feature_base}_debug_label_assist.json")
        if rail_debug:
            if "lidar_rail_refined_nonzero_ratio" in rail_debug:
                br["refined_lidar_rail_nonzero_ratio"] = rail_debug["lidar_rail_refined_nonzero_ratio"]
            elif "lidar_rail_raw_nonzero_ratio" in rail_debug:
                br["refined_lidar_rail_nonzero_ratio"] = rail_debug.get("lidar_rail_refined_nonzero_ratio", rail_debug.get("lidar_rail_raw_nonzero_ratio", 0.0))
        if edge_debug:
            for k in ("edge_raw_count", "edge_kept_count", "edge_range_gt_50_count", "edge_near_track_count"):
                if k in edge_debug:
                    br[k] = edge_debug[k]
        if label_debug:
            for k in (
                "label_assist_enabled",
                "label_feature_used",
                "unsupervised_feature_used",
                "image_track_vs_label_score",
                "sam_vehicle_vs_label_iou",
                "sam_person_vs_label_iou",
            ):
                if k in label_debug:
                    br[k] = label_debug[k]
        br = _apply_final_rail_hard_gate(br, sem_cfg, bool(locals().get("oracle_rail", False)))
        if output_file and os.path.isfile(output_file):
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"final_pose_valid: {br.get('final_pose_valid', 1.0)}\n")
                f.write(f"rail_gate_failed: {br.get('rail_gate_failed', 0.0)}\n")
                if br.get("invalid_reason"):
                    f.write(f"invalid_reason: {br.get('invalid_reason')}\n")
                if "refined_lidar_rail_nonzero_ratio" in br:
                    f.write(f"refined_lidar_rail_nonzero_ratio: {br['refined_lidar_rail_nonzero_ratio']}\n")
        write_unified_debug_json(
            os.path.join(calib_dir, f"{frame_id:010d}_debug_score_breakdown.json"),
            stage="calib",
            frame_id=f"{frame_id:010d}",
            input_pose={"rvec": list(r_use), "tvec": list(t_use)},
            output_pose=pose_out,
            breakdown=br,
            elapsed_sec=elapsed,
            meta={"optimizer_argv_len": len(cmd)},
        )

    print(f"\n[完成] 标定结果已保存到: {calib_dir}")
