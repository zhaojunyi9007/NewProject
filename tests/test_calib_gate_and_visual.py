import os
import json
import tempfile
import yaml
from types import SimpleNamespace
from unittest import mock

from pipeline.context import RuntimeContext
from pipeline.stages import calib_stage, visual_stage


def test_rail_mismatch_does_not_skip_when_strong_label_features_exist(tmp_path):
    strong = tmp_path / "0000000012_label_strong_features.tsv"
    strong.write_text("# header\ntrack\ttrk1\tpoint0\t1.0\t1\t2\t3\t0\t0\t0\tpolyline\t2\t0\t0\t1\t1\n", encoding="utf-8")
    sem_cfg = {"skip_optimizer_on_rail_refinement_mismatch": True}
    label_cfg = {"enabled": True, "strong_features_enabled": True}
    rail_debug = {"rail_refinement_valid": True}
    align_debug = {"rail_bev_alignment_valid": False}

    skip, reason = calib_stage._should_skip_optimizer_for_rail_mismatch(
        sem_cfg, True, rail_debug, align_debug, label_cfg, str(strong)
    )

    assert skip is False
    assert reason == ""
    assert calib_stage._has_rail_mismatch(rail_debug, align_debug) is True
    assert calib_stage._strong_label_can_override_rail_mismatch(label_cfg, str(strong)) is True


def test_rail_mismatch_still_skips_without_label_assist(tmp_path):
    strong = tmp_path / "0000000012_label_strong_features.tsv"
    strong.write_text("track\ttrk1\tpoint0\t1.0\t1\t2\t3\t0\t0\t0\tpolyline\t2\t0\t0\t1\t1\n", encoding="utf-8")
    sem_cfg = {"skip_optimizer_on_rail_refinement_mismatch": True}
    label_cfg = {"enabled": False, "strong_features_enabled": True}

    skip, reason = calib_stage._should_skip_optimizer_for_rail_mismatch(
        sem_cfg, True, {"rail_refinement_valid": True}, {"rail_bev_alignment_valid": False}, label_cfg, str(strong)
    )

    assert skip is True
    assert reason == "rail_bev_alignment_mismatch"



def test_label_assisted_candidate_selection_ignores_bev_by_default():
    init_r = [0.0, 0.0, 0.0]
    init_t = [0.0, 0.0, 0.0]
    bev_pose = {"rvec": [0.0, 0.0, 0.1], "tvec": [5.0, 0.0, 0.0]}

    def fake_score(source, pose, *_args, **_kwargs):
        return {
            "source": source,
            "selection_score": 10.0 if source == "bev_accepted" else 1.0,
            "pose_jump_m": 5.0 if source == "bev_accepted" else 0.0,
            "yaw_jump_deg": 5.0 if source == "bev_accepted" else 0.0,
        }

    with mock.patch.object(calib_stage, "_score_initial_pose_candidate", side_effect=fake_score):
        r, t, source, rejected, scores = calib_stage._select_initial_pose_candidate(
            12,
            init_r,
            init_t,
            "feature",
            "sam",
            object(),
            {"enable_init_candidate_scoring": True, "use_bev_candidate_in_label_assist": False},
            {"enabled": True, "strong_features_enabled": True},
            {12: bev_pose},
            {},
            True,
        )

    assert source == "original_init"
    assert r == init_r
    assert t == init_t
    assert rejected is False
    assert {s["source"] for s in scores} == {"original_init"}


def test_osdar_strong_label_stability_defaults_are_static_anchor_first():
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "osdar23.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    label_cfg = cfg["label_assist"]
    sem_cfg = cfg["semantic_calib"]
    assert label_cfg["track_weight"] < label_cfg["switch_weight"] < label_cfg["catenary_pole_weight"] < label_cfg["buffer_stop_weight"]
    assert label_cfg["max_track_samples_per_object"] <= 120
    assert label_cfg["track_initial_max_dist_px"] == 300.0
    assert label_cfg["track_use_as_primary_gate"] is False
    assert sem_cfg["use_bev_candidate_in_label_assist"] is False
    assert sem_cfg["strong_label_min_translation_prior_weight"] >= 50.0
    assert sem_cfg["strong_label_rotation_prior_weight"] >= 10.0
    assert sem_cfg["strong_label_static_score_tolerance"] == 0.0
    assert label_cfg["strong_label_static_overlay_enabled"] is True
    assert label_cfg["max_switch_samples_per_object"] == 80
    assert label_cfg["stage_a_use_switch"] is False
    assert label_cfg["stage_b_use_track"] is False
    assert label_cfg["stage_b_use_switch"] is False
    assert label_cfg["stage_b_track_min_score"] == 0.25
    assert label_cfg["stage_b_switch_min_score"] == 0.25
    vis_cfg = cfg["visualization"]
    assert vis_cfg["lidar_projection_enabled"] is True
    assert vis_cfg["lidar_projection_point_source"] == "all"
    assert vis_cfg["lidar_projection_color_mode"] == "depth"
    assert vis_cfg["lidar_projection_max_points"] == 120000
    assert vis_cfg["lidar_projection_point_radius"] == 1
    assert vis_cfg["lidar_projection_background"] == "grayscale"


def test_stage_a_reject_reason_prefers_static_score_degradation():
    reason = calib_stage._stage_a_reject_reason(
        static_before=0.738577,
        static_after=0.729426,
        pose_jump_m=0.5,
        yaw_jump_deg=0.5,
        tolerance=0.0,
        max_pose_jump_m=1.0,
        max_yaw_jump_deg=3.0,
    )

    assert reason == "static_score_degraded"


def test_strong_label_debug_includes_stage_a_fields():
    br = {
        "static_score_before": 0.7,
        "static_score_after_stage_a": 0.69,
        "stage_a_attempted": 1.0,
        "stage_a_static_score_before": 0.7,
        "stage_a_static_score_after": 0.69,
        "stage_a_static_score_delta": -0.01,
        "stage_a_pose_jump_m": 0.2,
        "stage_a_yaw_jump_deg": 0.1,
        "stage_a_rejected": 1.0,
        "stage_a_reject_reason": "static_score_degraded",
        "strong_label_pose_reverted": 1.0,
        "strong_label_feature_count": 8.0,
    }

    debug = calib_stage._extract_strong_label_debug(br)

    assert debug["static_score_before"] == 0.7
    assert debug["static_score_after_stage_a"] == 0.69
    assert debug["stage_a_attempted"] == 1.0
    assert debug["stage_a_static_score_delta"] == -0.01
    assert debug["stage_a_rejected"] == 1.0
    assert debug["stage_a_reject_reason"] == "static_score_degraded"
    assert debug["strong_label_pose_reverted"] == 1.0


def test_osdar_openlabel_intrinsics_calib_file_uses_openlabel_k(tmp_path):
    root = tmp_path / "1_calibration_1.1"
    root.mkdir()
    calib_file = root / "calibration.txt"
    calib_file.write_text(
        "\n".join(
            [
                "CAMERA",
                "data_folder: rgb_highres_center",
                "camera_matrix: [7267.0, 0, 2056.0;",
                " 0, 7267.0, 1486.0;",
                " 0, 0, 1]",
                "homogeneous transform:",
                "[1, 0, 0, 0;",
                " 0, 1, 0, 0;",
                " 0, 0, 1, 0;",
                " 0, 0, 0, 1]",
            ]
        ),
        encoding="utf-8",
    )
    label_json = root / "1_calibration_1.1_labels.json"
    label_json.write_text(
        json.dumps(
            {
                "openlabel": {
                    "streams": {
                        "rgb_highres_center": {
                            "stream_properties": {
                                "intrinsics_pinhole": {
                                    "camera_matrix": [
                                        7267.0,
                                        0.0,
                                        2056.0,
                                        0.0,
                                        0.0,
                                        7267.0,
                                        1232.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        1.0,
                                        0.0,
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "data": {
            "dataset_format": "osdar23",
            "image_sensor": "rgb_highres_center",
            "osdar_sequence_root": str(root),
        },
        "label_assist": {"enabled": True, "use_openlabel_extrinsic": True},
    }
    out, debug = calib_stage._prepare_optimizer_calib_file(
        cfg,
        str(calib_file),
        str(tmp_path / "calibration"),
        12,
        "openlabel_coordinate_systems",
    )

    text = open(out, "r", encoding="utf-8").read()

    assert out.endswith("0000000012_openlabel_intrinsics_calib.txt")
    assert "data_folder: rgb_highres_center" in text
    assert "1232" in text
    assert "1486" not in text
    assert debug["openlabel_intrinsics_available"] is True
    assert debug["optimizer_intrinsics_source"] == "openlabel_streams"
    assert debug["intrinsics_cx_delta_px"] == 0.0
    assert debug["intrinsics_cy_delta_px"] == 0.0
    assert debug["calibration_intrinsics_cy_delta_px"] == 254.0


def test_strong_label_debug_includes_intrinsics_diagnostics():
    debug = calib_stage._extract_strong_label_debug(
        {
            "strong_label_feature_count": 8.0,
            "openlabel_intrinsics_available": True,
            "optimizer_intrinsics_source": "openlabel_streams",
            "intrinsics_cx_delta_px": 0.0,
            "intrinsics_cy_delta_px": 0.0,
            "calibration_intrinsics_cy_delta_px": 254.0,
        }
    )

    assert debug["openlabel_intrinsics_available"] is True
    assert debug["optimizer_intrinsics_source"] == "openlabel_streams"
    assert debug["intrinsics_cy_delta_px"] == 0.0
    assert debug["calibration_intrinsics_cy_delta_px"] == 254.0

def test_visual_stage_omits_json_rail_overlay_by_default(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    lidar_dir = tmp_path / "lidar"
    calib_dir = tmp_path / "calib"
    visual_dir = tmp_path / "visual"
    lidar_dir.mkdir()
    calib_dir.mkdir()
    visual_dir.mkdir()
    (lidar_dir / "0000000012_label_strong_features.tsv").write_text(
        "buffer_stop\tbuf1\tcenter0\t1\t1\t2\t3\t0\t0\t0\tbbox\t10\t20\t30\t40\n",
        encoding="utf-8",
    )
    (calib_dir / "0000000012_calib_result.txt").write_text(
        "r: 0 0 0\nt: 0 0 0\nfinal_pose_valid: 1\n", encoding="utf-8"
    )

    class Adapter:
        def resolve_image(self, frame_id):
            return str(image)

    cfg = {
        "data": {
            "lidar_output_dir": str(lidar_dir),
            "calib_output_dir": str(calib_dir),
            "visual_output_dir": str(visual_dir),
            "calib_file": "",
            "dataset_format": "osdar23",
            "image_sensor": "rgb_center",
            "image_features_output_dir": str(tmp_path / "image_features"),
            "refinement_output_dir": str(tmp_path / "refinement"),
        },
        "visualization": {"enable_diag_panels": False},
    }
    ctx = RuntimeContext(config=cfg, frame_ids=[12], paths={})

    with mock.patch.object(visual_stage, "get_adapter", return_value=Adapter()), mock.patch.object(visual_stage.subprocess, "run") as run:
        visual_stage.run(ctx)

    cmd = run.call_args.args[0]
    assert "--overlay-lidar-rail-samples" in cmd
    assert "--overlay-json-rail" not in cmd
    assert "--strong-label-features" in cmd
    assert str(lidar_dir / "0000000012_label_strong_features.tsv") in cmd
    assert "--strong-static-overlay-output" in cmd
    assert str(visual_dir / "0000000012_result_diag_static_strong.png") in cmd


def test_visual_stage_can_enable_json_rail_overlay(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    lidar_dir = tmp_path / "lidar"
    calib_dir = tmp_path / "calib"
    visual_dir = tmp_path / "visual"
    lidar_dir.mkdir()
    calib_dir.mkdir()
    visual_dir.mkdir()
    (calib_dir / "0000000012_calib_result.txt").write_text(
        "r: 0 0 0\nt: 0 0 0\nfinal_pose_valid: 1\n", encoding="utf-8"
    )

    class Adapter:
        def resolve_image(self, frame_id):
            return str(image)

    cfg = {
        "data": {
            "lidar_output_dir": str(lidar_dir),
            "calib_output_dir": str(calib_dir),
            "visual_output_dir": str(visual_dir),
            "calib_file": "",
            "dataset_format": "osdar23",
            "image_sensor": "rgb_center",
            "image_features_output_dir": str(tmp_path / "image_features"),
            "refinement_output_dir": str(tmp_path / "refinement"),
        },
        "visualization": {"enable_diag_panels": False, "overlay_json_rail": True},
    }
    ctx = RuntimeContext(config=cfg, frame_ids=[12], paths={})

    with mock.patch.object(visual_stage, "get_adapter", return_value=Adapter()), mock.patch.object(visual_stage.subprocess, "run") as run:
        visual_stage.run(ctx)

    cmd = run.call_args.args[0]
    assert "--overlay-json-rail" in cmd


def test_visual_stage_prefers_optimizer_calib_file_for_result_and_static_overlay(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    lidar_dir = tmp_path / "lidar"
    calib_dir = tmp_path / "calib"
    visual_dir = tmp_path / "visual"
    lidar_dir.mkdir()
    calib_dir.mkdir()
    visual_dir.mkdir()
    fallback_calib = tmp_path / "calibration.txt"
    fallback_calib.write_text("old K\n", encoding="utf-8")
    optimizer_calib = calib_dir / "0000000012_openlabel_intrinsics_calib.txt"
    optimizer_calib.write_text("openlabel K\n", encoding="utf-8")
    (lidar_dir / "0000000012_label_strong_features.tsv").write_text(
        "buffer_stop\tbuf1\tcenter0\t1\t1\t2\t3\t0\t0\t0\tbbox\t10\t20\t30\t40\n",
        encoding="utf-8",
    )
    (calib_dir / "0000000012_calib_result.txt").write_text(
        "\n".join(
            [
                "r: 0 0 0",
                "t: 0 0 0",
                "final_pose_valid: 1",
                f"optimizer_calib_file: {optimizer_calib}",
                "optimizer_intrinsics_source: openlabel_streams",
            ]
        ),
        encoding="utf-8",
    )

    class Adapter:
        def resolve_image(self, frame_id):
            return str(image)

    cfg = {
        "data": {
            "lidar_output_dir": str(lidar_dir),
            "calib_output_dir": str(calib_dir),
            "visual_output_dir": str(visual_dir),
            "calib_file": str(fallback_calib),
            "dataset_format": "osdar23",
            "image_sensor": "rgb_highres_center",
            "image_features_output_dir": str(tmp_path / "image_features"),
            "refinement_output_dir": str(tmp_path / "refinement"),
        },
        "visualization": {"enable_diag_panels": False},
        "label_assist": {"strong_label_static_overlay_enabled": True},
    }
    ctx = RuntimeContext(config=cfg, frame_ids=[12], paths={})

    with mock.patch.object(visual_stage, "get_adapter", return_value=Adapter()), mock.patch.object(visual_stage.subprocess, "run") as run:
        visual_stage.run(ctx)

    cmd = run.call_args.args[0]
    assert cmd[cmd.index("--calib_file") + 1] == str(optimizer_calib)
    assert "--strong-static-overlay-output" in cmd
    assert str(visual_dir / "0000000012_result_diag_static_strong.png") in cmd
    assert "--lidar-projection-output" in cmd
    assert str(visual_dir / "0000000012_result_lidar_projection.png") in cmd
    assert "--lidar-projection-point-source" in cmd
    assert cmd[cmd.index("--lidar-projection-point-source") + 1] == "all"
    assert "--lidar-projection-color-mode" in cmd
    assert cmd[cmd.index("--lidar-projection-color-mode") + 1] == "depth"


def test_visual_stage_falls_back_to_config_calib_when_optimizer_calib_missing(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    lidar_dir = tmp_path / "lidar"
    calib_dir = tmp_path / "calib"
    visual_dir = tmp_path / "visual"
    lidar_dir.mkdir()
    calib_dir.mkdir()
    visual_dir.mkdir()
    fallback_calib = tmp_path / "calibration.txt"
    fallback_calib.write_text("old K\n", encoding="utf-8")
    (calib_dir / "0000000012_calib_result.txt").write_text(
        "\n".join(
            [
                "r: 0 0 0",
                "t: 0 0 0",
                "final_pose_valid: 1",
                f"optimizer_calib_file: {tmp_path / 'missing_openlabel_intrinsics_calib.txt'}",
                "optimizer_intrinsics_source: openlabel_streams",
            ]
        ),
        encoding="utf-8",
    )

    class Adapter:
        def resolve_image(self, frame_id):
            return str(image)

    cfg = {
        "data": {
            "lidar_output_dir": str(lidar_dir),
            "calib_output_dir": str(calib_dir),
            "visual_output_dir": str(visual_dir),
            "calib_file": str(fallback_calib),
            "dataset_format": "osdar23",
            "image_sensor": "rgb_highres_center",
            "image_features_output_dir": str(tmp_path / "image_features"),
            "refinement_output_dir": str(tmp_path / "refinement"),
        },
        "visualization": {"enable_diag_panels": False},
    }
    ctx = RuntimeContext(config=cfg, frame_ids=[12], paths={})

    with mock.patch.object(visual_stage, "get_adapter", return_value=Adapter()), mock.patch.object(visual_stage.subprocess, "run") as run:
        visual_stage.run(ctx)

    cmd = run.call_args.args[0]
    assert cmd[cmd.index("--calib_file") + 1] == str(fallback_calib)


def test_visual_stage_can_disable_lidar_projection_output(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    lidar_dir = tmp_path / "lidar"
    calib_dir = tmp_path / "calib"
    visual_dir = tmp_path / "visual"
    lidar_dir.mkdir()
    calib_dir.mkdir()
    visual_dir.mkdir()
    (calib_dir / "0000000012_calib_result.txt").write_text(
        "r: 0 0 0\nt: 0 0 0\nfinal_pose_valid: 1\n", encoding="utf-8"
    )

    class Adapter:
        def resolve_image(self, frame_id):
            return str(image)

    cfg = {
        "data": {
            "lidar_output_dir": str(lidar_dir),
            "calib_output_dir": str(calib_dir),
            "visual_output_dir": str(visual_dir),
            "calib_file": "",
            "dataset_format": "osdar23",
            "image_sensor": "rgb_center",
            "image_features_output_dir": str(tmp_path / "image_features"),
            "refinement_output_dir": str(tmp_path / "refinement"),
        },
        "visualization": {"enable_diag_panels": False, "lidar_projection_enabled": False},
    }
    ctx = RuntimeContext(config=cfg, frame_ids=[12], paths={})

    with mock.patch.object(visual_stage, "get_adapter", return_value=Adapter()), mock.patch.object(visual_stage.subprocess, "run") as run:
        visual_stage.run(ctx)

    cmd = run.call_args.args[0]
    assert "--lidar-projection-output" not in cmd
