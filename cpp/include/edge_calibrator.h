#pragma once

#include <string>
#include <vector>

#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

#include "include/common.h"
#include "include/optimizer_data_loader.h"
#include "include/optimizer_scoring.h"

struct EdgeCalibratorConfig {
    std::string lidar_base;
    std::string sam_base;
    std::string calib_file;
    std::string output_file;
    std::string history_file;
    double init_r[3] = {0.0, 0.0, 0.0};
    double init_t[3] = {0.0, 0.0, 0.0};

    // Phase B2: semantic-probability calibration inputs / weights (plumbed from CLI).
    std::string semantic_probs_path;          // EDGESEM1 binary (exported from semantic_probs.npy)
    std::string lidar_semantic_points_path;   // <lidar_base>_semantic_points.txt
    std::string init_pose_from_bev_path;      // pose_after_bev.txt (optional)
    double semantic_js_weight = 3.0;
    double histogram_weight = 0.5;
    double edge_weight = 1.0;
    // Phase D7 (sam_2d): rail term weight (new).
    double rail_weight = 1.2;
    double vehicle_object_weight = 0.8;
    double person_object_weight = 0.5;
    bool label_assist_enabled = false;
    bool strong_label_enabled = false;
    std::string label_object_points_path;
    std::string label_strong_features_path;
    double label_track_weight = 1.5;
    double label_static_weight = 1.0;
    double strong_track_weight = 0.25;
    double strong_pole_weight = 3.0;
    double strong_switch_weight = 2.0;
    double strong_buffer_stop_weight = 4.0;
    bool strong_stage_a_use_switch = false;
    double strong_label_min_translation_prior_weight = 50.0;
    double strong_label_rotation_prior_weight = 10.0;
    double strong_label_max_pose_jump_m = 1.0;
    double strong_label_max_yaw_jump_deg = 3.0;
    double strong_label_static_score_tolerance = 0.0;
    double label_vehicle_weight = 0.4;
    double label_person_weight = 0.2;
    int lidar_semantic_max_points = 0;        // 0 means keep all points.
    bool stratified_semantic_sampling = false;
    int semantic_sample_budget_rail = 800;
    int semantic_sample_budget_ballast = 800;
    int semantic_sample_budget_vertical = 600;
    int semantic_sample_budget_platform_building = 500;
    int semantic_sample_budget_other = 300;
    bool rail_early_reject_enabled = false;
    int rail_early_reject_visible_count = 10;
    double rail_early_reject_visible_ratio = 0.02;
    int optimizer_num_threads = 0;
    std::string rail_low_visible_policy = "zero";
    int min_rail_visible_count = 50;
    double min_rail_visible_ratio = 0.08;
    double rail_low_visible_penalty = 0.50;
    double rail_visibility_residual_weight = 0.0;
    double rail_oob_residual_weight = 0.0;
    std::string edge_low_visible_policy = "zero";
    int min_edge_visible_count = 50;
    double edge_low_visible_penalty = 0.15;
    std::vector<double> class_weights;        // same order as image semantic classes
    std::vector<double> pyramid_scales;       // e.g. 1.0,0.5,0.25

    // Phase B2/B? refine-only scaffolding (parsed but not yet used in current optimizer flow).
    std::string mode = "full_calib";          // full_calib | refine_only
    double max_delta_deg = 0.0;
    double max_delta_m = 0.0;
};

class EdgeCalibrator {
public:
    explicit EdgeCalibrator(const EdgeCalibratorConfig& config);

    bool LoadData();
    void PerformCoarseSearch();
    void PerformFineOptimization();
    void ApplyTemporalSmoothing();
    bool SaveResult() const;

private:
    // Phase E11 (sam_2d): sample rail points from 3D rail lines.
    void BuildRailSamplePoints();
    void DownsampleSemanticPoints();

    // Phase B5: new high-level flow (kept private to preserve public API compatibility).
    void ApplyPoseFromBEVIfProvided();
    void PerformSemanticCoarseOptimizationIfEnabled();
    void PerformSemanticFineOptimizationIfEnabled();
    void PerformGeometricRegularizedRefinement();  // existing Ceres-based fine stage

    EdgeCalibratorConfig config_;

    Eigen::Matrix3d K_;
    Eigen::Matrix3d R_rect_;
    Eigen::Matrix<double, 3, 4> P_rect_;

    double r_curr_[3];
    double t_curr_[3];

    cv::Mat edge_dist_;
    cv::Mat edge_weight_;
    cv::Mat semantic_map_;
    // Phase D7 (sam_2d): rail maps (loaded from *_rail_*.png in later phases).
    cv::Mat rail_dist_;
    cv::Mat rail_weight_;
    cv::Mat rail_region_;
    cv::Mat rail_centerline_;
    cv::Mat person_dist_;
    cv::Mat person_weight_;
    cv::Mat vehicle_dist_;
    cv::Mat vehicle_weight_;
    cv::Mat label_track_dist_;
    cv::Mat label_track_weight_;
    cv::Mat label_static_dist_;
    cv::Mat label_static_weight_;
    cv::Mat label_vehicle_dist_;
    cv::Mat label_vehicle_weight_;
    cv::Mat label_person_dist_;
    cv::Mat label_person_weight_;

    std::vector<PointFeature> edge_points_;
    std::vector<PointFeature> points_;
    std::vector<PointFeature> rail_sample_points_;
    std::vector<PointFeature> object_points_;
    std::vector<PointFeature> label_teacher_points_;
    std::vector<StrongLabelFeature> strong_label_features_;
    std::vector<Line3D> lines3d_;
    std::string rail_sample_source_ = "none";
    int label_residual_count_ = 0;
    int label_track_residual_count_ = 0;
    int label_object_residual_count_ = 0;
    int strong_track_residual_count_ = 0;
    int strong_pole_residual_count_ = 0;
    int strong_switch_residual_count_ = 0;
    int strong_buffer_stop_residual_count_ = 0;
    int strong_label_optimizer_residual_count_ = 0;
    bool strong_residuals_added_to_optimizer_ = false;
    double strong_label_score_before_optimization_ = 0.0;
    double strong_label_score_after_optimization_ = 0.0;
    double strong_static_score_before_optimization_ = 0.0;
    double strong_static_score_after_optimization_ = 0.0;
    bool strong_label_pose_reverted_ = false;
    bool stage_a_attempted_ = false;
    double stage_a_static_score_before_ = 0.0;
    double stage_a_static_score_after_ = 0.0;
    double stage_a_pose_jump_m_ = 0.0;
    double stage_a_yaw_jump_deg_ = 0.0;
    bool stage_a_rejected_ = false;
    std::string stage_a_reject_reason_ = "none";
    std::string strong_label_revert_reason_ = "none";

    CalibHistory history_;

    // Phase B5: semantic-probability inputs and breakdown caching.
    bool semantic_inputs_ready_ = false;
    SemanticProbMaps semantic_probs_;
    std::vector<SemanticPointRecord> semantic_points_;
    SemanticScoringConfig sem_cfg_;
    TotalScoreBreakdown last_score_breakdown_;

    int W_ = 0;
    int H_ = 0;

    double best_score_ = -1e8;
    Eigen::Vector3d r_result_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d t_result_ = Eigen::Vector3d::Zero();
};
