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
    double line_weight = 0.6;
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

    std::vector<PointFeature> edge_points_;
    std::vector<PointFeature> points_;
    std::vector<Line2D> lines2d_;
    std::vector<Line3D> lines3d_;

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
