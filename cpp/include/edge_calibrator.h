#pragma once

#include <string>
#include <vector>

#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

#include "include/common.h"
#include "include/optimizer_data_loader.h"

struct EdgeCalibratorConfig {
    std::string lidar_base;
    std::string sam_base;
    std::string calib_file;
    std::string output_file;
    std::string history_file;
    double init_r[3] = {0.0, 0.0, 0.0};
    double init_t[3] = {0.0, 0.0, 0.0};
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

    int W_ = 0;
    int H_ = 0;

    double best_score_ = -1e8;
    Eigen::Vector3d r_result_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d t_result_ = Eigen::Vector3d::Zero();
};
