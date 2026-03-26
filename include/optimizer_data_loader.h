#pragma once

#include "include/common.h"

#include <deque>
#include <string>
#include <vector>

#include <eigen3/Eigen/Core>

struct CalibHistory {
    std::deque<Eigen::Vector3d> rotation_history;
    std::deque<Eigen::Vector3d> translation_history;
    std::deque<double> score_history;
    int max_history = 3;

    void push(const Eigen::Vector3d& r, const Eigen::Vector3d& t, double score);
    bool is_smooth(const Eigen::Vector3d& r_new, const Eigen::Vector3d& t_new, double threshold = 0.05) const;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> get_smoothed() const;
};

bool LoadCalibHistory(const std::string& history_file, CalibHistory& history);
bool SaveCalibHistory(const std::string& history_file, const CalibHistory& history);
std::vector<PointFeature> LoadEdgePointsCustom(const std::string& filename);

bool LoadCalib(const std::string& calib_file,
               Eigen::Matrix3d& K,
               Eigen::Matrix3d& R_rect,
               Eigen::Matrix<double, 3, 4>& P_rect,
               bool* used_default = nullptr);
