#pragma once

#include "include/common.h"
#include "include/optimizer_cost_functions.h"

#include <vector>

#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

bool Project(const Eigen::Vector3d& p_lidar,
             const Eigen::Matrix3d& R_rect,
             const Eigen::Matrix<double, 3, 4>& P_rect,
             const Eigen::Matrix3d& R,
             const Eigen::Vector3d& t,
             int& u, int& v, int W, int H);

struct ProjectionDebugStats {
    int total = 0;
    int behind = 0;
    int out_of_bounds = 0;
    int in_bounds = 0;
    int labeled = 0;
    int unlabeled = 0;
};

ProjectionDebugStats CountProjectionStats(const std::vector<PointFeature>& points,
                                          const Eigen::Matrix3d& R_rect,
                                          const Eigen::Matrix<double, 3, 4>& P_rect,
                                          const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& t,
                                          int W, int H);

std::vector<PointFeature> FilterPointsInView(const std::vector<PointFeature>& points,
                                             const Eigen::Matrix3d& R_rect,
                                             const Eigen::Matrix<double, 3, 4>& P_rect,
                                             const Eigen::Matrix3d& R,
                                             const Eigen::Vector3d& t,
                                             int W, int H,
                                             int margin = 20);

std::vector<LabelStats> ComputeLabelStats(const std::vector<PointFeature>& points,
                                          const cv::Mat& semantic_map,
                                          const Eigen::Matrix3d& R_rect,
                                          const Eigen::Matrix<double, 3, 4>& P_rect,
                                          int W, int H,
                                          const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& t);

double MaskIntensityVarianceScore(const std::vector<PointFeature>& points,
                                  const cv::Mat& semantic_map,
                                  const Eigen::Matrix3d& R_rect,
                                  const Eigen::Matrix<double, 3, 4>& P_rect,
                                  int W, int H,
                                  const Eigen::Matrix3d& R,
                                  const Eigen::Vector3d& t);

double EdgeAttractionScore(const std::vector<PointFeature>& points,
                           const cv::Mat& dist_map,
                           const cv::Mat& weight_map,
                           const Eigen::Matrix3d& R_rect,
                           const Eigen::Matrix<double, 3, 4>& P_rect,
                           int W, int H,
                           const Eigen::Matrix3d& R,
                           const Eigen::Vector3d& t);
