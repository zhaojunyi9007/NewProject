#pragma once

#include "include/common.h"
#include "include/optimizer_cost_function.h"
#include "include/optimizer_semantic_scoring.h"

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

// Phase C5: unify breakdown structure in common.h
using TotalScoreBreakdown = ScoreBreakdown;

double ComputeTotalCalibrationScoreSemanticDominant(const std::vector<PointFeature>& edge_points,
                                                    const cv::Mat& edge_dist,
                                                    const cv::Mat& edge_weight,
                                                    const std::vector<Line3D>& lines3d,
                                                    const std::vector<PointFeature>& rail_points,
                                                    const cv::Mat& rail_dist,
                                                    const cv::Mat& rail_weight,
                                                    const std::vector<SemanticPointRecord>& lidar_semantic_points,
                                                    const SemanticProbMaps& image_semantic_probs,
                                                    const Eigen::Matrix3d& R_rect,
                                                    const Eigen::Matrix<double, 3, 4>& P_rect,
                                                    int W,
                                                    int H,
                                                    const Eigen::Matrix3d& R,
                                                    const Eigen::Vector3d& t,
                                                    double w_semantic_js,
                                                    double w_semantic_hist,
                                                    double w_edge,
                                                    double w_rail,
                                                    const SemanticScoringConfig& sem_cfg,
                                                    TotalScoreBreakdown* breakdown);
