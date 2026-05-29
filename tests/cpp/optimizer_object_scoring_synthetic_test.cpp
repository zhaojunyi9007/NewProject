#include "include/optimizer_scoring.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

int main() {
    const int W = 80;
    const int H = 60;
    cv::Mat edge_dist(H, W, CV_32FC1, cv::Scalar(0.5f));
    cv::Mat edge_weight(H, W, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat rail_dist(H, W, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat rail_weight(H, W, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat vehicle_dist(H, W, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat vehicle_weight(H, W, CV_32FC1, cv::Scalar(0.0f));
    vehicle_dist.at<float>(30, 40) = 0.0f;
    vehicle_weight.at<float>(30, 40) = 1.0f;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_rect = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 3, 4> P = Eigen::Matrix<double, 3, 4>::Zero();
    P(0, 0) = 1.0;
    P(1, 1) = 1.0;
    P(0, 2) = 40.0;
    P(1, 2) = 30.0;
    P(2, 2) = 1.0;

    std::vector<PointFeature> rail_points;
    PointFeature rail;
    rail.p = Eigen::Vector3d(0.0, 0.0, 1.0);
    rail.label = SEM_RAIL_LIKE;
    rail.weight = 1.0;
    rail_points.push_back(rail);

    std::vector<PointFeature> object_points;
    PointFeature obj;
    obj.p = Eigen::Vector3d(0.0, 0.0, 1.0);
    obj.label = SEM_VEHICLE_LIKE;
    obj.weight = 1.0;
    object_points.push_back(obj);

    std::vector<PointFeature> edge_points;
    std::vector<Line3D> lines;
    std::vector<SemanticPointRecord> semantic_points;
    SemanticProbMaps semantic_probs;
    std::vector<PointFeature> empty_points;
    cv::Mat empty_map;

    SemanticScoringConfig cfg;
    cfg.rail_early_reject_enabled = true;
    cfg.rail_early_reject_visible_count = 50;
    cfg.rail_early_reject_visible_ratio = 0.5;
    cfg.rail_low_visible_policy = "penalty";
    cfg.min_rail_visible_count = 50;
    cfg.min_rail_visible_ratio = 0.5;
    cfg.rail_low_visible_penalty = 0.5;

    TotalScoreBreakdown good;
    const double good_score = ComputeTotalCalibrationScoreSemanticDominant(
        edge_points, edge_dist, edge_weight, lines, rail_points, rail_dist, rail_weight,
        object_points, cv::Mat{}, cv::Mat{}, vehicle_dist, vehicle_weight,
        empty_points, empty_map, empty_map, empty_map, empty_map, empty_map, empty_map, empty_map, empty_map,
        semantic_points, semantic_probs, R_rect, P, W, H, R, Eigen::Vector3d(0, 0, 0),
        1.0, 1.0, 0.1, 0.1, 0.8, 0.5, 0.0, 0.0, 0.0, 0.0, cfg, &good);

    TotalScoreBreakdown bad;
    const double bad_score = ComputeTotalCalibrationScoreSemanticDominant(
        edge_points, edge_dist, edge_weight, lines, rail_points, rail_dist, rail_weight,
        object_points, cv::Mat{}, cv::Mat{}, vehicle_dist, vehicle_weight,
        empty_points, empty_map, empty_map, empty_map, empty_map, empty_map, empty_map, empty_map, empty_map,
        semantic_points, semantic_probs, R_rect, P, W, H, R, Eigen::Vector3d(10, 0, 0),
        1.0, 1.0, 0.1, 0.1, 0.8, 0.5, 0.0, 0.0, 0.0, 0.0, cfg, &bad);

    assert(good.rail_early_reject_applied == 1.0);
    assert(good.object_term_used == 1.0);
    assert(good.object_visible_count == 1.0);
    assert(good.object_score_norm > 0.7);
    assert(good.object_score > 0.5);
    assert(good.semantic_term_used == 0.0);
    assert(good.edge_term_used == 0.0);
    assert(good_score > bad_score);
    std::cout << "optimizer_object_scoring_synthetic_test PASS\\n";
    return 0;
}
