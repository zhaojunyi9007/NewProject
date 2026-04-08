#ifndef RAIL_BEV_EXTRACTOR_H
#define RAIL_BEV_EXTRACTOR_H

#include "bev_builder.h"
#include "include/common.h"
#include <opencv2/core.hpp>
#include <vector>

struct RailPriorConfig {
    float rail_prob_thresh = 0.15f;
    float min_component_cells = 30.f;
    int smooth_ksize = 5;
};

struct RailBEVResult {
    std::vector<cv::Point2f> centerline_pts;
    std::vector<cv::Point2f> left_rail_pts;
    std::vector<cv::Point2f> right_rail_pts;
    float confidence = 0.f;
};

// 基于 BEV 的轨条结构：多连通域 + 各域内 PCA 拟合，避免单条全局 RANSAC 直线。
RailBEVResult ExtractRailStructureFromBEV(
    const BEVChannels& bev,
    const RailPriorConfig& cfg);

// 将 2D BEV 线段提升到 3D（z = reference_plane_z），写入 Line3D type=0。
std::vector<Line3D> RailBEVToLine3D(
    const BEVChannels& bev,
    const RailBEVResult& rail,
    double reference_plane_z);

// 对每个足够大的轨概率连通域分别 PCA 拟合，支持多段/分叉区域。
std::vector<Line3D> ExtractMultiRailLinesFromBEV(
    const BEVChannels& bev,
    const RailPriorConfig& cfg,
    double reference_plane_z,
    int max_lines = 6,
    float* out_rail_confidence = nullptr,
    bool* out_branch_detected = nullptr);

#endif
