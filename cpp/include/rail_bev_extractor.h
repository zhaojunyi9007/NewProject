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
    float min_length_m = 8.0f;
    float max_width_m = 1.2f;
    float min_linearity = 8.0f;
    float gauge_m = 1.435f;
    float gauge_tolerance_m = 0.45f;
    float max_parallel_angle_deg = 5.0f;
    float min_pair_overlap_m = 5.0f;
    int morph_open_kernel_cells = 1;
    int hough_threshold = 12;
    float hough_min_line_length_m = 8.0f;
    float hough_max_line_gap_m = 1.2f;
    float accepted_line_band_m = 0.35f;
    float blob_max_width_m = 1.8f;
    float blob_max_fill_ratio = 0.45f;
    float blob_min_linearity = 4.0f;
    int blob_max_area_cells = 3500;
};

struct RailBEVDebug {
    int candidate_component_count = 0;
    int rejected_blob_component_count = 0;
    int line_candidate_count = 0;
    int line_component_count = 0;
    int pair_count = 0;
    float best_pair_gauge_m = 0.f;
    float best_pair_angle_deg = 0.f;
    float best_pair_overlap_m = 0.f;
    float refined_nonzero_ratio = 0.f;
    std::vector<float> refined_bbox_m;
    std::vector<float> raw_bbox_m;
    float mean_probability_on_accepted_lines = 0.f;
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
    bool* out_branch_detected = nullptr,
    RailBEVDebug* out_debug = nullptr,
    std::vector<float>* out_refined_rail_probability = nullptr);

#endif
