#include "include/bev_matcher.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static cv::Mat RailToMat(const BEVChannels& bev) {
    const int nx = bev.nx;
    const int ny = bev.ny;
    cv::Mat m(ny, nx, CV_32F, const_cast<float*>(bev.rail_probability.data()));
    return m.clone();
}

static cv::Mat ResampleRailToGrid(const BEVChannels& src, const BEVChannels& dst_grid) {
    cv::Mat out = cv::Mat::zeros(dst_grid.ny, dst_grid.nx, CV_32F);
    if (src.nx <= 0 || src.ny <= 0 || src.resolution <= 1e-12 || src.rail_probability.empty()) {
        return out;
    }
    for (int y = 0; y < dst_grid.ny; ++y) {
        const double wy = dst_grid.ymin + (static_cast<double>(y) + 0.5) * dst_grid.resolution;
        const int sy = static_cast<int>(std::floor((wy - src.ymin) / src.resolution));
        if (sy < 0 || sy >= src.ny) continue;
        for (int x = 0; x < dst_grid.nx; ++x) {
            const double wx = dst_grid.xmin + (static_cast<double>(x) + 0.5) * dst_grid.resolution;
            const int sx = static_cast<int>(std::floor((wx - src.xmin) / src.resolution));
            if (sx < 0 || sx >= src.nx) continue;
            const int idx = sy * src.nx + sx;
            if (idx >= 0 && idx < static_cast<int>(src.rail_probability.size())) {
                out.at<float>(y, x) = src.rail_probability[static_cast<size_t>(idx)];
            }
        }
    }
    return out;
}

static cv::Mat BuildDistanceMapMeters(const cv::Mat& image_rail, double resolution, double cap_m) {
    cv::Mat mask;
    cv::threshold(image_rail, mask, 1e-4, 255.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);
    cv::Mat inv;
    cv::bitwise_not(mask, inv);
    cv::Mat dist_px;
    cv::distanceTransform(inv, dist_px, cv::DIST_L2, 3);
    cv::Mat dist_m = dist_px * static_cast<float>(resolution);
    cv::min(dist_m, static_cast<float>(cap_m), dist_m);
    return dist_m;
}

static double WeightedChamferScore(const cv::Mat& lidar_rail, const cv::Mat& warped_distance_m, const BEVOptimizeConfig& cfg) {
    const double sigma = std::max(1e-6, cfg.chamfer_sigma_m);
    double score = 0.0;
    double weight_sum = 0.0;
    for (int y = 0; y < lidar_rail.rows; ++y) {
        for (int x = 0; x < lidar_rail.cols; ++x) {
            const float w = lidar_rail.at<float>(y, x);
            if (w <= 1e-4f) continue;
            const double d = static_cast<double>(warped_distance_m.at<float>(y, x));
            score += static_cast<double>(w) * std::exp(-(d * d) / (sigma * sigma));
            weight_sum += static_cast<double>(w);
        }
    }
    if (weight_sum < cfg.min_lidar_rail_weight_sum) {
        return 0.0;
    }
    return score / weight_sum;
}

bool EstimateBEVDelta(
    const BEVChannels& lidar_bev,
    const BEVChannels& image_bev,
    const BEVOptimizeConfig& cfg,
    PoseDeltaBev* out_delta,
    ScoreBreakdown* debug_score) {
    if (!out_delta) {
        return false;
    }
    if (lidar_bev.nx <= 0 || lidar_bev.ny <= 0 || lidar_bev.rail_probability.empty()) {
        std::cerr << "[BEVMatcher] Invalid lidar BEV\n";
        return false;
    }
    if (image_bev.rail_probability.empty()) {
        std::cerr << "[BEVMatcher] Invalid image BEV (need rail channel)\n";
        return false;
    }

    cv::Mat L = RailToMat(lidar_bev);
    cv::Mat I = ResampleRailToGrid(image_bev, lidar_bev);
    cv::patchNaNs(L, 0);
    cv::patchNaNs(I, 0);
    double lmax = 0.0;
    cv::minMaxLoc(L, nullptr, &lmax);
    if (lmax > 1e-6) {
        L = L / static_cast<float>(lmax);
    }
    cv::max(I, 0.0f, I);
    cv::min(I, 1.0f, I);
    cv::Mat D = BuildDistanceMapMeters(I, lidar_bev.resolution, cfg.chamfer_distance_cap_m);

    const double lidar_weight_sum = static_cast<double>(cv::sum(L)[0]);
    if (lidar_weight_sum < cfg.min_lidar_rail_weight_sum) {
        *out_delta = PoseDeltaBev{};
        if (debug_score) {
            debug_score->best_score_raw = 0.0;
            debug_score->best_score_norm = 0.0;
            debug_score->rail_score = 0.0;
            debug_score->pole_score = 0.0;
            debug_score->total = 0.0;
        }
        std::cerr << "[BEVMatcher] Not enough LiDAR rail weight: " << lidar_weight_sum << std::endl;
        return true;
    }

    const cv::Point2f center(static_cast<float>(L.cols - 1) * 0.5f, static_cast<float>(L.rows - 1) * 0.5f);

    double best_score = -1e300;
    PoseDeltaBev best;
    best.yaw_rad = 0.0;
    best.tx_m = 0.0;
    best.ty_m = 0.0;

    for (double yaw_deg = cfg.yaw_min_deg; yaw_deg <= cfg.yaw_max_deg + 1e-6; yaw_deg += cfg.yaw_step_deg) {
        cv::Mat M = cv::getRotationMatrix2D(center, yaw_deg, 1.0);
        cv::Mat D_rot;
        cv::warpAffine(
            D, D_rot, M, L.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
            static_cast<double>(cfg.chamfer_distance_cap_m));

        for (double tx_m = cfg.tx_min_m; tx_m <= cfg.tx_max_m + 1e-6; tx_m += cfg.trans_step_m) {
            for (double ty_m = cfg.ty_min_m; ty_m <= cfg.ty_max_m + 1e-6; ty_m += cfg.trans_step_m) {
                const int dx = static_cast<int>(std::llround(tx_m / lidar_bev.resolution));
                const int dy = static_cast<int>(std::llround(ty_m / lidar_bev.resolution));
                cv::Mat T = (cv::Mat_<double>(2, 3) << 1.0, 0.0, static_cast<double>(dx), 0.0, 1.0,
                             static_cast<double>(dy));
                cv::Mat D_warp;
                cv::warpAffine(
                    D_rot, D_warp, T, L.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                    static_cast<double>(cfg.chamfer_distance_cap_m));

                const double s = WeightedChamferScore(L, D_warp, cfg);
                if (s > best_score) {
                    best_score = s;
                    best.yaw_rad = yaw_deg * M_PI / 180.0;
                    best.tx_m = tx_m;
                    best.ty_m = ty_m;
                }
            }
        }
    }

    *out_delta = best;
    if (debug_score) {
        debug_score->best_score_raw = best_score;
        debug_score->best_score_norm = best_score;
        debug_score->rail_score = debug_score->best_score_norm;
        debug_score->pole_score = 0.0;
        debug_score->total = debug_score->rail_score;
    }
    std::cout << "[BEVMatcher] best score=" << best_score << " yaw_deg=" << (best.yaw_rad * 180.0 / M_PI)
              << " tx=" << best.tx_m << " ty=" << best.ty_m << std::endl;
    return true;
}

bool SaveBEVDebugImages(
    const std::string& path_overlay,
    const std::string& path_match,
    const BEVChannels& lidar_bev,
    const BEVChannels& image_bev,
    const PoseDeltaBev& delta) {
    cv::Mat L = RailToMat(lidar_bev);
    cv::Mat I = ResampleRailToGrid(image_bev, lidar_bev);
    cv::patchNaNs(L, 0);
    cv::patchNaNs(I, 0);

    const cv::Point2f center(static_cast<float>(L.cols - 1) * 0.5f, static_cast<float>(L.rows - 1) * 0.5f);
    const double yaw_deg = delta.yaw_rad * 180.0 / M_PI;
    cv::Mat M = cv::getRotationMatrix2D(center, yaw_deg, 1.0);
    cv::Mat I_rot;
    cv::warpAffine(I, I_rot, M, L.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    const int dx = static_cast<int>(std::llround(delta.tx_m / lidar_bev.resolution));
    const int dy = static_cast<int>(std::llround(delta.ty_m / lidar_bev.resolution));
    cv::Mat T = (cv::Mat_<double>(2, 3) << 1.0, 0.0, static_cast<double>(dx), 0.0, 1.0, static_cast<double>(dy));
    cv::Mat I_final;
    cv::warpAffine(I_rot, I_final, T, L.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    cv::Mat Ln, In;
    cv::normalize(L, Ln, 0, 255, cv::NORM_MINMAX);
    cv::normalize(I_final, In, 0, 255, cv::NORM_MINMAX);
    Ln.convertTo(Ln, CV_8U);
    In.convertTo(In, CV_8U);

    cv::Mat overlay(L.rows, L.cols, CV_8UC3);
    for (int y = 0; y < L.rows; ++y) {
        for (int x = 0; x < L.cols; ++x) {
            overlay.at<cv::Vec3b>(y, x) = cv::Vec3b(Ln.at<uint8_t>(y, x), In.at<uint8_t>(y, x), Ln.at<uint8_t>(y, x));
        }
    }
    cv::Mat pair;
    cv::hconcat(Ln, In, pair);
    cv::imwrite(path_overlay, pair);
    cv::imwrite(path_match, overlay);
    return true;
}
