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
    cv::Mat I = RailToMat(image_bev);
    cv::resize(I, I, L.size(), 0, 0, cv::INTER_LINEAR);
    cv::patchNaNs(L, 0);
    cv::patchNaNs(I, 0);
    cv::normalize(L, L, 0, 1, cv::NORM_MINMAX);
    cv::normalize(I, I, 0, 1, cv::NORM_MINMAX);

    const cv::Point2f center(static_cast<float>(L.cols - 1) * 0.5f, static_cast<float>(L.rows - 1) * 0.5f);

    double best_score = -1e300;
    PoseDeltaBev best;
    best.yaw_rad = 0.0;
    best.tx_m = 0.0;
    best.ty_m = 0.0;

    for (double yaw_deg = cfg.yaw_min_deg; yaw_deg <= cfg.yaw_max_deg + 1e-6; yaw_deg += cfg.yaw_step_deg) {
        cv::Mat M = cv::getRotationMatrix2D(center, yaw_deg, 1.0);
        cv::Mat I_rot;
        cv::warpAffine(I, I_rot, M, L.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

        for (double tx_m = cfg.tx_min_m; tx_m <= cfg.tx_max_m + 1e-6; tx_m += cfg.trans_step_m) {
            for (double ty_m = cfg.ty_min_m; ty_m <= cfg.ty_max_m + 1e-6; ty_m += cfg.trans_step_m) {
                const int dx = static_cast<int>(std::llround(tx_m / lidar_bev.resolution));
                const int dy = static_cast<int>(std::llround(ty_m / lidar_bev.resolution));
                cv::Mat T = (cv::Mat_<double>(2, 3) << 1.0, 0.0, static_cast<double>(dx), 0.0, 1.0,
                             static_cast<double>(dy));
                cv::Mat I_warp;
                cv::warpAffine(I_rot, I_warp, T, L.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

                cv::Mat prod;
                cv::multiply(L, I_warp, prod);
                const double s = static_cast<double>(cv::sum(prod)[0]);
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
        const double denom = static_cast<double>(std::max(1, L.rows * L.cols));
        debug_score->rail_score = best_score / denom;
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
    cv::Mat I = RailToMat(image_bev);
    cv::resize(I, I, L.size(), 0, 0, cv::INTER_LINEAR);
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
