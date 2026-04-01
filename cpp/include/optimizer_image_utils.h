#pragma once

#include <opencv2/opencv.hpp>

#include "include/math_utils.h"

inline int GetMaxLabel(const cv::Mat& semantic_map) {
    if (semantic_map.empty()) {
        return 0;
    }
    if (semantic_map.type() == CV_16U || semantic_map.type() == CV_32S) {
        double min_v = 0.0;
        double max_v = 0.0;
        cv::minMaxLoc(semantic_map, &min_v, &max_v);
        return static_cast<int>(max_v);
    }
    if (semantic_map.type() == CV_8U) {
        double min_v = 0.0;
        double max_v = 0.0;
        cv::minMaxLoc(semantic_map, &min_v, &max_v);
        return static_cast<int>(max_v);
    }
    return 0;
}

inline int GetSemanticLabel(const cv::Mat& semantic_map, int u, int v) {
    if (semantic_map.empty()) return 0;
    if (semantic_map.type() == CV_16U) return semantic_map.at<ushort>(v, u);
    if (semantic_map.type() == CV_8U) return semantic_map.at<uchar>(v, u);
    if (semantic_map.type() == CV_32S) return semantic_map.at<int>(v, u);
    return 0;
}

inline float GetDistanceValue(const cv::Mat& dist_map, int u, int v) {
    return GetDistanceValueT(dist_map, u, v);
}
