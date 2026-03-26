#pragma once

#include <cmath>

#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

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
    if (dist_map.empty()) return 1.0f;
    if (dist_map.type() == CV_16U) return static_cast<float>(dist_map.at<ushort>(v, u)) / 65535.0f;
    if (dist_map.type() == CV_32F) return dist_map.at<float>(v, u);
    if (dist_map.type() == CV_64F) return static_cast<float>(dist_map.at<double>(v, u));
    return 1.0f;
}

inline double BilinearInterpolate(const cv::Mat& img, double x, double y) {
    int x1 = static_cast<int>(std::floor(x));
    int y1 = static_cast<int>(std::floor(y));
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    if (x1 < 0 || x2 >= img.cols || y1 < 0 || y2 >= img.rows) {
        if (x1 >= 0 && x1 < img.cols && y1 >= 0 && y1 < img.rows) return GetDistanceValue(img, x1, y1);
        return 1.0;
    }

    double v11 = GetDistanceValue(img, x1, y1);
    double v12 = GetDistanceValue(img, x1, y2);
    double v21 = GetDistanceValue(img, x2, y1);
    double v22 = GetDistanceValue(img, x2, y2);

    double wx = x - x1;
    double wy = y - y1;

    double top = v11 * (1.0 - wx) + v21 * wx;
    double bot = v12 * (1.0 - wx) + v22 * wx;

    return top * (1.0 - wy) + bot * wy;
}

template <typename T>
inline T BilinearInterpolateT(const cv::Mat& img, const T& x, const T& y) {
    const double x_scalar = ceres::JetOps<T>::GetScalar(x);
    const double y_scalar = ceres::JetOps<T>::GetScalar(y);
    int x1 = static_cast<int>(std::floor(x_scalar));
    int y1 = static_cast<int>(std::floor(y_scalar));
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    if (x1 < 0 || x2 >= img.cols || y1 < 0 || y2 >= img.rows) {
        if (x1 >= 0 && x1 < img.cols && y1 >= 0 && y1 < img.rows) {
            return T(GetDistanceValue(img, x1, y1));
        }
        return T(1.0);
    }

    const T v11 = T(GetDistanceValue(img, x1, y1));
    const T v12 = T(GetDistanceValue(img, x1, y2));
    const T v21 = T(GetDistanceValue(img, x2, y1));
    const T v22 = T(GetDistanceValue(img, x2, y2));

    const T wx = x - T(x1);
    const T wy = y - T(y1);
    const T top = v11 * (T(1.0) - wx) + v21 * wx;
    const T bot = v12 * (T(1.0) - wx) + v22 * wx;
    return top * (T(1.0) - wy) + bot * wy;
}
