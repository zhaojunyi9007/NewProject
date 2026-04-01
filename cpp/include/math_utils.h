#pragma once

#include <cmath>

#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <opencv2/opencv.hpp>

template <typename T>
inline double ScalarValue(const T& value) {
    return static_cast<double>(value);
}

template <typename T, int N>
inline double ScalarValue(const ceres::Jet<T, N>& value) {
    return static_cast<double>(value.a);
}

inline float GetDistanceValueT(const cv::Mat& dist_map, int u, int v) {
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
        if (x1 >= 0 && x1 < img.cols && y1 >= 0 && y1 < img.rows) return GetDistanceValueT(img, x1, y1);
        return 1.0;
    }

    double v11 = GetDistanceValueT(img, x1, y1);
    double v12 = GetDistanceValueT(img, x1, y2);
    double v21 = GetDistanceValueT(img, x2, y1);
    double v22 = GetDistanceValueT(img, x2, y2);

    double wx = x - x1;
    double wy = y - y1;

    double top = v11 * (1.0 - wx) + v21 * wx;
    double bot = v12 * (1.0 - wx) + v22 * wx;

    return top * (1.0 - wy) + bot * wy;
}

template <typename T>
inline T BilinearInterpolateT(const cv::Mat& img, const T& x, const T& y) {
    double x_val = ScalarValue(x);
    double y_val = ScalarValue(y);

    int x1 = static_cast<int>(std::floor(x_val));
    int y1 = static_cast<int>(std::floor(y_val));
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    if (x1 < 0 || x2 >= img.cols || y1 < 0 || y2 >= img.rows) {
        if (x1 >= 0 && x1 < img.cols && y1 >= 0 && y1 < img.rows) {
            return T(GetDistanceValueT(img, x1, y1));
        }
        return T(1.0);
    }

    T v11 = T(GetDistanceValueT(img, x1, y1));
    T v12 = T(GetDistanceValueT(img, x1, y2));
    T v21 = T(GetDistanceValueT(img, x2, y1));
    T v22 = T(GetDistanceValueT(img, x2, y2));

    T wx = x - T(x1);
    T wy = y - T(y1);

    T top = v11 * (T(1.0) - wx) + v21 * wx;
    T bot = v12 * (T(1.0) - wx) + v22 * wx;
    return top * (T(1.0) - wy) + bot * wy;
}
