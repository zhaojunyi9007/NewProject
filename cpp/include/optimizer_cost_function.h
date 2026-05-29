#pragma once

#include "include/common.h"
#include "include/optimizer_image_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "/usr/include/ceres/ceres.h"
#include "/usr/include/ceres/rotation.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <opencv2/opencv.hpp>

struct LabelStats {
    double mean_intensity = 0.0;
    double var_intensity = 0.0;
    Eigen::Vector3d mean_normal = Eigen::Vector3d::Zero();
    int count = 0;

    bool valid() const { return count > 10; }
};

struct TranslationPriorCost {
    TranslationPriorCost(const Eigen::Vector3d& t_ref, double weight)
        : t_ref_(t_ref), sqrt_weight_(std::sqrt(std::max(0.0, weight))) {}

    template <typename T>
    bool operator()(const T* const t, T* residual) const {
        residual[0] = T(sqrt_weight_) * (t[0] - T(t_ref_.x()));
        residual[1] = T(sqrt_weight_) * (t[1] - T(t_ref_.y()));
        residual[2] = T(sqrt_weight_) * (t[2] - T(t_ref_.z()));
        return true;
    }

    Eigen::Vector3d t_ref_;
    double sqrt_weight_;
};

struct EdgeConsistencyCost {
    EdgeConsistencyCost(const std::vector<PointFeature>* points,
                        const std::vector<int>* indices,
                        const cv::Mat* dist_map,
                        const cv::Mat* semantic_map,
                        const std::vector<LabelStats>* label_stats,
                        const Eigen::Matrix3d& R_rect,
                        const Eigen::Matrix<double, 3, 4>& P_rect,
                        int W, int H,
                        double w_edge,
                        double w_consistency)
        : points_(points),
          indices_(indices),
          dist_map_(dist_map),
          semantic_map_(semantic_map),
          label_stats_(label_stats),
          R_rect_(R_rect),
          P_rect_(P_rect),
          W_(W),
          H_(H),
          w_edge_(w_edge),
          w_consistency_(w_consistency) {}

    bool operator()(const double* const r, const double* const t, double* residual) const {
        double total_error = 0.0;
        int visible_count = 0;

        for (int idx : *indices_) {
            const auto& pt = (*points_)[idx];

            double p_raw[3] = { pt.p.x(), pt.p.y(), pt.p.z() };
            double p_rotated[3];
            ceres::AngleAxisRotatePoint(r, p_raw, p_rotated);

            Eigen::Vector3d p_cam;
            p_cam.x() = p_rotated[0] + t[0];
            p_cam.y() = p_rotated[1] + t[1];
            p_cam.z() = p_rotated[2] + t[2];

            Eigen::Vector3d p_rect = R_rect_ * p_cam;
            if (p_rect.z() < 0.1) continue;

            Eigen::Vector4d p_rect_h;
            p_rect_h[0] = p_rect.x();
            p_rect_h[1] = p_rect.y();
            p_rect_h[2] = p_rect.z();
            p_rect_h[3] = 1.0;
            Eigen::Vector3d uv = P_rect_ * p_rect_h;

            int u = static_cast<int>(uv.x() / uv.z());
            int v = static_cast<int>(uv.y() / uv.z());
            if (u < 0 || u >= W_ || v < 0 || v >= H_) continue;

            double edge_error = 0.0;
            if (dist_map_ && !dist_map_->empty()) {
                edge_error = static_cast<double>(GetDistanceValue(*dist_map_, u, v));
                edge_error = std::min(std::max(edge_error, 0.0), 1.0);
            }

            double consistency_error = 0.0;
            if (semantic_map_ && !semantic_map_->empty()) {
                int img_label = GetSemanticLabel(*semantic_map_, u, v);

                if (pt.label != 0 && img_label != pt.label) {
                    consistency_error += 1.0;
                }

                if (label_stats_ && img_label >= 0 && img_label < static_cast<int>(label_stats_->size())) {
                    const auto& stats = (*label_stats_)[img_label];
                    if (stats.valid()) {
                        double std_intensity = std::sqrt(stats.var_intensity) + 1e-3;
                        double intensity_error = std::abs(pt.intensity - stats.mean_intensity) / std_intensity;
                        Eigen::Vector3d n = pt.normal;
                        if (n.norm() > 1e-6) {
                            n.normalize();
                        }
                        double normal_error = 1.0 - std::max(-1.0, std::min(1.0, n.dot(stats.mean_normal)));
                        consistency_error += intensity_error + normal_error;
                    }
                }
            }

            double weight = pt.weight;
            total_error += weight * (w_edge_ * edge_error + w_consistency_ * consistency_error);
            visible_count++;
        }

        if (visible_count < 50) {
            residual[0] = 1e3;
        } else {
            residual[0] = total_error / static_cast<double>(visible_count);
        }
        return true;
    }

    const std::vector<PointFeature>* points_;
    const std::vector<int>* indices_;
    const cv::Mat* dist_map_;
    const cv::Mat* semantic_map_;
    const std::vector<LabelStats>* label_stats_;

    Eigen::Matrix3d R_rect_;
    Eigen::Matrix<double, 3, 4> P_rect_;
    int W_;
    int H_;
    double w_edge_;
    double w_consistency_;
};

struct SinglePointEdgeCost {
    SinglePointEdgeCost(const PointFeature& pt,
                        const cv::Mat* dist_map,
                        const Eigen::Matrix3d& R_rect,
                        const Eigen::Matrix<double, 3, 4>& P_rect,
                        int W, int H)
        : pt_(pt), dist_map_(dist_map), R_rect_(R_rect), P_rect_(P_rect), W_(W), H_(H) {}

    template <typename T>
    bool operator()(const T* const r, const T* const t, T* residual) const {
        T p_raw[3] = { T(pt_.p.x()), T(pt_.p.y()), T(pt_.p.z()) };
        T p_rotated[3];
        ceres::AngleAxisRotatePoint(r, p_raw, p_rotated);

        Eigen::Matrix<T, 3, 1> p_cam;
        p_cam.x() = p_rotated[0] + t[0];
        p_cam.y() = p_rotated[1] + t[1];
        p_cam.z() = p_rotated[2] + t[2];

        Eigen::Matrix<T, 3, 1> p_rect = R_rect_.cast<T>() * p_cam;
        if (ScalarValue(p_rect.z()) < 0.1) {
            residual[0] = T(0.0);
            return true;
        }

        Eigen::Matrix<T, 4, 1> p_rect_h;
        p_rect_h[0] = p_rect.x();
        p_rect_h[1] = p_rect.y();
        p_rect_h[2] = p_rect.z();
        p_rect_h[3] = T(1.0);
        Eigen::Matrix<T, 3, 1> uv = P_rect_.cast<T>() * p_rect_h;

        T u_f = uv.x() / uv.z();
        T v_f = uv.y() / uv.z();

        const double u_scalar = ScalarValue(u_f);
        const double v_scalar = ScalarValue(v_f);

        T edge_error = T(0.0);
        if (dist_map_ && !dist_map_->empty()) {
            edge_error = BilinearInterpolateT(*dist_map_, u_f, v_f);
            const double edge_scalar = ScalarValue(edge_error);
            if (edge_scalar < 0.0) {
                edge_error = T(0.0);
            } else if (edge_scalar > 1.0) {
                edge_error = T(1.0);
            }
        }

        // Keep non-zero, geometry-dependent gradient for out-of-FOV points.
        // Using smooth hinge avoids hard discontinuity at borders.
        const T eps = T(1e-6);
        auto smooth_hinge = [&](const T& x) -> T {
            return (x + ceres::sqrt(x * x + eps)) * T(0.5);
        };

        T boundary_penalty = T(0.0);
        if (u_scalar < 0 || u_scalar >= W_ - 1 || v_scalar < 0 || v_scalar >= H_ - 1) {
            const T left = smooth_hinge(-u_f);
            const T right = smooth_hinge(u_f - T(W_ - 1));
            const T top = smooth_hinge(-v_f);
            const T bottom = smooth_hinge(v_f - T(H_ - 1));
            boundary_penalty = (left + right + top + bottom) / T(W_ + H_);
        }

        residual[0] = (edge_error + boundary_penalty) * T(pt_.weight);
        return true;
    }

    PointFeature pt_;
    const cv::Mat* dist_map_;
    Eigen::Matrix3d R_rect_;
    Eigen::Matrix<double, 3, 4> P_rect_;
    int W_;
    int H_;
};

struct WeightedRailEdgeCost {
    WeightedRailEdgeCost(const PointFeature& pt,
                         const cv::Mat* dist_map,
                         const cv::Mat* weight_map,
                         const Eigen::Matrix3d& R_rect,
                         const Eigen::Matrix<double, 3, 4>& P_rect,
                         int W, int H,
                         double visibility_residual_weight = 0.0,
                         double oob_residual_weight = 0.0)
        : pt_(pt), dist_map_(dist_map), weight_map_(weight_map),
          R_rect_(R_rect), P_rect_(P_rect), W_(W), H_(H),
          visibility_residual_weight_(visibility_residual_weight),
          oob_residual_weight_(oob_residual_weight) {}

    template <typename T>
    bool operator()(const T* const r, const T* const t, T* residual) const {
        T p_raw[3] = { T(pt_.p.x()), T(pt_.p.y()), T(pt_.p.z()) };
        T p_rotated[3];
        ceres::AngleAxisRotatePoint(r, p_raw, p_rotated);

        Eigen::Matrix<T, 3, 1> p_cam;
        p_cam.x() = p_rotated[0] + t[0];
        p_cam.y() = p_rotated[1] + t[1];
        p_cam.z() = p_rotated[2] + t[2];

        Eigen::Matrix<T, 3, 1> p_rect = R_rect_.cast<T>() * p_cam;
        if (ScalarValue(p_rect.z()) < 0.1) {
            residual[0] = T(oob_residual_weight_) * T(std::sqrt(std::max(0.0, pt_.weight)));
            return true;
        }

        Eigen::Matrix<T, 4, 1> p_rect_h;
        p_rect_h[0] = p_rect.x();
        p_rect_h[1] = p_rect.y();
        p_rect_h[2] = p_rect.z();
        p_rect_h[3] = T(1.0);
        Eigen::Matrix<T, 3, 1> uv = P_rect_.cast<T>() * p_rect_h;

        T u_f = uv.x() / uv.z();
        T v_f = uv.y() / uv.z();
        const double u_scalar = ScalarValue(u_f);
        const double v_scalar = ScalarValue(v_f);
        const T eps = T(1e-6);
        auto smooth_hinge = [&](const T& x) -> T {
            return (x + ceres::sqrt(x * x + eps)) * T(0.5);
        };
        if (u_scalar < 0 || u_scalar >= W_ - 1 || v_scalar < 0 || v_scalar >= H_ - 1) {
            const T left = smooth_hinge(-u_f);
            const T right = smooth_hinge(u_f - T(W_ - 1));
            const T top = smooth_hinge(-v_f);
            const T bottom = smooth_hinge(v_f - T(H_ - 1));
            residual[0] = T(oob_residual_weight_) * (left + right + top + bottom) / T(W_ + H_) *
                          T(std::sqrt(std::max(0.0, pt_.weight)));
            return true;
        }

        T dist = T(0.0);
        if (dist_map_ && !dist_map_->empty()) {
            dist = BilinearInterpolateT(*dist_map_, u_f, v_f);
            const double d = ScalarValue(dist);
            if (d < 0.0) dist = T(0.0);
            if (d > 1.0) dist = T(1.0);
        }

        T image_weight = T(1.0);
        if (weight_map_ && !weight_map_->empty()) {
            image_weight = BilinearInterpolateT(*weight_map_, u_f, v_f);
            const double iw = ScalarValue(image_weight);
            if (!std::isfinite(iw)) {
                residual[0] = T(visibility_residual_weight_) * T(std::sqrt(std::max(0.0, pt_.weight)));
                return true;
            }
            if (iw < 0.0) image_weight = T(0.0);
            if (iw > 1.0) image_weight = T(1.0);
            if (iw <= 1e-4) {
                residual[0] = T(visibility_residual_weight_) * T(std::sqrt(std::max(0.0, pt_.weight))) *
                              (T(1.0) - image_weight);
                return true;
            }
        }

        residual[0] = dist * ceres::sqrt(T(std::max(0.0, pt_.weight)) * image_weight);
        return true;
    }

    PointFeature pt_;
    const cv::Mat* dist_map_;
    const cv::Mat* weight_map_;
    Eigen::Matrix3d R_rect_;
    Eigen::Matrix<double, 3, 4> P_rect_;
    int W_;
    int H_;
    double visibility_residual_weight_;
    double oob_residual_weight_;
};

inline double ClampDouble(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

template <typename T>
T PointSegmentDistance2DT(const T& px, const T& py,
                          double ax, double ay, double bx, double by) {
    const T vx = T(bx - ax);
    const T vy = T(by - ay);
    const T wx = px - T(ax);
    const T wy = py - T(ay);
    const T denom = vx * vx + vy * vy + T(1e-9);
    T alpha = (wx * vx + wy * vy) / denom;
    const double a = ScalarValue(alpha);
    if (a < 0.0) alpha = T(0.0);
    if (a > 1.0) alpha = T(1.0);
    const T qx = T(ax) + alpha * vx;
    const T qy = T(ay) + alpha * vy;
    const T dx = px - qx;
    const T dy = py - qy;
    return ceres::sqrt(dx * dx + dy * dy + T(1e-6));
}

template <typename T>
T StrongProjectU(const Eigen::Vector3d& p,
                 const Eigen::Matrix3d& R_rect,
                 const Eigen::Matrix<double, 3, 4>& P_rect,
                 const T* const r,
                 const T* const t,
                 T* v_out,
                 bool* valid_depth) {
    T p_raw[3] = {T(p.x()), T(p.y()), T(p.z())};
    T p_rotated[3];
    ceres::AngleAxisRotatePoint(r, p_raw, p_rotated);
    Eigen::Matrix<T, 3, 1> p_cam;
    p_cam.x() = p_rotated[0] + t[0];
    p_cam.y() = p_rotated[1] + t[1];
    p_cam.z() = p_rotated[2] + t[2];
    Eigen::Matrix<T, 3, 1> p_rect = R_rect.cast<T>() * p_cam;
    *valid_depth = ScalarValue(p_rect.z()) >= 0.1;
    Eigen::Matrix<T, 4, 1> p_rect_h;
    p_rect_h[0] = p_rect.x();
    p_rect_h[1] = p_rect.y();
    p_rect_h[2] = p_rect.z();
    p_rect_h[3] = T(1.0);
    Eigen::Matrix<T, 3, 1> uv = P_rect.cast<T>() * p_rect_h;
    *v_out = uv.y() / uv.z();
    return uv.x() / uv.z();
}

struct TrackPolylineProjectionCost {
    TrackPolylineProjectionCost(const StrongLabelFeature& feature,
                                const Eigen::Matrix3d& R_rect,
                                const Eigen::Matrix<double, 3, 4>& P_rect,
                                int W, int H,
                                double weight)
        : feature_(feature), R_rect_(R_rect), P_rect_(P_rect), W_(W), H_(H),
          sqrt_weight_(std::sqrt(std::max(0.0, weight * feature.weight))) {}

    template <typename T>
    bool operator()(const T* const r, const T* const t, T* residual) const {
        if (feature_.image_points.size() < 2) {
            residual[0] = T(0.0);
            return true;
        }
        bool valid_depth = true;
        T v_f = T(0.0);
        T u_f = StrongProjectU(feature_.p1, R_rect_, P_rect_, r, t, &v_f, &valid_depth);
        if (!valid_depth) {
            residual[0] = T(sqrt_weight_);
            return true;
        }
        T best = T(1e6);
        double best_scalar = 1e6;
        for (size_t i = 1; i < feature_.image_points.size(); ++i) {
            const auto& a = feature_.image_points[i - 1];
            const auto& b = feature_.image_points[i];
            T d = PointSegmentDistance2DT(u_f, v_f, a.x(), a.y(), b.x(), b.y());
            const double ds = ScalarValue(d);
            if (ds < best_scalar) {
                best_scalar = ds;
                best = d;
            }
        }
        const T eps = T(1e-6);
        auto smooth_hinge = [&](const T& x) -> T { return (x + ceres::sqrt(x * x + eps)) * T(0.5); };
        T oob = (smooth_hinge(-u_f) + smooth_hinge(u_f - T(W_ - 1)) +
                 smooth_hinge(-v_f) + smooth_hinge(v_f - T(H_ - 1))) / T(W_ + H_);
        residual[0] = T(sqrt_weight_) * (best / T(80.0) + oob);
        return true;
    }

    StrongLabelFeature feature_;
    Eigen::Matrix3d R_rect_;
    Eigen::Matrix<double, 3, 4> P_rect_;
    int W_;
    int H_;
    double sqrt_weight_;
};

struct PoleCenterlineProjectionCost {
    PoleCenterlineProjectionCost(const StrongLabelFeature& feature,
                                 const Eigen::Matrix3d& R_rect,
                                 const Eigen::Matrix<double, 3, 4>& P_rect,
                                 int W, int H,
                                 double weight)
        : feature_(feature), R_rect_(R_rect), P_rect_(P_rect), W_(W), H_(H),
          sqrt_weight_(std::sqrt(std::max(0.0, weight * feature.weight))) {}

    template <typename T>
    bool operator()(const T* const r, const T* const t, T* residual) const {
        if (feature_.image_points.size() < 2) {
            residual[0] = residual[1] = residual[2] = T(0.0);
            return true;
        }
        bool ok1 = true, ok2 = true;
        T v1 = T(0.0), v2 = T(0.0);
        T u1 = StrongProjectU(feature_.p1, R_rect_, P_rect_, r, t, &v1, &ok1);
        T u2 = StrongProjectU(feature_.p2, R_rect_, P_rect_, r, t, &v2, &ok2);
        const auto& a = feature_.image_points[0];
        const auto& b = feature_.image_points[1];
        T d1 = PointSegmentDistance2DT(u1, v1, a.x(), a.y(), b.x(), b.y());
        T d2 = PointSegmentDistance2DT(u2, v2, a.x(), a.y(), b.x(), b.y());
        T du = u2 - u1;
        T dv = v2 - v1;
        T angle = ceres::sqrt(du * du + T(1e-6)) / (ceres::sqrt(du * du + dv * dv + T(1e-6)));
        if (!ok1) d1 = T(80.0);
        if (!ok2) d2 = T(80.0);
        residual[0] = T(sqrt_weight_) * d1 / T(60.0);
        residual[1] = T(sqrt_weight_) * d2 / T(60.0);
        residual[2] = T(sqrt_weight_) * angle;
        return true;
    }

    StrongLabelFeature feature_;
    Eigen::Matrix3d R_rect_;
    Eigen::Matrix<double, 3, 4> P_rect_;
    int W_;
    int H_;
    double sqrt_weight_;
};

struct BufferStopBBoxProjectionCost {
    BufferStopBBoxProjectionCost(const StrongLabelFeature& feature,
                                 const Eigen::Matrix3d& R_rect,
                                 const Eigen::Matrix<double, 3, 4>& P_rect,
                                 int W, int H,
                                 double weight)
        : feature_(feature), R_rect_(R_rect), P_rect_(P_rect), W_(W), H_(H),
          sqrt_weight_(std::sqrt(std::max(0.0, weight * feature.weight))) {}

    template <typename T>
    bool operator()(const T* const r, const T* const t, T* residual) const {
        bool valid_depth = true;
        T v = T(0.0);
        T u = StrongProjectU(feature_.p1, R_rect_, P_rect_, r, t, &v, &valid_depth);
        if (!valid_depth) {
            residual[0] = T(sqrt_weight_);
            return true;
        }
        const T x0 = T(feature_.bbox[0]);
        const T y0 = T(feature_.bbox[1]);
        const T x1 = T(feature_.bbox[2]);
        const T y1 = T(feature_.bbox[3]);
        const T eps = T(1e-6);
        auto smooth_hinge = [&](const T& x) -> T { return (x + ceres::sqrt(x * x + eps)) * T(0.5); };
        T outside = smooth_hinge(x0 - u) + smooth_hinge(u - x1) + smooth_hinge(y0 - v) + smooth_hinge(v - y1);
        T scale = ceres::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) + T(1.0));
        if (feature_.role.find("center") != std::string::npos) {
            const T cx = (x0 + x1) * T(0.5);
            const T cy = (y0 + y1) * T(0.5);
            T center_dist = ceres::sqrt((u - cx) * (u - cx) + (v - cy) * (v - cy) + T(1e-6));
            residual[0] = T(sqrt_weight_) * (outside + T(0.35) * center_dist) / scale;
        } else {
            residual[0] = T(sqrt_weight_) * outside / scale;
        }
        return true;
    }

    StrongLabelFeature feature_;
    Eigen::Matrix3d R_rect_;
    Eigen::Matrix<double, 3, 4> P_rect_;
    int W_;
    int H_;
    double sqrt_weight_;
};
