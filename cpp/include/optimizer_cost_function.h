#pragma once

#include "include/common.h"
#include "include/optimizer_image_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
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
