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

struct LineReprojectionError {
    LineReprojectionError(const Line3D& l3d,
                          const std::vector<Line2D>& l2ds,
                          const Eigen::Matrix3d& R_rect,
                          const Eigen::Matrix<double, 3, 4>& P_rect,
                          double match_threshold = 50.0,
                          bool soft_penalty = false,
                          double soft_cap = 100.0,
                          double behind_penalty = 0.0,
                          double unmatched_penalty = 0.0,
                          double threshold_fail_penalty = 0.0)
        : l3d_(l3d), l2ds_(l2ds), R_rect_(R_rect), P_rect_(P_rect),
          match_threshold_(match_threshold), soft_penalty_(soft_penalty), soft_cap_(soft_cap),
          behind_penalty_(behind_penalty), unmatched_penalty_(unmatched_penalty),
          threshold_fail_penalty_(threshold_fail_penalty) {}

    template <typename T>
    bool operator()(const T* const r, const T* const t, T* residual) const {
        T p1[3], p2[3];
        T p1_raw[3] = {T(l3d_.p1.x()), T(l3d_.p1.y()), T(l3d_.p1.z())};
        T p2_raw[3] = {T(l3d_.p2.x()), T(l3d_.p2.y()), T(l3d_.p2.z())};

        ceres::AngleAxisRotatePoint(r, p1_raw, p1);
        ceres::AngleAxisRotatePoint(r, p2_raw, p2);
        p1[0] += t[0]; p1[1] += t[1]; p1[2] += t[2];
        p2[0] += t[0]; p2[1] += t[1]; p2[2] += t[2];

        Eigen::Matrix<T, 3, 1> p1_cam(p1[0], p1[1], p1[2]);
        Eigen::Matrix<T, 3, 1> p2_cam(p2[0], p2[1], p2[2]);
        Eigen::Matrix<T, 3, 1> p1_rect = R_rect_.cast<T>() * p1_cam;
        Eigen::Matrix<T, 3, 1> p2_rect = R_rect_.cast<T>() * p2_cam;

        if (p1_rect[2] < T(0.1) || p2_rect[2] < T(0.1)) {
            residual[0] = T(behind_penalty_);
            return true;
        }
        Eigen::Matrix<T, 4, 1> p1_h(p1_rect[0], p1_rect[1], p1_rect[2], T(1.0));
        Eigen::Matrix<T, 4, 1> p2_h(p2_rect[0], p2_rect[1], p2_rect[2], T(1.0));
        Eigen::Matrix<T, 3, 4> P_rect_t = P_rect_.cast<T>();
        Eigen::Matrix<T, 3, 1> uv1 = P_rect_t * p1_h;
        Eigen::Matrix<T, 3, 1> uv2 = P_rect_t * p2_h;

        T u1 = uv1[0] / uv1[2];
        T v1 = uv1[1] / uv1[2];
        T u2 = uv2[0] / uv2[2];
        T v2 = uv2[1] / uv2[2];

        T min_dist = T(1000.0);
        bool found = false;

        for (const auto& l2d : l2ds_) {
            if (l2d.type != l3d_.type) continue;

            double lx1 = l2d.p1.x(), ly1 = l2d.p1.y();
            double lx2 = l2d.p2.x(), ly2 = l2d.p2.y();

            double A = ly1 - ly2;
            double B = lx2 - lx1;
            double C = lx1 * ly2 - lx2 * ly1;
            double norm = std::sqrt(A * A + B * B);

            if (norm < 1e-6) continue;

            T d1 = ceres::abs(T(A) * u1 + T(B) * v1 + T(C)) / T(norm);
            T d2 = ceres::abs(T(A) * u2 + T(B) * v2 + T(C)) / T(norm);

            T avg_dist = (d1 + d2) / T(2.0);
            if (avg_dist < min_dist) {
                min_dist = avg_dist;
                found = true;
            }
        }

        if (found && min_dist < T(match_threshold_)) {
            residual[0] = min_dist;
        } else if (soft_penalty_) {
            T soft_cap = T(soft_cap_);
            if (found) {
                residual[0] = soft_cap * (T(1.0) - ceres::exp(-min_dist / soft_cap));
            } else {
                residual[0] = soft_cap;
            }
        } else {
            if (!found) {
                residual[0] = T(unmatched_penalty_);
            } else {
                residual[0] = T(threshold_fail_penalty_);
            }
        }
        return true;
    }

    Line3D l3d_;
    std::vector<Line2D> l2ds_;
    Eigen::Matrix3d R_rect_;
    Eigen::Matrix<double, 3, 4> P_rect_;
    double match_threshold_;
    bool soft_penalty_;
    double soft_cap_;
    double behind_penalty_;
    double unmatched_penalty_;
    double threshold_fail_penalty_;
};

struct LineMatchStats {
    bool in_front = false;
    bool found_type_match = false;
    bool active = false;
    double min_dist = 1e9;
};

inline const char* GetLineStatusReason(const LineMatchStats& stats, double threshold) {
    if (!stats.in_front) return "behind";
    if (!stats.found_type_match) return "unmatched_type";
    if (!stats.active) {
        if (stats.min_dist >= 1e8) return "unmatched_or_invalid_2d";
        if (stats.min_dist >= threshold) return "threshold_fail";
        return "inactive_other";
    }
    return "active";
}

inline LineMatchStats EvaluateLineMatchStats(const Line3D& l3d,
                                             const std::vector<Line2D>& l2ds,
                                             const Eigen::Matrix3d& R_rect,
                                             const Eigen::Matrix<double, 3, 4>& P_rect,
                                             const double* r,
                                             const double* t,
                                             double threshold = 50.0) {
    LineMatchStats stats;
    double p1_raw[3] = {l3d.p1.x(), l3d.p1.y(), l3d.p1.z()};
    double p2_raw[3] = {l3d.p2.x(), l3d.p2.y(), l3d.p2.z()};
    double p1[3];
    double p2[3];
    ceres::AngleAxisRotatePoint(r, p1_raw, p1);
    ceres::AngleAxisRotatePoint(r, p2_raw, p2);
    p1[0] += t[0]; p1[1] += t[1]; p1[2] += t[2];
    p2[0] += t[0]; p2[1] += t[1]; p2[2] += t[2];

    Eigen::Vector3d p1_cam(p1[0], p1[1], p1[2]);
    Eigen::Vector3d p2_cam(p2[0], p2[1], p2[2]);
    Eigen::Vector3d p1_rect = R_rect * p1_cam;
    Eigen::Vector3d p2_rect = R_rect * p2_cam;
    if (p1_rect.z() < 0.1 || p2_rect.z() < 0.1) {
        return stats;
    }
    stats.in_front = true;

    Eigen::Vector4d p1_h(p1_rect.x(), p1_rect.y(), p1_rect.z(), 1.0);
    Eigen::Vector4d p2_h(p2_rect.x(), p2_rect.y(), p2_rect.z(), 1.0);
    Eigen::Vector3d uv1 = P_rect * p1_h;
    Eigen::Vector3d uv2 = P_rect * p2_h;
    double u1 = uv1.x() / uv1.z();
    double v1 = uv1.y() / uv1.z();
    double u2 = uv2.x() / uv2.z();
    double v2 = uv2.y() / uv2.z();

    for (const auto& l2d : l2ds) {
        if (l2d.type != l3d.type) continue;
        stats.found_type_match = true;

        double lx1 = l2d.p1.x(), ly1 = l2d.p1.y();
        double lx2 = l2d.p2.x(), ly2 = l2d.p2.y();
        double A = ly1 - ly2;
        double B = lx2 - lx1;
        double C = lx1 * ly2 - lx2 * ly1;
        double norm = std::sqrt(A * A + B * B);
        if (norm < 1e-6) continue;

        double d1 = std::abs(A * u1 + B * v1 + C) / norm;
        double d2 = std::abs(A * u2 + B * v2 + C) / norm;
        double avg_dist = 0.5 * (d1 + d2);
        if (avg_dist < stats.min_dist) {
            stats.min_dist = avg_dist;
        }
    }
    stats.active = stats.found_type_match && stats.min_dist < threshold;
    return stats;
}

inline bool IsLineConstraintActive(const Line3D& l3d,
                                   const std::vector<Line2D>& l2ds,
                                   const Eigen::Matrix3d& R_rect,
                                   const Eigen::Matrix<double, 3, 4>& P_rect,
                                   const double* r,
                                   const double* t,
                                   double threshold = 50.0) {
    LineMatchStats stats = EvaluateLineMatchStats(l3d, l2ds, R_rect, P_rect, r, t, threshold);
    return stats.active;
}

struct LineConstraintSummary {
    int total = 0;
    int active = 0;
    int behind = 0;
    int unmatched_type = 0;
    int threshold_fail = 0;
    int unmatched_or_invalid_2d = 0;
    int inactive_other = 0;
    double min_dist_sum = 0.0;
    int min_dist_count = 0;
};

inline LineConstraintSummary SummarizeLineConstraints(const std::vector<Line3D>& lines3d,
                                                      const std::vector<Line2D>& lines2d,
                                                      const Eigen::Matrix3d& R_rect,
                                                      const Eigen::Matrix<double, 3, 4>& P_rect,
                                                      const double* r,
                                                      const double* t,
                                                      double threshold) {
    LineConstraintSummary summary;
    summary.total = static_cast<int>(lines3d.size());
    for (const auto& l3d : lines3d) {
        LineMatchStats stats = EvaluateLineMatchStats(l3d, lines2d, R_rect, P_rect, r, t, threshold);
        if (stats.active) {
            summary.active++;
        }
        if (stats.min_dist < 1e8) {
            summary.min_dist_sum += stats.min_dist;
            summary.min_dist_count++;
        }

        const std::string reason(GetLineStatusReason(stats, threshold));
        if (reason == "behind") {
            summary.behind++;
        } else if (reason == "unmatched_type") {
            summary.unmatched_type++;
        } else if (reason == "threshold_fail") {
            summary.threshold_fail++;
        } else if (reason == "unmatched_or_invalid_2d") {
            summary.unmatched_or_invalid_2d++;
        } else if (reason != "active") {
            summary.inactive_other++;
        }
    }
    return summary;
}

inline void PrintLineConstraintSummary(const std::string& tag,
                                       const LineConstraintSummary& summary,
                                       double threshold) {
    std::cout << "  [LineSummary][" << tag << "] total=" << summary.total
              << ", active=" << summary.active
              << ", behind=" << summary.behind
              << ", unmatched_type=" << summary.unmatched_type
              << ", threshold_fail=" << summary.threshold_fail
              << ", unmatched_or_invalid_2d=" << summary.unmatched_or_invalid_2d
              << ", inactive_other=" << summary.inactive_other
              << ", threshold_px=" << threshold;
    if (summary.min_dist_count > 0) {
        std::cout << ", avg_min_dist=" << (summary.min_dist_sum / summary.min_dist_count);
    } else {
        std::cout << ", avg_min_dist=NA";
    }
    std::cout << std::endl;
}

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
            p_rect_h << p_rect.x(), p_rect.y(), p_rect.z(), 1.0;
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
        p_rect_h << p_rect.x(), p_rect.y(), p_rect.z(), 1.0;
        Eigen::Matrix<T, 3, 1> uv = P_rect_.cast<T>() * p_rect_h;

        T u_f = uv.x() / uv.z();
        T v_f = uv.y() / uv.z();

        const double u_scalar = ScalarValue(u_f);
        const double v_scalar = ScalarValue(v_f);
        if (u_scalar < 0 || u_scalar >= W_ - 1 || v_scalar < 0 || v_scalar >= H_ - 1) {
            residual[0] = T(pt_.weight);
            return true;
        }

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

        residual[0] = edge_error * T(pt_.weight);
        return true;
    }

    PointFeature pt_;
    const cv::Mat* dist_map_;
    Eigen::Matrix3d R_rect_;
    Eigen::Matrix<double, 3, 4> P_rect_;
    int W_;
    int H_;
};
