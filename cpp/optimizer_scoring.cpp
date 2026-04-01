#include "include/optimizer_scoring.h"

#include "include/optimizer_image_utils.h"

#include <algorithm>
#include <cmath>

bool Project(const Eigen::Vector3d& p_lidar,
             const Eigen::Matrix3d& R_rect,
             const Eigen::Matrix<double, 3, 4>& P_rect,
             const Eigen::Matrix3d& R,
             const Eigen::Vector3d& t,
             int& u, int& v, int W, int H) {
    Eigen::Vector3d p_cam = R * p_lidar + t;

    Eigen::Vector3d p_rect = R_rect * p_cam;
    if (p_rect.z() < 0.1) return false;
    Eigen::Vector4d p_rect_h;
    p_rect_h << p_rect.x(), p_rect.y(), p_rect.z(), 1.0;
    Eigen::Vector3d uv = P_rect * p_rect_h;
    u = static_cast<int>(uv.x() / uv.z());
    v = static_cast<int>(uv.y() / uv.z());
    return (u >= 0 && u < W && v >= 0 && v < H);
}

ProjectionDebugStats CountProjectionStats(const std::vector<PointFeature>& points,
                                          const Eigen::Matrix3d& R_rect,
                                          const Eigen::Matrix<double, 3, 4>& P_rect,
                                          const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& t,
                                          int W, int H) {
    ProjectionDebugStats stats;
    for (const auto& pt : points) {
        stats.total++;
        if (pt.label == 0) {
            stats.unlabeled++;
        } else {
            stats.labeled++;
        }
        Eigen::Vector3d p_cam = R * pt.p + t;
        Eigen::Vector3d p_rect = R_rect * p_cam;
        if (p_rect.z() < 0.1) {
            stats.behind++;
            continue;
        }
        Eigen::Vector4d p_rect_h;
        p_rect_h << p_rect.x(), p_rect.y(), p_rect.z(), 1.0;
        Eigen::Vector3d uv = P_rect * p_rect_h;
        int u = static_cast<int>(uv.x() / uv.z());
        int v = static_cast<int>(uv.y() / uv.z());
        if (u >= 0 && u < W && v >= 0 && v < H) {
            stats.in_bounds++;
        } else {
            stats.out_of_bounds++;
        }
    }
    return stats;
}

std::vector<PointFeature> FilterPointsInView(const std::vector<PointFeature>& points,
                                             const Eigen::Matrix3d& R_rect,
                                             const Eigen::Matrix<double, 3, 4>& P_rect,
                                             const Eigen::Matrix3d& R,
                                             const Eigen::Vector3d& t,
                                             int W, int H,
                                             int margin) {
    std::vector<PointFeature> filtered;
    filtered.reserve(points.size());
    for (const auto& pt : points) {
        Eigen::Vector3d p_cam = R * pt.p + t;
        Eigen::Vector3d p_rect = R_rect * p_cam;
        if (p_rect.z() < 0.1) {
            continue;
        }
        Eigen::Vector4d p_rect_h;
        p_rect_h << p_rect.x(), p_rect.y(), p_rect.z(), 1.0;
        Eigen::Vector3d uv = P_rect * p_rect_h;
        int u = static_cast<int>(uv.x() / uv.z());
        int v = static_cast<int>(uv.y() / uv.z());
        if (u >= -margin && u < W + margin && v >= -margin && v < H + margin) {
            filtered.push_back(pt);
        }
    }
    return filtered;
}

std::vector<LabelStats> ComputeLabelStats(const std::vector<PointFeature>& points,
                                          const cv::Mat& semantic_map,
                                          const Eigen::Matrix3d& R_rect,
                                          const Eigen::Matrix<double, 3, 4>& P_rect,
                                          int W, int H,
                                          const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& t) {
    int max_label = GetMaxLabel(semantic_map);
    if (max_label <= 0) return std::vector<LabelStats>();

    std::vector<double> sum_intensity(max_label + 1, 0.0);
    std::vector<double> sum_intensity_sq(max_label + 1, 0.0);
    std::vector<Eigen::Vector3d> sum_normal(max_label + 1, Eigen::Vector3d::Zero());
    std::vector<int> counts(max_label + 1, 0);

    for (const auto& pt : points) {
        int u, v;
        if (!Project(pt.p, R_rect, P_rect, R, t, u, v, W, H)) continue;

        int img_label = GetSemanticLabel(semantic_map, u, v);
        if (img_label <= 0 || img_label > max_label) continue;
        sum_intensity[img_label] += pt.intensity;
        sum_intensity_sq[img_label] += pt.intensity * pt.intensity;
        Eigen::Vector3d n = pt.normal;
        if (n.norm() > 1e-6) {
            n.normalize();
        }

        sum_normal[img_label] += n;
        counts[img_label] += 1;
    }

    std::vector<LabelStats> stats(max_label + 1);
    for (int i = 1; i <= max_label; ++i) {
        if (counts[i] == 0) continue;
        stats[i].count = counts[i];
        stats[i].mean_intensity = sum_intensity[i] / counts[i];
        double mean_sq = sum_intensity_sq[i] / counts[i];
        stats[i].var_intensity = std::max(0.0, mean_sq - stats[i].mean_intensity * stats[i].mean_intensity);
        Eigen::Vector3d mean_n = sum_normal[i];
        if (mean_n.norm() > 1e-6) {
            mean_n.normalize();
        }
        stats[i].mean_normal = mean_n;
    }

    return stats;
}

double MaskIntensityVarianceScore(const std::vector<PointFeature>& points,
                                  const cv::Mat& semantic_map,
                                  const Eigen::Matrix3d& R_rect,
                                  const Eigen::Matrix<double, 3, 4>& P_rect,
                                  int W, int H,
                                  const Eigen::Matrix3d& R,
                                  const Eigen::Vector3d& t) {
    if (semantic_map.empty()) return -1e6;

    int max_label = GetMaxLabel(semantic_map);
    if (max_label <= 0) return -1e6;

    std::vector<double> sum_intensity(max_label + 1, 0.0);
    std::vector<double> sum_intensity_sq(max_label + 1, 0.0);
    std::vector<int> counts(max_label + 1, 0);

    for (const auto& pt : points) {
        int u, v;
        if (!Project(pt.p, R_rect, P_rect, R, t, u, v, W, H)) continue;

        int img_label = GetSemanticLabel(semantic_map, u, v);

        if (img_label <= 0 || img_label > max_label) continue;
        sum_intensity[img_label] += pt.intensity;
        sum_intensity_sq[img_label] += pt.intensity * pt.intensity;
        counts[img_label] += 1;
    }

    double total_variance = 0.0;
    int valid_labels = 0;
    for (int i = 1; i <= max_label; ++i) {
        if (counts[i] < 10) continue;
        double mean = sum_intensity[i] / counts[i];
        double mean_sq = sum_intensity_sq[i] / counts[i];
        double variance = std::max(0.0, mean_sq - mean * mean);
        total_variance += variance;
        valid_labels++;
    }

    if (valid_labels == 0) return -1e6;
    return -total_variance;
}

double EdgeAttractionScore(const std::vector<PointFeature>& points,
                           const cv::Mat& dist_map,
                           const cv::Mat& weight_map,
                           const Eigen::Matrix3d& R_rect,
                           const Eigen::Matrix<double, 3, 4>& P_rect,
                           int W, int H,
                           const Eigen::Matrix3d& R,
                           const Eigen::Vector3d& t) {
    if (dist_map.empty()) return -1e6;
    double total_score = 0.0;
    int visible_count = 0;

    for (const auto& pt : points) {
        int u, v;

        if (!Project(pt.p, R_rect, P_rect, R, t, u, v, W, H)) {
            continue;
        }

        float dist_value = GetDistanceValue(dist_map, u, v);
        float edge_weight = 1.0f;
        if (!weight_map.empty()) {
            edge_weight = GetDistanceValue(weight_map, u, v);
            if (edge_weight < 0.05f) edge_weight = 1.0f;
        }

        dist_value = std::min(std::max(dist_value, 0.0f), 1.0f);
        edge_weight = std::min(std::max(edge_weight, 0.0f), 1.0f);

        double score = (1.0f - dist_value) * edge_weight * pt.weight;
        total_score += score;
        visible_count++;
    }

    // When fewer than 50 points project in-image the edge score is too flat to
    // distinguish poses.  Fall back to a score that simply prefers candidates
    // that get more points into the image, so the coarse search still drives
    // toward better alignment rather than returning an identical sentinel for
    // every candidate.
    if (visible_count < 50) return -1e5 + visible_count;
    return total_score;
}
