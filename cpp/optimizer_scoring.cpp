#include "include/optimizer_scoring.h"

#include "include/optimizer_image_utils.h"

#include <algorithm>
#include <cmath>

namespace {
double ComputeLineAlignmentScoreWeighted(const std::vector<Line3D>& lines3d,
                                         const std::vector<Line2D>& lines2d,
                                         const Eigen::Matrix3d& R_rect,
                                         const Eigen::Matrix<double, 3, 4>& P_rect,
                                         const Eigen::Matrix3d& R,
                                         const Eigen::Vector3d& t) {
    if (lines3d.empty() || lines2d.empty()) return 0.0;
    constexpr double kMinZ = 0.1;
    constexpr double kScalePx = 40.0;
    double score_sum = 0.0;
    int used = 0;

    for (const auto& l3d : lines3d) {
        Eigen::Vector3d p1_cam = R * l3d.p1 + t;
        Eigen::Vector3d p2_cam = R * l3d.p2 + t;
        Eigen::Vector3d p1_rect = R_rect * p1_cam;
        Eigen::Vector3d p2_rect = R_rect * p2_cam;
        if (p1_rect.z() < kMinZ || p2_rect.z() < kMinZ) continue;

        Eigen::Vector4d p1_h(p1_rect.x(), p1_rect.y(), p1_rect.z(), 1.0);
        Eigen::Vector4d p2_h(p2_rect.x(), p2_rect.y(), p2_rect.z(), 1.0);
        Eigen::Vector3d uv1 = P_rect * p1_h;
        Eigen::Vector3d uv2 = P_rect * p2_h;
        const double u1 = uv1.x() / uv1.z();
        const double v1 = uv1.y() / uv1.z();
        const double u2 = uv2.x() / uv2.z();
        const double v2 = uv2.y() / uv2.z();

        double best = -1.0;
        for (const auto& l2d : lines2d) {
            if (l2d.type != l3d.type) continue;
            if (l3d.class_id > 0 && l2d.class_id >= 0 && l2d.class_id != l3d.class_id) continue;

            const double lx1 = l2d.p1.x(), ly1 = l2d.p1.y();
            const double lx2 = l2d.p2.x(), ly2 = l2d.p2.y();
            const double A = ly1 - ly2;
            const double B = lx2 - lx1;
            const double C = lx1 * ly2 - lx2 * ly1;
            const double norm = std::sqrt(A * A + B * B);
            if (norm < 1e-6) continue;

            const double d1 = std::abs(A * u1 + B * v1 + C) / norm;
            const double d2 = std::abs(A * u2 + B * v2 + C) / norm;
            const double avg_dist = 0.5 * (d1 + d2);

            const double l3_conf = std::max(0.05f, std::min(1.0f, l3d.confidence));
            const double l2_sem = std::max(0.0, std::min(1.0, l2d.semantic_support));
            const double l2_conf = std::max(0.05f, std::min(1.0f, l2d.confidence));
            const double pair_w = std::max(0.05, l3_conf * (0.5 * l2_sem + 0.5 * l2_conf));
            const double sim = std::exp(-avg_dist / kScalePx) * pair_w;  // [0,1]
            if (sim > best) best = sim;
        }
        if (best >= 0.0) {
            score_sum += best;
            used++;
        }
    }
    if (used <= 0) return 0.0;
    return score_sum / static_cast<double>(used);
}
}  // namespace

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
            if (!std::isfinite(edge_weight)) {
                edge_weight = 0.0f;
            }
            edge_weight = std::min(std::max(edge_weight, 0.0f), 1.0f);

            // 低权重区域直接不给分，不要恢复成 1.0
            if (edge_weight <= 1e-4f) {
                continue;
            }
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

double ComputeTotalCalibrationScoreSemanticDominant(const std::vector<PointFeature>& edge_points,
                                                    const cv::Mat& edge_dist,
                                                    const cv::Mat& edge_weight,
                                                    const std::vector<Line3D>& lines3d,
                                                    const std::vector<Line2D>& lines2d,
                                                    const std::vector<SemanticPointRecord>& lidar_semantic_points,
                                                    const SemanticProbMaps& image_semantic_probs,
                                                    const Eigen::Matrix3d& R_rect,
                                                    const Eigen::Matrix<double, 3, 4>& P_rect,
                                                    int W,
                                                    int H,
                                                    const Eigen::Matrix3d& R,
                                                    const Eigen::Vector3d& t,
                                                    double w_semantic_js,
                                                    double w_semantic_hist,
                                                    double w_edge,
                                                    double w_line,
                                                    const SemanticScoringConfig& sem_cfg,
                                                    TotalScoreBreakdown* breakdown) {
    TotalScoreBreakdown bd;

    // Semantic terms (dominant).
    SemanticScoreBreakdown sem_bd;
    bd.semantic_js_divergence =
        ComputeSemanticJSDivergence(lidar_semantic_points, image_semantic_probs, R_rect, P_rect, R, t, sem_cfg, &sem_bd);
    bd.semantic_hist_similarity = 
        ComputeSemanticHistogramConsistency(lidar_semantic_points, image_semantic_probs, R_rect, P_rect, R, t, sem_cfg, &sem_bd);

    bd.semantic_js_score = -bd.semantic_js_divergence;
    bd.semantic_hist_score = bd.semantic_hist_similarity;

    // Edge term: normalize by point count for scale stability.
    bd.edge_score_norm = 0.0;
    if (!edge_dist.empty() && !edge_points.empty()) {
        const double raw = EdgeAttractionScore(edge_points, edge_dist, edge_weight, R_rect, P_rect, W, H, R, t);
        const double n = static_cast<double>(std::max<size_t>(1, edge_points.size()));
        bd.edge_score_norm = raw / n;
    }

    // Phase C5: weighted line alignment score with semantic/class support.
    bd.line_score_norm = ComputeLineAlignmentScoreWeighted(lines3d, lines2d, R_rect, P_rect, R, t);

    bd.edge_score = w_edge * bd.edge_score_norm;
    bd.line_score = w_line * bd.line_score_norm;

    bd.total_score = w_semantic_js * bd.semantic_js_score + w_semantic_hist * bd.semantic_hist_score + bd.edge_score + bd.line_score;

    if (breakdown) {
        *breakdown = bd;
    }
    return bd.total_score;
}
