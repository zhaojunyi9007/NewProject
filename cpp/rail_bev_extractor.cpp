#include "include/rail_bev_extractor.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

RailBEVResult ExtractRailStructureFromBEV(const BEVChannels& bev, const RailPriorConfig& cfg) {
    RailBEVResult out;
    if (bev.nx <= 0 || bev.ny <= 0 || bev.rail_probability.empty()) {
        return out;
    }

    cv::Mat rail(bev.ny, bev.nx, CV_32F, const_cast<float*>(bev.rail_probability.data()));
    rail = rail.clone();
    if (cfg.smooth_ksize >= 3) {
        int k = cfg.smooth_ksize | 1;
        cv::GaussianBlur(rail, rail, cv::Size(k, k), 0);
    }

    cv::Mat mask;
    cv::threshold(rail, mask, cfg.rail_prob_thresh, 255.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);

    cv::Mat labels, stats, centroids;
    const int ncomp = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);

    float best_area = 0.f;
    int best_lbl = -1;
    for (int l = 1; l < ncomp; ++l) {
        const int a = stats.at<int>(l, cv::CC_STAT_AREA);
        if (static_cast<float>(a) >= cfg.min_component_cells && static_cast<float>(a) > best_area) {
            best_area = static_cast<float>(a);
            best_lbl = l;
        }
    }

    if (best_lbl < 0) {
        std::cout << "[RailBEV] No salient rail component\n";
        return out;
    }

    std::vector<cv::Point2f> pts2d;
    for (int y = 0; y < bev.ny; ++y) {
        for (int x = 0; x < bev.nx; ++x) {
            if (labels.at<int>(y, x) == best_lbl) {
                const float wx =
                    static_cast<float>(bev.xmin + (static_cast<double>(x) + 0.5) * bev.resolution);
                const float wy =
                    static_cast<float>(bev.ymin + (static_cast<double>(y) + 0.5) * bev.resolution);
                pts2d.emplace_back(wx, wy);
            }
        }
    }

    if (pts2d.size() < 5) {
        return out;
    }

    cv::Mat data(static_cast<int>(pts2d.size()), 2, CV_32F);
    for (size_t i = 0; i < pts2d.size(); ++i) {
        data.at<float>(static_cast<int>(i), 0) = pts2d[i].x;
        data.at<float>(static_cast<int>(i), 1) = pts2d[i].y;
    }
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
    cv::Point2f mean(pca.mean.at<float>(0, 0), pca.mean.at<float>(0, 1));
    cv::Point2f ev(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
    const float len = std::sqrt(ev.x * ev.x + ev.y * ev.y) + 1e-6f;
    ev.x /= len;
    ev.y /= len;

    float tmin = 1e9f, tmax = -1e9f;
    for (const auto& p : pts2d) {
        const float t = (p.x - mean.x) * ev.x + (p.y - mean.y) * ev.y;
        tmin = std::min(tmin, t);
        tmax = std::max(tmax, t);
    }
    const float ext = std::max(5.f, (tmax - tmin) * 0.5f);
    cv::Point2f c0 = mean + ev * (tmin - 0.1f * ext);
    cv::Point2f c1 = mean + ev * (tmax + 0.1f * ext);
    out.centerline_pts.push_back(c0);
    out.centerline_pts.push_back(c1);

    cv::Point2f n(-ev.y, ev.x);
    const float half_w = std::max(0.5f, 0.08f * (tmax - tmin + 1.f));
    out.left_rail_pts.push_back(c0 + n * half_w);
    out.left_rail_pts.push_back(c1 + n * half_w);
    out.right_rail_pts.push_back(c0 - n * half_w);
    out.right_rail_pts.push_back(c1 - n * half_w);

    out.confidence = std::min(1.f, best_area / (cfg.min_component_cells * 4.f));
    std::cout << "[RailBEV] Extracted rail centerline, confidence=" << out.confidence << std::endl;
    return out;
}

std::vector<Line3D> ExtractMultiRailLinesFromBEV(
    const BEVChannels& bev,
    const RailPriorConfig& cfg,
    double reference_plane_z,
    int max_lines,
    float* out_rail_confidence,
    bool* out_branch_detected,
    RailBEVDebug* out_debug) {
    std::vector<Line3D> lines;
    if (bev.nx <= 0 || bev.ny <= 0 || bev.rail_probability.empty()) return lines;

    cv::Mat rail(bev.ny, bev.nx, CV_32F, const_cast<float*>(bev.rail_probability.data()));
    rail = rail.clone();
    if (cfg.smooth_ksize >= 3) {
        int k = cfg.smooth_ksize | 1;
        cv::GaussianBlur(rail, rail, cv::Size(k, k), 0);
    }
    cv::Mat mask;
    cv::threshold(rail, mask, cfg.rail_prob_thresh, 255.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);

    cv::Mat labels, stats, centroids;
    const int ncomp = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);

    struct Comp {
        int id;
        float area;
        float mean_prob;
        cv::Point2f mean;
        cv::Point2f dir;
        float tmin;
        float tmax;
        float nmin;
        float nmax;
        float length_m;
        float width_m;
        float linearity;
    };
    std::vector<Comp> line_comps;
    int candidate_count = 0;
    for (int l = 1; l < ncomp; ++l) {
        const int a = stats.at<int>(l, cv::CC_STAT_AREA);
        if (static_cast<float>(a) < cfg.min_component_cells) continue;
        candidate_count++;
        std::vector<cv::Point2f> pts2d;
        double prob_sum = 0.0;
        for (int y = 0; y < bev.ny; ++y) {
            for (int x = 0; x < bev.nx; ++x) {
                if (labels.at<int>(y, x) == l) {
                    const float wx =
                        static_cast<float>(bev.xmin + (static_cast<double>(x) + 0.5) * bev.resolution);
                    const float wy =
                        static_cast<float>(bev.ymin + (static_cast<double>(y) + 0.5) * bev.resolution);
                    pts2d.emplace_back(wx, wy);
                    prob_sum += static_cast<double>(rail.at<float>(y, x));
                }
            }
        }
        if (pts2d.size() < 5) continue;

        cv::Mat data(static_cast<int>(pts2d.size()), 2, CV_32F);
        for (size_t i = 0; i < pts2d.size(); ++i) {
            data.at<float>(static_cast<int>(i), 0) = pts2d[i].x;
            data.at<float>(static_cast<int>(i), 1) = pts2d[i].y;
        }
        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
        cv::Point2f mean(pca.mean.at<float>(0, 0), pca.mean.at<float>(0, 1));
        cv::Point2f ev(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
        const float l0 = std::max(0.f, pca.eigenvalues.at<float>(0, 0));
        const float l1 = std::max(0.f, pca.eigenvalues.at<float>(0, 1));
        const float len = std::sqrt(ev.x * ev.x + ev.y * ev.y) + 1e-6f;
        ev.x /= len;
        ev.y /= len;
        float tmin = 1e9f, tmax = -1e9f;
        float nmin = 1e9f, nmax = -1e9f;
        const cv::Point2f normal(-ev.y, ev.x);
        for (const auto& p : pts2d) {
            const float t = (p.x - mean.x) * ev.x + (p.y - mean.y) * ev.y;
            const float q = (p.x - mean.x) * normal.x + (p.y - mean.y) * normal.y;
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
            nmin = std::min(nmin, q);
            nmax = std::max(nmax, q);
        }
        const float length_m = std::max(0.f, tmax - tmin);
        const float width_m = std::max(static_cast<float>(bev.resolution), nmax - nmin + static_cast<float>(bev.resolution));
        const float linearity = l0 / (l1 + 1e-6f);
        const float mean_prob = static_cast<float>(prob_sum / std::max<size_t>(1, pts2d.size()));
        if (length_m < cfg.min_length_m || width_m > cfg.max_width_m ||
            linearity < cfg.min_linearity || mean_prob < cfg.rail_prob_thresh) {
            continue;
        }
        line_comps.push_back({l, static_cast<float>(a), mean_prob, mean, ev, tmin, tmax, nmin, nmax, length_m, width_m, linearity});
    }

    std::vector<bool> keep(line_comps.size(), false);
    int pair_count = 0;
    float best_gauge = 0.f;
    float best_angle = 0.f;
    float best_overlap = 0.f;
    for (size_t i = 0; i < line_comps.size(); ++i) {
        for (size_t j = i + 1; j < line_comps.size(); ++j) {
            cv::Point2f di = line_comps[i].dir;
            cv::Point2f dj = line_comps[j].dir;
            float dot = di.x * dj.x + di.y * dj.y;
            if (dot < 0.f) {
                dot = -dot;
                dj.x = -dj.x;
                dj.y = -dj.y;
            }
            dot = std::max(-1.f, std::min(1.f, dot));
            const float angle_deg = std::acos(dot) * 180.0f / static_cast<float>(CV_PI);
            if (angle_deg > cfg.max_parallel_angle_deg) continue;
            const cv::Point2f normal(-di.y, di.x);
            const cv::Point2f dm = line_comps[j].mean - line_comps[i].mean;
            const float gauge = std::abs(dm.x * normal.x + dm.y * normal.y);
            if (std::abs(gauge - cfg.gauge_m) > cfg.gauge_tolerance_m) continue;
            const float mj = dm.x * di.x + dm.y * di.y;
            const float jtmin = line_comps[j].tmin + mj;
            const float jtmax = line_comps[j].tmax + mj;
            const float overlap = std::max(0.f, std::min(line_comps[i].tmax, jtmax) - std::max(line_comps[i].tmin, jtmin));
            if (overlap < cfg.min_pair_overlap_m) continue;
            keep[i] = true;
            keep[j] = true;
            pair_count++;
            if (overlap > best_overlap) {
                best_overlap = overlap;
                best_gauge = gauge;
                best_angle = angle_deg;
            }
        }
    }

    for (size_t ci = 0; ci < line_comps.size() && static_cast<int>(lines.size()) < max_lines; ++ci) {
        if (!keep[ci]) continue;
        const float ext = std::max(1.f, line_comps[ci].length_m * 0.03f);
        cv::Point2f c0 = line_comps[ci].mean + line_comps[ci].dir * (line_comps[ci].tmin - ext);
        cv::Point2f c1 = line_comps[ci].mean + line_comps[ci].dir * (line_comps[ci].tmax + ext);
        Line3D L;
        L.p1 = Eigen::Vector3d(static_cast<double>(c0.x), static_cast<double>(c0.y), reference_plane_z);
        L.p2 = Eigen::Vector3d(static_cast<double>(c1.x), static_cast<double>(c1.y), reference_plane_z);
        L.type = 0;
        L.class_id = SEM_RAIL_LIKE;
        L.confidence = std::min(1.0f, line_comps[ci].mean_prob);
        lines.push_back(L);
    }

    if (out_rail_confidence) {
        *out_rail_confidence = pair_count > 0
            ? std::min(1.f, best_overlap / std::max(1.f, cfg.min_pair_overlap_m * 2.f))
            : 0.f;
    }
    if (out_branch_detected) {
        *out_branch_detected = pair_count > 1 || line_comps.size() > 2;
    }
    if (out_debug) {
        out_debug->candidate_component_count = candidate_count;
        out_debug->line_component_count = static_cast<int>(line_comps.size());
        out_debug->pair_count = pair_count;
        out_debug->best_pair_gauge_m = best_gauge;
        out_debug->best_pair_angle_deg = best_angle;
        out_debug->best_pair_overlap_m = best_overlap;
    }
    std::cout << "[RailBEV] Multi-segment rail lines: " << lines.size();
    if (out_rail_confidence) {
        std::cout << ", rail_confidence=" << *out_rail_confidence;
    }
    if (out_branch_detected) {
        std::cout << ", branch_detected=" << (*out_branch_detected ? 1 : 0);
    }
    std::cout << ", line_components=" << line_comps.size() << ", rail_pairs=" << pair_count;
    std::cout << std::endl;
    return lines;
}

std::vector<Line3D> RailBEVToLine3D(const BEVChannels& bev, const RailBEVResult& rail, double reference_plane_z) {
    (void)bev;
    std::vector<Line3D> lines;
    if (rail.centerline_pts.size() >= 2) {
        const auto& a = rail.centerline_pts.front();
        const auto& b = rail.centerline_pts.back();
        Line3D L;
        L.p1 = Eigen::Vector3d(static_cast<double>(a.x), static_cast<double>(a.y), reference_plane_z);
        L.p2 = Eigen::Vector3d(static_cast<double>(b.x), static_cast<double>(b.y), reference_plane_z);
        L.type = 0;
        L.class_id = SEM_RAIL_LIKE;
        L.confidence = std::max(0.05f, rail.confidence);
        lines.push_back(L);
    }
    return lines;
}
