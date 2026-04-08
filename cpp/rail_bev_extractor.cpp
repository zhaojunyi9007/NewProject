#include "include/rail_bev_extractor.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>

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
    bool* out_branch_detected) {
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
    };
    std::vector<Comp> comps;
    float total_area = 0.f;
    for (int l = 1; l < ncomp; ++l) {
        const int a = stats.at<int>(l, cv::CC_STAT_AREA);
        if (static_cast<float>(a) >= cfg.min_component_cells) {
            const float af = static_cast<float>(a);
            comps.push_back({l, af});
            total_area += af;
        }
    }
    std::sort(comps.begin(), comps.end(), [](const Comp& A, const Comp& B) { return A.area > B.area; });

    // Phase C4: simple confidence + branch detection.
    // - confidence: total rail component area normalized by a soft cap.
    // - branch_detected: at least two sizable components with comparable areas.
    if (out_rail_confidence) {
        const float denom = std::max(1.f, cfg.min_component_cells * 8.f);
        *out_rail_confidence = std::min(1.f, total_area / denom);
    }
    if (out_branch_detected) {
        bool branch = false;
        if (comps.size() >= 2) {
            const float a0 = comps[0].area;
            const float a1 = comps[1].area;
            if (a0 > 1e-3f) {
                const float ratio = a1 / a0;
                // If the 2nd component is not negligible, it often indicates a split / fork.
                branch = (ratio >= 0.35f);
            }
        }
        *out_branch_detected = branch;
    }

    for (size_t ci = 0; ci < comps.size() && static_cast<int>(lines.size()) < max_lines; ++ci) {
        const int lbl = comps[ci].id;
        std::vector<cv::Point2f> pts2d;
        for (int y = 0; y < bev.ny; ++y) {
            for (int x = 0; x < bev.nx; ++x) {
                if (labels.at<int>(y, x) == lbl) {
                    const float wx =
                        static_cast<float>(bev.xmin + (static_cast<double>(x) + 0.5) * bev.resolution);
                    const float wy =
                        static_cast<float>(bev.ymin + (static_cast<double>(y) + 0.5) * bev.resolution);
                    pts2d.emplace_back(wx, wy);
                }
            }
        }
        if (pts2d.size() < 5) continue;

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
        const float ext = std::max(2.f, (tmax - tmin) * 0.05f);
        cv::Point2f c0 = mean + ev * (tmin - ext);
        cv::Point2f c1 = mean + ev * (tmax + ext);
        Line3D L;
        L.p1 = Eigen::Vector3d(static_cast<double>(c0.x), static_cast<double>(c0.y), reference_plane_z);
        L.p2 = Eigen::Vector3d(static_cast<double>(c1.x), static_cast<double>(c1.y), reference_plane_z);
        L.type = 0;
        lines.push_back(L);
    }
    std::cout << "[RailBEV] Multi-segment rail lines: " << lines.size();
    if (out_rail_confidence) {
        std::cout << ", rail_confidence=" << *out_rail_confidence;
    }
    if (out_branch_detected) {
        std::cout << ", branch_detected=" << (*out_branch_detected ? 1 : 0);
    }
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
        lines.push_back(L);
    }
    return lines;
}
