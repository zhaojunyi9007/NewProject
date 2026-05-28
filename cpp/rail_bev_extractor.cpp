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

static float DistPointToSegment(cv::Point2f p, cv::Point2f a, cv::Point2f b) {
    const cv::Point2f ab = b - a;
    const float denom = ab.x * ab.x + ab.y * ab.y;
    if (denom <= 1e-9f) return std::sqrt((p.x - a.x) * (p.x - a.x) + (p.y - a.y) * (p.y - a.y));
    float t = ((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / denom;
    t = std::max(0.f, std::min(1.f, t));
    const cv::Point2f q = a + ab * t;
    return std::sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y));
}

static std::vector<float> MaskBBoxM(const cv::Mat& mask, const BEVChannels& bev) {
    int xmin = bev.nx, ymin = bev.ny, xmax = -1, ymax = -1;
    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            if (mask.at<uint8_t>(y, x) == 0) continue;
            xmin = std::min(xmin, x);
            ymin = std::min(ymin, y);
            xmax = std::max(xmax, x);
            ymax = std::max(ymax, y);
        }
    }
    if (xmax < xmin || ymax < ymin) return {};
    return {
        static_cast<float>(bev.xmin + static_cast<double>(xmin) * bev.resolution),
        static_cast<float>(bev.ymin + static_cast<double>(ymin) * bev.resolution),
        static_cast<float>(bev.xmin + static_cast<double>(xmax + 1) * bev.resolution),
        static_cast<float>(bev.ymin + static_cast<double>(ymax + 1) * bev.resolution),
    };
}

static float NonzeroRatio(const std::vector<float>& v) {
    if (v.empty()) return 0.f;
    int n = 0;
    for (float x : v) {
        if (x > 1e-4f) ++n;
    }
    return static_cast<float>(n) / static_cast<float>(v.size());
}

std::vector<Line3D> ExtractMultiRailLinesFromBEV(
    const BEVChannels& bev,
    const RailPriorConfig& cfg,
    double reference_plane_z,
    int max_lines,
    float* out_rail_confidence,
    bool* out_branch_detected,
    RailBEVDebug* out_debug,
    std::vector<float>* out_refined_rail_probability) {
    std::vector<Line3D> lines;
    const int ncell = bev.nx * bev.ny;
    if (out_refined_rail_probability) out_refined_rail_probability->assign(std::max(0, ncell), 0.f);
    if (out_rail_confidence) *out_rail_confidence = 0.f;
    if (out_branch_detected) *out_branch_detected = false;
    if (out_debug) *out_debug = RailBEVDebug{};
    if (bev.nx <= 0 || bev.ny <= 0 || bev.rail_probability.empty()) return lines;

    cv::Mat rail(bev.ny, bev.nx, CV_32F, const_cast<float*>(bev.rail_probability.data()));
    rail = rail.clone();
    if (cfg.smooth_ksize >= 3) {
        const int k = cfg.smooth_ksize | 1;
        cv::GaussianBlur(rail, rail, cv::Size(k, k), 0);
    }
    cv::Mat raw_mask;
    cv::threshold(rail, raw_mask, cfg.rail_prob_thresh, 255.0, cv::THRESH_BINARY);
    raw_mask.convertTo(raw_mask, CV_8U);

    cv::Mat mask = raw_mask.clone();
    if (cfg.morph_open_kernel_cells >= 3) {
        const int k = cfg.morph_open_kernel_cells | 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    }

    cv::Mat labels, stats, centroids;
    const int ncomp = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    cv::Mat filtered = cv::Mat::zeros(mask.size(), CV_8U);
    int candidate_count = 0;
    int rejected_blob_count = 0;
    int accepted_component_count = 0;

    for (int l = 1; l < ncomp; ++l) {
        const int area = stats.at<int>(l, cv::CC_STAT_AREA);
        if (static_cast<float>(area) < cfg.min_component_cells) continue;
        candidate_count++;
        std::vector<cv::Point2f> pts;
        pts.reserve(static_cast<size_t>(area));
        double prob_sum = 0.0;
        for (int y = 0; y < bev.ny; ++y) {
            for (int x = 0; x < bev.nx; ++x) {
                if (labels.at<int>(y, x) != l) continue;
                pts.emplace_back(
                    static_cast<float>(bev.xmin + (static_cast<double>(x) + 0.5) * bev.resolution),
                    static_cast<float>(bev.ymin + (static_cast<double>(y) + 0.5) * bev.resolution));
                prob_sum += rail.at<float>(y, x);
            }
        }
        if (pts.size() < 5) continue;
        cv::Mat data(static_cast<int>(pts.size()), 2, CV_32F);
        for (size_t i = 0; i < pts.size(); ++i) {
            data.at<float>(static_cast<int>(i), 0) = pts[i].x;
            data.at<float>(static_cast<int>(i), 1) = pts[i].y;
        }
        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
        const float l0 = std::max(0.f, pca.eigenvalues.at<float>(0, 0));
        const float l1 = std::max(0.f, pca.eigenvalues.at<float>(0, 1));
        cv::Point2f dir(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
        const float dlen = std::sqrt(dir.x * dir.x + dir.y * dir.y) + 1e-6f;
        dir.x /= dlen;
        dir.y /= dlen;
        const cv::Point2f mean(pca.mean.at<float>(0, 0), pca.mean.at<float>(0, 1));
        const cv::Point2f normal(-dir.y, dir.x);
        float tmin = 1e9f, tmax = -1e9f, nmin = 1e9f, nmax = -1e9f;
        for (const auto& p2 : pts) {
            const float t = (p2.x - mean.x) * dir.x + (p2.y - mean.y) * dir.y;
            const float q = (p2.x - mean.x) * normal.x + (p2.y - mean.y) * normal.y;
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
            nmin = std::min(nmin, q);
            nmax = std::max(nmax, q);
        }
        const float length_m = std::max(0.f, tmax - tmin);
        const float width_m = std::max(static_cast<float>(bev.resolution), nmax - nmin + static_cast<float>(bev.resolution));
        const float linearity = l0 / (l1 + 1e-6f);
        const float fill_ratio = static_cast<float>(area) / std::max(1.f, (length_m / static_cast<float>(bev.resolution)) * (width_m / static_cast<float>(bev.resolution)));
        const bool blob_like =
            width_m > cfg.blob_max_width_m ||
            (fill_ratio > cfg.blob_max_fill_ratio && linearity < cfg.blob_min_linearity) ||
            (area > cfg.blob_max_area_cells && linearity < cfg.min_linearity);
        if (blob_like) {
            rejected_blob_count++;
            continue;
        }
        accepted_component_count++;
        for (int y = 0; y < bev.ny; ++y) {
            for (int x = 0; x < bev.nx; ++x) {
                if (labels.at<int>(y, x) == l) filtered.at<uint8_t>(y, x) = 255;
            }
        }
        (void)prob_sum;
    }

    struct Segment {
        cv::Point2f a;
        cv::Point2f b;
        cv::Point2f mid;
        cv::Point2f dir;
        float length_m;
        float mean_prob;
        float tmin;
        float tmax;
    };
    std::vector<Segment> candidates;
    std::vector<cv::Vec4i> hough;
    const double min_len_px = std::max(1.0, static_cast<double>(cfg.hough_min_line_length_m) / bev.resolution);
    const double max_gap_px = std::max(1.0, static_cast<double>(cfg.hough_max_line_gap_m) / bev.resolution);
    const cv::Mat& hough_mask = accepted_component_count > 0 ? mask : filtered;
    cv::HoughLinesP(hough_mask, hough, 1.0, CV_PI / 180.0, cfg.hough_threshold, min_len_px, max_gap_px);

    auto sample_mean_prob = [&](cv::Point2f a, cv::Point2f b) -> float {
        const int ax = static_cast<int>(std::floor((a.x - bev.xmin) / bev.resolution));
        const int ay = static_cast<int>(std::floor((a.y - bev.ymin) / bev.resolution));
        const int bx = static_cast<int>(std::floor((b.x - bev.xmin) / bev.resolution));
        const int by = static_cast<int>(std::floor((b.y - bev.ymin) / bev.resolution));
        cv::LineIterator it(rail, cv::Point(ax, ay), cv::Point(bx, by), 8);
        double sum = 0.0;
        int cnt = 0;
        for (int i = 0; i < it.count; ++i, ++it) {
            const cv::Point p = it.pos();
            if (p.x < 0 || p.x >= rail.cols || p.y < 0 || p.y >= rail.rows) continue;
            sum += rail.at<float>(p.y, p.x);
            cnt++;
        }
        return cnt > 0 ? static_cast<float>(sum / static_cast<double>(cnt)) : 0.f;
    };

    for (const auto& v : hough) {
        cv::Point2f a(
            static_cast<float>(bev.xmin + (static_cast<double>(v[0]) + 0.5) * bev.resolution),
            static_cast<float>(bev.ymin + (static_cast<double>(v[1]) + 0.5) * bev.resolution));
        cv::Point2f b(
            static_cast<float>(bev.xmin + (static_cast<double>(v[2]) + 0.5) * bev.resolution),
            static_cast<float>(bev.ymin + (static_cast<double>(v[3]) + 0.5) * bev.resolution));
        cv::Point2f d = b - a;
        const float length_m = std::sqrt(d.x * d.x + d.y * d.y);
        if (length_m < cfg.min_length_m) continue;
        const float inv = 1.0f / std::max(1e-6f, length_m);
        d.x *= inv;
        d.y *= inv;
        const float mean_prob = sample_mean_prob(a, b);
        if (mean_prob < cfg.rail_prob_thresh) continue;
        const cv::Point2f mid = (a + b) * 0.5f;
        candidates.push_back({a, b, mid, d, length_m, mean_prob, -0.5f * length_m, 0.5f * length_m});
    }

    std::vector<bool> keep(candidates.size(), false);
    int pair_count = 0;
    float best_gauge = 0.f;
    float best_angle = 0.f;
    float best_overlap = 0.f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            cv::Point2f di = candidates[i].dir;
            cv::Point2f dj = candidates[j].dir;
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
            const cv::Point2f dm = candidates[j].mid - candidates[i].mid;
            const float gauge = std::abs(dm.x * normal.x + dm.y * normal.y);
            if (std::abs(gauge - cfg.gauge_m) > cfg.gauge_tolerance_m) continue;
            const float proj = dm.x * di.x + dm.y * di.y;
            const float jmin = candidates[j].tmin + proj;
            const float jmax = candidates[j].tmax + proj;
            const float overlap = std::max(0.f, std::min(candidates[i].tmax, jmax) - std::max(candidates[i].tmin, jmin));
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

    std::vector<float> refined(static_cast<size_t>(ncell), 0.f);
    double accepted_prob_sum = 0.0;
    int accepted_prob_count = 0;
    for (size_t i = 0; i < candidates.size() && static_cast<int>(lines.size()) < max_lines; ++i) {
        if (!keep[i]) continue;
        Line3D L;
        L.p1 = Eigen::Vector3d(static_cast<double>(candidates[i].a.x), static_cast<double>(candidates[i].a.y), reference_plane_z);
        L.p2 = Eigen::Vector3d(static_cast<double>(candidates[i].b.x), static_cast<double>(candidates[i].b.y), reference_plane_z);
        L.type = 0;
        L.class_id = SEM_RAIL_LIKE;
        L.confidence = std::min(1.0f, candidates[i].mean_prob);
        lines.push_back(L);
        accepted_prob_sum += candidates[i].mean_prob;
        accepted_prob_count++;
    }

    if (pair_count > 0 && !lines.empty()) {
        for (int y = 0; y < bev.ny; ++y) {
            for (int x = 0; x < bev.nx; ++x) {
                const cv::Point2f p(
                    static_cast<float>(bev.xmin + (static_cast<double>(x) + 0.5) * bev.resolution),
                    static_cast<float>(bev.ymin + (static_cast<double>(y) + 0.5) * bev.resolution));
                float best = 0.f;
                for (size_t i = 0; i < candidates.size(); ++i) {
                    if (!keep[i]) continue;
                    if (DistPointToSegment(p, candidates[i].a, candidates[i].b) <= cfg.accepted_line_band_m) {
                        best = std::max(best, rail.at<float>(y, x));
                    }
                }
                refined[static_cast<size_t>(y * bev.nx + x)] = best;
            }
        }
    }
    if (out_refined_rail_probability) *out_refined_rail_probability = refined;

    if (out_rail_confidence) {
        *out_rail_confidence = pair_count > 0
            ? std::min(1.f, best_overlap / std::max(1.f, cfg.min_pair_overlap_m * 2.f))
            : 0.f;
    }
    if (out_branch_detected) {
        *out_branch_detected = pair_count > 1 || lines.size() > 2;
    }
    if (out_debug) {
        out_debug->candidate_component_count = candidate_count;
        out_debug->rejected_blob_component_count = rejected_blob_count;
        out_debug->line_candidate_count = static_cast<int>(hough.size());
        out_debug->line_component_count = static_cast<int>(candidates.size());
        out_debug->pair_count = pair_count;
        out_debug->best_pair_gauge_m = best_gauge;
        out_debug->best_pair_angle_deg = best_angle;
        out_debug->best_pair_overlap_m = best_overlap;
        out_debug->refined_nonzero_ratio = NonzeroRatio(refined);
        out_debug->raw_bbox_m = MaskBBoxM(raw_mask, bev);
        cv::Mat refined_mask = cv::Mat::zeros(bev.ny, bev.nx, CV_8U);
        for (int y = 0; y < bev.ny; ++y) {
            for (int x = 0; x < bev.nx; ++x) {
                if (refined[static_cast<size_t>(y * bev.nx + x)] > 1e-4f) refined_mask.at<uint8_t>(y, x) = 255;
            }
        }
        out_debug->refined_bbox_m = MaskBBoxM(refined_mask, bev);
        out_debug->mean_probability_on_accepted_lines = accepted_prob_count > 0
            ? static_cast<float>(accepted_prob_sum / static_cast<double>(accepted_prob_count))
            : 0.f;
    }
    std::cout << "[RailBEV] Multi-segment rail lines: " << lines.size();
    if (out_rail_confidence) std::cout << ", rail_confidence=" << *out_rail_confidence;
    if (out_branch_detected) std::cout << ", branch_detected=" << (*out_branch_detected ? 1 : 0);
    std::cout << ", line_candidates=" << candidates.size() << ", rail_pairs=" << pair_count << std::endl;
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
