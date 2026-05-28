#include "include/bev_builder.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <cstdlib>

int BEVGridSpec::nx() const {
    if (resolution <= 1e-9) return 0;
    return std::max(1, static_cast<int>(std::ceil((xmax - xmin) / resolution)));
}

int BEVGridSpec::ny() const {
    if (resolution <= 1e-9) return 0;
    return std::max(1, static_cast<int>(std::ceil((ymax - ymin) / resolution)));
}

static int CellIndex(int ix, int iy, int nx) {
    return iy * nx + ix;
}

static float Clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

static double GetEnvDoubleLocal(const char* key, double def) {
    const char* v = std::getenv(key);
    if (!v) return def;
    try { return std::stod(v); } catch (...) { return def; }
}

static int GetEnvIntLocal(const char* key, int def) {
    const char* v = std::getenv(key);
    if (!v) return def;
    try { return std::stoi(v); } catch (...) { return def; }
}

static float Percentile(std::vector<float> values, double q) {
    if (values.empty()) return std::numeric_limits<float>::quiet_NaN();
    q = std::max(0.0, std::min(1.0, q));
    const size_t idx = static_cast<size_t>(std::floor(q * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(idx), values.end());
    return values[idx];
}

static float TriangularScore(float v, float lo, float peak_lo, float peak_hi, float hi) {
    if (v < lo || v > hi) return 0.f;
    if (v >= peak_lo && v <= peak_hi) return 1.f;
    if (v < peak_lo) return Clamp01((v - lo) / std::max(1e-6f, peak_lo - lo));
    return Clamp01((hi - v) / std::max(1e-6f, hi - peak_hi));
}


bool BuildLidarBEV(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals,
    const BEVGridSpec& spec,
    double rail_band_zmin,
    double rail_band_zmax,
    BEVChannels* out) {
    if (!cloud || !out || !normals || cloud->size() != normals->size()) {
        std::cerr << "[BEV] Invalid input\n";
        return false;
    }

    const int nx = spec.nx();
    const int ny = spec.ny();
    if (nx <= 0 || ny <= 0) {
        std::cerr << "[BEV] Invalid grid size\n";
        return false;
    }

    const int local_ground_window = std::max(1, GetEnvIntLocal("EDGECALIB_LIDAR_RAIL_LOCAL_GROUND_WINDOW_CELLS", 4));
    const double ground_percentile = GetEnvDoubleLocal("EDGECALIB_LIDAR_RAIL_GROUND_PERCENTILE", 0.10);
    const float residual_min = static_cast<float>(GetEnvDoubleLocal("EDGECALIB_LIDAR_RAIL_RESIDUAL_MIN_M", 0.03));
    const float residual_max = static_cast<float>(GetEnvDoubleLocal("EDGECALIB_LIDAR_RAIL_RESIDUAL_MAX_M", 0.30));
    const float max_candidate_height_range = static_cast<float>(GetEnvDoubleLocal("EDGECALIB_LIDAR_RAIL_MAX_CANDIDATE_HEIGHT_RANGE_M", 0.55));
    const int min_candidate_density = std::max(1, GetEnvIntLocal("EDGECALIB_LIDAR_RAIL_MIN_CANDIDATE_DENSITY", 1));
    const int linearity_window = std::max(1, GetEnvIntLocal("EDGECALIB_LIDAR_RAIL_LINEARITY_WINDOW_CELLS", 3));

    const int ncells = nx * ny;
    std::vector<int> count(ncells, 0);
    std::vector<double> sum_z(ncells, 0.0);
    std::vector<double> sum_i(ncells, 0.0);
    std::vector<double> sum_nz(ncells, 0.0);
    std::vector<double> sum_side_normal(ncells, 0.0);
    std::vector<float> min_z(ncells, std::numeric_limits<float>::infinity());
    std::vector<float> max_z(ncells, -std::numeric_limits<float>::infinity());

    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& p = cloud->points[i];
        const auto& n = normals->points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        const int ix = static_cast<int>(std::floor((p.x - spec.xmin) / spec.resolution));
        const int iy = static_cast<int>(std::floor((p.y - spec.ymin) / spec.resolution));
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) continue;
        const int ci = CellIndex(ix, iy, nx);
        count[ci]++;
        sum_z[ci] += static_cast<double>(p.z);
        sum_i[ci] += static_cast<double>(p.intensity);
        sum_nz[ci] += static_cast<double>(std::abs(n.normal_z));
        sum_side_normal[ci] += static_cast<double>(std::sqrt(n.normal_x * n.normal_x + n.normal_y * n.normal_y));
        min_z[ci] = std::min(min_z[ci], p.z);
        max_z[ci] = std::max(max_z[ci], p.z);
    }

    out->nx = nx;
    out->ny = ny;
    out->xmin = spec.xmin;
    out->ymin = spec.ymin;
    out->resolution = spec.resolution;
    out->density.assign(ncells, 0.f);
    out->max_height.assign(ncells, 0.f);
    out->min_height.assign(ncells, 0.f);
    out->mean_height.assign(ncells, 0.f);
    out->mean_intensity.assign(ncells, 0.f);
    out->verticality.assign(ncells, 0.f);
    out->rail_probability.assign(ncells, 0.f);

    for (int c = 0; c < ncells; ++c) {
        if (count[c] <= 0) continue;
        out->density[c] = static_cast<float>(count[c]);
        out->mean_height[c] = static_cast<float>(sum_z[c] / static_cast<double>(count[c]));
        out->min_height[c] = min_z[c];
        out->max_height[c] = max_z[c];
        out->mean_intensity[c] = static_cast<float>(sum_i[c] / static_cast<double>(count[c]));
        out->verticality[c] = static_cast<float>(sum_nz[c] / static_cast<double>(count[c]));
    }

    std::vector<float> rail_evidence(ncells, 0.f);
    std::vector<uint8_t> candidate(ncells, 0);

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            const int c = CellIndex(ix, iy, nx);
            if (count[c] < min_candidate_density) continue;
            std::vector<float> local_ground_values;
            std::vector<float> local_intensity_values;
            std::vector<float> local_density_values;
            local_ground_values.reserve(static_cast<size_t>((2 * local_ground_window + 1) * (2 * local_ground_window + 1)));
            for (int oy = -local_ground_window; oy <= local_ground_window; ++oy) {
                for (int ox = -local_ground_window; ox <= local_ground_window; ++ox) {
                    const int xx = ix + ox;
                    const int yy = iy + oy;
                    if (xx < 0 || xx >= nx || yy < 0 || yy >= ny) continue;
                    const int nc = CellIndex(xx, yy, nx);
                    if (count[nc] <= 0) continue;
                    local_ground_values.push_back(min_z[nc]);
                    local_intensity_values.push_back(out->mean_intensity[nc]);
                    local_density_values.push_back(out->density[nc]);
                }
            }
            if (local_ground_values.empty()) continue;
            const float local_ground = Percentile(local_ground_values, ground_percentile);
            const float height_residual = out->max_height[c] - local_ground;
            const float height_range = out->max_height[c] - out->min_height[c];
            const bool in_legacy_band = out->max_height[c] >= rail_band_zmin && out->min_height[c] <= rail_band_zmax;
            if (!in_legacy_band || height_residual < residual_min || height_residual > residual_max ||
                height_range > max_candidate_height_range) {
                continue;
            }
            candidate[c] = 1;
            const float height_score = TriangularScore(height_residual, residual_min, 0.08f, 0.22f, residual_max);

            float intensity_score = 0.f;
            if (local_intensity_values.size() > 1) {
                double mean = 0.0;
                for (float v : local_intensity_values) mean += v;
                mean /= static_cast<double>(local_intensity_values.size());
                double var = 0.0;
                for (float v : local_intensity_values) var += (static_cast<double>(v) - mean) * (static_cast<double>(v) - mean);
                var /= static_cast<double>(local_intensity_values.size());
                const double stdv = std::sqrt(var) + 1e-6;
                intensity_score = Clamp01(static_cast<float>(std::abs(static_cast<double>(out->mean_intensity[c]) - mean) / (2.0 * stdv)));
            }
            const float local_density_p90 = std::max(1e-6f, Percentile(local_density_values, 0.90));
            const float density_score = Clamp01(out->density[c] / local_density_p90);
            const float normal_side_score = Clamp01(static_cast<float>(sum_side_normal[c] / std::max(1, count[c])));
            rail_evidence[c] = Clamp01(
                0.30f * height_score +
                0.25f * intensity_score +
                0.25f * normal_side_score +
                0.20f * density_score);
        }
    }

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            const int c = CellIndex(ix, iy, nx);
            if (!candidate[c] || rail_evidence[c] <= 0.f) continue;
            double sw = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
            for (int oy = -linearity_window; oy <= linearity_window; ++oy) {
                for (int ox = -linearity_window; ox <= linearity_window; ++ox) {
                    const int xx = ix + ox;
                    const int yy = iy + oy;
                    if (xx < 0 || xx >= nx || yy < 0 || yy >= ny) continue;
                    const int nc = CellIndex(xx, yy, nx);
                    const double w = static_cast<double>(rail_evidence[nc]);
                    if (w <= 1e-6) continue;
                    const double mx = spec.xmin + (static_cast<double>(xx) + 0.5) * spec.resolution;
                    const double my = spec.ymin + (static_cast<double>(yy) + 0.5) * spec.resolution;
                    sw += w;
                    sx += w * mx;
                    sy += w * my;
                    sxx += w * mx * mx;
                    syy += w * my * my;
                    sxy += w * mx * my;
                }
            }
            float linearity_score = 0.f;
            if (sw > 1e-6) {
                const double mx = sx / sw;
                const double my = sy / sw;
                const double cxx = std::max(0.0, sxx / sw - mx * mx);
                const double cyy = std::max(0.0, syy / sw - my * my);
                const double cxy = sxy / sw - mx * my;
                const double tr = cxx + cyy;
                const double det_term = std::sqrt(std::max(0.0, (cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy));
                const double l0 = 0.5 * (tr + det_term);
                const double l1 = 0.5 * (tr - det_term);
                const double linearity = l0 / (l1 + 1e-6);
                linearity_score = Clamp01(static_cast<float>((linearity - 2.0) / 8.0));
            }
            out->rail_probability[c] = Clamp01(rail_evidence[c] * (0.35f + 0.65f * linearity_score));
        }
    }

    std::cout << "[BEV] Built grid " << nx << " x " << ny << " cells\n";
    return true;
}

bool ComputeBEVStructuralEdgeStrength(const BEVChannels& bev, std::vector<float>* out_strength) {
    if (!out_strength || bev.nx < 2 || bev.ny < 2) {
        return false;
    }
    const int nx = bev.nx;
    const int ny = bev.ny;
    if (static_cast<int>(bev.mean_height.size()) < nx * ny ||
        static_cast<int>(bev.rail_probability.size()) < nx * ny ||
        static_cast<int>(bev.density.size()) < nx * ny) {
        return false;
    }

    cv::Mat acc = cv::Mat::zeros(ny, nx, CV_32F);

    auto accumulate_gradient = [&](const std::vector<float>& channel) {
        cv::Mat m(ny, nx, CV_32F, const_cast<float*>(channel.data()));
        m = m.clone();
        cv::patchNaNs(m, 0);
        double mn = 0.0, mx = 0.0;
        cv::minMaxLoc(m, &mn, &mx);
        if (mx - mn > 1e-6) {
            m = (m - static_cast<float>(mn)) / static_cast<float>(mx - mn) * 255.0f;
        } else {
            m = 0.0f;
        }
        cv::Mat gx, gy, mag;
        cv::Sobel(m, gx, CV_32F, 1, 0, 3);
        cv::Sobel(m, gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, mag);
        acc += mag;
    };

    accumulate_gradient(bev.mean_height);
    accumulate_gradient(bev.rail_probability);
    accumulate_gradient(bev.density);

    cv::normalize(acc, acc, 0.0, 1.0, cv::NORM_MINMAX);
    out_strength->resize(static_cast<size_t>(nx * ny));
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            (*out_strength)[static_cast<size_t>(y * nx + x)] = acc.at<float>(y, x);
        }
    }
    return true;
}
