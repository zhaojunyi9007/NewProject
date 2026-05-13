#include "include/bev_builder.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

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

    const int ncells = nx * ny;
    std::vector<int> count(ncells, 0);
    std::vector<double> sum_z(ncells, 0.0);
    std::vector<double> sum_i(ncells, 0.0);
    std::vector<double> sum_nz(ncells, 0.0);
    std::vector<double> sum_side_normal(ncells, 0.0);
    std::vector<double> sum_x(ncells, 0.0);
    std::vector<double> sum_y(ncells, 0.0);
    std::vector<double> sum_xx(ncells, 0.0);
    std::vector<double> sum_yy(ncells, 0.0);
    std::vector<double> sum_xy(ncells, 0.0);
    std::vector<float> min_z(ncells, std::numeric_limits<float>::infinity());
    std::vector<float> max_z(ncells, -std::numeric_limits<float>::infinity());
    std::vector<double> rail_acc(ncells, 0.0);

    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& p = cloud->points[i];
        const auto& n = normals->points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) continue;

        const int ix = static_cast<int>(std::floor((p.x - spec.xmin) / spec.resolution));
        const int iy = static_cast<int>(std::floor((p.y - spec.ymin) / spec.resolution));
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) continue;

        const int ci = CellIndex(ix, iy, nx);
        count[ci]++;
        sum_z[ci] += static_cast<double>(p.z);
        sum_i[ci] += static_cast<double>(p.intensity);
        const float nz = std::abs(n.normal_z);
        sum_nz[ci] += static_cast<double>(nz);
        const float side_normal = std::sqrt(n.normal_x * n.normal_x + n.normal_y * n.normal_y);
        sum_side_normal[ci] += static_cast<double>(side_normal);
        sum_x[ci] += static_cast<double>(p.x);
        sum_y[ci] += static_cast<double>(p.y);
        sum_xx[ci] += static_cast<double>(p.x) * static_cast<double>(p.x);
        sum_yy[ci] += static_cast<double>(p.y) * static_cast<double>(p.y);
        sum_xy[ci] += static_cast<double>(p.x) * static_cast<double>(p.y);
        min_z[ci] = std::min(min_z[ci], p.z);
        max_z[ci] = std::max(max_z[ci], p.z);

        if (p.z >= rail_band_zmin && p.z <= rail_band_zmax) {
            rail_acc[ci] += 1.0;
        }
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

    float max_dens = 0.f;
    for (int c = 0; c < ncells; ++c) {
        if (count[c] <= 0) continue;
        const float dens = static_cast<float>(count[c]);
        out->density[c] = dens;
        max_dens = std::max(max_dens, dens);
        out->mean_height[c] = static_cast<float>(sum_z[c] / static_cast<double>(count[c]));
        out->min_height[c] = min_z[c];
        out->max_height[c] = max_z[c];
        out->mean_intensity[c] = static_cast<float>(sum_i[c] / static_cast<double>(count[c]));
        out->verticality[c] = static_cast<float>(sum_nz[c] / static_cast<double>(count[c]));
    }

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            const int c = CellIndex(ix, iy, nx);
            if (count[c] <= 0 || rail_acc[c] <= 0.0) {
                out->rail_probability[c] = 0.f;
                continue;
            }

            float local_ground = std::numeric_limits<float>::infinity();
            double local_i_sum = 0.0;
            double local_i_sq_sum = 0.0;
            int local_i_count = 0;
            for (int oy = -2; oy <= 2; ++oy) {
                for (int ox = -2; ox <= 2; ++ox) {
                    const int nx0 = ix + ox;
                    const int ny0 = iy + oy;
                    if (nx0 < 0 || nx0 >= nx || ny0 < 0 || ny0 >= ny) continue;
                    const int nc = CellIndex(nx0, ny0, nx);
                    if (count[nc] <= 0) continue;
                    local_ground = std::min(local_ground, min_z[nc]);
                    const double mi = static_cast<double>(out->mean_intensity[nc]);
                    local_i_sum += mi;
                    local_i_sq_sum += mi * mi;
                    local_i_count++;
                }
            }
            if (!std::isfinite(local_ground)) {
                local_ground = min_z[c];
            }

            const float density_norm = (max_dens > 1e-6f) ? Clamp01(out->density[c] / max_dens) : 0.f;
            const float height_residual = std::max(0.f, out->max_height[c] - local_ground);
            const float height_residual_score = Clamp01(height_residual / 0.25f);

            float intensity_contrast_score = 0.f;
            if (local_i_count > 1) {
                const double mean_i = local_i_sum / static_cast<double>(local_i_count);
                const double var_i = std::max(0.0, local_i_sq_sum / static_cast<double>(local_i_count) - mean_i * mean_i);
                const double std_i = std::sqrt(var_i) + 1e-6;
                intensity_contrast_score = Clamp01(static_cast<float>(std::abs(out->mean_intensity[c] - mean_i) / (2.0 * std_i)));
            }

            const float normal_side_score =
                Clamp01(static_cast<float>(sum_side_normal[c] / std::max(1, count[c])));

            float local_linearity_score = 0.f;
            if (count[c] >= 3) {
                const double inv_n = 1.0 / static_cast<double>(count[c]);
                const double mx = sum_x[c] * inv_n;
                const double my = sum_y[c] * inv_n;
                const double cxx = std::max(0.0, sum_xx[c] * inv_n - mx * mx);
                const double cyy = std::max(0.0, sum_yy[c] * inv_n - my * my);
                const double cxy = sum_xy[c] * inv_n - mx * my;
                const double tr = cxx + cyy;
                const double det_term = std::sqrt(std::max(0.0, (cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy));
                const double l0 = 0.5 * (tr + det_term);
                const double l1 = 0.5 * (tr - det_term);
                if (l0 > 1e-9) {
                    local_linearity_score = Clamp01(static_cast<float>((l0 - l1) / l0));
                }
            }

            out->rail_probability[c] = Clamp01(
                0.15f * density_norm +
                0.20f * height_residual_score +
                0.20f * intensity_contrast_score +
                0.20f * normal_side_score +
                0.25f * local_linearity_score);
        }
    }

    for (int c = 0; c < ncells; ++c) {
        if (count[c] <= 0 || rail_acc[c] <= 0.0) {
            out->rail_probability[c] = 0.f;
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
