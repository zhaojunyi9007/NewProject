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
    float max_rail = 0.f;
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
        out->rail_probability[c] = static_cast<float>(rail_acc[c]);
        max_rail = std::max(max_rail, out->rail_probability[c]);
    }

    // 归一化 rail 概率：轨带内点数相对强度 × 与局部密度的关系
    for (int c = 0; c < ncells; ++c) {
        if (max_rail > 1e-6f) {
            float r = out->rail_probability[c] / max_rail;
            float d = (max_dens > 1e-6f) ? (out->density[c] / max_dens) : 0.f;
            out->rail_probability[c] = std::min(1.f, 0.55f * r + 0.45f * d);
        } else {
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
