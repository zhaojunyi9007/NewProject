#include "include/vertical_structure_extractor.h"
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <iostream>

std::vector<Line3D> ExtractVerticalStructures(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const std::vector<int>& per_point_labels,
    const VerticalStructureConfig& cfg,
    int structure_label_id) {
    std::vector<Line3D> lines;
    if (!cloud || cloud->empty()) return lines;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cand(new pcl::PointCloud<pcl::PointXYZI>);
    cand->reserve(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& p = cloud->points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        if (p.z < cfg.z_min || p.z > cfg.z_max) continue;
        const double range = std::sqrt(static_cast<double>(p.x) * p.x + static_cast<double>(p.y) * p.y);
        if (cfg.max_range_m > 0.0 && range > cfg.max_range_m) continue;
        if (structure_label_id >= 0 && static_cast<int>(per_point_labels.size()) > static_cast<int>(i)) {
            if (per_point_labels[i] != structure_label_id) continue;
        }
        cand->push_back(p);
    }

    if (cand->size() < static_cast<size_t>(cfg.min_cluster_size)) {
        std::cout << "[VertStruct] Not enough candidate points: " << cand->size() << std::endl;
        return lines;
    }

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cand);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(static_cast<float>(cfg.cluster_tolerance));
    ec.setMinClusterSize(cfg.min_cluster_size);
    ec.setMaxClusterSize(1000000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cand);
    ec.extract(cluster_indices);

    std::cout << "[VertStruct] Clusters: " << cluster_indices.size() << std::endl;

    for (const auto& ind : cluster_indices) {
        if (ind.indices.size() < static_cast<size_t>(cfg.min_cluster_size)) continue;

        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        double szmin = 1e9, szmax = -1e9;
        for (int id : ind.indices) {
            const auto& p = cand->points[static_cast<size_t>(id)];
            mean += Eigen::Vector3d(p.x, p.y, p.z);
            szmin = std::min(szmin, static_cast<double>(p.z));
            szmax = std::max(szmax, static_cast<double>(p.z));
        }
        const double n = static_cast<double>(ind.indices.size());
        mean /= std::max(1.0, n);
        const double height = szmax - szmin;
        const double range = std::sqrt(mean.x() * mean.x() + mean.y() * mean.y());
        if (height < cfg.min_height_m) continue;
        if (cfg.max_range_m > 0.0 && range > cfg.max_range_m) continue;

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        double radius_xy = 0.0;
        for (int id : ind.indices) {
            const auto& p = cand->points[static_cast<size_t>(id)];
            Eigen::Vector3d q(p.x, p.y, p.z);
            Eigen::Vector3d d = q - mean;
            cov += d * d.transpose();
            const double rxy = std::hypot(q.x() - mean.x(), q.y() - mean.y());
            radius_xy = std::max(radius_xy, rxy);
        }
        cov /= std::max(1.0, n - 1.0);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        if (solver.info() != Eigen::Success) continue;
        Eigen::Vector3d ev = solver.eigenvalues();
        Eigen::Matrix3d evec = solver.eigenvectors();
        const double l1 = std::max(1e-9, ev[2]);
        const double l2 = std::max(0.0, ev[1]);
        const double linearity = (l1 - l2) / l1;
        const double verticality = std::abs(evec.col(2).z());
        const double plane_extent = 2.0 * std::sqrt(std::max(0.0, l2));

        if (radius_xy > cfg.max_radius_m) continue;
        if (linearity < cfg.min_linearity) continue;
        if (verticality < cfg.min_verticality) continue;
        if (plane_extent > cfg.max_plane_extent_m) continue;

        Line3D L;
        L.p1 = Eigen::Vector3d(mean.x(), mean.y(), szmin);
        L.p2 = Eigen::Vector3d(mean.x(), mean.y(), szmax);
        L.type = 1;
        L.class_id = SEM_VERTICAL_STRUCTURE;
        const double h_score = std::min(1.0, height / std::max(1e-6, cfg.min_height_m * 2.0));
        const double r_score = std::max(0.0, 1.0 - radius_xy / std::max(1e-6, cfg.max_radius_m));
        const double c = 0.35 * linearity + 0.35 * verticality + 0.20 * h_score + 0.10 * r_score;
        L.confidence = static_cast<float>(std::min(1.0, std::max(0.05, c)));
        lines.push_back(L);
        if (static_cast<int>(lines.size()) >= std::max(0, cfg.max_lines)) break;
    }

    return lines;
}
