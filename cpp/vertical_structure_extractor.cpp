#include "include/vertical_structure_extractor.h"
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
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

        double sx = 0, sy = 0, szmin = 1e9, szmax = -1e9;
        for (int id : ind.indices) {
            const auto& p = cand->points[static_cast<size_t>(id)];
            sx += p.x;
            sy += p.y;
            szmin = std::min(szmin, static_cast<double>(p.z));
            szmax = std::max(szmax, static_cast<double>(p.z));
        }
        const double n = static_cast<double>(ind.indices.size());
        sx /= n;
        sy /= n;

        Line3D L;
        L.p1 = Eigen::Vector3d(sx, sy, szmin);
        L.p2 = Eigen::Vector3d(sx, sy, szmax);
        L.type = 1;
        lines.push_back(L);
        if (static_cast<int>(lines.size()) >= 32) break;
    }

    return lines;
}
