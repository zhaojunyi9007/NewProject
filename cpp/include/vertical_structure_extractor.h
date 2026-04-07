#ifndef VERTICAL_STRUCTURE_EXTRACTOR_H
#define VERTICAL_STRUCTURE_EXTRACTOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "include/common.h"
#include <vector>

struct VerticalStructureConfig {
    double z_min = -1.0;
    double z_max = 5.0;
    double cluster_tolerance = 0.35;
    int min_cluster_size = 25;
    double vertical_line_distance_thresh = 0.12;
};

// 候选（高程 + 可选标签）→ 欧式聚类 → 各簇内拟合竖直线（平行 Z），替代整片点云全局 RANSAC。
std::vector<Line3D> ExtractVerticalStructures(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const std::vector<int>& per_point_labels,
    const VerticalStructureConfig& cfg,
    int structure_label_id = 3);

#endif
