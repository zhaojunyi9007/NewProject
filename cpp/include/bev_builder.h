#ifndef BEV_BUILDER_H
#define BEV_BUILDER_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "include/bev_io.h"
#include <string>

// LiDAR 坐标系（OSDaR23）：X 前向，Y 左向，Z 向上。BEV 在 Z = reference_plane 上铺栅格，向 XY 投影。

struct BEVGridSpec {
    double xmin = 0.0;
    double xmax = 80.0;
    double ymin = -20.0;
    double ymax = 20.0;
    double resolution = 0.2;
    double reference_plane_z = 0.0;

    int nx() const;
    int ny() const;
};

// 从点云（需与法向数组一一对应）构建 BEV 特征；rail 概率在近地轨带内用密度与法向增强。
bool BuildLidarBEV(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals,
    const BEVGridSpec& spec,
    double rail_band_zmin,
    double rail_band_zmax,
    BEVChannels* out);

// Phase3：对 mean_height / rail_probability / density 做 Sobel，融合为归一化边缘强度图 [0,1]，栅格顺序 iy*nx+ix。
bool ComputeBEVStructuralEdgeStrength(const BEVChannels& bev, std::vector<float>* out_strength);


#endif
