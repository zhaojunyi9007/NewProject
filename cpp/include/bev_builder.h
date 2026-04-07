#ifndef BEV_BUILDER_H
#define BEV_BUILDER_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
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

struct BEVChannels {
    int nx = 0;
    int ny = 0;
    double xmin = 0.0;
    double ymin = 0.0;
    double resolution = 0.2;
    std::vector<float> density;
    std::vector<float> max_height;
    std::vector<float> min_height;
    std::vector<float> mean_height;
    std::vector<float> mean_intensity;
    std::vector<float> verticality;
    std::vector<float> rail_probability;
};

// 从点云（需与法向数组一一对应）构建 BEV 特征；rail 概率在近地轨带内用密度与法向增强。
bool BuildLidarBEV(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals,
    const BEVGridSpec& spec,
    double rail_band_zmin,
    double rail_band_zmax,
    BEVChannels* out);

// 将 BEV 多通道写入二进制 + JSON 侧车文件，供 Python 打包为 bev_maps.npz。
bool SaveBEVChannelsRaw(const std::string& output_base, const BEVChannels& bev);

// 读取 SaveBEVChannelsRaw / Python 导出的单通道 rail 的 EDGEBEV1 二进制。
bool LoadBEVChannelsBin(const std::string& path, BEVChannels* out);

// Phase3：对 mean_height / rail_probability / density 做 Sobel，融合为归一化边缘强度图 [0,1]，栅格顺序 iy*nx+ix。
bool ComputeBEVStructuralEdgeStrength(const BEVChannels& bev, std::vector<float>* out_strength);

inline bool BEVWorldToCell(double x, double y, const BEVChannels& bev, int* ix, int* iy) {
    if (!ix || !iy || bev.resolution <= 1e-12) return false;
    *ix = static_cast<int>(std::floor((x - bev.xmin) / bev.resolution));
    *iy = static_cast<int>(std::floor((y - bev.ymin) / bev.resolution));
    return *ix >= 0 && *ix < bev.nx && *iy >= 0 && *iy < bev.ny;
}

#endif
