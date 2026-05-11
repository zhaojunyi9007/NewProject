#ifndef BEV_IO_H
#define BEV_IO_H

#include <cmath>
#include <string>
#include <vector>

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

// 将 BEV 多通道写入二进制 + JSON 侧车文件，供 Python 打包为 bev_maps.npz。
bool SaveBEVChannelsRaw(const std::string& output_base, const BEVChannels& bev);

// 读取 SaveBEVChannelsRaw / Python 导出的单通道 rail 的 EDGEBEV1 二进制。
bool LoadBEVChannelsBin(const std::string& path, BEVChannels* out);

inline bool BEVWorldToCell(double x, double y, const BEVChannels& bev, int* ix, int* iy) {
    if (!ix || !iy || bev.resolution <= 1e-12) return false;
    *ix = static_cast<int>(std::floor((x - bev.xmin) / bev.resolution));
    *iy = static_cast<int>(std::floor((y - bev.ymin) / bev.resolution));
    return *ix >= 0 && *ix < bev.nx && *iy >= 0 && *iy < bev.ny;
}

#endif
