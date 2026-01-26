#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

// 基础常量
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 语义标签定义 (ID Mapping)
enum SemanticLabel {
    LABEL_UNKNOWN = 0,
    LABEL_ROAD = 1,      // 平面 (Ground/Road)
    LABEL_VEGETATION = 2,// 散乱 (Vegetation/Tree)
    LABEL_STRUCTURE = 3, // 线性/垂直 (Pole/Building/Fence)
    LABEL_VEHICLE = 4    // 车辆 (Car/Truck)
};

// --- 数据结构定义 ---

// 1. 用于 Coarse 阶段的点特征 (包含法向量用于一致性校验)
struct PointFeature {
    Eigen::Vector3d p;      // 位置 x, y, z
    float intensity;        // 反射强度
    Eigen::Vector3d normal; // 法向量 nx, ny, nz
    int label;              // 语义标签 (0-4)
    double weight;          // 时空一致性权重 (0.0~1.0, 越高越静态)
    
    // 默认构造函数
    PointFeature() : p(0, 0, 0), intensity(0.0f), normal(0, 0, 0), label(0), weight(1.0) {}
    
    // 带参数构造函数
    PointFeature(const Eigen::Vector3d& pos, float intens, const Eigen::Vector3d& norm, int lbl=0, double w=1.0)
        : p(pos), intensity(intens), normal(norm), label(lbl), weight(w) {}
};

// 2. 用于 Fine 阶段的 3D 线特征 
struct Line3D {
    Eigen::Vector3d p1; // 端点1
    Eigen::Vector3d p2; // 端点2
    int type;           // 0: 铁轨 (Horizontal), 1: 立柱 (Vertical)
    
    // 默认构造函数
    Line3D() : p1(0, 0, 0), p2(0, 0, 0), type(0) {}
    
    // 带参数构造函数
    Line3D(const Eigen::Vector3d& pt1, const Eigen::Vector3d& pt2, int t)
        : p1(pt1), p2(pt2), type(t) {}
};

// 3. 用于 Fine 阶段的 2D 线特征 
struct Line2D {
    Eigen::Vector2d p1; // 端点1 (u, v)
    Eigen::Vector2d p2; // 端点2 (u, v)
    int type;           // 0: Horizontal, 1: Vertical
    
    // 默认构造函数
    Line2D() : p1(0, 0), p2(0, 0), type(0) {}
    
    // 带参数构造函数
    Line2D(const Eigen::Vector2d& pt1, const Eigen::Vector2d& pt2, int t)
        : p1(pt1), p2(pt2), type(t) {}
};

// --- 命名空间：IO工具函数声明 ---
namespace IOUtils {
    // 加载 KITTI 格式的 bin 文件
    bool LoadKittiBin(const std::string& bin_file, std::vector<PointFeature>& points);
    
    // 加载点特征文件 (x y z intensity nx ny nz)
    bool LoadPointFeatures(const std::string& filepath, std::vector<PointFeature>& points);
    
    // 加载 3D 线特征文件 (x1 y1 z1 x2 y2 z2 type)
    bool LoadLines3D(const std::string& filepath, std::vector<Line3D>& lines);
    
    // 加载 2D 线特征文件 (u1 v1 u2 v2 type)
    bool LoadLines2D(const std::string& filepath, std::vector<Line2D>& lines);
    
    // 加载 KITTI 标定文件
    bool LoadKittiCalib(const std::string& calib_file,
                        Eigen::Matrix3d& K,
                        Eigen::Matrix3d& R_rect,
                        Eigen::Matrix<double, 3, 4>& P_rect);
}

#endif // COMMON_H
