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

// Phase C1: 更适合铁路场景的细粒度语义 ID（用于 semantic_points.txt 的 semantic_id 列）。
// 注意：这不是深度学习分割，仅用于把现有几何/BEV 线索组织成更可用的语义输入。
enum SemanticIdRailway {
    SEM_UNKNOWN = 0,
    SEM_RAIL_LIKE = 1,
    SEM_BALLAST_GROUND = 2,
    SEM_VERTICAL_STRUCTURE = 3,
    SEM_PLATFORM_OR_BUILDING = 4,
    SEM_VEHICLE_LIKE = 5,
    SEM_VEGETATION_LIKE = 6,
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
    // Phase C5: 扩展字段（用于优化/诊断）
    int class_id;       // 语义类别（与 SemanticIdRailway/图像通道映射一致的约定）
    float confidence;   // 置信度/可用性 (0..1)
    
    // 默认构造函数
    Line3D() : p1(0, 0, 0), p2(0, 0, 0), type(0), class_id(0), confidence(1.0f) {}
    
    // 带参数构造函数
    Line3D(const Eigen::Vector3d& pt1, const Eigen::Vector3d& pt2, int t)
        : p1(pt1), p2(pt2), type(t), class_id(0), confidence(1.0f) {}
};

// Phase 3: 语义点记录（与 PointFeature 兼容，增加显式 semantic_id）
struct SemanticPointRecord {
    Eigen::Vector3d p;
    float intensity;
    Eigen::Vector3d normal;
    int label;
    double weight;
    // 细粒度语义 ID，优先使用 SemanticIdRailway（见上），用于与图像侧 semantic_probs 通道对齐/映射。
    int semantic_id;
};

// Phase 4+: 粗位姿增量占位
struct PoseDelta {
    double rx = 0, ry = 0, rz = 0, tx = 0, ty = 0, tz = 0;
};
// 兼容旧名（逐步降级）：统一使用 PoseDelta。
using PoseDelta6 = PoseDelta;

// 3. 用于 Fine 阶段的 2D 线特征 
struct Line2D {
    Eigen::Vector2d p1; // 端点1 (u, v)
    Eigen::Vector2d p2; // 端点2 (u, v)
    int type;           // 0: Horizontal, 1: Vertical
    // Phase C5: 扩展字段（用于把 2D 线与语义概率场关联）
    double semantic_support; // 线段区域的语义支持度（例如沿线的语义概率均值/一致性），(0..1)
    int class_id;            // 线段对应类别（可选）
    float confidence;        // 线段质量/置信度 (0..1)
    
    // 默认构造函数
    Line2D()
        : p1(0, 0), p2(0, 0), type(0),
          semantic_support(0.0), class_id(0), confidence(1.0f) {}
    
    // 带参数构造函数
    Line2D(const Eigen::Vector2d& pt1, const Eigen::Vector2d& pt2, int t)
        : p1(pt1), p2(pt2), type(t),
          semantic_support(0.0), class_id(0), confidence(1.0f) {}
};

// Phase C5: 统一的 score breakdown（供优化器/下游统一读取）。
// 说明：当前字段与 Phase A3 的输出键保持一致，并兼容 Phase B4 的加权分解。
struct ScoreBreakdown {
    // raw components
    double semantic_js_divergence = 0.0;     // lower is better
    double semantic_hist_similarity = 0.0;   // higher is better (0..1)
    double edge_score_norm = 0.0;            // normalized edge attraction (higher is better)
    double line_score_norm = 0.0;            // placeholder (higher is better)

    // weighted components used in total
    double semantic_js_score = 0.0;          // typically = -semantic_js_divergence
    double semantic_hist_score = 0.0;        // typically = semantic_hist_similarity
    double edge_score = 0.0;
    double line_score = 0.0;

    // Phase A3 keys (observability-related)
    double rail_confidence = 0.0;
    double vertical_structure_confidence = 0.0;

    double total_score = 0.0;
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

    // Load OSDaR23 calibration.txt for a given camera folder (e.g. rgb_center).
    // Output:
    //   K: camera intrinsics
    //   T_lidar_to_cam: 4x4 transform (LiDAR/parent frame -> camera frame)
    bool LoadOSDaRCalib(const std::string& calib_file,
                        const std::string& camera_folder,
                        Eigen::Matrix3d& K,
                        Eigen::Matrix4d& T_lidar_to_cam);
}

#endif // COMMON_H
