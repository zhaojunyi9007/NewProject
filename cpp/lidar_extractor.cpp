#include "include/common.h"
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

// PCL Headers
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/registration/ndt.h>

// ========================================
// 深度不连续性边缘提取 (README2.0 第二阶段 2.2)
// 使用球面投影构建深度图并检测邻域跳变
// ========================================
struct DepthEdgeResult {
    std::vector<bool> edge_flags;
    std::vector<int> edge_types; // 0=Unknown, 1=Horizontal, 2=Vertical
    int edge_count = 0;
};

//作用：通过球面投影构建深度图，对相邻像素深度差做阈值判断，标记深度跳变点为边缘。
//输入：点云、投影图尺寸、视场角与深度差阈值。
//输出：DepthEdgeResult，包含每个点是否为边缘及边缘类型。
//调用关系：在 main 阶段 4 被调用，结果用于后续权重提升与保存边缘点文件。
DepthEdgeResult DetectDepthDiscontinuityEdges(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    int width = 1024,
    int height = 64,
    float v_fov_up_deg = 2.0f,
    float v_fov_down_deg = -24.9f,
    float depth_jump_threshold = 0.5f) {
    // 输入: cloud(点云), 视场与深度阈值参数
    // 输出: DepthEdgeResult(边缘标记/类型/数量)
    DepthEdgeResult result;
    if (!cloud || cloud->empty()) {
        return result;
    }

    const float v_fov_up = v_fov_up_deg * static_cast<float>(M_PI) / 180.0f;
    const float v_fov_down = v_fov_down_deg * static_cast<float>(M_PI) / 180.0f;
    const float v_fov_total = std::abs(v_fov_down) + std::abs(v_fov_up);

    std::vector<float> range_image(width * height, std::numeric_limits<float>::infinity());
    std::vector<int> index_image(width * height, -1);

    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& p = cloud->points[i];
        float depth = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (!std::isfinite(depth) || depth <= 1e-3f) {
            continue;
        }

        float yaw = std::atan2(p.y, p.x);
        float pitch = std::asin(p.z / depth);

        float proj_x = 0.5f * (1.0f - yaw / static_cast<float>(M_PI));
        float proj_y = 1.0f - (pitch + std::abs(v_fov_down)) / v_fov_total;

        int u = static_cast<int>(proj_x * width);
        int v = static_cast<int>(proj_y * height);

        if (u < 0 || u >= width || v < 0 || v >= height) {
            continue;
        }

        int idx = v * width + u;
        if (depth < range_image[idx]) {
            range_image[idx] = depth;
            index_image[idx] = static_cast<int>(i);
        }
    }

    result.edge_flags.assign(cloud->size(), false);
    result.edge_types.assign(cloud->size(), 0);

    auto mark_edge = [&](int idx_a, int idx_b, int edge_type) {
        if (idx_a >= 0) {
            result.edge_flags[static_cast<size_t>(idx_a)] = true;
            result.edge_types[static_cast<size_t>(idx_a)] = std::max(result.edge_types[static_cast<size_t>(idx_a)], edge_type);
        }
        if (idx_b >= 0) {
            result.edge_flags[static_cast<size_t>(idx_b)] = true;
            
        }
    };

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            int idx = v * width + u;
            int idx_center = index_image[idx];
            if (idx_center < 0) {
                continue;
            }

            float range_center = range_image[idx];

            if (u + 1 < width) {
                int idx_right = index_image[idx + 1];
                if (idx_right >= 0) {
                    float diff = std::abs(range_center - range_image[idx + 1]);
                    if (diff > depth_jump_threshold) {
                        mark_edge(idx_center, idx_right, 1);
                    }
                }
            }

            if (v + 1 < height) {
                int idx_down = index_image[idx + width];
                if (idx_down >= 0) {
                    float diff = std::abs(range_center - range_image[idx + width]);
                    if (diff > depth_jump_threshold) {
                        mark_edge(idx_center, idx_down, 2);
                    }
                }
            }
        }
    }

    result.edge_count = static_cast<int>(std::count(result.edge_flags.begin(), result.edge_flags.end(), true));
    return result;
}


// IOUtils 用于加载 KITTI bin 文件到 PCL 点云
namespace LocalIO {
    bool LoadKittiBin(const std::string& bin_file, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
        std::ifstream file(bin_file, std::ios::binary);
        if (!file.good()) {
            std::cerr << "[Error] Cannot open bin file: " << bin_file << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::beg);
        
        // KITTI bin: x, y, z, intensity (float)
        while (file.good() && !file.eof()) {
            float data[4];
            file.read(reinterpret_cast<char*>(data), sizeof(float) * 4);
            
            // 添加安全检查
            if (file.gcount() < static_cast<std::streamsize>(sizeof(float) * 4)) break;
            
            pcl::PointXYZI p;
            p.x = data[0]; 
            p.y = data[1]; 
            p.z = data[2]; 
            p.intensity = data[3];
            cloud->push_back(p);
        }
        
        std::cout << "[Info] Loaded " << cloud->size() << " points from bin file" << std::endl;
        return cloud->size() > 0;
    }

    bool LoadPCD(const std::string& pcd_file, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
        if (!cloud) {
            cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
        }
        pcl::PCLPointCloud2 blob;
        if (pcl::io::loadPCDFile(pcd_file, blob) < 0) {
            std::cerr << "[Error] Cannot load PCD file: " << pcd_file << std::endl;
            return false;
        }

        // Convert by field names so extra fields like timestamp/sensor_index won't break loading.
        // OSDaR23 example: FIELDS x y z intensity timestamp sensor_index
        const bool has_x = pcl::getFieldIndex(blob, "x") >= 0;
        const bool has_y = pcl::getFieldIndex(blob, "y") >= 0;
        const bool has_z = pcl::getFieldIndex(blob, "z") >= 0;
        const bool has_i = pcl::getFieldIndex(blob, "intensity") >= 0;
        if (!(has_x && has_y && has_z)) {
            std::cerr << "[Error] PCD missing required fields x/y/z: " << pcd_file << std::endl;
            return false;
        }

        const int idx_x = pcl::getFieldIndex(blob, "x");
        const int idx_y = pcl::getFieldIndex(blob, "y");
        const int idx_z = pcl::getFieldIndex(blob, "z");
        const int idx_i = pcl::getFieldIndex(blob, "intensity");

        const auto& fx = blob.fields[static_cast<size_t>(idx_x)];
        const auto& fy = blob.fields[static_cast<size_t>(idx_y)];
        const auto& fz = blob.fields[static_cast<size_t>(idx_z)];
        const pcl::PCLPointField* fi = (idx_i >= 0) ? &blob.fields[static_cast<size_t>(idx_i)] : nullptr;

        auto read_as_float = [&](const uint8_t* ptr, uint8_t datatype) -> float {
            switch (datatype) {
                case pcl::PCLPointField::INT8: {
                    int8_t v;
                    std::memcpy(&v, ptr, sizeof(v));
                    return static_cast<float>(v);
                }
                case pcl::PCLPointField::UINT8: {
                    uint8_t v;
                    std::memcpy(&v, ptr, sizeof(v));
                    return static_cast<float>(v);
                }
                case pcl::PCLPointField::INT16: {
                    int16_t v;
                    std::memcpy(&v, ptr, sizeof(v));
                    return static_cast<float>(v);
                }
                case pcl::PCLPointField::UINT16: {
                    uint16_t v;
                    std::memcpy(&v, ptr, sizeof(v));
                    return static_cast<float>(v);
                }
                case pcl::PCLPointField::INT32: {
                    int32_t v;
                    std::memcpy(&v, ptr, sizeof(v));
                    return static_cast<float>(v);
                }
                case pcl::PCLPointField::UINT32: {
                    uint32_t v;
                    std::memcpy(&v, ptr, sizeof(v));
                    return static_cast<float>(v);
                }
                case pcl::PCLPointField::FLOAT32: {
                    float v;
                    std::memcpy(&v, ptr, sizeof(v));
                    return v;
                }
                case pcl::PCLPointField::FLOAT64: {
                    double v;
                    std::memcpy(&v, ptr, sizeof(v));
                    return static_cast<float>(v);
                }
                default:
                    return 0.0f;
            }
        };

        const size_t n = static_cast<size_t>(blob.width) * static_cast<size_t>(blob.height);
        cloud->clear();
        cloud->reserve(n);
        if (blob.data.empty() || blob.point_step == 0) {
            std::cerr << "[Error] Empty PCD data or invalid point_step: " << pcd_file << std::endl;
            return false;
        }

        for (size_t idx = 0; idx < n; ++idx) {
            const size_t base = idx * static_cast<size_t>(blob.point_step);
            if (base + blob.point_step > blob.data.size()) {
                break;
            }
            const uint8_t* p0 = blob.data.data() + base;
            pcl::PointXYZI p;
            p.x = read_as_float(p0 + fx.offset, fx.datatype);
            p.y = read_as_float(p0 + fy.offset, fy.datatype);
            p.z = read_as_float(p0 + fz.offset, fz.datatype);
            p.intensity = fi ? read_as_float(p0 + fi->offset, fi->datatype) : 1.0f;
            cloud->push_back(p);
        }

        std::cout << "[Info] Loaded " << cloud->size() << " points from PCD file" << std::endl;
        return cloud->size() > 0;
    }
}

// ========================================
// 作用：对多帧点云进行 NDT 配准和融合，输出融合点云与对齐后的历史帧列表。
// 调用：main 阶段 2 调用，融合点云用于后续降采样和特征提取；aligned_frames 用于时空一致性计算。
// ========================================
struct NDTFusionResult {
    pcl::PointCloud<pcl::PointXYZI>::Ptr fused_cloud;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> aligned_frames;  // 配准后的历史帧
};

double GetEnvDouble(const char* name, double default_value) {
    const char* value = std::getenv(name);
    if (!value) return default_value;
    char* end = nullptr;
    double parsed = std::strtod(value, &end);
    if (end == value) return default_value;
    return parsed;
}

int GetEnvInt(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (!value) return default_value;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value) return default_value;
    return static_cast<int>(parsed);
}

NDTFusionResult FuseMultiFrameNDT(
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& frames) 
{
    NDTFusionResult result;
    
    if (frames.empty()) {
        std::cerr << "[Error] No frames to fuse" << std::endl;
        return result;
    }
    
    if (frames.size() == 1) {
        std::cout << "[Info] Single frame, no fusion needed" << std::endl;
        result.fused_cloud = frames[0];
        // 单帧情况，没有历史帧
        return result;
    }
    
    std::cout << "\n[NDT Fusion] Fusing " << frames.size() << " frames..." << std::endl;
    
    // 使用最后一帧（当前帧）作为目标点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr target = frames.back();
    pcl::PointCloud<pcl::PointXYZI>::Ptr fused(new pcl::PointCloud<pcl::PointXYZI>);
    *fused = *target;  // 先添加目标帧
    
    std::cout << "  Target frame (current): " << target->size() << " points" << std::endl;
    
    // 配置NDT参数 (按照README2.0方案)
    pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
    ndt.setTransformationEpsilon(GetEnvDouble("EDGECALIB_LIDAR_NDT_TRANS_EPS", 0.01));  // 变换收敛阈值
    ndt.setStepSize(GetEnvDouble("EDGECALIB_LIDAR_NDT_STEP_SIZE", 0.1));                 // More/Newton线搜索步长
    ndt.setResolution(GetEnvDouble("EDGECALIB_LIDAR_NDT_RESOLUTION", 1.0));              // NDT网格分辨率 (m)
    ndt.setMaximumIterations(GetEnvInt("EDGECALIB_LIDAR_NDT_MAX_ITERS", 30));            // 最大迭代次数
    
    ndt.setInputTarget(target);
    
    // 依次配准前面的帧到当前帧
    for (size_t i = 0; i < frames.size() - 1; ++i) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr source = frames[i];
        pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>);
        
        std::cout << "  Aligning frame " << i << " (" << source->size() << " points)..." << std::endl;
        
        ndt.setInputSource(source);
        
        // 初始变换矩阵 (假设帧间变化不大，使用单位矩阵)
        Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
        
        // 执行NDT配准
        ndt.align(*aligned, init_guess);
        
        if (ndt.hasConverged()) {
            Eigen::Matrix4f transformation = ndt.getFinalTransformation();
            double fitness_score = ndt.getFitnessScore();
            
            std::cout << "    Converged! Fitness score: " << fitness_score << std::endl;
            // std::cout << "    Transformation:\n" << transformation << std::endl;
            
            // 保存配准后的点云（用于时空一致性计算）
            result.aligned_frames.push_back(aligned);
            
            // 将配准后的点云加入融合结果
            *fused += *aligned;
        } else {
            std::cout << "    [Warning] NDT did not converge, using raw alignment" << std::endl;
            // 即使不收敛，也保存原始点云（可能帧间变化太小）
            result.aligned_frames.push_back(source);
            *fused += *source;
        }
    }
    
    result.fused_cloud = fused;
    std::cout << "[NDT Fusion] Complete. Total points: " << fused->size() << std::endl;
    return result;
}

// ========================================
// 新增：时空一致性加权 (README2.0 第二阶段 2.3)
// 优化版本：使用NDT配准后的点云，预先构建KD-Tree
// ========================================

struct RangeImageMapping {
    int rows = 64;
    int cols = 1024;
    float min_el = -0.25f;
    float max_el = 0.45f;
};

struct RangeImageData {
    int rows;
    int cols;
    std::vector<float> ranges;
    std::vector<int> indices;
};

bool ProjectToRangeImage(const pcl::PointXYZI& p,
                        const RangeImageMapping& cfg,
                        int& row, int& col, float& range) {
    range = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (!std::isfinite(range) || range < 1e-3f) {
        return false;
    }

    float az = std::atan2(p.y, p.x);
    float el = std::atan2(p.z, std::sqrt(p.x * p.x + p.y * p.y));

    float col_f = (az + static_cast<float>(M_PI)) / (2.0f * static_cast<float>(M_PI)) * cfg.cols;
    float row_f = (el - cfg.min_el) / (cfg.max_el - cfg.min_el) * cfg.rows;

    col = static_cast<int>(std::floor(col_f));
    row = static_cast<int>(std::floor(row_f));
    if (row < 0 || row >= cfg.rows || col < 0 || col >= cfg.cols) {
        return false;
    }
    return true;
}

RangeImageData BuildRangeImage(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                               const RangeImageMapping& cfg) {
    RangeImageData data;
    data.rows = cfg.rows;
    data.cols = cfg.cols;
    data.ranges.assign(cfg.rows * cfg.cols, std::numeric_limits<float>::infinity());
    data.indices.assign(cfg.rows * cfg.cols, -1);

    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& p = cloud->points[i];
        /*float range = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (!std::isfinite(range) || range < 1e-3f) continue;

        float az = std::atan2(p.y, p.x);
        float el = std::atan2(p.z, std::sqrt(p.x * p.x + p.y * p.y));

        float col_f = (az + static_cast<float>(M_PI)) / (2.0f * static_cast<float>(M_PI)) * cfg.cols;
        float row_f = (el - cfg.min_el) / (cfg.max_el - cfg.min_el) * cfg.rows;

        int col = static_cast<int>(std::floor(col_f));
        int row = static_cast<int>(std::floor(row_f));
        if (row < 0 || row >= cfg.rows || col < 0 || col >= cfg.cols) continue;*/

        int row = 0;
        int col = 0;
        float range = 0.0f;
        if (!ProjectToRangeImage(p, cfg, row, col, range)) {
            continue;
        }

        int idx = row * cfg.cols + col;
        if (range < data.ranges[idx]) {
            data.ranges[idx] = range;
            data.indices[idx] = static_cast<int>(i);
        }
    }
    return data;
}

/*`std::vector<bool> DetectDepthDiscontinuities(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    float depth_threshold = 0.5f) {
    RangeImageMapping cfg;
    RangeImageData range_img = BuildRangeImage(cloud, cfg);
    std::vector<bool> is_edge(cloud->size(), false);

    const int rows = range_img.rows;
    const int cols = range_img.cols;

    auto idx = [cols](int r, int c) { return r * cols + c; };
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int center_idx = range_img.indices[idx(r, c)];
            float center_range = range_img.ranges[idx(r, c)];
            if (center_idx < 0 || !std::isfinite(center_range)) continue;

            for (int k = 0; k < 4; ++k) {
                int rn = r + dr[k];
                int cn = c + dc[k];
                if (rn < 0 || rn >= rows || cn < 0 || cn >= cols) continue;

                int nbr_idx = range_img.indices[idx(rn, cn)];
                float nbr_range = range_img.ranges[idx(rn, cn)];
                if (nbr_idx < 0 || !std::isfinite(nbr_range)) continue;

                if (std::abs(center_range - nbr_range) > depth_threshold) {
                    is_edge[center_idx] = true;
                    is_edge[nbr_idx] = true;
                }
            }
        }
    }
    return is_edge;
}
*/

// 优化版本：接受配准后的点云列表和降采样后的当前点云
std::vector<double> ComputeTemporalConsistencyOptimized(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& current_downsampled,
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& aligned_frames,
    double position_threshold = 0.5,  // 位置一致性阈值 (m)
    double projection_threshold = 0.5,  // 投影一致性阈值 (m, range image)
    double static_weight = 1.0,
    double dynamic_weight = 1.0)
{
    std::vector<double> weights;
    weights.reserve(current_downsampled->size());
    
    if (aligned_frames.empty() || !current_downsampled) {
        // 单帧情况，所有点权重为1.0
        weights.assign(current_downsampled->size(), 1.0);
        return weights;
    }
    
    std::cout << "\n[Temporal Filter] Computing spatial-temporal consistency..." << std::endl;
    std::cout << "  Current points: " << current_downsampled->size() << std::endl;
    std::cout << "  Historical frames: " << aligned_frames.size() << std::endl;
    
    if (aligned_frames.size() == 0) {
        // 单帧情况，所有点权重为1.0
        weights.assign(current_downsampled->size(), 1.0);
        std::cout << "  Single frame, all weights = 1.0" << std::endl;
        return weights;
    }
    
    // 优化1: 预先为所有历史帧构建KD-Tree（一次性构建）
    std::vector<pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr> hist_kdtrees;
    hist_kdtrees.reserve(aligned_frames.size());
    
    std::cout << "  Pre-building KD-Trees for " << aligned_frames.size() << " historical frames..." << std::endl;
    for (size_t i = 0; i < aligned_frames.size(); ++i) {
        pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree(new pcl::KdTreeFLANN<pcl::PointXYZI>);
        kdtree->setInputCloud(aligned_frames[i]);
        hist_kdtrees.push_back(kdtree);
    }
    std::cout << "  KD-Trees built." << std::endl;

    // 预先构建历史帧的球面投影图用于投影一致性评估
    RangeImageMapping range_cfg;
    std::vector<RangeImageData> hist_range_images;
    hist_range_images.reserve(aligned_frames.size());
    for (const auto& frame : aligned_frames) {
        hist_range_images.push_back(BuildRangeImage(frame, range_cfg));
    }
    
    int static_count = 0, dynamic_count = 0;
    int processed = 0;
    
    // 优化2: 只对降采样后的点计算权重
    for (const auto& p : current_downsampled->points) {
        // 统计该点在历史帧中的出现次数
        int appearance_count = 1;  // 当前帧算一次
        int projection_consistent = 0;
        
        // 优化3: 使用预先构建的KD-Tree
        for (size_t i = 0; i < hist_kdtrees.size(); ++i) {
            std::vector<int> indices;
            std::vector<float> distances;
            
            if (hist_kdtrees[i]->radiusSearch(p, position_threshold, indices, distances) > 0) {
                appearance_count++;
            }
        }

        int row = 0;
        int col = 0;
        float range = 0.0f;
        if (ProjectToRangeImage(p, range_cfg, row, col, range)) {
            int idx = row * range_cfg.cols + col;
            for (const auto& range_img : hist_range_images) {
                if (idx < 0 || idx >= static_cast<int>(range_img.ranges.size())) continue;
                float hist_range = range_img.ranges[static_cast<size_t>(idx)];
                if (std::isfinite(hist_range) && std::abs(hist_range - range) < projection_threshold) {
                    projection_consistent++;
                }
            }
        }
        
        // 计算权重: 出现次数越多，权重越高（表示静态物体）
        // weight = appearance_count / (total_frames + 1)  // +1 是当前帧
        double appearance_ratio = static_cast<double>(appearance_count) / (aligned_frames.size() + 1);
        double projection_ratio = aligned_frames.empty()
            ? 0.0
            : static_cast<double>(projection_consistent) / aligned_frames.size();
        double raw_weight = 0.5 * appearance_ratio + 0.5 * projection_ratio;
        raw_weight = std::max(0.0, std::min(1.0, raw_weight));
        double weight = dynamic_weight + (static_weight - dynamic_weight) * raw_weight;
        weights.push_back(weight);
        
        // 分类统计
        if (raw_weight > 0.7) {
            static_count++;
        } else {
            dynamic_count++;
        }
        
        processed++;
        if (processed % 10000 == 0) {
            std::cout << "  Processed " << processed << " / " << current_downsampled->size() << " points..." << std::endl;
        }
    }
    
    std::cout << "  Static points (weight > 0.7): " << static_count << std::endl;
    std::cout << "  Dynamic points (weight <= 0.7): " << dynamic_count << std::endl;
    
    return weights;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./lidar_extractor <cloud_path_1(.bin/.pcd)> [<cloud_path_2> ...] <output_base_path>" << std::endl;
        return -1;
    }

    // 解析参数：最后一个是输出路径，前面都是输入点云
    std::vector<std::string> cloud_paths;
    for (int i = 1; i < argc - 1; ++i) {
        cloud_paths.emplace_back(argv[i]);
    }
    std::string out_base = argv[argc - 1];
    // Ensure output directory exists (out_base is a prefix, not a directory).
    {
        const std::string::size_type slash = out_base.find_last_of("/\\");
        if (slash != std::string::npos) {
            const std::string out_dir = out_base.substr(0, slash);
            if (!out_dir.empty()) {
                ::mkdir(out_dir.c_str(), 0755);  // best-effort; ignore EEXIST
            }
        }
    }

    std::cout << "=== EdgeCalib v2.0 - LiDAR Feature Extractor ===" << std::endl;
    std::cout << "Input frames: " << cloud_paths.size() << std::endl;
    for (size_t i = 0; i < cloud_paths.size(); ++i) {
        std::cout << "  [" << i << "] " << cloud_paths[i] << std::endl;
    }
    std::cout << "Output base: " << out_base << "_*.txt" << std::endl;

    // ========================================
    // 阶段1: 加载多帧点云
    // ========================================
    std::cout << "\n[Stage 1] Loading point clouds..." << std::endl;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> frames;
    
    for (const auto& path : cloud_paths) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr frame(new pcl::PointCloud<pcl::PointXYZI>);
        const std::string ext = (path.size() >= 4) ? path.substr(path.size() - 4) : std::string();
        const std::string ext_lower = [&] {
            std::string s = ext;
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            return s;
        }();
        bool ok = false;
        if (ext_lower == ".bin") {
            ok = LocalIO::LoadKittiBin(path, frame);
        } else if (ext_lower == ".pcd") {
            ok = LocalIO::LoadPCD(path, frame);
        } else {
            std::cerr << "[Error] Unsupported cloud extension (expect .bin/.pcd): " << path << std::endl;
            ok = false;
        }
        if (!ok) {
            std::cerr << "[Error] Failed to load cloud: " << path << std::endl;
            return -1;
        }
        frames.push_back(frame);
    }
    
    // ========================================
    // 阶段2: NDT多帧融合增密 (README2.0 第二阶段 2.1)
    // ========================================
    NDTFusionResult fusion_result = FuseMultiFrameNDT(frames);
    if (!fusion_result.fused_cloud) {
        std::cerr << "[Error] Frame fusion failed" << std::endl;
        return -1;
    }
    pcl::PointCloud<pcl::PointXYZI>::Ptr fused_cloud = fusion_result.fused_cloud;
    
    // ========================================
    // 阶段3: 降采样
    // ========================================
    std::cout << "\n[Stage 2] Downsampling..." << std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ds(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::VoxelGrid<pcl::PointXYZI> sor;
    sor.setInputCloud(fused_cloud);
    const float voxel_size = static_cast<float>(GetEnvDouble("EDGECALIB_LIDAR_VOXEL_SIZE", 0.1));
    sor.setLeafSize(voxel_size, voxel_size, voxel_size); // 体素大小来自配置/环境变量
    sor.filter(*cloud_ds);
    std::cout << "  Downsampled to " << cloud_ds->size() << " points" << std::endl;

    // ========================================
    // 阶段4: 深度不连续性边缘提取 (README2.0 第二阶段 2.2)
    // ========================================

    std::cout << "\n[Stage 4] Depth discontinuity edge extraction..." << std::endl;
    DepthEdgeResult depth_edges = DetectDepthDiscontinuityEdges(cloud_ds, 1024, 64, 2.0f, -24.9f, 0.5f);
    std::cout << "  Edge points detected: " << depth_edges.edge_count << std::endl;

    // ========================================
    // 阶段5: 法向量估计
    // ========================================
    std::cout << "\n[Stage 5] Computing normals..." << std::endl;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
    ne.setInputCloud(cloud_ds);
    ne.setSearchMethod(tree);
    ne.setKSearch(std::max(1, GetEnvInt("EDGECALIB_LIDAR_NORMAL_K", 20)));
    ne.compute(*normals);
    std::cout << "  Computed normals for " << normals->size() << " points" << std::endl;

    // ========================================
    // 阶段6: 时空一致性加权 (README2.0 第二阶段 2.3)
    // 优化版本：使用NDT配准后的点云，只对降采样后的点计算权重
    // ========================================
    std::vector<double> temporal_weights = ComputeTemporalConsistencyOptimized(
        cloud_ds, 
        fusion_result.aligned_frames, 
        GetEnvDouble("EDGECALIB_LIDAR_TEMPORAL_POS_THRESH", 0.5),
        GetEnvDouble("EDGECALIB_LIDAR_TEMPORAL_PROJ_THRESH", 0.5),
        GetEnvDouble("EDGECALIB_LIDAR_TEMPORAL_STATIC_WEIGHT", 1.0),
        GetEnvDouble("EDGECALIB_LIDAR_TEMPORAL_DYNAMIC_WEIGHT", 1.0)
    );


    // ========================================
    // 阶段7: PCA几何分类与标签生成
    // ========================================
    std::cout << "\n[Stage 7] PCA-based semantic labeling..." << std::endl;
    const int LABEL_UNKNOWN = 0;
    const int LABEL_ROAD = 1;
    const int LABEL_VEGETATION = 2;
    const int LABEL_STRUCTURE = 3;

    std::vector<int> point_labels_1to1(cloud_ds->size(), LABEL_UNKNOWN);
    std::vector<PointFeature> labeled_points;
    double search_radius = GetEnvDouble("EDGECALIB_LIDAR_PCA_RADIUS", 0.5); // PCA 搜索半径
    int stats[5] = {0};

    for (size_t i = 0; i < cloud_ds->size(); ++i) {
        const auto& p_pcl = cloud_ds->points[i];
        const auto& n_pcl = normals->points[i];

        // 过滤无效点
        if (!std::isfinite(p_pcl.x) || !std::isfinite(n_pcl.normal_x)) continue;

        // 1. 邻域搜索
        std::vector<int> nn_indices;
        std::vector<float> nn_dists;
        if (tree->radiusSearch(p_pcl, search_radius, nn_indices, nn_dists) < 5) continue;

        // 2. PCA 计算
        Eigen::Vector4f centroid;
        Eigen::Matrix3f cov;
        pcl::computeMeanAndCovarianceMatrix(*cloud_ds, nn_indices, cov, centroid);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
        Eigen::Vector3f ev = solver.eigenvalues(); // ev[0] <= ev[1] <= ev[2]
        
        float l1 = ev[2]; float l2 = ev[1]; float l3 = ev[0];
        if (l1 < 1e-4) continue;

        float linearity = (l1 - l2) / l1;
        float planarity = (l2 - l3) / l1;
        float scattering = l3 / l1;

        // 3. 语义规则判决
        int label = LABEL_UNKNOWN;
        
        if (planarity > 0.5) {
            // 平面特征：高度小于 -1.2m 大概率是路面
            if (p_pcl.z < -1.2) label = LABEL_ROAD;
            else label = LABEL_STRUCTURE; // 墙面或其他平面结构
        } else if (linearity > 0.5) {
            label = LABEL_STRUCTURE; // 杆状物
        } else if (scattering > 0.3) {
            if (p_pcl.z > -1.0) label = LABEL_VEGETATION; // 高处散乱点是植被
        }
        // 【新增】：将算出的标签存入严格对齐的数组中
        point_labels_1to1[i] = label;

        // 4. 获取时空一致性权重
        double temporal_weight = 1.0;
        if (i < temporal_weights.size()) {
            temporal_weight = temporal_weights[i];
        }
        if (i < depth_edges.edge_flags.size() && depth_edges.edge_flags[i]) {
            temporal_weight = std::min(1.0, temporal_weight * 1.2);
        }

        // 5. 构建 PointFeature
        PointFeature pf;
        pf.p = Eigen::Vector3d(p_pcl.x, p_pcl.y, p_pcl.z);
        pf.intensity = p_pcl.intensity;
        pf.normal = Eigen::Vector3d(n_pcl.normal_x, n_pcl.normal_y, n_pcl.normal_z);
        pf.label = label;
        pf.weight = temporal_weight;  // 存储时空权重
        
        labeled_points.push_back(pf);
        if(label < 5) stats[label]++;
    }

    // ========================================
    // 阶段8: 保存点特征
    // ========================================
    std::cout << "\n[Stage 8] Saving point features..." << std::endl;
    std::ofstream out_pts(out_base + "_points.txt");
    if (!out_pts.is_open()) {
        std::cerr << "[Error] Cannot create output file: " << out_base << "_points.txt" << std::endl;
        return -1;
    }

    out_pts << "# Point Features: x y z intensity nx ny nz label weight\n";
    for (const auto& pf : labeled_points) {
        out_pts << pf.p.x() << " " << pf.p.y() << " " << pf.p.z() << " " 
                << pf.intensity << " "
                << pf.normal.x() << " " << pf.normal.y() << " " << pf.normal.z() << " "
                << pf.label << " " << pf.weight << "\n";
    }
    out_pts.close();
    std::cout << "  Saved " << labeled_points.size() << " labeled points with weights." << std::endl;
    std::cout << "  Stats: Road=" << stats[LABEL_ROAD] 
              << ", Veg=" << stats[LABEL_VEGETATION] 
              << ", Struct=" << stats[LABEL_STRUCTURE] << std::endl;

    // ========================================
    // 阶段9: 保存边缘点特征 (深度不连续性)
    // ========================================
    std::cout << "\n[Stage 9] Saving depth edge points..." << std::endl;
    std::ofstream out_edges(out_base + "_edge_points.txt");

    // 日志统计变量
    int count_total = 0, count_veg = 0, count_ground = 0;
    int keep_unknown = 0, keep_road = 0, keep_struct = 0;

    if (!out_edges.is_open()) {
        std::cerr << "[Warning] Cannot create edge output file: " << out_base << "_edge_points.txt" << std::endl;
    } else {
        out_edges << "# Depth Edge Points: x y z intensity\n";
        int saved_edges = 0;
        for (size_t i = 0; i < cloud_ds->size(); ++i) {
            if (!depth_edges.edge_flags.empty() && depth_edges.edge_flags[i]) {
                count_total++;
                const auto& p = cloud_ds->points[i];
                int label = point_labels_1to1[i];

                // 过滤1：抛弃植被 (树叶)
                if (label == 2) {
                    count_veg++;
                    continue;
                }
                
                // 过滤2：抛弃地面 (把阈值提高到 -1.2，确保彻底杀死路面噪点)
                if (p.z <= -1.2) {
                    count_ground++;
                    continue;
                }

                // 统计存活下来的点到底是什么身份
                if (label == 0) keep_unknown++;
                else if (label == 1) keep_road++;
                else if (label == 3) keep_struct++;
                
                if (p.z > -1.4) {
                    out_edges << p.x << " " << p.y << " " << p.z << " " << p.intensity << "\n";
                    saved_edges++;
                }
            }
        }
        out_edges.close();
        std::cout << "  Saved " << saved_edges << " edge points." << std::endl;

        // 打印诊断日志
        std::cout << "  [DEBUG] Total Raw Edges: " << count_total << std::endl;
        std::cout << "  [DEBUG] Removed Vegetation: " << count_veg << std::endl;
        std::cout << "  [DEBUG] Removed Ground (z <= -1.2): " << count_ground << std::endl;
        std::cout << "  [DEBUG] Kept Edges: " << saved_edges << std::endl;
        std::cout << "          -> Unknown (Label 0): " << keep_unknown << " (可能是噪点或锐角车框)" << std::endl;
        std::cout << "          -> Road (Label 1): " << keep_road << std::endl;
        std::cout << "          -> Structure (Label 3): " << keep_struct << " (高质量物理边缘!)" << std::endl;
    }
    
    // ========================================
    // 阶段10: 提取3D线特征 (铁轨 + 立柱)
    // ========================================
    std::cout << "\n[Stage 10] Extracting 3D line features..." << std::endl;
    std::vector<Line3D> lines;
    const double ground_z_min = GetEnvDouble("EDGECALIB_LIDAR_GROUND_Z_MIN", -3.0);
    const double ground_z_max = GetEnvDouble("EDGECALIB_LIDAR_GROUND_Z_MAX", -1.2);
    const double rail_ransac_threshold = GetEnvDouble("EDGECALIB_LIDAR_RAIL_RANSAC_THRESHOLD", 0.15);
    const int rail_max_lines = std::max(0, GetEnvInt("EDGECALIB_LIDAR_RAIL_MAX_LINES", 2));
    const double pole_z_min = GetEnvDouble("EDGECALIB_LIDAR_POLE_Z_MIN", -1.0);
    const double pole_z_max = GetEnvDouble("EDGECALIB_LIDAR_POLE_Z_MAX", 5.0);
    const double pole_ransac_threshold = GetEnvDouble("EDGECALIB_LIDAR_POLE_RANSAC_THRESHOLD", 0.2);
    const double pole_vertical_tolerance = GetEnvDouble("EDGECALIB_LIDAR_POLE_VERTICAL_TOLERANCE", 0.2);
    const int pole_max_lines = std::max(0, GetEnvInt("EDGECALIB_LIDAR_POLE_MAX_LINES", 5));
    
    // 8.1 提取铁轨 (Ground Parallel Lines)
    std::cout << "  [6.1] Extracting Rail Lines..." << std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr rail_region(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(fused_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(static_cast<float>(ground_z_min), static_cast<float>(ground_z_max)); // 地面区域
    pass.filter(*rail_region);
    std::cout << "      Filtered " << rail_region->size() << " ground points" << std::endl;

    if (rail_region->size() > 100) {
        pcl::SACSegmentation<pcl::PointXYZI> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_LINE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(rail_ransac_threshold);
        seg.setMaxIterations(200);

        // 尝试提取多条最显著的直线 (左右轨)
        for (int i=0; i<rail_max_lines; ++i) {
            if (rail_region->size() < 100) break;
            
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            seg.setInputCloud(rail_region);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.size() > 50) {
                // coefficients: [px, py, pz, dx, dy, dz]
                float cx = coefficients->values[0];
                float cy = coefficients->values[1];
                float cz = coefficients->values[2];
                float dx = coefficients->values[3];
                float dy = coefficients->values[4];
                float dz = coefficients->values[5];
                
                // 构造一段足够长的线段用于投影 (以点为中心前后延伸)
                Line3D line;
                line.p1 = Eigen::Vector3d(cx - dx * 20.0, cy - dy * 20.0, cz - dz * 20.0);
                line.p2 = Eigen::Vector3d(cx + dx * 20.0, cy + dy * 20.0, cz + dz * 20.0);
                line.type = 0; // 铁轨
                lines.push_back(line);
                
                std::cout << "      Found rail line " << (i+1) << " with " << inliers->indices.size() << " inliers" << std::endl;

                // 剔除已提取的点
                pcl::ExtractIndices<pcl::PointXYZI> extract;
                extract.setInputCloud(rail_region);
                extract.setIndices(inliers);
                extract.setNegative(true);
                extract.filter(*rail_region);
            }
        }
    }

    // 8.2 提取立柱 (Vertical Lines)
    std::cout << "  [6.2] Extracting Pole Lines..." << std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pole_region(new pcl::PointCloud<pcl::PointXYZI>);
    pass.setInputCloud(fused_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(static_cast<float>(pole_z_min), static_cast<float>(pole_z_max)); // 地面以上
    pass.filter(*pole_region);
    std::cout << "      Filtered " << pole_region->size() << " elevated points" << std::endl;
    
    if (pole_region->size() > 50) {
        // 使用 SACMODEL_PARALLEL_LINE 约束垂直方向
        pcl::SACSegmentation<pcl::PointXYZI> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PARALLEL_LINE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setAxis(Eigen::Vector3f(0, 0, 1)); // Z轴
        seg.setEpsAngle(pole_vertical_tolerance); // 允许偏离垂直角度
        seg.setDistanceThreshold(pole_ransac_threshold);
        seg.setMaxIterations(200);
        seg.setInputCloud(pole_region);

        for (int i=0; i<pole_max_lines; ++i) { // 尝试提取若干立柱
            if (pole_region->size() < 50) break;
            
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.size() > 30) {
                // 取inliers的z最大最小值作为端点
                float min_z = 1000.0f, max_z = -1000.0f;
                float cx = coefficients->values[0];
                float cy = coefficients->values[1];
                
                for(int idx : inliers->indices) {
                    float z = pole_region->points[idx].z;
                    if(z < min_z) min_z = z;
                    if(z > max_z) max_z = z;
                }
                
                // 构造垂直线段
                Line3D line;
                line.p1 = Eigen::Vector3d(cx, cy, min_z);
                line.p2 = Eigen::Vector3d(cx, cy, max_z);
                line.type = 1; // 立柱
                lines.push_back(line);
                
                std::cout << "      Found pole " << (i+1) << " with " << inliers->indices.size() << " inliers" << std::endl;

                // 剔除已提取的点
                pcl::ExtractIndices<pcl::PointXYZI> extract;
                extract.setInputCloud(pole_region);
                extract.setIndices(inliers);
                extract.setNegative(true);
                extract.filter(*pole_region);
            }
        }
    }

    // 保存线特征
    std::ofstream out_lines(out_base + "_lines_3d.txt");
    if (!out_lines.is_open()) {
        std::cerr << "[Error] Cannot create output file: " << out_base << "_lines_3d.txt" << std::endl;
        return -1;
    }
    
    out_lines << "# 3D Line Features: x1 y1 z1 x2 y2 z2 type (0=Rail, 1=Pole)\n";
    for (const auto& l : lines) {
        out_lines << l.p1.x() << " " << l.p1.y() << " " << l.p1.z() << " " 
                  << l.p2.x() << " " << l.p2.y() << " " << l.p2.z() << " " << l.type << "\n";
    }
    out_lines.close();
    std::cout << "  Saved " << lines.size() << " 3D line features" << std::endl;

    std::cout << "\n=== Extraction Complete ===" << std::endl;
    std::cout << "Output files:" << std::endl;
    std::cout << "  - " << out_base << "_points.txt" << std::endl;
    std::cout << "  - " << out_base << "_edge_points.txt" << std::endl;
    std::cout << "  - " << out_base << "_lines_3d.txt" << std::endl;
    
    return 0;
}
