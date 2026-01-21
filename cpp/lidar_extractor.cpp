#include "include/common.h"
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>

// PCL Headers
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
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
}

// ========================================
// 新增：NDT多帧融合函数 (README2.0 第二阶段 2.1)
// 优化版本：返回融合点云和配准后的点云列表（用于时空一致性计算）
// ========================================
struct NDTFusionResult {
    pcl::PointCloud<pcl::PointXYZI>::Ptr fused_cloud;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> aligned_frames;  // 配准后的历史帧
};

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
    ndt.setTransformationEpsilon(0.01);  // 变换收敛阈值
    ndt.setStepSize(0.1);                 // More/Newton线搜索步长
    ndt.setResolution(1.0);               // NDT网格分辨率 (m)
    ndt.setMaximumIterations(30);         // 最大迭代次数
    
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
struct PointWithWeight {
    pcl::PointXYZI point;
    double weight;  // 0.0 (动态) ~ 1.0 (静态)
    
    PointWithWeight(const pcl::PointXYZI& p, double w) : point(p), weight(w) {}
};

// 优化版本：接受配准后的点云列表和降采样后的当前点云
std::vector<double> ComputeTemporalConsistencyOptimized(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& current_downsampled,
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& aligned_frames,
    double position_threshold = 0.5)  // 位置一致性阈值 (m)
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
    
    int static_count = 0, dynamic_count = 0;
    int processed = 0;
    
    // 优化2: 只对降采样后的点计算权重
    for (const auto& p : current_downsampled->points) {
        // 统计该点在历史帧中的出现次数
        int appearance_count = 1;  // 当前帧算一次
        
        // 优化3: 使用预先构建的KD-Tree
        for (size_t i = 0; i < hist_kdtrees.size(); ++i) {
            std::vector<int> indices;
            std::vector<float> distances;
            
            if (hist_kdtrees[i]->radiusSearch(p, position_threshold, indices, distances) > 0) {
                appearance_count++;
            }
        }
        
        // 计算权重: 出现次数越多，权重越高（表示静态物体）
        // weight = appearance_count / (total_frames + 1)  // +1 是当前帧
        double weight = static_cast<double>(appearance_count) / (aligned_frames.size() + 1);
        weights.push_back(weight);
        
        // 分类统计
        if (weight > 0.7) {
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
        std::cerr << "Usage: ./lidar_extractor <kitti_bin_path_1> [<kitti_bin_path_2> ...] <output_base_path>" << std::endl;
        std::cerr << "Example: ./lidar_extractor data/velodyne/000000.bin data/velodyne/000001.bin result/lidar_features/000001" << std::endl;
        return -1;
    }

    // 解析参数：最后一个是输出路径，前面都是输入点云
    std::vector<std::string> bin_paths;
    for (int i = 1; i < argc - 1; ++i) {
        bin_paths.emplace_back(argv[i]);
    }
    std::string out_base = argv[argc - 1];

    std::cout << "=== EdgeCalib v2.0 - LiDAR Feature Extractor ===" << std::endl;
    std::cout << "Input frames: " << bin_paths.size() << std::endl;
    for (size_t i = 0; i < bin_paths.size(); ++i) {
        std::cout << "  [" << i << "] " << bin_paths[i] << std::endl;
    }
    std::cout << "Output base: " << out_base << "_*.txt" << std::endl;

    // ========================================
    // 阶段1: 加载多帧点云
    // ========================================
    std::cout << "\n[Stage 1] Loading point clouds..." << std::endl;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> frames;
    
    for (const auto& path : bin_paths) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr frame(new pcl::PointCloud<pcl::PointXYZI>);
        if (!LocalIO::LoadKittiBin(path, frame)) {
            std::cerr << "[Error] Failed to load bin: " << path << std::endl;
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
    sor.setLeafSize(0.1f, 0.1f, 0.1f); // 10cm 体素
    sor.filter(*cloud_ds);
    std::cout << "  Downsampled to " << cloud_ds->size() << " points" << std::endl;

    // ========================================
    // 阶段4: 法向量估计
    // ========================================
    std::cout << "\n[Stage 3] Computing normals..." << std::endl;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
    ne.setInputCloud(cloud_ds);
    ne.setSearchMethod(tree);
    ne.setKSearch(20);
    ne.compute(*normals);
    std::cout << "  Computed normals for " << normals->size() << " points" << std::endl;

    // ========================================
    // 阶段5: 时空一致性加权 (README2.0 第二阶段 2.3)
    // 优化版本：使用NDT配准后的点云，只对降采样后的点计算权重
    // ========================================
    std::vector<double> temporal_weights = ComputeTemporalConsistencyOptimized(
        cloud_ds, 
        fusion_result.aligned_frames, 
        0.5
    );

    // ========================================
    // 阶段6: PCA几何分类与标签生成
    // ========================================
    std::cout << "\n[Stage 4] PCA-based semantic labeling..." << std::endl;
    const int LABEL_UNKNOWN = 0;
    const int LABEL_ROAD = 1;
    const int LABEL_VEGETATION = 2;
    const int LABEL_STRUCTURE = 3;

    std::vector<PointFeature> labeled_points;
    double search_radius = 0.5; // PCA 搜索半径 0.5m
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

        // 4. 获取时空一致性权重
        double temporal_weight = 1.0;
        if (i < temporal_weights.size()) {
            temporal_weight = temporal_weights[i];
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
    // 阶段7: 保存点特征
    // ========================================
    std::cout << "\n[Stage 5] Saving point features..." << std::endl;
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
    // 阶段8: 提取3D线特征 (铁轨 + 立柱)
    // ========================================
    std::cout << "\n[Stage 6] Extracting 3D line features..." << std::endl;
    std::vector<Line3D> lines;
    
    // 8.1 提取铁轨 (Ground Parallel Lines)
    std::cout << "  [6.1] Extracting Rail Lines..." << std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr rail_region(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(fused_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-3.0, -1.2); // 粗略的地面区域
    pass.filter(*rail_region);
    std::cout << "      Filtered " << rail_region->size() << " ground points" << std::endl;

    if (rail_region->size() > 100) {
        pcl::SACSegmentation<pcl::PointXYZI> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_LINE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.15); // 15cm 容差
        seg.setMaxIterations(200);

        // 尝试提取两条最显著的直线 (左右轨)
        for (int i=0; i<2; ++i) {
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
    pass.setFilterLimits(-1.0, 5.0); // 地面以上
    pass.filter(*pole_region);
    std::cout << "      Filtered " << pole_region->size() << " elevated points" << std::endl;
    
    if (pole_region->size() > 50) {
        // 使用 SACMODEL_PARALLEL_LINE 约束垂直方向
        pcl::SACSegmentation<pcl::PointXYZI> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PARALLEL_LINE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setAxis(Eigen::Vector3f(0, 0, 1)); // Z轴
        seg.setEpsAngle(0.2); // 允许偏离垂直 ~11度
        seg.setDistanceThreshold(0.2);
        seg.setMaxIterations(200);
        seg.setInputCloud(pole_region);

        for (int i=0; i<5; ++i) { // 尝试提取5根立柱
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
    std::cout << "  - " << out_base << "_lines_3d.txt" << std::endl;
    
    return 0;
}
